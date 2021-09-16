"""`BiphasicAxonMapModel`"""
import numpy as np

from pulse2percept.models import AxonMapModel, AxonMapSpatial, TemporalModel, Model
from pulse2percept.implants import ProsthesisSystem, ElectrodeArray
from pulse2percept.stimuli import BiphasicPulseTrain, Stimulus
from pulse2percept.datasets import load_nanduri2012

from ._granley2021 import fast_biphasic_axon_map


class DefaultBrightModel():
    """
    Default model to be used for brightness scaling in BiphasicAxonMapModel
    Implements Eq 4 from [Granley2021]_ 
    Fit using data from [Nanduri2012]_ 

    Parameters:
    ------------
    do_thresholding : bool, optional
        Set to true to enable probabilistic phosphene appearance at near-threshold 
        amplitudes
    """

    def __init__(self,  do_thresholding=False):
        self.do_thresholding = do_thresholding

    def scale_threshold(self, pdur):
        """ 
        Based on eq 3 in paper, this function produces the factor that amplitude
        will be scaled by to produce a_tilde. Computes (A_0 * t + A_1)^-1
        Fit using color threshold of [Weitz2015]_
        """
        return 1 / (0.8825 + 0.27*pdur)

    def predict_freq_amp(self, amp, freq):
        """ Eq 4 in paper, A_2*A_tilde + A_3*f + A_4 """
        return 1.84*amp + 0.2*freq + 3.0986

    def __call__(self, freq, amp, pdur):
        """
        Main function to be called by BiphasicAxonMapModel
        Outputs value by which brightness contribution for each electrode should
        be scaled by (F_bright)
        """
        # Scale amp according to pdur (Eq 3 in paper) and then calculate F_{bright}
        F_bright = self.predict_freq_amp(
            amp * self.scale_threshold(pdur), freq)

        # If thresholding is enabled, phosphene only has a chance of appearing, determined by this sigmoid
        # p = -1/96 + 97 / (96(1+96 exp(-ln(98)amp)))
        # Sigmoid with p[0] = 0, p[1] = 0.5, p[2] = 0.99
        p = -0.01041666666667 + 1.0104166666666667 / (
            1+96 * np.exp(-4.584967478670572 * amp * self.predict_pdur(pdur)))
        if not self.do_thresholding or np.random.random() < p:
            return F_bright
        else:
            return 0


class DefaultSizeModel():
    """
    Default model to be used for size (rho) scaling in BiphasicAxonMapModel
    Implements Eq 5 from [Granley2021]_ 
    Fit using data from [Nanduri2012]_ 

    Parameters:
    ------------
    rho :  float32
        Rho parameter of BiphasicAxonMapModel (spatial decay rate)
    """

    def __init__(self, rho):
        self.min_rho = 10  # dont let rho be scaled below this threshold
        self.min_f_size = self.min_rho**2 / rho**2

    def scale_threshold(self, pdur):
        """ 
        Based on eq 3 in paper, this function produces the factor that amplitude
        will be scaled by to produce a_tilde. Computes (A_0 * t + A_1)^-1
        Fit using color threshold of [Weitz2015]_
        """
        return 1 / (0.8825 + 0.27*pdur)

    def __call__(self, freq, amp, pdur):
        """
        Main function to be called by BiphasicAxonMapModel
        Outputs value for each electrode that rho should be scaled by (F_size)
        """
        F_size = 1.0812 * amp * self.scale_threshold(pdur) - 0.35338
        if F_size > self.min_f_size:
            return F_size
        else:
            return self.min_f_size


class DefaultStreakModel():
    """
    Default model to be used for streak length (lambda) scaling in BiphasicAxonMapModel
    Implements Eq 6 from [Granley2021]_ 
    Fit using data from [Weitz2015]_

    Parameters:
    ------------
    axlambda :  float32
        Axlambda parameter of BiphasicAxonMapModel (axonal decay rate)
    """

    def __init__(self, axlambda):
        # never decrease lambda to less than 10
        self.min_lambda = 10
        self.min_f_streak = self.min_lambda**2 / axlambda ** 2

    def __call__(self, freq, amp, pdur):
        """
        Main function to be called by BiphasicAxonMapModel
        Outputs value for each electrode that lambda should be scaled by (F_streak)
        """
        F_streak = 1.56 - 0.54 * pdur ** 0.21
        if F_streak > self.min_f_streak:
            return F_streak
        else:
            return self.min_f_streak


class BiphasicAxonMapSpatial(AxonMapSpatial):
    """ BiphasicAxonMapModel of [Granley2021]_ (spatial model)
    An AxonMapModel where phosphene brightness, size, and streak length scale
    according to amplitude, frequency, and pulse duration

    All stimuli must be BiphasicPulseTrains.

    This model is different than other spatial models in that it calculates
    one representative percept from all time steps of the stimulus.

    Brightness, size, and streak length scaling are controlled by the parameters
    bright_model, size_model, and streak model respectively. By default, these are
    set to classes that implement Eqs 3-6 from Granley 2021. These models can be
    individually customized by setting the bright_model, size_model, or streak_model
    to any python callable with signature f(freq, amp, pdur)

    .. note: :
        Using this model in combination with a temporal model is not currently
        supported and will give unexpected results

    Parameters
    ----------
    bright_model: callable, optional
        Model used to modulate percept brightness with amplitude, frequency,
        and pulse duration
    size_model: callable, optional
        Model used to modulate percept size with amplitude, frequency, and
        pulse duration
    streak_model: callable, optional
        Model used to modulate percept streak length with amplitude, frequency,
        and pulse duration
    do_thresholding: boolean
        Use probabilistic sigmoid thresholding, default: False
    **params: dict, optional
        Arguments to be passed to AxonMapSpatial
        Options:
        ---------
        axlambda: double, optional
            Exponential decay constant along the axon(microns).
        rho: double, optional
            Exponential decay constant away from the axon(microns).
        eye: {'RE', LE'}, optional
            Eye for which to generate the axon map.
        xrange : (x_min, x_max), optional
            A tuple indicating the range of x values to simulate (in degrees of
            visual angle). In a right eye, negative x values correspond to the
            temporal retina, and positive x values to the nasal retina. In a left
            eye, the opposite is true.
        yrange : tuple, (y_min, y_max)
            A tuple indicating the range of y values to simulate (in degrees of
            visual angle). Negative y values correspond to the superior retina,
            and positive y values to the inferior retina.
        xystep : int, double, tuple
            Step size for the range of (x,y) values to simulate (in degrees of
            visual angle). For example, to create a grid with x values [0, 0.5, 1]
            use ``x_range=(0, 1)`` and ``xystep=0.5``.
        grid_type : {'rectangular', 'hexagonal'}
            Whether to simulate points on a rectangular or hexagonal grid
        loc_od, loc_od: (x,y), optional
            Location of the optic disc in degrees of visual angle. Note that the
            optic disc in a left eye will be corrected to have a negative x
            coordinate.
        n_axons: int, optional
            Number of axons to generate.
        axons_range: (min, max), optional
            The range of angles(in degrees) at which axons exit the optic disc.
            This corresponds to the range of $\\phi_0$ values used in
            [Jansonius2009]_.
        n_ax_segments: int, optional
            Number of segments an axon is made of.
        ax_segments_range: (min, max), optional
            Lower and upper bounds for the radial position values(polar coords)
            for each axon.
        min_ax_sensitivity: float, optional
            Axon segments whose contribution to brightness is smaller than this
            value will be pruned to improve computational efficiency. Set to a
            value between 0 and 1.
        axon_pickle: str, optional
            File name in which to store precomputed axon maps.
        ignore_pickle: bool, optional
            A flag whether to ignore the pickle file in future calls to
            ``model.build()``.
    """

    def __init__(self, **params):
        super(BiphasicAxonMapSpatial, self).__init__(**params)

    def get_default_params(self):
        base_params = super(BiphasicAxonMapSpatial, self).get_default_params()
        params = {
            # Callable model used to modulate percept brightness with amplitude,
            # frequency, and pulse duration
            'bright_model': None,
            # Callable model used to modulate percept size with amplitude,
            # frequency, and pulse duration
            'size_model': None,
            # Callable model used to modulate percept streak length with amplitude,
            # frequency, and pulse duration
            'streak_model': None,
            # Use probabilistic thresholding
            'do_thresholding': False
        }
        params.update(base_params)
        return params

    def _build(self):
        if self.bright_model is None:
            self.bright_model = DefaultBrightModel(
                do_thresholding=self.do_thresholding)
        if self.size_model is None:
            self.size_model = DefaultSizeModel(self.rho)
        if self.streak_model is None:
            self.streak_model = DefaultStreakModel(self.axlambda)
        assert(callable(self.bright_model))
        assert(callable(self.size_model))
        assert(callable(self.streak_model))
        super(BiphasicAxonMapSpatial, self)._build()

    def _predict_spatial(self, earray, stim):
        """Predicts the percept"""
        assert isinstance(earray, ElectrodeArray)
        assert isinstance(stim, Stimulus)

        # Calculate model effects before losing GIL
        bright_effects = []
        size_effects = []
        streak_effects = []
        amps = []
        for e in stim.electrodes:
            amp = stim.metadata['electrodes'][str(e)]['metadata']['amp']
            freq = stim.metadata['electrodes'][str(e)]['metadata']['freq']
            pdur = stim.metadata['electrodes'][str(e)]['metadata']['phase_dur']
            bright_effects.append(self.bright_model(freq, amp, pdur))
            size_effects.append(self.size_model(freq, amp, pdur))
            streak_effects.append(self.streak_model(freq, amp, pdur))
            amps.append(amp)

        return fast_biphasic_axon_map(
            np.array(amps, dtype=np.float32),
            np.array(bright_effects, dtype=np.float32),
            np.array(size_effects, dtype=np.float32),
            np.array(streak_effects, dtype=np.float32),
            np.array([earray[e].x for e in stim.electrodes], dtype=np.float32),
            np.array([earray[e].y for e in stim.electrodes], dtype=np.float32),
            self.axon_contrib, 
            self.axon_idx_start.astype(np.uint32),
            self.axon_idx_end.astype(np.uint32),
            self.rho, self.thresh_percept, stim.shape[1])


class BiphasicAxonMapModel(Model):
    """ BiphasicAxonMapModel of [Granley2021]_ (standalone model)
    AxonMapModel that scales percept size, brightness, and streak length according to
    the amplitude, frequency, and pulse duration of the BiphasicPulseTrain.

    All stimuli must be BiphasicPulseTrains with no delay dur.

    This model is different than other spatial models in that it calculates the brightest percept 
    from all time steps of the stimulus, and then returns this same percept at each time step. 

    The three new parameters are the models to be used to scale brightness, size, and streak length. 
    These models can be any python callable with function signature f(amp, freq, pdur) that return a float.


    .. note: :

        Using this model in combination with a temporal model is not supported and may give unexpected results

    Parameters
    ----------
    bright_model: callable, optional
        Model used to modulate percept brightness with amplitude, frequency, and pulse duration
    size_model: callable, optional
        Model used to modulate percept size with amplitude, frequency, and pulse duration
    streak_model: callable, optional
        Model used to modulate percept streak length with amplitude, frequency, and pulse duration
    do_thresholding: boolean
        Use probabilistic sigmoid thresholding, default=True
    **params: dict, optional
        Arguments to be passed to AxonMapModel
    """

    def __init__(self, **params):
        super(
            BiphasicAxonMapModel, self).__init__(
            spatial=BiphasicAxonMapSpatial(),
            temporal=None, **params)

    def predict_percept(self, implant, t_percept=None):
        # Make sure stimulus is a BiphasicPulseTrain:
        if not isinstance(implant.stim, BiphasicPulseTrain):
            # Could still be a stimulus where each electrode has a biphasic pulse train
            for ele, params in implant.stim.metadata['electrodes'].items():
                if params['type'] != BiphasicPulseTrain or params['metadata'][
                        'delay_dur'] != 0:
                    raise TypeError(
                        "All stimuli must be BiphasicPulseTrains with no delay dur (Failing electrode: %s)" % (ele))

        return super(
            BiphasicAxonMapModel, self).predict_percept(
            implant, t_percept=t_percept)
