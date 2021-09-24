"""`BiphasicAxonMapModel`"""
from typing import Type
import numpy as np
import sys

from . import AxonMapModel, AxonMapSpatial, TemporalModel, Model
from ..implants import ProsthesisSystem, ElectrodeArray
from ..stimuli import BiphasicPulseTrain, Stimulus
from ..percepts import Percept
from ..utils import FreezeError
from .base import BaseModel, NotBuiltError
from ._granley2021 import fast_biphasic_axon_map


class DefaultBrightModel(BaseModel):
    """
    Default model to be used for brightness scaling in BiphasicAxonMapModel
    Implements Eq 4 from [Granley2021]_ 
    Fit using data from [Nanduri2012]_ and [Weitz2015]_

    Parameters:
    ------------
    do_thresholding : bool, optional
        Set to true to enable probabilistic phosphene appearance at near-threshold 
        amplitudes
    a0, a1 : float, optional
        Linear regression coefficients (slope and intercept) of pulse_duration
        vs threshold curve (Eq 3). Amplitude factor will be scaled by 
        (a0*pdur + a1)^-1.
    a2, a3, a4: float, optional
        Linear regression coefficients for brightness vs amplitude and frequency (Eq 4)
        F_bright = a2*scaled_amp + a3*freq + a4 
    """

    def __init__(self, **params):
        super(DefaultBrightModel, self).__init__(**params)
        self.build()

    def get_default_params(self):
        params = {
            'a0': 0.27,
            'a1': 0.8825,
            'a2': 1.84,
            'a3': 0.2,
            'a4': 3.0986,
            'do_thresholding': False
        }
        return params

    def scale_threshold(self, pdur):
        """ 
        Based on eq 3 in paper, this function produces the factor that amplitude
        will be scaled by to produce a_tilde. Computes (A_0 * t + A_1)^-1
        """
        return 1 / (self.a1 + self.a0*pdur)

    def predict_freq_amp(self, amp, freq):
        """ Eq 4 in paper, A_2*A_tilde + A_3*f + A_4 """
        return self.a2*amp + self.a3*freq + self.a4

    def __call__(self, freq, amp, pdur):
        """
        Main function to be called by BiphasicAxonMapModel
        Outputs value by which brightness contribution for each electrode should
        be scaled by (F_bright)
        """
        # Scale amp according to pdur (Eq 3 in paper) and then calculate F_{bright}
        F_bright = self.predict_freq_amp(amp * self.scale_threshold(pdur), freq)

        if not self.do_thresholding:
            return F_bright
        else:
            # If thresholding is enabled, phosphene only has a chance of appearing
            # p = -1/96 + 97 / (96(1+96 exp(-ln(98)amp)))
            # Sigmoid with p[0] = 0, p[1] = 0.5, p[2] = 0.99
            p = -0.01041666666667 + 1.0104166666666667 / (
                1+96 * np.exp(-4.584967478670572 * amp * self.scale_threshold(pdur)))
            if np.random.random() < p:
                return F_bright
            else:
                return 0


class DefaultSizeModel(BaseModel):
    """
    Default model to be used for size (rho) scaling in BiphasicAxonMapModel
    Implements Eq 5 from [Granley2021]_ 
    Fit using data from [Nanduri2012]_ and [Weitz2015]_

    Parameters:
    ------------
    rho :  float32
        Rho parameter of BiphasicAxonMapModel (spatial decay rate)
    a0, a1 : float, optional
        Linear regression coefficients (slope and intercept) of pulse_duration
        vs threshold curve (Eq 3). Amplitude factor will be scaled by 
        (a0*pdur + a1)^-1.
    a5, a6 : float, optional
        Linear regression coefficients for size vs amplitude (Eq 5)
        F_size = a5*scaled_amp + a6 
    """

    def __init__(self, rho, **params):
        super(DefaultSizeModel, self).__init__(**params)
        self.rho = rho
        self.build()

    def get_default_params(self):
        params = {
            'a0': 0.27,
            'a1': 0.8825,
            'a5': 1.0812,
            'a6': -0.35338,
            # dont let rho be scaled below this threshold
            'min_rho': 10,
        }
        return params

    def scale_threshold(self, pdur):
        """ 
        Based on eq 3 in paper, this function produces the factor that amplitude
        will be scaled by to produce a_tilde. Computes (A_0 * t + A_1)^-1
        """
        return 1 / (self.a1 + self.a0*pdur)

    def __call__(self, freq, amp, pdur):
        """
        Main function to be called by BiphasicAxonMapModel
        Outputs value for each electrode that rho should be scaled by (F_size)
        """
        min_f_size = self.min_rho**2 / self.rho**2
        F_size = self.a5 * amp * self.scale_threshold(pdur) + self.a6
        if F_size > min_f_size:
            return F_size
        else:
            return min_f_size


class DefaultStreakModel(BaseModel):
    """
    Default model to be used for streak length (lambda) scaling in BiphasicAxonMapModel
    Implements Eq 6 from [Granley2021]_ 
    Fit using data from [Weitz2015]_

    Parameters:
    ------------
    axlambda :  float32
        Axlambda parameter of BiphasicAxonMapModel (axonal decay rate)
    a7, a8, a9: float, optional
        Regression coefficients for streak length vs pulse duration (Eq 6)
        F_streak = -a7*pdur^a8 + a9
    """

    def __init__(self, axlambda, **params):
        super(DefaultStreakModel, self).__init__(**params)
        self.axlambda = axlambda
        self.build()

    def get_default_params(self):
        params = {
            'a7': 0.54,
            'a8': 0.21,
            'a9': 1.56,
            # dont let lambda be scaled below this threshold
            'min_lambda': 10,
        }
        return params

    def __call__(self, freq, amp, pdur):
        """
        Main function to be called by BiphasicAxonMapModel
        Outputs value for each electrode that lambda should be scaled by (F_streak)
        """
        min_f_streak = self.min_lambda**2 / self.axlambda ** 2
        F_streak = self.a9 - self.a7 * pdur ** self.a8
        if F_streak > min_f_streak:
            return F_streak
        else:
            return min_f_streak


class BiphasicAxonMapSpatial(AxonMapSpatial):
    """ BiphasicAxonMapModel of [Granley2021]_ (spatial model)
    An AxonMapModel where phosphene brightness, size, and streak length scale
    according to amplitude, frequency, and pulse duration

    All stimuli must be BiphasicPulseTrains.

    This model is different than other spatial models in that it calculates
    one representative percept from all time steps of the stimulus.

    Brightness, size, and streak length scaling are controlled by the effects models
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
    **params: optional
        Additional params for AxonMapModel. 

        Options:
        --------
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
        if self.bright_model is None:
            self.bright_model = DefaultBrightModel(
                do_thresholding=self.do_thresholding)
        if self.size_model is None:
            self.size_model = DefaultSizeModel(self.rho)
        if self.streak_model is None:
            self.streak_model = DefaultStreakModel(self.axlambda)

    def __getattr__(self, attr):
        # Called when normal get attribute fails
        if sys._getframe(2).f_code.co_name == '__init__' or  \
           sys._getframe(3).f_code.co_name == '__init__':
            # We can set new class attributes in the constructor. Reaching this
            # point means the default attribute access failed - most likely
            # because we are trying to create a variable. In this case, simply
            # raise an exception:
            raise AttributeError("%s not found" % attr)
        # Check if bright/size/streak model has param
        for m in [self.bright_model, self.size_model, self.streak_model]:
            if hasattr(m, attr):
                return getattr(m, attr)
        raise AttributeError("%s not found" % attr)

    def __setattr__(self, name, value):
        """Called when an attribute is set

        This method is called when a new attribute is set(e.g.,
        ``model.a=2``). This is allowed in the constructor, but will raise a
        ``FreezeError`` elsewhere.

        ``model.a = X`` can be used as a shorthand to set ``model.bright_model.a``,
        etc
        """
        try:
            if sys._getframe(2).f_code.co_name == '__init__' or  \
               sys._getframe(3).f_code.co_name == '__init__':
                super().__setattr__(name, value)
                return
        except FreezeError:
            pass
        # Check whether the attribute is a part of any
        # bright/size/streak model
        found = False
        # try to set it ourselves, but can't use get_attr
        try:
            self.__getattribute__(name)
            # if we get here, we have the attribute, not (neccesarily) an effects model
            super().__setattr__(name, value)
            found = True
        except AttributeError:
            pass
        for m in [self.bright_model, self.size_model, self.streak_model]:
            try:
                if hasattr(m, name):
                    setattr(m, name, value)
                    found = True
            except (AttributeError, FreezeError):
                pass
        if not found:
            err_str = ("'%s' not found. You cannot add attributes to %s "
                       "outside the constructor." % (name,
                                                     self.__class__.__name__))
            raise FreezeError(err_str)

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
        if not callable(self.bright_model):
            raise TypeError("bright_model needs to be callable")
        if not callable(self.size_model):
            raise TypeError("size_model needs to be callable")
        if not callable(self.streak_model):
            raise TypeError("streak_model needs to be callable")
        super(BiphasicAxonMapSpatial, self)._build()

    def _predict_spatial(self, earray, stim):
        """Predicts the percept"""
        if not isinstance(earray, ElectrodeArray):
            raise TypeError("Implant must be of type ElectrodeArray but it is " +
                str(type(earray)))
        if not isinstance(stim, Stimulus):
            raise TypeError(
                "Stim must be of type Stimulus but it is " + str(type(stim)))

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

        if self.engine != 'jax':
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
                self.rho, self.thresh_percept)
        else:
            raise NotImplementedError("Jax will be supported in future release")

    def predict_percept(self, implant, t_percept=None):
        """ Predicts the spatial response
        Override base predict percept to have desired timesteps and 
        remove unneccesary computation

        Parameters
        ----------
        implant: :py:class:`~pulse2percept.implants.ProsthesisSystem`
            A valid prosthesis system. A stimulus can be passed via
            :py:meth:`~pulse2percept.implants.ProsthesisSystem.stim`.
        t_percept: float or list of floats, optional
            The time points at which to output a percept (ms).
            If None, ``implant.stim.time`` is used.

        Returns
        -------
        percept: :py:class:`~pulse2percept.models.Percept`
            A Percept object whose ``data`` container has dimensions Y x X x 1.
            Will return None if ``implant.stim`` is None.
        """
        # Make sure stimulus is a BiphasicPulseTrain:
        if not isinstance(implant.stim, BiphasicPulseTrain):
            # Could still be a stimulus where each electrode has a biphasic pulse train
            # or a 0 stimulus
            for i, (ele, params) in enumerate(implant.stim.metadata
                                              ['electrodes'].items()):
                if (params['type'] != BiphasicPulseTrain or
                        params['metadata']['delay_dur'] != 0) and \
                        np.any(implant.stim[i]):
                    raise TypeError(
                        "All stimuli must be BiphasicPulseTrains with no " +
                        "delay dur (Failing electrode: %s)" % (ele))
        if isinstance(implant, ProsthesisSystem):
            if implant.eye != self.eye:
                raise ValueError(("The implant is in %s but the model was "
                                  "built for %s.") % (implant.eye,
                                                      self.eye))
        if not self.is_built:
            raise NotBuiltError("Yout must call ``build`` first.")
        if not isinstance(implant, ProsthesisSystem):
            raise TypeError(("'implant' must be a ProsthesisSystem object, "
                             "not %s.") % type(implant))

        if implant.stim is None:
            return None
        stim = implant.stim
        if t_percept is None:
            n_time = 1
        else:
            n_time = len(t_percept)
        if not np.any(stim.data):
            # Stimulus is 0
            resp = np.zeros(list(self.grid.x.shape) + [n_time],
                            dtype=np.float32)
        else:
            # Make sure stimulus is in proper format
            stim = Stimulus(stim)
            resp = np.zeros(list(self.grid.x.shape) + [n_time])
            # Response goes in first frame
            resp[:, :, 0] = self._predict_spatial(
                implant.earray, stim).reshape(self.grid.x.shape)
        return Percept(resp, space=self.grid, time=t_percept,
                       metadata={'stim': stim.metadata})


class BiphasicAxonMapModel(Model):
    """ BiphasicAxonMapModel of [Granley2021]_ (standalone model)
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
        super(BiphasicAxonMapModel, self).__init__(
            spatial=BiphasicAxonMapSpatial(), temporal=None, **params)

    def predict_percept(self, implant, t_percept=None):
        """Predict a percept
        Overrides base predict percept to keep desired time axes
        .. important ::

            You must call ``build`` before calling ``predict_percept``.

        Parameters
        ----------
        implant: :py:class:`~pulse2percept.implants.ProsthesisSystem`
            A valid prosthesis system. A stimulus can be passed via
            :py:meth:`~pulse2percept.implants.ProsthesisSystem.stim`.
        t_percept: float or list of floats, optional
            The time points at which to output a percept (ms).
            If None, ``implant.stim.time`` is used.

        Returns
        -------
        percept: :py:class:`~pulse2percept.models.Percept`
            A Percept object whose ``data`` container has dimensions Y x X x T.
            Will return None if ``implant.stim`` is None.
        """
        if not self.is_built:
            raise NotBuiltError("You must call ``build`` first.")
        if not isinstance(implant, ProsthesisSystem):
            raise TypeError("'implant' must be a ProsthesisSystem object, not "
                            "%s." % type(implant))
        if implant.stim is None or (not self.has_space and not self.has_time):
            # Nothing to see here:
            return None
        resp = self.spatial.predict_percept(implant, t_percept=t_percept)
        return resp
