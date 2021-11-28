"""`BiphasicAxonMapModel`"""
from jax._src.dtypes import dtype
import numpy as np
import sys
import time

from . import AxonMapModel, AxonMapSpatial, TemporalModel, Model
from ..implants import ProsthesisSystem, ElectrodeArray
from ..stimuli import BiphasicPulseTrain, Stimulus
from ..percepts import Percept
from ..utils import FreezeError
from .base import BaseModel, NotBuiltError
from ._granley2021 import fast_biphasic_axon_map


try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    has_jax = True
except ImportError:
    has_jax = False

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
            'a0': 2.095,
            'a1': 0.054326,
            'a2': 0.1492147,
            'a3': 0.0163851,
            'a4': 0.25191869,
            'do_thresholding': False
        }
        return params

    def scale_threshold(self, pdur):
        """ 
        Based on eq 3 in paper, this function produces the factor that amplitude
        will be scaled by to produce a_tilde. Computes A_0 * t + A_1 (1/threshold)
        """
        return self.a1 + self.a0*pdur

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

        return F_bright

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
            'a0': 2.095,
            'a1': 0.054326,
            'a5': 1.0812,
            'a6': -0.35338,
            # dont let rho be scaled below this threshold
            'min_rho': 10,
        }
        return params

    def scale_threshold(self, pdur):
        """ 
        Based on eq 3 in paper, this function produces the factor that amplitude
        will be scaled by to produce a_tilde. Computes A_0 * t + A_1 (1/threshold)
        """
        return self.a1 + self.a0*pdur

    def __call__(self, freq, amp, pdur):
        """
        Main function to be called by BiphasicAxonMapModel
        Outputs value for each electrode that rho should be scaled by (F_size)
        """
        min_f_size = self.min_rho**2 / self.rho**2
        F_size = self.a5 * amp * self.scale_threshold(pdur) + self.a6
        return np.maximum(F_size, min_f_size)


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
        return jnp.maximum(F_streak, min_f_streak)


def predict_one_point_jax(axon, eparams, rho, axlambda):
    d2_el = (axon[:, 0, None] - eparams[:, 3])**2 + (axon[:, 1, None] - eparams[:, 4])**2
    intensities = eparams[:, 0] * jnp.exp(-d2_el / (2. * rho**2 * eparams[:, 1])) * (axon[:, 2, None] ** (1./eparams[:, 2]))
    return jnp.sum(intensities, axis=1)

@jit
def biphasic_axon_map_jax(eparams, axon_segments, rho, axlambda, thresh_percept):
    I = jnp.max(jax.vmap(predict_one_point_jax, in_axes=[0, None, None, None])(
                            axon_segments, 
                            eparams, 
                            rho, 
                            axlambda), 
            axis=1)
    I = (I > thresh_percept) * I
    return I

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
        retinotopy : :py:class:`~pulse2percept.utils.VisualFieldMap`, optional
            An instance of a :py:class:`~pulse2percept.utils.VisualFieldMap`
            object that provides ``ret2dva`` and ``dva2ret`` methods.
            By default, :py:class:`~pulse2percept.utils.Watson2014Map` is
            used.
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
        if self.engine == 'jax' and not has_jax:
            raise ImportError("Engine was chosen as jax, but jax is not installed. "
                              "You can install it with 'pip install \"jax[cpu]\"' for cpu "
                              "or following https://github.com/google/jax#installation for gpu")

        for key, val in params.items():
            if key in ['bright_model', 'size_model', 'streak_model']:
                continue
            setattr(self, key, val)

    def __getattr__(self, attr):
        # Called when normal get attribute fails
        # If we are in the initializer, or if trying to access
        # an effects model, raise an error which is caught and causes
        # the parameter to be created.
        if (sys._getframe(3).f_code.co_name == '__init__' and
                "pulse2percept/models/base.py" in
                sys._getframe(3).f_code.co_filename) or \
                (attr in ['bright_model', 'streak_model', 'size_model']):
            # We can set new class attributes in the constructor. Reaching this
            # point means the default attribute access failed - most likely
            # because we are trying to create a variable. In this case, simply
            # raise an exception:
            # Note that this gets called from __init__ of BaseModel, not directly from
            # BiphasicAxonMap
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
        found = False
        # try to set it ourselves, but can't use get_attr
        try:
            self.__getattribute__(name)
            # if we get here, we have the attribute, not (neccesarily) an effects model
            super().__setattr__(name, value)
            found = True
        except AttributeError:
            pass
        # Check whether the attribute is a part of any
        # bright/size/streak model
        if name not in ['bright_model', 'size_model', 'streak_model']:
            try:
                for m in [self.bright_model, self.size_model, self.streak_model]:
                    if hasattr(m, name):
                        setattr(m, name, value)
                        found = True
            except (AttributeError, FreezeError):
                pass
        if not found:
            try:
                if sys._getframe(2).f_code.co_name == '__init__' or  \
                sys._getframe(3).f_code.co_name == '__init__':
                    super().__setattr__(name, value)
                    return
            except FreezeError:
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
        return {**base_params, **params}

    def _build(self):
        if not callable(self.bright_model):
            raise TypeError("bright_model needs to be callable")
        if not callable(self.size_model):
            raise TypeError("size_model needs to be callable")
        if not callable(self.streak_model):
            raise TypeError("streak_model needs to be callable")
        super(BiphasicAxonMapSpatial, self)._build()
        if self.engine == 'jax':
            # Cache axon_contrib for fast access later
            self.axon_contrib = jax.device_put(self.axon_contrib, jax.devices()[0])

    def _predict_spatial(self, earray, stim):
        """Predicts the percept"""
        if not isinstance(earray, ElectrodeArray):
            raise TypeError("Implant must be of type ElectrodeArray but it is " +
                            str(type(earray)))
        if not isinstance(stim, Stimulus):
            raise TypeError(
                "Stim must be of type Stimulus but it is " + str(type(stim)))

        elec_params = []
        for e in stim.electrodes:
            amp = stim.metadata['electrodes'][str(e)]['metadata']['amp']
            if amp == 0:
                continue
            freq = stim.metadata['electrodes'][str(e)]['metadata']['freq']
            pdur = stim.metadata['electrodes'][str(e)]['metadata']['phase_dur']
            elec_params.append([freq, amp, pdur, earray[e].x, earray[e].y])
        elec_params = np.array(elec_params, dtype='float32')

        if self.engine != 'jax':
            bright_effects = self.bright_model(elec_params[:, 0], elec_params[:, 1], elec_params[:, 2])
            size_effects = self.size_model(elec_params[:, 0], elec_params[:, 1], elec_params[:, 2])
            streak_effects = self.streak_model(elec_params[:, 0], elec_params[:, 1], elec_params[:, 2])
            return fast_biphasic_axon_map(
                elec_params[:, 1],
                bright_effects,
                size_effects,
                streak_effects,
                elec_params[:, 3], elec_params[:, 4],
                self.axon_contrib,
                self.axon_idx_start.astype(np.uint32),
                self.axon_idx_end.astype(np.uint32),
                self.rho, self.thresh_percept)
        else:
            return self._predict_spatial_jax(elec_params)

    def _predict_spatial_jax(self, elec_params):
        """
        A stripped version of predict_percept that takes only electrode parameters, and returns only a numpy array
        This is a better function to use when the stimulus is guaranteed to be safe,
        and the percept object isn't used (e.g. inside a neural network), just the data in the percept

        To jit / differentiate this function:
        - Make effect models jit-able (maximum operation)
        - remove state (pass in axon_contib, rho, lambda, thresh_percept) (optional, speeds up)

        Parameters:
        ------------
        elec_params : np.array of shape (n_elecs, 5)
            The 5 columns correspond to freq, amp, pulse duration, x, y on each electrode 

        Returns:
        ------------
        resp : np.array() representing the resulting percept, shape (:, 1)
        """
        bright_effects = self.bright_model(elec_params[:, 0], elec_params[:, 1], elec_params[:, 2])
        size_effects = self.size_model(elec_params[:, 0], elec_params[:, 1], elec_params[:, 2])
        streak_effects = self.streak_model(elec_params[:, 0], elec_params[:, 1], elec_params[:, 2])
        eparams = np.stack([bright_effects, size_effects, streak_effects, elec_params[:, 3], elec_params[:, 4]], axis=1)
        
        resp = biphasic_axon_map_jax(eparams,
                                        self.axon_contrib,
                                        self.rho, 
                                        self.axlambda, 
                                        self.thresh_percept).block_until_ready()
        return resp


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
            if self.engine == 'jax':
                return self._predict_spatial(implant.earray, stim)
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
        retinotopy : :py:class:`~pulse2percept.utils.VisualFieldMap`, optional
            An instance of a :py:class:`~pulse2percept.utils.VisualFieldMap`
            object that provides ``ret2dva`` and ``dva2ret`` methods.
            By default, :py:class:`~pulse2percept.utils.Watson2014Map` is
            used.
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

        Note: The stimuli should use amplitude as a factor of threshold,
        NOT raw amplitude in microamps

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
