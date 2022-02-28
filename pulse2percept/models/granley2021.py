"""`BiphasicAxonMapModel`"""
from functools import partial
import numpy as np
import sys
import copy

from . import AxonMapSpatial, Model
from ..implants import ProsthesisSystem, ElectrodeArray
from ..stimuli import BiphasicPulseTrain, Stimulus
from ..percepts import Percept
from ..utils import FreezeError
from .base import BaseModel, NotBuiltError
from ._granley2021 import fast_biphasic_axon_map


try:
    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = '0'
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, lax
    has_jax = True
except ImportError:
    has_jax = False


def cond_jit(fn, static_argnums=None):
    """ Conditional decorator for jax jit"""
    if has_jax:
        if static_argnums is None:
            return jit(fn)
        else:
            return jit(fn, static_argnums=static_argnums)
    else:
        return fn


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
            'a4': 0
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
        be scaled by (F_bright).
        Must support batching (freq, amp, pdur may be arrays)
        """
        # Scale amp according to pdur (Eq 3 in paper) and then calculate F_{bright}
        F_bright = self.predict_freq_amp(amp * self.scale_threshold(pdur), freq)
        return F_bright


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

    def __init__(self, rho, engine="serial", **params):
        super(DefaultSizeModel, self).__init__(**params)
        self.rho = rho
        self.engine = engine
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
        Must support batching (freq, amp, pdur may be arrays)
        """
        min_f_size = self.min_rho**2 / self.rho**2
        F_size = self.a5 * amp * self.scale_threshold(pdur) + self.a6
        if self.engine == 'jax':
            return jnp.maximum(F_size, min_f_size)
        else:
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

    def __init__(self, axlambda, engine='serial', **params):
        super(DefaultStreakModel, self).__init__(**params)
        self.axlambda = axlambda
        self.engine = engine
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
        Must support batching (freq, amp, pdur may be arrays)
        """
        min_f_streak = self.min_lambda**2 / self.axlambda ** 2
        F_streak = self.a9 - self.a7 * pdur ** self.a8
        if self.engine == 'jax':
            return jnp.maximum(F_streak, min_f_streak)
        else:
            return np.maximum(F_streak, min_f_streak)


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

    .. note::
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
        n_gray : int, optional
            The number of gray levels to use. If an integer is given, k-means
            clustering is used to compress the color space of the percept into
            ``n_gray`` bins. If None, no compression is performed.
        noise : float or int, optional
            Adds salt-and-pepper noise to each percept frame. An integer will be
            interpreted as the number of pixels to subject to noise in each 
            frame. A float between 0 and 1 will be interpreted as a ratio of 
            pixels to subject to noise in each frame.
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
        n_threads: int, optional
            Number of CPU threads to use during parallelization using OpenMP. 
            Defaults to max number of user CPU cores.
    """

    def __init__(self, **params):
        super(BiphasicAxonMapSpatial, self).__init__(**params)
        if self.bright_model is None:
            self.bright_model = DefaultBrightModel()
        if self.size_model is None:
            self.size_model = DefaultSizeModel(self.rho, self.engine)
        if self.streak_model is None:
            self.streak_model = DefaultStreakModel(self.axlambda, self.engine)
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
            raise AttributeError(f"{attr} not found")
        # Check if bright/size/streak model has param
        for m in [self.bright_model, self.size_model, self.streak_model]:
            if hasattr(m, attr):
                return getattr(m, attr)
        raise AttributeError(f"{attr} not found")

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
        if name not in ['bright_model', 'size_model', 'streak_model', 'is_built', '_is_built']:
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
            err_str = (f"'{name}' not found. You cannot add attributes to "
                       f"{self.__class__.__name__} outside the constructor.")
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
        }
        return {**base_params, **params}

    def _build(self):
        if not callable(self.bright_model):
            raise TypeError("bright_model needs to be callable")
        if not callable(self.size_model):
            raise TypeError("size_model needs to be callable")
        if not callable(self.streak_model):
            raise TypeError("streak_model needs to be callable")
        if self.engine == 'jax' and not has_jax:
            raise ImportError("Engine was chosen as jax, but jax is not installed. "
                              "You can install it with 'pip install \"jax[cpu]\"' for cpu "
                              "or following https://github.com/google/jax#installation for gpu")

        super(BiphasicAxonMapSpatial, self)._build()
        if self.engine == 'jax':
            # Clear previously cached functions
            self._predict_spatial_jax = jit(self._predict_spatial_jax)
            self._predict_spatial_batched = jit(self._predict_spatial_batched)
            # Cache axon_contrib for fast access later
            self.axon_contrib = jax.device_put(
                jnp.array(self.axon_contrib), jax.devices()[0])

    def _predict_spatial(self, earray, stim):
        """Predicts the percept"""
        if not isinstance(earray, ElectrodeArray):
            raise TypeError("Implant must be of type ElectrodeArray but it is " +
                            str(type(earray)))
        if not isinstance(stim, Stimulus):
            raise TypeError(
                "Stim must be of type Stimulus but it is " + str(type(stim)))
        elec_params = []
        x = []
        y = []
        for e in stim.electrodes:
            amp = stim.metadata['electrodes'][str(e)]['metadata']['amp']
            if amp == 0:
                continue
            freq = stim.metadata['electrodes'][str(e)]['metadata']['freq']
            pdur = stim.metadata['electrodes'][str(e)]['metadata']['phase_dur']
            elec_params.append([freq, amp, pdur])
            x.append(earray[e].x)
            y.append(earray[e].y)
        elec_params = np.array(elec_params, dtype=np.float32)
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        if self.engine != 'jax':
            bright_effects = np.array(self.bright_model(elec_params[:, 0], elec_params[:, 1], elec_params[:, 2]),
                                      dtype=np.float32).reshape((-1))
            size_effects = np.array(self.size_model(elec_params[:, 0], elec_params[:, 1], elec_params[:, 2]),
                                    dtype=np.float32).reshape((-1))
            streak_effects = np.array(self.streak_model(elec_params[:, 0], elec_params[:, 1], elec_params[:, 2]),
                                      dtype=np.float32).reshape((-1))
            amps = np.array(elec_params[:, 1], dtype=np.float32).reshape((-1))
            return fast_biphasic_axon_map(
                amps,
                bright_effects,
                size_effects,
                streak_effects,
                x, y,
                self.axon_contrib,
                self.axon_idx_start.astype(np.uint32),
                self.axon_idx_end.astype(np.uint32),
                self.rho, self.thresh_percept,
                self.n_threads)
        else:
            return self._predict_spatial_jax(elec_params[:, :3], x, y)

    def predict_one_point_jax(self, axon, eparams, x, y, rho):
        """ Predicts the brightness contribution from each axon segment for each pixel"""
        d2_el = (axon[:, 0, None] - x)**2 + (axon[:, 1, None] - y)**2
        intensities = eparams[:, 0] * jnp.exp(-d2_el / (2. * rho**2 * eparams[:, 1])) * (
            axon[:, 2, None] ** (1./eparams[:, 2]))
        return jnp.sum(intensities, axis=1)

    @partial(cond_jit, static_argnums=[0])
    def biphasic_axon_map_jax(self, eparams, x, y, axon_segments, rho, thresh_percept):
        """ Predicts the spatial response of BiphasicAxonMapModel using Jax

        Parameters:
        -------------
        eparams : jnp.array with shape (n_elecs, 3)
            Brightness, size, and streak length effect on each electrode
        x, y : jnp.array with shape (n_elecs)
            x and y coordinate of each electrode
        axon_segments : jnp.array with shape (n_points, n_ax_segments, 3)
            Closest axon segment to each simulated point, as returned by calc_axon_sensitivities
        rho : float
            The rho parameter of the axon map model: exponential decay constant
            (microns) away from the axon.
        axlambda : float
            The lambda parameter of the axon map model: exponential decay constant
            (microns) away from the cell body along the axon
        thresh_percept : float
            Spatial responses smaller than ``thresh_percept`` will be set to zero
        """
        I = jnp.max(jax.vmap(self.predict_one_point_jax, in_axes=[0, None, None, None, None])(
            axon_segments,
            eparams, x, y,
            rho), axis=1)
        I = (I > thresh_percept) * I
        return I

    @partial(cond_jit, static_argnums=[0])
    def _predict_spatial_jax(self, elec_params, x, y):
        """
        A stripped version of _predict_spatial that takes only electrode parameters, 
        and returns only a numpy array
        This is a better function to use when the stimulus is guaranteed to be safe,
        and the percept object isn't used, just the data in the percept (e.g. inside a neural network)

        Parameters:
        ------------
        elec_params : np.array with shape (n_electrodes, 3)
            Frequency, amplitude, and pulse duration for each electrode
        x, y: np.array with shape (n_electrodes)
            x and y coordinates of electrodes

        Returns:
        ------------
        resp : flattened np.array() representing the resulting percept, shape (:, 1)
        """
        bright_effects = jnp.array(self.bright_model(elec_params[:, 0],
                                                     elec_params[:, 1],
                                                     elec_params[:, 2])).reshape((-1))
        size_effects = jnp.array(self.size_model(elec_params[:, 0],
                                                 elec_params[:, 1],
                                                 elec_params[:, 2])).reshape((-1))
        streak_effects = jnp.array(self.streak_model(elec_params[:, 0],
                                                     elec_params[:, 1],
                                                     elec_params[:, 2])).reshape((-1))
        eparams = jnp.stack(
            [bright_effects, size_effects, streak_effects], axis=1)

        resp = self.biphasic_axon_map_jax(eparams, x, y,
                                          self.axon_contrib,
                                          self.rho,
                                          self.thresh_percept)
        return resp

    @partial(cond_jit, static_argnums=[0])
    def _predict_spatial_batched(self, elec_params, x, y):
        """ A batched version of _predict_spatial_jax
        Parameters:
        -------------
        elec_params : np.array with shape (batch_size, n_electrodes, 3)
            The 3 columns are freq, amp, pdur for each electrode
        x, y: np.array with shape (n_electrodes)
            x and y coordinates of electrodes
        Returns:
        ------------
        resp : np.array() representing the resulting percepts, shape (batch_size, :, 1)
        """
        bright_effects = self.bright_model(elec_params[:, :, 0],
                                           elec_params[:, :, 1],
                                           elec_params[:, :, 2])
        size_effects = self.size_model(elec_params[:, :, 0],
                                       elec_params[:, :, 1],
                                       elec_params[:, :, 2])
        streak_effects = self.streak_model(elec_params[:, :, 0],
                                           elec_params[:, :, 1],
                                           elec_params[:, :, 2])
        eparams = jnp.stack(
            [bright_effects, size_effects, streak_effects], axis=2)

        def predict_one(e_params):
            return self.biphasic_axon_map_jax(e_params, x, y,
                                              self.axon_contrib,
                                              self.rho,
                                              self.thresh_percept)
        resps = lax.map(predict_one, eparams)

        return resps

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
                        f"All stimuli must be BiphasicPulseTrains with no " +
                        f"delay dur (Failing electrode: {ele})")
        if isinstance(implant, ProsthesisSystem):
            if implant.eye != self.eye:
                raise ValueError(f"The implant is in {implant.eye} but the model was "
                                 f"built for {self.eye}.")
        if not self.is_built:
            raise NotBuiltError("Yout must call ``build`` first.")
        if not isinstance(implant, ProsthesisSystem):
            raise TypeError(f"'implant' must be a ProsthesisSystem object, "
                            f"not {type(implant)}.")
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

    def predict_percept_batched(self, implant, stims, t_percept=None):
        """
        Batched version of predict_percept
        Only supported with jax engine

        This is significantly faster if you do not batch ALL of your percepts, but rather, split
        them into chunks (128 - 256 percepts each) and repeatedly call that. 
        This is because jax has to compile on the first call, so repeated calls 
        is much faster.

        Parameters
            ----------
            implant: :py:class:`~pulse2percept.implants.ProsthesisSystem`
                A valid prosthesis system. 
            stims : list of stimuli
                A percept will be predicted for each stimulus. Each stimulus
                must be a collection of :py:class:`~pulse2percept.stimuli.BiphasicPulseTrains'
            t_percept: float or list of floats, optional
                The time points at which to output a percept (ms).
                If None, ``implant.stim.time`` is used.

            Returns
            -------
            percepts: list of :py:class:`~pulse2percept.models.Percept`
                A list of Percept objects whose ``data`` container has dimensions Y x X x 1.
        """
        if self.engine != 'jax':
            raise ImportError(
                "Batched predict percept is not supported unless engine is jax")

        stims = [Stimulus(s) for s in stims]
        # Make sure all stimuli are BiphasicPulseTrains
        for stim in stims:
            if not isinstance(stim, BiphasicPulseTrain):
                # Could still be a stimulus where each electrode has a biphasic pulse train
                # or a 0 stimulus
                for i, (ele, params) in enumerate(stim.metadata
                                                  ['electrodes'].items()):
                    if (params['type'] != BiphasicPulseTrain or
                            params['metadata']['delay_dur'] != 0) and \
                            np.any(stim[i]):
                        raise TypeError(
                            f"All stimuli must be BiphasicPulseTrains with no "
                            f"delay dur (Failing electrode: {ele})")

        if not self.is_built:
            raise NotBuiltError("Yout must call ``build`` first.")
        if not isinstance(implant, ProsthesisSystem):
            raise TypeError(f"'implant' must be a ProsthesisSystem object, "
                            f"not {type(implant)}.")
        if implant.eye != self.eye:
            raise ValueError(f"The implant is in {implant.eye} but the model was "
                             f"built for {self.eye}.")

        # Currently all stimuli must have same electrodes
        all_elecs = list(set().union(
            *[s.metadata['electrodes'].keys() for s in stims]))
        max_elecs = len(all_elecs)
        # Compute stimulus parameters for all electrodes
        eparams = np.zeros((len(stims), max_elecs, 3))
        for idx_stim, stim in enumerate(stims):
            if stim is None:
                continue
            for elec in stim.metadata['electrodes'].keys():
                idx_elec = all_elecs.index(elec)
                elec_metadata = stim.metadata['electrodes'][elec]['metadata']
                eparams[idx_stim, idx_elec, 0] = elec_metadata['freq']
                eparams[idx_stim, idx_elec, 1] = elec_metadata['amp']
                eparams[idx_stim, idx_elec, 2] = elec_metadata['phase_dur']
        x = np.array([implant.earray[elec].x for elec in all_elecs])
        y = np.array([implant.earray[elec].y for elec in all_elecs])
        # Predict percepts (returns only numpy array)
        percepts_data = self._predict_spatial_batched(eparams, x, y)
        # Convert into percepts
        percepts = []
        for idx_percept, pdata in enumerate(percepts_data):
            if t_percept is None:
                n_time = 1
            else:
                n_time = len(t_percept)
            resp = np.zeros(list(self.grid.x.shape) + [n_time])
            # Response goes in first frame
            resp[:, :, 0] = pdata.reshape(self.grid.x.shape)
            percepts.append(Percept(resp, space=self.grid, time=t_percept,
                                    metadata={'stim': stims[idx_percept].metadata}))
        return percepts


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

    .. note::
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
        n_gray : int, optional
            The number of gray levels to use. If an integer is given, k-means
            clustering is used to compress the color space of the percept into
            ``n_gray`` bins. If None, no compression is performed.
        noise : float or int, optional
            Adds salt-and-pepper noise to each percept frame. An integer will be
            interpreted as the number of pixels to subject to noise in each 
            frame. A float between 0 and 1 will be interpreted as a ratio of 
            pixels to subject to noise in each frame.
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
        n_threads: int, optional
            Number of CPU threads to use during parallelization using OpenMP. 
            Defaults to max number of user CPU cores.

    """

    def __init__(self, **params):
        super(BiphasicAxonMapModel, self).__init__(
            spatial=BiphasicAxonMapSpatial(), temporal=None, **params)

    def predict_percept(self, implant, t_percept=None):
        """Predict a percept
        Overrides base predict percept to keep desired time axes
        .. important::

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
            raise TypeError(f"'implant' must be a ProsthesisSystem object, not "
                            f"{type(implant)}.")
        if implant.stim is None or (not self.has_space and not self.has_time):
            # Nothing to see here:
            return None
        resp = self.spatial.predict_percept(implant, t_percept=t_percept)
        return resp