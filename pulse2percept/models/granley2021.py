""":py:class:`~pulse2percept.models.BiphasicAxonMapModel`,
   :py:class:`~pulse2percept.models.BiphasicAxonMapSpatial` [Granley2021]_"""
import numpy as np
import sys

from . import AxonMapSpatial, Model
from ..implants import ProsthesisSystem, ElectrodeArray
from ..stimuli import BiphasicPulseTrain, Stimulus
from ..percepts import Percept
from ..units import as_value, um
from ..utils import FreezeError, rename_parameter
from ..utils.base import has_own_attr
from .base import NotBuiltError, BaseModel, _require_stim_dimension
from ._granley2021 import fast_biphasic_axon_map

# `find_threshold` bisects on a scaled copy of the stimulus *data*. This model
# reads amplitude from the stimulus metadata instead, where it means a
# multiple of threshold rather than a current, so scaling the data leaves the
# prediction untouched and the search cannot converge:
_FIND_THRESHOLD_MSG = (
    "{cls} does not support find_threshold. It takes amplitude as a multiple "
    "of threshold and reads it from the stimulus metadata, not from the "
    "stimulus data, so scaling the data leaves the percept unchanged. Vary "
    "`amp` when building the BiphasicPulseTrain instead."
)


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
        a0*pdur + a1.
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
        .. note::
            This equation has been updated from the original paper, and has been refit
            to data from Argus II users from Horsager et al. 2009.
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
        a0*pdur + a1.
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

    def get_param_units(self):
        """Return a dict of the units that parameters are stored in"""
        # `rho` is a constructor argument rather than a default parameter, but
        # it is still an attribute this object normalizes, and declaring it
        # here is what makes `DefaultSizeModel(0.2 * mm)` work. a0-a6 are
        # regression coefficients from the paper and take plain numbers:
        return {**super().get_param_units(), 'rho': um, 'min_rho': um}

    def scale_threshold(self, pdur):
        """ 
        Based on eq 3 in paper, this function produces the factor that amplitude
        will be scaled by to produce a_tilde. Computes A_0 * t + A_1 (1/threshold)
        .. note::
            This equation has been updated from the original paper, and has been refit
            to data from Argus II users from Horsager et al. 2009.
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
        return np.maximum(F_size, min_f_size)


class DefaultStreakModel(BaseModel):
    """
    Default model to be used for streak length (lambda) scaling in BiphasicAxonMapModel
    Implements Eq 6 from [Granley2021]_ 
    Fit using data from [Weitz2015]_

    Parameters:
    ------------
    lam :  float32
        ``lam`` parameter of BiphasicAxonMapModel (axonal decay rate)

        .. versionchanged:: 0.10.0

            Renamed from ``axlambda``. The old name still works as a keyword
            argument, but is deprecated and will be removed in v0.11.0.
    a7, a8, a9: float, optional
        Regression coefficients for streak length vs pulse duration (Eq 6)
        F_streak = -a7*pdur^a8 + a9
    """

    @rename_parameter('axlambda', 'lam', deprecated_version='0.10.0',
                      removed_version='0.11.0')
    def __init__(self, lam, **params):
        super(DefaultStreakModel, self).__init__(**params)
        self.lam = lam
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

    def get_param_units(self):
        """Return a dict of the units that parameters are stored in"""
        # `lam` is a constructor argument rather than a default parameter; see
        # `DefaultSizeModel.get_param_units`. `min_lambda` is a floor on it,
        # so it is a length too:
        return {**super().get_param_units(), 'lam': um, 'min_lambda': um}

    def __call__(self, freq, amp, pdur):
        """
        Main function to be called by BiphasicAxonMapModel
        Outputs value for each electrode that lambda should be scaled by (F_streak)
        Must support batching (freq, amp, pdur may be arrays)
        """
        min_f_streak = self.min_lambda**2 / self.lam ** 2
        F_streak = self.a9 - self.a7 * pdur ** self.a8
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

    .. important::
    
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
        lam: double, optional
            Exponential decay constant along the axon(microns).

            .. versionchanged:: 0.10.0

                Renamed from ``axlambda``, which reads poorly next to ``rho``.
                The old name still works, but is deprecated and will be
                removed in v0.11.0.
        rho: double, optional
            Exponential decay constant away from the axon(microns).
        min_current_spread: float, optional
            An electrode is skipped at axon segments where its current spread
            has decayed below this fraction of its peak. The decay is scaled
            per electrode by that electrode's ``F_size``. The default (1e-8)
            is a deliberate approximation: what gets dropped is the
            exponential *times* that electrode's ``F_bright``, summed over
            the skipped electrodes, so the error at a point is bounded by
            ``min_current_spread`` times the summed ``F_bright`` across
            electrodes.
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
        step : int, double, tuple
            Step size for the range of (x,y) values to simulate (in degrees of
            visual angle). For example, to create a grid with x values [0, 0.5, 1]
            use ``xrange=(0, 1)`` and ``step=0.5``. Pass a tuple to give the x
            and y axes different step sizes.

            .. versionchanged:: 0.10.0

                Renamed from ``xystep``, which suggested that one step size
                applies to both axes. The old name still works, but is
                deprecated and will be removed in v0.11.0.
        grid_type : {'rectangular', 'hexagonal'}
            Whether to simulate points on a rectangular or hexagonal grid
        vfmap : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
            An instance of a :py:class:`~pulse2percept.topography.VisualFieldMap`
            object that provides retinotopic mappings.
            By default, :py:class:`~pulse2percept.topography.Watson2014Map` is
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
        n_threads : int, optional
            Number of CPU threads to use during parallelization using OpenMP.
            Defaults to max number of user CPU cores.
        n_jobs : int, optional
            Alias for ``n_threads``; ``None`` or ``-1`` uses every core.
    """

    def __init__(self, **params):
        super(BiphasicAxonMapSpatial, self).__init__(**params)
        if self.bright_model is None:
            self.bright_model = DefaultBrightModel()
        if self.size_model is None:
            self.size_model = DefaultSizeModel(self.rho)
        if self.streak_model is None:
            self.streak_model = DefaultStreakModel(self.lam)
        for key, val in params.items():
            if key in ['bright_model', 'size_model', 'streak_model']:
                continue
            # `super().__init__` has already warned about any deprecated name
            # among these, so set the current one rather than warn twice:
            spec = self._renamed_params.get(key)
            setattr(self, spec.new_name if spec else key, val)

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
        # Try to set it ourselves, but can't use get_attr. Probe the type
        # rather than read the attribute, so that a `deprecated_alias` does
        # not warn just for being asked whether it exists:
        if has_own_attr(self, name):
            # if we get here, we have the attribute, not (neccesarily) an effects model
            try:
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

        super(BiphasicAxonMapSpatial, self)._build()

    def _predict_spatial(self, earray, stim):
        """Predicts the percept"""
        if not isinstance(earray, ElectrodeArray):
            raise TypeError("Implant must be of type ElectrodeArray but it is " +
                            str(type(earray)))
        if not isinstance(stim, Stimulus):
            raise TypeError(
                "Stim must be of type Stimulus but it is " + str(type(stim)))
        elec_params = []
        active = []
        try:
            for e in stim.electrodes:
                amp = stim.metadata['electrodes'][str(e)]['metadata']['amp']
                if amp == 0:
                    continue
                freq = stim.metadata['electrodes'][str(e)]['metadata']['freq']
                pdur = stim.metadata['electrodes'][str(e)]['metadata']['phase_dur']
                elec_params.append([freq, amp, pdur])
                active.append(e)
        except KeyError:
            raise TypeError(f"All stimuli must be BiphasicPulseTrains with no " +
                            f"delay dur")
        elec_params = np.array(elec_params, dtype=np.float32)
        # Only the electrodes that are actually driven, in the order they were
        # collected above:
        xyz = earray.coordinates(self.space_unit, electrodes=active)
        x = np.ascontiguousarray(xyz[:, 0], dtype=np.float32)
        y = np.ascontiguousarray(xyz[:, 1], dtype=np.float32)

        bright_effects = np.array(self.bright_model(elec_params[:, 0], elec_params[:, 1], elec_params[:, 2]),
                                  dtype=np.float32).reshape((-1))
        size_effects = np.array(self.size_model(elec_params[:, 0], elec_params[:, 1], elec_params[:, 2]),
                                dtype=np.float32).reshape((-1))
        streak_effects = np.array(self.streak_model(elec_params[:, 0], elec_params[:, 1], elec_params[:, 2]),
                                  dtype=np.float32).reshape((-1))
        amps = np.array(elec_params[:, 1], dtype=np.float32).reshape((-1))
        # A non-finite factor propagates through the kernel's exponent into
        # the segment brightness, where abs(nan) > abs(px_bright). So we need
        # to catch it here:
        for name, effects in (('bright_model', bright_effects),
                              ('size_model', size_effects),
                              ('streak_model', streak_effects)):
            if not np.all(np.isfinite(effects)):
                raise ValueError(f"{type(self).__name__}.{name} returned a "
                                 f"non-finite scaling factor. Scaling factors "
                                 f"must be finite.")
        # The kernel rescales each segment's sensitivity by 1 / F_streak and
        # each phosphene's size by F_size, both in an exponent, so neither of
        # them may be negative:
        for name, effects in (('size_model', size_effects),
                              ('streak_model', streak_effects)):
            if np.any(effects <= 0):
                raise ValueError(f"{type(self).__name__}.{name} returned a "
                                 f"non-positive scaling factor "
                                 f"({effects.min()}). Scaling factors must be "
                                 f"greater than zero.")
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
            self._cutoff_r2(self.rho),
            self.n_threads)

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
            The time points at which to output a percept (ms). This
            model's numerical contract is fixed to milliseconds.
            If None, ``implant.stim.time`` is used.
            May be given as a unitful quantity (e.g. ``[0, 20] * ms``);
            see :py:mod:`pulse2percept.units`.

        Returns
        -------
        percept: :py:class:`~pulse2percept.models.Percept`
            A Percept object whose ``data`` container has dimensions Y x X x 1.
            Will return None if ``implant.stim`` is None.
        """
        if not self.is_built:
            raise NotBuiltError("Yout must call ``build`` first.")
        if not isinstance(implant, ProsthesisSystem):
            raise TypeError(f"'implant' must be a ProsthesisSystem object, "
                            f"not {type(implant)}.")
        if implant.eye != self.eye:
            raise ValueError(f"The implant is in {implant.eye} but the model "
                             f"was built for {self.eye}.")
        if implant.stim is None:
            return None
        # What physical quantity the stimulus is comes before what waveform
        # it is: a picture is not an unsuitable pulse train, it is not a
        # current at all, and saying so is more use than asking it for pulse
        # metadata it was never going to have.
        _require_stim_dimension(self, implant.stim)
        # Make sure stimulus is a BiphasicPulseTrain:
        if not isinstance(implant.stim, BiphasicPulseTrain):
            # Could still be a stimulus where each electrode has a biphasic pulse train
            # or a 0 stimulus
            try:
                for i, (ele, params) in enumerate(implant.stim.metadata
                                                ['electrodes'].items()):
                    if (params['type'] != BiphasicPulseTrain or
                            params['metadata']['delay_dur'] != 0) and \
                            np.any(implant.stim[i]):
                        raise TypeError(
                            f"All stimuli must be BiphasicPulseTrains with no " +
                            f"delay dur (Failing electrode: {ele})")
            except KeyError:
                raise TypeError(f"All stimuli must be BiphasicPulseTrains with no " +
                                f"delay dur")
        t_percept = as_value(t_percept, self.time_unit, 't_percept')
        stim = implant.stim
        # `np.array([t]).size` rather than `len(t)`, so that the documented
        # scalar spelling `t_percept=20` counts as one time point instead
        # of raising -- the same idiom `SpatialModel.predict_percept` uses:
        n_time = 1 if t_percept is None else np.array([t_percept]).size
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
                       time_unit=self.time_unit,
                       metadata={'stim': stim.metadata})

    def find_threshold(self, implant, bright_th, amp_range=(0, 999), amp_tol=1,
                       bright_tol=0.1, max_iter=100):
        """Not supported by this model

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(_FIND_THRESHOLD_MSG.format(
            cls=type(self).__name__))


class BiphasicAxonMapModel(Model):
    """ BiphasicAxonMapModel of [Granley2021]_ (standalone model)

    An AxonMapModel where phosphene brightness, size, and streak length scale
    according to amplitude, frequency, and pulse duration.

    All stimuli must be BiphasicPulseTrains.

    This model is different than other spatial models in that it calculates
    one representative percept from all time steps of the stimulus.

    Brightness, size, and streak length scaling are controlled by the parameters
    bright_model, size_model, and streak model respectively. By default, these are
    set to classes that implement Eqs 3-6 from Granley 2021. These models can be
    individually customized by setting the bright_model, size_model, or streak_model
    to any python callable with signature f(freq, amp, pdur).

    .. important::

        Stimuli should pass amplitude as a factor of threshold, NOT as raw
        amplitude in microamps.

        This model interacts with `Stimulus` objects by reading the intended
        amplitude, frequency, and pulse duration from their metadata, not
        from the raw stimulus data. The arithmetic operators keep that
        metadata in sync, so scaling a pulse train (``pt * 2``) or the
        stimulus assembled from one (``implant.stim * 2``) does change the
        percept, while editing the data array in place does not.

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
        ^^^^^^^^
        lam: double, optional
            Exponential decay constant along the axon(microns).

            .. versionchanged:: 0.10.0

                Renamed from ``axlambda``, which reads poorly next to ``rho``.
                The old name still works, but is deprecated and will be
                removed in v0.11.0.
        rho: double, optional
            Exponential decay constant away from the axon(microns).
        min_current_spread: float, optional
            An electrode is skipped at axon segments where its current spread
            has decayed below this fraction of its peak. The decay is scaled
            per electrode by that electrode's ``F_size``. The default (1e-8)
            is a deliberate approximation: what gets dropped is the
            exponential *times* that electrode's ``F_bright``, summed over
            the skipped electrodes, so the error at a point is bounded by
            ``min_current_spread`` times the summed ``F_bright`` across
            electrodes. That is negligible for a typical array, but it grows
            with both array size and brightness scaling, and it can zero out
            points that are merely dim. Set to 0 to sum over every electrode
            and get the exact result.
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
        step : int, double, tuple
            Step size for the range of (x,y) values to simulate (in degrees of
            visual angle). For example, to create a grid with x values [0, 0.5, 1]
            use ``xrange=(0, 1)`` and ``step=0.5``. Pass a tuple to give the x
            and y axes different step sizes.

            .. versionchanged:: 0.10.0

                Renamed from ``xystep``, which suggested that one step size
                applies to both axes. The old name still works, but is
                deprecated and will be removed in v0.11.0.
        grid_type : {'rectangular', 'hexagonal'}
            Whether to simulate points on a rectangular or hexagonal grid
        vfmap : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
            An instance of a :py:class:`~pulse2percept.topography.VisualFieldMap`
            object that provides retinotopic mappings.
            By default, :py:class:`~pulse2percept.topography.Watson2014Map` is
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
        n_threads : int, optional
            Number of CPU threads to use during parallelization using OpenMP.
            Defaults to max number of user CPU cores.
        n_jobs : int, optional
            Alias for ``n_threads``; ``None`` or ``-1`` uses every core.

    """

    def __init__(self, **params):
        super(BiphasicAxonMapModel, self).__init__(
            spatial=BiphasicAxonMapSpatial(), temporal=None, **params)

    def predict_percept(self, implant, t_percept=None):
        """Predict a percept.

        Overrides base predict percept to keep desired time axes

        .. note::

            You must call ``build`` before calling ``predict_percept``.

        .. important::

            Stimuli should pass amplitude as a factor of threshold,
            NOT as raw amplitude in microamps.

            The model interacts with `Stimulus` objects by reading the
            intended amplitude, frequency, and pulse duration from their
            metadata, not from the raw stimulus data. Editing the data
            array in place will not change the predicted percept.

        Parameters
        ----------
        implant: :py:class:`~pulse2percept.implants.ProsthesisSystem`
            A valid prosthesis system. A stimulus can be passed via
            :py:meth:`~pulse2percept.implants.ProsthesisSystem.stim`.
        t_percept: float or list of floats, optional
            The time points at which to output a percept (ms). This
            model's numerical contract is fixed to milliseconds.
            If None, ``implant.stim.time`` is used.
            May be given as a unitful quantity (e.g. ``[0, 20] * ms``);
            see :py:mod:`pulse2percept.units`.

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
        _require_stim_dimension(self, implant.stim)
        resp = self.spatial.predict_percept(implant, t_percept=t_percept)
        return resp

    def find_threshold(self, implant, bright_th, amp_range=(0, 999), amp_tol=1,
                       bright_tol=0.1, max_iter=100, t_percept=None):
        """Not supported by this model

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(_FIND_THRESHOLD_MSG.format(
            cls=type(self).__name__))
