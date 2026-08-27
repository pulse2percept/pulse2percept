""":py:class:`~pulse2percept.models.BiphasicAxonMapModel`,
   :py:class:`~pulse2percept.models.BiphasicAxonMapSpatial` [Granley2021]_"""
import numpy as np
from copy import deepcopy

from . import AxonMapSpatial, Model
from ..implants import ElectrodeArray
from ..stimuli import BiphasicPulseTrain, Stimulus
from ..percepts import Percept
from ..units import as_value, um, xTh
from ..utils import FreezeError
from ..utils.base import has_own_attr
from .base import BaseModel, _require_stim_dimension
from ._granley2021 import fast_biphasic_axon_map

# Safety limit for locating delayed temporal peaks.
_PEAK_SEARCH_DOUBLINGS = 4


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
    a7, a8, a9: float, optional
        Regression coefficients for streak length vs pulse duration (Eq 6)
        F_streak = -a7*pdur^a8 + a9
    """

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


#: What to do about a pulse whose amplitude is a current of unknown
#: threshold. Formatted with the failing electrode.
_NO_THRESHOLD_MSG = (
    "This model takes amplitude as a multiple of perceptual threshold, and "
    "electrode {electrode} is driven at {amp:.1f} uA with no threshold to "
    "measure that against. Either give `amp` in threshold units (e.g. "
    "`2 * xTh`), pass `threshold_amp` to the BiphasicPulseTrain, or set "
    "`implant.thresholds`."
)


def _pulse_train_params(stim):
    """Return Granley pulse parameters for each driven electrode.

    Parameters are read from retained ``BiphasicPulseTrain`` objects
    without rendering their waveforms. Amplitudes are returned in
    multiples of threshold; zero-amplitude electrodes are omitted.

    Raises
    ------
    TypeError
        If a driven electrode is not a zero-delay
        :class:`~pulse2percept.stimuli.BiphasicPulseTrain`.
    ValueError
        If a current-valued amplitude has no threshold calibration.
    """
    sources = stim._structured_sources()
    if sources is None:
        # No pulse train is retained anywhere. A stimulus of nothing but
        # zeros is not one either, but it has always been answered with a
        # zero percept rather than an error -- and it is the only case that
        # costs a waveform this model otherwise never asks for:
        if not np.any(stim.data):
            return []
        raise TypeError("All stimuli must be BiphasicPulseTrains with no "
                        "delay dur")
    params = []
    for electrode, source in sources:
        # The exact class, not a subclass: this model's parameters are the
        # ones `BiphasicPulseTrain` is made of, and nothing else has been
        # shown to mean the same thing to it.
        if type(source) is not BiphasicPulseTrain or source.delay_dur != 0:
            raise TypeError(f"All stimuli must be BiphasicPulseTrains with "
                            f"no delay dur (Failing electrode: {electrode})")
        if source.amp == 0:
            continue
        if source.amp_factor is None:
            raise ValueError(_NO_THRESHOLD_MSG.format(electrode=electrode,
                                                      amp=source.amp))
        params.append((electrode, source.freq, source.amp_factor,
                       source.phase_dur, source.stim_dur))
    return params


#: The three effect models are set up by ``BiphasicAxonMapSpatial.__init__``
#: and are excluded from its attribute forwarding, in both directions.
_EFFECT_MODELS = ('bright_model', 'size_model', 'streak_model')


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

        When combined with a temporal model, the Granley prediction is treated
        as the peak spatial percept. The temporal model supplies a normalized
        brightness time course, yielding a space-time-separable percept
        ``P(x, y, t) = G(x, y) k(t)``. ``thresh_percept`` of the temporal
        model is ignored, because its response is normalized before being
        applied to the Granley percept.

        :py:class:`~pulse2percept.models.FadingTemporal` and
        :py:class:`~pulse2percept.models.AlphaTemporal` are the generic
        envelopes to reach for: the first fades from stimulus onset, the
        second rises to a peak first.

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`
        The implant whose stimulation this model predicts. Required before
        building or predicting.

        .. versionadded:: 0.11.0

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
        axons_range: (min, max) of float or Quantity, optional
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
        meridian_blend : float, optional
            Gaussian standard deviation (dva) for smoothing across the
            horizontal meridian. Default: 1. Set to 0 to disable.

            .. versionadded:: 0.10.0
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
    extra_stimulus_units = (xTh,)

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
        # Called when normal attribute access fails. The effect models
        # themselves are never forwarded through: asking one of them for
        # itself would recurse.
        if attr in _EFFECT_MODELS:
            raise AttributeError(f"{attr} not found")
        # `has_own_attr` rather than `getattr`, which would re-enter this
        # method: until the constructor has set all three, there is nothing to
        # forward through, and the caller (`Parametrized.__init__` setting a
        # default, say) wants the AttributeError anyway.
        if not all(has_own_attr(self, name) for name in _EFFECT_MODELS):
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
        # bright/size/streak model. Note that this runs even when the spatial
        # model already took the assignment above: `rho` and `lam` live in
        # both places and have to stay in step.
        if name not in _EFFECT_MODELS + ('is_built', '_is_built'):
            try:
                for m in [self.bright_model, self.size_model, self.streak_model]:
                    if hasattr(m, name):
                        setattr(m, name, value)
                        found = True
            except (AttributeError, FreezeError):
                pass
        if not found:
            # No legitimate destination, so let `Frozen` decide: a new
            # attribute is fine during construction and a `FreezeError`
            # afterwards.
            super().__setattr__(name, value)

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
        params = _pulse_train_params(stim)
        active = [p[0] for p in params]
        elec_params = np.array([p[1:4] for p in params],
                               dtype=np.float32).reshape((-1, 3))
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

    def _predict_prepared(self, stim, t_percept=None):
        """Predict the spatial response with model-specific time handling.

        Avoids intermediate computation used by the generic spatial path.

        Parameters
        ----------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus`
            The stimulus the bound implant delivers.
        t_percept: float or list of floats, optional
            The time points at which to output a percept (ms). This
            model's numerical contract is fixed to milliseconds.
            If None, the stimulus' own time points are used.
            May be given as a unitful quantity (e.g. ``[0, 20] * ms``);
            see :py:mod:`pulse2percept.units`.

        Returns
        -------
        percept: :py:class:`~pulse2percept.models.Percept`
            A Percept object whose ``data`` container has dimensions Y x X x 1.
            Will return None if ``stim`` is None.
        """
        if not self.is_built:
            self.build()
        if stim is None:
            return None
        # Determine what physical quantity the stimulus is:
        _require_stim_dimension(self, stim)
        # Determine which pulse trains the stimulus is made of:
        params = _pulse_train_params(stim)
        t_percept = as_value(t_percept, self.time_unit, 't_percept')
        n_time = 1 if t_percept is None else np.array([t_percept]).size
        if not params:
            # Nothing is driven above zero amplitude:
            resp = np.zeros(list(self.grid.x.shape) + [n_time],
                            dtype=np.float32)
        else:
            resp = np.zeros(list(self.grid.x.shape) + [n_time])
            # Response goes in first frame
            resp[:, :, 0] = self._predict_spatial(
                self.implant.earray, stim).reshape(self.grid.x.shape)
        # This override bypasses SpatialModel._predict_prepared:
        resp = self._postprocess_spatial(resp)
        return Percept(resp, space=self.grid, time=t_percept,
                       time_unit=self.time_unit,
                       metadata={'stim': stim.metadata})

    def _combine_temporal(self, percept, temporal, stim, t_percept):
        """Apply a normalized temporal response to the spatial percept."""
        dur = self._envelope_dur(stim)
        # A unit drive of the polarity this temporal model responds to (see
        # `TemporalModel._drive_sign`), held for exactly the stimulation
        # duration:
        envelope = Stimulus(np.array([[float(temporal._drive_sign), 0.0]]),
                            electrodes=['envelope'], time=[0, dur],
                            metadata=stim.metadata.get('user'))
        # Leave the caller's own model untouched:
        probe = deepcopy(temporal)
        probe.thresh_percept = 0
        peak = self._envelope_peak(probe, envelope)
        resp = probe.predict_percept(envelope, t_percept=t_percept)
        fade = resp.data.reshape(-1) / peak
        return Percept(percept.data[..., 0][..., np.newaxis] * fade,
                       space=self.grid, time=resp.time,
                       time_unit=probe.time_unit,
                       metadata={'stim': stim.metadata})

    @staticmethod
    def _envelope_peak(temporal, envelope):
        """Return the peak response to the canonical temporal drive."""
        dt = temporal.dt
        episode = envelope.times(temporal.time_unit)[-1]

        for _ in range(_PEAK_SEARCH_DOUBLINGS):
            t = np.arange(int(round(episode / dt)) + 1) * dt
            resp = temporal.predict_percept(envelope, t_percept=t).data
            if np.argmax(resp) < resp.size - 1:
                break
            episode *= 2
        else:
            raise ValueError(
                f"Could not locate the peak response of "
                f"{type(temporal).__name__} within {t[-1]:g} "
                f"{temporal.time_unit}."
            )

        peak = resp.max()
        if not np.isfinite(peak) or peak <= 0:
            raise ValueError(
                f"{type(temporal).__name__} produced no finite positive "
                f"response to the canonical drive."
            )
        return peak

    def _envelope_dur(self, stim):
        """Return the common duration of the active pulse trains."""
        durs = {p[4] for p in _pulse_train_params(stim)}
        if len(durs) > 1:
            raise NotImplementedError(
                f"{type(self).__name__} requires active electrodes to share "
                f"one stim_dur, not {sorted(durs)}."
            )
        if durs:
            return float(durs.pop())

        # No active electrodes; duration only determines the output time axis.
        sources = stim._structured_sources()
        if sources:
            return float(max(src.stim_dur for _, src in sources))
        return float(stim.time[-1])


class BiphasicAxonMapModel(Model):
    """ BiphasicAxonMapModel of [Granley2021]_ (standalone model)

    An AxonMapModel where phosphene brightness, size, and streak length scale
    according to amplitude, frequency, and pulse duration.

    All stimuli must be BiphasicPulseTrains.

    This model is different than other spatial models in that it calculates
    one representative percept from all time steps of the stimulus.

    Brightness, size, and streak length scaling are controlled by the
    parameters bright_model, size_model, and streak model respectively.
    By default, these are set to classes that implement Eqs 3-6 from
    [Granley2021]_. These models can be individually customized by setting
    the bright_model, size_model, or streak_model to any python callable
    with signature ``f(freq, amp, pdur)``.

    .. important::

        This model works in multiples of perceptual threshold, so give
        amplitude in :py:data:`~pulse2percept.units.xTh` (``2 * xTh``), or
        calibrate the electrode with ``threshold_amp`` or
        :py:attr:`~pulse2percept.implants.ProsthesisSystem.thresholds` and
        give it in microamps.

        Threshold is the 50%-detection current for a train at the same
        frequency and 0.45 ms phase duration [Granley2021]_. If frequency
        changes, the threshold may change too. The model applies the
        phase-duration correction itself; do not pre-correct the supplied
        threshold.

        This model reads amplitude, frequency and pulse duration off the
        :py:class:`~pulse2percept.stimuli.BiphasicPulseTrain` objects the
        stimulus is made of, not off its samples. Scaling a pulse train
        (``pt * 2``) or the stimulus assembled from one
        (``stim * 2``) gives a train at the new amplitude and does
        change the percept, while editing the data array in place does not.
        A stimulus that is only samples -- a raw waveform, an appended
        sequence of two trains, anything whose amplitudes were rewritten --
        is refused rather than predicted from numbers that no longer
        describe it.

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`
        The implant whose stimulation this model predicts. Required before
        building or predicting.

        .. versionadded:: 0.11.0

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
        axons_range: (min, max) of float or Quantity, optional
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
        meridian_blend : float, optional
            Gaussian standard deviation (dva) for smoothing across the
            horizontal meridian. Default: 1. Set to 0 to disable.

            .. versionadded:: 0.10.0
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
