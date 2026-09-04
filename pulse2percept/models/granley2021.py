""":py:class:`~pulse2percept.models.BiphasicAxonMapModel`,
   :py:class:`~pulse2percept.models.BiphasicAxonMapSpatial` [Granley2021]_"""
import numpy as np
from copy import deepcopy

from . import AxonMapSpatial, Model
from ..implants import ElectrodeArray
from ..stimuli import BiphasicPulseTrain, Stimulus
from ..percepts import Percept
from ..units import as_value, um, xTh
from .base import BaseModel, _require_stim_dimension
from ._granley2021 import fast_biphasic_axon_map

#: Maximum horizon expansions when locating delayed temporal peaks.
_PEAK_SEARCH_DOUBLINGS = 4


class DefaultBrightModel(BaseModel):
    r"""Brightness scaling from [Granley2021]_.

    Implements the amplitude- and frequency-dependent factor in Eq. 4, using the
    phase-duration threshold correction from Eq. 3. Coefficients were fit to
    [Nanduri2012]_ and [Weitz2015]_.

    The default brightness factor is

    .. math::

        F_{\mathrm{bright}} =
        a_2 \tilde{a} + a_3 f + a_4,

    where :math:`f` is pulse frequency and :math:`\tilde{a}` is
    threshold-scaled amplitude.

    Parameters
    ----------
    a0, a1 : float, optional
        Slope and intercept of the phase-duration threshold correction,
        ``a0 * pdur + a1``.
    a2 : float, optional
        Coefficient for threshold-scaled amplitude in Eq. 4.
    a3 : float, optional
        Coefficient for frequency in Eq. 4.
    a4 : float, optional
        Intercept in Eq. 4."""

    def __init__(self, *, a0=2.095, a1=0.054326, a2=0.1492147, a3=0.0163851,
                 a4=0):
        super().__init__(a0=a0, a1=a1, a2=a2, a3=a3, a4=a4)
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
        """Return the phase-duration threshold scaling from Eq. 3.

        Uses the Argus II refit to [Horsager2009]_ data rather than the
        coefficients in the original publication."""
        return self.a1 + self.a0*pdur

    def predict_freq_amp(self, amp, freq):
        """Return the Eq. 4 brightness factor."""
        return self.a2*amp + self.a3*freq + self.a4

    def __call__(self, freq, amp, pdur):
        """Return the brightness scaling factor.

        ``freq``, ``amp``, and ``pdur`` may be scalars or arrays."""
        F_bright = self.predict_freq_amp(amp * self.scale_threshold(pdur), freq)
        return F_bright


class DefaultSizeModel(BaseModel):
    r"""Phosphene-size scaling from [Granley2021]_.

    Implements the size scaling of [Granley2021]_, using p2p's Argus II
    phase-duration threshold refit (i.e., different from Eq. 5 in the published
    paper).

    With threshold-scaled amplitude :math:`\tilde{a}`, the default size
    factor is

    .. math::

        F_{\mathrm{size}} =
        \max\left(
            a_5 \tilde{a} + a_6,\,
            \frac{\mathrm{min\_rho}^2}{\rho^2}
        \right),

    so that

    .. math::

        \rho_{\mathrm{eff}} = \rho \sqrt{F_{\mathrm{size}}}
        \geq \mathrm{min\_rho}.
        
    Parameters
    ----------
    rho : float or Quantity, optional
        Baseline Gaussian spatial decay away from the axon, in microns.
        Larger values produce broader phosphenes. The effective spatial
        spread is additionally modulated by amplitude and pulse duration
        according to [Granley2021]_.

        .. important::

            Electrode-retina distance (``z``) does not directly affect ``rho``.
            Electrode-specific detection thresholds may capture its effect on
            stimulation sensitivity, but not on baseline spatial spread.

    min_rho : float or Quantity, optional
        Minimum effective ``rho``, in microns. Prevents pulse-dependent
        size scaling from shrinking the spatial decay below this value.
    a0, a1 : float, optional
        Slope and intercept of the phase-duration threshold correction,
        ``a0 * pdur + a1``.
    a5 : float, optional
        Coefficient for threshold-scaled amplitude in Eq. 5.
    a6 : float, optional
        Intercept in Eq. 5.
    """

    def __init__(self, rho, *, a0=2.095, a1=0.054326, a5=1.0812, a6=-0.35338,
                 min_rho=10):
        super().__init__(a0=a0, a1=a1, a5=a5, a6=a6, min_rho=min_rho)
        self.rho = rho
        self.build()

    def get_default_params(self):
        params = {
            'a0': 2.095,
            'a1': 0.054326,
            'a5': 1.0812,
            'a6': -0.35338,
            'min_rho': 10,
        }
        return params

    def get_param_units(self):
        """Return units used to store model parameters."""
        # ``rho`` is a constructor argument but still needs normalization.
        return {**super().get_param_units(), 'rho': um, 'min_rho': um}

    def scale_threshold(self, pdur):
        """Return the phase-duration threshold scaling from Eq. 3.

        Uses the Argus II refit to [Horsager2009]_ data rather than the
        coefficients in the original publication."""
        return self.a1 + self.a0*pdur

    def __call__(self, freq, amp, pdur):
        """Return ``F_size`` for each stimulus condition.

        ``F_size`` scales ``rho ** 2`` and is floored so the effective ``rho`` does
        not fall below ``min_rho``. Inputs may be scalars or arrays."""
        min_f_size = self.min_rho**2 / self.rho**2
        F_size = self.a5 * amp * self.scale_threshold(pdur) + self.a6
        return np.maximum(F_size, min_f_size)


class DefaultStreakModel(BaseModel):
    r"""Phosphene-streak scaling from [Granley2021]_.

    Implements the pulse-duration-dependent factor in Eq. 6. Coefficients were
    fit to [Weitz2015]_.

    The default streak-length factor is

    .. math::

        F_{\mathrm{streak}} =
        \max\left(
            a_9 - a_7 t^{a_8},\,
            \frac{\mathrm{min\_lambda}^2}{\lambda^2}
        \right),

    where :math:`t` is phase duration. Therefore

    .. math::

        \lambda_{\mathrm{eff}} =
        \lambda \sqrt{F_{\mathrm{streak}}}
        \geq \mathrm{min\_lambda}.
        
    Parameters
    ----------
    lam : float or Quantity, optional
        Baseline Gaussian decay along the axon, in microns. Larger values
        produce longer phosphene streaks. The effective decay is additionally
        modulated by pulse duration according to [Granley2021]_.
    min_lambda : float or Quantity, optional
        Minimum effective ``lam``, in microns. Prevents pulse-dependent
        streak-length scaling from shrinking the axonal decay below this value.
    a7, a8, a9 : float, optional
        Regression coefficients in ``a9 - a7 * pdur ** a8``.
    """
    def __init__(self, lam, *, a7=0.54, a8=0.21, a9=1.56, min_lambda=10):
        super().__init__(a7=a7, a8=a8, a9=a9, min_lambda=min_lambda)
        self.lam = lam
        self.build()

    def get_default_params(self):
        params = {
            'a7': 0.54,
            'a8': 0.21,
            'a9': 1.56,
            'min_lambda': 10,
        }
        return params

    def get_param_units(self):
        """Return units used to store model parameters."""
        # ``lam`` is a constructor argument but still needs normalization.
        return {**super().get_param_units(), 'lam': um, 'min_lambda': um}

    def __call__(self, freq, amp, pdur):
        """Return ``F_streak`` for each stimulus condition.

        ``F_streak`` scales ``lam ** 2`` and is floored so the effective ``lam`` does
        not fall below ``min_lambda``. Inputs may be scalars or arrays."""
        min_f_streak = self.min_lambda**2 / self.lam ** 2
        F_streak = self.a9 - self.a7 * pdur ** self.a8
        return np.maximum(F_streak, min_f_streak)


#: Error for a current-valued pulse without threshold calibration.
_NO_THRESHOLD_MSG = (
    "This model takes amplitude as a multiple of perceptual threshold, and "
    "electrode {electrode} is driven at {amp:.1f} uA with no threshold to "
    "measure that against. Either give `amp` in threshold units (e.g. "
    "`2 * xTh`), pass `threshold_amp` to the BiphasicPulseTrain, or set "
    "`implant.thresholds`."
)


def _amp_factor(electrode, amp, unit, thresholds):
    """Amplitude in multiples of threshold, calibrating current if need be"""
    if unit.dimension == xTh.dimension:
        return amp
    threshold = (thresholds or {}).get(electrode)
    if threshold is None:
        raise ValueError(_NO_THRESHOLD_MSG.format(electrode=electrode,
                                                  amp=amp))
    return amp / threshold


def _pulse_train_params(stim, thresholds=None):
    """Return Granley pulse parameters for each active electrode."""
    described = getattr(stim, '_biphasic_params', None)
    if described is not None:
        encoded = described()
        if encoded is None:
            raise TypeError(
                "All stimuli must be BiphasicPulseTrains with no delay dur. "
                "This one was encoded with a 'pulse' shape of its own, whose "
                "phase duration this model cannot read off.")
        return [(electrode, freq,
                 _amp_factor(electrode, amp, stim.unit, thresholds),
                 phase_dur, stim_dur)
                for electrode, freq, amp, phase_dur, stim_dur in encoded]
    sources = stim._structured_sources()
    if sources is None:
        # Preserve the historical zero-stimulus result; this is the only case
        # that requires rendering a waveform.
        if not np.any(stim.data):
            return []
        raise TypeError("All stimuli must be BiphasicPulseTrains with no "
                        "delay dur")
    params = []
    for electrode, source in sources:
        # Require the exact pulse-train contract used to derive the model.
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


#: Spatial parameters mirrored to effect models.
_SHARED_WITH_EFFECT = {'rho': 'size_model', 'lam': 'streak_model'}


class BiphasicAxonMapSpatial(AxonMapSpatial):
    r"""Biphasic axon-map model of [Granley2021]_ (spatial module only).

    Extends :py:class:`~pulse2percept.models.AxonMapSpatial` with the
    stimulus-dependent brightness, size, and streak-length scaling of
    [Granley2021]_. The model returns one representative spatial percept for the
    full biphasic pulse train.

    Stimuli must describe the pulse train they deliver, rather than only its
    samples: either retained
    :py:class:`~pulse2percept.stimuli.BiphasicPulseTrain` objects, or a still
    image encoded with the standard biphasic encoder pulse (see
    :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`). Amplitude may be
    given in multiples of perceptual threshold
    (:py:data:`~pulse2percept.units.xTh`) or as current when a threshold
    calibration is available.

    Encoded still images use the device-resolved amplitude, phase duration, and
    frequency; exact pulse-onset timing is ignored. Videos are not supported.

    Custom effect models must be callables with signature ``f(freq, amp, pdur)``.
    Their arguments are frequency, amplitude in multiples of threshold, and phase
    duration.

    When paired with a temporal model, this spatial prediction is treated as the
    peak percept and multiplied by a normalized temporal response.

    The spatial response is

    .. math::

        I(r, \theta) =
        \max_{p \in R(\theta)}
        \sum_{e \in E}
        F_{\mathrm{bright}}
        \exp\left(
            -\frac{d_e^2}{2 \rho^2 F_{\mathrm{size}}}
            -\frac{d_{\mathrm{soma}}^2}
                {2 \lambda^2 F_{\mathrm{streak}}}
        \right),

    where :math:`d_e` is the distance from an axon segment to electrode
    :math:`e`, and :math:`d_{\mathrm{soma}}` is the path length from that
    segment to the ganglion cell body. Thus the effective spatial scales are

    .. math::

        \rho_{\mathrm{eff}} = \rho \sqrt{F_{\mathrm{size}}},
        \qquad
        \lambda_{\mathrm{eff}} = \lambda \sqrt{F_{\mathrm{streak}}}.

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.Implant`
        Implant whose electrode geometry and eye are modeled.

        .. versionadded:: 0.11.0

    bright_model : callable, optional
        Maps ``(freq, amp, pdur)`` to a multiplicative brightness factor.
        Defaults to :class:`DefaultBrightModel`.
    size_model : callable, optional
        Maps ``(freq, amp, pdur)`` to ``F_size``, which scales ``rho ** 2``.
        Defaults to :class:`DefaultSizeModel`.
    streak_model : callable, optional
        Maps ``(freq, amp, pdur)`` to ``F_streak``, which scales
        ``lam ** 2``. Defaults to :class:`DefaultStreakModel`.
    rho : float or Quantity, optional
        Gaussian decay constant for spread from an electrode to nearby axon
        segments, in microns. Larger values broaden the percept.
    lam : float or Quantity, optional
        Gaussian decay constant along the axon between stimulation site and
        soma, in microns. Larger values lengthen the percept.

        .. versionchanged:: 0.10.0
            Renamed from ``axlambda``; ``axlambda`` was removed in 0.11.0.

    xrange : (float, float) or Quantity, optional
        Horizontal visual-field extent in degrees of visual angle. A physical
        retinal extent may instead be resolved through ``visual_field_map``.
    yrange : (float, float) or Quantity, optional
        Vertical visual-field extent in degrees of visual angle. A physical
        retinal extent may instead be resolved through ``visual_field_map``.
    step : float, (float, float), or Quantity, optional
        Grid spacing in degrees of visual angle. A pair specifies separate x
        and y spacing.

        .. versionchanged:: 0.10.0
            Renamed from ``xystep``; ``xystep`` was removed in 0.11.0.

    grid_type : {'rect', 'hex'}, optional
        Sampling lattice used for the visual-field grid.
    thresh_percept : float, optional
        Brightness values below this threshold are set to zero.
    min_current_spread : float, optional
        Fraction of peak current spread below which an electrode may be
        skipped at an axon segment. The cutoff is scaled by ``F_size``.
        Set to 0 to disable.
    visual_field_map : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
        Retinotopic map between visual-field and retinal coordinates. Defaults
        to :py:class:`~pulse2percept.topography.Watson2014Map`.
    n_gray : int or None, optional
        Number of gray levels in the returned percept. ``None`` disables
        gray-level quantization.
    noise : float, int, or None, optional
        Salt-and-pepper noise applied to each percept frame. An integer gives
        the number of affected pixels; a float in [0, 1] gives their fraction.
    implant_pos : (x, y) or Quantity, optional
        Where the implant's local ``(0, 0)`` origin sits. A bare pair or
        a length is a tissue position in microns; ``(6, -2) * dva`` is a
        visual field location, resolved through ``visual_field_map``.

        .. versionadded:: 0.11.0

    implant_z : float or Quantity, optional
        Depth (um) the implant is placed at, added to every electrode's
        local ``z``.

        .. versionadded:: 0.11.0

    location_noise : float or None, optional
        Standard deviation of fixed electrode-specific phosphene offsets, in dva.
        Requires an invertible 2D ``visual_field_map``. ``None`` or 0 disables it.
        Location-dependent models may also change phosphene shape or size.

        .. versionadded:: 0.11.0

    loc_od : (float, float) or Quantity, optional
        Optic-disc location in degrees of visual angle. Its horizontal sign is
        set from the bound implant's eye.
    n_axons : int, optional
        Number of nerve fiber bundles generated.
    axons_range : (float, float) or Quantity, optional
        Range of initial bundle angles ``phi0`` in the Jansonius model.
    n_ax_segments : int, optional
        Number of radial samples used to generate each bundle.
    ax_segments_range : (float, float), optional
        Radial-coordinate range used to generate each bundle in the Jansonius
        model.
    min_ax_sensitivity : float, optional
        Minimum relative axon sensitivity retained during precomputation.
    meridian_blend : float or Quantity, optional
        Gaussian standard deviation for blending across the horizontal
        meridian, in degrees of visual angle. Set to 0 to disable.

        .. versionadded:: 0.10.0

    axon_pickle : str, optional
        File used to cache generated axon bundles.
    ignore_pickle : bool, optional
        If True, regenerate axon bundles instead of loading ``axon_pickle``.
    verbose : bool, optional
        Whether to print status messages.
    ndim : list of int, optional
        Dimensionalities of ``visual_field_map`` accepted by the model.
    n_threads : int, optional
        Number of OpenMP threads.
    n_jobs : int or None, optional
        Alias for ``n_threads``. ``None`` and -1 use all available CPU cores.

    Notes
    -----
    ``ax_segments_range`` values above 90 are outside the range for which this
    axon-map construction is considered reliable."""
    extra_stimulus_units = (xTh,)

    def __init__(self, implant, *, bright_model=None, size_model=None,
                 streak_model=None, rho=300, lam=500, xrange=(-15, 15),
                 yrange=(-15, 15), step=0.25, grid_type='rect',
                 thresh_percept=0, min_current_spread=1e-8,
                 visual_field_map=None,
                 n_gray=None, noise=None,
                 implant_pos=(0, 0), implant_z=0,
                 location_noise=None, loc_od=(15.5, 1.5), n_axons=1000,
                 axons_range=(-180, 180), n_ax_segments=500,
                 ax_segments_range=(0, 50), min_ax_sensitivity=1e-3,
                 meridian_blend=1, axon_pickle='axons.pickle',
                 ignore_pickle=False, verbose=True, ndim=None,
                 n_threads=None, n_jobs=None):
        # Install default effect models after AxonMapSpatial initialization.
        super().__init__(
            implant, rho=rho, lam=lam, xrange=xrange, yrange=yrange,
            step=step, grid_type=grid_type, thresh_percept=thresh_percept,
            min_current_spread=min_current_spread,
            visual_field_map=visual_field_map,
            n_gray=n_gray, noise=noise,
            implant_pos=implant_pos, implant_z=implant_z,
            location_noise=location_noise, loc_od=loc_od, n_axons=n_axons,
            axons_range=axons_range, n_ax_segments=n_ax_segments,
            ax_segments_range=ax_segments_range,
            min_ax_sensitivity=min_ax_sensitivity,
            meridian_blend=meridian_blend, axon_pickle=axon_pickle,
            ignore_pickle=ignore_pickle, verbose=verbose, ndim=ndim,
            n_threads=n_threads, n_jobs=n_jobs)
        self.bright_model = (DefaultBrightModel() if bright_model is None
                             else bright_model)
        self.size_model = (DefaultSizeModel(self.rho) if size_model is None
                           else size_model)
        self.streak_model = (DefaultStreakModel(self.lam)
                             if streak_model is None else streak_model)
        # Synchronize rho/lam now that the effect models exist:
        self.rho = rho
        self.lam = lam

    def __setattr__(self, name, value):
        """Set a spatial parameter and synchronize shared effect-model parameters.

        ``rho`` and ``lam`` are mirrored to the size and streak models.
        Other effect-model parameters are not forwarded.
        """
        super().__setattr__(name, value)
        target = _SHARED_WITH_EFFECT.get(name)
        if target is None:
            return
        effect = getattr(self, target, None)
        if hasattr(effect, name):
            # Mirror the stored value, which `super()` has already normalized:
            setattr(effect, name, getattr(self, name))

    def get_default_params(self):
        base_params = super(BiphasicAxonMapSpatial, self).get_default_params()
        params = {
            'bright_model': None,
            'size_model': None,
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

    def _predict_spatial(self, electrode_array, stim):
        """Predict the representative spatial percept."""
        if not isinstance(electrode_array, ElectrodeArray):
            raise TypeError("Implant must be of type ElectrodeArray but it is " +
                            str(type(electrode_array)))
        if not isinstance(stim, Stimulus):
            raise TypeError(
                "Stim must be of type Stimulus but it is " + str(type(stim)))
        params = _pulse_train_params(stim, self.implant.thresholds)
        active = [p[0] for p in params]
        elec_params = np.array([p[1:4] for p in params],
                               dtype=np.float32).reshape((-1, 3))
        # Match coordinates to the active-electrode order above.
        x, y, _ = self._electrode_coords(electrode_array, stim,
                                         electrodes=active)

        bright_effects = np.array(self.bright_model(elec_params[:, 0], elec_params[:, 1], elec_params[:, 2]),
                                  dtype=np.float32).reshape((-1))
        size_effects = np.array(self.size_model(elec_params[:, 0], elec_params[:, 1], elec_params[:, 2]),
                                dtype=np.float32).reshape((-1))
        streak_effects = np.array(self.streak_model(elec_params[:, 0], elec_params[:, 1], elec_params[:, 2]),
                                  dtype=np.float32).reshape((-1))
        amps = np.array(elec_params[:, 1], dtype=np.float32).reshape((-1))
        # Reject values that would make the kernel's exponent non-finite.
        for name, effects in (('bright_model', bright_effects),
                              ('size_model', size_effects),
                              ('streak_model', streak_effects)):
            if not np.all(np.isfinite(effects)):
                raise ValueError(f"{type(self).__name__}.{name} returned a "
                                 f"non-finite scaling factor. Scaling factors "
                                 f"must be finite.")
        # ``F_size`` and ``F_streak`` appear in exponent denominators.
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
        """Predict from an already prepared stimulus.

        This model summarizes the full pulse train as one spatial percept. If
        ``t_percept`` contains multiple times, the representative percept occupies
        the first output frame and later frames are zero."""
        if not self.is_built:
            self.build()
        if stim is None:
            return None
        _require_stim_dimension(self, stim)
        params = _pulse_train_params(stim, self.implant.thresholds)
        t_percept = as_value(t_percept, self.time_unit, 't_percept')
        n_time = 1 if t_percept is None else np.array([t_percept]).size
        if not params:
            resp = np.zeros(list(self.grid.x.shape) + [n_time],
                            dtype=np.float32)
        else:
            resp = np.zeros(list(self.grid.x.shape) + [n_time])
            # The representative Granley percept occupies the first frame.
            resp[:, :, 0] = self._predict_spatial(
                self.implant.electrode_array, stim).reshape(self.grid.x.shape)
        # Apply the same spatial postprocessing as the generic path.
        resp = self._postprocess_spatial(resp)
        return Percept(resp, space=self.grid, time=t_percept,
                       time_unit=self.time_unit, metadata={'stim': stim},
                       n_gray=self.n_gray, noise=self.noise)

    def _combine_temporal(self, percept, temporal, stim, t_percept):
        """Apply a normalized temporal response to the spatial percept."""
        dur = self._envelope_dur(stim)
        # Canonical unit drive, held for the stimulation duration.
        envelope = Stimulus(np.array([[float(temporal._drive_sign), 0.0]]),
                            electrodes=['envelope'], time=[0, dur],
                            metadata=stim.metadata.get('user'))
        # Do not modify the caller's temporal model.
        probe = deepcopy(temporal)
        probe.thresh_percept = 0
        peak = self._envelope_peak(probe, envelope)
        resp = probe.predict_percept(envelope, t_percept=t_percept)
        fade = resp.data.reshape(-1) / peak
        return Percept(percept.data[..., 0][..., np.newaxis] * fade,
                       space=self.grid, time=resp.time,
                       time_unit=probe.time_unit, metadata={'stim': stim})

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
        durs = {p[4] for p in _pulse_train_params(stim,
                                                  self.implant.thresholds)}
        if len(durs) > 1:
            raise NotImplementedError(
                f"{type(self).__name__} requires active electrodes to share "
                f"one stim_dur, not {sorted(durs)}."
            )
        if durs:
            return float(durs.pop())

        # No active electrodes; duration only determines the output time axis.
        if getattr(stim, '_biphasic_params', None) is not None:
            return float(stim.duration)
        sources = stim._structured_sources()
        if sources:
            return float(max(src.stim_dur for _, src in sources))
        return float(stim.time[-1])


class BiphasicAxonMapModel(Model):
    r"""Biphasic axon-map model of [Granley2021]_.

    Extends :py:class:`~pulse2percept.models.AxonMapModel` with the
    stimulus-dependent brightness, size, and streak-length scaling of
    [Granley2021]_. The model returns one representative percept for the full
    biphasic pulse train.

    Stimuli must describe the pulse train they deliver, rather than only its
    samples: either retained
    :py:class:`~pulse2percept.stimuli.BiphasicPulseTrain` objects, or a still
    image encoded with the standard biphasic encoder pulse (see
    :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`). Give amplitude in
    multiples of perceptual threshold (:py:data:`~pulse2percept.units.xTh`) or
    provide a threshold calibration for current-valued amplitudes. Threshold is
    the 50%-detection current for a train at the same frequency and 0.45 ms
    phase duration [Granley2021]_; the model applies its own phase-duration
    correction.

    Encoded still images use the device-resolved amplitude, phase duration, and
    frequency; exact pulse-onset timing is ignored. Videos are not supported.

    Custom effect models must be callables with signature ``f(freq, amp, pdur)``.
    Their arguments are frequency, amplitude in multiples of threshold, and phase
    duration.

    The spatial response is

    .. math::

        I(r, \theta) =
        \max_{p \in R(\theta)}
        \sum_{e \in E}
        F_{\mathrm{bright}}
        \exp\left(
            -\frac{d_e^2}{2 \rho^2 F_{\mathrm{size}}}
            -\frac{d_{\mathrm{soma}}^2}
                {2 \lambda^2 F_{\mathrm{streak}}}
        \right),

    where :math:`d_e` is the distance from an axon segment to electrode
    :math:`e`, and :math:`d_{\mathrm{soma}}` is the path length from that
    segment to the ganglion cell body. Thus the effective spatial scales are

    .. math::

        \rho_{\mathrm{eff}} = \rho \sqrt{F_{\mathrm{size}}},
        \qquad
        \lambda_{\mathrm{eff}} = \lambda \sqrt{F_{\mathrm{streak}}}.

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.Implant`
        Implant whose electrode geometry and eye are modeled.

        .. versionadded:: 0.11.0

    bright_model : callable, optional
        Maps ``(freq, amp, pdur)`` to a multiplicative brightness factor.
        Defaults to :class:`DefaultBrightModel`.
    size_model : callable, optional
        Maps ``(freq, amp, pdur)`` to ``F_size``, which scales ``rho ** 2``.
        Defaults to :class:`DefaultSizeModel`.
    streak_model : callable, optional
        Maps ``(freq, amp, pdur)`` to ``F_streak``, which scales
        ``lam ** 2``. Defaults to :class:`DefaultStreakModel`.
    rho : float or Quantity, optional
        Gaussian decay constant for spread from an electrode to nearby axon
        segments, in microns. Larger values broaden the percept.
    lam : float or Quantity, optional
        Gaussian decay constant along the axon between stimulation site and
        soma, in microns. Larger values lengthen the percept.

        .. versionchanged:: 0.10.0
            Renamed from ``axlambda``; ``axlambda`` was removed in 0.11.0.

    xrange : (float, float) or Quantity, optional
        Horizontal visual-field extent in degrees of visual angle. A physical
        retinal extent may instead be resolved through ``visual_field_map``.
    yrange : (float, float) or Quantity, optional
        Vertical visual-field extent in degrees of visual angle. A physical
        retinal extent may instead be resolved through ``visual_field_map``.
    step : float, (float, float), or Quantity, optional
        Grid spacing in degrees of visual angle. A pair specifies separate x
        and y spacing.

        .. versionchanged:: 0.10.0
            Renamed from ``xystep``; ``xystep`` was removed in 0.11.0.

    grid_type : {'rect', 'hex'}, optional
        Sampling lattice used for the visual-field grid.
    thresh_percept : float, optional
        Brightness values below this threshold are set to zero.
    min_current_spread : float, optional
        Fraction of peak current spread below which an electrode may be
        skipped at an axon segment. The cutoff is scaled by ``F_size``.
        Set to 0 to disable.
    visual_field_map : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
        Retinotopic map between visual-field and retinal coordinates. Defaults
        to :py:class:`~pulse2percept.topography.Watson2014Map`.
    n_gray : int or None, optional
        Number of gray levels in the returned percept. ``None`` disables
        gray-level quantization.
    noise : float, int, or None, optional
        Salt-and-pepper noise applied to each percept frame. An integer gives
        the number of affected pixels; a float in [0, 1] gives their fraction.
    implant_pos : (x, y) or Quantity, optional
        Where the implant's local ``(0, 0)`` origin sits. A bare pair or
        a length is a tissue position in microns; ``(6, -2) * dva`` is a
        visual field location, resolved through ``visual_field_map``.

        .. versionadded:: 0.11.0

    implant_z : float or Quantity, optional
        Depth (um) the implant is placed at, added to every electrode's
        local ``z``.

        .. versionadded:: 0.11.0

    location_noise : float or None, optional
        Standard deviation of fixed electrode-specific phosphene offsets, in dva.
        Requires an invertible 2D ``visual_field_map``. ``None`` or 0 disables it.
        Location-dependent models may also change phosphene shape or size.
        
        .. versionadded:: 0.11.0

    loc_od : (float, float) or Quantity, optional
        Optic-disc location in degrees of visual angle. Its horizontal sign is
        set from the bound implant's eye.
    n_axons : int, optional
        Number of nerve fiber bundles generated.
    axons_range : (float, float) or Quantity, optional
        Range of initial bundle angles ``phi0`` in the Jansonius model.
    n_ax_segments : int, optional
        Number of radial samples used to generate each bundle.
    ax_segments_range : (float, float), optional
        Radial-coordinate range used to generate each bundle in the Jansonius
        model.
    min_ax_sensitivity : float, optional
        Minimum relative axon sensitivity retained during precomputation.
    meridian_blend : float or Quantity, optional
        Gaussian standard deviation for blending across the horizontal
        meridian, in degrees of visual angle. Set to 0 to disable.

        .. versionadded:: 0.10.0

    axon_pickle : str, optional
        File used to cache generated axon bundles.
    ignore_pickle : bool, optional
        If True, regenerate axon bundles instead of loading ``axon_pickle``.
    verbose : bool, optional
        Whether to print status messages.
    ndim : list of int, optional
        Dimensionalities of ``visual_field_map`` accepted by the model.
    n_threads : int, optional
        Number of OpenMP threads.
    n_jobs : int or None, optional
        Alias for ``n_threads``. ``None`` and -1 use all available CPU cores.

    Notes
    -----
    ``ax_segments_range`` values above 90 are outside the range for which this
    axon-map construction is considered reliable.

    Examples
    --------
    A picture, a device that encodes it, and a participant's measured
    threshold:

    .. code-block:: python

        import pulse2percept as p2p

        implant = p2p.implants.ArgusII(thresholds=80 * p2p.units.uA)
        model = p2p.models.BiphasicAxonMapModel(implant=implant)
        percept = model.predict_percept(p2p.stimuli.LogoBVL())

    An encoder that asks for threshold multiples in the first place needs no
    measured threshold:

    .. code-block:: python

        encoder = p2p.stimuli.AmplitudeEncoder(
            amp_range=(0 * p2p.units.xTh, 3 * p2p.units.xTh))
        implant = p2p.implants.ArgusII(encoder=encoder)
        model = p2p.models.BiphasicAxonMapModel(implant=implant)
        percept = model.predict_percept(p2p.stimuli.LogoBVL())
    """

    def __init__(self, implant, *, bright_model=None, size_model=None,
                 streak_model=None, rho=300, lam=500, xrange=(-15, 15),
                 yrange=(-15, 15), step=0.25, grid_type='rect',
                 thresh_percept=0, min_current_spread=1e-8,
                 visual_field_map=None,
                 n_gray=None, noise=None,
                 implant_pos=(0, 0), implant_z=0,
                 location_noise=None, loc_od=(15.5, 1.5), n_axons=1000,
                 axons_range=(-180, 180), n_ax_segments=500,
                 ax_segments_range=(0, 50), min_ax_sensitivity=1e-3,
                 meridian_blend=1, axon_pickle='axons.pickle',
                 ignore_pickle=False, verbose=True, ndim=None,
                 n_threads=None, n_jobs=None):
        super().__init__(
            spatial=BiphasicAxonMapSpatial(
                implant, bright_model=bright_model, size_model=size_model,
                streak_model=streak_model, rho=rho, lam=lam, xrange=xrange,
                yrange=yrange, step=step, grid_type=grid_type,
                thresh_percept=thresh_percept,
                min_current_spread=min_current_spread,
                visual_field_map=visual_field_map,
                n_gray=n_gray, noise=noise,
                implant_pos=implant_pos, implant_z=implant_z,
                location_noise=location_noise, loc_od=loc_od, n_axons=n_axons,
                axons_range=axons_range, n_ax_segments=n_ax_segments,
                ax_segments_range=ax_segments_range,
                min_ax_sensitivity=min_ax_sensitivity,
                meridian_blend=meridian_blend, axon_pickle=axon_pickle,
                ignore_pickle=ignore_pickle, verbose=verbose, ndim=ndim,
                n_threads=n_threads, n_jobs=n_jobs),
            temporal=None)
