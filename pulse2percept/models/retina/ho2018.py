""":py:class:`~pulse2percept.models.retina.Ho2018Model`,
   :py:class:`~pulse2percept.models.retina.Ho2018Spatial`,
   :py:class:`~pulse2percept.models.retina.Ho2018Temporal` [Ho2018]_"""

import numpy as np

from ...stimuli.encoders import _OpticalStimulus
from ...topography.retina import Watson2014Map
from ...units import as_value, mW, mm, ms
from ...percepts import Percept
from ..base import (Model, TemporalModel, _require_stim_dimension,
                    _thread_params)
from .beyeler2019 import ScoreboardSpatial

#: Peak irradiance (mW/mm^2) of the [Ho2018]_ white-noise stimulation
#: condition, which the activation law is normalized against.
REF_IRRADIANCE = 9.0

#: ON duration (ms) of the [Ho2018]_ white-noise stimulation condition.
REF_PULSE_DUR = 4.0

#: Irradiance the model reads its stimulus in.
_IRRADIANCE = mW / mm ** 2


def _radiant_exposure(stim):
    """Return per-pixel, per-pulse-period drive for an optical schedule.

    Radiant exposure (irradiance x ON duration) in multiples of the [Ho2018]_
    reference pulse. Shape is electrodes x pulse periods.
    """
    reference = REF_IRRADIANCE * REF_PULSE_DUR
    drive = stim.irradiance * stim.pulse_dur / reference
    return np.ascontiguousarray(drive, dtype=np.float32)


def _cascade(t, tau, n):
    """Return ``(t/tau)**n * exp(-n * (t/tau - 1))``, zero for ``t <= 0``.

    Impulse response of a cascade of ``n`` low-pass filters, normalized to
    peak at ``t = tau`` with unit amplitude.
    """
    t = np.asarray(t, dtype=np.float64)
    out = np.zeros(t.shape, dtype=np.float64)
    on = t > 0
    scaled = t[on] / tau
    out[on] = np.exp(n * (np.log(scaled) - scaled + 1.0))
    return out


class Ho2018Temporal(TemporalModel):
    r"""Network-mediated temporal response of [Ho2018]_.

    Photovoltaic spike-triggered-average time courses in [Ho2018]_ were
    summarized by a difference of two cascades of low-pass filters
    ([Chichilnisky2002]_):

    .. math::

        h(t) =
        p_1 \left(\frac{t}{\tau_1}\right)^{n}
            e^{-n\left(t/\tau_1 - 1\right)}
        - p_2 \left(\frac{t}{\tau_2}\right)^{n}
            e^{-n\left(t/\tau_2 - 1\right)},
        \qquad t \geq 0.

    Each term peaks at :math:`t = \tau_i` with amplitude :math:`p_i`, so
    :math:`\tau_1 < \tau_2` gives a biphasic, band-pass response.

    Every input sample is one pulse period's radiant-exposure drive and is
    applied as an impulse at that sample time, so the response to a schedule
    with drives :math:`d_k` at pulse onsets :math:`t_k` is

    .. math::

        B(t) = \max\left[
            g \sum_k d_k \, h(t - t_k), \; 0 \right],

    where :math:`g` normalizes :math:`h` to unit peak, so one reference pulse
    (9 mW/mm^2 for 4 ms) produces a peak drive of 1. The response is evaluated
    in closed form at the requested output times; ``dt`` only fixes the
    lattice those times must lie on.

    Half-wave rectification represents the pON pathway: a negative filter
    excursion is a drop below the spontaneous firing rate, which
    :py:class:`~pulse2percept.percepts.Percept` brightness cannot express.

    .. warning::

        The functional form and the timing landmarks come from [Ho2018]_,
        the default coefficients do not. [Ho2018]_ publishes no
        population-average coefficient set, so ``tau1``, ``tau2`` and ``p2``
        are a pulse2percept summary-matched parameterization: given ``n=6``,
        ``p1=1`` and zero DC
        gain (:math:`p_1\tau_1 = p_2\tau_2`, which makes the filter purely
        transient), they are the unique solution reproducing the reported
        degenerate-retina pON landmarks of a 50 ms first peak and a 94 ms
        first zero crossing. They are not fitted coefficients.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    n : float, optional
        Order of both low-pass cascades. The default 6 is a typical cascade
        order for ganglion-cell temporal filters of this form; the landmarks
        above can be matched for any ``n >= 4``.
    tau1 : float or Quantity, optional
        Time constant (ms) of the fast, positive cascade.
    tau2 : float or Quantity, optional
        Time constant (ms) of the slow, negative cascade.
    p1 : float, optional
        Peak amplitude of the fast cascade.
    p2 : float, optional
        Peak amplitude of the slow cascade.
    dt : float or Quantity, optional
        Output time lattice (ms). The response itself is evaluated in closed
        form, not integrated.
    thresh_percept : float, optional
        Brightness values below this threshold are set to zero.
    reduce : {'peak', 'last'}, optional
        How automatically chosen output points summarize the preceding
        interval.
    verbose : bool, optional
        Whether to print status messages.
    n_threads : int, optional
        Number of OpenMP threads.
    n_jobs : int or None, optional
        Alias for ``n_threads``. ``None`` and -1 use all available CPU cores.
    """

    #: Brightness is driven by light, which the encoders emit as positive
    #: irradiance.
    _drive_sign = 1

    def __init__(self, *, n=6, tau1=51.3, tau2=137.1, p1=1.0, p2=0.3743,
                 dt=0.005, thresh_percept=0, reduce='peak', verbose=True,
                 n_threads=None, n_jobs=None):
        super().__init__(n=n, tau1=tau1, tau2=tau2, p1=p1, p2=p2, dt=dt,
                         thresh_percept=thresh_percept, reduce=reduce,
                         verbose=verbose,
                         **_thread_params(n_threads, n_jobs))
        # Peak-normalization factor; `_build` recomputes it.
        self._gain = 1.0

    def get_default_params(self):
        """Return all settable parameters of the temporal response."""
        return {**super().get_default_params(),
                'n': 6, 'tau1': 51.3, 'tau2': 137.1, 'p1': 1.0, 'p2': 0.3743,
                'reduce': 'peak'}

    def get_param_units(self):
        """Return units used to store model parameters."""
        return {**super().get_param_units(), 'tau1': ms, 'tau2': ms}

    def _build(self):
        for name in ('n', 'tau1', 'tau2'):
            if getattr(self, name) <= 0:
                raise ValueError(f'"{name}" must be positive, not '
                                 f'{getattr(self, name)}.')
        # Peak of the impulse response, so that one reference pulse produces a
        # peak drive of 1 whatever the coefficients are:
        t = np.arange(0, 10 * max(self.tau1, self.tau2), self.dt)
        peak = self.impulse_response(t).max()
        if not np.isfinite(peak) or peak <= 0:
            raise ValueError("These coefficients produce no positive "
                             "response: the fast cascade (p1, tau1) has to "
                             "lead the slow one (p2, tau2).")
        self._gain = 1.0 / peak

    def impulse_response(self, t):
        """Return the unnormalized filter :math:`h(t)`.

        Parameters
        ----------
        t : array-like
            Time (ms) since the pulse. Negative times return zero.
        """
        fast = self.p1 * _cascade(t, self.tau1, self.n)
        slow = self.p2 * _cascade(t, self.tau2, self.n)
        return fast - slow

    def _predict_temporal(self, stim, t_percept):
        """Predict the temporal response."""
        t_pulse = np.asarray(self._stim_times(stim), dtype=np.float64)
        drive = np.asarray(self._stim_values(stim),
                           dtype=np.float64).reshape((-1, t_pulse.size))
        t_out = np.asarray(t_percept, dtype=np.float64)
        # One kernel column per output time; the same for every pixel, so the
        # spatial dimension only enters the matrix product.
        kernel = self._gain * self.impulse_response(
            t_out[np.newaxis, :] - t_pulse[:, np.newaxis])
        resp = np.maximum(drive @ kernel, 0.0)
        resp[resp < self.thresh_percept] = 0
        return resp.astype(np.float32)


class Ho2018Spatial(ScoreboardSpatial):
    r"""Network-mediated spatial response of [Ho2018]_ (spatial module only).

    Sums a circular Gaussian per illuminated pixel, as in
    :py:class:`~pulse2percept.models.retina.ScoreboardSpatial`, but drives it
    with the optical schedule a photovoltaic encoder produced rather than with
    normalized time-averaged drive. Use
    :py:class:`~pulse2percept.models.retina.Ho2018Model` for a standalone
    spatiotemporal model.

    Each pulse period is reduced to a radiant-exposure drive per pixel,

    .. math::

        d_e = \frac{E_e \, T_e}{E_\mathrm{ref} \, T_\mathrm{ref}},
        \qquad E_\mathrm{ref} = 9\ \mathrm{mW/mm}^2, \;
        T_\mathrm{ref} = 4\ \mathrm{ms},

    where :math:`E_e` is peak irradiance and :math:`T_e` the ON duration of
    pixel :math:`e` in that period, and the Ho reference pulse has
    :math:`d_e = 1`. Those drives are then summed over pixels,

    .. math::

        I(x, y) = \sum_{e \in E} d_e
        \exp\left(-\frac{(x-x_e)^2 + (y-y_e)^2}{2\rho^2}\right).

    The result is one drive map per pulse period, on the schedule's own
    pulse-period clock. The waveform behind the schedule is never rendered.

    .. warning::

        The linear radiant-exposure law is a pulse2percept baseline
        assumption, not a dose-response function reported by [Ho2018]_.
        Irradiance, pulse duration and repetition rate stay available
        separately on the stimulus so their effects can be fitted
        independently later.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.Implant`
        Photovoltaic implant whose pixel geometry is modeled. Its encoder
        converts images and videos into an optical schedule.
    rho : float or Quantity, optional
        Gaussian spatial decay constant in microns. The default 97.5 um is one
        standard deviation of the 195 um pON receptive-field diameter [Ho2018]_
        reports for degenerate (RCS) rat retina, which is the diameter of the
        fitted 1-sigma contour.

        .. important::

            Electrode-retina distance (``z``) does not affect ``rho``, and
            nonzero ``z`` raises a warning.

    xrange : (float, float) or Quantity, optional
        Horizontal visual-field extent in degrees of visual angle, or a
        retinal extent resolved through ``visual_field_map``.
    yrange : (float, float) or Quantity, optional
        Vertical visual-field extent in degrees of visual angle, or a retinal
        extent resolved through ``visual_field_map``.
    step : float, (float, float), or Quantity, optional
        Grid spacing in degrees of visual angle.
    grid_type : {'rect', 'hex'}, optional
        Sampling lattice used for the visual-field grid.
    thresh_percept : float, optional
        Drive values below this threshold are set to zero.
    min_current_spread : float, optional
        Fraction of peak Gaussian spread below which a pixel may be skipped at
        a grid point. Set to 0 to disable the cutoff.
    visual_field_map : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
        Retinotopic map between visual-field and retinal coordinates.
    n_gray : int or None, optional
        Number of gray levels in the returned percept.
    implant_position : (x, y) or Quantity, optional
        Position of the device-local origin, in tissue coordinates or dva.
    implant_rotation : float or Quantity, optional
        In-plane rotation (deg), positive counter-clockwise.
    implant_depth : float or Quantity, optional
        Signed offset (um) along the normal of a 2D tissue map.
    location_noise : float or None, optional
        Standard deviation of fixed pixel-specific phosphene offsets, in dva.
    verbose : bool, optional
        Whether to print status messages.
    ndim : list of int, optional
        Dimensionalities of ``visual_field_map`` accepted by the model.
    n_threads : int, optional
        Number of OpenMP threads.
    n_jobs : int or None, optional
        Alias for ``n_threads``. ``None`` and -1 use all available CPU cores.
    """

    #: Irradiance, not current: this model reads an optical schedule.
    stimulus_unit = _IRRADIANCE

    #: A normalized drive has already thrown away irradiance and ON duration,
    #: which is exactly what the activation law needs.
    extra_stimulus_units = ()

    #: The activation law reads peak irradiance and per-pixel ON duration off
    #: the schedule, so it must not be reduced to a spatial view first.
    _needs_structured_stim = True

    def __init__(self, implant, *, rho=97.5, xrange=(-15, 15),
                 yrange=(-15, 15), step=0.25, grid_type='rect',
                 thresh_percept=0, min_current_spread=1e-8,
                 visual_field_map=None,
                 n_gray=None,
                 implant_position=(0, 0), implant_rotation=0,
                 implant_depth=0,
                 location_noise=None, verbose=True, ndim=None,
                 n_threads=None, n_jobs=None):
        super().__init__(
            implant, rho=rho, xrange=xrange, yrange=yrange, step=step,
            grid_type=grid_type, thresh_percept=thresh_percept,
            min_current_spread=min_current_spread,
            visual_field_map=visual_field_map, n_gray=n_gray,
            implant_position=implant_position,
            implant_rotation=implant_rotation, implant_depth=implant_depth,
            location_noise=location_noise, verbose=verbose, ndim=ndim,
            n_threads=n_threads, n_jobs=n_jobs)

    def get_default_params(self):
        """Return all settable parameters of the spatial response."""
        return {**super().get_default_params(),
                'rho': 97.5, 'visual_field_map': Watson2014Map()}

    def _stim_values(self, stim):
        """Return radiant-exposure drive for an optical schedule."""
        if isinstance(stim, _OpticalStimulus):
            return _radiant_exposure(stim)
        return super()._stim_values(stim)

    def _predict_prepared(self, stim, t_percept=None):
        """Predict one drive map per pulse period.

        Output times are the schedule's pulse onsets. Explicit ``t_percept``
        values are served by zero-order hold, since drive is constant within a
        pulse period.
        """
        if not self.is_built:
            self.build()
        if stim is None:
            return None
        _require_stim_dimension(self, stim)
        if not isinstance(stim, _OpticalStimulus):
            raise TypeError(
                f"{type(self).__name__} reads peak irradiance and per-pixel "
                f"ON duration off a pulsed-illumination schedule, and this "
                f"stimulus does not carry one. Encode an image or a video "
                f"with pulse2percept.stimuli.PhotovoltaicEncoder, or hand a "
                f"photovoltaic implant the picture and let its encoder do "
                f"it.")
        t_percept = as_value(t_percept, self.time_unit, 't_percept')
        t_pulse = stim.pulse_time
        resp = self._predict_spatial(self.implant.electrode_array, stim)
        resp = self._postprocess_spatial(resp)
        resp = resp.reshape(list(self.grid.x.shape) + [-1])
        time = t_pulse
        if t_percept is not None:
            time = np.sort(np.array([t_percept], dtype=np.float64).ravel())
            at = np.searchsorted(t_pulse, time, side='right') - 1
            resp = resp[..., np.clip(at, 0, t_pulse.size - 1)]
        # The pulse-period clock is what a temporal stage should report on;
        # `_frame_clock` picks it up from here.
        return Percept(resp, space=self.grid, time=time,
                       time_unit=self.time_unit, n_gray=self.n_gray,
                       metadata={'stim': stim,
                                 'encoder': {'frame_time': t_pulse,
                                             'frame_dur': 1e3 / stim.freq}})


class Ho2018Model(Model):
    """Network-mediated photovoltaic response model of [Ho2018]_.

    Pairs :py:class:`~pulse2percept.models.retina.Ho2018Spatial` with
    :py:class:`~pulse2percept.models.retina.Ho2018Temporal`: an optical
    schedule from a photovoltaic encoder becomes a per-pulse-period drive map,
    which the temporal filter turns into a transient, spatially localized
    percept.

    .. code-block:: python

        implant = p2p.implants.retina.PRIMAPivotal()
        model = p2p.models.retina.Ho2018Model(implant, xrange=(-3, 3),
                                              yrange=(-3, 3), step=0.05)
        percept = model.predict_percept(image)

    Unlike
    :py:class:`~pulse2percept.models.retina.ScoreboardModel` driven by a
    :py:class:`~pulse2percept.stimuli.PhotovoltaicEncoder`, which visualizes
    normalized optical drive, this model predicts a phenomenological
    network-mediated retinal response.

    .. warning::

        This is a structural and timing-level reconstruction of [Ho2018]_, not
        a validated model of PRIMA percepts. It models the pON center response
        of degenerate (RCS) rat retina only: no antagonistic surround, no pOFF
        pathway, a linear radiant-exposure activation law with no fitted
        irradiance, pulse-duration or frequency nonlinearity, no photovoltaic
        circuit or electric-field model, no electrode-retina distance effect,
        and no calibration to human brightness or contrast perception.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.Implant`
        Photovoltaic implant whose pixel geometry is modeled.
    rho : float or Quantity, optional
        Gaussian spatial decay constant in microns; see
        :py:class:`~pulse2percept.models.retina.Ho2018Spatial`.
    xrange : (float, float) or Quantity, optional
        Horizontal visual-field extent in degrees of visual angle.
    yrange : (float, float) or Quantity, optional
        Vertical visual-field extent in degrees of visual angle.
    step : float, (float, float), or Quantity, optional
        Grid spacing in degrees of visual angle.
    grid_type : {'rect', 'hex'}, optional
        Sampling lattice used for the visual-field grid.
    min_current_spread : float, optional
        Fraction of peak Gaussian spread below which a pixel may be skipped at
        a grid point.
    visual_field_map : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
        Retinotopic map between visual-field and retinal coordinates.
    n_gray : int or None, optional
        Number of gray levels in the returned percept.
    implant_position : (x, y) or Quantity, optional
        Position of the device-local origin, in tissue coordinates or dva.
    implant_rotation : float or Quantity, optional
        In-plane rotation (deg), positive counter-clockwise.
    implant_depth : float or Quantity, optional
        Signed offset (um) along the normal of a 2D tissue map.
    location_noise : float or None, optional
        Standard deviation of fixed pixel-specific phosphene offsets, in dva.
    ndim : list of int, optional
        Dimensionalities of ``visual_field_map`` accepted by the model.
    n : float, optional
        Order of both low-pass cascades.
    tau1, tau2 : float or Quantity, optional
        Time constants (ms) of the fast and slow cascade.
    p1, p2 : float, optional
        Peak amplitudes of the fast and slow cascade.
    dt : float or Quantity, optional
        Output time lattice (ms).
    reduce : {'peak', 'last'}, optional
        How automatically chosen output points summarize the preceding
        interval.
    thresh_percept : float, optional
        Brightness values below this threshold are set to zero.
    verbose : bool, optional
        Whether to print status messages.
    n_threads : int, optional
        Number of OpenMP threads.
    n_jobs : int or None, optional
        Alias for ``n_threads``. ``None`` and -1 use all available CPU cores.
    """

    def __init__(self, implant, *, rho=97.5, xrange=(-15, 15),
                 yrange=(-15, 15), step=0.25, grid_type='rect',
                 min_current_spread=1e-8, visual_field_map=None, n_gray=None,
                 implant_position=(0, 0), implant_rotation=0, implant_depth=0,
                 location_noise=None, ndim=None,
                 n=6, tau1=51.3, tau2=137.1, p1=1.0, p2=0.3743, dt=0.005,
                 reduce='peak', thresh_percept=0, verbose=True,
                 n_threads=None, n_jobs=None):
        # `thresh_percept`, `verbose` and the thread count are declared by both
        # components and are applied to both.
        super().__init__(
            spatial=Ho2018Spatial(
                implant, rho=rho, xrange=xrange, yrange=yrange, step=step,
                grid_type=grid_type, thresh_percept=thresh_percept,
                min_current_spread=min_current_spread,
                visual_field_map=visual_field_map, n_gray=n_gray,
                implant_position=implant_position,
                implant_rotation=implant_rotation,
                implant_depth=implant_depth,
                location_noise=location_noise, verbose=verbose, ndim=ndim,
                n_threads=n_threads, n_jobs=n_jobs),
            temporal=Ho2018Temporal(
                n=n, tau1=tau1, tau2=tau2, p1=p1, p2=p2, dt=dt,
                reduce=reduce, thresh_percept=thresh_percept, verbose=verbose,
                n_threads=n_threads, n_jobs=n_jobs))
