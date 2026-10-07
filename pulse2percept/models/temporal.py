""":py:class:`~pulse2percept.models.FadingTemporal`,
:py:class:`~pulse2percept.models.AlphaTemporal`"""
import math

import numpy as np
from .base import TemporalModel
from ..units import ms


def _percept_steps(t_percept, dt):
    """Return output times as distinct ``dt`` step indices."""
    # Round before casting so floating-point noise cannot shift a sample.
    idx_percept = np.uint32(np.round(t_percept / dt))
    if np.unique(idx_percept).size < t_percept.size:
        raise ValueError(f"All times 't_percept' must be distinct multiples "
                         f"of `dt`={dt:.2e}")
    return idx_percept


def _temporal_runs(t_stim, idx_percept, dt):
    """Group simulation steps into constant-stimulus runs.

    A run ends before a stimulus frame change and at each output step.
    Returns the stimulus frame, length in ``dt`` steps, and output column (or
    -1) of each run.
    """
    idx_percept = np.asarray(idx_percept, dtype=np.int64)
    n_sim = int(idx_percept[-1]) + 1
    t_stim = np.asarray(t_stim, dtype=np.float32)
    # float32 step times, as in the deleted Cython kernels:
    t_sim = np.arange(n_sim).astype(np.float32) * np.float32(dt)
    # First step of each later frame; several frames may start at one step:
    start = np.searchsorted(t_sim, t_stim[1:])
    end = np.union1d(start[(start > 0) & (start < n_sim)] - 1, idx_percept)
    out = np.full(end.size, -1)
    out[np.searchsorted(end, idx_percept)] = np.arange(idx_percept.size)
    # Sample-and-hold: the latest frame with t_stim <= t_sim:
    frame = np.searchsorted(t_stim[1:], t_sim[end], side='right')
    return frame.tolist(), np.diff(end, prepend=-1), out.tolist()


def _run_powers(n, dt_tau):
    """Return float64 ``q**n`` and ``1 - q**n``, with ``q = 1 - dt_tau``."""
    n = np.asarray(n, dtype=np.float64)
    if dt_tau == 1:
        # tau == dt: q = 0, so q**0 = 1 and q**n = 0 otherwise:
        return (n == 0).astype(np.float64), (n != 0).astype(np.float64)
    n_log_q = n * np.log1p(-np.float64(dt_tau))
    return np.exp(n_log_q), -np.expm1(n_log_q)


def _on_torch(core, data, *args):
    """Return ``core`` applied to NumPy ``data`` as float32 Torch, as NumPy."""
    import torch
    # float32, as in the deleted Cython kernels. Copied, since Torch cannot
    # wrap a read-only array:
    data = torch.from_numpy(np.array(data, dtype=np.float32))
    with torch.inference_mode():
        return core(data, *args).numpy()


def _thresholded(cols, thresh):
    """Return output columns stacked, with ``|value| < thresh`` set to 0."""
    import torch
    resp = torch.stack(cols, dim=1)
    return torch.where(resp.abs() >= thresh, resp, 0.0)


def _alpha_interior_peak(x0, y0, drive, n, dt_tau, peak):
    """Return ``peak`` updated with the samples around the turning point of
    a constant-drive Alpha run of ``n > 2`` steps (requires ``tau > dt``).

    ``y[k+1] - y[k]`` has the sign of ``q (x0 - y0) - k a (x0 - drive)``, so
    the run has at most one interior maximum, after step
    ``k* = q (x0 - y0) / (a (x0 - drive))``.
    """
    import torch
    a = dt_tau
    q = float(np.float32(1) - np.float32(a))
    # Sample selection is discrete, so it carries no gradient:
    with torch.no_grad():
        u0 = x0 - drive
        k_star = torch.where(u0 > 0, q * (x0 - y0) / (a * u0), 0.0)
        # Otherwise the maximum is at the first or last step. Half a step of
        # slack absorbs rounding in `k_star`:
        if not torch.any((k_star >= 0.5) & (k_star < n)):
            return peak
        # Every candidate is an exact sample of the run, 1 <= k <= n:
        k = k_star.clamp(max=n).floor().to(torch.float64)[:, None]
        k = (k + torch.arange(3, dtype=torch.float64,
                              device=k.device)).clamp(1, n)
        log_q = math.log1p(-a)
        qk = torch.exp(k * log_q)
        ck = k * a * torch.exp((k - 1) * log_q)
        # Forms the small residual in float64 to avoid cancellation:
        rk = -torch.expm1(k * log_q) - ck
        qk, ck, rk = (c.to(y0.dtype) for c in (qk, ck, rk))
    cand = (y0[:, None] * qk + x0[:, None] * ck +
            drive[:, None] * rk).amax(dim=1)
    return torch.where(cand > peak, cand, peak)


class FadingTemporal(TemporalModel):
    r"""Generic temporal model for phosphene fading.

    Cathodic current is half-wave rectified into the drive

    .. math::

        D(t) = \max[-A(t), 0],

    where :math:`A(t)` is stimulus amplitude. Brightness follows a first-order
    leaky integrator,

    .. math::

        \tau \frac{dB}{dt} = D(t) - B(t).

    For constant drive :math:`D`, the continuous-time response is

    .. math::

        B(t) = D + [B(0) - D] e^{-t/\tau}.

    Thus :math:`\tau` controls both rise and decay. Larger values produce
    slower responses and lower peaks for brief pulses. Anodic current does not
    drive brightness.

    The model is evaluated with the explicit-Euler recurrence

    .. math::

        B_{k+1} =
        B_k + \frac{\Delta t}{\tau}\left(D_k - B_k\right),

    with :math:`\Delta t =` ``dt``. The implementation requires
    :math:`\tau \geq \Delta t`, so the discrete-time pole
    :math:`1-\Delta t/\tau` is nonnegative.

    This is a generic temporal response model, not a perceptually validated
    fit.

    Parameters
    ----------
    dt : float or Quantity, optional
        Simulation time step, in milliseconds. Default: 0.005 ms.
    tau : float or Quantity, optional
        Leaky-integrator time constant, in milliseconds. Larger values slow
        both rise and decay and reduce the peak response to brief pulses.
        Must be at least ``dt``. Default: 100 ms.
    thresh_percept : float, optional
        Brightness values below this threshold are set to zero. Default: 0.
    reduce : {'peak', 'last'}, optional
        How automatically chosen output points summarize the preceding
        interval. ``'peak'`` reports the maximum brightness reached;
        ``'last'`` reports brightness at the output instant. Explicit
        ``t_percept`` values always request those instants. Default:
        ``'peak'``.
    verbose : bool, optional
        Whether to print status messages. Default: True.

    .. versionchanged:: 0.10.0

        The drive is half-wave rectified, so only cathodic current increases
        brightness.

    .. versionchanged:: 0.12.0

        Runs on Torch; ``n_threads`` and ``n_jobs`` were removed.

    .. versionadded:: 0.7.1
    """

    #: The kernel tracks interval peaks directly.
    _reduces_intervals = True

    def __init__(self, *, tau=100, dt=0.005, thresh_percept=0, reduce='peak',
                 verbose=True):
        super().__init__(tau=tau, dt=dt, thresh_percept=thresh_percept,
                         reduce=reduce, verbose=verbose)

    def get_default_params(self):
        base_params = super(FadingTemporal, self).get_default_params()
        params = {
            'tau': 100,
            'reduce': 'peak',
        }
        base_params.update(params)
        # Torch runs on its own thread pool:
        del base_params['n_threads'], base_params['n_jobs']
        return base_params

    def get_param_units(self):
        """Return units used to store model parameters."""
        return {**super().get_param_units(), 'tau': ms}

    def _build(self):
        if self.tau <= 0:
            raise ValueError(f'"tau" must be positive, not {self.tau}.')
        # Require a nonnegative explicit-Euler pole, 1 - dt / tau.
        if self.tau < self.dt:
            raise ValueError(
                f'"tau" must be at least dt={self.dt}, not {self.tau}. A time '
                f'constant shorter than one simulation step makes the '
                f'integrator overshoot its drive by dt/tau and oscillate. '
                f'tau=dt is the fastest meaningful setting: brightness then '
                f'reaches its drive within one step, which makes the model a '
                f'half-wave rectifier. Shorten "dt" to go faster than that.')

    def _predict_temporal(self, stim, t_percept, reduce='last'):
        """Predict the float32 NumPy temporal response."""
        time = self._stim_times(stim)
        return _on_torch(self._predict_fading, self._stim_values(stim),
                         time, t_percept, reduce)

    def _predict_temporal_tensor(self, stim, t_percept, reduce='last'):
        """Predict the temporal response to tensor data, keeping its dtype
        and autograd graph."""
        return self._predict_fading(stim.data, self._stim_times(stim),
                                    t_percept, reduce)

    def _predict_fading(self, data, time, t_percept, reduce):
        """Return the Torch response to ``data`` sampled at ``time``.

        Composes each constant-drive run of ``n`` explicit-Euler steps into
        ``b <- b * q**n + drive * (1 - q**n)``, with ``q = 1 - dt/tau``.
        Timing is fixed.
        """
        import torch
        data = data.reshape((-1, len(time)))
        frame, length, out = _temporal_runs(
            time, _percept_steps(t_percept, self.dt), self.dt)
        # float32 `dt/tau`, float64 powers:
        dt_tau = np.float32(self.dt) / np.float32(self.tau)
        run_q, run_p = (c.tolist() for c in _run_powers(length, dt_tau))
        # Half-wave rectify: only cathodic current drives brightness.
        # `unbind` keeps the backward pass linear in the number of runs:
        drive = torch.clamp(-data, min=0).unbind(1)
        bright = peak = data.new_zeros(data.shape[0])
        cols = [None] * t_percept.size
        for k, col in enumerate(out):
            bright = bright * run_q[k] + drive[frame[k]] * run_p[k]
            if reduce == 'peak':
                # Brightness is monotonic within a run, so its endpoint is
                # the run's peak:
                peak = torch.where(bright > peak, bright, peak)
            if col >= 0:
                cols[col] = peak if reduce == 'peak' else bright
                # Brightness is continuous, so the boundary value starts the
                # next interval's peak:
                peak = bright
        return _thresholded(cols, self.thresh_percept)


class AlphaTemporal(TemporalModel):
    r"""Generic alpha-shaped temporal model.

    Cathodic current is half-wave rectified into

    .. math::

        D(t) = \max[-A(t), 0].

    Two identical first-order stages are then cascaded:

    .. math::

        \tau \frac{dx}{dt} = D(t) - x(t),

    .. math::

        \tau \frac{dB}{dt} = x(t) - B(t).

    For a unit-area impulse, the continuous-time impulse response is

    .. math::

        h(t) = \frac{t}{\tau^2} e^{-t/\tau},
        \qquad t \geq 0,

    which rises from zero, peaks at :math:`t=\tau`, and then decays. Anodic
    current does not drive brightness.

    The model is evaluated with the explicit-Euler recurrences

    .. math::

        x_{k+1} =
        x_k + \frac{\Delta t}{\tau}\left(D_k - x_k\right),

    .. math::

        B_{k+1} =
        B_k + \frac{\Delta t}{\tau}\left(x_k - B_k\right),

    with :math:`\Delta t =` ``dt``. The implementation requires
    :math:`\tau \geq \Delta t`.

    This is a generic temporal response model, not a perceptually validated
    fit.

    Parameters
    ----------
    dt : float or Quantity, optional
        Simulation time step, in milliseconds. Default: 0.005 ms.
    tau : float or Quantity, optional
        Time constant of both stages, in milliseconds. Larger values delay and
        broaden the response. Must be at least ``dt``. Default: 100 ms.
    thresh_percept : float, optional
        Brightness values below this threshold are set to zero. Default: 0.
    reduce : {'peak', 'last'}, optional
        How automatically chosen output points summarize the preceding
        interval. ``'peak'`` reports the maximum brightness reached;
        ``'last'`` reports brightness at the output instant. Explicit
        ``t_percept`` values always request those instants. Default:
        ``'peak'``.
    verbose : bool, optional
        Whether to print status messages. Default: True.

    .. versionchanged:: 0.12.0

        Runs on Torch; ``n_threads`` and ``n_jobs`` were removed.

    .. versionadded:: 0.10.0
    """

    #: The kernel tracks interval peaks directly.
    _reduces_intervals = True

    def __init__(self, *, tau=100, dt=0.005, thresh_percept=0, reduce='peak',
                 verbose=True):
        super().__init__(tau=tau, dt=dt, thresh_percept=thresh_percept,
                         reduce=reduce, verbose=verbose)

    def get_default_params(self):
        base_params = super(AlphaTemporal, self).get_default_params()
        params = {
            'tau': 100,
            'reduce': 'peak',
        }
        base_params.update(params)
        # Torch runs on its own thread pool:
        del base_params['n_threads'], base_params['n_jobs']
        return base_params

    def get_param_units(self):
        """Return units used to store model parameters."""
        return {**super().get_param_units(), 'tau': ms}

    def _build(self):
        if self.tau <= 0:
            raise ValueError(f'"tau" must be positive, not {self.tau}.')
        # Require a nonnegative explicit-Euler pole in each stage.
        if self.tau < self.dt:
            raise ValueError(
                f'"tau" must be at least dt={self.dt}, not {self.tau}. A time '
                f'constant shorter than one simulation step makes each stage '
                f'overshoot its input by dt/tau and oscillate. Shorten "dt" '
                f'to go faster than that.')

    def _predict_temporal(self, stim, t_percept, reduce='last'):
        """Predict the float32 NumPy temporal response."""
        time = self._stim_times(stim)
        return _on_torch(self._predict_alpha, self._stim_values(stim), time,
                         t_percept, reduce)

    def _predict_temporal_tensor(self, stim, t_percept, reduce='last'):
        """Predict the temporal response to tensor data, keeping its dtype
        and autograd graph."""
        return self._predict_alpha(stim.data, self._stim_times(stim),
                                   t_percept, reduce)

    def _predict_alpha(self, data, time, t_percept, reduce):
        """Return the Torch response to ``data`` sampled at ``time``.

        For a constant drive ``d``, with ``a = dt/tau`` and ``q = 1 - a``,
        ``n`` explicit-Euler steps from states ``x0``, ``y0`` compose to

            x_n = d + q**n (x0 - d)
            y_n = d + q**n (y0 - d) + n a q**(n-1) (x0 - d).

        Timing is fixed. With ``reduce='peak'``, each run's maximum is taken
        over its first and last steps and the samples around its turning
        point, which is exact.
        """
        import torch
        data = data.reshape((-1, len(time)))
        frame, length, out = _temporal_runs(
            time, _percept_steps(t_percept, self.dt), self.dt)
        # float32 `dt/tau`, float64 powers:
        dt_tau = np.float32(self.dt) / np.float32(self.tau)
        a = float(dt_tau)
        run_q, run_p = _run_powers(length, dt_tau)
        run_c = length * a * _run_powers(length - 1, dt_tau)[0]
        # Nonnegative weights on y0, x0, and d avoid cancellation between
        # O(a) terms while the response rises:
        run_r = run_p - run_c
        # One step is the plain recurrence, in which stage 2 reads the old
        # stage 1, so the drive weight is exactly zero:
        one = length == 1
        run_q[one] = np.float32(1) - dt_tau
        run_p[one], run_c[one], run_r[one] = a, a, 0
        run_q, run_p, run_c, run_r = (c.tolist() for c in
                                      (run_q, run_p, run_c, run_r))
        # Half-wave rectify: only cathodic current drives the cascade.
        # `unbind` keeps the backward pass linear in the number of runs:
        drive = torch.clamp(-data, min=0).unbind(1)
        first = second = peak = data.new_zeros(data.shape[0])
        cols = [None] * t_percept.size
        for k, col in enumerate(out):
            d, n = drive[frame[k]], int(length[k])
            x0, y0 = first, second
            first = x0 * run_q[k] + d * run_p[k]
            # Clip negative roundoff:
            second = torch.clamp(y0 * run_q[k] + x0 * run_c[k] + d * run_r[k],
                                 min=0)
            if reduce == 'peak':
                # The run's first and last steps:
                step = y0 + a * (x0 - y0)
                peak = torch.where(step > peak, step, peak)
                peak = torch.where(second > peak, second, peak)
                if n > 2 and a < 1:
                    peak = _alpha_interior_peak(x0, y0, d, n, a, peak)
            if col >= 0:
                cols[col] = peak if reduce == 'peak' else second
                # Brightness is continuous, so the boundary value starts the
                # next interval's peak:
                peak = second
        return _thresholded(cols, self.thresh_percept)
