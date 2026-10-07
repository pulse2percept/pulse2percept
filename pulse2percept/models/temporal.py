""":py:class:`~pulse2percept.models.FadingTemporal`,
:py:class:`~pulse2percept.models.AlphaTemporal`"""
import math
from contextlib import contextmanager

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
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
    if dt_tau >= 1:
        # tau <= dt: q <= 0 has no logarithm (0**0 = 1):
        qn = np.power(1 - np.float64(dt_tau), n)
        return qn, 1 - qn
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


#: Approximate working-memory target (bytes) per chunk in ``_charge_chunks``.
_CASCADE_BLOCK_BYTES = 32 * 2 ** 20

#: Longest chunk, in ``dt`` steps.
_CASCADE_MAX_STEPS = 2 ** 16

#: Largest ``(m + 3, n)`` slow-cascade map of a chunk, in entries.
_CASCADE_MAX_ENTRIES = 2 ** 20


def _cascade_chunks(time, t_percept, dt, n_space, itemsize, merge=True):
    """Split the simulation into chunks of constant stimulus.

    Returns the stimulus frame, length in ``dt`` steps, and 1-based output
    steps of each chunk. Consecutive ``_temporal_runs`` of one frame share a
    chunk while it fits; with ``merge=False``, each output ends a chunk.
    """
    frame, length, out = _temporal_runs(time, _percept_steps(t_percept, dt),
                                        dt)
    # About eight (n, n_space) temporaries per chunk:
    max_steps = int(np.clip(_CASCADE_BLOCK_BYTES // (8 * n_space * itemsize),
                            1, _CASCADE_MAX_STEPS))
    chunks, cur = [], None
    for f, n, col in zip(frame, length.tolist(), out):
        if cur is not None and (
                not merge or cur[0] != f or cur[1] + n > max_steps or
                (cur[1] + n) * (len(cur[2]) + 4) > _CASCADE_MAX_ENTRIES):
            chunks.append(cur)
            cur = None
        while n > max_steps:
            chunks.append([f, max_steps, []])
            n -= max_steps
        if cur is None:
            cur = [f, 0, []]
        cur[1] += n
        if col >= 0:
            cur[2].append(cur[1])
    chunks.append(cur)
    return [(f, n, tuple(offs)) for f, n, offs in chunks]


def _rising(x, d):
    """Return ``C(x + d - 1, d)`` for ``d`` in {0, 1, 2}, else 0."""
    return np.select([d == 0, d == 1, d == 2],
                     [np.ones_like(x), x, x * (x + 1) / 2], 0)


def _fast_coeffs(n, a1, a2, eps):
    """Return the float64 ``(n + 2, 5)`` map from ``(r1, drive, r2, charge,
    dq)`` to ``r1 - eps r2`` after each of ``n`` steps, then to ``r1`` and
    ``r2`` after the last.

    ``r1`` relaxes to the drive at rate ``a1``; ``r2`` relaxes at rate ``a2``
    to the charge, which grows by ``dq`` per step and is updated first.
    """
    k = np.arange(1, n + 1, dtype=np.float64)
    q1, p1 = _run_powers(k, a1)
    q2, p2 = _run_powers(k, a2)
    # Weight of dq in r2: a2 * sum_i i q2**(k - i):
    c2 = k - (1 - a2) / a2 * p2
    coeffs = np.zeros((n + 2, 5))
    coeffs[:n] = np.stack((q1, p1, -eps * q2, -eps * p2, -eps * c2), axis=1)
    coeffs[n, :2] = q1[-1], p1[-1]
    coeffs[n + 1, 2:] = q2[-1], p2[-1], c2[-1]
    return coeffs


def _slow_coeffs(n, offsets, a, inputs=True):
    """Return float64 maps of ``n`` steps of three identical explicit-Euler
    stages at rate ``a``, each reading the updated stage before it.

    The ``(m + 3, 3)`` map takes the stage states, and the ``(m + 3, n)`` map
    the inputs, to stage 3 after each of the ``m`` 1-based ``offsets``, then
    to stages 1-3 after step ``n``. The input map is None if not ``inputs``.
    """
    k = np.array(list(offsets) + [n] * 3)
    stage = np.array([3] * len(offsets) + [1, 2, 3])
    # State j reaches stage i after k steps as C(k + d - 1, d) a**d q**k,
    # d = i - j:
    d = stage[:, None] - np.arange(1, 4)
    carry = (a ** d * _rising(k[:, None].astype(np.float64), d) *
             _run_powers(k, a)[0][:, None])
    if not inputs:
        return carry, None
    # An input reaches stage i after lag steps as C(lag + i - 1, i - 1)
    # a**i q**lag:
    lag = np.arange(n, dtype=np.float64)
    kernel = (a ** np.arange(1, 4)[:, None] *
              _rising(lag + 1, np.arange(3)[:, None]) * _run_powers(lag, a)[0])
    # Input j (1-based) reaches stage i at step k with kernel[i, k - j], which
    # is `padded[i, n - k + j - 1]`, or 0 for j > k:
    padded = np.concatenate((kernel[:, ::-1], np.zeros((3, n))), axis=1)
    weights = sliding_window_view(padded, n, axis=1)[stage - 1, n - k]
    return carry, weights


@contextmanager
def _flush_denormals():
    """Flush subnormal floats to zero on this thread, as the deleted Cython
    kernels did. Decaying integrators produce them, and arithmetic on them
    is ~20x slower."""
    import torch
    # Torch has no getter; probe whether flushing is already on:
    was_on = (torch.tensor(1e-39, dtype=torch.float32) * 1.0).item() == 0
    torch.set_flush_denormal(True)
    try:
        yield
    finally:
        torch.set_flush_denormal(was_on)


def _quiet_after(r1, r2, charge, eps, a1):
    """Return the steps after which ``r1 - eps r2`` stays negative at every
    location under zero drive, or None if that cannot be shown.

    Under zero drive, ``r1`` decays as ``r1 q1**k``, and ``r2`` stays between
    its value and the charge, both nonnegative. ``r1 q1**k <= eps min(r2,
    charge) / 2`` thus bounds ``r1 - eps r2`` below zero, with margin for
    rounding.
    """
    import torch
    with torch.no_grad():
        r1 = r1.double().cpu().numpy()
        floor = (eps * torch.minimum(r2, charge)).double().cpu().numpy()
    pos = r1 > 0
    if not pos.any():
        # A sum of nonpositive terms rounds to a nonpositive value:
        return 0
    if not np.all(floor[pos] > 0):
        return None
    k = np.log(floor[pos] / (2 * r1[pos])) / np.log1p(-a1)
    return max(0, int(np.ceil(k.max())))


def _charge_chunks(model, data, time, t_percept, drive_sign, merge=True,
                   skip_quiet=True):
    """Yield the Horsager/Nanduri rectifier input of each chunk.

    For each constant-stimulus chunk of ``n`` steps (``_cascade_chunks``),
    yields ``r1 - eps/1000 r2``, shape ``(n, n_space)``, then the
    ``_slow_coeffs`` maps of the chunk and its output offsets. ``r1`` is
    driven by ``drive_sign`` times the stimulus; charge accumulates its
    anodic part. Coefficients are float32-rounded, as in the deleted Cython
    kernels; maps have the dtype and device of ``data``.

    With ``skip_quiet``, steps of zero drive at which ``r1 - eps/1000 r2`` is
    provably negative everywhere, and so rectifies to zero, are yielded as
    one chunk whose input and input map are None.
    """
    import torch
    f32 = np.float32
    dt = f32(model.dt)
    a1, a2, a3 = (float(dt / f32(tau))
                  for tau in (model.tau1, model.tau2, model.tau3))
    # `eps` was fit with a microsecond time step:
    eps = float(f32(model.eps) / f32(1000))
    # Time-major: Torch is several times faster reducing and multiplying
    # along the leading axis here:
    data = data.reshape((-1, len(time))).T.contiguous()
    kw = {'dtype': data.dtype, 'device': data.device}
    # `unbind` keeps the backward pass linear in the number of chunks:
    drive = (drive_sign * data).unbind(0)
    dq = (torch.clamp(data, min=0) * float(dt)).unbind(0)
    # The bound in `_quiet_after` requires finite, nonnegative eps and
    # decaying, nonnegative poles; otherwise every step is computed:
    skip_quiet = (skip_quiet and np.isfinite(eps) and eps >= 0 and
                  0 < a1 <= 1 and 0 < a2 <= 1)
    zero = (data == 0).all(dim=1).tolist()
    r1 = r2 = charge = data.new_zeros(data.shape[1])
    fast, slow, carry = {}, {}, {}
    for f, n, offs in _cascade_chunks(time, t_percept, model.dt,
                                      data.shape[1], data.element_size(),
                                      merge):
        quiet = (_quiet_after(r1, r2, charge, eps, a1)
                 if skip_quiet and zero[f] else None)
        if quiet is not None:
            # Coarse steps let periodic stimuli reuse coefficients:
            quiet = -(-quiet // 256) * 256
        if quiet is None or quiet >= n:
            quiet = n
        if quiet:
            # Steps that need the rectifier input:
            head = tuple(k for k in offs if k <= quiet)
            if quiet not in fast:
                fast[quiet] = torch.as_tensor(
                    _fast_coeffs(quiet, a1, a2, eps), **kw)
            if (quiet, head) not in slow:
                slow[quiet, head] = tuple(
                    torch.as_tensor(c, **kw)
                    for c in _slow_coeffs(quiet, head, a3))
            y = fast[quiet] @ torch.stack((r1, drive[f], r2, charge, dq[f]))
            r1, r2 = y[quiet], y[quiet + 1]
            charge = charge + quiet * dq[f]
            yield (y[:quiet], *slow[quiet, head], head)
        if quiet < n:
            m = n - quiet
            tail = tuple(k - quiet for k in offs if k > quiet)
            if (m, tail) not in carry:
                carry[m, tail] = torch.as_tensor(
                    _slow_coeffs(m, tail, a3, inputs=False)[0], **kw)
            # Zero drive: r1 decays and r2 relaxes to the constant charge:
            q1 = float(_run_powers(m, a1)[0])
            q2, p2 = (float(c) for c in _run_powers(m, a2))
            r1, r2 = r1 * q1, r2 * q2 + charge * p2
            yield None, carry[m, tail], None, tail


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
