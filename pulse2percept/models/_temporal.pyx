from libc.math cimport(pow as c_pow, exp as c_exp, fabs as c_abs,
                       sqrt as c_sqrt, log1p as c_log1p, expm1 as c_expm1)
from cython.parallel import prange
from cython.parallel cimport threadid
from pulse2percept.utils._fpmode cimport c_denormals_off, c_fpmode_restore
from cython import cdivision  # modulo, division by zero
import numpy as np
cimport numpy as cnp
cnp.import_array()

ctypedef cnp.float32_t float32
ctypedef cnp.int32_t int32
ctypedef cnp.uint32_t uint32
ctypedef Py_ssize_t index_t

# How many spatial locations one thread integrates at a time. The running
# brightness for a block this size stays in L1 next to the stimulus row it is
# read against, and the block is wide enough to fill the vector registers the
# inner loop compiles down to.
cdef index_t BLOCK = 64

# How a percept time point summarizes the interval that led up to it.
cdef uint32 REDUCE_LAST = 0
cdef uint32 REDUCE_PEAK = 1


cdef index_t _build_runs(const float32[::1] t_stim,
                         const uint32[::1] idx_t_percept,
                         float32 dt,
                         index_t n_sim,
                         index_t n_stim,
                         index_t n_percept,
                         int32[::1] run_frame,
                         int32[::1] run_len,
                         int32[::1] run_out) noexcept nogil:
    """Group the simulation clock into runs of steps that share one frame.

    Serial and location-independent, so it is walked once per call rather than
    once per location: O(n_sim) against the O(n_sim x n_space) it saves.

    A run ends where the frame changes or where a percept is due, so within
    one run the drive is constant and an integrator can be advanced across it
    in closed form. ``run_out`` carries the percept column to write at the end
    of the run, or -1.

    This is the only place the per-step frame rule lives. Each step is asked
    which frame it reads with the same comparison the per-step loops used to
    make, so a frame that no step ever lands on is skipped here exactly as it
    was skipped there.

    ``while``, not ``if``: more than one stimulus frame can fall inside a
    single simulation step, and skipping only one of them leaves the
    integrator reading a frame that is already in the past. Encoded pulses
    make that the normal case rather than a corner case -- their edges sit on
    the DT=1e-3 ms grid while ``dt`` defaults to 5e-3 ms, so a pulse edge and
    the sample after it routinely share a step.

    Returns the number of runs written.
    """
    cdef:
        index_t idx_sim, idx_stim = 0, idx_frame = 0
        index_t n_runs = 0, length = 0, prev_frame = -1
        float32 t_sim

    for idx_sim in range(n_sim):
        t_sim = idx_sim * dt
        while idx_stim + 1 < n_stim and t_sim >= t_stim[idx_stim + 1]:
            idx_stim = idx_stim + 1
        if length > 0 and idx_stim != prev_frame:
            run_frame[n_runs] = <int32>prev_frame
            run_len[n_runs] = <int32>length
            run_out[n_runs] = -1
            n_runs = n_runs + 1
            length = 0
        prev_frame = idx_stim
        length = length + 1
        if (idx_frame < n_percept and
                idx_sim == <index_t>idx_t_percept[idx_frame]):
            run_frame[n_runs] = <int32>idx_stim
            run_len[n_runs] = <int32>length
            run_out[n_runs] = <int32>idx_frame
            n_runs = n_runs + 1
            length = 0
            idx_frame = idx_frame + 1
    return n_runs


@cdivision(True)
cpdef fading_fast(const float32[:, ::1] stim,
                  const float32[::1] t_stim,
                  const uint32[::1] idx_t_percept,
                  float32 dt,
                  float32 tau,
                  float32 thresh_percept,
                  uint32 n_threads,
                  uint32 reduce=REDUCE_LAST):
    """Cython implementation of the generic fading model

    The leaky integrator has to be stepped in order, so the loop over time is
    serial. But, each spatial location integrates independently of every
    other. Time is therefore the *outer* loop and space the inner one, which
    leaves the inner loop free of any carried dependency and lets it
    vectorize. Threads take a block of locations each and run the whole time
    loop over it, so the parallel region is still entered only once.

    The outer loop runs over *runs* of simulation steps rather than over the
    steps themselves. Within a run the stimulus frame does not change, so the
    drive is constant and the Euler recurrence
    ``b <- b + (dt / tau) * (drive - b)`` is a fixed affine map; ``n`` of them
    compose into ``b <- b * q**n + drive * (1 - q**n)``, for
    ``q = 1 - dt/tau``.
    That is the closed form of the *discrete* recurrence, not of the ODE it
    approximates, so a pulse falling entirely between two simulation steps is
    still missed exactly as before -- which frame each step reads is settled
    by the same comparison, one step at a time, in the pre-pass below.

    ``q**n`` and ``1 - q**n`` are both carried because neither alone stays
    accurate: for a short run ``q**n`` is near 1 and ``drive + (b - drive) *
    q**n`` cancels, while for a long one ``1 - q**n`` rounds to 1 and the
    decaying tail is lost. Weighting the two endpoints avoids both, and
    ``expm1`` supplies ``1 - q**n`` without cancelling when it is small.

    Parameters
    ----------
    stim : 2D float32 array
        A ``Stimulus.data`` container that contains spatial locations as rows
        and time points as columns. This is the output of the spatial model.
        The time points are specified in ``t_stim``.
    t_stim : 1D float32 array
        The time points for ``stim`` above.
    dt : float32
        Sampling time step (ms)
    tau: float32
        Time decay constant for the fast leaky integrater (ms).
    thresh_percept : float32
        Spatial responses smaller than ``thresh_percept`` will be set to zero
    n_threads: uint32
        Number of CPU threads to use during parallelization using OpenMP. 
        Defaults to maximum number of cores on user CPU
    reduce : uint32
        How each percept time point summarizes the interval since the previous
        one: 0 reports the brightness at that instant, 1 reports the peak
        brightness reached over the interval.

        Electrical stimulation is pulsatile, so brightness rises and falls
        within one output interval. Reporting the instant the interval happens
        to end on samples a signal whose energy lives in sub-millisecond
        transients, and the sampling phase then walks through the pulse cycle:
        neighbouring frames come out orders of magnitude apart for no reason a
        viewer would recognize. Within a constant-drive run brightness is
        monotonic, so checking the run endpoints gives the exact interval
        peak without evaluating every simulation step.

    Returns
    -------
    percept : 2D float32 array
        space x time

    """
    cdef:
        float32 t_sim, amp, drive, bright, peak, dt_tau, q, p
        float32[:, ::1] percept
        # `const`: the caller's stimulus can be read-only, and the transpose
        # below is a no-op for a single location, so this is not always a copy
        const float32[:, ::1] stim_t
        float32[:, ::1] scratch
        float32[:, ::1] running
        float32[::1] run_q
        float32[::1] run_p
        int32[::1] run_frame
        int32[::1] run_len
        int32[::1] run_out
        index_t idx_space, idx_sim, idx_stim, idx_frame, idx_block, idx_run
        index_t n_space, n_stim, n_percept, n_sim, n_blocks, lo, hi, tid
        index_t n_runs, max_runs
        double log_q, n_log_q
        unsigned long long fpmode

    n_percept = len(idx_t_percept)  # Py overhead
    n_stim = len(t_stim)  # Py overhead
    n_sim = idx_t_percept[n_percept - 1] + 1  # no negative indices
    n_space = stim.shape[0]
    if n_threads < 1:  # `num_threads(0)` is not conforming OpenMP
        n_threads = 1

    percept = np.zeros((n_space, n_percept), dtype=np.float32)  # Py overhead
    # `dt / tau` is the step of the recurrence being composed below. Rounding
    # it to float32 here rather than carrying `dt` and `tau` separately is what
    # keeps the composed map a power of the step the per-step loop would have
    # taken:
    dt_tau = dt / tau
    # One run reads the same stimulus frame for every location, so transpose
    # once and that read becomes a contiguous run:
    stim_t = np.ascontiguousarray(np.asarray(stim).T)  # Py overhead
    n_blocks = (n_space + BLOCK - 1) // BLOCK
    # Running brightness, one row per thread. Rows are BLOCK floats apart, so
    # no two threads share a cache line.
    scratch = np.empty((n_threads, BLOCK), dtype=np.float32)  # Py overhead
    # Peak brightness since the last percept time point, laid out the same way.
    # Allocated even when it is not used, so that the loop below can be written
    # once:
    running = np.empty((n_threads, BLOCK), dtype=np.float32)  # Py overhead

    # A run ends when the frame changes or a percept lands on it, so there can
    # be no more runs than there are frames plus output points -- nor, of
    # course, than there are simulation steps:
    max_runs = n_stim + n_percept + 1
    if max_runs > n_sim:
        max_runs = n_sim
    run_frame = np.empty(max_runs, dtype=np.int32)  # Py overhead
    run_len = np.empty(max_runs, dtype=np.int32)  # Py overhead
    run_out = np.empty(max_runs, dtype=np.int32)  # Py overhead
    run_q = np.empty(max_runs, dtype=np.float32)  # Py overhead
    run_p = np.empty(max_runs, dtype=np.float32)  # Py overhead

    log_q = c_log1p(-<double>dt_tau)
    with nogil:
        n_runs = _build_runs(t_stim, idx_t_percept, dt, n_sim, n_stim,
                             n_percept, run_frame, run_len, run_out)
        # `q**n` and `1 - q**n` for each run. Both are the same number at
        # every location, so they are settled here rather than inside the
        # parallel loop:
        for idx_run in range(n_runs):
            n_log_q = run_len[idx_run] * log_q
            run_q[idx_run] = <float32>c_exp(n_log_q)
            run_p[idx_run] = <float32>(-c_expm1(n_log_q))

    for idx_block in prange(n_blocks, schedule='static', nogil=True,
                            num_threads=n_threads):
        # Brightness decays down through the subnormal range between pulses,
        # where the arithmetic costs ~100x what it does on normal floats; see
        # `utils/_fpmode.pxd`. The mode is per-thread, hence set here rather
        # than around the `prange`:
        fpmode = c_denormals_off()
        tid = threadid()
        lo = idx_block * BLOCK
        hi = lo + BLOCK
        if hi > n_space:
            hi = n_space
        for idx_space in range(hi - lo):
            scratch[tid, idx_space] = <float32>0.0
            running[tid, idx_space] = <float32>0.0
        for idx_run in range(n_runs):
            idx_stim = run_frame[idx_run]
            q = run_q[idx_run]
            p = run_p[idx_run]
            for idx_space in range(hi - lo):
                amp = stim_t[idx_stim, lo + idx_space]
                bright = scratch[tid, idx_space]
                # Half-wave rectify: only cathodic (negative) current drives
                # brightness. Without this the model cannot see a
                # charge-balanced pulse at all:
                drive = -amp
                if drive < 0.0:
                    drive = 0.0
                # The whole run in one affine step. Brightness stays in
                # [0, inf[ without a clamp: `q` and `p` are both in [0, 1]
                # (`tau >= dt` is enforced), so this is a convex combination
                # of two nonnegative numbers.
                bright = bright * q + drive * p
                scratch[tid, idx_space] = bright
                # `drive` is constant across the run, so brightness moves
                # monotonically from one end of it to the other and the peak
                # over the run is at an endpoint. Comparing here therefore
                # tracks the same peak the per-step compare did:
                if bright > running[tid, idx_space]:
                    running[tid, idx_space] = bright
            idx_frame = run_out[idx_run]
            if idx_frame >= 0:
                # `idx_t_percept` stores the time points at which we need to
                # output a percept. The pre-pass ended a run on each of them,
                # so reaching one is a property of the run rather than a
                # comparison to make here:
                for idx_space in range(hi - lo):
                    if reduce == REDUCE_PEAK:
                        bright = running[tid, idx_space]
                    else:
                        bright = scratch[tid, idx_space]
                    if c_abs(bright) >= thresh_percept:
                        percept[lo + idx_space, idx_frame] = bright
                    # Start the next interval's peak from where this one left
                    # off, not from zero: brightness is continuous, so the
                    # value carried across the boundary is a floor on what the
                    # next interval reaches:
                    running[tid, idx_space] = scratch[tid, idx_space]
        # Hand the thread back in the floating-point mode it arrived in:
        c_fpmode_restore(fpmode)

    return np.asarray(percept)  # Py overhead


@cdivision(True)
cpdef alpha_fast(const float32[:, ::1] stim,
                 const float32[::1] t_stim,
                 const uint32[::1] idx_t_percept,
                 float32 dt,
                 float32 tau,
                 float32 thresh_percept,
                 uint32 n_threads,
                 uint32 reduce=REDUCE_LAST):
    """Cython implementation of the generic alpha model

    Two leaky integrators in series, both with time constant ``tau`` and unit
    DC gain, driven by the cathodic half of the stimulus. The cascade's
    impulse response is ``t / tau**2 * exp(-t / tau)``, which starts at zero,
    peaks at ``t = tau`` and decays afterwards.

    Blocking, the run schedule, denormal handling and interval peak tracking
    follow the same principles as ``fading_fast``; see there. Each location
    carries two states rather than one, and both compose across a run of
    ``n`` steps at constant drive ``d``. Writing ``a = dt/tau``, ``q = 1 - a``
    and taking the run's starting states as ``x0``, ``y0``::

        x_n = d + q**n * (x0 - d)
        y_n = d + q**n * (y0 - d) + n * a * q**(n - 1) * (x0 - d)

    the second term on ``y`` being what the first stage feeds in over the run.

    The peak needs more care than it does for one stage, because ``y`` is not
    monotonic within a run: the cascade is what gives the model a rise time,
    so brightness can climb after the drive has gone. It is however
    *unimodal*, which is enough. Since ``y[k+1] - y[k] = a * (x[k] - y[k])``
    and::

        x[k] - y[k] = q**(k - 1) * (q * (x0 - y0) - k * a * (x0 - d))

    the bracket is linear in ``k`` and ``q**(k-1)`` is positive, so the
    difference changes sign at most once across the run. A strictly interior
    maximum therefore exists only when ``x0 > d`` -- stage one above the
    drive, brightness still catching up -- and sits at the crossing
    ``k* = q * (x0 - y0) / (a * (x0 - d))``. Evaluating ``y`` at the two ends
    and either side of ``k*`` gives the exact interval peak in O(1), so no
    part of this kernel walks the simulation clock per location.

    Parameters
    ----------
    stim : 2D float32 array
        A ``Stimulus.data`` container that contains spatial locations as rows
        and time points as columns. This is the output of the spatial model.
        The time points are specified in ``t_stim``.
    t_stim : 1D float32 array
        The time points for ``stim`` above.
    dt : float32
        Sampling time step (ms)
    tau : float32
        Time constant of both leaky stages (ms).
    thresh_percept : float32
        Spatial responses smaller than ``thresh_percept`` will be set to zero
    n_threads : uint32
        Number of CPU threads to use during parallelization using OpenMP.
        Defaults to maximum number of cores on user CPU
    reduce : uint32
        How each percept time point summarizes the interval since the previous
        one: 0 reports the brightness at that instant, 1 reports the peak
        brightness reached over the interval. The second stage is what is
        reported and what the peak is tracked on.

    Returns
    -------
    percept : 2D float32 array
        space x time

    """
    cdef:
        float32 amp, drive, x0, y0, xn, yn, u0, kstar, cand, dt_tau, q
        float32 qn, pn, cn, rn
        double d_qn, d_pn, d_cn
        float32[:, ::1] percept
        # `const`: the caller's response can be read-only, and the transpose
        # below is a no-op for a single location, so this is not always a copy
        const float32[:, ::1] stim_t
        float32[:, ::1] first
        float32[:, ::1] second
        float32[:, ::1] running
        float32[::1] run_q
        float32[::1] run_p
        float32[::1] run_c
        float32[::1] run_r
        float32[::1] pow_q
        float32[::1] pow_c
        float32[::1] pow_r
        int32[::1] run_frame
        int32[::1] run_len
        int32[::1] run_out
        index_t idx_space, idx_frame, idx_block, idx_run, idx_stim
        index_t n_space, n_stim, n_percept, n_sim, n_blocks, lo, hi, tid
        index_t n_runs, max_runs, max_len, n_run, k, kk, j
        double log_q, n_log_q
        unsigned long long fpmode

    n_percept = len(idx_t_percept)  # Py overhead
    n_stim = len(t_stim)  # Py overhead
    n_sim = idx_t_percept[n_percept - 1] + 1  # no negative indices
    n_space = stim.shape[0]
    if n_threads < 1:  # `num_threads(0)` is not conforming OpenMP
        n_threads = 1

    percept = np.zeros((n_space, n_percept), dtype=np.float32)  # Py overhead
    dt_tau = dt / tau
    q = <float32>1.0 - dt_tau
    stim_t = np.ascontiguousarray(np.asarray(stim).T)  # Py overhead
    n_blocks = (n_space + BLOCK - 1) // BLOCK
    # One row per thread for each of the two stages and for the running peak,
    # so no two threads share a cache line:
    first = np.empty((n_threads, BLOCK), dtype=np.float32)  # Py overhead
    second = np.empty((n_threads, BLOCK), dtype=np.float32)  # Py overhead
    running = np.empty((n_threads, BLOCK), dtype=np.float32)  # Py overhead

    max_runs = n_stim + n_percept + 1
    if max_runs > n_sim:
        max_runs = n_sim
    run_frame = np.empty(max_runs, dtype=np.int32)  # Py overhead
    run_len = np.empty(max_runs, dtype=np.int32)  # Py overhead
    run_out = np.empty(max_runs, dtype=np.int32)  # Py overhead
    run_q = np.empty(max_runs, dtype=np.float32)  # Py overhead
    run_p = np.empty(max_runs, dtype=np.float32)  # Py overhead
    run_c = np.empty(max_runs, dtype=np.float32)  # Py overhead
    run_r = np.empty(max_runs, dtype=np.float32)  # Py overhead

    # Both states come out of a run as a weighted average of where the run
    # started and the drive it ran at, never as a difference of two larger
    # numbers. That matters most at the onset of a response, where `y` is
    # orders of magnitude below both `drive` and the terms an algebraically
    # equal grouping would subtract. The three weights on `y` are
    # nonnegative and sum to one -- `1 - q**n - n*a*q**(n-1)` is the chance
    # of at least two successes in `n` Bernoulli(`a`) trials -- so the result
    # is bracketed by the values going in.
    log_q = c_log1p(-<double>dt_tau)
    with nogil:
        n_runs = _build_runs(t_stim, idx_t_percept, dt, n_sim, n_stim,
                             n_percept, run_frame, run_len, run_out)
        max_len = 1
        for idx_run in range(n_runs):
            n_run = run_len[idx_run]
            if n_run > max_len:
                max_len = n_run
            if n_run == 1:
                # A single step is the plain recurrence, and its weights are
                # exact: `1 - q - a` is zero, not the 1-ulp residue `expm1`
                # would leave. That residue is the whole output of the first
                # step of a response, which has to be exactly zero -- stage
                # two reads a stage one that has not moved yet.
                run_q[idx_run] = q
                run_p[idx_run] = dt_tau
                run_c[idx_run] = dt_tau
                run_r[idx_run] = <float32>0.0
            else:
                n_log_q = n_run * log_q
                d_qn = c_exp(n_log_q)
                d_pn = -c_expm1(n_log_q)
                d_cn = n_run * <double>dt_tau * c_exp((n_run - 1) * log_q)
                run_q[idx_run] = <float32>d_qn
                run_p[idx_run] = <float32>d_pn
                run_c[idx_run] = <float32>d_cn
                # Differenced in double, where the two are still far enough
                # apart to leave the small result its significant digits:
                run_r[idx_run] = <float32>(d_pn - d_cn)

    # The same three weights at every k a peak can land on. Only the peak
    # search indexes these, and only at the steps either side of the turning
    # point:
    if reduce == REDUCE_PEAK:
        pow_q = np.empty(max_len + 1, dtype=np.float32)  # Py overhead
        pow_c = np.empty(max_len + 1, dtype=np.float32)  # Py overhead
        pow_r = np.empty(max_len + 1, dtype=np.float32)  # Py overhead
        with nogil:
            pow_q[0] = <float32>1.0
            pow_c[0] = <float32>0.0
            pow_r[0] = <float32>0.0
            pow_q[1] = q
            pow_c[1] = dt_tau
            pow_r[1] = <float32>0.0
            for k in range(2, max_len + 1):
                d_qn = c_exp(k * log_q)
                d_pn = -c_expm1(k * log_q)
                d_cn = k * <double>dt_tau * c_exp((k - 1) * log_q)
                pow_q[k] = <float32>d_qn
                pow_c[k] = <float32>d_cn
                pow_r[k] = <float32>(d_pn - d_cn)
    else:
        pow_q = np.empty(1, dtype=np.float32)  # Py overhead
        pow_c = np.empty(1, dtype=np.float32)  # Py overhead
        pow_r = np.empty(1, dtype=np.float32)  # Py overhead

    for idx_block in prange(n_blocks, schedule='static', nogil=True,
                            num_threads=n_threads):
        fpmode = c_denormals_off()
        tid = threadid()
        lo = idx_block * BLOCK
        hi = lo + BLOCK
        if hi > n_space:
            hi = n_space
        for idx_space in range(hi - lo):
            first[tid, idx_space] = <float32>0.0
            second[tid, idx_space] = <float32>0.0
            running[tid, idx_space] = <float32>0.0
        for idx_run in range(n_runs):
            idx_stim = run_frame[idx_run]
            n_run = run_len[idx_run]
            qn = run_q[idx_run]
            pn = run_p[idx_run]
            cn = run_c[idx_run]
            rn = run_r[idx_run]
            for idx_space in range(hi - lo):
                amp = stim_t[idx_stim, lo + idx_space]
                x0 = first[tid, idx_space]
                y0 = second[tid, idx_space]
                # Half-wave rectify: only cathodic (negative) current drives
                # the cascade.
                drive = -amp
                if drive < 0.0:
                    drive = 0.0
                # How far stage one starts above the drive. This is what the
                # first stage feeds the second over the run, and its sign is
                # also what decides whether the run can peak in its interior:
                u0 = x0 - drive
                xn = x0 * qn + drive * pn
                yn = y0 * qn + x0 * cn + drive * rn
                # Brightness is bounded in [0, inf[. Both stages are convex
                # combinations of nonnegative quantities, so this only ever
                # catches rounding:
                if yn < 0.0:
                    yn = 0.0
                first[tid, idx_space] = xn
                second[tid, idx_space] = yn
                if reduce == REDUCE_PEAK:
                    # `y` is unimodal across the run, so the ends always have
                    # to be looked at. The first is one plain step from the
                    # state the run began in:
                    cand = y0 + dt_tau * (x0 - y0)
                    if cand > running[tid, idx_space]:
                        running[tid, idx_space] = cand
                    if yn > running[tid, idx_space]:
                        running[tid, idx_space] = yn
                    # An interior maximum needs stage one to start above the
                    # drive; otherwise the interior turning point is a
                    # minimum and the ends already have it:
                    if u0 > 0.0 and n_run > 2:
                        kstar = q * (x0 - y0) / (dt_tau * u0)
                        if kstar >= 1.0:
                            kk = <index_t>kstar + 1
                            # Either side of the crossing, since which of the
                            # two neighbouring steps is higher is decided by
                            # a rounding of `kstar`:
                            for j in range(3):
                                k = kk - 1 + j
                                if k < 1:
                                    k = 1
                                if k > n_run:
                                    k = n_run
                                cand = (y0 * pow_q[k] + x0 * pow_c[k] +
                                        drive * pow_r[k])
                                if cand > running[tid, idx_space]:
                                    running[tid, idx_space] = cand
            idx_frame = run_out[idx_run]
            if idx_frame >= 0:
                for idx_space in range(hi - lo):
                    if reduce == REDUCE_PEAK:
                        yn = running[tid, idx_space]
                    else:
                        yn = second[tid, idx_space]
                    if c_abs(yn) >= thresh_percept:
                        percept[lo + idx_space, idx_frame] = yn
                    # Brightness is continuous, so the value carried across
                    # the boundary is a floor on the next interval's peak:
                    running[tid, idx_space] = second[tid, idx_space]
        c_fpmode_restore(fpmode)

    return np.asarray(percept)  # Py overhead
