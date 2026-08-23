from libc.math cimport(pow as c_pow, exp as c_exp, fabs as c_abs,
                       sqrt as c_sqrt)
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
        viewer would recognize. The peak is tracked across every ``dt`` step,
        so it costs one compare per step and is exact at any output rate.

    Returns
    -------
    percept : 2D float32 array
        space x time

    """
    cdef:
        float32 t_sim, amp, drive, bright, peak, dt_tau
        float32[:, ::1] percept
        float32[:, ::1] stim_t
        float32[:, ::1] scratch
        float32[:, ::1] running
        index_t idx_space, idx_sim, idx_stim, idx_frame, idx_block
        index_t n_space, n_stim, n_percept, n_sim, n_blocks, lo, hi, tid
        unsigned long long fpmode

    n_percept = len(idx_t_percept)  # Py overhead
    n_stim = len(t_stim)  # Py overhead
    n_sim = idx_t_percept[n_percept - 1] + 1  # no negative indices
    n_space = stim.shape[0]
    if n_threads < 1:  # `num_threads(0)` is not conforming OpenMP
        n_threads = 1

    percept = np.zeros((n_space, n_percept), dtype=np.float32)  # Py overhead
    # The integrator below steps by `dt * (drive - bright) / tau`, and `dt/tau`
    # is the same number on every step at every location. Written that way it
    # is still a division per step, because reassociating it is not a
    # transformation a C compiler may make on its own: the two forms round
    # differently, and neither `/fp:fast` nor `-ffast-math` is on. Dividing
    # once here takes ~14 cycles of latency off the dependency chain that the
    # loop cannot start the next step without:
    dt_tau = dt / tau
    # One simulation step reads the same stimulus frame for every location, so
    # transpose once and that read becomes a contiguous run:
    stim_t = np.ascontiguousarray(np.asarray(stim).T)  # Py overhead
    n_blocks = (n_space + BLOCK - 1) // BLOCK
    # Running brightness, one row per thread. Rows are BLOCK floats apart, so
    # no two threads share a cache line.
    scratch = np.empty((n_threads, BLOCK), dtype=np.float32)  # Py overhead
    # Peak brightness since the last percept time point, laid out the same way.
    # Allocated even when it is not used, so that the loop below can be written
    # once:
    running = np.empty((n_threads, BLOCK), dtype=np.float32)  # Py overhead

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
        idx_stim = 0
        idx_frame = 0
        for idx_sim in range(n_sim):
            t_sim = idx_sim * dt
            # Since the stimulus is compressed ('sparse'), we need to access
            # the right frame. Each frame is associated with a time, `t_stim`.
            # We use that frame until `t_sim` advances past it. In other words,
            # we use the `idx_stim`-th frame for all times
            # t_stim[idx_stim] <= t_sim < t_stim[idx_stim + 1]. Which frame
            # that is does not depend on the location, so it is settled here
            # rather than inside the loop below.
            #
            # `while`, not `if`: more than one stimulus frame can fall inside a
            # single simulation step, and skipping only one of them leaves the
            # integrator reading a frame that is already in the past. Encoded
            # pulses make that the normal case rather than a corner case --
            # their edges sit on the DT=1e-3 ms grid while `dt` defaults to
            # 5e-3 ms, so a pulse edge and the sample after it routinely share
            # a step. Advancing one frame per step would let a blip that has
            # already ended drive brightness at a later instant:
            while idx_stim + 1 < n_stim and t_sim >= t_stim[idx_stim + 1]:
                idx_stim = idx_stim + 1
            for idx_space in range(hi - lo):
                amp = stim_t[idx_stim, lo + idx_space]
                bright = scratch[tid, idx_space]
                # Half-wave rectify: only cathodic (negative) current drives
                # brightness. Without this the model cannot see a
                # charge-balanced pulse at all:
                drive = -amp
                if drive < 0.0:
                    drive = 0.0
                bright = bright + dt_tau * (drive - bright)
                # Brightness is bounded in [0, \inf[
                if bright < 0.0:
                    bright = 0.0
                scratch[tid, idx_space] = bright
                # One compare per step, and it keeps the peak exact however
                # coarse the output rate is:
                if bright > running[tid, idx_space]:
                    running[tid, idx_space] = bright
            if idx_sim == idx_t_percept[idx_frame]:
                # `idx_t_percept` stores the time points at which we need to
                # output a percept. We compare `idx_sim` to `idx_t_percept`
                # rather than `t_sim` to `t_percept` because there is no good
                # (fast) way to compare two floating point numbers:
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
                idx_frame = idx_frame + 1
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

    Loop structure, sparse stimulus advancement, blocking, denormal handling
    and interval peak tracking all work exactly as in ``fading_fast``; see
    there. The only difference is that each location carries two states.

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
        float32 t_sim, amp, drive, stage1, stage1_old, bright, dt_tau
        float32[:, ::1] percept
        float32[:, ::1] stim_t
        float32[:, ::1] first
        float32[:, ::1] second
        float32[:, ::1] running
        index_t idx_space, idx_sim, idx_stim, idx_frame, idx_block
        index_t n_space, n_stim, n_percept, n_sim, n_blocks, lo, hi, tid
        unsigned long long fpmode

    n_percept = len(idx_t_percept)  # Py overhead
    n_stim = len(t_stim)  # Py overhead
    n_sim = idx_t_percept[n_percept - 1] + 1  # no negative indices
    n_space = stim.shape[0]
    if n_threads < 1:  # `num_threads(0)` is not conforming OpenMP
        n_threads = 1

    percept = np.zeros((n_space, n_percept), dtype=np.float32)  # Py overhead
    dt_tau = dt / tau
    stim_t = np.ascontiguousarray(np.asarray(stim).T)  # Py overhead
    n_blocks = (n_space + BLOCK - 1) // BLOCK
    # One row per thread for each of the two stages and for the running peak,
    # so no two threads share a cache line:
    first = np.empty((n_threads, BLOCK), dtype=np.float32)  # Py overhead
    second = np.empty((n_threads, BLOCK), dtype=np.float32)  # Py overhead
    running = np.empty((n_threads, BLOCK), dtype=np.float32)  # Py overhead

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
        idx_stim = 0
        idx_frame = 0
        for idx_sim in range(n_sim):
            t_sim = idx_sim * dt
            while idx_stim + 1 < n_stim and t_sim >= t_stim[idx_stim + 1]:
                idx_stim = idx_stim + 1
            for idx_space in range(hi - lo):
                amp = stim_t[idx_stim, lo + idx_space]
                stage1 = first[tid, idx_space]
                bright = second[tid, idx_space]
                # Half-wave rectify: only cathodic (negative) current drives
                # the cascade.
                drive = -amp
                if drive < 0.0:
                    drive = 0.0
                # The second stage reads the first stage as it was at the
                # start of the step. Feeding it the already-updated value
                # would let a drive reach the output within a single step and
                # remove the rise delay the cascade exists for:
                stage1_old = stage1
                stage1 = stage1 + dt_tau * (drive - stage1)
                bright = bright + dt_tau * (stage1_old - bright)
                # Brightness is bounded in [0, \inf[
                if bright < 0.0:
                    bright = 0.0
                first[tid, idx_space] = stage1
                second[tid, idx_space] = bright
                if bright > running[tid, idx_space]:
                    running[tid, idx_space] = bright
            if idx_sim == idx_t_percept[idx_frame]:
                for idx_space in range(hi - lo):
                    if reduce == REDUCE_PEAK:
                        bright = running[tid, idx_space]
                    else:
                        bright = second[tid, idx_space]
                    if c_abs(bright) >= thresh_percept:
                        percept[lo + idx_space, idx_frame] = bright
                    # Brightness is continuous, so the value carried across
                    # the boundary is a floor on the next interval's peak:
                    running[tid, idx_space] = second[tid, idx_space]
                idx_frame = idx_frame + 1
        c_fpmode_restore(fpmode)

    return np.asarray(percept)  # Py overhead
