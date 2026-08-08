from libc.math cimport(pow as c_pow, exp as c_exp, fabs as c_abs,
                       sqrt as c_sqrt)
from cython.parallel import prange
from cython.parallel cimport threadid
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


@cdivision(True)
cpdef fading_fast(const float32[:, ::1] stim,
                  const float32[::1] t_stim,
                  const uint32[::1] idx_t_percept,
                  float32 dt,
                  float32 tau,
                  float32 thresh_percept,
                  uint32 n_threads):
    """Cython implementation of the generic fading model

    The leaky integrator has to be stepped in order, so the loop over time is
    serial -- but each spatial location integrates independently of every
    other. Time is therefore the *outer* loop and space the inner one, which
    leaves the inner loop free of any carried dependency and lets it
    vectorize. Threads take a block of locations each and run the whole time
    loop over it, so the parallel region is still entered only once.

    The arithmetic per step is unchanged, and each location still sees its
    steps in the same order, so the result is bit-for-bit what the
    space-outer version produced.

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
        Number of CPU threads to use during parallelization using OpenMP. Defaults to maximum number of cores on user CPU

    Returns
    -------
    percept : 2D float32 array
        space x time

    """
    cdef:
        float32 t_sim, amp, bright
        float32[:, ::1] percept
        float32[:, ::1] stim_t
        float32[:, ::1] scratch
        index_t idx_space, idx_sim, idx_stim, idx_frame, idx_block
        index_t n_space, n_stim, n_percept, n_sim, n_blocks, lo, hi, tid

    n_percept = len(idx_t_percept)  # Py overhead
    n_stim = len(t_stim)  # Py overhead
    n_sim = idx_t_percept[n_percept - 1] + 1  # no negative indices
    n_space = stim.shape[0]
    if n_threads < 1:  # `num_threads(0)` is not conforming OpenMP
        n_threads = 1

    percept = np.zeros((n_space, n_percept), dtype=np.float32)  # Py overhead
    # One simulation step reads the same stimulus frame for every location, so
    # transpose once and that read becomes a contiguous run:
    stim_t = np.ascontiguousarray(np.asarray(stim).T)  # Py overhead
    n_blocks = (n_space + BLOCK - 1) // BLOCK
    # Running brightness, one row per thread. Rows are BLOCK floats apart, so
    # no two threads share a cache line.
    scratch = np.empty((n_threads, BLOCK), dtype=np.float32)  # Py overhead

    for idx_block in prange(n_blocks, schedule='static', nogil=True,
                            num_threads=n_threads):
        tid = threadid()
        lo = idx_block * BLOCK
        hi = lo + BLOCK
        if hi > n_space:
            hi = n_space
        for idx_space in range(hi - lo):
            scratch[tid, idx_space] = <float32>0.0
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
            # rather than inside the loop below:
            if idx_stim + 1 < n_stim:
                if t_sim >= t_stim[idx_stim + 1]:
                    idx_stim = idx_stim + 1
            for idx_space in range(hi - lo):
                amp = stim_t[idx_stim, lo + idx_space]
                bright = scratch[tid, idx_space]
                # Invert stimulus polarity and apply leaky integrator:
                bright = bright + dt * (-amp - bright) / tau
                # Brightness is bounded in [0, \inf[
                if bright < 0.0:
                    bright = 0.0
                scratch[tid, idx_space] = bright
            if idx_sim == idx_t_percept[idx_frame]:
                # `idx_t_percept` stores the time points at which we need to
                # output a percept. We compare `idx_sim` to `idx_t_percept`
                # rather than `t_sim` to `t_percept` because there is no good
                # (fast) way to compare two floating point numbers:
                for idx_space in range(hi - lo):
                    bright = scratch[tid, idx_space]
                    if c_abs(bright) >= thresh_percept:
                        percept[lo + idx_space, idx_frame] = bright
                idx_frame = idx_frame + 1

    return np.asarray(percept)  # Py overhead
