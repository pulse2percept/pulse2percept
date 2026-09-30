from pulse2percept.utils._fast_math cimport c_fmax
from pulse2percept.utils._fpmode cimport c_denormals_off, c_fpmode_restore
from libc.math cimport powf as c_pow, fabs as c_abs, sqrtf as c_sqrt
from cython.parallel import prange
from cython import cdivision  # modulo, division by zero
import numpy as np
cimport numpy as cnp
cnp.import_array()

ctypedef cnp.float32_t float32
ctypedef cnp.int32_t int32
ctypedef cnp.uint32_t uint32
ctypedef Py_ssize_t index_t


@cdivision(True)
cpdef temporal_fast(const float32[:, ::1] stim,
                    const float32[::1] t_stim,
                    const uint32[::1] idx_t_percept,
                    float32 dt,
                    float32 tau1,
                    float32 tau2,
                    float32 tau3,
                    float32 eps,
                    float32 beta,
                    float32 thresh_percept,
                    uint32 n_threads):
    """Cython implementation of the Horsager 2009 temporal model

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
    tau1: float32
        Time decay constant for the fast leaky integrater (ms).
    tau2: float32
        Time decay constant for the charge accumulation (ms).
    tau3: float32
        Time decay constant for the slow leaky integrator (ms).
    eps: float32
        Scaling factor applied to charge accumulation.
    beta: float32
        Power nonlinearity (exponent of the half-wave rectification).
    thresh_percept : float32
        Spatial responses smaller than ``thresh_percept`` will be set to zero
    n_threads: uint32
        Number of CPU threads to use during parallelization using OpenMP.

    Returns
    -------
    percept : 2D float32 array
        space x time

    """
    cdef:
        float32 ca, r1, r2, r3_a, r3, r4a, r4b, r4c
        float32 t_sim, amp, zero_pow
        float32 dt_tau1, dt_tau2, dt_tau3
        float32[:, ::1] percept
        index_t idx_space, idx_sim, idx_stim, idx_frame
        index_t n_space, n_stim, n_percept, n_sim
        unsigned long long fpmode

    # Note that eps must be divided by 1000, because the original model was fit
    # with a microsecond time step and now we are running milliseconds:
    eps = eps / 1000.0

    n_percept = len(idx_t_percept)  # Py overhead
    n_stim = len(t_stim)  # Py overhead
    n_sim = idx_t_percept[n_percept - 1] + 1  # no negative indices
    n_space = stim.shape[0]

    percept = np.zeros((n_space, n_percept), dtype=np.float32)  # Py overhead
    # Power nonlinearity of a rectified-to-zero argument (>99% of steps for a
    # typical pulse train). Not simply zero: `pow(0, beta)` is 1 for
    # `beta == 0` and inf for `beta < 0`:
    zero_pow = c_pow(<float32>0.0, beta)
    # Precompute `dt / tau`: without fast-math the compiler cannot reassociate
    # the per-step divisions, which sit on the loop's dependency chain:
    dt_tau1 = dt / tau1
    dt_tau2 = dt / tau2
    dt_tau3 = dt / tau3

    for idx_space in prange(n_space, schedule='static', nogil=True, num_threads=n_threads):
        # Between pulses the integrators decay into subnormals, where arithmetic
        # is ~100x slower; see `utils/_fpmode.pxd`. The FP mode is per-thread,
        # so set it inside the `prange`:
        fpmode = c_denormals_off()
        # Because the stationary nonlinearity depends on `max_R3`, which is the
        # largest value of R3 over all time points, we have to process the
        # stimulus in two steps.
        # Step 1: Calculate `r3` for all time points and extract `max_r3`:
        ca = 0.0
        r1 = 0.0
        r2 = 0.0
        r4a = 0.0
        r4b = 0.0
        r4c = 0.0
        idx_stim = 0
        idx_frame = 0
        for idx_sim in range(n_sim):
            t_sim = idx_sim * dt
            # Since the stimulus is compressed ('sparse'), we need to access
            # the right frame. Each frame is associated with a time, `t_stim`.
            # We use that frame until `t_sim` advances past it. In other words,
            # we use the `idx_stim`-th frame for all times
            # t_stim[idx_stim] <= t_sim < t_stim[idx_stim + 1].
            # `while`, not `if`: encoded pulse edges lie on the DT=1e-3 ms grid,
            # finer than `dt`, so several frames can fall inside one step:
            while idx_stim + 1 < n_stim and t_sim >= t_stim[idx_stim + 1]:
                idx_stim = idx_stim + 1
            amp = stim[idx_space, idx_stim]
            # Fast ganglion cell response. Note the negative sign before `amp`,
            # which is required to reproduce e.g. Fig.3 in the paper,
            # indicating that the model was trained on what we now call
            # "anodic" current:
            r1 = r1 + dt_tau1 * (-amp - r1)  # += in threads is a reduction
            # Charge accumulation:
            # ca = ca + dt * c_fmax(amp, 0) # SLOW
            ca = ca + dt * (amp if amp > 0.0 else 0.0)
            r2 = r2 + dt_tau2 * (ca - r2)
            # Half-rectification and power nonlinearity:
            # r3 = c_pow(c_fmax(r1 - eps * r2, 0), beta) # SLOW
            # `powf` is the costliest call; reuse `zero_pow` when the
            # argument is rectified to zero:
            r3_a = r1 - eps * r2
            if r3_a > 0.0:
                r3 = c_pow(r3_a, beta)
            else:
                r3 = zero_pow
            # Slow response (3-stage leaky integrator):
            r4a = r4a + dt_tau3 * (r3 - r4a)
            r4b = r4b + dt_tau3 * (r4a - r4b)
            r4c = r4c + dt_tau3 * (r4b - r4c)
            if idx_sim == idx_t_percept[idx_frame]:
                # `idx_t_percept` stores the time points at which we need to
                # output a percept. We compare `idx_sim` to `idx_t_percept`
                # rather than `t_sim` to `t_percept` because there is no good
                # (fast) way to compare two floating point numbers:
                if c_abs(r4c) >= thresh_percept:
                    percept[idx_space, idx_frame] = r4c
                idx_frame = idx_frame + 1
        # Restore the thread's floating-point mode:
        c_fpmode_restore(fpmode)

    return np.asarray(percept)  # Py overhead
