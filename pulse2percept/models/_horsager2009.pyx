import numpy as np
cimport numpy as cnp
from cython import cdivision  # modulo, division by zero
from cython.parallel import prange, parallel
from libc.math cimport(pow as c_pow, exp as c_exp, fabs as c_abs,
                       sqrt as c_sqrt)

ctypedef cnp.float32_t float32
ctypedef cnp.int32_t int32
ctypedef cnp.uint32_t uint32

@cdivision(True)
cdef inline float32 expit(float32 x) nogil:
    return 1.0 / (1.0 + c_exp(-x))

cdef inline float32 float_max(float32 a, float32 b) nogil:
    return a if a >= b else b


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
                    float32 thresh_percept):
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

    Returns
    -------
    percept : 2D float32 array
        space x time

    """
    cdef:
        float32 ca, r1, r2, r3, r4a, r4b, r4c
        float32 t_sim, amp
        float32[:, ::1] percept
        uint32 idx_space, idx_sim, idx_stim, idx_frame
        uint32 n_space, n_stim, n_percept, n_sim

    # Note that eps must be divided by 1000, because the original model was fit
    # with a microsecond time step and now we are running milliseconds:
    eps = eps / 1000.0

    n_percept = len(idx_t_percept)  # Py overhead
    n_stim = len(t_stim)  # Py overhead
    n_sim = idx_t_percept[n_percept - 1] + 1  # no negative indices
    n_space = stim.shape[0]

    percept = np.zeros((n_space, n_percept), dtype=np.float32)  # Py overhead

    for idx_space in prange(n_space, schedule='dynamic', nogil=True):
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
            # t_stim[idx_stim] <= t_sim < t_stim[idx_stim + 1]:
            if idx_stim + 1 < n_stim:
                if t_sim >= t_stim[idx_stim + 1]:
                    idx_stim = idx_stim + 1
            amp = stim[idx_space, idx_stim]
            # Fast ganglion cell response. Note the negative sign before `amp`,
            # which is required to reproduce e.g. Fig.3 in the paper,
            # indicating that the model was trained on what we know call
            # "anodic" current:
            r1 = r1 + dt * (-amp - r1) / tau1  # += in threads is a reduction
            # Charge accumulation:
            ca = ca + dt * float_max(amp, 0)
            r2 = r2 + dt * (ca - r2) / tau2
            # Half-rectification and power nonlinearity:
            r3 = c_pow(float_max(r1 - eps * r2, 0), beta)
            # Slow response (3-stage leaky integrator):
            r4a = r4a + dt * (r3 - r4a) / tau3
            r4b = r4b + dt * (r4a - r4b) / tau3
            r4c = r4c + dt * (r4b - r4c) / tau3
            if idx_sim == idx_t_percept[idx_frame]:
                # `idx_t_percept` stores the time points at which we need to
                # output a percept. We compare `idx_sim` to `idx_t_percept`
                # rather than `t_sim` to `t_percept` because there is no good
                # (fast) way to compare two floating point numbers:
                if c_abs(r4c) >= thresh_percept:
                    percept[idx_space, idx_frame] = r4c
                idx_frame = idx_frame + 1

    return np.asarray(percept)  # Py overhead

