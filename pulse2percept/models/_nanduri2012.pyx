import numpy as np
cimport numpy as cnp
from cython import cdivision  # modulo, division by zero
from cython.parallel import prange, parallel
from libc.math cimport(pow as c_pow, exp as c_exp, fabs as c_abs,
                       sqrt as c_sqrt)

ctypedef cnp.float32_t float32
ctypedef cnp.int32_t int32

@cdivision(True)
cdef inline float32 expit(float32 x) nogil:
    return 1.0 / (1.0 + c_exp(-x))

cdef inline float32 float_max(float32 a, float32 b) nogil:
    return a if a >= b else b


@cdivision(True)
cpdef spatial_fast(const float32[:, ::1] stim,
                   const float32[::1] xel,
                   const float32[::1] yel,
                   const float32[::1] zel,
                   const float32[::1] rel,
                   const float32[::1] xgrid,
                   const float32[::1] ygrid,
                   float32 atten_a,
                   float32 atten_n,
                   float32 thresh_percept):
    """Fast spatial response of the Nanduri et al. (2012) model

    Parameters
    ----------
    stim : 2D float32 array
        A ``Stimulus.data`` container that contains electrodes as rows and
        time points as columns. The spatial response will be calculated for
        each column independently.
    xel, yel : 1D float32 array
        An array of x or y coordinates for each electrode (microns)
    xgrid, ygrid : 1D float32 array
        An array of x or y coordinates at which to calculate the spatial
        response (microns)
    rho : float32
        The rho parameter of the scoreboard model (microns): exponential decay
        constant for the current spread
    thresh_percept : float32
        Spatial responses smaller than ``thresh_percept`` will be set to zero

    """
    cdef:
        size_t idx_el, idx_time, idx_space, n_el, n_time, n_space
        size_t idx_bright, n_bright
        float32[:, ::1] bright
        float32 px_bright, denom, d2c, d2e, amp

    n_el = stim.shape[0]
    n_time = stim.shape[1]
    n_space = len(xgrid)
    n_bright = n_time * n_space

    # A flattened array containing n_time x n_space entries:
    bright = np.empty((n_time, n_space), dtype=np.float32)  # Py overhead

    for idx_bright in prange(n_bright, schedule='dynamic', nogil=True):
        # For each entry in the output matrix:
        idx_space = idx_bright % n_space
        idx_time = idx_bright / n_space

        px_bright = 0.0
        for idx_el in range(n_el):
            amp = stim[idx_el, idx_time]
            if c_abs(amp) > 0:
                d2c = (c_pow(xgrid[idx_space] - xel[idx_el], 2) +
                       c_pow(ygrid[idx_space] - yel[idx_el], 2))
                if d2c < c_pow(rel[idx_el], 2):
                    denom = atten_a + c_pow(zel[idx_el], atten_n)
                    px_bright = px_bright + amp * atten_a / denom
                else:
                    d2e = (c_pow(c_sqrt(d2c) - rel[idx_el], 2) +
                           c_pow(zel[idx_el], 2))
                    denom = atten_a + c_pow(c_sqrt(d2e), atten_n)
                    px_bright = px_bright + atten_a / denom
        if c_abs(px_bright) < thresh_percept:
            px_bright = 0.0
        bright[idx_time, idx_space] = px_bright  # Py overhead
    return np.asarray(bright)  # Py overhead


@cdivision(True)
cpdef temporal_fast(const float32[:, ::1] stim,
                    const float32[::1] t_stim,
                    const size_t[::1] idx_t_percept,
                    float32 dt,
                    float32 tau1,
                    float32 tau2,
                    float32 tau3,
                    float32 asymptote,
                    float32 shift,
                    float32 slope,
                    float32 eps):
    """Cython implementation of the Nanduri 2012 temporal model"""
    cdef:
        float32 ca, r1, r2, r3, max_r3, r4a, r4b, r4c
        float32 t_sim, amp, scale
        float32[::1] all_r3
        float32[:, ::1] percept
        size_t idx_space, idx_sim, idx_stim, idx_frame
        size_t n_space, n_stim, n_percept
        int32 n_sim

    n_percept = len(idx_t_percept)  # Py overhead
    n_stim = len(t_stim)  # Py overhead
    n_sim = idx_t_percept[n_percept - 1] + 1  # no negative indices
    n_space = stim.shape[1]

    all_r3 = np.empty(n_sim, dtype=np.float32)  # Py overhead
    percept = np.empty((n_percept, n_space), dtype=np.float32)  # Py overhead

    for idx_space in prange(n_space, schedule='dynamic', nogil=True):
        # Because the stationary nonlinearity depends on `max_R3`, which is the
        # largest value of R3 over all time points, we have to process the
        # stimulus in two steps.
        # Step 1: Calculate `r3` for all time points and extract `max_r3`:
        ca = 0.0
        r1 = 0.0
        r2 = 0.0
        idx_stim = 0
        max_r3 = 1e-37
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
            amp = stim[idx_stim, idx_space]
            # Fast ganglion cell response:
            r1 = r1 + dt * (-amp - r1) / tau1  # += in threads is a reduction
            # Charge accumulation:
            ca = ca + dt * float_max(amp, 0)
            r2 = r2 + dt * (ca - r2) / tau2
            # Half-rectification:
            r3 = float_max(r1 - eps * r2, 0)
            # Store `r3` for Step 2:
            all_r3[idx_sim] = r3
            # Find the largest `r3` across time:
            if r3 > max_r3:
                max_r3 = r3

        # Step 2: Use `max_R3` to calculate the slow response:
        r4a = 0.0
        r4b = 0.0
        r4c = 0.0
        idx_stim = 0
        idx_frame = 0
        scale = asymptote * expit((max_r3 - shift) / slope) / max_r3
        for idx_sim in range(n_sim):
            t_sim = idx_sim * dt
            # Access the right stimulus frame (same as above):
            if idx_stim + 1 < n_stim:
                if t_sim >= t_stim[idx_stim + 1]:
                    idx_stim = idx_stim + 1
            # Slow response:
            r4a = r4a + dt * (all_r3[idx_sim] * scale - r4a) / tau3
            r4b = r4b + dt * (r4a - r4b) / tau3
            r4c = r4c + dt * (r4b - r4c) / tau3
            if idx_sim == idx_t_percept[idx_frame]:
                # `idx_t_percept` stores the time points at which we need to
                # output a percept. We compare `idx_sim` to `idx_t_percept`
                # rather than `t_sim` to `t_percept` because there is no good
                # (fast) way to compare two floating point numbers:
                percept[idx_frame, idx_space] = r4c
                idx_frame = idx_frame + 1

    return np.asarray(percept)  # Py overhead

