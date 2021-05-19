from libc.math cimport(pow as c_pow, exp as c_exp, fabs as c_abs,
                       sqrt as c_sqrt)
from cython.parallel import prange
from cython import cdivision  # modulo, division by zero
import numpy as np
cimport numpy as cnp

ctypedef cnp.float32_t float32
ctypedef cnp.int32_t int32
ctypedef cnp.uint32_t uint32


@cdivision(True)
cpdef fading_fast(const float32[:, ::1] stim,
                  const float32[::1] t_stim,
                  const uint32[::1] idx_t_percept,
                  float32 dt,
                  float32 tau,
                  float32 thresh_percept):
    """Cython implementation of the generic fading model

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

    Returns
    -------
    percept : 2D float32 array
        space x time

    """
    cdef:
        float32 t_sim, amp, bright
        float32[:, ::1] percept
        int32 idx_space, idx_sim, idx_stim, idx_frame
        int32 n_space, n_stim, n_percept, n_sim

    n_percept = len(idx_t_percept)  # Py overhead
    n_stim = len(t_stim)  # Py overhead
    n_sim = idx_t_percept[n_percept - 1] + 1  # no negative indices
    n_space = stim.shape[0]

    percept = np.zeros((n_space, n_percept), dtype=np.float32)  # Py overhead

    for idx_space in prange(n_space, schedule='static', nogil=True):
        bright = 0.0
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
            # Invert stimulus polarity and apply leaky integrator:
            bright = bright + dt * (-amp - bright) / tau
            # Brightness is bounded in [0, \inf[
            if bright < 0.0:
                bright = 0.0
            if idx_sim == idx_t_percept[idx_frame]:
                # `idx_t_percept` stores the time points at which we need to
                # output a percept. We compare `idx_sim` to `idx_t_percept`
                # rather than `t_sim` to `t_percept` because there is no good
                # (fast) way to compare two floating point numbers:
                if c_abs(bright) >= thresh_percept:
                    percept[idx_space, idx_frame] = bright
                idx_frame = idx_frame + 1

    return np.asarray(percept)  # Py overhead
