from libc.math cimport(pow as c_pow, exp as c_exp, tanh as c_tanh,
                       sin as c_sin, cos as c_cos, fabs as c_abs,
                       isnan as c_isnan)
from cython.parallel import prange
from cython import cdivision  # for modulo operator
import numpy as np
cimport numpy as cnp


ctypedef cnp.float32_t float32
ctypedef cnp.uint32_t uint32
ctypedef cnp.int32_t int32
ctypedef cnp.uint8_t uint8


@cdivision(True)
cpdef fast_thompson2003(const float32[:, ::1] stim,
                        const float32[::1] xel,
                        const float32[::1] yel,
                        const float32[::1] xgrid,
                        const float32[::1] ygrid,
                        const uint8[:, ::1] dropout,
                        float32 radius,
                        float32 thresh_percept):
    """Fast spatial response of the scoreboard model

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
        int32 idx_el, idx_time, idx_space, idx_bright
        int32 n_el, n_time, n_space, n_bright
        float32[:, ::1] bright
        float32 px_bright, dist2, gauss, amp

    n_el = stim.shape[0]
    n_time = stim.shape[1]
    n_space = len(xgrid)
    n_bright = n_time * n_space

    # A flattened array containing n_time x n_space entries:
    bright = np.empty((n_space, n_time), dtype=np.float32)  # Py overhead

    for idx_bright in prange(n_bright, schedule='static', nogil=True):
        # For each entry in the output matrix:
        idx_space = idx_bright % n_space
        idx_time = idx_bright / n_space

        if c_isnan(xgrid[idx_space]) or c_isnan(ygrid[idx_space]):
            bright[idx_space, idx_time] = 0.0
            continue

        px_bright = 0.0
        for idx_el in range(n_el):
            if dropout[idx_el, idx_time]:
                continue
            amp = stim[idx_el, idx_time]
            if c_abs(amp) > 0:
                dist2 = (c_pow(xgrid[idx_space] - xel[idx_el], 2) +
                         c_pow(ygrid[idx_space] - yel[idx_el], 2))
                if dist2 < radius * radius:
                    px_bright = px_bright + amp
        if c_abs(px_bright) < thresh_percept:
            px_bright = 0.0
        bright[idx_space, idx_time] = px_bright  # Py overhead
    return np.asarray(bright)  # Py overhead
