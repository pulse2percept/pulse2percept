import numpy as np
cimport numpy as cnp
from cython import cdivision  # for modulo operator
from cython.parallel import prange
from libc.math cimport(pow as c_pow, exp as c_exp)

ctypedef cnp.float32_t float32


@cdivision(True)
cpdef scoreboard_fast(const float32[:, ::1] stim,
                      const float32[::1] xel,
                      const float32[::1] yel,
                      const float32[::1] xgrid,
                      const float32[::1] ygrid,
                      float32 rho,
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
        size_t idx_el, idx_time, idx_space, n_el, n_time, n_space
        size_t idx_bright, n_bright
        float32[::1] bright
        float32 px_bright, dist2, gauss

    n_el = stim.shape[0]
    n_time = stim.shape[1]
    n_space = len(xgrid)
    n_bright = n_time * n_space

    # A flattened array containing n_time x n_space entries:
    bright = np.empty(n_bright, dtype=np.float32)  # Py overhead

    for idx_bright in prange(n_bright, nogil=True):
        # For each entry in the output matrix:
        idx_space = idx_bright % n_space
        idx_time = idx_bright / n_space

        px_bright = 0.0
        for idx_el in range(n_el):
            dist2 = (c_pow(xgrid[idx_space] - xel[idx_el], 2) +
                     c_pow(ygrid[idx_space] - yel[idx_el], 2))
            gauss = c_exp(-dist2 / (2.0 * rho * rho))
            px_bright = px_bright + stim[idx_el, idx_time] * gauss
        if px_bright < thresh_percept:
            px_bright = 0.0
        bright[idx_bright] = px_bright  # Py overhead
    return np.asarray(bright)  # Py overhead
