"""Gaussian current-spread kernels shared by the retinal and cortical
scoreboard models."""
from libc.math cimport(expf as c_exp, fabsf as c_abs, isnan as c_isnan)
from cython.parallel import prange
from cython import cdivision
import numpy as np
cimport numpy as cnp
cnp.import_array()

ctypedef cnp.float32_t float32
ctypedef cnp.uint32_t uint32
ctypedef Py_ssize_t index_t


cdef cnp.uint8_t[::1] _active_electrodes(const float32[:, ::1] stim):
    """Flag the electrodes that carry a nonzero amplitude at any time point.

    Spatial kernels loop over electrodes outside the time loop, so they can
    only skip electrodes that are zero for the whole stimulus (most of them
    for a sparse stimulus).
    """
    cdef:
        index_t idx_el, idx_time
        index_t n_el = stim.shape[0]
        index_t n_time = stim.shape[1]
        cnp.uint8_t[::1] active = np.zeros(n_el, dtype=np.uint8)

    with nogil:
        for idx_el in range(n_el):
            for idx_time in range(n_time):
                if c_abs(stim[idx_el, idx_time]) > 0:
                    active[idx_el] = 1
                    break
    return active


@cdivision(True)
cpdef fast_scoreboard(const float32[:, ::1] stim,
                      const float32[::1] xel,
                      const float32[::1] yel,
                      const float32[::1] xgrid,
                      const float32[::1] ygrid,
                      float32 rho,
                      float32 thresh_percept,
                      float32 cutoff_r2,
                      uint32 separate,
                      float32 offset,
                      uint32 n_threads):
    """Fast spatial response of the scoreboard model

    The Gaussian spread is time-independent, so it is computed once per
    (grid point, electrode) pair. The innermost loop runs over time (the
    contiguous axis of ``stim`` and the output) with independent iterations,
    so it vectorizes without relaxed floating-point semantics.

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
    cutoff_r2 : float32
        Squared distance (microns^2) beyond which an electrode is treated as
        contributing nothing to a grid point. Pass ``inf`` to sum over every
        electrode. See ``min_current_spread`` on the model for how this is
        derived.
    separate: uint32 :
        If nonzero, then points on different side of x=offset than the electrode
        will not contribute to the percept (used for cortical models)
    offset : float32
         Boundary for separation
    n_threads: uint32
        Number of CPU threads to use during parallelization using OpenMP.
    """
    cdef:
        index_t idx_el, idx_time, idx_space
        index_t n_el, n_time, n_space
        float32[:, ::1] bright
        float32 xdiff, ydiff, r2, gauss
        cnp.uint8_t[::1] active

    n_el = stim.shape[0]
    n_time = stim.shape[1]
    n_space = len(xgrid)
    if n_threads < 1:  # `num_threads(0)` is not conforming OpenMP
        n_threads = 1

    bright = np.zeros((n_space, n_time), dtype=np.float32)  # Py overhead
    active = _active_electrodes(stim)  # Py overhead

    # Parallel loop over all pixels to be rendered:
    for idx_space in prange(n_space, schedule='guided', nogil=True,
                            num_threads=n_threads):
        if c_isnan(xgrid[idx_space]) or c_isnan(ygrid[idx_space]):
            continue
        for idx_el in range(n_el):
            if active[idx_el] == 0:
                continue
            if separate != 0:
                if ((xel[idx_el] < offset) != (xgrid[idx_space] < offset)):
                    continue
            xdiff = xgrid[idx_space] - xel[idx_el]
            ydiff = ygrid[idx_space] - yel[idx_el]
            r2 = xdiff * xdiff + ydiff * ydiff
            if r2 > cutoff_r2:
                continue
            gauss = c_exp(-r2 / (<float32>2.0 * rho * rho))
            for idx_time in range(n_time):
                bright[idx_space, idx_time] = (bright[idx_space, idx_time] +
                                               gauss * stim[idx_el, idx_time])
        for idx_time in range(n_time):
            if c_abs(bright[idx_space, idx_time]) < thresh_percept:
                bright[idx_space, idx_time] = <float32>0.0
    return np.asarray(bright)  # Py overhead


@cdivision(True)
cpdef fast_scoreboard_3d(const float32[:, ::1] stim,
                      const float32[::1] xel,
                      const float32[::1] yel,
                      const float32[::1] zel,
                      const float32[::1] xgrid,
                      const float32[::1] ygrid,
                      const float32[::1] zgrid,
                      float32 rho,
                      float32 thresh_percept,
                      float32 cutoff_r2,
                      uint32 separate,
                      float32 offset,
                      uint32 n_threads):
    """Fast spatial response of the scoreboard model

    3D counterpart of :func:`fast_scoreboard`.

    Parameters
    ----------
    stim : 2D float32 array
        A ``Stimulus.data`` container that contains electrodes as rows and
        time points as columns. The spatial response will be calculated for
        each column independently.
    xel, yel, zel : 1D float32 array
        An array of x, y, or z coordinates for each electrode (microns)
    xgrid, ygrid, zgrid : 1D float32 array
        An array of x, y, or z coordinates at which to calculate the spatial
        response (microns)
    rho : float32
        The rho parameter of the scoreboard model (microns): exponential decay
        constant for the current spread
    thresh_percept : float32
        Spatial responses smaller than ``thresh_percept`` will be set to zero
    cutoff_r2 : float32
        Squared distance (microns^2) beyond which an electrode is treated as
        contributing nothing to a grid point. Pass ``inf`` to sum over every
        electrode. See ``min_current_spread`` on the model for how this is
        derived.
    separate: uint32 :
        If nonzero, then points on different side of x=offset than the electrode
        will not contribute to the percept (used for cortical models)
    offset : float32
         Boundary for separation
    n_threads: uint32
        Number of CPU threads to use during parallelization using OpenMP.
    """
    cdef:
        index_t idx_el, idx_time, idx_space
        index_t n_el, n_time, n_space
        float32[:, ::1] bright
        float32 xdiff, ydiff, zdiff, r2, gauss
        cnp.uint8_t[::1] active

    n_el = stim.shape[0]
    n_time = stim.shape[1]
    n_space = len(xgrid)
    if n_threads < 1:  # `num_threads(0)` is not conforming OpenMP
        n_threads = 1

    bright = np.zeros((n_space, n_time), dtype=np.float32)  # Py overhead
    active = _active_electrodes(stim)  # Py overhead

    # Parallel loop over all pixels to be rendered:
    for idx_space in prange(n_space, schedule='guided', nogil=True,
                            num_threads=n_threads):
        if c_isnan(xgrid[idx_space]) or c_isnan(ygrid[idx_space]):
            continue
        for idx_el in range(n_el):
            if active[idx_el] == 0:
                continue
            if separate != 0:
                if ((xel[idx_el] < offset) != (xgrid[idx_space] < offset)):
                    continue
            xdiff = xgrid[idx_space] - xel[idx_el]
            ydiff = ygrid[idx_space] - yel[idx_el]
            zdiff = zgrid[idx_space] - zel[idx_el]
            r2 = xdiff * xdiff + ydiff * ydiff + zdiff * zdiff
            if r2 > cutoff_r2:
                continue
            gauss = c_exp(-r2 / (<float32>2.0 * rho * rho))
            for idx_time in range(n_time):
                bright[idx_space, idx_time] = (bright[idx_space, idx_time] +
                                               gauss * stim[idx_el, idx_time])
        for idx_time in range(n_time):
            if c_abs(bright[idx_space, idx_time]) < thresh_percept:
                bright[idx_space, idx_time] = <float32>0.0
    return np.asarray(bright)  # Py overhead
