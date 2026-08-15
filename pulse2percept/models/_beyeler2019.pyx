from libc.math cimport(powf as c_pow, expf as c_exp, tanhf as c_tanh,
                       sinf as c_sin, cosf as c_cos, fabsf as c_abs,
                       isnan as c_isnan)
from cython.parallel import prange
from cython.parallel cimport threadid
from cython import cdivision  # for modulo operator
import numpy as np
cimport numpy as cnp
cnp.import_array()

ctypedef cnp.float32_t float32
ctypedef cnp.uint32_t uint32
ctypedef cnp.int32_t int32
ctypedef Py_ssize_t index_t

cdef float32 deg2rad = <float32>(3.14159265358979323846 / 180.0)


cdef cnp.uint8_t[::1] _active_electrodes(const float32[:, ::1] stim):
    """Flag the electrodes that carry a nonzero amplitude at any time point.

    The spatial kernels loop over electrodes *outside* the loop over time, so
    they cannot skip an electrode that happens to be zero at one time point
    the way a time-innermost loop could. Electrodes that are zero for the
    whole stimulus can still be skipped, and for a sparse stimulus that is
    most of them -- hence this one-off pass.
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

    The Gaussian current spread of an electrode at a grid point depends only
    on the two of them, not on time, so it is computed once per
    (grid point, electrode) pair and then applied to every time point. The
    innermost loop is over time, which is the contiguous axis of both ``stim``
    and the output, and whose iterations are independent -- so it vectorizes
    without needing relaxed floating-point semantics.

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

    The three-dimensional counterpart of :func:`fast_scoreboard`; see there
    for why the loop nest is ordered the way it is.

    Parameters
    ----------
    stim : 2D float32 array
        A ``Stimulus.data`` container that contains electrodes as rows and
        time points as columns. The spatial response will be calculated for
        each column independently.
    xel, yel, zel : 1D float32 array
        An array of x or y coordinates for each electrode (microns)
    xgrid, ygrid, zgrid : 1D float32 array
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



cpdef fast_jansonius(float32[::1] rho, float32 phi0, float32 beta_s,
                     float32 beta_i):
    cdef:
        float32[::1] xprime, yprime
        float32 b, c, rho_min, tmp_phi, tmp_rho
        index_t idx

    if phi0 > 0:
        # Axon is in superior retina, compute `b` (real number) from Eq. 5:
        b = c_exp(beta_s + <float32>3.9 * c_tanh(-(phi0 - <float32>121.0) / <float32>14.0))
        # Equation 3, `c` a positive real number:
        c = <float32>1.9 + <float32>1.4 * c_tanh((phi0 - <float32>121.0) / <float32>14.0)
    else:
        # Axon is in inferior retina: compute `b` (real number) from Eq. 6:
        b = -c_exp(beta_i + <float32>1.5 * c_tanh(-(-phi0 - <float32>90.0) / <float32>25.0))
        # Equation 4, `c` a positive real number:
        c = <float32>1.0 + <float32>0.5 * c_tanh((-phi0 - <float32>90.0) / <float32>25.0)

    xprime = np.empty_like(rho)
    yprime = np.empty_like(rho)
    rho_min = np.min(rho)
    with nogil:
        for idx in range(len(rho)):
            tmp_rho = rho[idx]
            tmp_phi = phi0 + b * c_pow(tmp_rho - rho_min, c)
            xprime[idx] = tmp_rho * c_cos(deg2rad * tmp_phi)
            yprime[idx] = tmp_rho * c_sin(deg2rad * tmp_phi)
    return np.asarray(xprime), np.asarray(yprime)


cdef index_t argmin_segment(float32[:, :] flat_bundles, float32 x, float32 y):
    cdef:
        float32 dist2, min_dist2, xdiff, ydiff
        index_t seg, n_seg
        index_t min_seg

    min_dist2 = <float32>1e12
    n_seg = flat_bundles.shape[0]
    for seg in range(n_seg):
        xdiff = flat_bundles[seg, 0] - x
        ydiff = flat_bundles[seg, 1] - y
        dist2 = xdiff * xdiff + ydiff * ydiff
        if dist2 < min_dist2:
            min_dist2 = dist2
            min_seg = seg
    return min_seg


cpdef fast_find_closest_axon(float32[:, :] flat_bundles,
                             float32[::1] xret,
                             float32[::1] yret):
    cdef:
        index_t[::1] closest_seg
        index_t n_xy, n_seg
        index_t pos
    closest_seg = np.empty(len(xret), dtype=np.uint32)
    n_xy = len(xret)
    n_seg = flat_bundles.shape[0]
    for pos in range(n_xy):
        closest_seg[pos] = argmin_segment(flat_bundles, xret[pos], yret[pos])
    return np.asarray(closest_seg)


@cdivision(True)
cpdef fast_axon_map(const float32[:, ::1] stim,
                    const float32[::1] xel,
                    const float32[::1] yel,
                    const float32[:, ::1] axon_segments,
                    const uint32[::1] idx_start,
                    const uint32[::1] idx_end,
                    float32 rho,
                    float32 thresh_percept,
                    float32 cutoff_r2,
                    uint32 n_threads):
    """Fast spatial response of the axon map model

    The Gaussian falloff from an electrode to an axon segment depends only on
    the two of them, not on time. The loop nest is therefore ordered
    pixel -> segment -> electrode -> time, so that the ``exp`` is evaluated
    once per (segment, electrode) pair and reused across every time point.
    A time-innermost ordering would evaluate it ``n_time`` times over.

    The innermost loop over time accumulates into independent slots of a
    scratch buffer, so it vectorizes without relaxed floating-point
    semantics. Nothing of size ``n_segments x n_electrodes`` or
    ``n_segments x n_time`` is ever materialized: each thread holds two
    buffers of ``n_time`` floats, and ``stim`` is small enough to stay in
    cache while every pixel streams past it.

    Parameters
    ----------
    stim : 2D float32 array
        A ``Stimulus.data`` container that contains electrodes as rows and
        time points as columns. The spatial response will be calculated for
        each column independently.
    xel, yel : 1D float32 array
        An array of x or y coordinates for each electrode (microns)
    axon_segments : 2D float32 array
        All axon segments concatenated into an Nx3 array.
        Each row has the x/y coordinate of a segment along with its
        contribution to a given pixel.
        ``idx_start`` and ``idx_end`` are used to slice the ``axon`` array.
        For example, the axon belonging to the i-th pixel has segments
        axon[idx_start[i]:idx_end[i]].
        This arrangement is necessary in order to access ``axon`` in parallel.
    idx_start, idx_end : 1D uint32 array
        Start and stop indices of the i-th axon.
    rho : float32
        The rho parameter of the axon map model: exponential decay constant
        (microns) away from the axon.
        Note that lambda was already taken into account when calculating the
        axon contribution (stored/passed in ``axon``).
    thresh_percept : float32
        Spatial responses smaller than ``thresh_percept`` will be set to zero
    cutoff_r2 : float32
        Squared distance (microns^2) beyond which an electrode is treated as
        contributing nothing to an axon segment. Pass ``inf`` to sum over
        every electrode. See ``min_current_spread`` on the model for how this
        is derived.
    n_threads: uint32
        Number of CPU threads to use during parallelization using OpenMP.

    """
    cdef:
        index_t idx_el, idx_time, idx_space, idx_ax, tid, row
        index_t n_el, n_time, n_space, stride
        float32[:, ::1] bright
        float32[:, ::1] scratch
        float32 xdiff, ydiff, r2, gauss, sens, ax_x, ax_y, sgm
        cnp.uint8_t[::1] active

    n_el = stim.shape[0]
    n_time = stim.shape[1]
    n_space = len(idx_start)
    # `num_threads(0)` is not conforming OpenMP, and the scratch buffer below
    # is sized on the assumption that no thread id can reach `n_threads`:
    if n_threads < 1:
        n_threads = 1

    # A flattened array containing n_space x n_time entries:
    bright = np.empty((n_space, n_time), dtype=np.float32)  # Py overhead
    active = _active_electrodes(stim)  # Py overhead

    # Per-thread scratch: row `tid` holds this thread's running per-time-point
    # segment brightness (first `n_time` entries) and pixel brightness (next
    # `n_time`). OpenMP may give the team fewer threads than requested but
    # never more, so `n_threads` rows always suffice. Rows are padded to a
    # 64-byte boundary so that two threads never share a cache line, which for
    # a single-frame stimulus they otherwise would. This is the only extra
    # memory the kernel needs, and allocating it through NumPy means a failure
    # raises MemoryError here rather than inside a nogil block.
    stride = ((2 * n_time + 15) // 16) * 16
    scratch = np.empty((n_threads, stride), dtype=np.float32)

    # Parallel loop over all pixels to be rendered. `guided` rather than
    # `static`: axons differ several-fold in how many segments they have, and
    # how many electrodes fall inside `cutoff_r2` varies with where the pixel
    # sits relative to the array, so equal-sized chunks are not equal-sized
    # work.
    for idx_space in prange(n_space, schedule='guided', nogil=True,
                            num_threads=n_threads):
        tid = threadid()
        # Brightness of this pixel over time, built up by taking the strongest
        # activated axon segment at each time point:
        for idx_time in range(n_time):
            scratch[tid, n_time + idx_time] = <float32>0.0
        # `idx_start` and `idx_end` serve as indexes into `axon_segments`.
        # For example, the axon belonging to the neuron sitting at pixel
        # `idx_space` has segments
        # `axon_segments[idx_start[idx_space]:idx_end[idx_space]]`:
        for idx_ax in range(idx_start[idx_space], idx_end[idx_space]):
            ax_x = axon_segments[idx_ax, 0]
            ax_y = axon_segments[idx_ax, 1]
            # A segment with no location cannot be activated. That is a
            # property of the segment, so it is checked once here rather than
            # once per electrode:
            if c_isnan(ax_x) or c_isnan(ax_y):
                continue
            # Activation as a function of distance to the cell body (depends
            # on `lam`, precalculated during `build`):
            sens = axon_segments[idx_ax, 2]
            # Activation of this segment over time, by adding up the
            # contribution of each electrode:
            for idx_time in range(n_time):
                scratch[tid, idx_time] = <float32>0.0
            for idx_el in range(n_el):
                # An electrode that is zero for the whole stimulus
                # contributes nothing at any time point:
                if active[idx_el] == 0:
                    continue
                # Calculate the distance between this axon segment and the
                # center of the stimulating electrode:
                xdiff = ax_x - xel[idx_el]
                ydiff = ax_y - yel[idx_el]
                r2 = xdiff * xdiff + ydiff * ydiff
                # Past the cutoff radius. Note this drops `gauss * stim`, not
                # just `gauss`:
                if r2 > cutoff_r2:
                    continue
                # Activation as a function of distance to the stimulating
                # electrode (depends on `rho`). Neither this nor `sens`
                # depends on time, which is why time is the innermost loop:
                gauss = sens * c_exp(-r2 / (<float32>2.0 * rho * rho))
                for idx_time in range(n_time):
                    scratch[tid, idx_time] = (scratch[tid, idx_time] +
                                              gauss * stim[idx_el, idx_time])
            # After summing up the currents from all the electrodes, we
            # compare the brightness of the segment to the previously
            # brightest segment. The brightest segment overall determines the
            # brightness of the pixel:
            for idx_time in range(n_time):
                sgm = scratch[tid, idx_time]
                if c_abs(sgm) > c_abs(scratch[tid, n_time + idx_time]):
                    scratch[tid, n_time + idx_time] = sgm
        for idx_time in range(n_time):
            sgm = scratch[tid, n_time + idx_time]
            if c_abs(sgm) < thresh_percept:
                bright[idx_space, idx_time] = <float32>0.0
            else:
                bright[idx_space, idx_time] = sgm
    return np.asarray(bright)  # Py overhead
