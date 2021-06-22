import numpy as np
cimport numpy as cnp
from cython import cdivision  # for modulo operator
from cython.parallel import prange
from libc.math cimport(pow as c_pow, exp as c_exp, tanh as c_tanh,
                       sin as c_sin, cos as c_cos, fabs as c_abs)

ctypedef cnp.float32_t float32
ctypedef cnp.uint32_t uint32
ctypedef cnp.int32_t int32

cdef float32 deg2rad = 3.14159265358979323846 / 180.0


@cdivision(True)
cpdef fast_scoreboard(const float32[:, ::1] stim,
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

        px_bright = 0.0
        for idx_el in range(n_el):
            amp = stim[idx_el, idx_time]
            if c_abs(amp) > 0:
                dist2 = (c_pow(xgrid[idx_space] - xel[idx_el], 2) +
                         c_pow(ygrid[idx_space] - yel[idx_el], 2))
                gauss = c_exp(-dist2 / (2.0 * rho * rho))
                px_bright = px_bright + amp * gauss
        if c_abs(px_bright) < thresh_percept:
            px_bright = 0.0
        bright[idx_space, idx_time] = px_bright  # Py overhead
    return np.asarray(bright)  # Py overhead




cpdef fast_jansonius(float32[::1] rho, float32 phi0, float32 beta_s,
                     float32 beta_i):
    cdef:
        float32[::1] xprime, yprime
        float32 b, c, rho_min, tmp_phi, tmp_rho
        int32 idx

    if phi0 > 0:
        # Axon is in superior retina, compute `b` (real number) from Eq. 5:
        b = c_exp(beta_s + 3.9 * c_tanh(-(phi0 - 121.0) / 14.0))
        # Equation 3, `c` a positive real number:
        c = 1.9 + 1.4 * c_tanh((phi0 - 121.0) / 14.0)
    else:
        # Axon is in inferior retina: compute `b` (real number) from Eq. 6:
        b = -c_exp(beta_i + 1.5 * c_tanh(-(-phi0 - 90.0) / 25.0))
        # Equation 4, `c` a positive real number:
        c = 1.0 + 0.5 * c_tanh((-phi0 - 90.0) / 25.0)

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


cdef uint32 argmin_segment(float32[:, :] flat_bundles, float32 x, float32 y):
    cdef:
        float32 dist2, min_dist2
        int32 seg, n_seg
        uint32 min_seg

    min_dist2 = 1e12
    n_seg = flat_bundles.shape[0]
    for seg in range(n_seg):
        dist2 = (c_pow(flat_bundles[seg, 0] - x, 2) +
                 c_pow(flat_bundles[seg, 1] - y, 2))
        if dist2 < min_dist2:
            min_dist2 = dist2
            min_seg = seg
    return min_seg


cpdef fast_find_closest_axon(float32[:, :] flat_bundles,
                             float32[::1] xret,
                             float32[::1] yret):
    cdef:
        uint32[::1] closest_seg
        int32 n_xy, n_seg
        int32 pos
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
                    float32 thresh_percept):
    """Fast spatial response of the axon map model

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
    """
    cdef:
        int32 idx_el, idx_time, idx_space, idx_ax, idx_bright
        int32 n_el, n_time, n_space, n_ax, n_bright
        float32[:, ::1] bright
        float32 px_bright, xdiff, ydiff, r2, gauss, sgm_bright, amp
        int32 i0, i1

    n_el = stim.shape[0]
    n_time = stim.shape[1]
    n_space = len(idx_start)
    n_bright = n_time * n_space

    # A flattened array containing n_space x n_time entries:
    bright = np.empty((n_space, n_time), dtype=np.float32)  # Py overhead

    # Parallel loop over all pixels to be rendered:
    for idx_space in prange(n_space, schedule='static', nogil=True):
        # Each frame in `stim` is treated independently, so we can have an
        # inner loop over all points in time:
        for idx_time in range(n_time):
            # Find the brightness value of each pixel (`px_bright`) by finding
            # the strongest activated axon segment:
            px_bright = 0.0
            # Slice `axon_contrib` (but don't assign the slice to a variable).
            # `idx_start` and `idx_end` serve as indexes into `axon_segments`.
            # For example, the axon belonging to the neuron sitting at pixel
            # `idx_space` has segments
            # `axon_segments[idx_start[idx_space]:idx_end[idx_space]]`:
            for idx_ax in range(idx_start[idx_space], idx_end[idx_space]):
                # Calculate the activation of each axon segment by adding up
                # the contribution of each electrode:
                sgm_bright = 0.0
                for idx_el in range(n_el):
                    amp = stim[idx_el, idx_time]
                    if c_abs(amp) > 0:
                        # Calculate the distance between this axon segment and
                        # the center of the stimulating electrode:
                        xdiff = axon_segments[idx_ax, 0] - xel[idx_el]
                        ydiff = axon_segments[idx_ax, 1] - yel[idx_el]
                        r2 = xdiff * xdiff + ydiff * ydiff
                        # Determine the activation level of this axon segment,
                        # consisting of two things:
                        # - activation as a function of distance to the
                        #   stimulating electrode (depends on `rho`):
                        gauss = c_exp(-r2 / (2.0 * rho * rho))
                        # - activation as a function of distance to the cell
                        #   body (depends on `axlambda`, precalculated during
                        #   `build` and stored in `axon_segments[idx_ax, 2]`:
                        sgm_bright = (sgm_bright +
                                      gauss * axon_segments[idx_ax, 2] * amp)
                # After summing up the currents from all the electrodes, we
                # compare the brightness of the segment (`sgm_bright`) to the
                # previously brightest segment. The brightest segment overall
                # determines the brightness of the pixel (`px_bright`):
                if c_abs(sgm_bright) > c_abs(px_bright):
                    px_bright = sgm_bright
            if c_abs(px_bright) < thresh_percept:
                px_bright = 0.0
            bright[idx_space, idx_time] = px_bright  # Py overhead
    return np.asarray(bright)  # Py overhead

