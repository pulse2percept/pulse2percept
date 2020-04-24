import numpy as np
cimport numpy as cnp
from cython import cdivision  # for modulo operator
from cython.parallel import prange
from libc.math cimport(pow as c_pow, exp as c_exp, tanh as c_tanh,
                       sin as c_sin, cos as c_cos, fabs as c_abs)

ctypedef cnp.float32_t float32
ctypedef cnp.uint32_t uint32
cdef float32 deg2rad = 3.14159265358979323846 / 180.0


cdef float32 c_min(float32[:] arr):
    cdef:
        float32 arr_min
        uint32 idx, arr_len

    arr_min = 1e12
    arr_len = len(arr)
    for idx in range(arr_len):
        if arr[idx] < arr_min:
            arr_min = arr[idx]
    return arr_min


cdef float32 c_max(float32[:] arr):
    cdef:
        float32 arr_max
        uint32 idx, arr_len

    arr_max = -1e12
    arr_len = len(arr)
    for idx in range(arr_len):
        if arr[idx] > arr_max:
            arr_max = arr[idx]
    return arr_max


@cdivision(True)
cpdef gauss2(float32[:, ::1] arr, float32 x, float32 y,
             float32 tau):
    cdef:
        uint32 idx, n_arr
        float32 dist2
        float32[::1] gauss
    n_arr = arr.shape[0]
    gauss = np.empty(n_arr, dtype=np.float32)
    with nogil:
        for idx in range(n_arr):
            dist2 = c_pow(arr[idx, 0] - x, 2) + c_pow(arr[idx, 1] - y, 2)
            gauss[idx] = c_exp(-dist2 / (2.0 * c_pow(tau, 2)))
    return np.asarray(gauss)


cpdef jansonius(float32[::1] rho, float32 phi0, float32 beta_s,
                float32 beta_i):
    cdef:
        float32[::1] xprime, yprime
        float32 b, c, rho_min, tmp_phi, tmp_rho
        uint32 idx

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
    rho_min = c_min(rho)
    with nogil:
        for idx in range(len(rho)):
            tmp_rho = rho[idx]
            tmp_phi = phi0 + b * c_pow(tmp_rho - rho_min, c)
            xprime[idx] = tmp_rho * c_cos(deg2rad * tmp_phi)
            yprime[idx] = tmp_rho * c_sin(deg2rad * tmp_phi)
    return np.asarray(xprime), np.asarray(yprime)


cdef uint32 argmin_segment(float32[:, ::1] bundles, float32 x, float32 y):
    cdef:
        float32 dist2, min_dist2
        uint32 seg, min_seg, n_seg

    min_dist2 = 1e12
    n_seg = bundles.shape[0]
    for seg in range(n_seg):
        dist2 = c_pow(bundles[seg, 0] - x, 2) + c_pow(bundles[seg, 1] - y, 2)
        if dist2 < min_dist2:
            min_dist2 = dist2
            min_seg = seg
    return min_seg


cpdef axon_contribution(float32[:, ::1] bundle,
                        float32[::1] xy,
                        float32 lmbd):
    cdef:
        uint32 p, c, argmin, n_seg
        float32 dist2
        float32[:, ::1] contrib

    # Find the segment that is closest to the soma `xy`:
    argmin = argmin_segment(bundle, xy[0], xy[1])

    # Add the exact location of the soma:
    bundle[argmin + 1, 0] = xy[0]
    bundle[argmin + 1, 1] = xy[1]

    # For every axon segment, calculate distance from soma by summing up the
    # individual distances between neighboring axon segments
    # (by "walking along the axon"):
    n_seg = argmin + 1
    contrib = np.zeros((n_seg, 3))
    dist2 = 0
    c = 0
    for p in range(argmin, -1, -1):
        dist2 += (c_pow(bundle[p, 0] - bundle[p + 1, 0], 2) +
                  c_pow(bundle[p, 1] - bundle[p + 1, 1], 2))
        contrib[c, 0] = bundle[p, 0]
        contrib[c, 1] = bundle[p, 1]
        contrib[c, 2] = c_exp(-dist2 / (2.0 * c_pow(lmbd, 2)))
        c += 1
    return np.asarray(contrib)


cpdef finds_closest_axons(float32[:, ::1] bundles,
                          float32[::1] xret,
                          float32[::1] yret):
    cdef:
        uint32[::1] closest_seg
        uint32 n_xy, n_seg
    closest_seg = np.empty(len(xret), dtype=np.uint32)
    n_xy = len(xret)
    n_seg = bundles.shape[0]
    for pos in range(n_xy):
        closest_seg[pos] = argmin_segment(bundles, xret[pos], yret[pos])
    return np.asarray(closest_seg)


@cdivision(True)
cpdef spatial_fast(const float32[:, ::1] stim,
                   const float32[::1] xel,
                   const float32[::1] yel,
                   const float32[:, ::1] axon,
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
    axon : 2D float32 array
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
        size_t idx_el, idx_time, idx_space, idx_ax, n_el, n_time, n_space, n_ax
        size_t idx_bright, n_bright
        float32[:, ::1] bright
        float32 px_bright, xdiff, ydiff, r2, gauss, sgm_bright, amp
        size_t i0, i1

    n_el = stim.shape[0]
    n_time = stim.shape[1]
    n_space = len(idx_start)
    n_bright = n_time * n_space

    # A flattened array containing n_space x n_time entries:
    bright = np.empty((n_space, n_time), dtype=np.float32)  # Py overhead

    for idx_space in prange(n_space, schedule='dynamic', nogil=True):
        # Parallel loop over all pixels to be rendered:
        for idx_time in range(n_time):
            # Each frame in ``stim`` is treated independently. Find the
            # brightness value of each pixel by finding the strongest activated
            # axon segment:
            px_bright = 0.0
            # Slice `axon_contrib` but don't assign the slice to a variable:
            for idx_ax in range(idx_start[idx_space], idx_end[idx_space]):
                # Calculate the activation of each axon segment:
                sgm_bright = 0.0
                for idx_el in range(n_el):
                    amp = stim[idx_el, idx_time]
                    if c_abs(amp) > 0:
                        xdiff = axon[idx_ax, 0] - xel[idx_el]
                        ydiff = axon[idx_ax, 1] - yel[idx_el]
                        r2 = xdiff * xdiff + ydiff * ydiff
                        # axon[idx_ax, 2] depends on axlambda:
                        gauss = c_exp(-r2 / (2.0 * rho * rho))
                        sgm_bright = sgm_bright + gauss * axon[idx_ax, 2] * amp
                if c_abs(sgm_bright) > c_abs(px_bright):
                    # The strongest activated axon segment determines the
                    # brightness of the pixel:
                    px_bright = sgm_bright
            if c_abs(px_bright) < thresh_percept:
                px_bright = 0.0
            bright[idx_space, idx_time] = px_bright  # Py overhead
    return np.asarray(bright)  # Py overhead

