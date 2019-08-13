import numpy as np
cimport numpy as np
import cython
from libc.math cimport(pow as c_pow, exp as c_exp, tanh as c_tanh,
                       sin as c_sin, cos as c_cos)


cdef deg2rad = 3.14159265358979323846 / 180.0

cdef double c_min(double[:] arr):
    cdef double arr_min
    cdef np.intp_t idx, arr_len

    arr_min = 1e12
    arr_len = len(arr)
    for idx in range(arr_len):
        if arr[idx] < arr_min:
            arr_min = arr[idx]
    return arr_min

cdef double c_max(double[:] arr):
    cdef double arr_max
    cdef np.intp_t idx, arr_len

    arr_max = -1e12
    arr_len = len(arr)
    for idx in range(arr_len):
        if arr[idx] > arr_max:
            arr_max = arr[idx]
    return arr_max


cpdef gauss2(double[:, ::1] arr, double x, double y, double tau):
    cdef np.intp_t idx, n_arr
    cdef double dist2
    n_arr = arr.shape[0]
    cdef double[:] gauss = np.empty(n_arr)
    with nogil:
        for idx in range(n_arr):
            dist2 = c_pow(arr[idx, 0] - x, 2) + c_pow(arr[idx, 1] - y, 2)
            gauss[idx] = c_exp(-dist2 / (2.0 * c_pow(tau, 2)))
    return np.asarray(gauss)


cpdef jansonius(double[:] rho, double phi0, double beta_s, double beta_i):
    cdef double b, c, rho_min, tmp_phi, tmp_rho
    cdef np.intp_t idx

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

    cdef double[:] xprime = np.empty_like(rho)
    cdef double[:] yprime = np.empty_like(rho)
    rho_min = c_min(rho)
    for idx in range(len(rho)):
        tmp_rho = rho[idx]
        tmp_phi = phi0 + b * c_pow(tmp_rho - rho_min, c)
        xprime[idx] = tmp_rho * c_cos(deg2rad * tmp_phi)
        yprime[idx] = tmp_rho * c_sin(deg2rad * tmp_phi)
    return np.asarray(xprime), np.asarray(yprime)


cdef np.intp_t argmin_segment(double[:, :] bundles, double x, double y):
    cdef double dist2, min_dist2
    cdef np.intp_t seg, min_seg, n_seg

    min_dist2 = 1e12
    n_seg = bundles.shape[0]
    for seg in range(n_seg):
        dist2 = c_pow(bundles[seg, 0] - x, 2) + c_pow(bundles[seg, 1] - y, 2)
        if dist2 < min_dist2:
            min_dist2 = dist2
            min_seg = seg
    return min_seg


cpdef axon_contribution(double[:, :] bundle, double[:] xy, double lmbd):
    cdef np.intp_t p, c, argmin, n_seg
    cdef double dist2
    cdef double[:, :] contrib

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


cpdef finds_closest_axons(double[:, :] bundles, double[:] xret,
                          double[:] yret):
    cdef np.intp_t[:] closest_seg = np.empty(len(xret), dtype=int)
    cdef np.intp_t n_xy, n_seg
    n_xy = len(xret)
    n_seg = bundles.shape[0]
    for pos in range(n_xy):
        closest_seg[pos] = argmin_segment(bundles, xret[pos], yret[pos])
    return np.asarray(closest_seg)


cpdef axon_map(double[:] stim, double[:] xel, double[:] yel,
               double[:, ::1] axon, double rho):
    cdef np.intp_t idx, n_stim, n_ax
    cdef double bright, gauss
    n_stim = len(stim)
    n_ax = axon.shape[0]
    cdef double[:] act = np.zeros(n_ax)
    with nogil:
        for i_stim in range(n_stim):
            for i_ax in range(n_ax):
                r2 = c_pow(axon[i_ax, 0] - xel[i_stim], 2)
                r2 += c_pow(axon[i_ax, 1] - yel[i_stim], 2)
                gauss = c_exp(-r2 / (2.0 * c_pow(rho, 2)))
                act[i_ax] += stim[i_stim] * axon[i_ax, 2] * gauss
    bright = c_max(act)
    return bright
