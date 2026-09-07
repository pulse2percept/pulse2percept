from libc.math cimport(powf as c_pow, expf as c_exp, tanhf as c_tanh,
                       sinf as c_sin, cosf as c_cos, fabsf as c_abs,
                       isnan as c_isnan, sqrtf as c_sqrt)
from cython.parallel import prange
from cython.parallel cimport threadid
from cython import cdivision  # for modulo operator
import numpy as np
cimport numpy as cnp
from .._scoreboard cimport _active_electrodes
cnp.import_array()

ctypedef cnp.float32_t float32
ctypedef cnp.uint32_t uint32
ctypedef cnp.int32_t int32
ctypedef Py_ssize_t index_t

cdef float32 deg2rad = <float32>(3.14159265358979323846 / 180.0)


cdef inline index_t _lower_bound(const float32[::1] xs, index_t n,
                                 float32 value) noexcept nogil:
    """Index of the first entry of sorted ``xs[:n]`` at or above ``value``."""
    cdef index_t lo = 0, hi = n, mid
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if xs[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


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

    Only electrodes within ``cutoff_r2`` of a segment can contribute to it.
    The active electrodes are sorted by x once per call and each segment
    binary-searches the band ``[ax_x - r, ax_x + r]``; with a finite cutoff
    that leaves the electrode loop walking the band rather than the whole
    array. Sorting also lets the inactive electrodes be dropped outright
    rather than branched past. The band is one-dimensional, so the exact
    ``r2 <= cutoff_r2`` test still decides which of its electrodes count; an
    infinite cutoff puts every electrode in the band and rejects none.

    The innermost loop over time accumulates into independent slots of a
    scratch buffer, so it vectorizes without relaxed floating-point
    semantics. Nothing of size ``n_segments x n_electrodes`` or
    ``n_segments x n_time`` is ever materialized: each thread holds two
    buffers of ``n_time`` floats.

    .. note::

        Electrodes are summed in order of increasing x rather than in the
        order they were passed, so results can differ in the last bits from
        a version that summed them in array order.

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
        index_t idx_el, idx_time, idx_space, idx_ax, tid, row, lo_el
        index_t n_el, n_time, n_space, stride
        float32[:, ::1] bright
        float32[:, ::1] scratch
        const float32[:, ::1] stim_s
        const float32[::1] xs
        const float32[::1] ys
        float32 xdiff, ydiff, r2, gauss, sens, ax_x, ax_y, sgm, cutoff_r, x_hi

    n_time = stim.shape[1]
    n_space = len(idx_start)
    # `num_threads(0)` is not conforming OpenMP, and the scratch buffer below
    # is sized on the assumption that no thread id can reach `n_threads`:
    if n_threads < 1:
        n_threads = 1

    # A flattened array containing n_space x n_time entries:
    bright = np.empty((n_space, n_time), dtype=np.float32)  # Py overhead

    # Keep the electrodes that carry current at some point, in order of
    # increasing x. `stim` is permuted alongside so that the band the loop
    # below walks is contiguous in all three arrays.  # Py overhead follows
    keep = np.asarray(_active_electrodes(stim)).view(np.bool_).nonzero()[0]
    keep = keep[np.argsort(np.asarray(xel)[keep], kind='stable')]
    xs = np.ascontiguousarray(np.asarray(xel)[keep])
    ys = np.ascontiguousarray(np.asarray(yel)[keep])
    stim_s = np.ascontiguousarray(np.asarray(stim)[keep])
    n_el = len(keep)
    # Half-width of the x band to search. `inf` (no cutoff) carries through:
    # every electrode then sorts into the band and none is ever rejected.
    cutoff_r = c_sqrt(cutoff_r2)

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
            # Only the electrodes whose x lies within `cutoff_r` of this
            # segment can clear the cutoff, and `xs` is sorted, so they are
            # one contiguous run. Seek its start, then walk until x leaves the
            # band -- no electrode outside it is ever looked at:
            lo_el = _lower_bound(xs, n_el, ax_x - cutoff_r)
            x_hi = ax_x + cutoff_r
            for idx_el in range(lo_el, n_el):
                if xs[idx_el] > x_hi:
                    break
                # Calculate the distance between this axon segment and the
                # center of the stimulating electrode:
                xdiff = ax_x - xs[idx_el]
                ydiff = ax_y - ys[idx_el]
                r2 = xdiff * xdiff + ydiff * ydiff
                # In the band on x, but still too far in y. Note this drops
                # `gauss * stim`, not just `gauss`:
                if r2 > cutoff_r2:
                    continue
                # Activation as a function of distance to the stimulating
                # electrode (depends on `rho`). Neither this nor `sens`
                # depends on time, which is why time is the innermost loop:
                gauss = sens * c_exp(-r2 / (<float32>2.0 * rho * rho))
                for idx_time in range(n_time):
                    scratch[tid, idx_time] = (scratch[tid, idx_time] +
                                              gauss * stim_s[idx_el, idx_time])
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
