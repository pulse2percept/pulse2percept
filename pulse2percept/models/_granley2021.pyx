from libc.math cimport(powf as c_pow, expf as c_exp, logf as c_log,
                       fabs as c_abs, isnan as c_isnan)
from cython.parallel import prange
from cython import cdivision  # for modulo operator
import numpy as np
cimport numpy as cnp
cnp.import_array()
cimport cython

ctypedef cnp.float32_t float32
ctypedef cnp.uint32_t uint32
ctypedef cnp.int32_t int32
ctypedef Py_ssize_t index_t
cdef float32 deg2rad = 3.14159265358979323846 / 180.0


@cython.boundscheck(False)
@cdivision(True)
cpdef fast_biphasic_axon_map(const float32[::1] amp_el,
                             const float32[::1] bright_model_el,
                             const float32[::1] size_model_el,
                             const float32[::1] streak_model_el,
                             const float32[::1] xel,
                             const float32[::1] yel,
                             const float32[:, ::1] axon_segments,
                             const uint32[::1] idx_start,
                             const uint32[::1] idx_end,
                             float32 rho,
                             float32 thresh_percept,
                             float32 cutoff_r2,
                             uint32 n_threads):
    """Fast spatial response of the biphasic axon map model
    Predicts representative percept using entire time interval,
    and returns this percept repeated at each time point

    The activation of a segment by an electrode is
    ``exp(-r^2 / (2 rho^2 F_size)) * sensitivity ** (1 / F_streak)``. Both
    factors are exponentials, so they are evaluated as a single ``exp`` of the
    summed exponents: the power becomes ``exp(log(sensitivity) / F_streak)``,
    and ``log(sensitivity)`` depends only on the segment, so it is taken once
    per segment instead of once per segment and electrode. That trades a
    ``powf`` per pair -- the most expensive call in the loop -- for one
    ``logf`` per segment.

    ``F_streak`` is strictly positive: the default streak model clamps it to
    ``min_lambda ** 2 / axlambda ** 2``, and ``_predict_spatial`` rejects a
    custom model that returns anything else. Sensitivities are likewise
    positive, so the logarithm is always defined.

    Parameters
    ----------
    amp_el : 1D float array 
        Amplitudes (as a factor of threshold) per electrode
    bright_model_el, size_model_el, streak_model_el : 1D float array
        Factors by which to scale brightness, rho (size), and lambda (streak length)
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
        Squared distance (microns^2) at which an electrode of unscaled size
        stops contributing; scaled per electrode by its ``F_size``. Pass
        ``inf`` to sum over every electrode. See ``min_current_spread`` on the
        model for how this is derived.
    n_threads: uint32
        Number of CPU threads to use during parallelization using OpenMP.

    Return Value
    -----------------
    Array with shape (n_points) representing the brightest frame of the percept
    """
    cdef:
        index_t idx_el, idx_space, idx_ax
        index_t n_el, n_space
        float32[::1] bright
        float32[::1] neg_inv_2rho2, inv_streak, cutoff_el
        cnp.uint8_t[::1] active
        float32 px_bright, xdiff, ydiff, r2, ax_x, ax_y, log_sens
        float32 sgm_bright

    n_el = xel.shape[0]
    n_space = len(idx_start)
    if n_threads < 1:  # `num_threads(0)` is not conforming OpenMP
        n_threads = 1

    # An array containing n_space entries
    bright = np.zeros((n_space), dtype=np.float32)  # Py overhead

    # Everything that depends only on the electrode is worked out once here,
    # rather than once for every (segment, electrode) pair:
    size_np = np.asarray(size_model_el, dtype=np.float32)
    neg_inv_2rho2 = (-1.0 / (2.0 * rho * rho * size_np)).astype(np.float32)
    inv_streak = (1.0 / np.asarray(streak_model_el,
                                   dtype=np.float32)).astype(np.float32)
    cutoff_el = (cutoff_r2 * size_np).astype(np.float32)
    active = (np.abs(np.asarray(amp_el, dtype=np.float32)) >
              0).astype(np.uint8)

    # Parallel loop over all pixels to be rendered. `guided` rather than
    # `static`: axons differ several-fold in how many segments they have, and
    # with the cutoff above, how many electrodes reach a given segment varies
    # too, so equal-sized chunks are not equal-sized work.
    for idx_space in prange(n_space, schedule='guided', nogil=True,
                            num_threads=n_threads):
        # Find the brightness value of each pixel (`px_bright`) by finding
        # the strongest activated axon segment:
        px_bright = 0.0
        # Slice `axon_contrib` (but don't assign the slice to a variable).
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
            # Sensitivity as a function of distance to the cell soma,
            # precalculated during `build` and stored in
            # `axon_segments[idx_ax, 2]`. The streak model rescales it by a
            # per-electrode exponent below; taking the logarithm here turns
            # that power into a multiply inside the electrode loop:
            log_sens = c_log(axon_segments[idx_ax, 2])
            # Calculate the activation of each axon segment by adding up
            # the contribution of each electrode:
            sgm_bright = 0.0
            for idx_el in range(n_el):
                if active[idx_el] == 0:
                    continue
                # Calculate the distance between this axon segment and
                # the center of the stimulating electrode:
                xdiff = ax_x - xel[idx_el]
                ydiff = ax_y - yel[idx_el]
                r2 = xdiff * xdiff + ydiff * ydiff
                # Too far away for the exponential below to resolve:
                if r2 > cutoff_el[idx_el]:
                    continue
                # Distance to the electrode and distance to the soma both
                # enter as exponentials, so they are summed in the exponent
                # and raised once:
                sgm_bright = (sgm_bright + bright_model_el[idx_el] *
                              c_exp(r2 * neg_inv_2rho2[idx_el] +
                                    log_sens * inv_streak[idx_el]))
            # After summing up the currents from all the electrodes, we
            # compare the brightness of the segment (`sgm_bright`) to the
            # previously brightest segment. The brightest segment overall
            # determines the brightness of the pixel (`px_bright`):
            if c_abs(sgm_bright) > c_abs(px_bright):
                px_bright = sgm_bright
        if c_abs(px_bright) < thresh_percept:
            px_bright = 0.0
        bright[idx_space] = px_bright  # Py overhead
    return np.asarray(bright)
