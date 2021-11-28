from libc.math cimport(pow as c_pow, exp as c_exp, tanh as c_tanh,
                       sin as c_sin, cos as c_cos, fabs as c_abs,
                       isnan as c_isnan)
from cython.parallel import prange
from cython import cdivision  # for modulo operator
import numpy as np
cimport numpy as cnp
cimport cython

ctypedef cnp.float32_t float32
ctypedef cnp.uint32_t uint32
ctypedef cnp.int32_t int32
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
                             float32 thresh_percept):
    """Fast spatial response of the biphasic axon map model
    Predicts representative percept using entire time interval, 
    and returns this percept repeated at each time point
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

    Return Value
    -----------------
    Array with shape (n_points) representing the brightest frame of the percept
    """
    cdef:
        int32 idx_el, idx_time, idx_space, idx_ax, idx_bright
        int32 n_el, n_time, n_space, n_ax, n_bright
        float32[::1] bright
        float32 px_bright, xdiff, ydiff, r2, amp, gauss_el, gauss_soma
        float32 sgm_bright, bright_effect, size_effect, streak_effect
        int32 i0, i1

    n_el = xel.shape[0]
    n_space = len(idx_start)
    n_bright = n_space

    # An array containing n_space entries
    bright = np.zeros((n_space), dtype=np.float32)  # Py overhead

    # Parallel loop over all pixels to be rendered:
    for idx_space in prange(n_space, schedule='static', nogil=True):
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
                amp = amp_el[idx_el]
                bright_effect = bright_model_el[idx_el]
                size_effect = size_model_el[idx_el]
                streak_effect = streak_model_el[idx_el]
                if c_abs(amp) > 0:
                    if (c_isnan(axon_segments[idx_ax, 0]) or
                            c_isnan(axon_segments[idx_ax, 1])):
                        continue
                    # Calculate the distance between this axon segment and
                    # the center of the stimulating electrode:
                    xdiff = axon_segments[idx_ax, 0] - xel[idx_el]
                    ydiff = axon_segments[idx_ax, 1] - yel[idx_el]
                    r2 = xdiff * xdiff + ydiff * ydiff
                    # Determine the activation level of this axon segment,
                    # consisting of two things:
                    # - activation as a function of distance to the
                    #   stimulating electrode (depends on `rho`):
                    gauss_el = c_exp(-r2 / (2.0 * rho * rho * size_effect))
                    # - activation as a function of distance to the cell
                    #   soma (depends on `axlambda`, precalculated during
                    #   `build` and stored in `axon_segments[idx_ax, 2]`
                    #   precalculated value does not include streak model
                    #   effect, which must be added now
                    gauss_soma = c_pow(
                        axon_segments[idx_ax, 2], 1 / streak_effect)
                    sgm_bright = (sgm_bright +
                                  bright_effect * gauss_el * gauss_soma)
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
