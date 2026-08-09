# distutils: language = c
# cython: language_level=3
cimport numpy as cnp
cnp.import_array()


ctypedef cnp.float64_t float64
ctypedef Py_ssize_t index_t

cpdef bint fast_is_strictly_increasing(float64[::1] a, float64[::1] b, float64 tol) noexcept nogil:
    """Check if b[i] - a[i] is strictly greater than tol for all i

    Takes float64 because it is used on stimulus time axes, which are stored
    as float64: float32 cannot resolve two time points a DT=1e-3 ms step apart
    once they are more than 8.4 s in, and would report a perfectly good time
    axis as non-increasing.
    """
    cdef index_t i, arr_len = a.shape[0]

    for i in range(arr_len):
        if b[i] - a[i] < tol:
            return False
    return True
