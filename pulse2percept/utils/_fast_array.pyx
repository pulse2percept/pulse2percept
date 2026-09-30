# distutils: language = c
# cython: language_level=3
cimport numpy as cnp
cnp.import_array()


ctypedef cnp.float64_t float64
ctypedef Py_ssize_t index_t

cpdef bint fast_is_strictly_increasing(const float64[::1] a,
                                      const float64[::1] b,
                                      float64 tol) noexcept nogil:
    """Check if b[i] - a[i] is strictly greater than tol for all i

    Uses float64 for stimulus time axes: float32 cannot resolve a 1e-3 ms
    step beyond 8.4 s.
    """
    cdef index_t i, arr_len = a.shape[0]

    for i in range(arr_len):
        if b[i] - a[i] < tol:
            return False
    return True
