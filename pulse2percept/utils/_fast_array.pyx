# distutils: language = c

import numpy as np
cimport numpy as cnp

ctypedef cnp.float32_t float32
ctypedef Py_ssize_t index_t

cpdef bint fast_is_strictly_increasing(float32[::1] a, float32[::1] b, float32 tol):
    """Check if b[i] - a[i] is strictly greater than tol for all i"""
    cdef index_t i, arr_len = a.shape[0]

    with nogil:
        for i in range(arr_len):
            if b[i] - a[i] < tol:
                return False
    return True
