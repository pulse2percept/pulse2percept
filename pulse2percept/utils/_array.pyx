# distutils: language = c++
# ^ needed for bool

from libcpp cimport bool
import numpy as np
cimport numpy as cnp

ctypedef cnp.float32_t float32
ctypedef cnp.int32_t int32


cpdef bool fast_is_strictly_increasing(float32[::1] a, float32[::1] b,
                                       float32 tol):
    cdef:
        int32 idx, arr_len
    arr_len = len(a)
    with nogil:
        for i in range(arr_len):
            if b[i] - a[i] < tol:
                return False
    return True
