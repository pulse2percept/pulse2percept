import numpy as np
cimport numpy as cnp
from ._array cimport float32, int32

cpdef bint fast_is_strictly_increasing(float32[::1] a, float32[::1] b, float32 tol):
    cdef:
        int32 idx, arr_len
    arr_len = len(a)
    with nogil:
        for i in range(arr_len):
            if b[i] - a[i] < tol:
                return False
    return True
