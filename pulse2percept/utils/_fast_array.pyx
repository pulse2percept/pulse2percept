# distutils: language = c++
# ^ needed for bool

from pulse2percept.utils._fast_array cimport float32, uint32
from libcpp cimport bool
import numpy as np
cimport numpy as cnp

cpdef bool fast_is_strictly_increasing(float32[::1] a, float32[::1] b, float32 tol):
    cdef:
        uint32 i, arr_len
    arr_len = len(a)
    with nogil:
        for i in range(arr_len):
            if b[i] - a[i] < tol:
                return False
    return True
