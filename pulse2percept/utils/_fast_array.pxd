# distutils: language = c++
# ^ needed for bool

from libcpp cimport bool
import numpy as np
cimport numpy as cnp

ctypedef cnp.float32_t float32
ctypedef cnp.uint32_t uint32

cpdef bool fast_is_strictly_increasing(float32[::1] a, float32[::1] b, float32 tol)
