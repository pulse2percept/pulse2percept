cimport numpy as cnp

ctypedef cnp.float32_t float32
ctypedef cnp.int32_t int32

cpdef bint fast_is_strictly_increasing(float32[::1] a, float32[::1] b, float32 tol)
