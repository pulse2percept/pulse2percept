# distutils: language = c

cimport numpy as cnp
ctypedef cnp.float32_t float32
ctypedef Py_ssize_t index_t

cpdef bint fast_is_strictly_increasing(float32[::1] a, float32[::1] b, float32 tol)
