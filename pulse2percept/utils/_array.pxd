ctypedef numpy.float32_t float32
ctypedef numpy.int32_t int32

cpdef bool fast_is_strictly_increasing(float32[::1] a, float32[::1] b, float32 tol)
