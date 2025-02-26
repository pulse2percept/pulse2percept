# distutils: language = c++
# ^ needed for bool

from libcpp cimport bool
cimport numpy as cnp

ctypedef cnp.float32_t float32
ctypedef cnp.uint32_t uint32
ctypedef cnp.int32_t int32
ctypedef Py_ssize_t index_t

# --- SCALAR FUNCTIONS ------------------------------------------------------- #

cdef float32 c_fmax(float32 a, float32 b) noexcept nogil

cdef float32 c_fmin(float32 a, float32 b) noexcept nogil

cdef bool c_isclose(float32 a, float32 b, float32 rel_tol=*, float32 abs_tol=*) noexcept nogil

cdef float32 c_expit(float32 x) noexcept nogil

cpdef float32 c_gcd(float32 a, float32 b, float32 rtol=*, float32 atol=*) noexcept nogil

# --- ARRAY FUNCTIONS -------------------------------------------------------- #

cdef float32 c_min(float32[::1] arr) noexcept nogil

cdef float32 c_max(float32[::1] arr) noexcept nogil

cdef void c_cumpow(float32* arr_in, float32* arr_out, index_t N, float32 exp) noexcept nogil