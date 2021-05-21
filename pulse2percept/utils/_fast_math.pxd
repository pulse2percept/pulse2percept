# distutils: language = c++
# ^ needed for bool

from libcpp cimport bool
cimport numpy as cnp

ctypedef cnp.float32_t float32
ctypedef cnp.uint32_t uint32
ctypedef cnp.int32_t int32

cdef float32 deg2rad = 3.14159265358979323846 / 180.0


# --- SCALAR FUNCTIONS ------------------------------------------------------- #

cdef float32 c_fmax(float32 a, float32 b) nogil

cdef bool c_isclose(float32 a, float32 b, float32 rel_tol=*,
                    float32 abs_tol=*) nogil

cdef float32 c_expit(float32 x) nogil

# --- ARRAY FUNCTIONS -------------------------------------------------------- #

cdef float32 c_min(float32[:] arr)

cdef float32 c_max(float32[:] arr)
