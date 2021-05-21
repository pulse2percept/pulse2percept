# distutils: language = c++
# ^ needed for bool

from cython import cdivision  # modulo, division by zero
from libc.math cimport fabs as c_abs, exp as c_exp, pow as c_pow
from libcpp cimport bool
import numpy as np
cimport numpy as cnp

# --- SCALAR FUNCTIONS ------------------------------------------------------- #

cdef inline float32 c_fmax(float32 a, float32 b) nogil:
    return a if a >= b else b

cdef inline bool c_isclose(float32 a, float32 b, float32 rel_tol=1e-09,
                           float32 abs_tol=0.0) nogil:
    return c_abs(a-b) <= c_fmax(rel_tol * c_fmax(c_abs(a), c_abs(b)), abs_tol)


@cdivision(True)
cdef inline float32 c_expit(float32 x) nogil:
    return 1.0 / (1.0 + c_exp(-x))


# --- ARRAY FUNCTIONS -------------------------------------------------------- #

cdef float32 c_min(float32[::1] arr):
    cdef:
        float32 arr_min
        int32 idx, arr_len

    arr_min = np.inf
    arr_len = len(arr)
    for idx in range(arr_len):
        if arr[idx] < arr_min:
            arr_min = arr[idx]
    return arr_min


cdef float32 c_max(float32[::1] arr):
    cdef:
        float32 arr_max
        int32 idx, arr_len

    arr_max = -np.inf
    arr_len = len(arr)
    for idx in range(arr_len):
        if arr[idx] > arr_max:
            arr_max = arr[idx]
    return arr_max


cdef void c_cumpow(float32* arr_in, float32* arr_out, int32 N, int32 exp) nogil:
    cdef:
        int32 i = 0
        float32 sum = 0
    for i in range(N):
        sum += arr_in[i]
        arr_out[i] = c_pow(sum, exp)
