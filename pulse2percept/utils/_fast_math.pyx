# distutils: language = c

from pulse2percept.utils._fast_math cimport float32, int32, index_t
from cython import cdivision  # modulo, division by zero
from libc.math cimport fabsf as c_abs, expf as c_exp, powf as c_pow, HUGE_VALF
cimport numpy as cnp


# --- SCALAR FUNCTIONS ------------------------------------------------------- #

cdef inline float32 c_fmax(float32 a, float32 b) noexcept nogil:
    return a if a >= b else b

cdef inline float32 c_fmin(float32 a, float32 b) noexcept nogil:
    return a if a <= b else b

cdef inline bint c_isclose(float32 a, float32 b, float32 rel_tol=1e-09,
                           float32 abs_tol=0.0) noexcept nogil:
    return c_abs(a-b) <= c_fmax(rel_tol * c_fmax(c_abs(a), c_abs(b)), abs_tol)


@cdivision(True)
cdef inline float32 c_expit(float32 x) noexcept nogil:
    return <float32>1.0 / (<float32>1.0 + c_exp(-x))


@cdivision(True)
cpdef inline float32 c_gcd(float32 a, float32 b, float32 rtol=1e-5,
                          float32 atol=1e-8) noexcept nogil:
    cdef float32 t
    t = c_fmin(c_abs(a), c_abs(b))
    while c_abs(b) > rtol * t + atol:
        a, b = b, a % b
    return a


# --- ARRAY FUNCTIONS -------------------------------------------------------- #

cdef float32 c_min(float32[::1] arr) noexcept nogil:
    cdef:
        float32 arr_min
        index_t idx, arr_len

    arr_min = HUGE_VALF
    with gil:
        arr_len = len(arr)

    for idx in range(arr_len):
        if arr[idx] < arr_min:
            arr_min = arr[idx]
    return arr_min


cdef float32 c_max(float32[::1] arr) noexcept nogil:
    cdef:
        float32 arr_max
        index_t idx, arr_len

    arr_max = -HUGE_VALF
    with gil:
        arr_len = len(arr)

    for idx in range(arr_len):
        if arr[idx] > arr_max:
            arr_max = arr[idx]
    return arr_max


cdef void c_cumpow(float32* arr_in, float32* arr_out, index_t N, float32 exp) noexcept nogil:
    cdef:
        index_t i = 0
        float32 sum = <float32>0

    for i in range(N):
        sum += arr_in[i]
        arr_out[i] = c_pow(sum, exp)
