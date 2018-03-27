import numpy as np
cimport numpy as np
import cython

cdef extern from "math.h":
    cpdef float expf(float x)

cdef inline float expit(float x):
    return 1.0 / (1.0 + expf(-x))

cdef inline float float_max(float a, float b):
    return a if a >= b else b

ctypedef double(*metric_ptr)(double[:], double[:])


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def nanduri2012_calc_layer_current(double[:] in_arr, double[:, ::1] pt_arr):
    cdef np.intp_t n_el = in_arr.shape[0]
    cdef np.intp_t n_time = pt_arr.shape[1]

    cdef double[:] pulse = np.zeros(pt_arr.shape[1])
    for el in range(n_el):
        with nogil:
            for t in range(n_time):
                pulse[t] += in_arr[el] * pt_arr[el, t]
    return pulse


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def nanduri2012_model_cascade(double[:] stimdata,
                              float dt, float tau1, float tau2,
                              float tau3, float asymptote, float shift,
                              float slope, float eps, float max_R3):
    """Cython implementation of the Nanduri 2012 model cascade"""
    cdef float tmp_ca = 0
    cdef float tmp_R1 = 0
    cdef float tmp_R2 = 0
    cdef float tmp_R3 = 0
    cdef float tmp_R4a = 0
    cdef float tmp_R4b = 0
    cdef float tmp_R4c = 0

    # Stationary nonlinearity: used to depend on future values of the
    # intermediary response, now has to be passed through `max_R3`
    # (because we can't look into the future):
    cdef float scale = asymptote * expit((max_R3 - shift) / slope)

    # Output array
    cdef np.intp_t arr_size = len(stimdata)
    cdef double[:] out_R4 = np.empty(arr_size, dtype=float)

    for i in range(arr_size):
        # Fast ganglion cell response:
        tmp_R1 += dt * (-stimdata[i] - tmp_R1) / tau1

        # Leaky integrated charge accumulation:
        tmp_ca += dt * float_max(stimdata[i], 0)
        tmp_R2 += dt * (tmp_ca - tmp_R2) / tau2
        tmp_R3 = float_max(tmp_R1 - eps * tmp_R2, 0) / max_R3 * scale

        # Slow response: 3-stage leaky integrator
        tmp_R4a += dt * (tmp_R3 - tmp_R4a) / tau3
        tmp_R4b += dt * (tmp_R4a - tmp_R4b) / tau3
        tmp_R4c += dt * (tmp_R4b - tmp_R4c) / tau3
        out_R4[i] = tmp_R4c
    return out_R4
