import cython
cimport numpy as np
import numpy as np


cdef extern from "math.h":
    cpdef float expf(float x)


cdef inline float expit(float x):
    return 1.0 / (1.0 + expf(x))

cdef inline float float_max(float a, float b):
    return a if a >= b else b

DTYPE = np.float
ctypedef np.float_t DTYPE_t


def nanduri2012_model_cascade(np.ndarray[DTYPE_t, ndim=1] stimdata,
                              float dt, float tau1, float tau2,
                              float tau3, float asymptote, float shift,
                              float slope, float eps, float max_R3):
    """Cython implementation of the Nanduri 2012 model cascade"""
    cdef float tmp_chargeacc = 0
    cdef float tmp_ca = 0
    cdef float tmp_cl = 0
    cdef float tmp_R1 = 0
    cdef float tmp_R2 = 0
    cdef float tmp_R3norm = 0
    cdef float tmp_R3 = 0
    cdef float sc_fac = 0
    tmp_R4a = [0, 0, 0, 0]
    cdef int arr_size = int(dt * stimdata.shape[-1])

    cdef np.ndarray[DTYPE_t] out_R4 = np.zeros(arr_size, dtype=DTYPE)

    for i in range(len(arr_size)):
        tmp_R1 += dt * (-stimdata[i] - tmp_R1) / tau1

        # Leaky integrated charge accumulation:
        tmp_chargeacc += dt * float_max(stimdata[i], 0)
        tmp_ca += dt * (tmp_chargeacc - tmp_ca) / tau2
        tmp_R3 = float_max(tmp_R1 - eps * tmp_ca, 0)

        # Stationary nonlinearity:
        sc_fac = asymptote * expit((max_R3 - shift) / slope)

        # R4: R3 passed through a cascade of 3 leaky integrators
        tmp_R4a[0] = tmp_R3b = tmp_R3 / max_R3 * sc_fac
        for j in range(3):
            dR4a = dt * (tmp_R4a[j] - tmp_R4a[j + 1]) / tau3
            tmp_R4a[j + 1] += dR4a

        out_R4[i] = tmp_R4a[-1]
    return out_R4
