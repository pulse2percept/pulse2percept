import numpy as np
cimport numpy as np
from libc.math cimport(fmax as c_fmax, exp as c_exp)


cdef inline double expit(double x):
    return 1.0 / (1.0 + c_exp(-x))


def fast_nanduri2012(double dt, double amp, double ca, double r1, double r2,
                     double r4a, double r4b, double r4c, double tau1,
                     double tau2, double tau3, double eps, double asymptote,
                     double shift, double slope, double max_r3):
    """Steps the temporal model"""
    cdef:
        double r3
        # Stationary nonlinearity: used to depend on future values of the
        # intermediary response, now has to be passed through `max_R3`
        # (because we can't look into the future):
        double scale = asymptote * expit((max_r3 - shift) / slope)

    with nogil:
        # Fast response:
        r1 += dt * (-amp - r1) / tau1
        # Leaky integrated charge accumulation:
        ca += dt * c_fmax(amp, 0)
        r2 += dt * (ca - r2) / tau2
        r3 = c_fmax(r1 - eps * r2, 0.0) / max_r3 * scale

        # Slow response: 3-stage leaky integrator
        r4a += dt * (r3 - r4a) / tau3
        r4b += dt * (r4a - r4b) / tau3
        r4c += dt * (r4b - r4c) / tau3

    return ca, r1, r2, r4a, r4b, r4c
