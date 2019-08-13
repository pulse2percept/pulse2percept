import numpy as np
cimport numpy as np
from libc.math cimport(fmax as c_fmax, pow as c_pow)


def step_horsager2009(double dt, double amp, double ca, double r1, double r2,
                      double r4a, double r4b, double r4c, double tau1,
                      double tau2, double tau3, double epsilon, double beta):
    """Steps the temporal model"""
    cdef:
        double r3

    with nogil:
        # Although the paper says to use cathodic-first, the code only
        # reproduces if we use what we now call anodic-first. So flip the sign
        # on the stimulus here:
        r1 += dt * (-amp - r1) / tau1
        # It's possible that charge accumulation was done on the anodic phase.
        # It might not matter too much (timing is slightly different, but the
        # data are not accurate enough to warrant using one over the other).
        # Thus use what makes the most sense: accumulate on cathodic
        ca += dt * c_fmax(amp, 0)
        r2 += dt * (ca - r2) / tau2
        r3 = c_pow(c_fmax(r1 - epsilon * r2, 0.0), beta)

        # Slow response: 3-stage leaky integrator
        r4a += dt * (r3 - r4a) / tau3
        r4b += dt * (r4a - r4b) / tau3
        r4c += dt * (r4b - r4c) / tau3

    return ca, r1, r2, r4a, r4b, r4c
