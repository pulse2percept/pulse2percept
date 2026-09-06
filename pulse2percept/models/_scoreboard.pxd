# Declared so the retinal axon-map kernel can reuse the same
# electrode-activity pass; see ``retina/_beyeler2019.pyx``.
cimport numpy as cnp

ctypedef cnp.float32_t float32

cdef cnp.uint8_t[::1] _active_electrodes(const float32[:, ::1] stim)
