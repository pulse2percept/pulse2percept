# -*- coding: utf-8 -*-
"""
===============================================================================
Granley et al. (2023): Oriented Gaussian phosphenes
===============================================================================

Granley et al. (2023) used a simple phosphene model for human-in-the-loop
optimization (HILO) of stimulus encoders. Each stimulated electrode produces an
elliptical Gaussian centered on the electrode. Unlike
:py:class:`~pulse2percept.models.retina.BiphasicAxonMapModel`, activation does
not spread along the axon: the local nerve fiber bundle sets only the
ellipse's orientation.

For amplitude :math:`\\tilde{a}` (multiples of threshold), frequency :math:`f`
(Hz), and phase duration :math:`t` (ms), each phosphene has

.. math::

    \\mathrm{brightness} &= a_0 \\tilde{a}^{a_1} + a_2 f, \\\\
    \\mathrm{area} &= \\max(\\rho a_3 \\tilde{a}, 1), \\\\
    \\mathrm{eccentricity} &= \\min(\\lambda (t / 0.45)^{a_4}, 0.99).

Amplitudes at or below ``amp_cutoff`` (0.25 xTh) produce no phosphene. Area
and eccentricity describe the ellipse at ``thresh_percept`` (:math:`e^{-2}`) of
peak brightness. The area is in output pixels: changing ``step`` changes the
phosphene's angular size.

The example varies one pulse parameter at a time on a single Argus II
electrode. Amplitude changes brightness and size, frequency changes brightness
only, and phase duration changes elongation only.
"""
import matplotlib.pyplot as plt

from pulse2percept.implants.retina import ArgusII
from pulse2percept.models.retina import Granley2023Model
from pulse2percept.stimuli import BiphasicPulseTrain
from pulse2percept.units import xTh


###############################################################################
# Model setup
# -----------
#
# The grid covers the region of electrode A4 in a right-eye Argus II, which
# sits on obliquely running nerve fiber bundles.

ELECTRODE = 'A4'
BASE_FREQ = 20     # Hz
BASE_AMP = 2       # xTh
BASE_PDUR = 0.45   # ms

model = Granley2023Model(ArgusII(), xrange=(-8, 2), yrange=(0, 10), step=0.1,
                         rho=400)
model.build()


def predict(freq, amp, pdur):
    """Return the percept for one biphasic pulse train."""
    train = BiphasicPulseTrain(freq, amp * xTh, pdur)
    return model.predict_percept({ELECTRODE: train}).data[..., 0]


###############################################################################
# Parameter sweeps
# ----------------
#
# ``rho = 400`` pixels at ``step = 0.1`` dva is an area of 4 deg² at 2 xTh.

AMPS = [0.25, 0.5, 1, 2, 4, 6]          # xTh, at 20 Hz / 0.45 ms
FREQS = [5, 10, 20, 40, 80, 120]        # Hz, at 2 xTh / 0.45 ms
PDURS = [0.1, 0.45, 1, 2, 4, 8]         # ms, at 20 Hz / 2 xTh

rows = [
    ('Amplitude',
     [f'{a:g}' + r'$\times$Th' for a in AMPS],
     [predict(BASE_FREQ, a, BASE_PDUR) for a in AMPS]),
    ('Frequency',
     [f'{f:g} Hz' for f in FREQS],
     [predict(f, BASE_AMP, BASE_PDUR) for f in FREQS]),
    ('Phase duration',
     [f'{t:g} ms' for t in PDURS],
     [predict(BASE_FREQ, BASE_AMP, t) for t in PDURS]),
]


###############################################################################
# One gray scale for all 18 panels, so brightness is comparable across panels:

vmax = max(frame.max() for _, _, frames in rows for frame in frames)

fig, axes = plt.subplots(3, 6, figsize=(11, 6.5), facecolor='k')
for row_axes, (label, titles, frames) in zip(axes, rows):
    for ax, title, frame in zip(row_axes, titles, frames):
        ax.imshow(frame, cmap='gray', vmin=0, vmax=vmax)
        ax.set_title(title, color='w', fontsize=10, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
    row_axes[0].set_ylabel(label, color='w', fontsize=11)
fig.tight_layout()


###############################################################################
# 0.25 xTh is at the cutoff and produces no phosphene. Short phase durations
# give more elongated phosphenes, clipped at eccentricity 0.99; the
# orientation does not change.
#
# Limitations
# -----------
#
# * The coefficients were fit for the HILO simulations of Granley et al.
#   (2023), not to individual patients.
# * Amplitude is used as given in multiples of threshold; there is no
#   phase-duration threshold correction.
# * Multi-electrode stimulation is modeled by linear summation.
