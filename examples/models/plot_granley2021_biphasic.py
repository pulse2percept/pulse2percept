# -*- coding: utf-8 -*-
"""
===============================================================================
Granley et al. (2021): Pulse parameters shape phosphene appearance
===============================================================================

The axon map model of [Beyeler2019]_ predicts one phosphene shape per
electrode, regardless of the pulse train. [Granley2021]_ adds three scaling
factors, fit to psychophysical and electrophysiological data: amplitude,
frequency, and phase duration modulate
brightness (:math:`F_\\mathrm{bright}`), spatial extent
(:math:`F_\\mathrm{size}`, scaling :math:`\\rho`), and streak length
(:math:`F_\\mathrm{streak}`, scaling :math:`\\lambda`).

This example qualitatively recreates Fig. 3 of [Granley2021]_. Each row
sweeps one pulse parameter on one Argus II electrode:

* **amplitude** makes the phosphene brighter *and* larger,
* **frequency** makes it brighter but not larger,
* **phase duration** shortens the axonal streak.
"""
import matplotlib.pyplot as plt

from pulse2percept.implants.retina import ArgusII
from pulse2percept.models.retina import BiphasicAxonMapModel
from pulse2percept.stimuli import BiphasicPulseTrain
from pulse2percept.units import xTh

###############################################################################
# One electrode, one model
# ------------------------
#
# ``rho`` and ``lam`` (um) are the baseline spread across and along axons.
# ``lam = 800`` um (longer than in the paper) elongates the baseline phosphene
# to about 2.5:1, so the streak effect in the bottom row is visible. The grid
# covers the region electrode A4 of a right-eye Argus II projects to.

ELECTRODE = 'A4'
BASE_FREQ = 5     # Hz
BASE_AMP = 1      # xTh
BASE_PDUR = 0.45  # ms

model = BiphasicAxonMapModel(implant=ArgusII(), rho=200, lam=800,
                             xrange=(-10.5, 1.5), yrange=(-2, 10), step=0.1)
model.build()


def predict(freq, amp, pdur):
    """Brightest frame of the percept for one biphasic pulse train"""
    train = BiphasicPulseTrain(freq, amp * xTh, pdur)
    return model.predict_percept({ELECTRODE: train}).data[..., 0]


###############################################################################
# The three sweeps
# ----------------
#
# Amplitude is in multiples of perceptual threshold (``xTh``), defined at
# 0.45 ms phase duration.
#
# In this model, threshold falls as phase duration grows (``a0 * pdur + a1``,
# Eq. 3): a fixed ``1 xTh`` at 100 ms would be ~200x the threshold-scaled
# amplitude at 0.45 ms and saturate. The phase-duration row therefore divides
# amplitude by the same factor, so phase duration acts only through
# :math:`F_\mathrm{streak}`.

AMPS = [1, 2, 3, 4, 5, 6]              # xTh, at 5 Hz / 0.45 ms
FREQS = [5, 10, 20, 40, 80, 120]       # Hz, at 1 xTh / 0.45 ms
PDURS = [0.1, 1, 5, 25, 50, 100]       # ms, at 5 Hz, amplitude compensated

scale = model.spatial.bright_model.scale_threshold
pdur_amps = [BASE_AMP * scale(BASE_PDUR) / scale(pdur) for pdur in PDURS]

rows = [
    ('Increasing amplitude',
     [f'{a:g}' + r'$\times$Th' for a in AMPS],
     [predict(BASE_FREQ, a, BASE_PDUR) for a in AMPS]),
    ('Increasing frequency',
     [f'{f:g} Hz' for f in FREQS],
     [predict(f, BASE_AMP, BASE_PDUR) for f in FREQS]),
    ('Increasing phase duration',
     [f'{t:g} ms' for t in PDURS],
     [predict(BASE_FREQ, a, t) for a, t in zip(pdur_amps, PDURS)]),
]

###############################################################################
# All 18 panels share one gray scale, ``[0, vmax]``, with ``vmax`` the
# brightest pixel in the figure (the 120 Hz panel, ~9x the 5 Hz reference).
# Per-panel autoscaling would make every panel equally bright.

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
# Amplitude (top) increases size and brightness. Frequency (middle) increases
# brightness only; the outline is identical across the row. Phase duration
# (bottom), with brightness and width held fixed, shortens the streak by about
# half between 0.1 and 100 ms.
#
# What this does not establish
# ----------------------------
#
# * The three factors are phenomenological fits to a few Argus I/II subjects,
#   not a biophysical model. They are linear or single power laws and
#   extrapolate poorly beyond the ranges swept here.
# * v0.11 uses an Argus II refit of the [Horsager2009]_ phase-duration
#   threshold relation, so the bottom row does not exactly reproduce the
#   published panel. The compensating amplitudes come from the model's own
#   ``scale_threshold``.
# * Brightness is in arbitrary units; compare only within this figure.
# * With several electrodes active, the model sums linearly; real
#   multi-electrode percepts do not.
