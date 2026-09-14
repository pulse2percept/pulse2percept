# -*- coding: utf-8 -*-
"""
===============================================================================
Granley et al. (2021): Pulse parameters shape phosphene appearance
===============================================================================

The axon map model of [Beyeler2019]_ predicts one phosphene shape per
electrode, no matter how that electrode is driven. [Granley2021]_ adds three
stimulus-dependent scaling factors on top of it, fit to psychophysical and
electrophysiological data: amplitude, frequency, and phase duration modulate
brightness (:math:`F_\\mathrm{bright}`), spatial extent
(:math:`F_\\mathrm{size}`, scaling :math:`\\rho`), and streak length
(:math:`F_\\mathrm{streak}`, scaling :math:`\\lambda`).

This example is a qualitative recreation of Fig. 3 in [Granley2021]_ with the
current pulse2percept parameterization. Each row sweeps one pulse parameter on
a single Argus II electrode:

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
# ``rho`` and ``lam`` are the baseline spatial decay away from the axon and
# along it. ``lam = 800`` um gives a baseline phosphene elongated enough
# (about 2.5:1) for the streak effect in the bottom row to be visible; the
# published figure used a shorter ``lam``. The grid is cropped to the corner of
# the visual field that electrode A4 of a right-eye Argus II projects to.

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
# Amplitude is in multiples of perceptual threshold (``xTh``), where threshold
# is defined at 0.45 ms phase duration.
#
# The phase-duration row needs a correction. In this model, threshold falls as
# phase duration grows (``a0 * pdur + a1``, Eq. 3), so a fixed ``1 xTh`` at
# 100 ms would deliver ~200x the threshold-scaled amplitude of the 0.45 ms
# reference and the row would show one enormous saturated blob rather than a
# streak. Dividing the nominal amplitude by that same factor holds
# threshold-scaled amplitude constant, leaving phase duration to act only
# through :math:`F_\mathrm{streak}`.

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
# All 18 panels share one grayscale range, ``[0, vmax]``, with ``vmax`` the
# brightest pixel anywhere in the figure (the 120 Hz panel, ~9x the 5 Hz
# reference, which is why the other two rows sit at the dim end). This
# matters: ``Percept.plot()`` and Matplotlib both autoscale each image to its
# own min and max by default, which would make every panel below equally
# bright and erase the result.

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
# Amplitude (top) recruits a wider patch of retina and drives it harder, so the
# phosphene grows in both size and brightness. Frequency (middle) leaves
# :math:`F_\mathrm{size}` untouched: the outline is pixel-for-pixel identical
# across the row, only brighter. Phase duration (bottom) is the opposite case,
# with brightness and width held fixed by the amplitude compensation: the
# streak along the axon shortens by about half between 0.1 and 100 ms.
#
# What this does not establish
# ----------------------------
#
# * The three factors are phenomenological fits to a handful of Argus I/II
#   subjects, not a biophysical account of how pulse parameters drive ganglion
#   cells. They are linear (or single-power-law) in their arguments and
#   extrapolate poorly outside the ranges swept here.
# * v0.11 uses an Argus II refit of the [Horsager2009]_ phase-duration
#   threshold relation rather than the equation in the original publication, so
#   the bottom row is not a bit-for-bit reproduction of the published panel.
#   The compensating amplitudes here are derived from the current model's own
#   ``scale_threshold``, not copied from the paper.
# * Brightness is in arbitrary units. Only relative comparisons within this
#   figure are meaningful.
# * A single electrode is a best case. With many electrodes active, the
#   summation across electrodes in the model is linear, which real
#   multi-electrode percepts are not.
