# -*- coding: utf-8 -*-
"""
===============================================================================
Granley et al. (2021): Pulse-parameter effects
===============================================================================

The axon map model predicts a fixed phosphene shape for each electrode,
independent of the pulse train (Beyeler et al., 2019). Granley et al. (2021)
extend the model with three scaling factors, fit to psychophysical and
electrophysiological data, that relate amplitude, frequency, and phase duration
to phosphene brightness, size, and streak length.

The scaling factors are:

.. math::

    F_\\mathrm{bright} &= a_2 \\tilde{a} + a_3 f + a_4, \\\\
    F_\\mathrm{size} &= a_5 \\tilde{a} + a_6, \\\\
    F_\\mathrm{streak} &= a_9 - a_7 t^{a_8},

where :math:`\\tilde{a}` is amplitude in multiples of threshold, :math:`f` is
frequency (Hz), and :math:`t` is phase duration (ms). :math:`F_\\mathrm{bright}`
scales peak brightness. The spatial factors modify the decay constants of the
axon map model [Beyeler2019]_:

.. math::

    \\rho_\\mathrm{eff} &= \\rho \\sqrt{F_\\mathrm{size}}, \\\\
    \\lambda_\\mathrm{eff} &= \\lambda \\sqrt{F_\\mathrm{streak}}.

The model also applies lower bounds ``min_rho`` and ``min_lambda``. The fitted
coefficients are documented in
:py:class:`~pulse2percept.models.retina.granley2021.DefaultBrightModel`,
:py:class:`~pulse2percept.models.retina.granley2021.DefaultSizeModel`, and
:py:class:`~pulse2percept.models.retina.granley2021.DefaultStreakModel`.
For phosphene size, pulse2percept uses an Argus II refit of the threshold
correction rather than Eq. 5 of [Granley2021]_.

The example qualitatively reproduces Fig. 3 of [Granley2021]_ by varying one
pulse parameter at a time on a single Argus II electrode. Increasing amplitude
increases brightness and size, increasing frequency increases brightness, and
increasing phase duration shortens the axonal streak.
"""
import matplotlib.pyplot as plt

from pulse2percept.implants.retina import ArgusII
from pulse2percept.models.retina import BiphasicAxonMapModel
from pulse2percept.stimuli import BiphasicPulseTrain
from pulse2percept.units import xTh


###############################################################################
# Model setup
# -----------
#
# ``rho`` and ``lam`` (um) set the baseline spread across and along axons.
# ``lam = 800`` um, longer than in the paper, produces a more elongated
# baseline phosphene so that the phase-duration effect is easier to see.
# The grid covers the region stimulated by electrode A4 in a right-eye
# Argus II.

ELECTRODE = 'A4'
BASE_FREQ = 5      # Hz
BASE_AMP = 1       # xTh
BASE_PDUR = 0.45   # ms

model = BiphasicAxonMapModel(
    implant=ArgusII(),
    rho=200,
    lam=800,
    xrange=(-10.5, 1.5),
    yrange=(-2, 10),
    step=0.1,
)
model.build()


def predict(freq, amp, pdur):
    """Return the brightest frame for one biphasic pulse train."""
    train = BiphasicPulseTrain(freq, amp * xTh, pdur)
    return model.predict_percept({ELECTRODE: train}).data[..., 0]


###############################################################################
# Parameter sweeps
# ----------------
#
# Amplitude is expressed in multiples of perceptual threshold (``xTh``),
# defined at a phase duration of 0.45 ms.
#
# Threshold varies with phase duration (``a0 * pdur + a1``, Eq. 3). Without
# compensation, a fixed ``1 xTh`` at long phase durations would therefore
# correspond to a much larger threshold-scaled amplitude. For the
# phase-duration sweep, amplitude is adjusted by the same threshold relation
# so that the row isolates the effect on streak length.

AMPS = [1, 2, 3, 4, 5, 6]              # xTh, at 5 Hz / 0.45 ms
FREQS = [5, 10, 20, 40, 80, 120]       # Hz, at 1 xTh / 0.45 ms
PDURS = [0.1, 1, 5, 25, 50, 100]       # ms, at 5 Hz, amplitude compensated

scale = model.spatial.bright_model.scale_threshold
pdur_amps = [BASE_AMP * scale(BASE_PDUR) / scale(pdur) for pdur in PDURS]

rows = [
    ('Amplitude',
     [f'{a:g}' + r'$\times$Th' for a in AMPS],
     [predict(BASE_FREQ, a, BASE_PDUR) for a in AMPS]),
    ('Frequency',
     [f'{f:g} Hz' for f in FREQS],
     [predict(f, BASE_AMP, BASE_PDUR) for f in FREQS]),
    ('Phase duration',
     [f'{t:g} ms' for t in PDURS],
     [predict(BASE_FREQ, a, t) for a, t in zip(pdur_amps, PDURS)]),
]


###############################################################################
# Use one gray scale for all 18 panels. Autoscaling each panel separately
# would obscure the modeled brightness differences.

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
# Amplitude increases phosphene size and brightness. Frequency changes
# brightness without changing the outline. With threshold compensation,
# increasing phase duration shortens the axonal streak.
#
# Limitations
# -----------
#
# * The scaling factors are phenomenological fits to data from a small number
#   of Argus I/II subjects, not a biophysical model. Extrapolation outside the
#   fitted stimulus ranges should therefore be treated cautiously.
# * v0.11 uses an Argus II refit of the [Horsager2009]_ phase-duration
#   threshold relation, so the bottom row does not exactly reproduce the
#   published panel. The compensating amplitudes use the model's own
#   ``scale_threshold``.
# * Brightness is in arbitrary units and is only comparable within this figure.
# * Multi-electrode stimulation is modeled by linear summation, which does not
#   capture the nonlinear percepts reported with simultaneous stimulation.