# -*- coding: utf-8 -*-
"""
===============================================================================
Nanduri et al. (2012): Amplitude and frequency effects on brightness and size
===============================================================================

Nanduri et al. (2012) measured phosphene brightness and size in Argus I users
while varying either the amplitude or frequency of a 0.45 ms cathodic-first
pulse train. Relative to a reference stimulus of 1.25x threshold at 20 Hz,
brightness saturated with amplitude but continued to increase with frequency;
phosphene size increased primarily with amplitude.

The model combines spatial current spread with a temporal cascade. Current
:math:`A_e(t)` from each disk electrode :math:`e` falls off with distance
:math:`d_e` from the electrode edge:

.. math::

    I(x, y, t) = \\sum_e A_e(t)
    \\frac{\\mathrm{atten\\_a}}
    {\\mathrm{atten\\_a} + d_e(x, y)^{\\mathrm{atten\\_n}}}.

At each retinal location, :math:`I` passes through a temporal model related to
[Horsager2009]_: a fast response :math:`R_1`, minus a slowly filtered charge
term :math:`R_2`, followed by half-wave rectification and three slow leaky
integrators. Before the final smoothing stage, a logistic nonlinearity rescales
the peak response:

.. math::

    \\mathrm{asymptote} \\cdot
    \\sigma\\left(
    \\frac{R_{3,\\max} - \\mathrm{shift}}{\\mathrm{slope}}
    \\right),
    \\qquad
    \\sigma(u) = \\frac{1}{1 + e^{-u}}.

Because the logistic operates on the peak fast response, brightness saturates
with pulse amplitude. Increasing frequency adds more responses within the
temporal integration window, allowing brightness to continue increasing.
Parameters are documented in
:py:class:`~pulse2percept.models.retina.Nanduri2012Spatial` and
:py:class:`~pulse2percept.models.retina.Nanduri2012Temporal`.

This example recreates Figs. 7 and 8 of [Nanduri2012]_ and compares the
predicted amplitude- and frequency-dependent brightness with the rating data.
The frequency comparison is quantitative. For amplitude, the dataset does not
provide thresholds for the rated electrodes, so the prediction depends on an
assumed threshold current.
"""
# sphinx_gallery_thumbnail_number = 3

import matplotlib.pyplot as plt
import numpy as np

from pulse2percept.datasets import load_nanduri2012
from pulse2percept.implants import DiskElectrode, ElectrodeArray, Implant
from pulse2percept.models.retina import Nanduri2012Model
from pulse2percept.stimuli import BiphasicPulseTrain


###############################################################################
# Stimulation setup
# -----------------
#
# The experiment stimulated one Argus I electrode at a time. Five of the eight
# rated electrodes were 500 um in diameter, so the simulation uses a single
# disk electrode with radius 250 um.
#
# ``AMP_TH`` is an assumed threshold because thresholds for the eight pooled
# electrodes are not reported in the dataset. At ``z = 0``, the current-spread
# term is 1 directly beneath the electrode.

AMP_TH = 30       # assumed threshold current (uA)
PHASE_DUR = 0.45  # cathodic/anodic phase duration (ms)
STIM_DUR = 500    # stimulus duration (ms), as in the experiment

implant = Implant(ElectrodeArray(DiskElectrode(0, 0, 0, 250)))


###############################################################################
# Temporal response
# -----------------
#
# Response at a single retinal location. Brightness is the maximum of the
# predicted temporal response:

model = Nanduri2012Model(implant=implant, xrange=(0, 0), yrange=(0, 0))

stim = BiphasicPulseTrain(
    20, AMP_TH, PHASE_DUR,
    interphase_dur=PHASE_DUR,
    stim_dur=STIM_DUR,
)
percept = model.predict_percept(stim, t_percept=np.arange(STIM_DUR))
delivered = implant.prepare_stim(stim)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(
    delivered.time,
    -0.02 + 0.01 * delivered.data[0, :] / delivered.data.max(),
    linewidth=2,
    label='pulse train',
)
ax.plot(percept.time, percept.data[0, 0, :], linewidth=2, label='percept')
ax.axhline(
    percept.data.max(),
    color='k',
    linestyle='--',
    label='max brightness',
)
ax.axhline(0, color='k')
ax.set_xlabel('time (ms)')
ax.set_ylabel('predicted brightness (a.u.)')
ax.set_xlim(0, STIM_DUR)
ax.legend(loc='center right')
fig.tight_layout()


###############################################################################
# Brightness ratings
# ------------------
#
# The rating data contain separate amplitude- and frequency-modulation
# conditions, both relative to the 1.25xTh / 20 Hz reference.

data = load_nanduri2012(task='rate')
amp_rows = data[data.varied_param == 'amp']
freq_rows = data[data.varied_param == 'freq']

amp_factors = sorted(amp_rows.amp_factor.unique())
freqs = sorted(freq_rows.freq.unique())


def brightness(amp_factor, freq):
    """Peak predicted brightness for one pulse-train condition."""
    train = BiphasicPulseTrain(
        freq,
        amp_factor * AMP_TH,
        PHASE_DUR,
        interphase_dur=PHASE_DUR,
        stim_dur=STIM_DUR,
    )
    return model.predict_percept(train).data.max()


reference = brightness(1.25, 20)
model_amp = np.array([brightness(f, 20) for f in amp_factors]) / reference
model_freq = np.array([brightness(1.25, f) for f in freqs]) / reference


###############################################################################
# Ratings and model output use different scales, so both are normalized to
# their reference condition; only curve shapes are comparable.

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey='row')

for electrode, group in amp_rows.groupby('electrode'):
    group = group.sort_values('amp_factor')
    ref = group[group.amp_factor == 1.25].brightness.values[0]
    axes[0, 0].plot(
        group.amp_factor,
        group.brightness / ref,
        'o-',
        color='0.6',
        markersize=4,
        linewidth=1,
    )

for electrode, group in freq_rows.groupby('electrode'):
    group = group.sort_values('freq')
    ref = group[group.freq == 20].brightness.values[0]
    axes[0, 1].plot(
        group.freq,
        group.brightness / ref,
        'o-',
        color='0.6',
        markersize=4,
        linewidth=1,
    )

axes[1, 0].plot(amp_factors, model_amp, 'ko-', linewidth=2)
axes[1, 0].axvspan(2, 6, color='0.9', zorder=0)
axes[1, 0].text(
    2.2,
    0.92,
    'nonlinearity saturated',
    fontsize=8,
    color='0.4',
    transform=axes[1, 0].get_xaxis_transform(),
)
axes[1, 1].plot(freqs, model_freq, 'ko-', linewidth=2)

axes[0, 0].set_title('amplitude modulation (20 Hz)')
axes[0, 1].set_title('frequency modulation (1.25xTh)')
axes[0, 0].set_ylabel('rated brightness\n(re reference)')
axes[1, 0].set_ylabel('predicted brightness\n(re reference)')
axes[1, 0].set_xlabel('amplitude (xTh)')
axes[1, 1].set_xlabel('frequency (Hz)')
fig.tight_layout()


###############################################################################
# Frequency modulation
# --------------------
#
# The model predicts relative brightness values of
# 0.79 / 1.00 / 1.20 / 1.84 / 3.47 / 5.00 at 15-120 Hz, compared with measured
# means of 0.76 / 1.00 / 1.12 / 1.72 / 2.39 / 3.03 (SD 0.15-1.38 across eight
# electrodes). The model response is steeper but follows the measured trend.
#
# This curve does not depend on ``AMP_TH`` because the model gain is normalized
# by the peak of its own fast response.


###############################################################################
# Amplitude modulation
# --------------------
#
# The amplitude curve is sensitive to ``AMP_TH`` and should not be treated as
# a quantitative reproduction of the ratings. The logistic nonlinearity acts
# on the peak fast response, approximately 0.66x the current for a 0.45 ms
# phase, with midpoint 16 and slope 3. It therefore spans only about 12-45 uA
# from threshold to saturation.
#
# With ``AMP_TH = 30`` uA, the sweep begins about 86% saturated and is fully
# saturated by 2xTh. The predicted 6xTh / 1.25xTh brightness ratio varies from
# 38.6 at a 5 uA threshold to 1.0 at 40 uA; the measured ratio is
# 1.83 +/- 0.67. No threshold reproduces the measured curve shape.


###############################################################################
# Phosphene size
# --------------
#
# Amplitude and frequency conditions from Fig. 7 on an 8 x 8 dva grid:

model = Nanduri2012Model(
    implant=implant,
    step=0.5,
    xrange=(-4, 4),
    yrange=(-4, 4),
)

t_percept = np.arange(0, STIM_DUR, 1)
fig7_amps = [1, 1.25, 1.5, 2, 4, 6]
fig7_freqs = [40.0 / 3, 20, 2.0 * 40 / 3, 40, 80, 120]

frames_amp = [
    model.predict_percept(
        BiphasicPulseTrain(
            20,
            a * AMP_TH,
            PHASE_DUR,
            interphase_dur=PHASE_DUR,
            stim_dur=STIM_DUR,
        ),
        t_percept=t_percept,
    ).max(axis='frames')
    for a in fig7_amps
]

frames_freq = [
    model.predict_percept(
        BiphasicPulseTrain(
            f,
            1.25 * AMP_TH,
            PHASE_DUR,
            interphase_dur=PHASE_DUR,
            stim_dur=STIM_DUR,
        ),
        t_percept=t_percept,
    ).max(axis='frames')
    for f in fig7_freqs
]

fig, axes = plt.subplots(
    nrows=2,
    ncols=len(fig7_amps),
    figsize=(14, 5),
)

for ax, amp, frame in zip(axes[0], fig7_amps, frames_amp):
    ax.imshow(frame, vmin=0, vmax=0.3, cmap='gray')
    ax.set_title(f'{amp:.2g}xTh / 20 Hz', fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
axes[0][0].set_ylabel('amplitude\nmodulation')

for ax, freq, frame in zip(axes[1], fig7_freqs, frames_freq):
    ax.imshow(frame, vmin=0, vmax=0.3, cmap='gray')
    ax.set_title(f'1.25xTh / {freq:.0f} Hz', fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
axes[1][0].set_ylabel('frequency\nmodulation')

fig.tight_layout()


###############################################################################
# Brightness and phosphene area
# -----------------------------
#
# Figure 8 compares suprathreshold area with peak brightness. Increasing
# amplitude increases both quantities, whereas frequency has little effect on
# area.

bright_th = brightness(1, 20)

plt.figure()
plt.plot(
    [np.max(frame) for frame in frames_amp],
    [np.sum(frame >= bright_th) for frame in frames_amp],
    'o-',
    label='amplitude modulation',
)
plt.plot(
    [np.max(frame) for frame in frames_freq],
    [np.sum(frame >= bright_th) for frame in frames_freq],
    'o-',
    label='frequency modulation',
)
plt.xlabel('brightness (a.u.)')
plt.ylabel('area (# suprathreshold pixels)')
plt.legend();


###############################################################################
# Limitations
# -----------
#
# * Model brightness is in arbitrary units and is not a perceptual rating
#   scale. Values should only be compared within a figure.
# * ``AMP_TH = 30`` uA is assumed, not measured, for these electrodes. The
#   amplitude prediction depends on this choice.
# * Changing electrode-retina distance rescales the current reaching the
#   nonlinearity but does not widen the nonlinearity itself, so electrode
#   geometry cannot recover the measured amplitude curve.
# * Argus I thresholds vary by more than an order of magnitude across subjects
#   and electrodes, and thresholds for the eight pooled electrodes are not
#   reported in the dataset.
# * The rating experiment included one subject and eight electrodes, so the
#   amplitude-frequency difference is a pooled trend, not a per-electrode
#   prediction.
# * Phosphene area is defined here as the number of model pixels above the
#   reference brightness, not the reported or drawn phosphene size.