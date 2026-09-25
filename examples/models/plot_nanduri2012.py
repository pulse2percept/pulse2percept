# -*- coding: utf-8 -*-
"""
===============================================================================
Nanduri et al. (2012): Amplitude and frequency are not interchangeable
===============================================================================

[Nanduri2012]_ had Argus I users rate brightness and size while the
amplitude or the frequency of a 0.45 ms cathodic-first pulse train was varied
from a reference of 1.25x threshold at 20 Hz. **Brightness saturated with
amplitude but kept growing with frequency; size grew mainly with amplitude.**

This example reproduces the frequency result with
:py:class:`~pulse2percept.models.retina.Nanduri2012Model` and recreates
Figs. 7 and 8. The amplitude result cannot be compared quantitatively: the
model's amplitude nonlinearity spans about a factor of 3.5 in current, the
experiment a factor of 6, so the predicted amplitude curve depends on the
assumed threshold rather than on the ratings.
"""
# sphinx_gallery_thumbnail_number = 3

import matplotlib.pyplot as plt
import numpy as np

from pulse2percept.datasets import load_nanduri2012
from pulse2percept.implants import DiskElectrode, ElectrodeArray, Implant
from pulse2percept.models.retina import Nanduri2012Model
from pulse2percept.stimuli import BiphasicPulseTrain

###############################################################################
# The stimulated electrode
# ------------------------
#
# The experiment stimulated one Argus I electrode at a time, so the
# simulation uses a single :py:class:`~pulse2percept.implants.DiskElectrode`.
# Five of the eight rated electrodes are 500 um across, so the radius is
# 250 um. The radius sets phosphene size, not brightness under the electrode.
#
# ``AMP_TH`` is an assumed threshold, not a published value. At ``z = 0`` the
# current-spread term is 1, so the model receives the full electrode current.

AMP_TH = 30       # assumed threshold current (uA)
PHASE_DUR = 0.45  # cathodic/anodic phase duration (ms)
STIM_DUR = 500    # stimulus duration (ms), as in the experiment

implant = Implant(ElectrodeArray(DiskElectrode(0, 0, 0, 250)))

###############################################################################
# Brightness over time
# --------------------
#
# The model is a cascade of linear filters and a static nonlinearity. At a
# single point, brightness rises within about 100 ms, then fades. Below,
# "brightness" means the maximum of this time course.

model = Nanduri2012Model(implant=implant, xrange=(0, 0), yrange=(0, 0))

stim = BiphasicPulseTrain(20, AMP_TH, PHASE_DUR, interphase_dur=PHASE_DUR,
                          stim_dur=STIM_DUR)
percept = model.predict_percept(stim, t_percept=np.arange(STIM_DUR))
delivered = implant.prepare_stim(stim)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(delivered.time,
        -0.02 + 0.01 * delivered.data[0, :] / delivered.data.max(),
        linewidth=2, label='pulse train')
ax.plot(percept.time, percept.data[0, 0, :], linewidth=2, label='percept')
ax.axhline(percept.data.max(), color='k', linestyle='--',
           label='max brightness')
ax.axhline(0, color='k')
ax.set_xlabel('time (ms)')
ax.set_ylabel('predicted brightness (a.u.)')
ax.set_xlim(0, STIM_DUR)
ax.legend(loc='center right')
fig.tight_layout()

###############################################################################
# The brightness asymmetry
# ------------------------
#
# The dataset holds the ratings for both modulations, relative to the
# 1.25xTh / 20 Hz reference. The model is run on the same conditions:

data = load_nanduri2012(task='rate')
amp_rows = data[data.varied_param == 'amp']
freq_rows = data[data.varied_param == 'freq']

amp_factors = sorted(amp_rows.amp_factor.unique())
freqs = sorted(freq_rows.freq.unique())


def brightness(amp_factor, freq):
    """Peak predicted brightness for one pulse-train condition"""
    train = BiphasicPulseTrain(freq, amp_factor * AMP_TH, PHASE_DUR,
                               interphase_dur=PHASE_DUR, stim_dur=STIM_DUR)
    return model.predict_percept(train).data.max()


reference = brightness(1.25, 20)
model_amp = np.array([brightness(f, 20) for f in amp_factors]) / reference
model_freq = np.array([brightness(1.25, f) for f in freqs]) / reference

###############################################################################
# Ratings and model output have unrelated scales, so both are normalized to
# the reference and plotted in separate rows. Compare curve shapes, not gains:

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey='row')

for electrode, group in amp_rows.groupby('electrode'):
    group = group.sort_values('amp_factor')
    ref = group[group.amp_factor == 1.25].brightness.values[0]
    axes[0, 0].plot(group.amp_factor, group.brightness / ref, 'o-',
                    color='0.6', markersize=4, linewidth=1)
for electrode, group in freq_rows.groupby('electrode'):
    group = group.sort_values('freq')
    ref = group[group.freq == 20].brightness.values[0]
    axes[0, 1].plot(group.freq, group.brightness / ref, 'o-',
                    color='0.6', markersize=4, linewidth=1)

axes[1, 0].plot(amp_factors, model_amp, 'ko-', linewidth=2)
axes[1, 0].axvspan(2, 6, color='0.9', zorder=0)
axes[1, 0].text(2.2, 0.92, 'nonlinearity saturated', fontsize=8, color='0.4',
                transform=axes[1, 0].get_xaxis_transform())
axes[1, 1].plot(freqs, model_freq, 'ko-', linewidth=2)

axes[0, 0].set_title('amplitude modulation (20 Hz)')
axes[0, 1].set_title('frequency modulation (1.25xTh)')
axes[0, 0].set_ylabel('rated brightness\n(re reference)')
axes[1, 0].set_ylabel('predicted brightness\n(re reference)')
axes[1, 0].set_xlabel('amplitude (xTh)')
axes[1, 1].set_xlabel('frequency (Hz)')
fig.tight_layout()

###############################################################################
# **Frequency** is reproduced. The model predicts
# 0.79 / 1.00 / 1.20 / 1.84 / 3.47 / 5.00 at 15-120 Hz; the measured means are
# 0.76 / 1.00 / 1.12 / 1.72 / 2.39 / 3.03 (SD 0.15-1.38 across eight
# electrodes). The model is steeper but within the spread of the data. This
# curve is independent of ``AMP_TH``, because the model's gain is normalized
# by the peak of its own fast response.
#
# **Amplitude** is not reproduced; the flat curve does not confirm
# saturation. The nonlinearity is a logistic in the peak fast response (about
# 0.66x the current for a 0.45 ms phase; midpoint 16, slope 3), which spans
# roughly 12-45 uA from threshold to saturation. At ``AMP_TH = 30`` uA the
# sweep starts 86% saturated and is fully clipped by 2xTh. The predicted
# 6xTh / 1.25xTh ratio depends only on the assumed threshold (38.6 at 5 uA,
# 6.5 at 15 uA, 2.7 at 20 uA, 1.5 at 25 uA, 1.15 at 30 uA, 1.0 at 40 uA);
# the measured ratio is 1.83 +/- 0.67. No threshold reproduces the measured
# shape.
#
# Phosphene size (Fig. 7)
# -----------------------
#
# Size requires a spatial grid, so the model is rebuilt over an 8 x 8 dva
# patch. The conditions are those of Fig. 7:

model = Nanduri2012Model(implant=implant, step=0.5, xrange=(-4, 4),
                         yrange=(-4, 4))

t_percept = np.arange(0, STIM_DUR, 1)
fig7_amps = [1, 1.25, 1.5, 2, 4, 6]
fig7_freqs = [40.0 / 3, 20, 2.0 * 40 / 3, 40, 80, 120]

frames_amp = [model.predict_percept(
    BiphasicPulseTrain(20, a * AMP_TH, PHASE_DUR, interphase_dur=PHASE_DUR,
                       stim_dur=STIM_DUR),
    t_percept=t_percept).max(axis='frames') for a in fig7_amps]

frames_freq = [model.predict_percept(
    BiphasicPulseTrain(f, 1.25 * AMP_TH, PHASE_DUR, interphase_dur=PHASE_DUR,
                       stim_dur=STIM_DUR),
    t_percept=t_percept).max(axis='frames') for f in fig7_freqs]

fig, axes = plt.subplots(nrows=2, ncols=len(fig7_amps), figsize=(14, 5))
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
# Size vs brightness (Fig. 8)
# ---------------------------
#
# Suprathreshold area against brightness separates the two modulations:
# amplitude increases area, frequency mostly does not.

bright_th = brightness(1, 20)

plt.figure()
plt.plot([np.max(frame) for frame in frames_amp],
         [np.sum(frame >= bright_th) for frame in frames_amp],
         'o-', label='amplitude modulation')
plt.plot([np.max(frame) for frame in frames_freq],
         [np.sum(frame >= bright_th) for frame in frames_freq],
         'o-', label='frequency modulation')
plt.xlabel('brightness (a.u.)')
plt.ylabel('area (# suprathreshold pixels)')
plt.legend()

###############################################################################
# What this does not establish
# ----------------------------
#
# * Model brightness is in arbitrary units, not a rating scale. Compare only
#   within one figure.
# * ``AMP_TH = 30`` uA is an assumption; the amplitude figure reflects it,
#   not the data. Raising the electrode above the retina rescales the current
#   reaching the nonlinearity but cannot widen it, so no geometry recovers the
#   measured amplitude curve.
# * Argus I thresholds vary by more than 10x across electrodes and subjects;
#   the dataset does not report thresholds for the eight pooled electrodes.
# * The rating task had 1 subject and 8 electrodes. The asymmetry is a
#   group-level trend, not a per-electrode prediction.
# * Area is counted in model pixels above the reference brightness, not the
#   drawn phosphene size.
