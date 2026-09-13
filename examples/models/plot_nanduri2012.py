# -*- coding: utf-8 -*-
"""
===============================================================================
Nanduri et al. (2012): Amplitude and frequency are not interchangeable
===============================================================================

Turning a phosphene up can mean two things: more current per pulse, or more
pulses per second. [Nanduri2012]_ showed that the two are not equivalent.
Argus I users rated brightness and size while either the amplitude or the
frequency of a 0.45 ms cathodic-first pulse train was varied from a common
reference of 1.25x threshold at 20 Hz.

The reported result is an asymmetry: **brightness saturates with amplitude but
keeps growing with frequency, while phosphene size grows mainly with
amplitude.** This example reproduces that asymmetry with
:py:class:`~pulse2percept.models.retina.Nanduri2012Model`, and recreates
Figs. 7 and 8 of the paper.
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
# The experiment stimulated one Argus I disk electrode at a time. That is what
# the simulation represents: a single 260 um radius
# :py:class:`~pulse2percept.implants.DiskElectrode` at the array origin, rather
# than a named multi-electrode device. Amplitudes are expressed as multiples of
# that electrode's threshold, taken here as 30 uA.

AMP_TH = 30       # threshold current (uA)
PHASE_DUR = 0.45  # cathodic/anodic phase duration (ms)
STIM_DUR = 500    # stimulus duration (ms), as in the experiment

implant = Implant(ElectrodeArray(DiskElectrode(0, 0, 0, 260)))

###############################################################################
# Brightness over time
# --------------------
#
# The model is a cascade of linear filters and a stationary nonlinearity.
# Predicted at a single point, (0, 0), it produces the time course the paper
# describes: brightness rises within about 100 ms, then fades. "Brightness of a
# stimulus" below always means the maximum of that time course.

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
# The dataset holds the measured ratings for both modulation directions, each
# relative to the same 1.25xTh / 20 Hz reference. Re-simulating exactly those
# conditions gives the model's counterpart:

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
# Measured ratings and model predictions are on unrelated scales -- one is a
# subject's rating relative to a reference stimulus, the other is in arbitrary
# model units -- so they are shown normalized to that shared reference and in
# separate rows, each row on its own shared axis. What is being compared is
# the *shape* of each curve, not its gain:

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
axes[1, 1].plot(freqs, model_freq, 'ko-', linewidth=2)

axes[0, 0].set_title('amplitude modulation (20 Hz)')
axes[0, 1].set_title('frequency modulation (1.25xTh)')
axes[0, 0].set_ylabel('rated brightness\n(re reference)')
axes[1, 0].set_ylabel('predicted brightness\n(re reference)')
axes[1, 0].set_xlabel('amplitude (xTh)')
axes[1, 1].set_xlabel('frequency (Hz)')
fig.tight_layout()

###############################################################################
# Both rows show the same asymmetry: raising amplitude buys progressively less
# brightness, while raising frequency keeps buying it. The model is steeper in
# frequency and flatter in amplitude than the ratings, which is expected --
# the parameters here are the published defaults, not refit to these eight
# electrodes, and a subject's rating scale is not linear in model brightness.
#
# Phosphene size (Fig. 7)
# -----------------------
#
# Size needs space, so the model is rebuilt over a patch of visual field rather
# than a single point. The conditions are those of Fig. 7:

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
# Plotting suprathreshold area against brightness separates the two
# modulations: amplitude buys area, frequency mostly does not.

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
# * Model brightness is in arbitrary units and is not calibrated to a
#   psychophysical rating scale. Only relative comparisons within one figure
#   are meaningful.
# * The 30 uA threshold and the single 260 um disk electrode stand in for one
#   Argus I electrode. Thresholds vary by more than an order of magnitude
#   across electrodes and subjects.
# * [Nanduri2012]_ measured 1 subject on 8 electrodes in the rating task. The
#   asymmetry is a group-level trend, not a per-electrode prediction.
# * Area here is counted in model pixels above the reference brightness, which
#   is not the drawn phosphene size the subjects reported.
