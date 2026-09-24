# -*- coding: utf-8 -*-
"""
===============================================================================
Horsager et al. (2009): Temporal sensitivity of the epiretinal percept
===============================================================================

Detection threshold depends on how charge is delivered in time.
[Horsager2009]_ measured thresholds in Argus I users while varying pulse
duration and pulse-train frequency, and fit a cascade of linear filters and a
nonlinearity.

This example reproduces Figs. 3B and 4B: threshold current against pulse
duration and against frequency, for subject S05, electrode C3.
"""
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from pulse2percept.datasets import load_horsager2009
from pulse2percept.models.retina import Horsager2009Temporal
from pulse2percept.stimuli import BiphasicPulse, BiphasicPulseTrain

###############################################################################
# What the model does
# -------------------
#
# The model is temporal only, so it needs no implant. Brightness rises over
# about 100 ms and decays over seconds; this integration window makes
# threshold depend on pulse timing.
#
# A single cathodic-first biphasic pulse, 0.075 ms per phase, 180 uA:

model = Horsager2009Temporal()
model.build()

stim_dur = 200
pulse = BiphasicPulse(180, 0.075, interphase_dur=0.075, stim_dur=stim_dur,
                      cathodic_first=True)
percept = model.predict_percept(pulse, t_percept=np.arange(stim_dur))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(pulse.time, -20 + 10 * pulse.data[0, :] / pulse.data.max(),
        linewidth=2, label='pulse')
ax.plot(percept.time, percept.data[0, 0, :], linewidth=2, label='percept')
ax.axhline(percept.data.max(), color='k', linestyle='--',
           label='max brightness')
ax.axhline(0, color='k')
ax.set_xlabel('time (ms)')
ax.set_ylabel('predicted brightness (a.u.)')
ax.set_xlim(0, stim_dur)
ax.legend(loc='center right')
fig.tight_layout()

###############################################################################
# Defining threshold
# ------------------
#
# Behavioral threshold is the amplitude detected on 50% of trials.
# [Horsager2009]_ equates it with the amplitude at which the peak model
# response reaches a constant :math:`\theta`, fit per subject and electrode
# and included in the dataset. Threshold is then a 1D root search over
# amplitude:


def threshold_amp(make_stim, theta, amp_range=(0, 300)):
    """Amplitude (uA) whose predicted max brightness matches ``theta``"""
    def objective(amp):
        stim = make_stim(amp)
        percept = model.predict_percept(stim,
                                        t_percept=np.arange(stim.duration))
        return percept.data.max() - theta
    return brentq(objective, *amp_range)


###############################################################################
# Threshold vs pulse duration (Fig. 3B)
# -------------------------------------
#
# Longer pulses need less current. Each condition is simulated at its
# published pulse and interphase duration:

single_pulse = load_horsager2009(subjects='S05', electrodes='C3',
                                 stim_types='single_pulse')

amp_th = []
for _, row in single_pulse.iterrows():
    def pulse_at(amp, row=row):
        return BiphasicPulse(amp, row['pulse_dur'],
                             interphase_dur=row['interphase_dur'],
                             stim_dur=row['stim_dur'],
                             cathodic_first=True)

    amp_th.append(threshold_amp(pulse_at, row['theta']))

plt.figure()
plt.semilogx(single_pulse.pulse_dur, single_pulse.stim_amp, 's', label='data')
plt.semilogx(single_pulse.pulse_dur, amp_th, 'k-', linewidth=2, label='model')
plt.xticks([0.1, 1, 4], ['0.1', '1', '4'])
plt.xlabel('pulse duration (ms)')
plt.ylabel('threshold current (uA)')
plt.legend()
plt.title('Fig. 3B: S05, electrode C3')

###############################################################################
# Threshold vs frequency (Fig. 4B)
# --------------------------------
#
# At fixed train duration, higher frequencies deliver more pulses into the
# integration window, so less current per pulse is needed. The stimulus is now
# a :py:class:`~pulse2percept.stimuli.BiphasicPulseTrain`:

fixed_dur = load_horsager2009(subjects='S05', electrodes='C3',
                              stim_types='fixed_duration')
fixed_dur = fixed_dur[fixed_dur.pulse_dur == 0.075]

amp_th = []
for _, row in fixed_dur.iterrows():
    def train_at(amp, row=row):
        return BiphasicPulseTrain(row['stim_freq'], amp, row['pulse_dur'],
                                  interphase_dur=row['interphase_dur'],
                                  stim_dur=row['stim_dur'],
                                  cathodic_first=True)

    amp_th.append(threshold_amp(train_at, row['theta']))

plt.figure()
plt.semilogx(fixed_dur.stim_freq, fixed_dur.stim_amp, 's', label='data')
plt.semilogx(fixed_dur.stim_freq, amp_th, 'k-', linewidth=2, label='model')
plt.xticks([5, 15, 75, 225], ['5', '15', '75', '225'])
plt.xlabel('frequency (Hz)')
plt.ylabel('threshold current (uA)')
plt.legend()
plt.title('Fig. 4B: S05, electrode C3, 0.075 ms pulses')

###############################################################################
# Both curves follow the measured thresholds over about two orders of
# magnitude, using the published :math:`\theta` without further fitting.
#
# What this does not establish
# ----------------------------
#
# * Temporal only: no phosphene location or shape.
# * :math:`\theta` and the filter parameters were fit per subject and
#   electrode; thresholds for a new electrode require comparable data.
# * Measured on Argus I, cathodic-first pulses, few subjects. Brightness
#   above threshold is not modeled here.
# * The paper's other conditions use the same procedure: bursting triplets
#   (:py:class:`~pulse2percept.stimuli.BiphasicTripletTrain`), variable
#   duration trains (``n_pulses``), and latent addition (appended
#   :py:class:`~pulse2percept.stimuli.MonophasicPulse` objects).
