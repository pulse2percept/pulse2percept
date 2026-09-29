# -*- coding: utf-8 -*-
"""
===============================================================================
Horsager et al. (2009): Temporal model of detection threshold
===============================================================================

Horsager et al. (2009) measured detection thresholds in Argus I users while
varying pulse duration and pulse-train frequency. Their model describes these
effects with a cascade of linear filters followed by a nonlinearity.

The stimulus current :math:`A(t)` first passes through a fast leaky integrator
:math:`R_1`. Accumulated charge :math:`C(t)` is filtered more slowly into
:math:`R_2` and subtracted from the response:

.. math::

    \\tau_1 \\frac{dR_1}{dt} &= -A(t) - R_1(t), \\\\
    \\tau_2 \\frac{dR_2}{dt} &= C(t) - R_2(t),
    \\qquad \\frac{dC}{dt} = \\max[A(t), 0], \\\\
    R_3(t) &= \\max\\left[R_1(t) - \\epsilon R_2(t),\\, 0\\right]^\\beta.

Three identical leaky integrators with time constant :math:`\\tau_3` then
smooth :math:`R_3` to produce brightness :math:`B(t)`. The default time
constants are :math:`\\tau_1 = 0.42` ms, :math:`\\tau_2 = 45.25` ms, and
:math:`\\tau_3 = 26.25` ms.

The model captures lower current thresholds for longer pulses and higher pulse
frequencies. Model parameters and the unit conversion of :math:`\\epsilon`
are documented in
:py:class:`~pulse2percept.models.retina.Horsager2009Temporal`.

This example reproduces Figs. 3B and 4B of [Horsager2009]_ for subject S05,
electrode C3.
"""
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from pulse2percept.datasets import load_horsager2009
from pulse2percept.models.retina import Horsager2009Temporal
from pulse2percept.stimuli import BiphasicPulse, BiphasicPulseTrain


###############################################################################
# Temporal response
# -----------------
#
# ``Horsager2009Temporal`` models the temporal response only and therefore
# does not require an implant. The example below shows the response to a
# single cathodic-first biphasic pulse.

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
# Detection threshold
# -------------------
#
# Behavioral threshold is the stimulus amplitude detected on 50% of trials.
# [Horsager2009]_ represents this as the amplitude for which the peak model
# response reaches a fitted criterion :math:`\theta`. The corresponding
# amplitude can be found with a one-dimensional root search.


def threshold_amp(make_stim, theta, amp_range=(0, 300)):
    """Amplitude (uA) whose predicted max brightness matches ``theta``."""
    def objective(amp):
        stim = make_stim(amp)
        percept = model.predict_percept(
            stim, t_percept=np.arange(stim.duration))
        return percept.data.max() - theta

    return brentq(objective, *amp_range)


###############################################################################
# Pulse duration
# --------------
#
# Figure 3B varies pulse duration. Each condition uses its measured pulse
# duration, interphase duration, and fitted detection criterion.

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
plt.title('Fig. 3B: S05, electrode C3');

###############################################################################
# The model reproduces the measured dependence of threshold on pulse duration
# using the published :math:`\theta` values.
#
# Limitations
# -----------
#
# * The model is temporal only and does not predict phosphene location or shape.
# * :math:`\theta` and the filter parameters were fit to individual subjects
#   and electrodes. Predictions for a new electrode require corresponding
#   parameter estimates.
# * The data were collected in a small number of Argus I users using
#   cathodic-first stimulation.
# * The model was fit to detection thresholds; predicted values above threshold
#   should not be interpreted as a validated brightness scale.
# * [Horsager2009]_ also tested bursting triplets
#   (:py:class:`~pulse2percept.stimuli.BiphasicTripletTrain`), variable-duration
#   trains (``n_pulses``), and latent addition using appended
#   :py:class:`~pulse2percept.stimuli.MonophasicPulse` objects.