# -*- coding: utf-8 -*-
"""
===============================================================================
Generating pulse trains
===============================================================================

This example shows how to use :py:class:`~pulse2percept.stimuli.PulseTrain`
and its variants.

Biphasic pulse trains
---------------------

A series of biphasic pulses can be created with the
:py:class:`~pulse2percept.stimuli.BiphasicPulseTrain` class.

You have the same options as when setting up a single
:py:class:`~pulse2percept.stimuli.BiphasicPulse`, in addition to specifying
a pulse train frequency (``freq``) and total stimulus duration (``stim_dur``).

For example, a 20 Hz pulse train lasting 500 ms and made from anodic-first
biphasic pulses (30 uA, 0.075 ms pulse duration, no interphase gap) can be
created as follows:
"""
# sphinx_gallery_thumbnail_number = 3

from pulse2percept.stimuli import BiphasicPulseTrain

pt = BiphasicPulseTrain(20, 30, 0.075, stim_dur=500, cathodic_first=False)
pt.plot()

###############################################################################
# You can also limit the number of pulses in the train, but still make the
# stimulus last 500 ms:

pt = BiphasicPulseTrain(20, 30, 0.075, n_pulses=3, stim_dur=500,
                        cathodic_first=False)
pt.plot()

###############################################################################
# Asymmetric biphasic pulse trains
# --------------------------------
#
# To create a 20 Hz pulse train lasting 500 ms created from asymmetric biphasic
# pulses, use :py:class:`~pulse2percept.stimuli.AsymmetricBiphasicPulseTrain`:

from pulse2percept.stimuli import AsymmetricBiphasicPulseTrain

# First pulse:
amp1 = 10
phase_dur1 = 1

# Second pulse
amp2 = 2
phase_dur2 = 5

pt = AsymmetricBiphasicPulseTrain(20, amp1, amp2, phase_dur1, phase_dur2,
                                  stim_dur=500)
pt.plot()

###############################################################################
# Biphasic triplet trains
# -----------------------
#
# To create a train of pulse triplets, use
# :py:class:`~pulse2percept.stimuli.BiphasicTripletTrain`:

from pulse2percept.stimuli import BiphasicTripletTrain

amp = 15
phase_dur = 0.075

pt = BiphasicTripletTrain(20, amp, phase_dur, stim_dur=500)
pt.plot()

###############################################################################
# Generic pulse trains
# --------------------
#
# Finally, you can concatenate any :py:class:`~pulse2percept.stimuli.Stimulus`
# object into a pulse train.
#
# For example, let's define a single ramp stimulus:

import numpy as np
from pulse2percept.stimuli import Stimulus, PulseTrain

# Single ramp:
dt = 1e-6
ramp = Stimulus([[0, 0, 1, 1, 2, 2, 0, 0]],
                time=[0, 1, 1 + dt, 2, 2 + dt, 3, 3 + dt, 5])
ramp.plot()

# Ramp train:
PulseTrain(20, ramp, stim_dur=200).plot()

# Biphasic ramp:
biphasic_ramp = Stimulus(np.concatenate((ramp.data, -ramp.data), axis=1),
                         time=np.concatenate((ramp.time, ramp.time + 5)))
biphasic_ramp.plot()

# Biphasic ramp train:
PulseTrain(20, biphasic_ramp, stim_dur=200).plot()
