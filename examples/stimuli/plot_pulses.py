# -*- coding: utf-8 -*-
"""
============================================================================
Building stimuli
============================================================================
This example shows how to build and visualize basic types of stimuli such as
:py:class:`~pulse2percept.stimuli.MonophasicPulse`,
:py:class:`~pulse2percept.stimuli.BiphasicPulse` or a
:py:class:`~pulse2percept.stimuli.PulseTrain` for a given implant.

A monophasic pulse has a single phase and can be either anodic or cathodic. A
biphasic pulse is generally charge balanced for safety reasons and defined
as either anodic first or cathodic first.
Multiple pulses can form a pulsetrain.


"""
##############################################################################

# 1. Simplest Stimulus
# ---------------------
# :py:class:`~pulse2percept.stimuli.Stimulus` is the base class to generate
# different types of stimulus. Simplest way to instantiate a Stimulus is
# to pass a scalar value which is interpreted as the current amplitude
# for a single electrode.


from pulse2percept.stimuli import Stimulus

stim = Stimulus(10)

##############################################################################
# Parameters you don't specify will take on default values. You can inspect
# all current model parameters as follows:

print(stim)



##############################################################################
# 2. A MonophasicPulse
# --------------------
# Let's start by importing necessary modules
from pulse2percept.stimuli import MonophasicPulse

import matplotlib.pyplot as plt

##############################################################################
# Then we can specify the arguments of the monophasic pulse

pulse_type = 'anodic'  # whether current has a positive or negative amplitude
pulse_dur = 0.0046  # in seconds
time_sample = 0.1 / 1000  # temporal sampling in seconds

##############################################################################
# By calling Stimulus with a MonophasicPulse source we can generate a single
# pulse
stim = Stimulus(source=MonophasicPulse(ptype=pulse_type, pdur=pulse_dur,
                tsample=time_sample))

print(stim)

##############################################################################
# Here data is a 2D NumPy array where rows are electrodes and columns are the
# points in time. Since we did not specify any electrodes in our source
# MonophasicPulse, there is only one row denoting a single electrode. So
# we generated a single pulse with an amplitude of 1.
#
# This command also reveals a number of other parameters to set, such as:
#
# * ``electrodes``: you can either specify the electrodes in the source
# or within  the stimulus. If none are specified it looks up the source
# electrode.
# * `` metadata``: optionally you can include metadata to the stimulus you
# generate
#
# To change parameter values, either pass them directly to the constructor
# above or set them by hand, like this:

stim.metadata = 'Threshold measurement'

##############################################################################
# Let's visualize the pulse we generated

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(stim.time, stim.data[0])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude ($\mu$A)')
