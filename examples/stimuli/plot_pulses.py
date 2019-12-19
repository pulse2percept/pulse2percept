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
# Simplest Stimulus
# ---------------------
# :py:class:`~pulse2percept.stimuli.Stimulus` is the base class to generate
# different types of stimulus. Simplest way to instantiate a Stimulus is
# to pass a scalar value which is interpreted as the current amplitude
# for a single electrode.

# Let's start by importing necessary modules
from pulse2percept.stimuli import (MonophasicPulse, BiphasicPulse,
                                   Stimulus, PulseTrain)

import matplotlib.pyplot as plt
import numpy as np

stim = Stimulus(10)

##############################################################################
# Parameters you don't specify will take on default values. You can inspect
# all current model parameters as follows:

print(stim)

##############################################################################
# A MonophasicPulse
# --------------------
# We can specify the arguments of the monophasic pulse

pulse_type = 'anodic'  # whether current has a positive or negative amplitude
pulse_dur = 0.0046  # in seconds
time_sample = 0.1 / 1000  # temporal sampling in seconds

##############################################################################
# By calling Stimulus with a MonophasicPulse source we can generate a single
# pulse
monophasic_stim = Stimulus(source=MonophasicPulse(ptype=pulse_type,
                           pdur=pulse_dur, tsample=time_sample))

print(monophasic_stim)

##############################################################################
# Here data is a 2D NumPy array where rows are electrodes and columns are the
# points in time. Since we did not specify any electrodes in our source
# MonophasicPulse, there is only one row denoting a single electrode. So
# we generated a single pulse with an amplitude of 1.
#
# This command also reveals a number of other parameters to set, such as:
#
# * ``electrodes``: you can either specify the electrodes in the source
#   or within  the stimulus. If none are specified it looks up the source
#   electrode.
#
# * `` metadata``: optionally you can include metadata to the stimulus you
#   generate
#
# To change parameter values, either pass them directly to the constructor
# above or set them by hand, like this:

monophasic_stim.metadata = 'Threshold measurement'

##############################################################################
# Let's visualize the pulse we generated

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(monophasic_stim.time, monophasic_stim.data[0])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude ($\mu$A)')

###############################################################################
# A BiphasicPulse
# ------------------
# Similarly, we can generate biphasic pulse by changing the source of the
# stimulus to :py:class:`~pulse2percept.stimuli.BiphasicPulse. This time
# parameter ``ptype`` can either be 'anodicfirst' or 'cathodicfirst'.

# set relevant parameters
pulse_type = 'cathodicfirst'
pulse_dur = 0.0046  # in seconds
time_sample = 0.1 / 1000   # temporal sampling in seconds

biphasic_stim = Stimulus(source=BiphasicPulse(ptype=pulse_type, pdur=pulse_dur,
                         tsample=time_sample))

###############################################################################
# If we visualize this stimulus we can see the difference between a monophasic
# and biphasic pulse
# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# First, plot monophasic pulse
axes[0].plot(monophasic_stim.time, monophasic_stim.data[0])
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude ($\mu$A)')

# Second, plot biphasic pulse
axes[1].plot(biphasic_stim.time, biphasic_stim.data[0])
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Amplitude ($\mu$A)')

###############################################################################
# Changing The Amplitude Of Pulses
# ---------------------------------
# For any given pulse, we can modify the amplitude simply by indexing the
# electrode. For this example, we only have 1 electrode with an index of 0.
# Let's say we want the amplitude of monophasic pulse to be 10 microAmps and
# the cathodic part of the biphasic pulse to be -5, anodic part of the biphasic
# pulse to be 20 (this is hypothetical, as this wouldn't be suitable for
# patient testing as it isn't charge balanced)
###############################################################################
# You can change the amplitude in two ways: You can either change the values
# directly or create a NumPy array and assign that to the data structure of the
# stimulus

# get the data structure by indexing the electrode at 0
monophasic_stim.data[0] = 10*monophasic_stim.data[0]
print(monophasic_stim)

# OR
# recreate the same stimulus with an amplitude 1 microAmps.
pulse_type = 'anodic'
monophasic_stim = Stimulus(source=MonophasicPulse(ptype=pulse_type,
                           pdur=pulse_dur, tsample=time_sample))
monophasic_stim.data[0] = 10*np.ones_like(monophasic_stim.data[0])
print(monophasic_stim)

###############################################################################
# The outputs show that both works for changing the amplitude from 1 to 10
# microAmps.

###############################################################################
# Now, let's change the biphasic pulse.
# We first need to find the halfway point where the current switches from
# cathodic to anodic. To do that we first get the length of the pulse by
# indexing the single electrode at 0
length = len(biphasic_stim.data[0])
print(length)

# Find the halfway where cathodic turns into anodic pulse
half = int(len(biphasic_stim.data[0])/2)
print("Halfway index is", half)

# change the first half of the pulse to be 5 times larger
biphasic_stim.data[0][0:half] = 5*biphasic_stim.data[0][0:half]

# change the second half to be 20 times larger
biphasic_stim.data[0][half:length] = 20*biphasic_stim.data[0][half:length]
###############################################################################
# Let's plot the monophasic and biphasic pulses again
# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(25, 15))

# First, plot monophasic pulse
axes[0].plot(monophasic_stim.time, monophasic_stim.data[0])
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude ($\mu$A)')

# Second, plot biphasic pulse
axes[1].plot(biphasic_stim.time, biphasic_stim.data[0])
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Amplitude ($\mu$A)')
