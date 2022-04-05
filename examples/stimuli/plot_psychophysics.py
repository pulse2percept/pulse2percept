# -*- coding: utf-8 -*-
"""
===============================================================================
Generating a drifting sinusoidal grating or drifting bar stimulus
===============================================================================

*This example shows how to use drifting psychophysics-based stimuli for a retinal implant.*

Along with images, videos, and oter built-in stimuli, pulse2percept supports
generating :py:class: `~pulse2percept.stimuli.GratingStimulus` and :py:class: `~pulse2percept.stimuli.BarStimulus` as stimuli
that can be passed as percepts to implants.

Creating a Stimulus
-------------------

Create a stimulus as such:

.. code:: python

    grating_sin_stim = GratingStimulus( (height, width) )
    bar_stim = BarStimulus = BarStimulus( (height, width) )

Shape is the only required parameter for creating either of the stimuli.

"""

#####################################################################################
# Drifting Sinusoidal Grating
# ---------------------------
# 
# A drifting sinusoidal grating is represented by :py:class: `~pulse2percept.stimuli.GratingStimulus`.
# A visual example of a basic sinusoidal grating can be generated as such:

from pulse2percept.stimuli.psychophysics import GratingStimulus
stim = GratingStimulus((50, 50), temporal_freq=0.1)
stim.play()

#####################################################################################
# A size is required for :py:class: `~pulse2percept.stimuli.GratingStimulus`,
# and here we have set a custom temporal_freq parameter to increase the speed of the drift.
# There are parameters for the direction of the drift, the size of the bars, and other modifications.

#####################################################################################
# Drifting Bar Stimulus
# ---------------------------
# 
# A drifting bar is represented by :py:class: `~pulse2percept.stimuli.BarStimulus`.
# A visual example of a basic sinusoidal grating can be generated as such:

from pulse2percept.stimuli.psychophysics import BarStimulus
stim = BarStimulus((50, 50), speed=1)
stim.play()

#####################################################################################
# A size is required for :py:class: `~pulse2percept.stimuli.BarStimulus`,
# and here we have set a custom speed parameter to increase the speed of the drift.
# If, for example, we wanted a longer stimulus, we cold set the time parameter
# to the number of milliseconds we want the stimulus to last:

from pulse2percept.stimuli.psychophysics import BarStimulus
stim = BarStimulus((50, 50), speed=1, time=1500)
stim.play()

#####################################################################################
# Passing to an Implant
# ---------------------
# 
# Psychophsyics stimuli can be passed to an implant and predicted by a model.
# To demonstrate, we will pass a GratingStimululus to an :py:class: `~pulse2percept.implants.ArgusII`
# and use the Beyeler 2019 :py:class: `~pulse2percept.models.AxonMapModel` to interpret it:
#
# .. important ::
#   
#   Don't forget to build the model before using predict_percept
#

from pulse2percept.implants import ArgusII
from pulse2percept.models import AxonMapModel
from pulse2percept.stimuli import GratingStimulus
model = AxonMapModel()
model.build()

implant = ArgusII()
implant.stim = GratingStimulus((25,25), temporal_freq=0.1)

percept = model.predict_percept(implant)
percept.play()

#####################################################################################
# Pre-processing Stimuli
# ---------------------
# 
# Since both :py:class: `~pulse2percept.stimuli.BarStimulus` and :py:class: `~pulse2percept.stimuli.GratingStimulus`
# inherit form VideoStimulus, you can apply processing methods from VideoStimulus. In this example,
# we will invert the stimulus before passing it to the implant.
#
from pulse2percept.implants import ArgusII
from pulse2percept.models import AxonMapModel
from pulse2percept.stimuli import GratingStimulus
model = AxonMapModel()
model.build()

implant = ArgusII()
implant.stim = GratingStimulus((25,25), temporal_freq=0.1)

percept = model.predict_percept(implant)
percept.play()