# -*- coding: utf-8 -*-
"""
===============================================================================
Generating a drifting sinusoidal grating or drifting bar stimulus
===============================================================================

*This example shows how to use drifting psychophysics-based stimuli for a retinal implant.*

Along with images, videos, and oter built-in stimuli, pulse2percept supports
generating :py:class:`~pulse2percept.stimuli.GratingStimulus` and :py:class:`~pulse2percept.stimuli.BarStimulus` as stimuli
that can be passed as percepts to implants.

Creating a Stimulus
-------------------

First, create the stimuli:


Shape (`(height, width)` in pixels) is the only required parameter for creating these stimuli.

A drifting sinusoidal grating is represented by :py:class:`~pulse2percept.stimuli.GratingStimulus`.
The following illustrates one frame of a grating stimulus.
"""
# sphinx_gallery_thumbnail_number = 1
import matplotlib.pyplot as plt
from pulse2percept.stimuli import GratingStimulus
stim = GratingStimulus((50, 50), spatial_freq=0.1, temporal_freq=0.1)
plt.imshow(stim.data[:, 0].reshape(50, 50), cmap='gray')
plt.title("Grating Stimulus")
plt.show()

########################################################################################
# You can view the entire stimulus over time using `stim.play()`
stim.play()

#####################################################################################
# Here, the spatial frequency of the grating (i.e., the inverse of how many pixels it
# takes to represent one cycle of the sinusoid) is given as 0.1 cycles/pixel, whereas
# the temporal frequency (i.e., the inverse of how many frames it takes to represent
# one cycle of the sinusoid) is given as 0.1 cycles/frame.
# By default, the drift direction of the grating will be to the right (0 degrees).
#
# A drifting bar is represented by :py:class:`~pulse2percept.stimuli.BarStimulus`.
# A visual example of a basic sinusoidal grating can be generated as such:

from pulse2percept.stimuli.psychophysics import BarStimulus
stim = BarStimulus((50, 50), speed=1)
stim.play()

#####################################################################################
# Here, the drift speed of the bar is given as 1 pixel/frame and, by default, the
# bar will drift to the right (0 degrees).
#
# Customizing the Stimulus
# ------------------------
#
# For both :py:class:`~pulse2percept.stimuli.BarStimulus` and 
# :py:class:`~pulse2percept.stimuli.GratingStimulus`,
# the only argument you must pass to the constructor is the shape `(height, width)` 
# in pixels.
# There are many optional arguments that can be passed to change various attributes 
# of the stimulus.
# In the examples above, we changed the speed at which the GratingStimulus changed 
# with temporal_freq *(scalar, cycles/frame)* 
# and the speed at which the BarStimulus moved with speed *(scalar, pixels/frame)* 
# in order to make the effect easier to visualize.
# If, for example, we wanted a longer stimulus, we cold set the time parameter 
# (in units of milliseconds)
# to change the duration of the stimulus:

stim = BarStimulus((50, 50), speed=1, time=1500)
stim.play()

#####################################################################################
# We can also change the direction *(scalar in [0, 360) degrees)*, where 0 degrees
# represents rightward motion, 90 degrees represents upward motion, 180 degrees
# represents leftward motion, and 270 degrees represents downward motion.
#
# .. code:: python
#
#     BarStimulus((height, width), direction=direction)
#
# Or the contrast *(scalar in [0,1])*
#
# .. code:: python
#
#     BarStimulus((height, width), contrast=contrast)
#
# For exact info on all of the arguments, please refer to
# :py:class:`~pulse2percept.stimuli.BarStimulus` and 
# :py:class:`~pulse2percept.stimuli.GratingStimulus`.
#
#
# Passing to an Implant
# ---------------------
# 
# Psychophsyics stimuli can be passed to an implant and combined with a model.
# To demonstrate, we will pass a ``GratingStimululus`` to an
# :py:class:`~pulse2percept.implants.ArgusII` implant and use the
# :py:class:`~pulse2percept.models.AxonMapModel` [Beyeler2019]_ to interpret it:
#
# .. important ::
#   
#   Don't forget to build the model before using ``predict_percept``
#

from pulse2percept.implants import ArgusII
from pulse2percept.models import AxonMapModel

model = AxonMapModel()
model.build()

implant = ArgusII()
implant.stim = GratingStimulus((25,25), temporal_freq=0.1)

percept = model.predict_percept(implant)
percept.play()

#####################################################################################
# As you can see in the above code segment, the stimulus passed to the implant does
# not necessarily have to have the same dimensions as the electrode grid.
# This is functionality built in to the implant code: The implant will automatically
# rescale the stimulus to the appropriate size.
# In the case of Argus II, the stimulus would thus be downscaled to a 6x10 image.
#
# Pre-Processing Stimuli
# ----------------------
# 
# Since both :py:class:`~pulse2percept.stimuli.BarStimulus` and 
# :py:class:`~pulse2percept.stimuli.GratingStimulus`
# inherit form :py:class:`~pulse2percept.stimuli.VideoStimulus`, we can apply 
# any video processing methods provided by ``VideoStimulus``.
#
# In the following example, we will invert the stimulus before passing it to the
# implant:

model = AxonMapModel()
model.build()

implant = ArgusII()
implant.stim = GratingStimulus((25,25), temporal_freq=0.1).invert()

percept = model.predict_percept(implant)
percept.play()
