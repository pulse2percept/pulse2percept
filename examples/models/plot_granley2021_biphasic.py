# -*- coding: utf-8 -*-
"""
==========================================================================================
Granley et al. (2021): Effects of Biphasic Pulse Parameters with the BiphasicAxonMapModel
==========================================================================================

This example shows how to use the
:py:class:`~pulse2percept.models.BiphasicAxonMapModel` to model the effects of 
biphasic pulse train parameters phosphene appearance in an epiretinal
implant such as :py:class:`~pulse2percept.implants.ArgusII`. 

Biphasic pulse trains are a commonly used type of stimulus in visual prostheses. 
This model enhances the :py:class:`~pulse2percept.models.AxonMapModel` to reflect
the effects of the amplitude, frequency, and pulse duration on threshold,
phosphene size, brightness, and streak length, according to previous
psychophysical and electrophysiological studies.

The :py:class:`~pulse2percept.models.BiphasicAxonMapModel` shares the same underlying 
assumptions as the axon map model. Namely, an axon's sensitivity to electrical stimulation
is assumed to decay exponentially:

*  with distance along the axon from the soma, with spatial decay
   constant :math:`\\lambda`,
*  with distance from the stimulated retinal location
   :math:`(x_{stim}, y_{stim})`, with spatial decay constant :math:`\\rho`.

In the biphasic model, the radial decay rate :math:`\\rho` is scaled by :math:`F_{size}`,
the axonal decay rate :math:`\\lambda` is scaled by :math:`F_{streak}`, and the brightness 
contribution from each electrode is scaled by :math:`F_{bright}`. The final equation for the
brightness intensity for a pixel located at polar coordinates :math:`(r, \\theta)` is given by:


.. math::

    I =  \max_{axon}\sum_{elecs}F_\mathrm{bright}  
            \exp\left(\frac{-d_{e}^2}{2\rho^2 F_\mathrm{size} } + 
            \frac{-d_{s}^2}{2\lambda^2 F_\mathrm{streak} }\right).

The biphasic axon map model can be instantiated and ran similarly to other models,
with the exception that all stimuli are required to be :py:class:`~pulse2percept.stimuli.BiphasicPulseTrain`

"""
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
from pulse2percept.implants import ArgusII
from pulse2percept.models import BiphasicAxonMapModel
from pulse2percept.stimuli import BiphasicPulseTrain
model = BiphasicAxonMapModel(rho=200, axlambda=800)

##############################################################################
# Parameters you don't specify will take on default values. You can inspect
# all current model parameters as follows:

print(model)

##############################################################################
# The most important parameters are ``rho`` and ``axlambda``, which control the 
# radial and axonal current spread, respectively.
#
# The parameters a0-a9 are coefficients for the size, streak, and bright
# models, which will be discusses later in this example.
#
# The biphasic axon map model supports both the default cython engine and a 
# faster, gpu-enabled jax engine. For details on running p2p on gpu, see
# TODO: jax
#
# The rest of the parameters are shared with 
# :py:class:`~pulse2percept.models.AxonMapModel`. For full details on these 
# parameters, see TODO: axon map tutorial link
#
##############################################################################


##############################################################################
# Build the model to perform expensive, one time calculations.

model.build()

##############################################################################
# .. important ::
#
#     You need to build a model only once. After that, you can apply any number
#     of stimuli -- or even apply the model to different implants -- without
#     having to rebuild (which takes time).
#
#     However, if you change model parameters
#     (e.g., by directly setting ``model.axlambda = 100``), you will have to
#     call ``model.build()`` again for your changes to take effect.
#
# Assigning a stimulus
# --------------------
# The second step is to specify a visual prosthesis from the
# :py:mod:`~pulse2percept.implants` module and giving it a stimulus
#
# Models with an axon map are well suited for epiretinal implants, such as 
# Argus II

implant = ArgusII()

##############################################################################
# You can visualize the location of the implant and the axon map

model.plot()
implant.plot()


##############################################################################
# As mentioned above, the Biphasic Axon Map Model only accepts 
# :py:class:`~pulse2percept.stimuli.BiphasicPulseTrain`
# stimuli with no delay_dur. The amplitude given to the BiphasicPulseTrain
# is interpreted as amplitude factor (i.e. an amp of 1 means 1xTh). 
#
# You can easily assign BiphasicPulseTrains to electrodes with a dictionary
# The following creates a train with 20Hz frequency, 1xTh amplitude, and 0.45ms
# pulse / phase duration.

implant.stim = {'A4' : BiphasicPulseTrain(20, 1, 0.45)}
implant.stim.plot()

##############################################################################
# Predicting the percept
# ----------------------
# Finally, you can predict the percept resulting from stimulation

percept = model.predict_percept(implant)
ax = percept.plot()
ax.set_title('Predicted percept')


##############################################################################
# Increasing the frequency will make phosphenes brighter
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
implant.stim = {'A4' : BiphasicPulseTrain(50, 1, 0.45)}
new_percept = model.predict_percept(implant)
new_percept.plot(ax=axes[1])
percept.plot(ax=axes[0], vmax=new_percept.max())
axes[0].set_title("20 Hz")
axes[1].set_title("40 Hz")
# Note that without setting vmax, matplotlib automatically rescales images to
# have the same max brightness and the difference isn't visible

##############################################################################
# Increasing amplitude increases both size and brightness
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
implant.stim = {'A4' : BiphasicPulseTrain(20, 3, 0.45)}
new_percept = model.predict_percept(implant)
new_percept.plot(ax=axes[1])
percept.plot(ax=axes[0], vmax=new_percept.max())
axes[0].set_title("1xTh")
axes[1].set_title("3xTh")

##############################################################################
# Increasing pulse duration decreases threshold, thus indirectly causing an 
# increase in size and brightness (amp factor is increased)
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
implant.stim = {'A4' : BiphasicPulseTrain(20, 1, 4)}
new_percept = model.predict_percept(implant)
new_percept.plot(ax=axes[1])
percept.plot(ax=axes[0], vmax=new_percept.max())
axes[0].set_title("0.45ms")
axes[1].set_title("4ms")
# If you account for the change in threshold by decreasing amplitude, then 
# The only affect is the streak length decreasing
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
implant.stim = {'A4' : BiphasicPulseTrain(20, 0.11856311, 4)}
new_percept = model.predict_percept(implant)
new_percept.plot(ax=axes[1])
percept.plot(ax=axes[0], vmax=new_percept.max())
axes[0].set_title("0.45ms")
axes[1].set_title("4ms, 0.11xTh")