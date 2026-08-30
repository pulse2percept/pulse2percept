# -*- coding: utf-8 -*-
"""
============================================================================
Simulating PRIMA in age-related macular degeneration (AMD)
============================================================================

Someone with geographic atrophy has lost vision in the center of their visual
field but still sees normally around it. A subretinal implant such as
:py:class:`~pulse2percept.implants.PRIMAPivotal` sits inside the blind
spot and gives back a coarse, grayscale percept there.

Four objects capture that situation:

*  :py:class:`~pulse2percept.vision.Scene`: what is visually present, and
   where native vision is lost.
*  :py:class:`~pulse2percept.implants.ProsthesisSystem`: where the implant's
   electrodes are, and how it turns gray levels into stimulation.
*  :py:class:`~pulse2percept.models.Model`: the retinotopy, which is what
   connects the two.
*  :py:class:`~pulse2percept.percepts.Percept`: what the simulated observer
   sees.
"""
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt

from pulse2percept.implants import PRIMAPivotal
from pulse2percept.models import ScoreboardModel
from pulse2percept.stimuli import LogoBVL
from pulse2percept.units import dva
from pulse2percept.vision import Scene, Scotoma

###############################################################################
# Geographic atrophy is rarely a circle centered on the fovea:

scotoma = Scotoma.ellipse(7 * dva, 5 * dva, center=(1, -1) * dva)

###############################################################################
# ``scotoma_fill`` determines what a user inside the scotoma sees.
# Biologically, no information from within the scotoma is supposed to reach
# the visual cortex. So researchers often set it to gray (or black).
#
# But people with AMD rarely report seeing a black spot in their vision.
# Often they're not even aware that they have a "blind spot".
# It is quite possible that the brain fills in the missing information.
# To mimic that, ``'inpaint'`` fills the scotoma from the vision around it
# by biharmonic inpainting (:py:func:`skimage.restoration.inpaint_biharmonic`).
# Note that this is a frame-local, boundary-driven approximation to
# perceptual filling-in, not a neural or generative model of it.

scene = Scene(LogoBVL(resize=(240, 300)), fov=40 * dva, scotoma=scotoma,
              scotoma_fill='inpaint')

scene.plot(gaze=(0, 0) * dva)
plt.title('Native vision alone')

###############################################################################

# PRIMA is driven by 880 nm light, not injected current, so the implant
# encodes gray levels as projector pulse durations by default:

implant = PRIMAPivotal()

###############################################################################

model = ScoreboardModel(implant=implant, rho=50, xrange=(-6, 6),
                        yrange=(-6, 6), step=0.05)

###############################################################################
# The scene is trial input, like any other stimulus:

percept = model.predict_percept(scene, gaze=(0, 0) * dva, vmax=2)

percept.plot()
plt.title('Native vision with a PRIMA percept in the scotoma')

###############################################################################

percept = model.predict_percept(scene, gaze=(8, -4) * dva, vmax=2)

percept.plot()
plt.title('Looking 8 degrees right and 4 degrees down')

###############################################################################

implant.preprocess = lambda stim: stim.filter('sobel')

# Edges drive far fewer pixels, so white sits much lower:

percept = model.predict_percept(scene, gaze=(0, 0) * dva, vmax=0.3)

percept.plot()
plt.title('Edge-filtered device input, intact vision around it')

###############################################################################
