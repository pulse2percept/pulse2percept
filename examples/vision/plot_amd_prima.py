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
*  :py:class:`~pulse2percept.implants.Implant`: where the implant's
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
# Geographic atrophy is rarely a circle centered on the fovea. The same center
# is used twice below: once for the lesion, and once to place the implant
# inside it.

center = (6, -2) * dva
scotoma = Scotoma.ellipse(5 * dva, 4 * dva, center=center)

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
#
# The logo is a transparent PNG, and what shows through belongs to the scene
# rather than to the picture, so ``background=1`` puts it on white instead of
# the default black. ``rings=True`` adds 5-degree rings about the
# fovea; they are drawn on top and change nothing about the scene.

logo = LogoBVL(resize=(240, 300))
filled_in = Scene(logo, fov=40 * dva, scotoma=scotoma,
                  scotoma_fill='inpaint', background=1)

filled_in.plot(gaze=(0, 0) * dva, rings=True)
plt.title('Native vision alone, with filling-in')

###############################################################################
# ``'inpaint'`` cannot be combined with a prosthetic percept: the inpainted
# scene would act as a brightness floor inside the scotoma, so an
# unstimulated electrode could never read dark, and how filling-in interacts
# with prosthetic vision is not modeled here. The scenes below therefore use
# a numeric fill, and black is the honest choice for what the device draws on.

scene = Scene(logo, fov=40 * dva, scotoma=scotoma, scotoma_fill=0,
              background=1)

###############################################################################

# PRIMA is driven by 880 nm light, not injected current, so the implant
# encodes gray levels as projector pulse durations by default:

implant = PRIMAPivotal()

###############################################################################
# ``PRIMAPivotal()`` is centered on the fovea, but this lesion is not, so the
# default position would put the array outside the atrophy it is meant to
# replace. ``implant_offset`` moves the whole array to the lesion center: it
# is a visual-field displacement, resolved through the model's
# ``visual_field_map`` into a single retinal translation, so the 100 um pixel
# pitch is unchanged and the implant object itself is left alone. The grid is
# widened to cover where the array now sits.

model = ScoreboardModel(implant=implant, implant_offset=center, rho=50,
                        xrange=(0, 12), yrange=(-8, 4), step=0.05)

###############################################################################
# The scene is trial input, like any other stimulus:

percept = model.predict_percept(scene, gaze=(0, 0) * dva, vmax=2)

percept.plot()
plt.title('Native vision with a PRIMA percept in the scotoma')

###############################################################################
# The lesion is eye-centered and the implant is on the retina, so both travel
# together when the eye moves:

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
