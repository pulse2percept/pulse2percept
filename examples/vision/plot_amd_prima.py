# -*- coding: utf-8 -*-
"""
===============================================================================
Simulating PRIMA in age-related macular degeneration
===============================================================================

Someone with geographic atrophy has lost vision in part of their visual field
but still sees normally around it. A subretinal photovoltaic implant such as
:py:class:`~pulse2percept.implants.retina.PRIMAPivotal` sits inside that blind
region and returns a coarse grayscale percept there, while native vision
continues to work everywhere else.

Simulating that means keeping four things separate: what is in the visual
field, where native vision is lost, what the device does with the light it
receives, and where in the field its percept lands. This example walks the
complete workflow end to end -- scene, scotoma, prosthetic percept, gaze,
device preprocessing, and both eyes.

The objects and conventions used here are documented under
:ref:`topics-vision`, :ref:`topics-coordinates` and :ref:`topics-stimulation`.
"""
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt

from pulse2percept.implants.retina import PRIMAPivotal
from pulse2percept.models.retina import ScoreboardModel
from pulse2percept.stimuli import samples
from pulse2percept.units import dva
from pulse2percept.vision import BinocularScene, Scene, Scotoma

###############################################################################
# Residual vision
# ---------------
#
# The lesion and the implant share an eccentric center: the device is placed
# where the vision is missing.

center = (6, -2) * dva
logo = samples.logo_bvl(resize=(240, 300))
scotoma = Scotoma.ellipse(5 * dva, 4 * dva, center=center)

###############################################################################
# ``scotoma_fill`` says what is seen inside the loss. Biologically no
# information from within the scotoma reaches cortex, so researchers often use
# black or gray. But people with AMD rarely report a black spot, and are often
# unaware of the loss at all; the brain appears to fill it in. ``'inpaint'``
# mimics that by filling the scotoma from the vision around it
# (:py:func:`skimage.restoration.inpaint_biharmonic`) -- a frame-local,
# boundary-driven approximation, not a neural or generative model of
# filling-in.
#
# The logo is a transparent PNG, and what shows through belongs to the scene
# rather than the picture, so ``background=1`` puts it on white.
# ``rings=True`` adds 5-degree eccentricity rings; they are drawn on top and
# change nothing.

filled_in = Scene(logo, fov=40 * dva, scotoma=scotoma,
                  scotoma_fill='inpaint', background=1)

filled_in.plot(gaze=(0, 0) * dva, rings=True)
plt.title('Native vision alone, with filling-in')

###############################################################################
# Filling-in is not modeled together with prosthetic vision, so the remaining
# simulations use a numeric fill.

scene = Scene(logo, fov=40 * dva, scotoma=scotoma, scotoma_fill=0,
              background=1)

###############################################################################
# Prosthetic vision inside the loss
# ---------------------------------
#
# PRIMA is driven by pulsed 880 nm light rather than injected current, and
# carries a :py:class:`~pulse2percept.stimuli.PRIMAEncoder` by default, so gray
# levels become projector ON durations without further setup. The model places
# the device-local origin at the lesion center and simulates the field around
# it.

implant = PRIMAPivotal()
model = ScoreboardModel(implant=implant, implant_position=center, rho=50,
                        xrange=(0, 12), yrange=(-8, 4), step=0.05)

percept = model.predict_percept(scene, gaze=(0, 0) * dva)

scene.plot(percept=percept, gaze=(0, 0) * dva, vmax=2)
plt.title('Native vision with a PRIMA percept in the scotoma')

###############################################################################
# The scene keeps its own 240 x 300 pixels and the percept its own
# 0.05-degree model grid; drawing them together does not resample either.
# ``scene.render(...)`` produces the dense RGB composite when one raster is
# actually needed.
#
# Gaze
# ----
#
# The lesion is eye-centered and the implant is on the retina, so both move
# with the eye. Only the scene stays put:

percept = model.predict_percept(scene, gaze=(8, -4) * dva)

scene.plot(percept=percept, gaze=(8, -4) * dva, vmax=2)
plt.title('Looking 8 degrees right and 4 degrees down')

###############################################################################
# Device preprocessing
# --------------------
#
# What the device does to its own input is not something the eye goes through.
# An edge filter applied to the prosthetic branch changes the percept while
# native vision around it is untouched. Edges drive far fewer pixels, so the
# display ceiling has to come down with them:

implant.preprocess = lambda stim: stim.filter('sobel')

percept = model.predict_percept(scene, gaze=(0, 0) * dva)

scene.plot(percept=percept, gaze=(0, 0) * dva, vmax=0.3)
plt.title('Edge-filtered device input, intact vision around it')

###############################################################################
# Both eyes
# ---------
#
# PRIMA is implanted in one eye, and a :py:class:`~pulse2percept.vision.Scene`
# is one monocular visual field, so the fellow eye needs a scene of its own.
# A :py:class:`~pulse2percept.vision.BinocularScene` holds the pair. Vision
# loss need not be symmetric: here the fellow eye has a smaller, partial loss.
#
# ``aperture='ellipse'`` inscribes an eye-centered ellipse in ``fov``; this
# affects rendering only.

implant.preprocess = False

worse_eye = Scene(logo, fov=40 * dva, scotoma=scotoma, scotoma_fill=0,
                  background=1, aperture='ellipse')
fellow_eye = Scene(logo, fov=40 * dva, background=1, aperture='ellipse',
                   scotoma=Scotoma.circle(3 * dva, center=center),
                   scotoma_fill=0.4)

binocular = BinocularScene(left=worse_eye, right=fellow_eye)

###############################################################################
# Models are monocular, so a percept is predicted for one eye and then assigned
# to that eye's side of the binocular scene. The fellow eye shows whatever its
# own scene says it sees.

prosthetic_input = Scene(logo, fov=40 * dva, background=1)
percept = model.predict_percept(prosthetic_input, gaze=(0, 0) * dva)

binocular.plot(left_percept=percept, vmax=2, rings=True)

###############################################################################
# What this does not establish
# ----------------------------
#
# * :py:class:`~pulse2percept.models.retina.ScoreboardModel` visualizes where
#   the light lands, normalized to a fully lit pixel. It does not model
#   photovoltaic transduction or retinal activation;
#   :py:class:`~pulse2percept.models.retina.Ho2018Model` is the model that
#   attempts a retinal response.
# * ``rho = 50`` is a spatial-spread choice for this illustration, not a
#   measured PRIMA point-spread function.
# * The composition inside the scotoma is a display convention. How prosthetic
#   and residual percepts actually interact perceptually is not modeled, which
#   is why nothing is drawn over intact vision.
# * Differences in residual acuity between the eyes, binocular combination,
#   disparity and depth are all outside the model.
