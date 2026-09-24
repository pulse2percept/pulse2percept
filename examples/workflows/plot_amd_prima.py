# -*- coding: utf-8 -*-
"""
===============================================================================
Simulating PRIMA in age-related macular degeneration
===============================================================================

In geographic atrophy, central vision is lost and peripheral vision remains.
The subretinal photovoltaic implant
:py:class:`~pulse2percept.implants.retina.PRIMAPivotal` is placed inside the
lesion and produces a coarse grayscale percept there.

This example covers scene, scotoma, prosthetic percept, gaze, device
preprocessing, and both eyes. The objects are documented under
:ref:`topics-vision`.
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
# The lesion and the implant share an eccentric center (dva).

center = (6, -2) * dva
logo = samples.logo_bvl(resize=(240, 300))
scotoma = Scotoma.ellipse(5 * dva, 4 * dva, center=center)

###############################################################################
# ``scotoma_fill`` sets what is shown inside the loss. Simulations often use
# black or gray, but people with AMD rarely report a black spot and are often
# unaware of the loss. ``'inpaint'`` fills the scotoma from the surrounding
# vision (:py:func:`skimage.restoration.inpaint_biharmonic`). This is a
# frame-local image operation, not a model of perceptual filling-in.
#
# ``background=1`` shows the logo's transparent pixels as white.
# ``rings=True`` draws 5 dva eccentricity rings.

filled_in = Scene(logo, fov=40 * dva, scotoma=scotoma,
                  scotoma_fill='inpaint', background=1)

filled_in.plot(gaze=(0, 0) * dva, rings=True)
plt.title('Native vision alone, with filling-in')

###############################################################################
# ``'inpaint'`` is not combined with prosthetic percepts, so the rest of the
# example uses a numeric fill.

scene = Scene(logo, fov=40 * dva, scotoma=scotoma, scotoma_fill=0,
              background=1)

###############################################################################
# Prosthetic vision inside the loss
# ---------------------------------
#
# PRIMA is driven by pulsed 880 nm light. Its default
# :py:class:`~pulse2percept.stimuli.PRIMAEncoder` converts gray levels to
# projector ON durations. ``implant_position`` places the array at the lesion
# center, and the model grid covers the field around it.

implant = PRIMAPivotal()
model = ScoreboardModel(implant=implant, implant_position=center, rho=50,
                        xrange=(0, 12), yrange=(-8, 4), step=0.05)

percept = model.predict_percept(scene, gaze=(0, 0) * dva)

scene.plot(percept=percept, gaze=(0, 0) * dva, vmax=2)
plt.title('Native vision with a PRIMA percept in the scotoma')

###############################################################################
# The scene is drawn at its own 240 x 300 pixels and the percept on its
# 0.05 dva model grid; neither is resampled. ``scene.render(...)`` returns a
# single RGB raster when one is needed.
#
# Gaze
# ----
#
# The lesion and the implant move with the eye; the scene does not:

percept = model.predict_percept(scene, gaze=(8, -4) * dva)

scene.plot(percept=percept, gaze=(8, -4) * dva, vmax=2)
plt.title('Looking 8 degrees right and 4 degrees down')

###############################################################################
# Device preprocessing
# --------------------
#
# ``implant.preprocess`` applies to the prosthetic input only. An edge filter
# changes the percept; native vision is unchanged. Edges light fewer pixels,
# so ``vmax`` is lowered:

implant.preprocess = lambda stim: stim.filter('sobel')

percept = model.predict_percept(scene, gaze=(0, 0) * dva)

scene.plot(percept=percept, gaze=(0, 0) * dva, vmax=0.3)
plt.title('Edge-filtered device input, intact vision around it')

###############################################################################
# Both eyes
# ---------
#
# A :py:class:`~pulse2percept.vision.Scene` is one eye, so the fellow eye
# gets its own scene, here with a smaller, partial loss. A
# :py:class:`~pulse2percept.vision.BinocularScene` holds the pair.
# ``aperture='round'`` draws each field as an ellipse (display only).

implant.preprocess = False

worse_eye = Scene(logo, fov=40 * dva, scotoma=scotoma, scotoma_fill=0,
                  background=1, aperture='round')
fellow_eye = Scene(logo, fov=40 * dva, background=1, aperture='round',
                   scotoma=Scotoma.circle(3 * dva, center=center),
                   scotoma_fill=0.4)

binocular = BinocularScene(left=worse_eye, right=fellow_eye)

###############################################################################
# Models are monocular: the percept is predicted for the implanted eye and
# drawn on that side. The fellow eye shows only its own scene.

prosthetic_input = Scene(logo, fov=40 * dva, background=1)
percept = model.predict_percept(prosthetic_input, gaze=(0, 0) * dva)

binocular.plot(left_percept=percept, vmax=2, rings=True)

###############################################################################
# What this does not establish
# ----------------------------
#
# * :py:class:`~pulse2percept.models.retina.ScoreboardModel` shows where the
#   light lands, normalized to a fully lit pixel. It does not model
#   photovoltaic transduction or retinal activation
#   (:py:class:`~pulse2percept.models.retina.Ho2018Model` does).
# * ``rho = 50`` um is an illustrative spread, not a measured PRIMA
#   point-spread function.
# * Combining prosthetic and residual vision inside the scotoma is a display
#   convention; their perceptual interaction is not modeled.
# * Residual acuity, binocular combination, disparity, and depth are not
#   modeled.
