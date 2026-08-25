# -*- coding: utf-8 -*-
"""
============================================================================
Simulating PRIMA in age-related macular degeneration
============================================================================

Someone with geographic atrophy has lost vision in the center of their visual
field but still sees normally around it. A subretinal implant such as
:py:class:`~pulse2percept.implants.PRIMA` sits inside that blind spot and
gives back a coarse, grayscale percept there.

Four objects say everything there is to say about that situation:

*  :py:class:`~pulse2percept.vision.Scene` -- what is visually present, and
   where native vision is lost.
*  :py:class:`~pulse2percept.implants.ProsthesisSystem` -- the device: where
   its electrodes are, and how it turns gray levels into stimulation.
*  :py:class:`~pulse2percept.models.Model` -- the retinotopy, which is what
   connects the two.
*  :py:class:`~pulse2percept.percepts.Percept` -- what the simulated observer
   sees.

None of the coordinate bookkeeping between them is yours to do.
"""
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt

from pulse2percept.implants import PRIMA
from pulse2percept.models import ScoreboardModel
from pulse2percept.stimuli import AmplitudeEncoder, LogoBVL
from pulse2percept.units import dva
from pulse2percept.vision import Scene, Scotoma

###############################################################################
# The scene
# ---------
#
# A picture, how much of the visual field it covers, and where native vision
# is missing. ``fov`` is the outer extent of the image; the vertical extent
# follows from its aspect ratio. The scotoma is *eye-centered*: 16 degrees
# across and fixed relative to the fovea.
#
# ``scene.plot`` draws what is left of native vision -- the picture outside
# the scotoma, and ``scotoma_fill`` (black, by default) inside it.

scotoma = Scotoma.circle(8 * dva)

scene = Scene(LogoBVL(resize=(240, 300)), fov=40 * dva, scotoma=scotoma)

scene.plot(gaze=(0, 0) * dva)
plt.title('Native vision alone')

###############################################################################
# .. note::
#
#     This logo has a transparent background, which composes against black the
#     way it does everywhere else in pulse2percept. That is why the panels
#     here sit on black rather than on white.
#
# The device
# ----------
#
# The implant lives in retinal coordinates, which is where an implant actually
# is, and its encoder says how a gray level becomes current. Those are two
# separate facts about the device, and neither of them is about the scene.

implant = PRIMA()
implant.encoder = AmplitudeEncoder(amp_range=(0, 40), freq=20)

###############################################################################
# The model
# ---------
#
# A model knows the retinotopy, so it is the one object that can say which
# part of the scene each electrode is looking at. Give it the scene and it
# does exactly that.

model = ScoreboardModel(scene=scene, rho=150, xrange=(-6, 6), yrange=(-6, 6),
                        step=0.1).build()

###############################################################################
# What the person sees
# --------------------
#
# ``gaze`` is the scene location that currently falls on the fovea. ``vmax``
# is the one thing that cannot be guessed: a percept is in arbitrary
# brightness units, so you have to say which brightness counts as white.
#
# The result is an ordinary :py:class:`~pulse2percept.percepts.Percept`:
# intact color vision outside the scotoma, and the prosthetic percept inside
# it, on the scene's own pixel grid.

percept = model.predict_percept(implant, gaze=(0, 0) * dva, vmax=40)

percept.plot()
plt.title('Native vision with a PRIMA percept in the scotoma')

###############################################################################
# Outside the scotoma the scene comes through untouched. Inside it, the
# grayscale percept shows wherever the implant reaches, and the ring between
# the edge of the array and the edge of the scotoma stays dark.
#
# Looking somewhere else
# ----------------------
#
# Changing ``gaze`` moves the scene past the eye. The scotoma is fixed
# relative to the fovea and the implant is fixed on the retina, so the two
# never move relative to each other: the blind spot travels across the scene,
# and the implant keeps sampling whatever is at the fovea.
#
# That is one argument, and nothing else changes. ``vmax`` is held at the same
# value so that a change of gaze does not silently rescale the display.

percept = model.predict_percept(implant, gaze=(8, -4) * dva, vmax=40)

percept.plot()
plt.title('Looking 8 degrees right and 4 degrees down')

###############################################################################
# Processing what the device sees
# -------------------------------
#
# ``implant.preprocess`` is what the device does to its own input, and it runs
# on the picture, before the scene is sampled at the electrodes -- an edge
# filter needs an image, and by sampling time there is one number per
# electrode. It belongs to the prosthetic branch alone: native vision outside
# the scotoma is unchanged, because the eye does not look through the camera.

implant.preprocess = lambda stim: stim.filter('sobel')

percept = model.predict_percept(implant, gaze=(0, 0) * dva, vmax=40)

percept.plot()
plt.title('Edge-filtered device input, intact vision around it')

###############################################################################
# .. note::
#
#     This example demonstrates the software workflow. Whether the electrode
#     geometry of a particular PRIMA device is faithfully reproduced is a
#     separate question; see :issue:`667` and :issue:`681`.
