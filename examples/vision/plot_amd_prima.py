# -*- coding: utf-8 -*-
"""
============================================================================
Simulating PRIMA in age-related macular degeneration
============================================================================

Someone with geographic atrophy has lost vision in the center of their visual
field but still sees normally around it. A subretinal implant such as
:py:class:`~pulse2percept.implants.PRIMA` sits inside that blind spot and
gives back a coarse, grayscale percept there.

This example puts the two together. The only thing you have to state is where
each piece physically is; pulse2percept does the coordinate bookkeeping.

Three coordinate systems meet here, and one number relates them::

    scene coordinates
        | subtract gaze
    eye-centered visual field
        |- scotoma
        \\- prosthetic percept

``gaze`` is the scene location that currently falls on the fovea. The scotoma
is fixed relative to the fovea and the implant is fixed on the retina, so the
two never move relative to each other -- the scene moves past both.
"""
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt

from pulse2percept.implants import PRIMA
from pulse2percept.models import ScoreboardModel
from pulse2percept.stimuli import AmplitudeEncoder, LogoBVL
from pulse2percept.units import dva
from pulse2percept.vision import Scotoma, compose_hybrid_vision

###############################################################################
# The scene
# ---------
#
# Any image will do, as long as it says how much of the visual field it covers.
# ``fov`` is the outer extent of the image; the vertical extent follows from
# its aspect ratio.

scene = LogoBVL(resize=(240, 300), fov=40 * dva)

scene.plot()
plt.title('The scene')

###############################################################################
# .. note::
#
#     This logo has a transparent background, which composes against black the
#     way it does everywhere else in pulse2percept. That is why the panels
#     below sit on black rather than on white.

###############################################################################
# The eye
# -------
#
# A central geographic-atrophy scotoma 16 degrees across, and a PRIMA implant
# in the middle of it. The implant is placed in retinal coordinates (microns),
# which is where an implant actually lives; nothing here converts anything.

scotoma = Scotoma.circle(8 * dva)

implant = PRIMA()
implant.encoder = AmplitudeEncoder(amp_range=(0, 40), freq=20)

model = ScoreboardModel(rho=150, xrange=(-6, 6), yrange=(-6, 6),
                        step=0.1).build()

###############################################################################
# What the implant sees
# ---------------------
#
# ``vfmap`` is the retinotopy the model was built with, and it is what carries
# each electrode from the retina out into the visual field. ``gaze`` says where
# the eye is pointing. Together they decide which part of the scene each
# electrode is looking at.

gaze = (0, 0) * dva

implant.stim = implant.encoder.encode(scene, implant=implant,
                                      vfmap=model.vfmap, gaze=gaze)
prosthetic = model.predict_percept(implant)

prosthetic.plot()
plt.title('Prosthetic percept alone')

###############################################################################
# Putting it together
# -------------------
#
# ``vmax`` is the one thing that cannot be guessed: a percept is in arbitrary
# brightness units, so you have to say which brightness counts as white.
# pulse2percept does not infer that transfer function; taking the peak of this
# percept is simply the normalization *this example* chooses, and it is held
# fixed below so that a change of gaze does not silently rescale the display.
# ``scotoma_fill`` is what the lost region looks like where the implant does
# not reach -- black here.

display_vmax = prosthetic.data.max()

combined = compose_hybrid_vision(scene, prosthetic, scotoma,
                                 vmax=display_vmax, gaze=gaze,
                                 scotoma_fill=0)

combined.plot()
plt.title('Native vision with a PRIMA percept in the scotoma')

###############################################################################
# Outside the scotoma the scene comes through untouched. Inside it, the
# grayscale percept shows wherever the implant reaches, and the ring between
# the edge of the array and the edge of the scotoma stays dark.
#
# Looking somewhere else
# ----------------------
#
# Changing ``gaze`` moves the scene past the eye. The scotoma and the implant
# stay where they are relative to each other, so the blind spot travels across
# the scene and the implant keeps sampling whatever is at the fovea.

gaze = (8, -4) * dva

implant.stim = implant.encoder.encode(scene, implant=implant,
                                      vfmap=model.vfmap, gaze=gaze)
shifted = compose_hybrid_vision(scene, model.predict_percept(implant),
                                scotoma, vmax=display_vmax, gaze=gaze)

shifted.plot()
plt.title('Looking 8 degrees right and 4 degrees down')

###############################################################################
# .. note::
#
#     This example demonstrates the software workflow. Whether the electrode
#     geometry of a particular PRIMA device is faithfully reproduced is a
#     separate question; see :issue:`667` and :issue:`681`.
