# -*- coding: utf-8 -*-
"""
===============================================================================
Comparing a stimulus to the percept it produces
===============================================================================

*This example shows how to view an input stimulus next to the percept a model
predicts from it.*

Every object in pulse2percept knows how to draw itself, and
:py:mod:`pulse2percept.viz` puts two of them side by side. The pair to reach
for here is :py:func:`~pulse2percept.viz.plot_stimulus_percept` for a still
image and :py:func:`~pulse2percept.viz.play_stimulus_percept` for a video.

A still image
-------------

Start with a logo, an implant, and a model:

"""
# sphinx_gallery_thumbnail_number = 1

import pulse2percept as p2p

stim = p2p.stimuli.LogoUCSB(resize=(60, 80))
implant = p2p.implants.ArgusII()
model = p2p.models.ScoreboardModel(implant=implant, rho=200,
                                   xrange=(-14, 12), yrange=(-12, 12),
                                   step=0.25).build()

###############################################################################
# Assigning the image to the implant encodes it: ``implant.stim`` now holds the
# electrical stimulus, not the picture. Hand the original image to the model
# and to the plotting function; that is where the picture still lives:

percept = model.predict_percept(stim)
p2p.viz.plot_stimulus_percept(stim, percept)

###############################################################################
# A video
# -------
#
# The same works for a video, except that both panels now move. Predicting a
# percept for every frame of a 30-Hz video is the expensive part here, so keep
# the video small:

video = p2p.stimuli.BostonTrain(resize=(60, 80), as_gray=True)
percept = model.predict_percept(video)

###############################################################################
# :py:func:`~pulse2percept.viz.play_stimulus_percept` drives both panels off a
# single clock. The percept's time axis is authoritative: each percept frame is
# shown next to the source frame that is up at the same physical time, so a
# source and a percept sampled at different rates stay in register:

p2p.viz.play_stimulus_percept(video, percept, fmt='jpg')

###############################################################################
# ``fps`` resamples the whole presentation rather than speeding it up: a
# three-second sequence takes three seconds either way, and a lower rate simply
# samples it more coarsely.
