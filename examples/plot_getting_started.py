# -*- coding: utf-8 -*-
"""
===============================================================================
Getting Started
===============================================================================

A pulse2percept simulation has three main pieces:

1. an **implant** describing the implanted device,
2. a **stimulus** describing what the implant delivers, and
3. a **model** describing how stimulation becomes a percept.

The pieces are deliberately swappable. In a few lines you can move from an
electrical pulse train to a predicted retinal percept, switch to cortex, feed
the system an image or video, or combine prosthetic and residual vision.

Your first retinal percept
--------------------------

Start with an Argus II retinal implant and the biphasic axon-map model
[Granley2021]_. Unlike a simple Gaussian model, it captures axonal streaks and
lets pulse amplitude, frequency, and phase duration change predicted
brightness, phosphene size, and streak length.
"""
# sphinx_gallery_thumbnail_number = 1

import matplotlib.pyplot as plt
import numpy as np

import pulse2percept as p2p
from pulse2percept.units import Hz, mm, ms, uA, xTh, dva


retinal_implant = p2p.implants.retina.ArgusII()
retinal_model = p2p.models.retina.BiphasicAxonMapModel(
    retinal_implant,
    rho=200,
    lam=800,
)

stim = {
    'C5': p2p.stimuli.BiphasicPulseTrain(
        freq=20 * Hz,
        amp=2 * xTh,
        phase_dur=0.45 * ms,
    )
}

percept = retinal_model.predict_percept(stim)
percept.plot()
plt.title('Argus II: one stimulated electrode')
plt.show()

###############################################################################
# That is the whole user-facing workflow:
#
# ``implant + model + stimulus -> percept``
#
# Each piece owns a different scientific choice:
#
# * **Implant:** swap ``ArgusII`` for another retinal device, choose
#   ``eye='LE'`` or ``'RE'``, provide measured electrode thresholds, or attach
#   a different image/video encoder.
# * **Model:** change ``rho`` (spread across axons), ``lam`` (spread along
#   axons), the visual-field map, or the model-side implant placement. Swap in
#   :class:`~pulse2percept.models.retina.BiphasicScoreboardModel` if your
#   scientific question calls for round rather than axon-shaped phosphenes.
# * **Stimulus:** change pulse amplitude, frequency, phase duration, electrode,
#   or stimulation pattern.
#
#
# Cortex: same workflow, different anatomy
# ----------------------------------------
#
# Cortical stimulation uses the same pattern. Here a CORTIVIS array is placed
# over V1 and one electrode is stimulated. The workflow is the same:

cortical_implant = p2p.implants.cortex.Cortivis()

cortical_model = p2p.models.cortex.ScoreboardModel(
    cortical_implant,
    implant_position=(20, -5) * mm,
    regions=['v1'],
    rho=300,
    xrange=(-4, 0.5),
    yrange=(-1, 3.5),
    step=0.05,
)

percept = cortical_model.predict_percept({'11': 100 * uA})
percept.plot()
plt.title('CORTIVIS: one stimulated electrode')
plt.show()

###############################################################################
# Retinal and cortical components live in explicit anatomical namespaces:
#
# ``p2p.implants.retina`` / ``p2p.models.retina`` /
# ``p2p.topography.retina``
#
# and
#
# ``p2p.implants.cortex`` / ``p2p.models.cortex`` /
# ``p2p.topography.cortex``.
#
#
# Images are stimuli too
# ----------------------
#
# Stimulus encoders convert pixels (gray levels) into stimulation.
# Typically, gray levels are converted either to stimulus amplitude
# (amplitude modulation) or frequency (frequency modulation).
#
# Example: Convert gray levels from 0 to 255 to pulse amplitudes from 0
# to 3 times perceptual threshold:

encoder = p2p.stimuli.AmplitudeEncoder(
    amp_range=(0 * xTh, 3 * xTh),
    freq=20 * Hz,
    phase_dur=0.45 * ms,
)

retinal_implant.encoder = encoder

image = p2p.stimuli.samples.logo_bvl()
percept = retinal_model.predict_percept(image)

percept.plot()
plt.title('An image encoded for Argus II')
plt.show()

###############################################################################
# Individual thresholds can be stored in ``implant.thresholds``.
# :func:`~pulse2percept.stimuli.samples.logo_bvl` is a bundled
# :class:`~pulse2percept.stimuli.ImageStimulus`; your own image can be loaded
# the same way. The model receives the image directly because the implant knows
# how to encode it.
#
# If you want to inspect what the device actually delivers, call
# :meth:`~pulse2percept.implants.Implant.prepare_stim` yourself:
#
# .. code-block:: python
#
#     delivered = retinal_implant.prepare_stim(image)
#     delivered.plot()
#
#
# Videos work the same way
# ------------------------
#
# A video is another visual source. A clip made here on the spot -- a drifting
# pattern of light and dark patches at 30 fps -- lets us see both the input
# and the predicted percept as interactive players; a
# :class:`~pulse2percept.stimuli.VideoStimulus` reads a movie file the same
# way.
#
# The Granley biphasic model above describes one biphasic pulse-train condition
# per electrode. For a video whose encoded amplitude changes frame by frame,
# the spatial :class:`~pulse2percept.models.retina.AxonMapModel` maps that
# modulation to a percept frame by frame.

n_frames, rows, cols = 30, 120, 160
x = np.linspace(0, 4 * np.pi, cols)[np.newaxis, :, np.newaxis]
y = np.cos(np.linspace(0, 3 * np.pi, rows))[:, np.newaxis, np.newaxis]
phase = 2 * np.pi * np.arange(n_frames) / n_frames
video = p2p.stimuli.VideoStimulus(0.5 + 0.5 * y * np.sin(x - phase),
                                  metadata={'fps': 30})
video.play()

###############################################################################

implant = p2p.implants.retina.ArgusII(
    encoder=p2p.stimuli.AmplitudeEncoder(
        amp_range=(0, 50 * uA),
        freq=30 * Hz,
    )
)
model = p2p.models.retina.AxonMapModel(implant)

percept = model.predict_percept(video)
percept.play()

###############################################################################
# Images and videos can be resized, filtered, cropped, rotated, inverted, or
# processed with your own function before they reach the implant. Encoders,
# preprocessing, and the percept model are separate choices, so changing one
# does not require rewriting the rest of the simulation.
#
#
# Residual and prosthetic vision together
# ---------------------------------------
#
# pulse2percept can also place the prosthetic percept back into a visual scene.
# This is useful when the person to be simulated still has native vision outside
# a scotoma.
#
# In this example, ``samples.logo_bvl()`` supplies the image, ``Scene``
# describes the visible world and an eccentric central-field loss, PRIMA
# supplies prosthetic input inside the scotoma, and the retinal model predicts
# the combined view.

center = (6, -2) * dva

image = p2p.stimuli.samples.logo_bvl(resize=(240, 300))
scotoma = p2p.vision.Scotoma.ellipse(
    5 * dva,
    4 * dva,
    center=center,
)
scene = p2p.vision.Scene(
    image,
    fov=40 * dva,
    scotoma=scotoma,
    scotoma_fill=0,
    background=1,
)

scene.plot(rings=True)
plt.title('Scene with residual vision and a scotoma')
plt.show()

###############################################################################
# PRIMA uses photovoltaic stimulation rather than injected current, and carries
# the corresponding encoder by default. Device geometry remains device-local;
# the model places the implant at the same visual-field location as the lesion.

implant = p2p.implants.retina.PRIMAPivotal()

model = p2p.models.retina.ScoreboardModel(
    implant,
    implant_position=center,
    rho=50,
    xrange=(0, 12),
    yrange=(-8, 4),
    step=0.1,
)

percept = model.predict_percept(scene, gaze=(0, 0) * dva, vmax=2)
percept.plot()
plt.title('Residual vision with a PRIMA percept in the scotoma')
plt.show()

###############################################################################
# A :class:`~pulse2percept.vision.Scene` can also represent gaze, videos,
# backgrounds, and other residual-vision conditions.
# The :ref:`example gallery <sphx_glr_examples>` contains complete simulations.
#
#
# New in v0.11
# ------------
#
# pulse2percept 0.11 ("Foundations") introduces a backwards-incompatible API.
# This page already uses the new API. If you are updating older code, these are
# the changes you are most likely to encounter.
#
# **Models are bound to implants and build automatically**
#
# .. code-block:: python
#
#     # before
#     model = p2p.models.BiphasicAxonMapModel()
#     model.build()
#     percept = model.predict_percept(implant, stim)
#
#     # v0.11
#     model = p2p.models.retina.BiphasicAxonMapModel(implant)
#     percept = model.predict_percept(stim)
#
# **Anatomy is explicit**
#
# .. code-block:: python
#
#     p2p.implants.retina.ArgusII()
#     p2p.models.retina.AxonMapModel(...)
#     p2p.topography.retina.Watson2014Map()
#
#     p2p.implants.cortex.Cortivis()
#     p2p.models.cortex.ScoreboardModel(...)
#     p2p.topography.cortex.Polimeni2006Map()
#
# Generic machinery such as :class:`~pulse2percept.implants.Implant`,
# :class:`~pulse2percept.models.Model`, and
# :class:`~pulse2percept.topography.VisualFieldMap` remains at the package
# root.
#
# **Implant geometry and implant placement are separate**
#
# .. code-block:: python
#
#     implant = p2p.implants.retina.ArgusII()
#
#     model = p2p.models.retina.AxonMapModel(
#         implant,
#         implant_position=(2, -1) * dva,
#         implant_rotation=15,
#     )
#
# Named implants now describe device-local geometry. ``implant_position``,
# ``implant_rotation``, and ``implant_depth`` belong to the model and place the
# device in tissue.
#
# **Composite-model parameters live on their component**
#
# .. code-block:: python
#
#     model.spatial.rho = 250
#
# Model constructors expose the parameters they support instead of accepting
# arbitrary ``**params``.
#
# **``ProsthesisSystem`` is now ``Implant``**
#
# The old name is deprecated through v0.11 and is scheduled for removal in
# v0.12.
#
# For the complete migration record, see the
# :doc:`release notes </users/release_notes>`.
#
#
# Where to go next
# ----------------
#
# The :ref:`basic concepts <topics-index>` explain implants, stimuli, models,
# encoders, units, and other pieces in more detail. The
# :ref:`example gallery <sphx_glr_examples>` contains complete scientific
# workflows for retinal and cortical stimulation, residual vision, plotting,
# datasets, and custom extensions.
