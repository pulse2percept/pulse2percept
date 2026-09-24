# -*- coding: utf-8 -*-
"""
===============================================================================
Argus II: images, video, and encoders
===============================================================================

Argus II converts each camera frame into biphasic pulses on 60 epiretinal
electrodes. This example follows an image and a video through that chain::

    source -> implant (encoder, raster) -> delivered pulses -> model -> percept

and changes one link at a time: the model, the encoder, and the source.
"""
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np

from pulse2percept.implants.retina import ArgusII
from pulse2percept.models.retina import AxonMapModel, BiphasicAxonMapModel
from pulse2percept.models.retina import ScoreboardModel
from pulse2percept.stimuli import (AmplitudeEncoder, FrequencyEncoder,
                                   psychophysics, samples)
from pulse2percept.units import Hz, dva, s, uA

###############################################################################
# From image to delivered stimulation
# -----------------------------------
#
# ``ArgusII()`` carries its device defaults: an
# :py:class:`~pulse2percept.stimuli.AmplitudeEncoder` that maps gray levels
# 0-1 to 0-50 uA at 6 Hz, and a
# :py:class:`~pulse2percept.implants.SequentialRaster` that pulses one row at
# a time. The image is stretched across the array, so each electrode receives
# the gray level at its own position:

implant = ArgusII()
image = samples.logo_ucsb()

fig, (ax_src, ax_dev) = plt.subplots(ncols=2, figsize=(11, 3.5))
image.plot(ax=ax_src)
ax_src.set_title('Source')
implant.plot(stim=image, stim_cmap=True, ax=ax_dev)
ax_dev.set_title('Delivered amplitude per electrode')
fig.tight_layout()

###############################################################################
# Same stimulation, two models
# ----------------------------
#
# The model sets how the delivered pulses appear.
# :py:class:`~pulse2percept.models.retina.ScoreboardModel` draws one round
# phosphene per electrode;
# :py:class:`~pulse2percept.models.retina.AxonMapModel` spreads each one along
# the nerve fiber bundles passing under it [Beyeler2019]_. ``rho`` (um) is the
# spread across bundles and ``lam`` (um) the spread along them:

models = {
    'ScoreboardModel': ScoreboardModel(implant=implant, rho=100),
    'AxonMapModel': AxonMapModel(implant=implant, rho=100, lam=300),
}

fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
for ax, (name, model) in zip(axes, models.items()):
    model.predict_percept(image).plot(ax=ax)
    ax.set_title(name)
fig.tight_layout()

###############################################################################
# Choosing an encoder
# -------------------
#
# Spatial-only models see one gray level per electrode, so the encoder's pulse
# parameters do not change their output. A spatiotemporal model reads the
# delivered pulse train.
# :py:class:`~pulse2percept.models.retina.BiphasicAxonMapModel` [Granley2021]_
# lets amplitude scale phosphene size and brightness, while frequency scales
# brightness only.
#
# The model takes amplitude as a multiple of perceptual threshold, so every
# electrode is given an assumed 50 uA threshold. Both encoders below drive a
# white pixel at the same charge per second (150 uA x 20 Hz = 50 uA x 60 Hz):

implant.thresholds = 50 * uA
model = BiphasicAxonMapModel(implant=implant, rho=100, lam=300)

encoders = {
    'Amplitude: 0-150 uA at 20 Hz':
        AmplitudeEncoder(amp_range=(0, 150) * uA, freq=20 * Hz),
    'Frequency: 0-60 Hz at 50 uA':
        FrequencyEncoder(amp=50 * uA, freq_range=(0, 60) * Hz),
}

percepts = {}
for label, encoder in encoders.items():
    implant.encoder = encoder
    percepts[label] = model.predict_percept(image)

vmax = max(p.data.max() for p in percepts.values())
fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
for ax, (label, percept) in zip(axes, percepts.items()):
    percept.plot(ax=ax, vmin=0, vmax=vmax)
    ax.set_title(label)
fig.tight_layout()

###############################################################################
# Both panels share one brightness scale. Amplitude encoding recruits wider
# phosphenes; frequency encoding keeps them narrow.
#
# Video
# -----
#
# A moving bar is a common Argus II motion task. The bar is defined in the
# visual field: a :py:class:`~pulse2percept.vision.Scene` 30 dva wide, sampled
# every 50 ms. Each electrode receives the part of the scene at its own
# visual-field location (see :ref:`topics-vision`).
#
# The default 6 Hz pulse rate is slower than the 20 fps video, so most frames
# would deliver no pulse. A 20 Hz encoder delivers one pulse per frame:

implant = ArgusII()
implant.encoder = AmplitudeEncoder(freq=20 * Hz)
model = AxonMapModel(implant=implant, rho=100, lam=300)

bar = psychophysics.bar(width=2 * dva, speed=20 * dva / s, offset=-12 * dva,
                        fov=30 * dva, time=np.arange(0, 1200, 50))
percept = model.predict_percept(bar, gaze=(0, 0) * dva)
percept.play()

###############################################################################
# What this does not establish
# ----------------------------
#
# * ``rho``, ``lam`` and the 50 uA threshold are illustrative values. Measured
#   values vary across subjects and electrodes.
# * Model brightness is in arbitrary units.
# * Argus II's camera-to-electrode mapping is configured per patient and is
#   not modeled; each electrode samples its own visual-field location.
