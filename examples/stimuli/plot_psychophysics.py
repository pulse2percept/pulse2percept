# -*- coding: utf-8 -*-
"""
===============================================================================
Gratings and bars in degrees of visual angle
=============================================

:py:mod:`pulse2percept.stimuli.psychophysics` generates gratings and bars in
visual rather than pixel units. The resulting
:py:class:`~pulse2percept.vision.Scene` carries the field of view needed to
register the stimulus with an implant.

A static grating
----------------

With ``time=None`` nothing changes over time, so the scene's source is an
:py:class:`~pulse2percept.stimuli.ImageStimulus` rather than a video of some
made-up duration:
"""
# sphinx_gallery_thumbnail_number = 1
import numpy as np
import matplotlib.pyplot as plt

from pulse2percept.stimuli import psychophysics
from pulse2percept.units import deg, dva, Hz, s

scene = psychophysics.grating(spatial_freq=0.5 / dva, fov=20 * dva,
                              shape=(256, 256))
print(type(scene.source).__name__, scene.fov)

scene.plot()
plt.show()

###############################################################################
# ``spatial_freq=0.5 / dva`` gives one cycle every two degrees. Changing
# ``shape`` changes only the raster resolution.
#
# ``direction`` is an ordinary angle, measured counterclockwise from the
# positive x axis, in the same visual-field frame the scene uses (x to the
# right, y upwards): 0 right, 90 up, 180 left, 270 down. The bars run
# perpendicular to it. ``phase`` shifts the pattern along that axis, and
# ``mask`` applies a circular or Gaussian aperture:

fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
for ax, kwargs in zip(axes, [{'direction': 0 * deg},
                             {'direction': 45 * deg},
                             {'direction': 90 * deg, 'mask': 'gauss'}]):
    psychophysics.grating(spatial_freq=0.5 / dva, fov=20 * dva,
                          shape=(256, 256), **kwargs).plot(ax=ax)
    ax.set_title(str(kwargs))
plt.show()

###############################################################################
# A drifting grating
# ------------------
#
# ``time`` contains the explicit sample times. Temporal phase is computed from
# those times rather than frame number.

time = np.arange(0, 1000, 20)  # ms
drift = psychophysics.grating(spatial_freq=0.5 / dva, temporal_freq=2 * Hz,
                              direction=0 * deg, fov=20 * dva,
                              shape=(128, 128), time=time)
drift.source.play()

###############################################################################
# At 2 Hz the pattern completes two cycles per second, and since one cycle is
# two degrees wide it travels ``temporal_freq / spatial_freq = 4`` dva/s.
#
# Because the clock is physical, the grating at a given timestamp does not
# depend on how densely the video was sampled. Sampling the same second every
# 40 ms instead of every 20 ms gives exactly the same frame at 200 ms:

coarse = psychophysics.grating(spatial_freq=0.5 / dva, temporal_freq=2 * Hz,
                               direction=0 * deg, fov=20 * dva,
                               shape=(128, 128), time=np.arange(0, 1000, 40))
fine_frame = drift.source.data.reshape(drift.source.vid_shape)[..., 10]
coarse_frame = coarse.source.data.reshape(coarse.source.vid_shape)[..., 5]
print(np.array_equal(fine_frame, coarse_frame))

###############################################################################
# A moving bar
# ------------
#
# :py:func:`~pulse2percept.stimuli.psychophysics.bar` draws a single bright
# bar of a given angular width, perpendicular to its direction of motion. Its
# center sits at ``offset + speed * t`` along the motion axis, measured from
# fixation, so ``offset`` is where it starts at ``t = 0``:

sweep = psychophysics.bar(width=2 * dva, direction=0 * deg,
                          speed=20 * dva / s, offset=-10 * dva,
                          edge_width=0.5 * dva, fov=20 * dva,
                          shape=(128, 128), time=np.arange(0, 1000, 20))
sweep.source.play()

###############################################################################
# ``width`` and ``edge_width`` are angular sizes, ``speed`` is in dva/s, and
# ``offset`` is the position at ``t = 0`` along the motion axis.
#
# Dropping ``time`` freezes it at ``t = 0``, which is a plain image again:

still = psychophysics.bar(width=2 * dva, offset=-5 * dva, fov=20 * dva,
                          shape=(256, 256))
still.plot()
plt.show()

###############################################################################
# Scenes carry visual-field geometry
# ----------------------------------
#
# Because these are scenes, their pixels have angular coordinates. That is
# what makes a bar "two degrees wide" rather than "sixteen pixels wide", and
# it is what a model needs in order to place the pattern on the retina:

print(still.fov)
print(still.dva_to_pixel(0, 0))
print(still.pixel_to_dva(0, 0))

###############################################################################
# Passing a scene to a model
# --------------------------
#
# A scene can be sampled at the implant's visual-field locations and encoded
# into stimulation:

from pulse2percept.implants.retina import ArgusII
from pulse2percept.models.retina import AxonMapModel
from pulse2percept.stimuli import AmplitudeEncoder
from pulse2percept.units import uA

implant = ArgusII(encoder=AmplitudeEncoder(amp_range=(0, 30) * uA, freq=50))
model = AxonMapModel(implant, xrange=(-10, 10), yrange=(-10, 10), step=0.25)

percept = model.predict_percept(drift, gaze=(0, 0) * dva,
                                t_percept=drift.time)
percept.play()

###############################################################################
# ``gaze`` says which point of the scene falls on the fovea, so moving the eye
# moves the grating across the array rather than redrawing it.
#
# Stating the grating in cycles/dva also makes it directly comparable with the
# array. Argus II spans about 19 x 11 degrees here, with roughly 2.1 degrees
# between neighboring electrodes, so it cannot resolve anything finer than
# about 0.24 cycles/dva: the 0.5 cycles/dva grating above is beyond what the
# array can sample. At 0.1 cycles/dva there are about five electrodes per
# cycle instead:

coarse = psychophysics.grating(spatial_freq=0.1 / dva, temporal_freq=2 * Hz,
                               fov=20 * dva, shape=(128, 128), time=time)
percept = model.predict_percept(coarse, gaze=(0, 0) * dva,
                                t_percept=coarse.time)
percept.play()

###############################################################################
# The scene's source is still an ordinary
# :py:class:`~pulse2percept.stimuli.VideoStimulus`, so its processing methods
# remain available if you need them; wrap the result back into a
# :py:class:`~pulse2percept.vision.Scene` with the same ``fov`` to keep the
# geometry:

from pulse2percept.vision import Scene

inverted = Scene(drift.source.invert(), fov=drift.fov)
percept = model.predict_percept(inverted, gaze=(0, 0) * dva,
                                t_percept=inverted.time)
percept.play()
