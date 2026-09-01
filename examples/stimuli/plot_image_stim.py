# -*- coding: utf-8 -*-
"""
===============================================================================
Generating a stimulus from an image
===============================================================================

*This example shows how to use images as input stimuli for a retinal implant.*

In addition to built-in stimuli such as
:py:class:`~pulse2percept.stimuli.BiphasicPulse` and
:py:class:`~pulse2percept.stimuli.BiphasicPulseTrain`,
you can also load conventional images and convert them to stimuli using
:py:class:`~pulse2percept.stimuli.ImageStimulus`.

Loading an image
----------------

An image can be loaded as follows:

.. code:: python

    stim = ImageStimulus('path-to-image.png')

By default, each pixel in the image is assigned to an electrode, and its
grayscale value is encoded as an amplitude.
If the image has more than 1 channel (e.g., RGB, RGBA), the image is flattened
before each pixel/channel is assigned a different electrode.
You can specify names for the electrodes, but the number of electrodes must
match the number of pixels. By default, electrodes are labeled 1...N.

A number of images come pre-installed with pulse2percept, such as the logo of
the Bionic Vision Lab (BVL) at UC Santa Barbara:

"""
# sphinx_gallery_thumbnail_number = 4

import pulse2percept as p2p
import numpy as np

logo = p2p.stimuli.LogoBVL()
print(logo)

##############################################################################
# Inspecting the ``LogoBVL`` object, we can see that gray levels are converted
# to floats in the range [0, 1], and that the original 576x720x4 image is
# flattened so that each pixel can be assigned to an electrode.
#
# We also notice that ``time=None``, indicating that the stimulus does not have
# a time component. Thus we cannot apply temporal models to it.
#
# ``LogoBVL`` can be assigned to a stimulus and used in conjunction with a
# phosphene model, just like any other
# :py:class:`~pulse2percept.stimuli.Stimulus` object.
#
# Preprocessing an image
# ----------------------
#
# :py:class:`~pulse2percept.stimuli.ImageStimulus` objects come with a number
# of methods to process an image before it is passed to an implant. We can:
#
# -  :py:meth:`~pulse2percept.stimuli.ImageStimulus.invert` the
#    polarity of the image (applied to all channels except the alpha channel),
# -  convert RGB and RGBA images to grayscale using
#    :py:meth:`~pulse2percept.stimuli.ImageStimulus.rgb2gray`
#    (note that a change in the number of pixels also means a change in the
#    number of electrodes),
# -  :py:meth:`~pulse2percept.stimuli.ImageStimulus.resize` the image
#    to a new height x width (optionally using anti-aliasing),
# -  :py:meth:`~pulse2percept.stimuli.ImageStimulus.scale`,
#    :py:meth:`~pulse2percept.stimuli.ImageStimulus.shift`, and
#    :py:meth:`~pulse2percept.stimuli.ImageStimulus.rotate` the image
#    foreground (i.e., anything that's not black),
# -  :py:meth:`~pulse2percept.stimuli.ImageStimulus.trim` any black borders
#    around the image.
# -  :py:meth:`~pulse2percept.stimuli.ImageStimulus.threshold` the image using
#    a number of commonly used techniques (e.g., Otsu's method, adaptive
#    thresholding, ISODATA),
# -  :py:meth:`~pulse2percept.stimuli.ImageStimulus.filter` the image and
#    extract edges (e.g., Sobel, Scharr, Canny, median filter),
# -  :py:meth:`~pulse2percept.stimuli.ImageStimulus.apply` any input-output
#    function not covered above (must accept an image as input and return
#    another image of the same size).
#
# Collectively, these methods should support arbitrarily complex image
# preprocessing strategies, including the ones commonly used by implants such
# as Argus II and Alpha-AMS.
#
# Let's look at a concrete example.
# To get the BVL logo into proper shape, we need to convert the 4-channel RGBA
# image to grayscale. This can be done with
# :py:meth:`~pulse2percept.stimuli.ImageStimulus.rgb2gray`.
# In addition, since grayscale values will be mapped to current ampltiudes,
# we may want to :py:meth:`~pulse2percept.stimuli.ImageStimulus.invert` the
# image so that image edges appear bright on a dark background.
#
# We can perform both actions in one line, and plot the result side-by-side
# with the original image:

logo_gray = logo.invert().rgb2gray()

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
logo.plot(ax=ax1)
logo_gray.plot(ax=ax2)

##############################################################################
# As demonstrated above, multiple image processing steps can be performed in
# one line. This is possible because each method returns a copy of the
# processed image (without altering the original).
#
# The following example takes the grayscale logo, shrinks it to 75% of its
# original size, rotates it by 30 degrees (counter-clockwise), and trims the
# black border around the image:

logo_gray.scale(0.75).rotate(30).trim().plot()

##############################################################################
# As mentioned in the introduction above, the
# :py:meth:`~pulse2percept.stimuli.ImageStimulus.filter` method provides
# a number of popular techniques to extract edges from the image, such as:
#
# -  ``'sobel'`` to extract edges using the `Sobel operator
#    <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel>`_,
# -  ``'scharr'`` to extract edges using the `Scharr operator
#    <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.scharr>`_,
#    and
# -  ``'canny'`` to extract edges using the `Canny algorithm
#    <https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.canny>`_.
#
# Additional parameters (e.g., the standard deviation of the Gaussian filter
# for the Canny algorithm) can be passed as keyword arguments (e.g.,
# ``filter('canny', sigma=3)``).
#
# For example, we can use the Scharr operator as follows:

logo_edge = logo_gray.filter('scharr')

##############################################################################
# If more advanced image processing methods are required, we can use the
# :py:meth:`~pulse2percept.stimuli.ImageStimulus.apply` method to apply
# literally any function to the image. The only requirement is that the
# function return an image of the same size.
#
# For example, we can thicken the edges in the image by using a morphological
# operator (i.e., dilation) provided by
# `scikit-image <https://scikit-image.org>`_:

from skimage.morphology import dilation
logo_dilate = logo_edge.apply(dilation)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
# Edges extracted with the Scharr operator:
logo_edge.plot(ax=ax1)
# Edges thickened with dilation:
logo_dilate.plot(ax=ax2)

##############################################################################
# We can also save the processed stimulus as an image:

logo_dilate.save('dilated_logo.png')

##############################################################################
# Using the image as input to a retinal implant
# ---------------------------------------------
#
# :py:class:`~pulse2percept.stimuli.ImageStimulus` can be used in
# combination with any :py:meth:`~pulse2percept.implants.Implant`.
# We just have to resize the image first so that the number of pixels in the
# image matches the number of electrodes in the implant.
#
# But let's start from the top. The first two steps are to choose an implant
# and create a model bound to it:

from pulse2percept.implants import AlphaAMS
implant = AlphaAMS()

# Simulate only what we need (14x14 deg sampled at 0.1 deg):
model = p2p.models.ScoreboardModel(implant=implant, xrange=(-7, 7),
                                   yrange=(-7, 7), step=0.1)
model.build()

# Show the visual field we're simulating (dashed lines) atop the implant:
model.plot()
implant.plot()

##############################################################################
# Since :py:class:`~pulse2percept.implants.AlphaAMS` is a 2D electrode grid,
# all we need to do is downscale the image to the size of the grid, and then
# *encode* it:

stim_gray = logo_gray.resize(implant.shape).encode()

##############################################################################
# The downscaling assigns the pixels of the image to the electrodes in
# row-by-row order (i.e., we don't need to specify the actual electrode names).
#
# The encoding is what turns those pixels into stimulation. A gray level is
# not a current, and a model reads current, so something has to say how much
# current a gray level stands for --
# :py:meth:`~pulse2percept.stimuli.ImageStimulus.encode` is that something. By
# default it maps gray levels in [0, 1] onto a 20 Hz train of
# :py:class:`~pulse2percept.stimuli.BiphasicPulse` with amplitudes in
# [0, 50] uA; the section below shows how to change that. Handing the model an
# un-encoded image raises a
# :py:class:`~pulse2percept.units.DimensionMismatchError`.
#
# .. note ::
#
#    If the implant is not a proper 2D grid, you will have to manually specify
#    the input to each electrode.
#
#    In the near future, this will be done automatically using an implant's
#    ``preprocess`` method.
#
# Then the stimulus can be passed to the model's
# :py:meth:`~pulse2percept.models.ScoreboardModel.predict_percept` method:

percept_gray = model.predict_percept(stim_gray)

##############################################################################
# .. note ::
#
#     :py:class:`~pulse2percept.models.ScoreboardModel` has no temporal
#     component, so it reports the instantaneous brightness of each frame of
#     the pulse train. :py:meth:`~pulse2percept.percepts.Percept.plot` shows
#     the brightest of them.
#
# To see what difference our image preprocessing makes on the quality of the
# resulting percept, we can re-run the model on ``logo_dilate`` and plot the
# two percepts side-by-side:

stim_dilate = logo_dilate.trim().resize(implant.shape).encode()
percept_dilate = model.predict_percept(stim_dilate)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
percept_gray.plot(ax=ax1)
percept_dilate.plot(ax=ax2)

##############################################################################
# Customizing the encoding
# ------------------------
#
# The :py:meth:`~pulse2percept.stimuli.ImageStimulus.encode` method used above
# converts an image into a series of pulse trains (i.e., into electrical
# stimuli with a time component).
#
# By default, it interprets the gray level of a pixel as the current amplitude
# of a 20 Hz train of :py:class:`~pulse2percept.stimuli.BiphasicPulse`
# (0.46 ms phase duration), lasting 500 ms overall. Gray levels in the range
# [0, 1] are mapped onto currents in the range [0, 50] uA:

stim_dilate = logo_dilate.trim().resize(implant.shape).encode()

##############################################################################
# We can customize the range of amplitudes to be used by passing a keyword
# argument; e.g. ``amp_range=(0, 20)`` to use currents in [0, 20] uA, and the
# pulse rate with ``freq=50``.
#
# We can also specify our own pulse shape to be repeated, by passing a keyword
# argument such as ``pulse=BiphasicPulse(1, 0.2)``. Its amplitude is normalized
# away, since that is what the encoding sets; only its shape is used.
#
# ``encode`` is a shorthand for
# :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`, which offers the full
# set of options. Give one to an implant and the implant encodes whatever
# picture it is handed -- sampling it at the electrode locations *before*
# building the pulse trains, which for a video is the difference between a
# stimulus of a few hundred kilobytes and one of a few hundred megabytes:
#
# .. code-block:: python
#
#     implant.encoder = p2p.stimuli.AmplitudeEncoder(amp_range=(0, 50))
#     delivered = implant.prepare_stim(p2p.stimuli.BostonTrain())
#
# :py:class:`~pulse2percept.implants.ArgusII` brings one along already, so
# ``model.predict_percept(p2p.stimuli.BostonTrain())`` on a model bound to one
# is the whole setup.
#
# The other way to encode a gray level is as a pulse *rate* at fixed amplitude,
# which is what :py:class:`~pulse2percept.stimuli.FrequencyEncoder` does. It is
# considerably more expensive to simulate, because electrodes pulsing at
# different rates no longer pulse at the same times; ``clock`` (the period of
# the stimulator's time base) is the lever that keeps that under control:
#
# .. code-block:: python
#
#     implant.encoder = p2p.stimuli.FrequencyEncoder(freq_range=(0, 300),
#                                                    amp=50, clock=1)
#     delivered = implant.prepare_stim(p2p.stimuli.BostonTrain())
#
# A real stimulator usually cannot drive every electrode at once, because the
# current it can source at any instant is limited. Give the implant a
# :py:class:`~pulse2percept.implants.Raster` and the electrodes take turns
# instead, a group at a time, all of them within one pulse period. Every group
# still pulses at the full requested rate; the raster only decides *when*
# within each period each one fires, so no two groups are ever active at the
# same instant:
#
# .. code-block:: python
#
#     implant.max_current = 1000  # uA, summed over electrodes
#     implant.raster = p2p.implants.SequentialRaster(6)  # one row at a time
#     delivered = implant.prepare_stim(video)
#
# The raster belongs to the implant, and is the only place the schedule is
# described: assigning one binds it to that implant, so
# :py:class:`~pulse2percept.implants.CheckerboardRaster` can work its pattern
# out from where the electrodes actually are, and
# :py:meth:`~pulse2percept.implants.Raster.plot` can draw it without being
# told what to draw:
#
# .. code-block:: python
#
#     implant.raster = p2p.implants.CheckerboardRaster(5)
#     implant.raster.plot()
#
# Using the image as input to a spatiotemporal model
# ---------------------------------------------------
#
# Now, if we passed the new stimulus to
# :py:class:`~pulse2percept.models.ScoreboardModel`, it would simply apply the
# model (in space) to every time point in the stimulus.
# To get a proper temporal response, we need to extend the scoreboard model
# with a proper temporal model, such as
# :py:class:`~pulse2percept.models.Horsager2009Temporal`:

model = p2p.models.Model(
    p2p.models.ScoreboardSpatial(implant, xrange=(-7, 7), yrange=(-7, 7),
                                 step=0.1, rho=50),
    p2p.models.Horsager2009Temporal())

##############################################################################
# .. note::
#
#    You can combine any spatial model (names ending in **Spatial**) with any
#    temporal model (names ending in **Temporal**).
#
# ``xrange``, ``yrange``, ``step``, and ``rho`` are spatial parameters, so
# they are set on ``ScoreboardSpatial`` above.
#
# The ``rho`` parameter of the scoreboard model controls how much blur we get
# in the resulting percept. The value of this parameter should be set
# empirically to match the quality of the vision reported behaviorally by each
# implant user. For the purpose of this tutorial, we set it to 50um.

model.build()

##############################################################################
# The predicted percept will now be a movie, where the spatial response (i.e.,
# each frame of the movie) is primarily determined by the scoreboard model, but
# the temporal evolution of these frames is determined by the Horsager model.
#
# By default, the model will output a movie frame every 20 ms (corresponding to
# a 50 Hz frame rate). The frame rate can be adjusted by passing a list of
# time points to :py:meth:`~pulse2percept.Model.predict_percept` (e.g.,
# ``t_percept=np.arange(500)`` to get an output every millisecond):

percept = model.predict_percept(stim_dilate)

##############################################################################
# The output of the model is a :py:class:`~pulse2percept.percepts.Percept`
# object, which can be animated in IPython or Jupyter Notebook using the
# :py:meth:`~pulse2percept.percepts.Percept.play` method:

percept.play()

##############################################################################
# You can also save the percept as a movie:

percept.save('logo_percept.mp4')
