# -*- coding: utf-8 -*-
"""
===============================================================================
Generating a stimulus from a video
===============================================================================

*This example shows how to use videos as input stimuli for a retinal implant.*

Loading a video
----------------

A video can be loaded as follows:

.. code:: python

    stim = p2p.stimuli.videos.VideoStimulus("path-to-video.mp4")

There is an example video that is pre-installed with pulse2percept. You can
load it like this.

"""
# sphinx_gallery_thumbnail_number = 1

import pulse2percept as p2p
import numpy as np

video = p2p.stimuli.BostonTrain(as_gray=True)
print(video)

##############################################################################
# There is a lot of useful information in this output.
#
# Firstly, note that ``vid_shape`` gives the dimension of the original video in
# (height, width, the number of frames).
#
# On the other hand, ``shape`` gives the dimension of the stimulation which is
# (the number of electrodes, the number of time steps). This is calculated from
# flattening the video to (height x width, the number of frames).
#
# The video data is stored as a 2D NumPy array in ``video.data``. You can
# reshape it to the original video dimensions as follows:

data = video.data.reshape(video.vid_shape)

##############################################################################
# Then it's possibly to access individual pixels or frames by indexing into the
# NumPy array. For example, to plot the first frame of the movie, use:

import matplotlib.pyplot as plt
plt.imshow(data[..., 0], cmap='gray')

##############################################################################
# Preprocessing a video
# ----------------------
#
# A :py:class:`~pulse2percept.stimuli.VideoStimulus` object comes with a number
# of methods to process a video before it is passed to an implant. Some
# examples include:
#
# -  :py:meth:`~pulse2percept.stimuli.VideoStimulus.invert` the gray levels of
#    the video,
# -  :py:meth:`~pulse2percept.stimuli.VideoStimulus.resize` the video,
# -  :py:meth:`~pulse2percept.stimuli.VideoStimulus.rotate` the video,
# -  :py:meth:`~pulse2percept.stimuli.VideoStimulus.filter` each frame of the
#    video and extract edges (e.g., Sobel, Scharr, Canny, median filter),
# -  :py:meth:`~pulse2percept.stimuli.VideoStimulus.apply` any input-output
#    function not provided: The function (applied to each frame of the video)
#    must accept a 2D or 3D image and return an image with the same dimensions.
#
# For a complete list, check out the documentation for
# :py:class:`~pulse2percept.stimuli.VideoStimulus`.
#
# Let's do some processing for our example video. Firstly, let's play the video
# so that we know what it looks like originally.
# (It might take a couple of seconds for the video to appear, because it needs
# to be converted to HTML & JavaScript code first.)

video.play()

##############################################################################
# For example, let's resize the video to 100 x 100, and then use the Sobel
# filter to extract edges. This can be done in one line. Then we will play
# the processed video:

edge_video = video.resize((100, 100)).filter('Sobel')
edge_video.play()

##############################################################################
# As demonstrated above, multiple video processing steps can be performed in
# one line. This is possible because each method returns a copy of the
# processed video (without altering the original).
# This means you can combine as many different processing steps as you like:

video.resize((40, 40)).rotate(10).invert().filter('median').play()

##############################################################################
# Using the video as input to a retinal implant
# ---------------------------------------------
#
# :py:class:`~pulse2percept.stimuli.VideoStimulus` can be used in
# combination with any :py:meth:`~pulse2percept.implants.ProsthesisSystem`.
# To make this work, the stimulus needs to have the same number of pixels as
# the implant has electrodes.
#
# In most cases, the implant contains a rectangular grid of electrodes, so
# we just have to resize the video first so that the number of pixels in each
# frame of the video matches the number of electrodes in the implant:

implant = p2p.implants.ArgusII()
# Assign the resized video to the implant:
implant.stim = video.resize(implant.shape)

##############################################################################
# Then you can feed the video directly into any of the available models
# described by a :py:class:`~pulse2percept.models.Model` object, such as the
# axon map model:

model = p2p.models.AxonMapModel()
model.build()
percept = model.predict_percept(implant)
percept.play()

##############################################################################
# Lastly, we can save the percept to disk:

percept.save('video_percept.mp4')
