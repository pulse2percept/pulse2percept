# -*- coding: utf-8 -*-
"""
===============================================================================
Shrinking inputs by content-aware image retargeting
===============================================================================

*This example shows how to use shrink by content-aware image retargeting for retinitis pigmentosa patients.*

pulse2percept.shrink implements improved content-aware image retargeting from https://doi.org/10.1186/1475-925X-9-52

You can shrink an image, a video, or an stimuli(ImageStimuli or VideoStimuli)

"""
##############################################################################
# First, let's take a look at the video using VideoStimuli.play()
#
import matplotlib.pyplot as plt
import pulse2percept as p2p
import numpy as np

# reader = get_reader('road.mp4')
# video = np.array([frame for frame in reader])
# stim = p2p.stimuli.videos.VideoStimulus(video[0:7].transpose(1,2,3,0))
stim = p2p.stimuli.VideoStimulus("road.mp4")

stim.play()

##############################################################################
# You can shrink only one frame of the video, which is an image
# The following would only shrink the first frame of the video


image = video[0]
img = p2p.processing.single_image_retargeting(image, wid=300, hei=50)
plt.imshow(img, cmap='gray')

##############################################################################
# It does not work quite well, because our function needs motion information
# to determine what is important. You can take the next frame of the image as
# another input to provide motion information

img = p2p.processing.image_retargeting(video[0], video[1], wid=300, hei=50)
plt.imshow(img, cmap='gray')


##############################################################################
# You can see that the people in the image is identified, and do not shrink a
# lot.
# The function also takes other parameters: N. boundary, L, num
#
# When calculating the motion map, the image will be divided into blocks of
# size ``N*N``, and the motion map Wt[x, y] will be determined by if there is
# change larger than the ``boundary`` in the block containing the pixel [x,y].
#
# We use seams to cut the input frame. ``num`` determines how many seams you
# would like to have when doing the seam carving.
#
# To preserve continuity between rows, we would apply a moving average window
# of size ``L*1`` to the importance matrix.
#
#
#
#
#
#
# Then let's try shrink the video. You can directly shrink it as a stimuli.
# It takes a long time.


new_stim = p2p.processing.video_retargeting(stim, wid=300, hei=50)
new_stim.play()


##############################################################################
# You can also shrink it when it is a video array

new_video = p2p.processing.stim_retargeting(video[0:5], wid=300, hei=50)
new_video_stim = p2p.stimuli.videos.VideoStimulus(np.dstack(new_video))
new_video_stim.play()
