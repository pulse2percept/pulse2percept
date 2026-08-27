# -*- coding: utf-8 -*-
"""
===============================================================================
Thompson et al. (2003): Circular phosphenes
===============================================================================

This example shows how to use the
:py:class:`~pulse2percept.models.Thompson2003Model`.

The model introduced in [Thompson2003]_ assumes that electrical stimulation
leads to circular percepts with discrete gray levels.
The model also allows for a fraction of phosphenes to be omitted at random
(dropout rate).

The model can be loaded as follows (using 10% dropout rate):
"""
# sphinx_gallery_thumbnail_number = 1

import matplotlib.pyplot as plt
import numpy as np
import pulse2percept as p2p

# A model predicts what a particular device produces, so it is bound to one.
# Here we will use an :py:class:`~pulse2percept.implants.ArgusII` implant:
implant = p2p.implants.ArgusII()
model = p2p.models.Thompson2003Model(implant=implant, step=0.2, dropout=0.1)

###############################################################################
# We are ready to predict percepts; the model builds itself on the first call.
#
# One way to describe a stimulus is a NumPy array with the same number of
# elements as there are electrodes in the array (i.e., 60).
# Choosing values from ``np.arange(60)`` will assign a different number to
# every electrode. We should thus expect to see 60 circular phosphenes that get
# gradually brighter from one electrode to the next:

stim = np.arange(60)
percept = model.predict_percept(stim)
percept.plot()

###############################################################################
# Setting a nonzero dropout rate will randomly choose a fraction of phosphenes
# to disappear:

fig, axes = plt.subplots(ncols=4, figsize=(15, 6))
for ax, drop in zip(axes, [0, 0.25, 0.5, 0.75]):
    model.build(dropout=drop)
    model.predict_percept(stim).plot(ax=ax)
    ax.set_title(f"{100*drop}% dropout")
fig.tight_layout()


###############################################################################
# Finally, the model can also be applied to
# :py:class:`~pulse2percept.stimuli.VideoStimulus` objects, where every frame
# of the video will be encoded with circular phosphenes and a given dropout
# rate.
#
# A video is a sequence of gray levels, and a model reads current, so the
# video has to be encoded first: that is the step that says how much current
# a gray level stands for (here 0-50 uA, sampled at the implant's electrodes).
# See :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`.
#
# The pulse rate has to keep up with the frame rate, or some frames go by
# without a pulse and are never seen; this video runs at 29.97 fps, so 30 Hz
# it is. Asking for a percept at the video's own frame times then gives one
# percept frame per video frame:

video = p2p.stimuli.BostonTrain()
encoded = video.encode(implant=implant, freq=30)
model.build(dropout=0.2)
model.predict_percept(encoded, t_percept=video.time).play()
