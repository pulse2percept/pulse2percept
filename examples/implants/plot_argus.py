"""
===============================================================================
Simulating Argus II
===============================================================================

Background
----------

:py:class:`~pulse2percept.implants.ArgusII` is an epiretinal implant developed
by Second Sight Medical Products, Inc. (Sylmar, CA).
Now discontinued, it was the first retinal implant to get FDA approval in the US
and the CE mark in Europe, and has been implanted in close to 500 patients
worldwide.

A number of studies have documented how the artificial vision provided by
:py:class:`~pulse2percept.implants.ArgusII` (Second Sight Medical Products, Inc.)
differs from normal sight.
Argus II contains 60 electrodes of 225 um diameter arranged in a 6 x 10
grid (575 um center-to-center separation) [Yue2020]_.
Some researchers therefore assumed that the stimulation of a grid of
electrodes on the retina would lead to the perception of a grid of luminous
dots ("phosphenes").
We refer to this as the :py:class:`~pulse2percept.models.ScoreboardModel`
of prosthetic vision.
However, a growing body of evidence has shown that retinal implant users often
report seeing distorted phosphenes and require extensive rehabilitative training 
to make use of their new vision [Beyeler2019]_, [EricksonDavis2021]_.

Although single-electrode phosphenes are consistent from trial to trial, they 
vary across electrodes and users [Luo2016]_, [Beyeler2019]_.
Phosphene shape strongly depends on stimulation parameters [Nanduri2012]_
as well as the retinal location of the stimulating electrode [Beyeler2019]_ 
due to inadvertent activation of passing axon fibers in the retina.
The result is a rich repertoire of phosphene shape that includes blobs, arcs,
wedges, and triangles [Beyeler2019]_:

"""
# sphinx_gallery_thumbnail_number = 1

import matplotlib.pyplot as plt
import pulse2percept as p2p

fig, axes = plt.subplots(ncols=3, figsize=(10, 3))
for ax, subject, scale in zip(axes, ['S2', 'S3', 'S4'], [1, 1, 0.5]):
    data = p2p.datasets.fetch_beyeler2019(subjects=subject)
    p2p.viz.plot_argus_phosphenes(data, ax=ax, scale=scale)
    ax.axis('off')
    ax.set_title(subject)
fig.tight_layout()

###############################################################################
# These phosphene shapes can be simulated with the
# :py:class:`~pulse2percept.models.AxonMapModel`, which was developed to fit
# behavioral data (see [Beyeler2019]_ for details).
#
# Boston Train sequence
# ---------------------
#
# To simulate the vision provided by Argus II, we first need to set up a new
# axon map model. We can specify phosphene size (``rho``) and elongation
# (``axlambda``) as well as the visual field we would like to simulate (given
# in degrees of visual angle):

model = p2p.models.AxonMapModel(rho=400, axlambda=200,
                                xrange=(-12, 12), yrange=(-8, 8))
model.build()

###############################################################################
# We can visualize where the implant sits on the axon map as follows:

model.plot()
p2p.implants.ArgusII().plot()

###############################################################################
# We then need to choose a stimulus to run through the model:

p2p.stimuli.BostonTrain().play()

###############################################################################
# In real life, the Argus II camera would capture a video just like the above,
# convert it to grayscale, downscale it, and then assign each pixel to an
# electrode in the implant.
#
# After feeding the resulting stimulus through the axon map model, we get a
# pretty good idea of what this video would look like to an Argus II patient:

implant = p2p.implants.ArgusII(stim=p2p.stimuli.BostonTrain())
model.predict_percept(implant).play()

###############################################################################
# The above simulation is basically equivalent to the scoreboard model, because
# each electrode appears as a large blob.
#
# However, as mentioned above, every patient sees phosphenes differently.
# To some they appear thin and elongated, to others they appear big and
# arc-like.
#
# To increase the arc length of individual phosphenes, we can choose a larger
# ``axlambda`` value:

model.axlambda = 600
model.build()
model.predict_percept(implant).play()

###############################################################################
# On the other hand, some retinal implant recipients (e.g., 51-009) report
# seeing thin and elongated phosphenes. In this case, the same video may appear
# very different to them:

model.rho = 50
model.axlambda = 1000
model.build()
model.predict_percept(implant).play()


###############################################################################
# Girl Pool sequence
# ------------------
#
# Another video shows a girl jumping into a swimming pool:

p2p.stimuli.GirlPool().play()

###############################################################################
# Similar to the above video, we can convert it to grayscale and downscale it,
# then feed it through the axon map model:

implant = p2p.implants.ArgusII(stim=p2p.stimuli.GirlPool())
model.rho = 400
model.axlambda = 200
model.build()
model.predict_percept(implant).play()

###############################################################################
# Here is the same video with longer phosphenes:

model.axlambda = 600
model.build()
model.predict_percept(implant).play()

###############################################################################
# Here is the same video with thin and long phosphenes:

model.rho = 50
model.axlambda = 1000
model.build()
model.predict_percept(implant).play()

###############################################################################
# In reality, the vision provided by Argus II may be even worse for several
# reasons:
#
# 1. The above simulations do not consider temporal distortions (e.g., flicker,
#    persistence, fading)
# 2. For some patients, individual phosphenes do not assemble into more complex
#    percepts, or do so in a highly unpredictable way.
