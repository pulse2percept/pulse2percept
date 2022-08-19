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

A number of studies have documented the artificial vision provided by
Argus II, which contains 60 electrodes of 225 um diameter arranged in a 6 x 10
grid (575 um center-to-center separation) [Yue2020]_.
Researchers often assume that the stimulation of a grid of
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
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import pulse2percept as p2p

fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
for ax, subject in zip(axes, ['S2', 'S3', 'S4']):
    data = p2p.datasets.fetch_beyeler2019(subjects=subject)
    p2p.viz.plot_argus_phosphenes(data, ax=ax)

###############################################################################
# Simulating video
# ----------------
#

model = p2p.models.AxonMapModel(rho=300, axlambda=100).build()
implant = p2p.implants.ArgusII(stim=p2p.stimuli.BostonTrain())
model.predict_percept(implant).play()

###############################################################################
# more cowbell

model = p2p.models.AxonMapModel(rho=300, axlambda=500).build()
model.predict_percept(implant).play()

###############################################################################
# more cowbell

model = p2p.models.AxonMapModel(rho=300, axlambda=1000).build()
model.predict_percept(implant).play()
