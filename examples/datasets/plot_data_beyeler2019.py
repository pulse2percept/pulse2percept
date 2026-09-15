# -*- coding: utf-8 -*-
"""
===============================================================================
Beyeler et al. (2019): Phosphenes are not pixels
===============================================================================

Simulated prosthetic vision is often drawn as a grid of independent dots, one
per electrode. [Beyeler2019]_ tested that assumption directly: Argus I and
Argus II users were asked to draw what they saw during single-electrode
stimulation.

Two results follow from those drawings. Phosphenes are **elongated**, not
round, and their orientation follows the trajectory of the retinal nerve fiber
bundles (NFBs) passing under the stimulated electrode. This example reproduces
both from the published data, and predicts the drawings with a
subject-specific :py:class:`~pulse2percept.models.retina.AxonMapModel`.

.. important ::

    This dataset requires `Pandas <https://pandas.pydata.org>`_
    (``pip install pandas``) and `h5py <https://www.h5py.org>`_
    (``pip install h5py``). The 66 MB archive is downloaded from the Open
    Science Framework on first use and cached in ``~/pulse2percept_data``.
"""
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np

from pulse2percept.datasets import fetch_beyeler2019
from pulse2percept.implants.retina import ArgusII
from pulse2percept.models.retina import AxonMapModel
from pulse2percept.plotting import (plot_argus_phosphenes,
                                    plot_argus_simulated_phosphenes)
from pulse2percept.stimuli import Stimulus
from pulse2percept.units import um

###############################################################################
# The measured drawings
# ---------------------
#
# The dataset contains 400 drawings. Each row is one trial: the stimulated
# electrode, the subject, the binary drawing itself, and shape descriptors
# measured from it. We work with Subject 2:

data = fetch_beyeler2019(subjects='S2')

###############################################################################
# [Beyeler2019]_ reports S2's Argus II as implanted at ``(-1331, -850)`` um
# with a rotation of -28.4 degrees, and the optic disc center 16.2 degrees
# nasally and 1.38 degrees superior to the fovea. Those four numbers are the
# subject-specific anatomy everything below depends on:

argus = ArgusII(eye='right')
implant_position = (-1331, -850) * um
implant_rotation = -28.4
loc_od = (16.2, 1.38)

###############################################################################
# Drawings from repeated trials on the same electrode are averaged and drawn at
# the electrode that produced them. This reproduces a panel of Fig. 2 in
# [Beyeler2019]_, with the model's NFB trajectories overlaid:

model = AxonMapModel(argus, loc_od=loc_od)
plot_argus_phosphenes(data, argus, axon_map=model)

###############################################################################
# Phosphenes are not round, and they are not oriented arbitrarily: each one
# runs along the bundle that passes under its electrode. Stimulating an
# electrode activates passing axons, not just the cells beneath it.
#
# Predicting the drawings
# -----------------------
#
# The axon map model formalizes that: current spreads by ``rho`` across
# bundles and by ``lam`` along them. [Beyeler2019]_ fit both per subject; for
# S2, ``rho = 315`` um and ``lam = 500`` um. ``thresh_percept`` is set to
# :math:`1/\sqrt{e}`, the contour at which the paper measured phosphene shape.

model = AxonMapModel(implant=argus, rho=315, lam=500, loc_od=loc_od,
                     implant_position=implant_position,
                     implant_rotation=implant_rotation,
                     xrange=(-30, 30), yrange=(-22.5, 22.5),
                     thresh_percept=1 / np.sqrt(np.e))

###############################################################################
# A stimulus is an (electrodes, time points) array, so an identity matrix
# activates exactly one electrode per frame. Each predicted frame is then the
# percept from one electrode, matching one drawing:

electrodes = data.electrode.unique()
stim = Stimulus(np.eye(len(electrodes)), electrodes=electrodes)
percepts = model.predict_percept(stim)

fig, (ax_data, ax_sim) = plt.subplots(ncols=2, figsize=(15, 5))
plot_argus_phosphenes(data, argus, scale=0.75, ax=ax_data)
plot_argus_simulated_phosphenes(percepts, argus, scale=1.25, ax=ax_sim,
                                implant_position=implant_position,
                                implant_rotation=implant_rotation)
ax_data.set_title('Drawn by S2')
ax_sim.set_title('Predicted by the axon map model')

###############################################################################
# The predicted phosphenes reproduce the orientation and elongation of the
# drawings across the array, which is the claim the model was built to support.
# Individual sizes are not expected to match trial by trial: ``rho`` and
# ``lam`` are single per-subject fits, and the drawings themselves vary between
# repetitions of the same electrode.
#
# Elongation across all subjects
# ------------------------------
#
# If phosphenes were pixels, their elongation would cluster at zero. The
# dataset ships the shape descriptors measured from each drawing, so the claim
# can be checked on all 400 trials at once. ``eccentricity`` here is the shape
# descriptor from the computer-vision literature -- 0 is a circle, 1 an
# infinitesimally thin line -- and has nothing to do with retinal eccentricity:

all_data = fetch_beyeler2019()
all_data.eccentricity.plot(kind='hist')
plt.xlabel('phosphene elongation')
plt.ylabel('number of drawings')

###############################################################################
# Most drawings are elongated, reproducing Fig. 3C of [Beyeler2019]_.
#
# What this does not establish
# ----------------------------
#
# * A drawing is a subjective report produced on a touchscreen, not a
#   measurement of the percept. Shape descriptors inherit that.
# * ``rho``, ``lam``, the optic disc location and the implant placement were
#   fit to each subject. The model is not predictive for a new subject without
#   comparable data.
# * The model describes single-electrode phosphene *shape*. It says nothing
#   about brightness on an absolute scale, temporal dynamics, or how
#   simultaneously stimulated electrodes combine.
