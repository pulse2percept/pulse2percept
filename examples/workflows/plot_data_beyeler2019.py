# -*- coding: utf-8 -*-
"""
===============================================================================
Beyeler et al. (2019): Phosphenes are not pixels
===============================================================================

Simulated prosthetic vision is often drawn as one dot per electrode.
[Beyeler2019]_ tested this: Argus I and II users drew what they saw during
single-electrode stimulation. Phosphenes were **elongated**, and oriented
along the retinal nerve fiber bundles (NFBs) passing under the electrode.

This example reproduces both results from the published data and predicts the
drawings with a subject-specific
:py:class:`~pulse2percept.models.retina.AxonMapModel`.

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
# The dataset contains 400 drawings. Each row is one trial: electrode,
# subject, binary drawing, and shape descriptors. This example uses subject S2:

data = fetch_beyeler2019(subjects='S2')

###############################################################################
# [Beyeler2019]_ reports S2's Argus II at ``(-1331, -850)`` um, rotated
# -28.4 deg, with the optic disc center 16.2 dva nasal and 1.38 dva superior
# to the fovea:

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
# Each phosphene runs along the bundle under its electrode: stimulation
# activates passing axons, not only the cells beneath the electrode.
#
# Predicting the drawings
# -----------------------
#
# In the axon map model, activation spreads by ``rho`` across bundles and by
# ``lam`` along them. The per-subject fits for S2 are ``rho = 315`` um and
# ``lam = 500`` um. ``thresh_percept`` is :math:`1/\sqrt{e}`, the contour at
# which the paper measured shape.

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
# The predictions reproduce the orientation and elongation of the drawings.
# Sizes do not match trial by trial: ``rho`` and ``lam`` are per-subject fits,
# and drawings vary across repetitions of the same electrode.
#
# Elongation across all subjects
# ------------------------------
#
# Round phosphenes would cluster at zero elongation. ``eccentricity`` is the
# shape descriptor (0: circle, 1: line), not retinal eccentricity:

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
