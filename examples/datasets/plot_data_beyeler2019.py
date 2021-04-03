"""
===============================================================================
Phosphene drawings from Beyeler et al. (2019)
===============================================================================

This example shows how to use the Beyeler et al. (2019) dataset.

[Beyeler2019]_ asked Argus I/II users to draw what they see in response to
single-electrode stimulation.

.. important ::

    For this dataset you will need to install both
    `Pandas <https://pandas.pydata.org>`_ (``pip install pandas``) and
    `HDF4 for Python <https://www.h5py.org>`_ (``pip install h5py``).

Loading the dataset
-------------------

Due to its size (66 MB), the dataset is not included with pulse2percept, but
can be downloaded from the Open Science Framework (OSF).

By default, the dataset will be stored in a local directory
‘~/pulse2percept_data/’ within your user directory (but a different path can be
specified).
This way, the datasets is only downloaded once, and future calls to the fetch
function will load the dataset from your local copy.

The data itself will be provided as a Pandas ``DataFrame``:

"""
# sphinx_gallery_thumbnail_number = 2

from pulse2percept.datasets import fetch_beyeler2019

data = fetch_beyeler2019()
print(data)

###############################################################################
#
# Inspecting the DataFrame tells us that there are 400 phosphene drawings
# (the rows) each with 16 different attributes (the columns).
#
# These attributes include specifiers such as "subject", "electrode", and
# "image". We can print all column names using:

data.columns

###############################################################################
# .. note ::
#
#     The meaning of all column names is explained in the docstring of
#     the :py:func:`~pulse2percept.datasets.fetch_beyeler2019` function.
#
# For example, "subject" contains the different subject IDs used in the study:

data.subject.unique()

###############################################################################
# To select all drawings from Subject 2, we can index into the DataFrame as
# follows:

print(data[data.subject == 'S2'])

###############################################################################
# This leaves us with 110 rows, each of which correspond to one phosphene
# drawings from a number of different electrodes and trials.
#
# An alternative to indexing into the DataFrame is to load only a subset of
# the data:

print(fetch_beyeler2019(subjects='S2'))

###############################################################################
# Plotting the data
# -----------------
#
# Arguably the most important column is "image". This is the phosphene drawing
# obtained during a particular trial.
#
# Each phosphene drawing is a 2D black-and-white NumPy array, so we can just
# plot it using Matplotlib like any other image:

import matplotlib.pyplot as plt
plt.imshow(data.loc[0, 'image'], cmap='gray')

###############################################################################
# However, we might be more interested in seeing how phosphene shape differs
# for different electrodes.
# For this we can use :py:func:`~pulse2percept.viz.plot_argus_phosphenes` from
# the :py:mod:`~pulse2percept.viz` module.
# In addition to the ``data`` matrix, the function will also want an
# :py:class:`~pulse2percept.implants.ArgusII` object implanted at the correct
# location.
#
# Consulting [Beyeler2019]_ tells us that the prosthesis was roughly implanted
# in the following location:

from pulse2percept.implants import ArgusII
argus = ArgusII(x=-1331, y=-850, rot=-28.4, eye='RE')

###############################################################################
# For now, let's focus on the data from Subject 2:

data = fetch_beyeler2019(subjects='S2')

###############################################################################
# Passing both ``data`` and ``argus`` to
# :py:func:`~pulse2percept.viz.plot_argus_phosphenes` will then allow the
# function to overlay the phosphene drawings over a schematic of the implant.
# Here, phosphene drawings from different trials are averaged, and aligned with
# the center of the electrode that was used to obtain the drawing:

from pulse2percept.viz import plot_argus_phosphenes
plot_argus_phosphenes(data, argus)

###############################################################################
# Great! We have just reproduced a panel from Figure 2 in [Beyeler2019]_.
#
# As [Beyeler2019]_ went on to show, the orientation of these phosphenes is
# well aligned with the map of nerve fiber bundles (NFBs) in each subject's
# eye.
#
# To see how the phosphene drawings line up with the NFBs, we can also pass an
# :py:class:`~pulse2percept.models.AxonMapModel` to the function.
# Of course, we need to make sure that we use the correct dimensions. Subject
# S2 had their optic disc center located 16.2 deg nasally, 1.38 deg superior
# from the fovea:

from pulse2percept.models import AxonMapModel
model = AxonMapModel(loc_od=(16.2, 1.38))
plot_argus_phosphenes(data, argus, axon_map=model)

###############################################################################
# Predicting phosphene shape
# --------------------------
#
# In addition, the :py:class:`~pulse2percept.models.AxonMapModel` is well
# suited to predict the shape of individual phosphenes. Using the values given
# in [Beyeler2019]_, we can tailor the axon map parameters to Subject 2:

import numpy as np
model = AxonMapModel(rho=315, axlambda=500, loc_od=(16.2, 1.38),
                     xrange=(-30, 30), yrange=(-22.5, 22.5),
                     thresh_percept=1 / np.sqrt(np.e))
model.build()

###############################################################################
# Now we need to activate one electrode at a time, and predict the resulting
# percept. We could build a :py:class:`~pulse2percept.stimuli.Stimulus` object
# with a for loop that does just that, or we can use the following trick.
#
# The stimulus' data container is a (electrodes, timepoints) shaped 2D NumPy
# array. Activating one electrode at a time is therefore the same as an
# identity matrix whose size is equal to the number of electrodes. In code:

# Find the names of all the electrodes in the dataset:
electrodes = data.electrode.unique()
# Activate one electrode at a time:
import numpy as np
from pulse2percept.stimuli import Stimulus
argus.stim = Stimulus(np.eye(len(electrodes)), electrodes=electrodes)

###############################################################################
# Using the model's
# :py:func:`~pulse2percept.models.AxonMapModel.predict_percept`, we then get
# a Percept object where each frame is the percept generated from activating
# a single electrode:

percepts = model.predict_percept(argus)
percepts.play()

###############################################################################
# Finally, we can visualize the ground-truth and simulated phosphenes
# side-by-side:

from pulse2percept.viz import plot_argus_simulated_phosphenes
fig, (ax_data, ax_sim) = plt.subplots(ncols=2, figsize=(15, 5))
plot_argus_phosphenes(data, argus, scale=0.75, ax=ax_data)
plot_argus_simulated_phosphenes(percepts, argus, scale=1.25, ax=ax_sim)
ax_data.set_title('Ground-truth phosphenes')
ax_sim.set_title('Simulated phosphenes')

###############################################################################
# Analyzing phosphene shape
# -------------------------
#
# The phosphene drawings also come annotated with different shape descriptors:
# area, orientation, and elongation.
# Elongation is also called eccentricity in the computer vision literature,
# which is not to be confused with retinal eccentricity. It is simply a number
# between 0 and 1, where 0 corresponds to a circle and 1 corresponds to an
# infinitesimally thin line (note that the Methods section of [Beyeler2019]_
# got it wrong).
#
# [Beyeler2019]_ made the point that if each phosphene could be considered a
# pixel (or essentially a blob), as is so often assumed in the literature, then
# most phosphenes should have zero elongation.
#
# Instead, using Matplotlib's histogram function, we can convince ourselves
# that most phosphenes are in fact elongated:

data = fetch_beyeler2019()
data.eccentricity.plot(kind='hist')
plt.xlabel('phosphene elongation')

###############################################################################
# Phosphenes are not pixels!
# And with that we have just reproduced Fig. 3C of [Beyeler2019]_.
