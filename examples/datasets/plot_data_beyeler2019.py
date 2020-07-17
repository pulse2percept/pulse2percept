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

Due to its size (263 MB), the dataset is not included with pulse2percept, but
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
argus = ArgusII(x=-1331, y=-850, rot=-0.495, eye='RE')

###############################################################################
# (We also need to specify the dimensions of the screens that the subject used,
# expressed in degrees of visual angle, so that we can scale the phosphene
# drawing appropriately. This should really be part of the Beyeler dataset and
# will be fixed in a future version.
# For now, we add the necessary columns ourselves.)
import pandas as pd
data = fetch_beyeler2019(subjects='S2')
data['img_x_dva'] = pd.Series([(-30, 30)] * len(data), index=data.index)
data['img_y_dva'] = pd.Series([(-22.5, 22.5)] * len(data), index=data.index)

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
# S2 had their optic disc center located 14 deg nasally, 2.4 deg superior from
# the fovea:

from pulse2percept.models import AxonMapModel
model = AxonMapModel(loc_od=(14, 2.4))
plot_argus_phosphenes(data, argus, axon_map=model)

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
plt.hist(data.eccentricity)
plt.xlabel('phosphene elongation')
plt.ylabel('count')

###############################################################################
# Phosphenes are not pixels!
# And we have just reproduced Fig. 3C of [Beyeler2019]_.
