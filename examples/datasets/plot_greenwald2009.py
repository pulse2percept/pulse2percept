"""
===============================================================================
 Data from Greenwald et al. (2009)
===============================================================================

This example shows how to use the Greenwald et al. (2009) dataset.

[Greenwald2009]_ investigated the relationship between electrical stimulation amplitude
and phosphene brightness in two Argus I users.

.. important ::

	You will need to install `Pandas <https://pandas.pydata.org>`_
	(``pip install pandas``) for this dataset.

Loading the dataset
-------------------

The dataset can be loaded as a Pandas ``DataFrame``:
"""
# sphinx_gallery_thumbnail_number = 4

from pulse2percept.datasets import load_greenwald2009
data = load_greenwald2009()
print(data)

###############################################################################
# Inspecting the DataFrame tells us that there are 83 measurements
# (the rows) each with 12 different attributes (the columns).
#
# These attributes include specifiers such as "subject", "electrode", and
# "task". We can print all column names using:
print(data.columns)

###############################################################################
# .. note ::
#
#     The meaning of all column names is explained in the docstring of
#     the :py:func:`~pulse2percept.datasets.load_greenwald2009` function.
#
# For example, "amp" corresponds to the amplitude of the stimulation used
# in a particular measurement:

data.amp.unique()

###############################################################################
# To select all the rows where the same subject was used, such as 'S05'
# we can index into the DataFrame as follows:

print(data[data.subject == 'S05'])

###############################################################################
# Likewise, we can perform the same operation when initially loading the data
# as follows:

print(load_greenwald2009(subjects='S05'))

# .. note ::
#
#     Please see the documentation for :py:func:`~pulse2percept.datasets.load_greenwald2009`
#     to see all available parameters for data subset loading.
#

###############################################################################
# Plotting the data
# -----------------
#
# To see the relationship between electrical stimulation amplitude and phosphene brightness,
# let us demonstrate partially recreating part of Figure 2 from the paper. Specifically,
# we will look at subject S06 and electrode C4. Please note, we omit the power fits in this demonstration.

import matplotlib.pyplot as plt
data = load_greenwald2009(subjects='S06', electrodes='C4')

# Adjust the x-axis scaling, and add title
plt.xlim(0, 400)
plt.xticks(ticks=[0, 200, 400])
plt.xlabel("Current (µA)")

# Adjust the y-axis scaling, and add title
plt.ylabel("Rating")

plt.title("S06")
plt.ylim(0, 25)

# Add figure title
plt.title("S06 Current Amplitude vs Brightness, Electrode C4")

# Plot the data
plt.scatter(data.amp, data.brightness)
