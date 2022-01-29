"""
===============================================================================
 Data from Perez Fornos et al. (2012)
===============================================================================

This example shows how to use the Perez Fornos et al. (2012) dataset.

[Fornos2012]_ had subjects record stimulated brightness via joystick position in nine Argus II users.

.. important ::

	You will need to install `Pandas <https://pandas.pydata.org>`_
	(``pip install pandas``) for this dataset.

Loading the dataset
-------------------

The dataset can be loaded as a Pandas ``DataFrame``:
"""
# sphinx_gallery_thumbnail_number = 3

from pulse2percept.datasets import load_fornos2012

data = load_fornos2012()
print(data)

###############################################################################
# Inspecting the DataFrame tells us that there are 45 measurements
# (the rows) each with 3 different attributes (the columns).
#
# These attributes include specifiers such as "subject", "figure", and
# "time_series". We can print all column names using:

data.columns

###############################################################################
# .. note ::
#
#     The meaning of all column names is explained in the docstring of
#     the :py:func:`~pulse2percept.datasets.load_fornos2012` function.
#
# For example, "subject" corresponds to each subject used in the paper:

data.subject.unique()

###############################################################################
# To select all the rows where the subject frequency was 'S2', we can index into the DataFrame as
# follows:

print(data[data.subject == 'S2'])

###############################################################################
# However, we can do the same operation when loading in the data as follows:
print(load_fornos2012(subjects='S2'))

###############################################################################
# This leaves us with 4 rows, each pertaining to subject 'S2'.


###############################################################################
# .. note ::
#
#     Please see the documentation for :py:func:`~pulse2percept.datasets.load_fornos2012`
#     to see all available parameters for data subset loading.
#

###############################################################################
# Plotting the data
# -----------------
#
# One of the important points in the paper is how the perceptual response varies amongst subjects.
# Let us recreate figure 3, for subject 2 to an example of this difference.

import matplotlib.pyplot as plt
import numpy as np

# Load in each figures data
figure_3_s1_data = load_fornos2012(figures='fig3_S1')['time_series'][0]
figure_3_s2_data = load_fornos2012(figures='fig3_S2')['time_series'][0]

# Set up the subplot
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
plt.xlim(-1, 30)
plt.ylim(-11, 11)
fig.suptitle('Average Joystick Response vs. Time')

time_steps = np.arange(start=-2.0, stop=75.5, step=0.5)

# Set up the pulse
pulse_value = []
for pulse_step in time_steps:
    if 10 >= pulse_step > 0:
        pulse_value.append(10)
    else:
        pulse_value.append(0)

# Plot subject 1
ax1.plot(time_steps, figure_3_s1_data, 'r', label='S1 Joystick Response')
ax1.step(time_steps, pulse_value, 'k--', label='Pulse')
ax1.spines['bottom'].set_position('center')
ax1.set_title('S1')
ax1.legend(loc='lower right')

# Plot subject 2
ax2.plot(time_steps, figure_3_s2_data, label='S2 Joystick Response')
ax2.step(time_steps, pulse_value, 'k--', label='Pulse')
ax2.spines['bottom'].set_position('center')
ax2.set_title('S2')
ax2.legend(loc='lower right')
