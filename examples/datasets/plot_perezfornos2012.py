"""
===============================================================================
Phosphene fading data from Perez Fornos et al. (2012)
===============================================================================

This example shows how to use the Perez Fornos et al. (2012) dataset.

[PerezFornos2012]_ had nine Argus II users report perceived phosphene brightness
via joystick position.

.. important ::

    You will need to install `Pandas <https://pandas.pydata.org>`_
    (``pip install pandas``) for this dataset.

Loading the dataset
-------------------

The dataset can be loaded as a Pandas ``DataFrame``:
"""
# sphinx_gallery_thumbnail_number = 1

import numpy as np
import matplotlib.pyplot as plt
from pulse2percept.datasets import load_perezfornos2012

data = load_perezfornos2012()
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
# To select all the rows where the subject frequency was 'S2', we can index
# into the DataFrame as follows:

print(data[data.subject == 'S2'])

###############################################################################
# However, we can do the same operation when loading in the data as follows:

print(load_perezfornos2012(subjects='S2'))

###############################################################################
# This leaves us with 4 rows, each pertaining to subject 'S2'.
#
# .. note ::
#
#     Please see the documentation for
#     :py:func:`~pulse2percept.datasets.load_perezfornos2012` to see all
#     available parameters for data subset loading.
#
# Plotting the data
# -----------------
#
# One of the important points in the paper is how the perceptual response varies
# amongst subjects.
# Let us recreate Figure 3, for Subject 2, to show an example of this
# difference:

# Load in each figures data:
figure_3_s1_data = load_perezfornos2012(figures='fig3_S1')['time_series'][0]
figure_3_s2_data = load_perezfornos2012(figures='fig3_S2')['time_series'][0]

###############################################################################
# Stimuli were delivered over a 10 second window. To recreate this, we build
# a vector of time points (``time_steps``) and set stimulation value to 10
# within the first 10 seconds, and 0 otherwise:

time_steps = np.arange(start=-2.0, stop=75.5, step=0.5)

# Set up the pulse:
pulse_value = []
for pulse_step in time_steps:
    if 10 >= pulse_step > 0:
        pulse_value.append(10)
    else:
        pulse_value.append(0)

###############################################################################
# Now we can plot the stimulus alongside the subject's response:

# Set up the subplot:
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
plt.xlim(-1, 30)
plt.ylim(-11, 11)
fig.suptitle('Average Joystick Response vs. Time')

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

fig.tight_layout()

###############################################################################
# The above plot suggests that Subject 1 saw phosphenes gradually get dimmer
# over the stimulus duration. Once the stimulus was removed, phosphene
# brightness was quickly reduced to zero.
#
# For Subject 2, the phosphene started bright (t=0) but rapidly went dark,
# even darker than the background (y-value < 0). Then the phosphene slowly
# got brighter again. Upon stimulus removal, the subject reported seeing
# another flash of light.
#
# The reason for these differences is currently unknown, but might have to do
# with retinal adaptation.
