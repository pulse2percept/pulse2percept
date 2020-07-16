"""
===============================================================================
Threshold data from Horsager et al. (2009)
===============================================================================

This example shows how to use the Horsager et al. (2009) dataset.

pulse2percept provides all the threshold data published in [Horsager2009]_.
This paper used a set of psychophysical detection tasks to determine perceptual
thresholds in Argus I users.
A typical task involved turning on a single electrode in the array, and asking
the subject if they saw something. If the subject did not detect a visual
percept (i.e., a phosphene), the stimulus amplitude was increased.
Using a staircase procedure, the researchers were able to determine the current
at which subjects were able to detect a phosphene 50% of the time.
This current is called the threshold current, and it is stored in the
"stim_amp" column of the dataset.

.. note ::

    Mean threshold currents were extracted from the main publication and its
    supplementary materials using Webplot Digitizer.
    Therefore, these data are provided without any guarantee of correctness or
    completeness.

Loading the dataset
-------------------

The dataset can be loaded as a Pandas ``DataFrame``:
"""
# sphinx_gallery_thumbnail_number = 3

from pulse2percept.datasets import load_horsager2009
data = load_horsager2009()
print(data)

###############################################################################
# Inspecting the DataFrame tells us that there are 552 threshold measurements
# (the rows) each with 21 different attributes.
#
# These attributes include specifiers such as "subject", "electrode", and
# "stim_type". We can print all column names using:

data.columns

###############################################################################
# .. note ::
#
#     The meaning of all column names is explained in the docstring of
#     the :py:func:`~pulse2percept.datasets.load_horsager2009` function.
#
# For example, "stim_type" corresponds to the different stimulus types that
# were used in the paper:

list(data.stim_type.unique())

###############################################################################
# The first entry, "single_pulse", corresponds to the single biphasic pulse
# used to produce Figure 3. We can verify which figure a data point came from
# by inspecting the "source" column.
#
# To select all the "single_pulse" rows, we can index into the DataFrame as
# follows:

print(data[data.stim_type == 'single_pulse'])

###############################################################################
# This leaves us with 80 rows, some of which come from Figure 3, others from
# Figure S3.1 in the supplementary material.
#
# An alternative to indexing into the DataFrame is to load only a subset of
# the data:

load_horsager2009(stim_types='single_pulse')

###############################################################################
# Plotting the data
# -----------------
#
# Arguably the most important column is "stim_amp". This is the current
# amplitude of the different stimuli (single pulse, pulse trains, etc.) used
# at threshold.
#
# We might be interested in seeing how threshold amplitude varies as a function
# of pulse duration. We could either use Matplotlib to generate a scatter plot
# or use pulse2percept's own visualization function:

from pulse2percept.viz import scatter_correlation
scatter_correlation(data.pulse_dur, data.stim_amp)

###############################################################################
# :py:func:`~pulse2percept.viz.scatter_correlation` above generates a scatter
# plot of the stimulus amplitude as a function of pulse duration, and performs
# linear regression to calculate a correlation $r$ and a $p$ value.
# As expected from the literature, now it becomes evident that stimulus
# amplitude is negatively correlated with pulse duration (no matter the exact
# stimulus used).
#
# Recreating the stimuli
# ----------------------
#
# To recreate the stimulus used to obtain a specific data point, we need to use
# the values specified in the different columns of a particular row.
#
# For example, the first row of the dataset specifies the stimulus used to
# obtain the threshold current for subject S05 on Electrode C3 using a single
# biphasic pulse:

row = data.loc[0, :]
print(row)

###############################################################################
# Using :py:class:`~pulse2percept.stimuli.BiphasicPulse`, we can create the
# pulse to specifications:

from pulse2percept.stimuli import BiphasicPulse
stim = BiphasicPulse(row['stim_amp'], row['pulse_dur'],
                     interphase_dur=row['interphase_dur'],
                     stim_dur=row['stim_dur'], electrode=row['electrode'])
stim.plot()
