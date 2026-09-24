# -*- coding: utf-8 -*-
"""
===============================================================================
van der Grinten et al. (2023): Cortical phosphene dynamics
===============================================================================

[vanderGrinten2023]_ models a cortical phosphene as the output of charge
accumulation: stimulation drives a tissue activation trace, the trace must
cross a threshold, and brightness follows it through a sigmoid. Brightness
therefore rises over a few hundred ms, fades under sustained stimulation, and
stays at zero for weak stimulation.

This example applies :py:class:`~pulse2percept.models.cortex.DynaphosModel`
to an :py:class:`~pulse2percept.implants.cortex.Orion` epicortical array and
reproduces the brightness-over-time family of Fig. 3. The authors' reference
implementation is `on GitHub <https://github.com/neuralcodinglab/dynaphos>`_.
"""
# sphinx_gallery_thumbnail_number = 3

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from pulse2percept.implants.cortex import Orion
from pulse2percept.models.cortex import DynaphosModel
from pulse2percept.stimuli import BiphasicPulseTrain

###############################################################################
# Where the array sits
# --------------------
#
# Dynaphos is one composite model, not separate spatial and temporal
# components, and covers V1 only. Phosphenes are small relative to the field,
# so the grid is fine (0.05 dva):

implant = Orion()
model = DynaphosModel(implant=implant, step=0.05)
model.build()

model.plot(show_implant=True)

###############################################################################
# Cortical retinotopy distorts the array layout: the phosphene arrangement
# differs from the electrode arrangement on cortex.
#
# The percept of a sustained train
# --------------------------------
#
# Dynaphos requires stimuli with a time course. Every electrode receives the
# same biphasic pulse train: 300 Hz, 0.17 ms phases, 100 uA, 2 s:

stim = {e: BiphasicPulseTrain(amp=100, freq=300, phase_dur=0.17,
                              stim_dur=2000)
        for e in implant.electrode_names}

percept = model.predict_percept(stim)

plt.figure()
plt.imshow(percept.max(axis='frames'), cmap='gray')
plt.title('Brightest frame')

###############################################################################
# The brightest pixel over time peaks early and decays while stimulation
# continues:

delivered = implant.prepare_stim(stim)
brightness = percept.data.max(axis=(0, 1))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(delivered.time,
        -0.02 + 0.01 * delivered.data[0, :] / delivered.data.max(),
        linewidth=2, label='pulse train')
ax.plot(percept.time, brightness, linewidth=2, label='percept')
ax.axhline(percept.max(), color='k', linestyle='--', label='max brightness')
ax.axhline(0, color='k')
ax.set_xlabel('time (ms)')
ax.set_ylabel('predicted brightness (a.u.)')
ax.set_xlim(0, 2000)
ax.legend(loc='center right')
fig.tight_layout()

###############################################################################
# Brightness over time vs amplitude (Fig. 3)
# ------------------------------------------
#
# A 166 ms train at 10-100 uA reproduces the traces of Fig. 3 in
# [vanderGrinten2023]_. Low amplitudes never cross the activation threshold
# ``a_thr`` and produce no phosphene:

amps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
t_percept = np.arange(0, 700, 20)

traces = []
for amp in amps:
    trial = {e: BiphasicPulseTrain(freq=300, amp=amp, phase_dur=0.17,
                                   stim_dur=166)
             for e in implant.electrode_names}
    traces.append(model.predict_percept(
        trial, t_percept=t_percept).data.max(axis=(0, 1)))

fig, ax = plt.subplots(figsize=(8, 3.5))
cmap = mpl.cm.YlOrBr
norm = mpl.colors.Normalize(vmin=0, vmax=100)
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8,
             ticks=[0, 100], label='stimulus amplitude (uA)')

for amp, trace in zip(amps, traces):
    ax.plot(t_percept / 1000, trace, color=cmap(amp / 100), linewidth=3)

ax.set_xlabel('time (s)')
ax.set_ylabel('brightness')
ax.set_xlim(0, 0.7)
ax.set_ylim(0, 1.1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()

###############################################################################
# Correspondence with the publication
# -----------------------------------
#
# These curves show percept brightness. [vanderGrinten2023]_ plots the
# model's internal brightness state, dashed where activation is below
# threshold. The two agree where a phosphene exists; below threshold this
# figure shows zero where the paper shows a dashed line.
#
# What this does not establish
# ----------------------------
#
# * Dynaphos is calibrated against reports from a few participants with
#   cortical implants. Brightness is in arbitrary units.
# * V1 only; V2 and V3 stimulation is not modeled.
# * Retinotopy is the population-average
#   :py:class:`~pulse2percept.topography.cortex.Polimeni2006Map`. Individual
#   retinotopy varies substantially.
# * All 60 electrodes are stimulated simultaneously for simplicity. Real
#   systems raster their electrodes, and interactions between simultaneously
#   stimulated sites are not modeled.
