# -*- coding: utf-8 -*-
"""
===============================================================================
van der Grinten et al. (2023): Cortical phosphene dynamics
===============================================================================

A cortical phosphene does not switch on and stay on. [vanderGrinten2023]_
models it as the output of a charge-accumulation process: stimulation drives a
tissue activation trace, the trace has to cross a threshold before anything is
seen, and brightness then follows the trace through a sigmoid. The consequence
is that phosphene brightness rises over a few hundred milliseconds and then
fades under sustained stimulation, and that weak stimulation produces nothing
at all.

This example applies
:py:class:`~pulse2percept.models.cortex.DynaphosModel` to an
:py:class:`~pulse2percept.implants.cortex.Orion` epicortical array and
reproduces the brightness-over-time family of Fig. 3 of that paper. The
reference implementation is available `here
<https://github.com/neuralcodinglab/dynaphos>`_.
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
# Unlike the retinal models, Dynaphos is a single composite model rather than
# separable spatial and temporal components, and it is defined for V1 only.
# Phosphenes are small compared with the simulated field, so the visual field
# is sampled finely:

implant = Orion()
model = DynaphosModel(implant=implant, step=0.05)
model.build()

model.plot(show_implant=True)

###############################################################################
# Each electrode maps to one visual-field location through cortical
# retinotopy, so the array's arrangement on cortex is not the arrangement of
# the phosphenes it produces.
#
# The percept of a sustained train
# --------------------------------
#
# Dynaphos requires stimuli with a time course. Here every electrode receives
# the same 300 Hz, 0.17 ms biphasic pulse train at 100 uA for 2 s:

stim = {e: BiphasicPulseTrain(amp=100, freq=300, phase_dur=0.17,
                              stim_dur=2000)
        for e in implant.electrode_names}

percept = model.predict_percept(stim)

plt.figure()
plt.imshow(percept.max(axis='frames'), cmap='gray')
plt.title('Brightest frame')

###############################################################################
# Following the brightest pixel over time shows the accumulate-then-fade
# behavior the model is built around: brightness peaks early and decays while
# stimulation continues.

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
# Repeating a shorter 166 ms train across stimulation amplitudes reproduces the
# family of brightness traces in Fig. 3 of [vanderGrinten2023]_. Low amplitudes
# never cross the tissue activation threshold ``a_thr`` and produce no
# phosphene at all:

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
# The curves above are the **generated percept brightness**, which is bounded
# by phosphene size and by the tissue activation threshold.
# [vanderGrinten2023]_ plots the model's **internal brightness state**, and
# draws it dashed at the time points where activation stayed below threshold
# and no phosphene was generated. The two agree where a phosphene exists;
# below threshold this figure reads zero where the paper's reads a dashed
# continuation.
#
# What this does not establish
# ----------------------------
#
# * Dynaphos is calibrated against phosphene reports from a small number of
#   participants with cortical implants, and its brightness is in arbitrary
#   units.
# * The model covers V1 only. V2 and V3 stimulation is not represented.
# * Cortical retinotopy here is the population-average
#   :py:class:`~pulse2percept.topography.cortex.Polimeni2006Map`. Individual
#   retinotopy varies substantially and changes where every phosphene lands.
# * Stimulating all 60 Orion electrodes simultaneously is a modeling
#   convenience; real systems raster their electrodes and interactions between
#   simultaneously stimulated sites are not modeled.
