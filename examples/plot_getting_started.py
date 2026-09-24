# -*- coding: utf-8 -*-
"""
===============================================================================
Quickstart
===============================================================================

A pulse2percept simulation has three parts:

1. an **implant**: the device and its electrodes,
2. a **stimulus**: what is delivered to the electrodes, and
3. a **model**: how stimulation becomes a predicted percept.

Retinal and cortical simulations use the same pattern.

A retinal percept
-----------------

An Argus II epiretinal implant with the biphasic axon map model
[Granley2021]_. Phosphenes are elongated along the nerve fiber bundles, and
their brightness, size, and streak length depend on pulse amplitude,
frequency, and phase duration. Amplitude is given as a multiple of perceptual
threshold (``xTh``).
"""
# sphinx_gallery_thumbnail_number = 1

import matplotlib.pyplot as plt

import pulse2percept as p2p
from pulse2percept.units import Hz, mm, ms, uA, xTh

retinal_implant = p2p.implants.retina.ArgusII()
retinal_model = p2p.models.retina.BiphasicAxonMapModel(retinal_implant)

stim = {
    'A5': p2p.stimuli.BiphasicPulseTrain(
        freq=20 * Hz,
        amp=2 * xTh,
        phase_dur=0.45 * ms,
    )
}

percept = retinal_model.predict_percept(stim)
percept.plot()
plt.title('Argus II: one stimulated electrode')
plt.show()

###############################################################################
# A cortical percept
# ------------------
#
# A Cortivis intracortical array (96 electrodes, 400 um spacing) with the
# cortical scoreboard model. Electrode coordinates are device-local, so the
# model's ``implant_position`` places the array on V1. ``(20, -5) * mm`` is
# measured from the foveal representation of the right hemisphere, so the
# phosphenes appear about 2 dva into the left visual field.
#
# Cortical magnification is high this close to the fovea, so a single
# phosphene is under 0.1 dva wide. Every electrode therefore receives the same
# pulse train, and the simulated field is zoomed in on the array.

cortical_implant = p2p.implants.cortex.Cortivis()
cortical_model = p2p.models.cortex.ScoreboardModel(
    cortical_implant,
    implant_position=(20, -5) * mm,
    xrange=(-3, -1),  # dva
    yrange=(0, 2),    # dva
    step=0.02,        # dva
)

train = p2p.stimuli.BiphasicPulseTrain(
    freq=20 * Hz,
    amp=100 * uA,
    phase_dur=0.45 * ms,
)
stim = {electrode: train for electrode in cortical_implant.electrode_names}

percept = cortical_model.predict_percept(stim)
percept.plot()
plt.title('Cortivis: all 96 electrodes')
plt.show()

###############################################################################
# Changing the simulation
# -----------------------
#
# * **Implant:** another retinal or cortical device, the implanted eye or
#   hemisphere, measured electrode thresholds, or a different encoder.
# * **Stimulus:** electrode, amplitude, frequency, phase duration, or an image
#   or video that the implant encodes into pulses.
# * **Model:** chosen for the implant and the question. Scoreboard models
#   produce round phosphenes; axon map models produce elongated retinal
#   phosphenes.
#
# Next steps
# ----------
#
# The Core Concepts pages cover each part in order, starting with
# :ref:`topics-implants`. The :ref:`key examples <examples-workflows>` show
# complete workflows with images, video, and encoders.
