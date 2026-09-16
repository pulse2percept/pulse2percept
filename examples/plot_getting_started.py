# -*- coding: utf-8 -*-
"""
===============================================================================
Getting Started
===============================================================================

A pulse2percept simulation has three main pieces:

1. an **implant** describing the implanted device,
2. a **stimulus** describing what the implant delivers, and
3. a **model** describing how stimulation becomes a percept.

The same workflow applies across supported retinal and cortical implants and
models.

Your first retinal percept
--------------------------

Start with an Argus II retinal implant and the biphasic axon-map model
[Granley2021]_. Unlike a simple Gaussian model, it captures axonal streaks and
lets pulse amplitude, frequency, and phase duration change predicted
brightness, phosphene size, and streak length.
"""
# sphinx_gallery_thumbnail_number = 1

import matplotlib.pyplot as plt

import pulse2percept as p2p
from pulse2percept.units import Hz, ms, xTh


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
# A cortical example
# ------------------
#
# Cortical simulations follow the same pattern: choose a cortical implant,
# pair it with a cortical model, and provide a stimulus.

cortical_implant = p2p.implants.cortex.Cortivis()
cortical_model = p2p.models.cortex.ScoreboardModel(cortical_implant)

stim = ...
percept = cortical_model.predict_percept(stim)
percept.plot()
plt.title('Cortivis: one stimulated electrode')
plt.show()

###############################################################################
# You can adapt each part of the simulation:
#
# * **Implant:** use a different retinal or cortical implant, change retinal
#   laterality or cortical hemisphere, provide measured electrode thresholds,
#   or attach a different encoder.
# * **Model:** choose a model appropriate for the implant and question. For
#   example, retinal scoreboard models produce round phosphenes, whereas
#   axon-map models capture elongated retinal phosphenes.
# * **Stimulus:** change the electrode, pulse amplitude, frequency, phase
#   duration, or stimulation pattern. Images and videos can also be encoded
#   into stimulation through the implant.
#
# Where to go next
# ----------------
#
# See :ref:`core concepts <topics-index>` for more on implants, stimulation,
# models, visual input, coordinates, and units. See the
# :ref:`example gallery <examples>` for complete workflows.