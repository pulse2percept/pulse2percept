# -*- coding: utf-8 -*-
"""
===============================================================================
Getting Started
===============================================================================

A pulse2percept simulation has three main pieces:

1. an **implant** describing the implanted device,
2. a **stimulus** describing what the implant delivers, and
3. a **model** describing how stimulation becomes a percept.

The pieces are deliberately swappable.

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
retinal_model = p2p.models.retina.BiphasicAxonMapModel(
    retinal_implant,
    rho=200,
    lam=800,
)

stim = {
    'C5': p2p.stimuli.BiphasicPulseTrain(
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
# That is the whole user-facing workflow:
#
# ``implant + model + stimulus -> percept``
#
# Each piece owns a different scientific choice:
#
# * **Implant:** swap ``ArgusII`` for another retinal device, choose
#   ``eye='left'`` or ``'right'``, provide measured electrode thresholds, or
#   attach a different image/video encoder.
# * **Model:** change ``rho`` (spread across axons), ``lam`` (spread along
#   axons), the visual-field map, or the model-side implant placement. Swap in
#   :class:`~pulse2percept.models.retina.BiphasicScoreboardModel` if your
#   scientific question calls for round rather than axon-shaped phosphenes.
# * **Stimulus:** change pulse amplitude, frequency, phase duration, electrode,
#   or stimulation pattern.
#
# Where to go next
# ----------------
#
# The :ref:`basic concepts <topics-index>` explain implants, stimuli, models,
# encoders, units, and other pieces in more detail. The
# :ref:`example gallery <examples>` contains complete scientific
# workflows.
