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

A single-electrode phosphene
----------------------------

As an epiretinal implant, Argus II elicits phosphenes that are elongated along
the underlying nerve fiber bundle [Beyeler2019]_.
This is best modeled by the biphasic axon map model [Granley2021]_, which
models phosphene appearance (shape, brightness, size) as a function of pulse
amplitude (given as a multiple of perceptual threshold, ``xTh``),
frequency (``Hz``), and pulse duration (``ms``).
"""
# sphinx_gallery_thumbnail_number = 1

import matplotlib.pyplot as plt

import pulse2percept as p2p
from pulse2percept.units import Hz, mm, ms, uA, xTh, dva

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
# A subretinal percept with residual vision
# -----------------------------------------
#
# Visual input can also be defined as a
# :py:class:`~pulse2percept.vision.Scene`: an image or video with a known field
# of view. A scene can include a :py:class:`~pulse2percept.vision.Scotoma` to
# represent where native vision has been lost.
#
# Here, a PRIMA photovoltaic implant restores part of a central scotoma while
# viewing a video recorded on the UCSB campus.
# :py:class:`~pulse2percept.implants.retina.PRIMAPivotal` uses a
# :py:class:`~pulse2percept.stimuli.PRIMAEncoder` by default, which samples the
# scene at the implant pixels and converts image intensity into the pulsed
# near-infrared illumination used by the device.

video = p2p.stimuli.samples.ucsb_pedestrians(resize=(173, 320))
scotoma = p2p.vision.Scotoma.circle(5 * dva)
scene = p2p.vision.Scene(
    video, fov=40 * dva, 
    scotoma=scotoma, 
    scotoma_fill=0,
)
prima = p2p.implants.retina.PRIMAPivotal()
prima_model = p2p.models.retina.Ho2018Model(
    prima, 
    xrange=(-6 * dva, 6 * dva), 
    yrange=(-6 * dva, 6 * dva), 
    step=0.1 * dva,
)
percept = prima_model.predict_percept(scene, gaze=(0, 0) * dva)
scene.render(
    percept=percept,
    gaze=(0, 0) * dva,
    vmax=percept.data.max(),
)
plt.title('PRIMA percept inside a central scotoma')
plt.show()

###############################################################################
# A cortical percept
# ------------------
#
# For cortical implants, electrode locations must be mapped from cortex to the
# visual field. Here we use the population-average
# :py:class:`~pulse2percept.topography.cortex.Polimeni2006Map` [Polimeni2006]_
# and simulate V1 only. Subject-specific retinotopy can instead be modeled with
# :py:class:`~pulse2percept.topography.cortex.NeuropythyMap`.
#
# The scoreboard model assumes that each stimulated electrode produces a round
# phosphene centered at its retinotopic location. We place a 96-electrode
# NeuroPort array on V1 and stimulate every electrode with the same pulse train.

polimeni_map = p2p.topography.cortex.Polimeni2006Map(regions=['v1'])

cortical_implant = p2p.implants.cortex.Cortivis()
cortical_model = p2p.models.cortex.ScoreboardModel(
    cortical_implant,
    visual_field_map = polimeni_map,
    implant_position=(20, -5) * mm,
    xrange=(-3 * dva, -1 * dva),
    yrange=(0, 2 * dva),
    step=0.02 * dva,
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
