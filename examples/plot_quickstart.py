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

In epiretinal implants like Argus II, phosphenes often appear elongated along
retinal nerve fiber bundles [Beyeler2019]_.
The [Granley2021]_ model captures this effect and predicts phosphene shape,
brightness, and size from pulse amplitude (given as a multiple of perceptual
threshold, ``xTh``), frequency (``Hz``), and pulse duration (``ms``):
"""
# sphinx_gallery_thumbnail_number = 1

import matplotlib.pyplot as plt

import pulse2percept as p2p
from pulse2percept.units import Hz, um, mm, ms, uA, xTh, dva

retinal_implant = p2p.implants.retina.ArgusII()
retinal_model = p2p.models.retina.BiphasicAxonMapModel(
    retinal_implant,
    rho=300 * um,  # microns
    lam=500 * um,  # microns
)

stim = {
    'A5': p2p.stimuli.BiphasicPulseTrain(
        freq=20 * Hz,         # Hertz (pulses/s)
        amp=2 * xTh,          # multiples of threshold
        phase_dur=0.45 * ms,  # milliseconds
    )
}

percept = retinal_model.predict_percept(stim)
percept.plot()
plt.title('Argus II: one stimulated electrode')
plt.show()

###############################################################################
# The biphasic axon map model [Granley2021]_ is based on human behavioral data
# collected across multiple retinal prosthesis studies. Its two spatial
# parameters, `rho` and `lam`, control phosphene spread perpendicular and
# parallel to the retinal nerve fiber bundles, respectively.
# These parameters vary across patients, so the values above are illustrative
# rather than universal.
#
#
# An image through a photovoltaic implant
# ---------------------------------------
#
# PRIMA is a subretinal photovoltaic implant driven by pulsed near-infrared
# light. :py:class:`~pulse2percept.implants.retina.PRIMAPivotal` includes a
# :py:class:`~pulse2percept.stimuli.PRIMAEncoder` that converts image intensity
# into the pulse durations delivered by the projector.
# 
# Below we pair the implant with 
# :py:class:`~pulse2percept.models.retina.Ho2018Model`, which models the
# transient retinal network response reported by [Ho2018]_.

prima = p2p.implants.retina.PRIMAPivotal()
prima_model = p2p.models.retina.Ho2018Model(
    prima,
    xrange=(-6 * dva, 6 * dva),  # degrees of visual angle
    yrange=(-6 * dva, 6 * dva),
    step=0.1 * dva,
)

image = p2p.stimuli.samples.ucsb_surf(resize=(180, 320))
percept = prima_model.predict_percept(image, t_percept=50 * ms)

###############################################################################
# The model is based on degenerated rat retina and should not be interpreted
# as a validated human model of PRIMA perception.
#
# Adding residual vision and gaze
# -------------------------------
#
# A :py:class:`~pulse2percept.vision.Scene` places an image or video in the
# visual field. It can also include a :py:class:`~pulse2percept.vision.Scotoma`
# describing where native vision has been lost.
#
# Here a video recorded on the UCSB campus spans 40 degrees of visual angle
# (dva), with a central scotoma representing vision loss around the implanted
# region.

video = p2p.stimuli.samples.ucsb_pedestrians(resize=(173, 320))
scotoma = p2p.vision.Scotoma.circle(5 * dva)

scene = p2p.vision.Scene(
    video,
    fov=40 * dva,
    scotoma=scotoma,
    scotoma_fill=0,
    aperture='round',
)

gaze = p2p.vision.Gaze(
    [(0, 0), (-15.5, -6), (12, -6)] * dva, 
    time=[0, 635, 1370] * ms
)
percept = prima_model.predict_percept(scene, gaze=gaze)
scene.play(
    percept=percept,
    gaze=gaze,
    vmax=percept.data.max(),
)

###############################################################################
# A form traced through visual cortex
# -----------------------------------
#
# Cortical phosphenes do not necessarily combine like pixels on a screen.
# In human experiments with the Orion visual cortical prosthesis, simultaneous
# stimulation of several electrodes often produced merged phosphenes rather than
# a recognizable shape. Beauchamp et al. [Beauchamp2020]_ instead stimulated
# electrodes sequentially, tracing letter-like forms through the retinotopic map.
#
# Here we borrow that stimulation strategy, but visualize the resulting
# spatiotemporal percept with the independently developed
# :py:class:`~pulse2percept.models.cortex.DynaphosModel`
# [vanderGrinten2023]_. An Orion array is mapped from cortex into the visual
# field, and a sequence of electrodes traces a letter over time:
# NeuroPort array on V1 and stimulate every electrode with the same pulse train.
#
# TODO: composed view, orion lighting up on the left, using ``TraceEncoder``
# to draw a Z on the right.

polimeni_map = p2p.topography.cortex.Polimeni2006Map(regions=['v1'])

orion = p2p.implants.cortex.Orion()
dynaphos = p2p.models.cortex.DynaphosModel(
    orion,
    visual_field_map = polimeni_map,
    implant_position=(20, -5) * mm,
    xrange=(-3 * dva, -1 * dva),
    yrange=(0, 2 * dva),
    step=0.01 * dva,
)

train = p2p.stimuli.BiphasicPulseTrain(
    freq=20 * Hz,
    amp=100 * uA,
    phase_dur=0.45 * ms,
)
stim = {electrode: train for electrode in cortical_implant.electrode_names}

percept = cortical_model.predict_percept(stim)
percept.plot()
plt.title('NeuroPort Array: all 96 electrodes')
plt.show()

###############################################################################
# This is therefore an illustrative simulation, not a reproduction of the
# Beauchamp participant: electrode locations, thresholds, and phosphene dynamics
# vary across people.
#
#
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
