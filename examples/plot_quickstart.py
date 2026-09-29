# -*- coding: utf-8 -*-
"""
===============================================================================
Quickstart Guide
===============================================================================

A pulse2percept [Beyeler2017]_ simulation has three parts:

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
import numpy as np

import pulse2percept as p2p
from pulse2percept.units import Hz, um, ms, xTh

argus = p2p.implants.retina.ArgusII()
axon_map = p2p.models.retina.BiphasicAxonMapModel(
    argus,
    rho=200 * um,  # microns
    lam=800 * um,  # microns
)
stim = {
    'A3': p2p.stimuli.BiphasicPulseTrain(
        freq=20 * Hz,         # Hertz (pulses/s)
        amp=2 * xTh,          # multiples of threshold
        phase_dur=0.45 * ms,  # milliseconds
    )
}

fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
percept = axon_map.predict_percept(stim)
percept.plot(ax=axes[1], rings=True, meridians=True)
axon_map.plot(show_implant=True, ax=axes[0])
fig.tight_layout()

###############################################################################
# The biphasic axon map model [Granley2021]_ is based on human behavioral data
# collected across multiple retinal prosthesis studies. Its two spatial
# parameters, `rho` and `lam`, control phosphene spread perpendicular and
# parallel to the retinal nerve fiber bundles, respectively.
# These parameters vary across patients, so the values above are illustrative
# rather than universal (see the
# :ref:`Granley 2021 reproduction 
# <sphx_glr_examples_models_plot_granley2021_biphasic.py>`).
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

from pulse2percept.units import dva, deg, ms
huang = p2p.implants.retina.Huang2021Array(30)
huang_model = p2p.models.retina.Ho2018Model(
    huang,
    xrange=(-3 * dva, 3 * dva),  # degrees of visual angle
    yrange=(-3 * dva, 3 * dva),
    step=0.025 * dva,
)

# 0.3 dva stroke ≈ 87 um ≈ 3 pixels of 30 um; the whole E is 1.5 dva across
e = p2p.stimuli.psychophysics.tumbling_e(
    stroke=0.3 * dva,
    orientation=90 * deg,  # bars point up
    fov=6 * dva,
    polarity='light',      # white E on black: only the E is stimulated
)
percept = huang_model.predict_percept(e, t_percept=50 * ms)
percept.plot();

###############################################################################
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

prima = p2p.implants.retina.PRIMAPivotal()
prima_model = p2p.models.retina.Ho2018Model(
    prima,
    xrange=(-6 * dva, 6 * dva),  # degrees of visual angle
    yrange=(-6 * dva, 6 * dva),
    step=0.05 * dva,
)

gaze = p2p.vision.Gaze([
    (0, 0, 0),
    (-29 * dva, -7 * dva, 605 * ms),
    (22 * dva, -10 * dva, 1270 * ms)
])
percept = prima_model.predict_percept(scene, gaze=gaze)
scene.play(
    percept=percept,
    gaze=gaze,
    vmax=percept.data.max(),
)

###############################################################################
# The [Ho2018]_ model is based on degenerated rat retina and should not be
# interpreted as a validated human model of PRIMA perception.
#
# 
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
# [vanderGrinten2023]_. An Orion array on right V1 covers part of the left
# visual field. :py:class:`~pulse2percept.stimuli.TraceEncoder` maps a
# trajectory in dva onto the nearest electrodes and stimulates them one at a
# time, ``step_dur`` each:

from pulse2percept.units import mm, ms, uA, dva

orion = p2p.implants.cortex.Orion()
dynaphos = p2p.models.cortex.DynaphosModel(
    orion,
    visual_field_map=p2p.topography.cortex.Polimeni2006Map(regions=['v1']),
    implant_position=(20, -5) * mm,
    xrange=(-6 * dva, 0 * dva),
    yrange=(-1 * dva, 4.5 * dva),
    step=0.05 * dva,
)

# The letter Z: top bar, diagonal, bottom bar, 30 samples per stroke (dva)
corners = [(-4.5, 3.4), (-1.3, 3.4), (-3.8, 0), (-1.2, 0)]
z = np.vstack([np.linspace(start, end, 30)
               for start, end in zip(corners[:-1], corners[1:])]) * dva

encoder = p2p.stimuli.TraceEncoder(
    dynaphos,
    amp=1000 * uA,             # phosphene diameter grows with sqrt(amp)
    freq=dynaphos.freq,        # Dynaphos simulates its own pulse timing
    phase_dur=dynaphos.p_dur,
    step_dur=100 * ms,         # per electrode
)
stim = encoder.encode(z)
percept = dynaphos.predict_percept(stim)

fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
dynaphos.plot(show_implant=True, ax=axes[0])
percept.play(ax=axes[1], rings=[1.25, 2.5, 5], meridians=True);

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
# :ref:`topics-implants`. The :ref:`model reproductions <examples-models>`
# recreate figures from published models.
