.. _topics-models:

====================
Models and Percepts
====================

A model turns stimulation into a predicted percept. A model with a spatial
component predicts the response to stimulation *by a particular device*, so it
is bound to an implant and handed the stimulus. A temporal-only model
describes one location's response over time and needs no implant.

.. code-block:: python

    import pulse2percept as p2p

    implant = p2p.implants.retina.ArgusII()
    model = p2p.models.retina.ScoreboardModel(implant=implant, rho=200)
    percept = model.predict_percept({'A8': 30})

Models build automatically on first prediction, and rebuild the affected
component when a parameter or the implant changes. The result of
``predict_percept`` is a :py:class:`~pulse2percept.percepts.Percept`.

The full pipeline distinguishes the source you provide, the stimulation the
device delivers, and the percept::

    source -> implant -> delivered stimulation -> model -> percept

Models call ``implant.prepare_stim(source)`` internally; see
:ref:`topics-stimulation` to inspect or control that step.

Choosing a model
----------------

Models are grouped by the tissue they stimulate. The root
:py:mod:`pulse2percept.models` namespace holds the abstract classes a model is
assembled from and temporal models that are not tied to a stimulation site;
models of a particular target live in :py:mod:`pulse2percept.models.retina`
and :py:mod:`pulse2percept.models.cortex`.

Which model to use depends on the scientific question. The published models
add assumptions specific to their experiments and should be chosen when those
assumptions are relevant. The API reference for each model documents its
assumptions, parameters, input requirements, and numerical units.

Retinal models
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 16 34

   * - Model
     - Reference
     - Type
     - Use it for
   * - :py:class:`~pulse2percept.models.retina.ScoreboardModel`
     - [Beyeler2019]_
     - spatial
     - A round phosphene per electrode; the simplest spatial baseline
   * - :py:class:`~pulse2percept.models.retina.AxonMapModel`
     - [Beyeler2019]_
     - spatial
     - Elongated phosphenes that follow retinal nerve fiber bundles
   * - :py:class:`~pulse2percept.models.retina.BiphasicScoreboardModel`
     - derived from [Granley2021]_
     - spatiotemporal
     - Round phosphenes whose brightness and size depend on the pulse train
   * - :py:class:`~pulse2percept.models.retina.BiphasicAxonMapModel`
     - [Granley2021]_
     - spatiotemporal
     - Pulse-dependent phosphenes with axonal elongation
   * - :py:class:`~pulse2percept.models.retina.Nanduri2012Model`
     - [Nanduri2012]_
     - spatial + temporal
     - Brightness and size as a function of amplitude and frequency
   * - :py:class:`~pulse2percept.models.retina.Horsager2009Model`
     - [Horsager2009]_
     - temporal
     - Single-electrode threshold as a function of pulse timing
   * - :py:class:`~pulse2percept.models.retina.Thompson2003Model`
     - [Thompson2003]_
     - spatial
     - Early scoreboard-style simulation with electrode dropout
   * - :py:class:`~pulse2percept.models.retina.Ho2018Model`
     - [Ho2018]_
     - spatiotemporal
     - Photovoltaic subretinal stimulation (see below)

The three most common spatial choices differ in what they commit to:

:py:class:`~pulse2percept.models.retina.ScoreboardModel`
    Fixed-width Gaussian per electrode; amplitude scales brightness. ``rho``
    is an effective perceptual spread fitted to subject reports, not a
    physical current-spread constant.

:py:class:`~pulse2percept.models.retina.BiphasicScoreboardModel`
    Adds [Granley2021]_-derived pulse-dependent brightness and width.
    Requires a described biphasic pulse train rather than a bare amplitude.

:py:class:`~pulse2percept.models.retina.BiphasicAxonMapModel`
    Additionally models axonal elongation, whose length follows phase
    duration [Granley2021]_. ``rho`` spreads across axons and ``lam`` along
    them.

Every retinal spatial model derives from
:py:class:`~pulse2percept.models.retina.RetinalSpatial`, which places
electrodes through a retinotopic map (see :ref:`topics-coordinates`) and
accepts a physical retinal extent as shorthand for ``xrange``/``yrange``.

Photovoltaic stimulation
^^^^^^^^^^^^^^^^^^^^^^^^

Photovoltaic arrays are driven by a pulsed near-infrared schedule rather than
injected current (see :ref:`topics-stimulation`). Two models consume it, and
they answer different questions:

:py:class:`~pulse2percept.models.retina.ScoreboardModel`
    Visualizes normalized optical drive: where the light lands, relative to a
    fully lit pixel. No retinal response.

:py:class:`~pulse2percept.models.retina.Ho2018Model`
    Predicts a phenomenological network-mediated retinal response. Each pulse
    period's radiant exposure (irradiance x ON duration) drives a Gaussian
    whose default ``rho`` is half the implant's median pixel pitch. A
    difference of low-pass cascades then filters the per-pulse drive maps, so
    the response is transient: a static image at a fixed pulse rate gives an
    onset response that adapts.

.. warning::

    :py:class:`~pulse2percept.models.retina.Ho2018Model` reconstructs the
    structure and timing of [Ho2018]_, not validated PRIMA percepts:

    *  pON center response of degenerate (RCS) rat retina only, with no
       antagonistic surround and no pOFF pathway;
    *  a device-scaled Gaussian spread, ``rho = pitch / 2``, which is a
       pulse2percept convention rather than a measured point-spread function
       or the receptive-field size [Ho2018]_ reports;
    *  a linear radiant-exposure activation law normalized to the 9 mW/mm^2,
       4 ms reference pulse of [Ho2018]_, with no fitted irradiance,
       pulse-duration or frequency nonlinearity;
    *  no photovoltaic circuit or electric-field model, no electrode-retina
       distance effect, and no wavelength or device-specific conversion
       efficiency, so an 880 nm PRIMA pixel and a 915 nm pixel of another
       design respond identically to the same radiant exposure;
    *  default temporal coefficients matched to the Table 1 timing landmarks
       of [Ho2018]_ rather than published by it, and no calibration to human
       brightness or contrast perception.

Cortical models
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 32 22 16 30

   * - Model
     - Reference
     - Type
     - Use it for
   * - :py:class:`~pulse2percept.models.cortex.ScoreboardModel`
     - [Beyeler2019]_, adapted
     - spatial
     - A spatial baseline where phosphene size follows cortical magnification
   * - :py:class:`~pulse2percept.models.cortex.DynaphosModel`
     - [vanderGrinten2023]_
     - spatiotemporal
     - Charge accumulation, thresholds, and phosphene dynamics over time

Cortical spatial models derive from
:py:class:`~pulse2percept.models.cortex.CortexSpatial`, which simulates one or
more visual areas (``'v1'``, ``'v2'``, ``'v3'``) and maps them through
cortical retinotopy. The cortical
:py:class:`~pulse2percept.models.cortex.ScoreboardModel` spreads current in
cortex rather than in the retina, so a fixed cortical ``rho`` produces
phosphenes whose visual-field size depends on eccentricity.

Generic temporal components
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`~pulse2percept.models.FadingTemporal` and
:py:class:`~pulse2percept.models.AlphaTemporal` describe how one location's
response decays after a pulse, without committing to where that location is.
They are the temporal half of a :py:class:`~pulse2percept.models.Model` whose
spatial half may be retinal or cortical.

Model limitations
-----------------

Electrode-tissue distance
~~~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`~pulse2percept.models.retina.ScoreboardModel`,
:py:class:`~pulse2percept.models.retina.AxonMapModel` and
:py:class:`~pulse2percept.models.retina.Thompson2003Model` use electrode ``x``
and ``y`` coordinates only. Nonzero ``z`` values therefore do not affect their
output and produce a warning.

This is a model limitation. Electrode-target distance is expected to affect
stimulation threshold and spatial recruitment, but pulse2percept does not
currently parameterize that relationship because the required psychophysical
evidence is insufficient. In the Scoreboard and AxonMap models, ``rho``
remains an effective perceptual spread parameter fitted to subject reports
rather than inferred from electrode-retina distance.

Monocular prediction
~~~~~~~~~~~~~~~~~~~~

Models are monocular in v0.11. A prediction is about one eye, and binocular
material is composed afterwards; see :ref:`topics-vision`.

Percepts
--------

A :py:class:`~pulse2percept.percepts.Percept` holds one of two layouts, with
time as the last axis in both::

    (Y, X, T)     perceived brightness in arbitrary units
    (Y, X, 3, T)  RGB intensities in [0, 1]

Prosthesis models produce brightness percepts.
:py:meth:`~pulse2percept.vision.Scene.render` composes one with residual
vision and returns an RGB percept:

.. code-block:: python

    import numpy as np
    from pulse2percept.percepts import Percept

    rgb = Percept(np.zeros((60, 80, 3, 1)))
    rgb.is_rgb                  # True
    rgb[..., 0].shape           # (60, 80, 3): one frame, still in color
    rgb.plot()                  # drawn as RGB, without a colormap

RGB values are display intensities and must be finite and lie in ``[0, 1]``;
anything else raises at construction rather than saturating quietly later. The
RGB axis is not a spatial dimension: ``space`` still describes ``(Y, X)``.

Operations defined on perceived brightness (i.e., ``n_gray``, ``argmax``,
``max``, ``vmin``, ``vmax``) raise a ``ValueError`` for an RGB percept rather
than inventing a conversion from color to brightness. Ranking three channels
by one number would have to pick a color metric, which is also why a
multi-frame RGB percept has no brightest frame to ``plot()``; animate it with
``play()`` instead. ``percept.data`` is always available for the plain
numerical answer.

Measuring a percept
-------------------

.. versionadded:: 0.11.0

Prediction stops at the percept. Measurement is optional post-processing,
performed on call and not part of the model::

    stimulus -> model -> Percept -> optional measurement

:py:meth:`~pulse2percept.percepts.Percept.measure` reports the brightness and
geometry of the phosphenes in each frame:

.. code-block:: python

    percept = model.predict_percept({'A8': 30})
    metrics = percept.measure()

    metrics.peak.diameter               # dva
    metrics.peak.centroid               # (x, y) in dva
    metrics.peak.integrated_brightness  # brightness units x dva^2

Support
~~~~~~~

Each frame is clipped at zero, so negative model output does not contribute.
Geometry is then measured on the *support*: the pixels at or above 50% of that
frame's own positive maximum. The threshold is relative and re-evaluated per
frame, so scaling a percept's brightness does not change its measured shape.
Lower it to include more of the falloff:

.. code-block:: python

    metrics = percept.measure(threshold=0.25)

What is measured
~~~~~~~~~~~~~~~~

``max_brightness`` is the largest positive pixel value, in the model's
arbitrary units. ``integrated_brightness`` is the pixel sum scaled by pixel
area (brightness units x dva^2), which approximates a spatial integral and is
therefore insensitive to sampling resolution. Neither is on a psychophysical
absolute scale.

The remaining measurements describe the support as a binary set of pixels;
brightness enters only through where the threshold falls. For a sufficiently
sampled circular phosphene, ``major_axis``, ``minor_axis`` and ``diameter``
approximately agree.

**area**
    Area of the support (dva^2).

**diameter**
    Diameter (dva) of the circle of equal area. At ``threshold=0.5`` this
    approximates the FWHM of a sufficiently sampled circular Gaussian.

**centroid**
    Mean position ``(x, y)`` of the support pixels, in dva.

**major_axis**, **minor_axis**
    Axes (dva) of the ellipse with the same second moments as the support.

**elongation**
    ``major_axis / minor_axis``; approaches 1 for a circular phosphene.

**n_components**
    Number of disconnected suprathreshold regions.

**touches_edge**
    Whether the support reaches the simulated field boundary.

A frame without positive brightness has no phosphene: brightness, area and
component count are zero, and position and shape are ``NaN``.

Multiple components and clipping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model output need not form a single connected phosphene. ``n_components``
counts the disconnected suprathreshold regions; the largest is not selected,
and every other measurement describes the combined support. An ``elongation``
of 4 is consistent with one elongated phosphene or with two round ones some
distance apart, which ``n_components`` distinguishes.

``touches_edge`` flags support reaching the border of the simulated visual
field. Where it is True, the measurements describe only the portion inside the
field, and no extrapolation is performed. Widen the model's ``xrange`` and
``yrange`` to capture the full percept.

Temporal percepts
~~~~~~~~~~~~~~~~~

Frames are measured independently, and each measurement is also available as
an array over frames:

.. code-block:: python

    metrics = percept.measure()

    metrics.integrated_brightness  # one value per frame
    metrics.peak_frame             # index of the brightest frame
    metrics.peak                   # that frame's measurements

``peak_frame`` ranks frames by integrated positive brightness, taking the
earliest on a tie. Frame timing remains in ``percept.time``; no time
integration or duration metrics are computed.

Scope
~~~~~

These measurements describe the modeled percept *image*. They do not estimate
visual acuity, phosphene discriminability, pairwise separability, behavioral
resolution, or object-recognition performance.

``measure()`` applies to model-produced brightness percepts and raises for the
RGB percepts :py:meth:`~pulse2percept.vision.Scene.render` returns, whose
values are display intensities. Positions and sizes are in degrees of visual
angle, so the percept must have been built on a
:py:class:`~pulse2percept.topography.Grid2D`; a percept holding bare pixel
indices is rejected.

Combining components
--------------------

Classes ending in ``Model`` are complete models with explicit constructor
parameters. Classes ending in ``Spatial`` or ``Temporal`` are components, and
:py:class:`~pulse2percept.models.Model` combines two of them:

.. code-block:: python

    spatial = p2p.models.retina.AxonMapSpatial(implant, rho=300, lam=500)
    temporal = p2p.models.FadingTemporal(tau=100)

    model = p2p.models.Model(spatial, temporal)

Use ``Model`` to combine spatial and temporal components from different
models. At least one component is required, and each must already be
constructed. The implant belongs to the spatial component.

Component parameters are accessed directly, and changing one rebuilds that
component on the next prediction:

.. code-block:: python

    model.spatial.rho = 250
    model.temporal.tau = 50

Named-model constructors expose the same parameters directly. Parameters
declared by both components, such as ``thresh_percept``, remain independent.
