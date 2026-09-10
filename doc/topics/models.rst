.. _topics-models:

====================
Computational Models
====================

A model with a spatial component predicts the response to stimulation *by a
particular device*, so it is bound to an implant and handed the stimulus.
A temporal-only model describes one location's response over time and needs
no implant. Most users work with
:py:class:`~pulse2percept.models.Model`, which can contain a spatial component,
a temporal component, or both:

* :py:class:`~pulse2percept.models.SpatialModel` determines where stimulation
  appears in the visual field.
* :py:class:`~pulse2percept.models.TemporalModel` determines how the response
  evolves over time.

Available models
----------------

Models are grouped by the tissue they stimulate. The root
:py:mod:`pulse2percept.models` namespace holds only the abstract classes a
model is assembled from and temporal models that are not tied to a stimulation
site; models of a particular target live in
:py:mod:`pulse2percept.models.retina` and
:py:mod:`pulse2percept.models.cortex`.

Generic model components
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Reference
     - Model
     - Type
   * - generic
     - :py:class:`~pulse2percept.models.FadingTemporal`
     - temporal
   * - generic
     - :py:class:`~pulse2percept.models.AlphaTemporal`
     - temporal

Both describe how one location's response decays after a pulse, without
committing to where that location is. They are the temporal half of a
:py:class:`~pulse2percept.models.Model` whose spatial half may be retinal or
cortical.

Retinal stimulation
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Reference
     - Model
     - Type
   * - [Thompson2003]_
     - :py:class:`~pulse2percept.models.retina.Thompson2003Model`
     - spatial
   * - [Horsager2009]_
     - :py:class:`~pulse2percept.models.retina.Horsager2009Model`
     - temporal
   * - [Nanduri2012]_
     - :py:class:`~pulse2percept.models.retina.Nanduri2012Model`
     - spatial + temporal
   * - [Beyeler2019]_
     - :py:class:`~pulse2percept.models.retina.ScoreboardModel`
     - spatial
   * - [Beyeler2019]_
     - :py:class:`~pulse2percept.models.retina.AxonMapModel`
     - spatial
   * - derived from [Granley2021]_
     - :py:class:`~pulse2percept.models.retina.BiphasicScoreboardModel`
     - spatiotemporal
   * - [Granley2021]_
     - :py:class:`~pulse2percept.models.retina.BiphasicAxonMapModel`
     - spatiotemporal

Every retinal spatial model derives from
:py:class:`~pulse2percept.models.retina.RetinalSpatial`, which places
electrodes through a retinotopic map and accepts a physical retinal extent as
shorthand for ``xrange``/``yrange``.

Which one to use depends on the scientific question. The three main choices
differ in what they model:

:py:class:`~pulse2percept.models.retina.ScoreboardModel`
    Fixed-width Gaussian per electrode; amplitude scales brightness.

:py:class:`~pulse2percept.models.retina.BiphasicScoreboardModel`
    Adds [Granley2021]_-derived pulse-dependent brightness and width.
    Requires a described biphasic pulse train rather than a bare
    amplitude.

:py:class:`~pulse2percept.models.retina.BiphasicAxonMapModel`
    Additionally models axonal elongation, whose length follows phase
    duration [Granley2021]_.

The published models add assumptions specific to their experiments and
should be chosen when those assumptions are relevant.

Cortical stimulation
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Reference
     - Model
     - Type
   * - [Beyeler2019]_, adapted
     - :py:class:`~pulse2percept.models.cortex.ScoreboardModel`
     - spatial
   * - [vanderGrinten2023]_
     - :py:class:`~pulse2percept.models.cortex.DynaphosModel`
     - spatiotemporal

Cortical spatial models derive from
:py:class:`~pulse2percept.models.cortex.CortexSpatial`, which simulates one or
more visual areas ('v1', 'v2', 'v3') and maps them through cortical
retinotopy. The cortical
:py:class:`~pulse2percept.models.cortex.ScoreboardModel` is a spatial baseline:
it spreads current in cortex rather than in the retina, so phosphene size in
the visual field follows cortical magnification.
:py:class:`~pulse2percept.models.cortex.DynaphosModel` adds the temporal
dynamics of the [vanderGrinten2023]_ phosphene model, including charge
accumulation and a stimulation threshold.

Basic usage
-----------

Models follow the same workflow: choose an implant, bind a model to it, then
predict a percept from a stimulus.

.. code-block:: python

    import pulse2percept as p2p

    implant = p2p.implants.retina.ArgusII()
    model = p2p.models.retina.ScoreboardModel(implant=implant, rho=200)
    percept = model.predict_percept({'A8': 30})

The result of ``predict_percept`` is a
:py:class:`~pulse2percept.percepts.Percept`.

Measuring a percept
-------------------

.. versionadded:: 0.11.0

Prediction stops at the percept. Measuring one is optional post-processing,
not part of the model::

    stimulus → model → Percept → optional measurement

:py:meth:`~pulse2percept.percepts.Percept.measure` summarizes how bright a
modeled percept is and what shape it has. Nothing is measured until you ask,
and nothing is cached afterwards:

.. code-block:: python

    percept = model.predict_percept({'A8': 30})
    metrics = percept.measure()

    metrics.peak.diameter          # 1.72 dva
    metrics.peak.centroid          # (5.32, 5.32) dva
    metrics.peak.total_brightness  # 102.7 brightness units x dva^2

Support
~~~~~~~

Geometry is measured on the pixels at or above 50% of a frame's own positive
peak brightness. The threshold is *relative* and re-evaluated per frame, so
doubling a percept's brightness leaves its measured shape where it was. Lower
it to measure a wider skirt:

.. code-block:: python

    metrics = percept.measure(threshold=0.25)

Negative model output never counts as brightness; each frame is clipped at
zero first.

What is measured
~~~~~~~~~~~~~~~~

Brightness comes in two flavors. ``max_brightness`` is the brightest positive
pixel, in the model's own arbitrary units. ``total_brightness`` integrates
positive brightness over the visual field --- a pixel sum scaled by the area
of a pixel, in brightness units x dva^2 --- so it approximates a spatial
integral instead of tracking how finely you sampled the field. Neither is on a
psychophysical absolute scale.

The remaining measurements describe the suprathreshold support:

**area**
    Area of the support (dva^2).

**diameter**
    Diameter (dva) of the circle with the same area. At ``threshold=0.5``
    this approximates the FWHM of a sufficiently sampled circular Gaussian.

**centroid**
    Brightness-weighted ``(x, y)`` center, in dva.

**major_axis**, **minor_axis**
    Axes (dva) of the ellipse with the same brightness-weighted second
    moments as the support.

**elongation**
    ``major_axis / minor_axis``; 1 for a circular phosphene.

**n_components**
    Number of disconnected suprathreshold regions.

**touches_edge**
    Whether the support reaches the simulated field boundary.

A frame with no positive brightness has no phosphene: its brightness, area and
component count are zero, and the quantities that would place or shape a
phosphene are ``NaN`` rather than a phantom blob at the origin.

Multiple components and clipping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model output is not always one tidy phosphene. ``n_components`` counts the
disconnected suprathreshold regions, and every other measurement describes the
*whole* support: pulse2percept does not quietly keep the largest blob and
throw the rest away. An ``elongation`` of 4 can mean one stretched phosphene
or two round ones far apart, and ``n_components`` is what tells them apart.

``touches_edge`` warns that the support runs into the border of the simulated
visual field. When it is True, area, diameter, axes, centroid and integrated
brightness describe only the part of the percept the simulation covers. No
extrapolation is attempted; widen the model's ``xrange``/``yrange`` if you
need the whole thing.

Temporal percepts
~~~~~~~~~~~~~~~~~

Every frame is measured independently, and each measurement is also available
as an array over frames:

.. code-block:: python

    metrics = percept.measure()

    metrics.total_brightness  # one value per frame
    metrics.peak_frame        # frame with the most integrated brightness
    metrics.peak              # that frame's measurements

``peak_frame`` ranks frames by integrated positive brightness, taking the
earliest of any tie. When each frame occurred stays where it was, in
``percept.time``; measurement adds no time integration or duration metrics.

Scope
~~~~~

These are measurements of a modeled percept *image*. They do not estimate
visual acuity, phosphene discriminability, pairwise separability, behavioral
resolution, or object-recognition performance --- those are psychophysical
questions that a picture of a percept does not answer.

``measure()`` applies to model-produced brightness percepts, and raises for
the RGB percepts scene composition returns, whose values are display
intensities rather than perceived brightness. Positions and sizes are reported
in degrees of visual angle, so the percept must have been built on a real
:py:class:`~pulse2percept.topography.Grid2D`; a percept carrying bare image
pixel indices is rejected rather than measured as though pixels were degrees.


Source, delivered stimulation, percept
--------------------------------------

The prediction pipeline distinguishes the source, delivered stimulation, and
percept::

    source → implant → delivered stimulation → model → percept

**Source**
    Input presented to the device: a
    :py:class:`~pulse2percept.stimuli.Stimulus` (or compatible scalar, array,
    or dict), :py:class:`~pulse2percept.stimuli.ImageStimulus`,
    :py:class:`~pulse2percept.stimuli.VideoStimulus`, or
    :py:class:`~pulse2percept.vision.Scene`.

**Delivered stimulation**
    Electrical stimulation after implant preprocessing, encoding, raster
    scheduling, threshold calibration, and safety checks. Models call
    ``implant.prepare_stim(source)`` internally; call it directly to inspect
    the delivered stimulus.

**Percept**
    Model output from ``model.predict_percept(source)``.

Building
--------

Models build automatically on first prediction. Changing a model parameter
invalidates the affected component, which is rebuilt when needed:

.. code-block:: python

    model = p2p.models.retina.AxonMapModel(implant=implant)

    # Builds automatically:
    percept = model.predict_percept(stim)

    # Rebuilds the spatial component:
    model.spatial.rho = 250
    percept = model.predict_percept(stim)

Rebinding the implant also invalidates the spatial build because it depends on
device geometry. ``model.build()`` forces a full rebuild; to set parameters as
you build a component, use ``model.spatial.build(rho=250)``.

Electrode-retina distance
-------------------------

:py:class:`~pulse2percept.models.retina.ScoreboardModel`,
:py:class:`~pulse2percept.models.retina.AxonMapModel` and
:py:class:`~pulse2percept.models.retina.Thompson2003Model` use electrode ``x`` and
``y`` coordinates only. Nonzero ``z`` values therefore do not affect their
output and produce a warning.

This is a model limitation. Electrode-target distance is expected to affect
stimulation threshold and spatial recruitment, but pulse2percept does not
currently parameterize that relationship because the required psychophysical
evidence is insufficient. In the Scoreboard and AxonMap models, ``rho`` remains
an effective perceptual spread parameter fitted to subject reports rather than
inferred from electrode-retina distance.

.. _topics-models-scene:

Simulating a visual scene
-------------------------

.. versionadded:: 0.11.0

The workflow above starts from a stimulus you built yourself. To start from
what someone is *looking at* instead, give the model a
:py:class:`~pulse2percept.vision.Scene`. A scene is **one monocular visual
field**, not the person's final vision: it says what is present in front of
one eye, and where that eye's native vision is lost.

.. code-block:: python

    from pulse2percept.units import dva

    scene = p2p.vision.Scene(p2p.stimuli.samples.logo_bvl(), fov=40 * dva)

    implant = p2p.implants.retina.ArgusII()
    implant.encoder = p2p.stimuli.AmplitudeEncoder(amp_range=(0, 50))

    model = p2p.models.retina.ScoreboardModel(implant=implant, rho=200)
    percept = model.predict_percept(scene, gaze=(0, 0) * dva)

Scene prediction separates four responsibilities:

=========  ==================================================================
Scene      What is visually present, and where native vision is lost.
Implant    Device geometry and encoding constraints.
Model      Knows the retinotopy, and so connects Scene to Implant.
Percept    What the simulated observer sees.
=========  ==================================================================

The model maps implant coordinates into the visual field through its
retinotopic map. Each electrode follows this chain::

    retinal coordinate (um)
      -> visual_field_map.ret_to_dva -> eye-centered visual field (dva)
      -> + gaze, for eye-coupled input only -> scene coordinate (dva)
      -> sample the scene

``gaze`` is the scene location that currently falls on the fovea, so
``scene = eye-centered visual field + gaze``. Gaze always decides where the
percept lands in scene coordinates. Whether it also decides what the
electrodes are given depends on the implant's
:py:attr:`~pulse2percept.implants.Implant.scene_input_frame`:

==========  =================================================================
``'eye'``   Input passes through the eye's optics (Alpha, PRIMA), so gaze
            moves the scene across the implant as well as moving the percept
            across the scene. This is the default.
``'head'``  Input comes from a head-fixed camera (Argus, BVT, IMIE) that the
            eye cannot move: the electrodes are handed the same scene
            whatever the gaze, and only the percept moves.
==========  =================================================================

Neither the implant nor an eye-centered
:py:class:`~pulse2percept.vision.Scotoma` moves when gaze does. Pass one
``(x, y)`` to fixate, or one per video frame to move the eye between frames.

For a ``'head'`` system, the sampling locations above are still the electrodes'
own visual-field positions, which assumes the device's camera-to-electrode
registration is aligned with them. Real systems configure that mapping
separately and it is not modeled here.

``scene_input_frame`` follows the device class but is a property of the
system, so one implant can be run the other way -- an Argus II with eye
tracking, which shifts the camera ROI with gaze:

.. code-block:: python

    implant = p2p.implants.retina.ArgusII()
    implant.scene_input_frame = 'eye'

The sampled values are passed to ``implant.encoder``, which maps gray levels
to current and applies device timing constraints. A scene is per-prediction
input and is not stored on the model or implant.

An implant's ``preprocess`` -- an edge filter, an inversion, a contrast
stretch -- is applied to the **prosthetic input branch only**, before the
scene is sampled at the electrode locations, because an image operation needs
an image and by sampling time there is one number per electrode. Native and
residual vision always use the original scene: what the device does to its
own input is not something the eye goes through. Spatial preprocessing
operates at the scene source's pixel resolution.

.. code-block:: python

    implant.preprocess = lambda stim: stim.filter('sobel')

For scene input, ``preprocess`` must return an
:py:class:`~pulse2percept.stimuli.ImageStimulus` or
:py:class:`~pulse2percept.stimuli.VideoStimulus`; conversion to electrical
stimulation belongs to the encoder. Pixel values and channels may change, but
spatial shape and frame timing must remain unchanged because ``fov`` and the
frame clock refer to the original scene.

Scene registration is a spatial-model capability: a model has to say where in
the visual field each of its electrodes lands. Only retinal models
(:py:class:`~pulse2percept.models.retina.RetinalSpatial`) implement it, through
their retinotopy; any other spatial model raises ``NotImplementedError``. A
retinal model given a non-retinotopic ``visual_field_map``, or an implant
without an ``encoder``, raises ``ValueError``.

Residual vision
~~~~~~~~~~~~~~~

If the scene also carries a :py:class:`~pulse2percept.vision.Scotoma`, the
result is what the person actually sees -- intact native vision outside the
lost region, and the prosthetic percept inside it -- as a single RGB
:py:class:`~pulse2percept.percepts.Percept` on the scene's own pixel grid:

.. code-block:: python

    scene = p2p.vision.Scene(p2p.stimuli.samples.logo_bvl(), fov=40 * dva,
                             scotoma=p2p.vision.Scotoma.circle(8 * dva))
    model = p2p.models.retina.ScoreboardModel(implant=implant, rho=200)

    percept = model.predict_percept(scene, gaze=(0, 0) * dva, vmax=50)

``vmax`` is required here and is not inferred: model brightness is in
arbitrary units, so which brightness counts as white is a claim about the
display, not about the model. Holding it fixed across calls is what keeps two
gazes comparable.

The scotoma affects *native* vision only. Prosthetic encoding samples the
unmasked scene, including locations inside the scotoma.

The rendered field boundary
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 0.11.0

A scene's source, pixel grid and sampling are rectangular. ``aperture='circle'``
renders an eye-centered disc of radius ``min(fov) / 2`` instead, blacking out
the corners around it and changing the rendered scene only:

.. code-block:: python

    scene = p2p.vision.Scene(image, fov=40 * dva, aperture='circle')

Like the scotoma and the eccentricity rings, the disc is
eye-centered, so gaze moves it through the scene.

Both eyes
~~~~~~~~~

.. versionadded:: 0.11.0

:py:class:`~pulse2percept.vision.BinocularScene` holds the left and right
monocular views:

.. code-block:: python

    binocular = p2p.vision.BinocularScene(
        left=p2p.vision.Scene(image, fov=40 * dva, scotoma=scotoma),
        right=p2p.vision.Scene(image, fov=40 * dva),
    )

    ax_left, ax_right = binocular.plot(left_percept=percept, vmax=2)

A bilateral loss is often symmetric about the vertical meridian.
:py:meth:`~pulse2percept.vision.Scotoma.mirror` reflects a scotoma across it
(``mirrored(x, y) == original(-x, y)``) and returns a new one:

.. code-block:: python

    left_scotoma = p2p.vision.Scotoma.circle(3 * dva, center=(6, 0) * dva)
    right_scotoma = left_scotoma.mirror()

Models are monocular in v0.11, so a prediction names the eye it is about:

.. code-block:: python

    percept = model.predict_percept(binocular.left)


Percept data layouts
--------------------

A :py:class:`~pulse2percept.percepts.Percept` holds one of two layouts, with
time as the last axis in both::

    (Y, X, T)     perceived brightness in arbitrary units
    (Y, X, 3, T)  RGB intensities in [0, 1]

Prosthesis models produce brightness percepts. When a
:py:class:`~pulse2percept.vision.Scene` has a scotoma, scene-driven prediction
composes that model output with residual vision and returns an RGB percept:

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
``max``, ``vmin``, ``vmax``) raise a ``ValueError`` for an RGB
percept rather than inventing a conversion from color to brightness. 
Ranking three channels by one number would have to pick a
color metric, which is also why a multi-frame RGB percept has no brightest
frame to ``plot()``; animate it with ``play()`` instead. ``percept.data`` is
always available for the plain numerical answer.

Spatial and temporal components
-------------------------------

Classes ending in ``Model`` are complete models with explicit constructor
parameters:

.. code-block:: python

    model = p2p.models.retina.AxonMapModel(
        implant,
        rho=300,
        lam=500,
    )
    percept = model.predict_percept(stim)

Classes ending in ``Spatial`` or ``Temporal`` are components, and
:py:class:`~pulse2percept.models.Model` combines two of them:

.. code-block:: python

    spatial = p2p.models.retina.AxonMapSpatial(
        implant,
        rho=300,
        lam=500,
    )

    temporal = p2p.models.FadingTemporal(tau=100)

    model = p2p.models.Model(spatial, temporal)

Use ``Model`` to combine spatial and temporal components from different
models. At least one component is required, and each must already be
constructed. The implant belongs to the spatial component.

Parameters
----------

Component parameters are accessed directly:

.. code-block:: python

    model.spatial.rho = 250
    model.temporal.tau = 50

Named-model constructors expose the same parameters directly. After
construction, access them through the component. Parameters declared by both
components, such as ``thresh_percept``, remain independent.

The API reference for each model documents its assumptions, parameters, input
requirements, and numerical units.
