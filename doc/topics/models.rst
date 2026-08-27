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
   * - [Thompson2003]_
     - :py:class:`~pulse2percept.models.Thompson2003Model`
     - spatial
   * - [Horsager2009]_
     - :py:class:`~pulse2percept.models.Horsager2009Model`
     - temporal
   * - [Nanduri2012]_
     - :py:class:`~pulse2percept.models.Nanduri2012Model`
     - spatial + temporal
   * - [Beyeler2019]_
     - :py:class:`~pulse2percept.models.ScoreboardModel`
     - spatial
   * - [Beyeler2019]_
     - :py:class:`~pulse2percept.models.AxonMapModel`
     - spatial
   * - [Granley2021]_
     - :py:class:`~pulse2percept.models.BiphasicAxonMapModel`
     - spatiotemporal
   * - [vanderGrinten2023]_
     - :py:class:`~pulse2percept.models.cortex.DynaphosModel`
     - spatiotemporal

Cortical stimulation also has
:py:class:`~pulse2percept.models.cortex.ScoreboardModel`, a spatial baseline
that maps cortical electrode locations through cortical retinotopy.

Which model to use depends on the scientific question. The scoreboard model is
a simple local baseline. The axon-map model adds retinal nerve-fiber effects
for epiretinal stimulation. Published temporal and spatiotemporal models add
assumptions specific to their experiments and should be chosen when those
assumptions are relevant.

Basic usage
-----------

Models follow the same workflow: choose an implant, bind a model to it, then
predict a percept from a stimulus.

.. code-block:: python

    import pulse2percept as p2p

    implant = p2p.implants.ArgusII()
    model = p2p.models.ScoreboardModel(implant=implant, rho=200)
    percept = model.predict_percept({'A8': 30})

The result of ``predict_percept`` is a
:py:class:`~pulse2percept.percepts.Percept`.

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

    model = p2p.models.AxonMapModel(implant=implant)

    # Builds automatically:
    percept = model.predict_percept(stim)

    # Rebuilds the spatial component:
    model.rho = 250
    percept = model.predict_percept(stim)

Rebinding the implant also invalidates the spatial build because it depends on
device geometry. ``model.build()`` forces a full rebuild and can be used to
build eagerly or pass build-time parameters (``model.build(rho=250)``).

Electrode-retina distance
-------------------------

:py:class:`~pulse2percept.models.ScoreboardModel`,
:py:class:`~pulse2percept.models.AxonMapModel` and
:py:class:`~pulse2percept.models.Thompson2003Model` use electrode ``x`` and
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
:py:class:`~pulse2percept.vision.Scene`:

.. code-block:: python

    from pulse2percept.units import dva

    scene = p2p.vision.Scene(p2p.stimuli.LogoBVL(), fov=40 * dva)

    implant = p2p.implants.ArgusII()
    implant.encoder = p2p.stimuli.AmplitudeEncoder(amp_range=(0, 50))

    model = p2p.models.ScoreboardModel(implant=implant, rho=200)
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
      -> vfmap.ret_to_dva -> eye-centered visual field (dva)
      -> + gaze            -> scene coordinate (dva)
      -> sample the scene

``gaze`` is the scene location that currently falls on the fovea, so
``scene = visual field + gaze``. The implant does not move when gaze does, and
neither does an eye-centered
:py:class:`~pulse2percept.vision.Scotoma`: the scene moves past them. Pass one
``(x, y)`` to fixate, or one per video frame to move the eye between frames.

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

Scene registration currently requires a retinal ``vfmap`` and an implant
``encoder``. A cortical ``vfmap`` or missing encoder raises ``ValueError``.

Residual vision
~~~~~~~~~~~~~~~

If the scene also carries a :py:class:`~pulse2percept.vision.Scotoma`, the
result is what the person actually sees -- intact native vision outside the
lost region, and the prosthetic percept inside it -- as a single RGB
:py:class:`~pulse2percept.percepts.Percept` on the scene's own pixel grid:

.. code-block:: python

    scene = p2p.vision.Scene(p2p.stimuli.LogoBVL(), fov=40 * dva,
                             scotoma=p2p.vision.Scotoma.circle(8 * dva))
    model = p2p.models.ScoreboardModel(implant=implant, rho=200)

    percept = model.predict_percept(scene, gaze=(0, 0) * dva, vmax=50)

``vmax`` is required here and is not inferred: model brightness is in
arbitrary units, so which brightness counts as white is a claim about the
display, not about the model. Holding it fixed across calls is what keeps two
gazes comparable.

The scotoma affects *native* vision only. Prosthetic encoding samples the
unmasked scene, including locations inside the scotoma.

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

Operations defined on perceived brightness -- ``n_gray``, ``noise``,
``argmax``, ``max``, and the ``vmin``/``vmax`` display range -- raise a
``ValueError`` for an RGB percept rather than inventing a conversion from
color to brightness. Ranking three channels by one number would have to pick a
color metric, which is also why a multi-frame RGB percept has no brightest
frame to ``plot()``; animate it with ``play()`` instead. ``percept.data`` is
always available for the plain numerical answer.

Spatial and temporal components
-------------------------------

Classes ending in ``Model`` are complete model objects. Classes ending in
``Spatial`` or ``Temporal`` are components that can be composed:

.. code-block:: python

    model = p2p.models.Model(
        implant=implant,
        spatial=p2p.models.ScoreboardSpatial(),
        temporal=p2p.models.Nanduri2012Temporal(),
    )

The implant belongs to the spatial component -- a temporal model never sees an
electrode -- and naming it on the parent is shorthand for that. Naming a
*different* one on both raises rather than silently picking a winner.

This is useful when the spatial and temporal assumptions come from different
models. The combined model handles the intermediate representation and returns
a Percept like any other Model.

Components may also be used directly, but normal simulations are simpler
through the complete Model interface.

Parameters
----------

A combined Model forwards parameters to its components. For example,
``model.rho`` may belong to the spatial component and ``model.dt`` to the
temporal component. If both components define the same parameter, setting it on
the parent updates both; set ``model.spatial.<name>`` or
``model.temporal.<name>`` when you need to distinguish them.

The API reference for each model documents its assumptions, parameters, input
requirements, and numerical units.
