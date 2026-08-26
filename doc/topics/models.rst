.. _topics-models:

====================
Computational Models
====================

A model predicts the response to stimulation. Most users work with
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

Models follow the same workflow: initialize, build, then predict a percept from
an implant containing a stimulus.

.. code-block:: python

    import pulse2percept as p2p

    implant = p2p.implants.ArgusII(stim={'A8': 30})
    model = p2p.models.ScoreboardModel(rho=200).build()
    percept = model.predict_percept(implant)

``build()`` performs one-time setup such as constructing an axon map. The result
of ``predict_percept`` is a
:py:class:`~pulse2percept.percepts.Percept`.

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

    model = p2p.models.ScoreboardModel(scene=scene, rho=200).build()
    percept = model.predict_percept(implant, gaze=(0, 0) * dva)

Four objects divide the problem between them:

=========  ==================================================================
Scene      What is visually present, and where native vision is lost.
Implant    Device geometry and encoding constraints.
Model      Knows the retinotopy, and so connects Scene to Implant.
Percept    What the simulated observer sees.
=========  ==================================================================

The model is the glue because it is the only object that holds a retinotopy
*and* is handed an implant. Each electrode is followed out along this chain::

    retinal coordinate (um)
      -> vfmap.ret_to_dva -> eye-centered visual field (dva)
      -> + gaze            -> scene coordinate (dva)
      -> sample the scene

``gaze`` is the scene location that currently falls on the fovea, so
``scene = visual field + gaze``. The implant does not move when gaze does, and
neither does an eye-centered
:py:class:`~pulse2percept.vision.Scotoma`: the scene moves past them. Pass one
``(x, y)`` to fixate, or one per video frame to move the eye between frames.

The sampled values go to ``implant.encoder``, so the implant still decides how
a gray level becomes current and which electrodes may pulse when. Prediction
does not touch ``implant.stim``: it runs against a stand-in copy, so asking
what someone sees never rewrites their device.

An implant's ``preprocess`` -- an edge filter, an inversion, a contrast
stretch -- is applied to the **prosthetic input branch only**, before the
scene is sampled at the electrode locations, because an image operation needs
an image and by sampling time there is one number per electrode. Native and
residual vision always use the original scene: what the device does to its
own input is not something the eye goes through. Spatial preprocessing
operates at the scene source's pixel resolution.

.. code-block:: python

    implant.preprocess = lambda stim: stim.filter('sobel')

For a scene, ``preprocess`` must return an
:py:class:`~pulse2percept.stimuli.ImageStimulus` or
:py:class:`~pulse2percept.stimuli.VideoStimulus`; a callable that produces
current directly has nothing left to place in the visual field, and that is
the encoder's job in any case. Pixel values and channels are free to change --
RGB to grayscale is fine -- but the spatial shape and the frame clock are not,
because ``fov`` describes the geometry of the source it was given, and a
resize or a re-timing would quietly reinterpret it.

Scene registration is retinal. A model whose ``vfmap`` is a cortical map
raises rather than pretending cortical registration is solved, and so does an
implant with no ``encoder``.

Residual vision
~~~~~~~~~~~~~~~

If the scene also carries a :py:class:`~pulse2percept.vision.Scotoma`, the
result is what the person actually sees -- intact native vision outside the
lost region, and the prosthetic percept inside it -- as a single RGB
:py:class:`~pulse2percept.percepts.Percept` on the scene's own pixel grid:

.. code-block:: python

    scene = p2p.vision.Scene(p2p.stimuli.LogoBVL(), fov=40 * dva,
                             scotoma=p2p.vision.Scotoma.circle(8 * dva))
    model = p2p.models.ScoreboardModel(scene=scene, rho=200).build()

    percept = model.predict_percept(implant, gaze=(0, 0) * dva, vmax=50)

``vmax`` is required here and is not inferred: model brightness is in
arbitrary units, so which brightness counts as white is a claim about the
display, not about the model. Holding it fixed across calls is what keeps two
gazes comparable.

The scotoma describes *native* vision only. What the implant is given to
encode is sampled from the scene itself, inside the lost region as well as
outside it: a camera does not go blind where its wearer has.

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
        spatial=p2p.models.ScoreboardSpatial(),
        temporal=p2p.models.Nanduri2012Temporal(),
    ).build()

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
