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

Percept data layouts
--------------------

A :py:class:`~pulse2percept.percepts.Percept` holds one of two layouts, with
time as the last axis in both::

    (Y, X, T)     perceived brightness in arbitrary units
    (Y, X, 3, T)  RGB intensities in [0, 1]

Models always produce the brightness form. The RGB form exists to display a
scene alongside a modeled percept:

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
