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
   * - [Grinten2023]_
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
