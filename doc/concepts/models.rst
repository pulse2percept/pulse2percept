.. _topics-models:

===================
Models and Percepts
===================

A model converts delivered stimulation into a predicted
:py:class:`~pulse2percept.percepts.Percept`.

pulse2percept includes models for retinal and cortical prostheses, ranging
from simple spatial baselines to models of axonal activation, pulse-dependent
perception, photovoltaic stimulation, and temporal dynamics.


Choosing a Model
================

Different models make different assumptions about how electrical stimulation
is transformed into a visual percept. Some describe each stimulated electrode
with a simple local phosphene, while others incorporate retinal anatomy,
stimulus-dependent changes in phosphene appearance, or temporal dynamics.

The consequences of those assumptions are visible even for the same implant
and stimulation pattern:

.. plot::

    import matplotlib.pyplot as plt
    import pulse2percept as p2p

    implant = p2p.implants.retina.ArgusII()
    stim = {'A8': 30}

    models = [
        ('Thompson2003Model',
         p2p.models.retina.Thompson2003Model(
             implant,
             xrange=(-10, 10),
             yrange=(-10, 10),
             step=0.1,
             verbose=False)),
        ('ScoreboardModel',
         p2p.models.retina.ScoreboardModel(
             implant,
             rho=200,
             xrange=(-10, 10),
             yrange=(-10, 10),
             step=0.1,
             verbose=False)),
        ('AxonMapModel',
         p2p.models.retina.AxonMapModel(
             implant,
             rho=200,
             xrange=(-10, 10),
             yrange=(-10, 10),
             step=0.1,
             verbose=False)),
    ]

    fig, axes = plt.subplots(
        1, 3, figsize=(12, 4), sharex=True, sharey=True
    )

    for ax, (title, model) in zip(axes, models):
        percept = model.predict_percept(stim)
        percept.plot(ax=ax)
        ax.set_title(title)

    fig.tight_layout()

These predictions reflect different modeling assumptions rather than a simple
progression from less to more accurate models. The appropriate choice depends
on the implant, stimulation protocol, and perceptual property being studied.

The sections below summarize the models included in pulse2percept and the
phenomena each one represents.


Retinal Models
==============

Retinal models include spatial baselines, models of epiretinal axonal
activation, pulse-dependent models, and models developed from specific
psychophysical or physiological datasets.

.. list-table::
   :header-rows: 1
   :widths: 29 19 17 35

   * - Model
     - Reference
     - Type
     - Use It For
   * - :py:class:`~pulse2percept.models.retina.ScoreboardModel`
     - [Beyeler2019]_
     - spatial
     - Round phosphenes; the simplest spatial baseline
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
     - spatiotemporal
     - Brightness and size as functions of amplitude and frequency
   * - :py:class:`~pulse2percept.models.retina.Horsager2009Model`
     - [Horsager2009]_
     - temporal
     - Single-electrode detection threshold as a function of pulse timing
   * - :py:class:`~pulse2percept.models.retina.Thompson2003Model`
     - [Thompson2003]_
     - spatial
     - Early scoreboard-style simulation with electrode dropout
   * - :py:class:`~pulse2percept.models.retina.Ho2018Model`
     - [Ho2018]_
     - spatiotemporal
     - Network-mediated response to photovoltaic stimulation

Retinal spatial models use retinotopy to place stimulation in the visual
field; see :ref:`topics-coordinates`.

``xrange`` and ``yrange`` may be specified either in degrees of visual angle
or in retinal length units.


Scoreboard and Axon Map Models
------------------------------

The scoreboard and axon map models provide two common spatial starting points.

:py:class:`~pulse2percept.models.retina.ScoreboardModel` represents each
stimulated electrode as a circular Gaussian. ``rho`` controls the spatial
spread of the predicted phosphene.

:py:class:`~pulse2percept.models.retina.AxonMapModel` additionally models
activation along retinal nerve fiber bundles. ``rho`` controls spread away
from the bundle and ``lam`` controls spread along it.

.. note::

   ``rho`` and ``lam`` are effective perceptual parameters fitted to subject
   reports. They vary substantially across patients and are not physical
   current-spread constants.

   Electrode-retina distance is not currently used to determine ``rho``.


Pulse-Dependent Models
----------------------

The biphasic models extend the same spatial families with pulse-dependent
effects described by [Granley2021]_.

They require a biphasic pulse train with amplitude specified in ``xTh``, or
``implant.thresholds`` so that current in uA can be converted to threshold
units.

Within these models:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Stimulation Parameter
     - Predicted Effect
   * - amplitude
     - brightness and phosphene size
   * - frequency
     - brightness
   * - phase duration
     - axonal streak length

See the :ref:`model reproductions <examples-models>` for examples based on the
published experiments.


Photovoltaic Stimulation
------------------------

Two retinal models accept the near-infrared stimulation schedule produced by
photovoltaic implants:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Model
     - Output
   * - :py:class:`~pulse2percept.models.retina.ScoreboardModel`
     - Normalized optical drive showing where light lands relative to a fully
       illuminated pixel; it does not model the retinal response
   * - :py:class:`~pulse2percept.models.retina.Ho2018Model`
     - Phenomenological network-mediated retinal response based on [Ho2018]_

``Ho2018Model`` converts each pulse period's radiant exposure
(irradiance x ON duration) into a Gaussian spatial response. A difference of
low-pass cascades then filters the response over time, producing a strong
onset response that adapts under a static image and fixed pulse rate.

.. note::

   ``Ho2018Model`` is a phenomenological reconstruction of the pON response
   reported in degenerate rat retina by [Ho2018]_, not a model calibrated to
   human PRIMA percepts. It does not model the photovoltaic circuit or electric
   field, and its spatial and temporal parameters should be interpreted as
   model assumptions rather than measured human perceptual parameters.


Cortical Models
===============

Cortical models predict percepts from epicortical or intracortical
stimulation.

.. list-table::
   :header-rows: 1
   :widths: 32 22 16 30

   * - Model
     - Reference
     - Type
     - Use It For
   * - :py:class:`~pulse2percept.models.cortex.ScoreboardModel`
     - [Beyeler2019]_, adapted
     - spatial
     - Spatial baseline with phosphene size determined by cortical
       magnification
   * - :py:class:`~pulse2percept.models.cortex.DynaphosModel`
     - [vanderGrinten2023]_
     - spatiotemporal
     - Charge accumulation, thresholds, and phosphene dynamics over time

Cortical spatial models use cortical retinotopy to transform stimulation into
visual space.

Depending on the model, one or more of ``'v1'``, ``'v2'``, and ``'v3'`` can
be selected through ``regions``.

The cortical ``rho`` parameter describes spread across cortex in um. Because
cortical magnification changes with eccentricity, a fixed cortical spread does
not produce a fixed phosphene size in degrees of visual angle.


Predicting a Percept
====================

All complete models use the same basic workflow:

.. code-block:: python

    import pulse2percept as p2p

    implant = p2p.implants.retina.ArgusII()

    model = p2p.models.retina.ScoreboardModel(
        implant,
        rho=200,
    )

    percept = model.predict_percept({
        'A8': 30,
    })

    percept.plot()

:py:meth:`~pulse2percept.models.Model.predict_percept` first passes the input
through ``implant.prepare_stim``; see :ref:`topics-stimulation`.

The model is built automatically the first time it is used. Changing a model
parameter or its implant rebuilds only the affected component on the next
prediction.


Percepts
========

A :py:class:`~pulse2percept.percepts.Percept` stores the predicted visual
output together with its visual-field and temporal coordinates.

Common operations are:

.. code-block:: python

    percept.plot()                 # static frame
    percept.play()                 # animation
    percept.save('percept.mp4')    # image or video, based on extension

Spatial coordinates are stored in ``xdva`` and ``ydva``. ``time`` uses
``time_unit`` (ms by default), or is ``None`` for a single timeless frame.


Percept Data
------------

The underlying ``data`` array has time on its final axis:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Shape
     - Contents
   * - ``(Y, X, T)``
     - Brightness in arbitrary model units
   * - ``(Y, X, 3, T)``
     - RGB display intensities in ``[0, 1]``

Prosthesis models produce brightness percepts.

:py:meth:`~pulse2percept.vision.Scene.render` instead returns an RGB percept;
see :ref:`topics-vision`.

RGB values must be finite and lie in ``[0, 1]``. Operations that require an
ordering of brightness values are not defined for RGB percepts.
``n_gray``, ``argmax``, ``max``, ``vmin``, ``vmax``, and ``plot()`` on a
multi-frame RGB percept therefore raise ``ValueError``. Use ``play()`` or
access ``percept.data`` directly instead.


Measuring a Percept
===================

.. versionadded:: 0.11.0

:py:meth:`~pulse2percept.percepts.Percept.measure` extracts brightness and
geometric measurements from each predicted frame:

.. code-block:: python

    metrics = percept.measure()

    metrics.peak.diameter               # dva
    metrics.peak.centroid               # (x, y) in dva
    metrics.peak.integrated_brightness  # brightness units x dva^2

Geometry is measured from the percept's **support**: pixels at or above
``threshold`` times that frame's maximum. The default is ``threshold=0.5``.

For example:

.. code-block:: python

    metrics = percept.measure(threshold=0.25)

Because the threshold is relative, multiplying an entire percept by a constant
does not change its measured shape.


Measurements
------------

.. list-table::
   :header-rows: 1
   :widths: 27 18 55

   * - Measurement
     - Units
     - Meaning
   * - ``max_brightness``
     - model units
     - Largest pixel value in the frame
   * - ``integrated_brightness``
     - brightness x dva^2
     - Pixel sum multiplied by pixel area; approximately independent of grid
       resolution
   * - ``area``
     - dva^2
     - Area of the suprathreshold support
   * - ``diameter``
     - dva
     - Diameter of a circle with the same area as the support
   * - ``centroid``
     - dva
     - Mean ``(x, y)`` position of the support pixels
   * - ``major_axis``
     - dva
     - Major axis of the ellipse defined by the support's second moments
   * - ``minor_axis``
     - dva
     - Minor axis of the same ellipse
   * - ``elongation``
     - ratio
     - ``major_axis / minor_axis``; 1 for a circle
   * - ``n_components``
     - count
     - Number of disconnected suprathreshold regions
   * - ``touches_edge``
     - boolean
     - Whether the support reaches the simulated field boundary

At ``threshold=0.5``, ``diameter`` approximates the FWHM of a well-sampled
circular Gaussian.

``major_axis``, ``minor_axis``, and ``elongation`` describe the union of all
support pixels. An elongation of 4 could therefore describe one elongated
phosphene or two separated round phosphenes. ``n_components`` distinguishes
those cases.

If ``touches_edge`` is true, the measurement covers only the part of the
percept inside the simulated field. Increase ``xrange`` or ``yrange`` before
interpreting its geometry.

A frame with no positive brightness has zero brightness, area, and components,
and ``NaN`` position and shape.


Temporal Percepts
-----------------

For a temporal percept, each measurement is an array over frames.

``metrics.peak_frame`` identifies the frame with the largest integrated
brightness, choosing the earliest frame in a tie. ``metrics.peak`` contains
the measurements from that frame.

``measure()`` does not compute time-integrated brightness or duration metrics.


Measurement Scope
-----------------

``measure()`` requires a brightness percept defined on a
:py:class:`~pulse2percept.topography.Grid2D`. RGB percepts and percepts without
dva coordinates are rejected.

The measurements describe the predicted percept image. They do not measure
acuity, discriminability, recognition, or task performance.


Temporal Models
===============

Temporal models describe how the predicted response changes after
stimulation.

Some complete models, including
:py:class:`~pulse2percept.models.retina.BiphasicAxonMapModel`,
:py:class:`~pulse2percept.models.retina.Ho2018Model`, and
:py:class:`~pulse2percept.models.cortex.DynaphosModel`, already include
temporal dynamics.

Generic temporal components are also available:

* :py:class:`~pulse2percept.models.FadingTemporal`
* :py:class:`~pulse2percept.models.AlphaTemporal`

They can be combined with retinal or cortical spatial components; see
`Combining Components`_.


Percept Timing
--------------

Spatial-only models return one frame for each stimulus time point. A timeless
stimulus, such as a static image, produces one frame.

For temporal models, ``t_percept`` controls when the output is sampled:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Setting
     - Behavior
   * - explicit ``t_percept``
     - Return the model response at exactly the requested times
   * - ``t_percept=None``
     - Use the encoder's frame clock for encoded video; otherwise return
       frames every 20 ms (50 Hz)

For example, to sample every 10 ms:

.. code-block:: python

    import numpy as np

    percept = model.predict_percept(
        stim,
        t_percept=np.arange(0, 500, 10),
    )

``reduce`` controls what a frame represents when an output interval covers
multiple internal time points:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Meaning
   * - ``'last'``
     - Brightness at the end of the interval; the default
   * - ``'peak'``
     - Maximum brightness reached within the interval


Model Configuration
===================

Most spatial and spatiotemporal models share a small set of conventions.

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Parameter
     - Meaning
     - Units / Values
   * - ``implant``
     - Device supplying the stimulation
     - :py:class:`~pulse2percept.implants.Implant`
   * - ``xrange``, ``yrange``
     - Bounds of the simulated visual field
     - dva, or retinal length for retinal spatial models
   * - ``step``
     - Visual-field sampling interval
     - dva per pixel
   * - ``t_percept``
     - Requested output times for temporal models
     - ms
   * - ``reduce``
     - Representation of activity within an output interval
     - ``'last'`` or ``'peak'``

Retinal models default to a 30 x 30 dva field sampled every 0.25 dva. Cortical
models default to 10 x 10 dva. Both are centered on the fovea.

Increase ``xrange`` or ``yrange`` if a predicted percept reaches the edge of
the simulated field. Reduce ``step`` when small phosphenes require finer
spatial sampling.

The API documentation for each model gives its complete parameter set,
assumptions, and units.


Combining Components
====================

Classes ending in ``Model`` are complete models. Classes ending in ``Spatial``
or ``Temporal`` are components that can be combined with
:py:class:`~pulse2percept.models.Model`.

For example, the retinal axon map can be combined with a generic fading
temporal response:

.. code-block:: python

    spatial = p2p.models.retina.AxonMapSpatial(
        implant,
        rho=300,
        lam=500,
    )

    temporal = p2p.models.FadingTemporal(
        tau=100,
    )

    model = p2p.models.Model(
        spatial,
        temporal,
    )

The components remain accessible independently:

.. code-block:: python

    model.spatial.rho = 250
    model.temporal.tau = 50

Either component may be ``None``. Components must be constructed before they
are passed to ``Model``, and the implant belongs to the spatial component.

Changing ``model.spatial.rho`` rebuilds the spatial component on the next
prediction. Changing ``model.temporal.tau`` affects only the temporal
component.

Parameters with the same name in both components, such as
``thresh_percept``, remain independent.


Model Limitations
=================

The models in pulse2percept reproduce specific assumptions, datasets, or
published modeling frameworks. Their predictions should be interpreted within
those scopes.

.. note::

   :py:class:`~pulse2percept.models.retina.ScoreboardModel`,
   :py:class:`~pulse2percept.models.retina.AxonMapModel`, and
   :py:class:`~pulse2percept.models.retina.Thompson2003Model` ignore electrode
   ``z`` and warn when it is nonzero.

   Electrode-retina distance is expected to affect threshold and spatial
   spread, but the psychophysical evidence represented by these models is
   insufficient to parameterize that dependence.

All prosthesis models are monocular: a prediction refers to one eye. See
:ref:`topics-vision` for binocular scene handling.

For model-specific assumptions and validation, see the individual API
documentation and the :ref:`model reproductions <examples-models>`.