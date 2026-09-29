.. _topics-models:

===================
Models and Percepts
===================

A model converts delivered stimulation into a predicted
:py:class:`~pulse2percept.percepts.Percept`.

Spatial models are attached to an implant. Temporal-only models describe the
response at a single location and do not require one.


Quick Start
===========

.. code-block:: python

    import pulse2percept as p2p

    implant = p2p.implants.retina.ArgusII()

    model = p2p.models.retina.ScoreboardModel(
        implant=implant,
        rho=200,
    )

    percept = model.predict_percept({'A8': 30})

:py:meth:`~pulse2percept.models.Model.predict_percept` first passes the source
through ``implant.prepare_stim``; see :ref:`topics-stimulation`. The model is
built automatically on first use.


Model Interface
===============

Most models share a small set of conventions:

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Parameter
     - Meaning
     - Units / values
   * - ``implant``
     - Device supplying the delivered stimulation
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
     - How activity within an output interval is represented
     - ``'last'`` or ``'peak'``

Retinal models default to a 30 x 30 dva field sampled every 0.25 dva. Cortical
models default to 10 x 10 dva. Both are centered on the fovea.

Widen ``xrange`` or ``yrange`` if a percept reaches the edge of the simulated
field. Reduce ``step`` when small phosphenes require finer spatial sampling.

Changing a model parameter or implant rebuilds only the affected component on
the next prediction.


Choosing a Model
================

Models are grouped by the tissue they stimulate:

* :mod:`pulse2percept.models.retina`
* :mod:`pulse2percept.models.cortex`

Generic temporal components and base classes live directly in
:mod:`pulse2percept.models`. The API documentation for each model gives its
full parameter set, assumptions, and units.


Retinal Models
--------------

.. list-table::
   :header-rows: 1
   :widths: 29 19 17 35

   * - Model
     - Reference
     - Type
     - Use it for
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
     - spatial + temporal
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

.. note::

   In the scoreboard and axon map models, ``rho`` (um) is an effective
   perceptual spread fitted to subject reports. It is not a physical
   current-spread constant.

   Axon map models add ``lam`` (um), which controls spread along retinal nerve
   fiber bundles.

.. note::

   The biphasic models require a biphasic pulse train with amplitude specified
   in ``xTh``, or ``implant.thresholds`` so that uA can be converted to
   threshold units.

   In these models, amplitude affects brightness and size, frequency affects
   brightness, and phase duration affects axonal streak length.

Retinal spatial models derive from
:py:class:`~pulse2percept.models.retina.RetinalSpatial` and place electrodes
through a retinotopic map; see :ref:`topics-coordinates`.

``xrange`` and ``yrange`` can also be specified in retinal length units.


Photovoltaic Stimulation
------------------------

Two retinal models accept the near-infrared stimulation schedule produced by a
photovoltaic implant:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Model
     - Output
   * - :py:class:`~pulse2percept.models.retina.ScoreboardModel`
     - Normalized optical drive: where light lands relative to a fully
       illuminated pixel. It does not model the retinal response.
   * - :py:class:`~pulse2percept.models.retina.Ho2018Model`
     - A phenomenological network-mediated retinal response based on
       [Ho2018]_.

``Ho2018Model`` converts each pulse period's radiant exposure
(irradiance x ON duration) into a Gaussian spatial response. Its default
``rho`` is half the median pixel pitch. A difference of low-pass cascades then
filters the response over time, producing an onset transient that adapts under
a static image and fixed pulse rate.

.. warning::

   ``Ho2018Model`` reconstructs the structure and timing of [Ho2018]_; it is
   not a validated model of PRIMA percepts.

   * It includes the pON center response of degenerate RCS rat retina only.
     There is no antagonistic surround or pOFF pathway.

   * ``rho = pitch / 2`` is a pulse2percept convention, not a measured
     point-spread function or the receptive-field size reported by [Ho2018]_.

   * Activation is linear in radiant exposure and normalized to the
     9 mW/mm^2, 4 ms reference pulse from [Ho2018]_. No irradiance,
     pulse-duration, or frequency nonlinearity was fit.

   * The model includes no photovoltaic circuit, electric field,
     electrode-retina distance, or wavelength dependence. An 880 nm PRIMA
     pixel and a 915 nm pixel from another design therefore respond identically
     to the same radiant exposure.

   * Temporal coefficients were chosen to match the timing landmarks in
     Table 1 of [Ho2018]_. They were not published by that study and are not
     calibrated to human brightness or contrast perception.


Cortical Models
---------------

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
     - Spatial baseline with phosphene size determined by cortical
       magnification
   * - :py:class:`~pulse2percept.models.cortex.DynaphosModel`
     - [vanderGrinten2023]_
     - spatiotemporal
     - Charge accumulation, thresholds, and phosphene dynamics over time

Cortical spatial models derive from
:py:class:`~pulse2percept.models.cortex.CortexSpatial` and use cortical
retinotopy to map stimulation into visual space.

Depending on the model, one or more of ``'v1'``, ``'v2'``, and ``'v3'`` can be
selected through ``regions``.

The cortical ``rho`` parameter is a spread on cortex, measured in um. Because
cortical magnification varies with eccentricity, a fixed cortical spread does
not correspond to a fixed phosphene size in dva.


Generic Temporal Models
-----------------------

:py:class:`~pulse2percept.models.FadingTemporal` and
:py:class:`~pulse2percept.models.AlphaTemporal` describe the response at a
single location after stimulation.

They can be combined with retinal or cortical spatial components; see
`Combining components`_ below.


Model Limitations
-----------------

.. note::

   :py:class:`~pulse2percept.models.retina.ScoreboardModel`,
   :py:class:`~pulse2percept.models.retina.AxonMapModel`, and
   :py:class:`~pulse2percept.models.retina.Thompson2003Model` ignore electrode
   ``z`` and warn when it is nonzero.

   Electrode-retina distance is expected to affect threshold and spread, but
   the psychophysical evidence represented by these models is insufficient to
   parameterize that dependence.

All prosthesis models are monocular: a prediction refers to one eye. See
:ref:`topics-vision` for binocular scene handling.


Percept Timing
==============

Spatial-only models return one frame for each stimulus time point. A timeless
stimulus, such as a static image, therefore produces one frame.

For temporal models, ``t_percept`` determines when the output is sampled:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Setting
     - Behavior
   * - Explicit ``t_percept``
     - Returns the model response at exactly the requested times.
   * - ``t_percept=None``
     - Uses the encoder's frame clock for encoded video; otherwise returns
       frames every 20 ms (50 Hz).

``reduce`` determines what a frame represents when it covers an interval:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Meaning
   * - ``'last'``
     - Brightness at the end of the interval; the default.
   * - ``'peak'``
     - Maximum brightness reached within the interval.

For example, to sample every 10 ms:

.. code-block:: python

    import numpy as np

    percept = model.predict_percept(
        stim,
        t_percept=np.arange(0, 500, 10),
    )


Percepts
========

A :py:class:`~pulse2percept.percepts.Percept` stores the predicted visual
output together with its spatial and temporal coordinates.

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

Spatial coordinates are stored in ``xdva`` and ``ydva``. ``time`` uses
``time_unit`` (ms by default), or is ``None`` for a single timeless frame.

Common operations are:

.. code-block:: python

    percept.plot()                 # brightest frame
    percept.play()                 # animation
    percept.save('percept.mp4')    # image or video, based on extension

Prosthesis models produce brightness percepts.
:py:meth:`~pulse2percept.vision.Scene.render` instead returns an RGB percept;
see :ref:`topics-vision`.

RGB values must be finite and lie in ``[0, 1]``. Other values are rejected at
construction.

Operations that require an ordering of brightness values are undefined for RGB
percepts. ``n_gray``, ``argmax``, ``max``, ``vmin``, ``vmax``, and
``plot()`` on a multi-frame RGB percept therefore raise ``ValueError``. Use
``play()`` or access ``percept.data`` directly instead.


Measuring a Percept
===================

.. versionadded:: 0.11.0

:py:meth:`~pulse2percept.percepts.Percept.measure` reports brightness and
geometry for each frame. Measurement is post-processing and does not change
the underlying model prediction.

.. code-block:: python

    metrics = percept.measure()

    metrics.peak.diameter                # dva
    metrics.peak.centroid                # (x, y) in dva
    metrics.peak.integrated_brightness   # brightness units x dva^2

Geometry is measured from the percept's *support*: pixels at or above
``threshold`` times that frame's maximum. The default threshold is 0.5.

Because the threshold is relative, multiplying the entire percept by a
constant does not change its measured shape. Lower the threshold to include
more of the spatial falloff:

.. code-block:: python

    metrics = percept.measure(threshold=0.25)


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

At the default ``threshold=0.5``, ``diameter`` approximates the FWHM of a
well-sampled circular Gaussian.

``major_axis``, ``minor_axis``, and ``elongation`` describe the union of all
support pixels. An elongation of 4, for example, could describe one elongated
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
brightness, choosing the earliest frame in a tie. ``metrics.peak`` provides
the measurements from that frame.

``measure()`` does not compute time-integrated brightness or duration metrics.


Measurement Scope
-----------------

``measure()`` requires a brightness percept defined on a
:py:class:`~pulse2percept.topography.Grid2D`. RGB percepts and percepts without
dva coordinates are rejected.

These measurements describe the predicted percept image. They do not measure
acuity, discriminability, recognition, or task performance.


Combining Components
====================

Classes ending in ``Model`` are complete models. Classes ending in ``Spatial``
or ``Temporal`` are components that can be combined with
:py:class:`~pulse2percept.models.Model`.

.. code-block:: python

    spatial = p2p.models.retina.AxonMapSpatial(
        implant,
        rho=300,
        lam=500,
    )

    temporal = p2p.models.FadingTemporal(tau=100)

    model = p2p.models.Model(spatial, temporal)

    model.spatial.rho = 250
    model.temporal.tau = 50

Either component may be ``None``. Components must be constructed before they
are passed to ``Model``, and the implant belongs to the spatial component.

Changing ``model.spatial.rho`` rebuilds the spatial component on the next
prediction; changing ``model.temporal.tau`` affects only the temporal
component.

Parameters with the same name in both components, such as
``thresh_percept``, remain independent.