.. _topics-models:

====================
Models and Percepts
====================

A model converts delivered stimulation into a predicted
:py:class:`~pulse2percept.percepts.Percept`. A model with a spatial component
is bound to an implant; a temporal-only model describes one location's
response over time and needs none.

.. code-block:: python

    import pulse2percept as p2p

    implant = p2p.implants.retina.ArgusII()
    model = p2p.models.retina.ScoreboardModel(implant=implant, rho=200)
    percept = model.predict_percept({'A8': 30})

``predict_percept`` calls ``implant.prepare_stim(source)`` first (see
:ref:`topics-stimulation`). Models build on first prediction and rebuild the
affected component when a parameter or the implant changes.

The simulated visual field is a grid set by ``xrange``, ``yrange`` (dva) and
``step`` (dva per pixel). Retinal models default to 30 x 30 dva at 0.25 dva,
cortical models to 10 x 10 dva, both centered on the fovea. Widen the grid if
a percept touches its edge; refine ``step`` for small phosphenes.

Choosing a model
----------------

Models live under the tissue they stimulate,
:py:mod:`pulse2percept.models.retina` and
:py:mod:`pulse2percept.models.cortex`. The root namespace holds generic
temporal models and the base classes. Each model's API page lists its
assumptions, parameters, and units.

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

*  ``rho`` (um) in the scoreboard and axon map models is an effective
   perceptual spread fitted to subject reports, not a physical current-spread
   constant. Axon map models add ``lam`` (um), the spread along an axon.
*  The Biphasic models require a biphasic pulse train with amplitude in
   ``xTh``, or ``implant.thresholds`` to convert uA. Amplitude scales
   phosphene brightness and size; frequency scales brightness; phase duration
   shortens axonal streaks.
*  Retinal spatial models derive from
   :py:class:`~pulse2percept.models.retina.RetinalSpatial` and place
   electrodes through a retinotopic map (see :ref:`topics-coordinates`).
   ``xrange``/``yrange`` may also be given as retinal lengths.

Photovoltaic stimulation
^^^^^^^^^^^^^^^^^^^^^^^^

Two models accept the near-infrared schedule of a photovoltaic array:

:py:class:`~pulse2percept.models.retina.ScoreboardModel`
    Shows normalized optical drive: where the light lands, relative to a
    fully lit pixel. No retinal response.

:py:class:`~pulse2percept.models.retina.Ho2018Model`
    Predicts a phenomenological network-mediated retinal response. Each pulse
    period's radiant exposure (irradiance x ON duration) drives a Gaussian
    with default ``rho`` of half the median pixel pitch. A difference of
    low-pass cascades filters the result, so a static image at a fixed pulse
    rate gives an onset response that adapts.

.. warning::

    :py:class:`~pulse2percept.models.retina.Ho2018Model` reconstructs the
    structure and timing of [Ho2018]_, not validated PRIMA percepts:

    *  pON center response of degenerate (RCS) rat retina only; no
       antagonistic surround and no pOFF pathway.
    *  ``rho = pitch / 2`` is a pulse2percept convention, not a measured
       point-spread function or the receptive-field size of [Ho2018]_.
    *  Activation is linear in radiant exposure, normalized to the
       9 mW/mm^2, 4 ms reference pulse of [Ho2018]_, with no fitted
       irradiance, pulse-duration, or frequency nonlinearity.
    *  No photovoltaic circuit, electric field, electrode-retina distance, or
       wavelength dependence: an 880 nm PRIMA pixel and a 915 nm pixel of
       another design respond identically to the same radiant exposure.
    *  Temporal coefficients are matched to the Table 1 timing landmarks of
       [Ho2018]_, not published by it, and are not calibrated to human
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
:py:class:`~pulse2percept.models.cortex.CortexSpatial`, simulate one or more
of ``'v1'``, ``'v2'``, ``'v3'`` (``regions``), and map them through cortical
retinotopy. The cortical ``rho`` (um) is a spread on cortex, so phosphene
size in dva grows with eccentricity.

Generic temporal models
~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`~pulse2percept.models.FadingTemporal` and
:py:class:`~pulse2percept.models.AlphaTemporal` describe how one location's
response decays after a pulse. They combine with retinal or cortical spatial
components (see `Combining components`_).

Limitations
~~~~~~~~~~~

*  :py:class:`~pulse2percept.models.retina.ScoreboardModel`,
   :py:class:`~pulse2percept.models.retina.AxonMapModel` and
   :py:class:`~pulse2percept.models.retina.Thompson2003Model` ignore
   electrode ``z`` and emit a warning if it is nonzero. Electrode-retina
   distance is expected to affect threshold and spread, but the
   psychophysical evidence is insufficient to parameterize it.
*  Models are monocular: a prediction is about one eye (see
   :ref:`topics-vision`).

Percept timing
--------------

Spatial-only models return one frame per stimulus time point (one frame for
a timeless stimulus or an image). For temporal models, ``t_percept`` sets the
output times (ms):

*  **Given explicitly**, it returns those exact instants.
*  **Omitted**, output frames follow the encoder's frame clock for encoded
   video, and are 20 ms apart (50 Hz) otherwise. ``reduce`` sets what each
   frame reports: ``'last'`` (default) the brightness at the end of its
   interval, ``'peak'`` the maximum within it.

.. code-block:: python

    import numpy as np

    percept = model.predict_percept(stim, t_percept=np.arange(0, 500, 10))

Percepts
--------

A :py:class:`~pulse2percept.percepts.Percept` stores ``data`` with time as
the last axis, the grid coordinates ``xdva`` and ``ydva``, and ``time`` (in
``time_unit``, ms by default; ``None`` for a single timeless frame):

.. code-block:: text

    (Y, X, T)     perceived brightness, arbitrary units
    (Y, X, 3, T)  RGB display intensities in [0, 1]

.. code-block:: python

    percept.plot()               # brightest frame
    percept.play()               # animation
    percept.save('percept.mp4')  # image or video, by extension

Prosthesis models produce brightness percepts.
:py:meth:`~pulse2percept.vision.Scene.render` returns an RGB percept (see
:ref:`topics-vision`). RGB values must be finite and lie in ``[0, 1]``;
anything else is rejected at construction with a ``ValueError``.
Brightness-only operations (``n_gray``, ``argmax``, ``max``, ``vmin``,
``vmax``, and ``plot()`` of a multi-frame percept) are rejected for RGB
percepts with a ``ValueError``, because ranking colors needs a color metric.
Use ``play()`` or ``percept.data`` instead.

Measuring a percept
-------------------

.. versionadded:: 0.11.0

:py:meth:`~pulse2percept.percepts.Percept.measure` reports brightness and
geometry of the phosphenes in each frame. It is post-processing, not part of
the model:

.. code-block:: python

    metrics = percept.measure()

    metrics.peak.diameter               # dva
    metrics.peak.centroid               # (x, y) in dva
    metrics.peak.integrated_brightness  # brightness units x dva^2

Negative values are clipped to zero. Geometry is measured on the *support*:
pixels at or above ``threshold`` (default 0.5) times that frame's own
maximum. The threshold is relative, so scaling brightness does not change
measured shape. Lower it to include more of the falloff:
``percept.measure(threshold=0.25)``.

``max_brightness``
    Largest pixel value (model units).

``integrated_brightness``
    Pixel sum times pixel area (brightness units x dva^2); approximately
    independent of grid resolution.

``area``
    Support area (dva^2).

``diameter``
    Diameter (dva) of the circle of equal area. At ``threshold=0.5`` this
    approximates the FWHM of a well-sampled circular Gaussian.

``centroid``
    Mean ``(x, y)`` of the support pixels (dva).

``major_axis``, ``minor_axis``, ``elongation``
    Axes (dva) of the ellipse with the support's second moments, and their
    ratio (1 for a circle).

``n_components``
    Number of disconnected suprathreshold regions. Other measurements
    describe their union, so an ``elongation`` of 4 can be one elongated
    phosphene or two round ones; ``n_components`` distinguishes them.

``touches_edge``
    Whether the support reaches the grid border. If so, measurements cover
    only the part inside the grid; widen ``xrange`` and ``yrange``.

A frame without positive brightness has zero brightness, area, and
components, and ``NaN`` position and shape.

For a temporal percept, each measurement is also an array over frames.
``metrics.peak_frame`` is the frame with the largest integrated brightness
(earliest on a tie), and ``metrics.peak`` its measurements. No time
integration or duration metrics are computed.

``measure()`` requires a brightness percept on a
:py:class:`~pulse2percept.topography.Grid2D`; RGB percepts and percepts
without dva coordinates are rejected. The measurements describe the percept
image only, not acuity, discriminability, or task performance.

Combining components
--------------------

Classes ending in ``Model`` are complete models. Classes ending in
``Spatial`` or ``Temporal`` are components, which
:py:class:`~pulse2percept.models.Model` combines:

.. code-block:: python

    spatial = p2p.models.retina.AxonMapSpatial(implant, rho=300, lam=500)
    temporal = p2p.models.FadingTemporal(tau=100)

    model = p2p.models.Model(spatial, temporal)

    model.spatial.rho = 250    # rebuilds the spatial component
    model.temporal.tau = 50

One of the two may be ``None``. Components must be constructed before they
are passed. The implant belongs to the spatial component. Parameters declared by
both components, such as ``thresh_percept``, stay independent.
