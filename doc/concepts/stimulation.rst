.. _topics-stimulation:
.. _topics-stimuli:

===========
Stimulation
===========

Stimulation in pulse2percept follows a simple path::

    source
      |
      v
    implant.prepare_stim()
      |
      +-- preprocessing
      +-- encoding
      +-- raster scheduling
      +-- threshold conversion
      +-- safety checks
      |
      v
    delivered stimulation

A **source** is what is presented to the device: an electrical stimulus, an
image or video, or a :py:class:`~pulse2percept.vision.Scene`.

The **delivered stimulation** is what the electrodes or pixels receive after
the implant has processed that source. It is always represented as a
:py:class:`~pulse2percept.stimuli.Stimulus`.

The implant stores neither one.


Quick Start
===========

Models call :py:meth:`~pulse2percept.implants.Implant.prepare_stim`
internally. Call it directly when you want to inspect what a device actually
delivers:

.. code-block:: python

    import pulse2percept as p2p

    implant = p2p.implants.retina.ArgusII()

    source = p2p.stimuli.VideoStimulus('movie.mp4')
    delivered = implant.prepare_stim(source)

    delivered.plot()
    implant.plot(stim=source, stim_cmap=True)


Sources and Delivered Stimulation
=================================

.. list-table::
   :header-rows: 1
   :widths: 28 34 38

   * - Input
     - Interpretation
     - Processing
   * - Scalar, array, or dict
     - Electrical electrode values
     - Converted directly to a ``Stimulus``
   * - :py:class:`~pulse2percept.stimuli.Stimulus`
     - Electrical stimulation
     - Bypasses image/video encoding
   * - :py:class:`~pulse2percept.stimuli.ImageStimulus`
     - Static visual source
     - Preprocessing, encoding, raster scheduling
   * - :py:class:`~pulse2percept.stimuli.VideoStimulus`
     - Time-varying visual source
     - Preprocessing, encoding, raster scheduling
   * - :py:class:`~pulse2percept.vision.Scene`
     - Visual source positioned in visual space
     - Rendered and then processed by the implant

See :ref:`topics-vision` for scenes and :ref:`topics-implants` for
device-specific defaults.


Electrical Waveforms
====================

Most electrical stimulation starts with a
:py:class:`~pulse2percept.stimuli.BiphasicPulseTrain`.

Bare numeric values use uA, ms, and Hz; see :ref:`topics-units`.

.. code-block:: python

    pulse_train = p2p.stimuli.BiphasicPulseTrain(
        freq=20,          # Hz
        amp=50,           # uA
        phase_dur=0.45,   # ms
        stim_dur=500,     # ms; default 1000
    )

    model = p2p.models.retina.ScoreboardModel(implant=implant)
    percept = model.predict_percept({'A5': pulse_train})

The dictionary key selects the electrode. Electrodes not present in the
dictionary receive no stimulation.


Available Pulse Classes
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Stimulus
     - Description
   * - :py:class:`~pulse2percept.stimuli.BiphasicPulse`
     - One symmetric biphasic pulse
   * - :py:class:`~pulse2percept.stimuli.BiphasicPulseTrain`
     - Repeated biphasic pulses
   * - :py:class:`~pulse2percept.stimuli.MonophasicPulse`
     - One cathodic or anodic phase
   * - :py:class:`~pulse2percept.stimuli.AsymmetricBiphasicPulse`
     - Biphasic pulse with unequal phases
   * - :py:class:`~pulse2percept.stimuli.AsymmetricBiphasicPulseTrain`
     - Repeated asymmetric biphasic pulses
   * - :py:class:`~pulse2percept.stimuli.BiphasicTripletTrain`
     - Repeated triplets of biphasic pulses
   * - :py:class:`~pulse2percept.stimuli.PulseTrain`
     - Repetition of an arbitrary pulse

Pulses are cathodic-first by default.

Pulse trains contain only complete pulses. A pulse that would extend beyond
``stim_dur`` is omitted rather than truncated.


The Stimulus Container
======================

:py:class:`~pulse2percept.stimuli.Stimulus` is the common representation of
delivered stimulation.

.. list-table::
   :header-rows: 1
   :widths: 24 46 30

   * - Attribute
     - Meaning
     - Shape / units
   * - ``data``
     - Stimulation values
     - ``(n_electrodes, n_times)``
   * - ``electrodes``
     - Row labels
     - Electrode names or indices
   * - ``time``
     - Physical time axis
     - ``time_unit``; ms by default
   * - ``time_unit``
     - Unit used by ``time``
     - ms by default

``time`` is ``None`` for a timeless stimulus.

A ``Stimulus`` can be created from arrays, scalars, lists, dictionaries, or
other stimuli:

.. code-block:: python

    stim = p2p.stimuli.Stimulus({
        'A1': 10,
        'A2': 20,
        'A3': 30,
    })

    stim['A1']
    stim['A1', 10]    # value at t=10 ms; interpolated if necessary

Use ``stim.data`` for ordinary NumPy indexing.


Operations
----------

Stimuli are read-only, and most operations return a new object.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Operation
     - Behavior
   * - ``stim.plot()``
     - Heatmap for many electrodes or traces for selected electrodes
   * - ``stim.shift(dt)``
     - Shift the stimulus in time
   * - ``stim >> dt``
     - Shorthand for ``shift``
   * - ``stim.pad(...)``
     - Extend the stimulus with zeros
   * - ``stim.compress()``
     - Compress redundant samples; modifies in place
   * - ``stim.remove(...)``
     - Remove electrodes or samples; modifies in place

Pulse classes retain their waveform parameters, such as ``freq`` and ``amp``,
and generate sampled waveform data only when ``data`` is first accessed.

Arithmetic preserves the pulse class when the result is still the same kind
of waveform:

.. code-block:: python

    pt * 2      # still a pulse train
    pt + 5      # generic Stimulus


Visual Sources
==============

:py:class:`~pulse2percept.stimuli.ImageStimulus` and
:py:class:`~pulse2percept.stimuli.VideoStimulus` store dimensionless image
values rather than electrical current.

When passed directly to an implant, an image is mapped across the electrode
array. To position visual content in degrees of visual angle instead, use a
:py:class:`~pulse2percept.vision.Scene`; see :ref:`topics-vision`.

Both classes support common image operations, each returning a new object:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Operation
     - Available for
   * - ``rgb2gray``
     - Images and videos
   * - ``invert``
     - Images and videos
   * - ``resize``
     - Images and videos
   * - ``crop``
     - Images and videos
   * - ``crop_square``
     - Images and videos
   * - ``rotate``
     - Images and videos
   * - ``filter``
     - Images and videos; e.g. ``'sobel'``
   * - ``threshold``
     - Images

Bundled sample data are available under
:mod:`pulse2percept.stimuli.samples`:

.. code-block:: python

    from pulse2percept.stimuli import samples

    image = (
        samples.ucsb_bike(as_gray=True)
        .crop_square()
        .resize((60, 60))
    )

    video = samples.big_buck_bunny(resize=(60, 80))

The module documentation lists the available samples.
``pulse2percept/stimuli/data/samples/README.rst`` records their provenance and
licenses.

.. note::

   Pass a complete video to ``predict_percept`` when temporal dynamics across
   frames matter.

   Iterating over a ``VideoStimulus`` yields independent
   :py:class:`~pulse2percept.stimuli.ImageStimulus` frames. Passing those
   frames separately therefore resets any temporal model state between frames.


Psychophysical Stimuli
----------------------

:mod:`pulse2percept.stimuli.psychophysics` generates standard visual patterns
directly in degrees of visual angle and physical time.

Each generator returns a :py:class:`~pulse2percept.vision.Scene`. ``shape``
controls raster resolution without changing the stimulus geometry.

.. code-block:: python

    import numpy as np

    from pulse2percept.stimuli import psychophysics
    from pulse2percept.units import Hz, deg, dva, s

    c = psychophysics.landolt_c(
        gap=0.5 * dva,
        position=(5, 0) * dva,
        orientation=90 * deg,
        fov=15 * dva,
    )

    e = psychophysics.tumbling_e(
        stroke=0.5 * dva,
        position=(5, 0) * dva,
        orientation=90 * deg,
        fov=15 * dva,
    )

    grating = psychophysics.grating(
        spatial_freq=0.5 / dva,
        temporal_freq=2 * Hz,
        fov=20 * dva,
        time=np.arange(0, 1000, 20),
    )

    bar = psychophysics.bar(
        width=2 * dva,
        speed=20 * dva / s,
        offset=-10 * dva,
        fov=20 * dva,
        time=np.arange(0, 1000, 20),
    )

.. list-table::
   :header-rows: 1
   :widths: 27 73

   * - Quantity
     - Convention
   * - Optotype size
     - ``gap`` or ``stroke`` defines the critical feature in dva
   * - ``spatial_freq``
     - cycles/dva
   * - ``temporal_freq``
     - Hz
   * - ``speed``
     - dva/s
   * - ``direction``
     - Counterclockwise from the positive x axis
   * - ``time=None``
     - Static stimulus
   * - ``time``
     - Explicit sample times in ms for moving stimuli

Optotype critical features must span at least three output pixels. Optotypes
are supersampled and area-averaged onto the requested raster.

Spatial or temporal frequencies above the corresponding Nyquist limit are
rejected with ``ValueError`` rather than aliased.

.. note::

   ``GratingStimulus`` and ``BarStimulus`` use the legacy pixel/frame API and
   are deprecated until v0.12.


.. _topics-encoders:

Encoders
========

An encoder maps visual source values to device stimulation.

:py:class:`~pulse2percept.stimuli.ImplantEncoder` subclasses are attached to an
implant. Electrical encoders derive from
:py:class:`~pulse2percept.stimuli.PulseEncoder`; photovoltaic encoders produce
optical stimulation.

Model-aware encoders such as
:py:class:`~pulse2percept.stimuli.TraceEncoder` are different: they are called
directly through ``encode`` rather than attached to the implant.


Encoding Through an Implant
---------------------------

.. code-block:: python

    implant = p2p.implants.retina.ArgusII()

    implant.encoder = p2p.stimuli.AmplitudeEncoder(
        amp_range=(0, 50),   # uA
        freq=20,             # Hz
    )

    model = p2p.models.retina.ScoreboardModel(implant=implant)

    percept = model.predict_percept(
        p2p.stimuli.VideoStimulus('movie.mp4')
    )

Electrical ``Stimulus`` objects bypass the encoder.

For images and videos, the encoder first samples the source at each electrode
position, producing one value per electrode. Encoding can also be called
directly:

.. code-block:: python

    source = p2p.stimuli.VideoStimulus('movie.mp4')

    encoder = p2p.stimuli.AmplitudeEncoder(
        implant,
        amp_range=(0, 50),
        freq=20,
    )

    stim = encoder.encode(source)


Amplitude and Frequency Encoding
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 36 34 30

   * - Encoder
     - Gray level controls
     - Held fixed
   * - :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`
     - Pulse amplitude (``amp_range``)
     - ``freq``
   * - :py:class:`~pulse2percept.stimuli.FrequencyEncoder`
     - Pulse frequency (``freq_range``)
     - ``amp``

For example:

.. code-block:: python

    implant.encoder = p2p.stimuli.FrequencyEncoder(
        amp=50,              # uA
        freq_range=(0, 60),  # Hz
    )

For video, pulse trains run continuously across frame boundaries. The source
frame rate determines when the requested modulation changes.

If the pulse period is longer than a video frame, some frames contain no pulse;
``prepare_stim`` issues a warning in that case.

An encoded ``Stimulus`` stores both the frame-level values and the delivered
pulse schedule:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Consumer
     - Representation used
   * - Spatial-only model
     - Frame-level gray values
   * - Temporal model
     - Delivered pulse schedule

Waveform samples are generated lazily, so encoding a long video does not
require allocating the complete electrical waveform up front.


Optical Encoding
----------------

.. versionadded:: 0.11.0

:py:class:`~pulse2percept.stimuli.PhotovoltaicEncoder` maps gray level to the
ON duration of near-infrared pulses at a fixed peak irradiance.

The delivered ``Stimulus`` is expressed in mW/mm^2.

Each photovoltaic implant defaults to the protocol used by its experimental
system; see :ref:`topics-implants`.

.. code-block:: python

    implant = p2p.implants.retina.PRIMAPivotal()

    stim = implant.prepare_stim(
        p2p.stimuli.samples.logo_bvl()
    )

    stim.unit    # mW/mm^2

A different optical protocol can be supplied explicitly:

.. code-block:: python

    implant.encoder = p2p.stimuli.PhotovoltaicEncoder(
        irradiance=4,      # mW/mm^2
        freq=40,           # Hz
        pulse_dur=4,       # ms
        wavelength=915,    # nm
    )

:py:class:`~pulse2percept.stimuli.PRIMAEncoder` reproduces the pivotal
projector settings: 30 Hz, 3.5 mW/mm^2, with ON duration quantized to 14
nonzero steps of 0.7 ms from 0.7 to 9.8 ms.

The generic ``PhotovoltaicEncoder`` does not quantize ON duration.

.. note::

   **Grayscale mapping.** No source paper specifies a grayscale transfer
   function for natural images. pulse2percept therefore maps gray level
   linearly to ON duration. This is a software convention, not a reconstruction
   of the clinical camera pipeline. ``grayscale=False`` selects binary
   encoding.

   **Clinical image processing.** Ambient-light adaptation, contrast
   enhancement, zoom, and contrast inversion used by clinical PRIMA systems
   are not applied automatically. They can be supplied as preprocessing.

   **Video timing.** Video frames are sampled on the projector clock using
   zero-order hold.

   **Spatial models.** Spatial-only models receive normalized optical drive.
   A value of 1.0 corresponds to a fully illuminated pixel at the current
   encoder settings, or to the documented projector maximum for
   ``PRIMAEncoder``.


.. _topics-rasters:

Raster Scheduling
=================

A :py:class:`~pulse2percept.implants.Raster` divides electrodes into groups
that are stimulated at different times.

This is useful for devices that cannot drive every electrode simultaneously.
The encoder schedules pulses around the implant's raster.

.. code-block:: python

    implant = p2p.implants.retina.ArgusII()

    implant.raster = p2p.implants.CheckerboardRaster(
        n_groups=5
    )

    implant.encoder = p2p.stimuli.AmplitudeEncoder(
        amp_range=(0, 50),
        freq=20,
    )

    delivered = implant.prepare_stim(
        p2p.stimuli.VideoStimulus('movie.mp4')
    )


Available Rasters
-----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Raster
     - Grouping
   * - :py:class:`~pulse2percept.implants.SequentialRaster`
     - Consecutive electrodes, or interleaved electrodes with
       ``interleave=True``
   * - :py:class:`~pulse2percept.implants.CheckerboardRaster`
     - Spatially distributed grid positions
   * - :py:class:`~pulse2percept.implants.CustomRaster`
     - User-defined groups; every electrode belongs to exactly one group

``raster.plot()`` displays the grouping, while ``raster.members(...)`` returns
the electrodes in a group.

Groups are activated in order, separated by ``group_dur`` milliseconds. With
``group_dur=None``, the groups are distributed across the pulse period.

Each group slot must be long enough to contain its pulse, and the full raster
sweep must fit within the pulse period.


Rastering and Pulse Rate
------------------------

Amplitude and frequency encoding interact with the raster differently:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Encoder
     - Raster behavior
   * - Amplitude encoding
     - All electrodes share one pulse period; the raster offsets groups within
       that period
   * - Frequency encoding
     - Each electrode's period is rounded upward to a whole number of raster
       sweeps, so delivered frequency may be lower than requested but never
       higher

Leave ``implant.raster`` unset when raster scheduling is not part of the
device or research question.

PRIMA does not use a raster; all pixels can be illuminated simultaneously.


Device Constraints
==================

Device-specific constraints are applied during ``prepare_stim``.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Constraint
     - Behavior
   * - ``clock``
     - Quantizes supported timing
   * - ``n_levels``
     - Quantizes encoded gray levels
   * - ``thresholds``
     - Converts amplitudes expressed in ``xTh`` to uA
   * - ``safe_mode``
     - Applies device-specific safety checks when available

Timing quantization can reduce a requested pulse rate but never increase it.

Amplitudes expressed in ``xTh`` are converted using ``implant.thresholds``;
see :ref:`topics-units`.

For the PRIMA projector envelope enforced by ``safe_mode``, see
:ref:`topics-implants`.