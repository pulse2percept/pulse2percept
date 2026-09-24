.. _topics-stimulation:

.. _topics-stimuli:

============
Stimulation
============

Stimulation always follows the same path::

    source -> implant.prepare_stim() -> delivered stimulation

The **source** is what you present to the device: a
:py:class:`~pulse2percept.stimuli.Stimulus` (or a scalar, array, or dict), an
:py:class:`~pulse2percept.stimuli.ImageStimulus`, a
:py:class:`~pulse2percept.stimuli.VideoStimulus`, or a
:py:class:`~pulse2percept.vision.Scene`.

The **delivered stimulation** is what the electrodes do after preprocessing,
encoding, rastering, threshold calibration, and safety checks. It is always a
:py:class:`~pulse2percept.stimuli.Stimulus`. The implant stores neither.

Models call :py:meth:`~pulse2percept.implants.Implant.prepare_stim`
internally. Call it directly to inspect the delivered stimulation:

.. code-block:: python

    import pulse2percept as p2p

    implant = p2p.implants.retina.ArgusII()

    source = p2p.stimuli.VideoStimulus('movie.mp4')
    delivered = implant.prepare_stim(source)

    delivered.plot()
    implant.plot(stim=source, stim_cmap=True)

Electrical waveforms
--------------------

Most electrical stimulation starts with a
:py:class:`~pulse2percept.stimuli.BiphasicPulseTrain`. Bare numbers are uA,
ms, and Hz (see :ref:`topics-units`):

.. code-block:: python

    pulse_train = p2p.stimuli.BiphasicPulseTrain(
        freq=20,          # Hz
        amp=50,           # uA
        phase_dur=0.45,   # ms
        stim_dur=500,     # ms, default 1000
    )

    model = p2p.models.retina.ScoreboardModel(implant=implant)
    percept = model.predict_percept({'A5': pulse_train})

The dictionary key selects the electrode; unlisted electrodes receive no
stimulation.

.. list-table::
   :header-rows: 1

   * - Stimulus
     - Description
   * - :py:class:`~pulse2percept.stimuli.BiphasicPulse`
     - One symmetric biphasic pulse
   * - :py:class:`~pulse2percept.stimuli.BiphasicPulseTrain`
     - Repeated biphasic pulses
   * - :py:class:`~pulse2percept.stimuli.MonophasicPulse`
     - One cathodic or anodic phase
   * - :py:class:`~pulse2percept.stimuli.AsymmetricBiphasicPulse`
     - Unequal biphasic phases
   * - :py:class:`~pulse2percept.stimuli.AsymmetricBiphasicPulseTrain`
     - Repeated asymmetric pulses
   * - :py:class:`~pulse2percept.stimuli.BiphasicTripletTrain`
     - Repeated biphasic triplets
   * - :py:class:`~pulse2percept.stimuli.PulseTrain`
     - Repeats an arbitrary pulse

Pulses are cathodic-first by default. Pulse trains contain only complete
pulses: a pulse that would extend past ``stim_dur`` is omitted, not truncated.

The Stimulus container
----------------------

A :py:class:`~pulse2percept.stimuli.Stimulus` is labeled 2D data:

``data``
    NumPy array, shape ``(n_electrodes, n_times)``.

``electrodes``
    Row labels.

``time``
    Time axis in ``time_unit`` (ms by default), or ``None`` for a timeless
    stimulus.

It can be built from arrays, scalars, lists, dicts, or other stimuli, and is
indexed by electrode label and physical time:

.. code-block:: python

    stim = p2p.stimuli.Stimulus({'A1': 10, 'A2': 20, 'A3': 30})

    stim['A1']
    stim['A1', 10]    # value at t = 10 ms, interpolated if not stored

Use ``stim.data`` for ordinary NumPy indexing.

Stimuli are read-only, and most operations return a new object
(:py:meth:`~pulse2percept.stimuli.Stimulus.compress` and
:py:meth:`~pulse2percept.stimuli.Stimulus.remove` still modify in place).
Pulse classes keep their parameters (``pt.freq``, ``pt.amp``) and generate
waveform samples on first access to ``data``. Arithmetic keeps the class
where the result is still that waveform: ``pt * 2`` is a pulse train,
``pt + 5`` is a plain Stimulus.

:py:meth:`~pulse2percept.stimuli.Stimulus.plot` draws a heatmap for many
electrodes and traces for selected ones (``stim.plot(electrodes=['A1'])``).
:py:meth:`~pulse2percept.stimuli.Stimulus.shift` (or ``stim >> dt``) moves a
stimulus in time; :py:meth:`~pulse2percept.stimuli.Stimulus.pad` extends it
with zeros.

Visual sources
--------------

:py:class:`~pulse2percept.stimuli.ImageStimulus` and
:py:class:`~pulse2percept.stimuli.VideoStimulus` hold dimensionless gray
levels, not currents. An :py:class:`~pulse2percept.stimuli.Encoder` converts
them to stimulation. Passed directly to an implant, the picture is stretched
across the array; to place it in the visual field, wrap it in a
:py:class:`~pulse2percept.vision.Scene` (see :ref:`topics-vision`).

Both classes load files or arrays and provide common image operations, each
returning a new stimulus: ``rgb2gray``, ``invert``, ``resize``, ``crop``,
``crop_square``, ``rotate``, ``filter`` (e.g. ``'sobel'``), and, for images,
``threshold``.

.. code-block:: python

    from pulse2percept.stimuli import samples

    image = samples.ucsb_bike(as_gray=True).crop_square().resize((60, 60))
    video = samples.big_buck_bunny(resize=(60, 80))

:py:mod:`~pulse2percept.stimuli.samples` lists the bundled images and videos;
``pulse2percept/stimuli/data/samples/README.rst`` records their provenance
and licenses.

Pass a whole video to ``predict_percept`` when temporal dynamics across frames
matter. Iterating over a video yields each frame as an ``ImageStimulus``,
which a model treats as an independent still image.

Psychophysical stimuli
~~~~~~~~~~~~~~~~~~~~~~

:py:mod:`pulse2percept.stimuli.psychophysics` generates visual patterns in
degrees of visual angle (dva) and physical time. Each generator returns a
:py:class:`~pulse2percept.vision.Scene`; ``shape`` sets the raster
resolution without changing the stimulus geometry.

.. code-block:: python

    import numpy as np
    from pulse2percept.stimuli import psychophysics
    from pulse2percept.units import Hz, deg, dva, s

    c = psychophysics.landolt_c(
        gap=0.5 * dva, position=(5, 0) * dva,
        orientation=90 * deg, fov=15 * dva)

    e = psychophysics.tumbling_e(
        stroke=0.5 * dva, position=(5, 0) * dva,
        orientation=90 * deg, fov=15 * dva)

    grating = psychophysics.grating(
        spatial_freq=0.5 / dva, temporal_freq=2 * Hz,
        fov=20 * dva, time=np.arange(0, 1000, 20))

    bar = psychophysics.bar(
        width=2 * dva, speed=20 * dva / s, offset=-10 * dva,
        fov=20 * dva, time=np.arange(0, 1000, 20))

*  Optotypes use standard proportions. The critical feature (``gap`` or
   ``stroke``) must span at least three output pixels; optotypes are
   supersampled and area-averaged onto the raster.
*  ``spatial_freq`` is in cycles/dva, ``temporal_freq`` in Hz, bar ``speed``
   in dva/s. ``direction`` is counterclockwise from the positive x axis.
*  ``time=None`` gives a still image. Moving stimuli require explicit sample
   times (ms); no frame rate is assumed.
*  Gratings above the spatial or temporal Nyquist limit are rejected
   (``ValueError``) rather than aliased.

``GratingStimulus`` and ``BarStimulus`` use the legacy pixel/frame API and are
deprecated until v0.12.

.. _topics-encoders:

Encoders
--------

An :py:class:`~pulse2percept.stimuli.Encoder` maps gray level to
stimulation. :py:class:`~pulse2percept.stimuli.StimulusEncoder` subclasses
drive current; :py:class:`~pulse2percept.stimuli.PhotovoltaicEncoder` drives
light. Attach one to an implant, then pass it an image or video:

.. code-block:: python

    implant = p2p.implants.retina.ArgusII()
    implant.encoder = p2p.stimuli.AmplitudeEncoder(
        amp_range=(0, 50),   # uA
        freq=20,             # Hz
    )

    model = p2p.models.retina.ScoreboardModel(implant=implant)
    percept = model.predict_percept(p2p.stimuli.VideoStimulus('movie.mp4'))

Electrical stimuli bypass the encoder. The encoder first samples the source
at each electrode's position, so the result has one row per electrode.
Encoding can also be called explicitly, with the same result:

.. code-block:: python

    stim = implant.encoder.encode(source, implant=implant)

Amplitude and frequency encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Encoder
     - Gray level sets
     - Fixed
   * - :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`
     - pulse amplitude (``amp_range``)
     - ``freq``
   * - :py:class:`~pulse2percept.stimuli.FrequencyEncoder`
     - pulse frequency (``freq_range``)
     - ``amp``

.. code-block:: python

    implant.encoder = p2p.stimuli.FrequencyEncoder(
        amp=50,              # uA
        freq_range=(0, 60),  # Hz
    )

For video, the pulse train runs continuously across frame boundaries; the
frame rate sets when the requested modulation changes. If the pulse period is
longer than a frame, some frames deliver no pulse, and ``prepare_stim`` warns.

An encoded Stimulus stores both the frame-level gray levels and the delivered
pulse schedule. Spatial-only models use the gray levels; temporal models use
the pulses. Waveform samples are generated only when needed, so encoding a
long video does not allocate the full waveform.

Optical encoding
~~~~~~~~~~~~~~~~

.. versionadded:: 0.11.0

:py:class:`~pulse2percept.stimuli.PhotovoltaicEncoder` maps gray level to
the ON duration of near-infrared pulses at fixed peak irradiance, and returns
irradiance in mW/mm^2. Each photovoltaic implant defaults to the protocol of
its own experimental system (see :ref:`topics-implants`); any compatible
encoder can replace it:

.. code-block:: python

    implant = p2p.implants.retina.PRIMAPivotal()
    stim = implant.prepare_stim(p2p.stimuli.samples.logo_bvl())
    stim.unit  # mW/mm^2

    implant.encoder = p2p.stimuli.PhotovoltaicEncoder(
        irradiance=4,      # mW/mm^2
        freq=40,           # Hz
        pulse_dur=4,       # ms
        wavelength=915,    # nm
    )

:py:class:`~pulse2percept.stimuli.PRIMAEncoder` defaults to the pivotal
projector (30 Hz, 3.5 mW/mm^2) and quantizes ON duration to 14 nonzero steps
of 0.7 ms (0.7-9.8 ms). The generic encoder does not quantize.

*  No source paper specifies a grayscale transfer function for natural
   images, so gray level maps linearly to ON duration. This is a
   pulse2percept convention, not a reconstruction of the camera pipeline.
   ``grayscale=False`` gives binary encoding.
*  Clinical PRIMA processing (ambient-light adaptation, contrast
   enhancement, zoom, contrast inversion) is not applied; add it as
   preprocessing.
*  Video frames are sampled on the projector clock (zero-order hold).
*  Spatial-only models read *normalized optical drive*: 1.0 is a fully lit
   pixel at the encoder's settings, or, for ``PRIMAEncoder``, the projector's
   documented maximum.

.. _topics-rasters:

Raster scheduling
-----------------

A :py:class:`~pulse2percept.implants.Raster` divides the electrodes into
groups that take turns, for stimulators that cannot drive every electrode at
once. The encoder schedules pulses against the implant's raster:

.. code-block:: python

    implant = p2p.implants.retina.ArgusII()
    implant.raster = p2p.implants.CheckerboardRaster(n_groups=5)
    implant.encoder = p2p.stimuli.AmplitudeEncoder(amp_range=(0, 50), freq=20)

    delivered = implant.prepare_stim(p2p.stimuli.VideoStimulus('movie.mp4'))

.. list-table::
   :header-rows: 1

   * - Raster
     - Groups
   * - :py:class:`~pulse2percept.implants.SequentialRaster`
     - Consecutive or interleaved (``interleave=True``) electrodes
   * - :py:class:`~pulse2percept.implants.CheckerboardRaster`
     - Spatially distributed grid positions
   * - :py:class:`~pulse2percept.implants.CustomRaster`
     - User-defined; every electrode in exactly one group

``raster.plot()`` shows the groups; ``raster.members(...)`` lists a group's
electrodes. Groups fire in order, ``group_dur`` (ms) apart. With
``group_dur=None``, groups are spread across the pulse period. Each slot must
fit a pulse, and the full sweep must fit within the pulse period.

With amplitude encoding, all electrodes share one pulse period and the raster
only offsets the groups. With frequency encoding, each electrode's period is
rounded up to whole raster sweeps, so delivered rates can be lower than
requested, never higher.

Leave ``implant.raster`` unset if rastering is not part of the question.
PRIMA has no raster; all pixels can be illuminated at once.

Device constraints
------------------

Encoders can quantize timing (``clock``) and gray levels (``n_levels``).
Quantization can lower a requested pulse rate, never increase it. For the
PRIMA projector envelope under ``safe_mode``, see :ref:`topics-implants`.

Amplitudes given in ``xTh`` (multiples of perceptual threshold) are converted
to uA using ``implant.thresholds``; see :ref:`topics-units`.
