.. _topics-stimulation:

.. _topics-stimuli:

============
Stimulation
============

Stimulation always follows the same path::

    source -> implant.prepare_stim() -> delivered stimulation

The **source** is what you hand the device: a
:py:class:`~pulse2percept.stimuli.Stimulus` (or a compatible scalar, array, or
dict), an :py:class:`~pulse2percept.stimuli.ImageStimulus`, a
:py:class:`~pulse2percept.stimuli.VideoStimulus`, or a
:py:class:`~pulse2percept.vision.Scene`.

The **delivered stimulation** is what the electrodes actually do. Getting
there may involve preprocessing, image/video encoding, resampling onto the
electrode array, raster scheduling, threshold calibration, and safety checks.
Only the result is a statement about the device; the source is a request.

A source is trial input, not implant state: the implant keeps nothing.
Models call :py:meth:`~pulse2percept.implants.Implant.prepare_stim`
internally, so call it directly when the delivered stimulation itself is of
interest:

.. code-block:: python

    import pulse2percept as p2p

    implant = p2p.implants.retina.ArgusII()

    source = p2p.stimuli.VideoStimulus('movie.mp4')
    delivered = implant.prepare_stim(source)

    delivered.plot()
    implant.plot(stim=source, stim_cmap=True)

The result is always a :py:class:`~pulse2percept.stimuli.Stimulus`.

Electrical waveforms
--------------------

For most electrical stimulation, start with a
:py:class:`~pulse2percept.stimuli.BiphasicPulseTrain`:

.. code-block:: python

    pulse_train = p2p.stimuli.BiphasicPulseTrain(
        freq=20,
        amp=50,
        phase_dur=0.45,
        stim_dur=500,
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

Pulse trains deliver only complete pulses. A pulse that would extend beyond
``stim_dur`` is omitted rather than truncated.

Electrical amplitudes, time, and frequency use the unit conventions described
in :ref:`topics-units`.

The Stimulus container
----------------------

A :py:class:`~pulse2percept.stimuli.Stimulus` is labeled two-dimensional data:
rows are electrodes and columns are points in time. Every Stimulus exposes:

``data``
    A NumPy array with shape ``(n_electrodes, n_times)``.

``electrodes``
    The labels corresponding to the rows.

``time``
    The time axis, or ``None`` for a timeless stimulus.

A Stimulus can be built from arrays, scalars, lists, dictionaries, or other
Stimulus objects:

.. code-block:: python

    stim = p2p.stimuli.Stimulus({'A1': 10, 'A2': 20, 'A3': 30})

Stimulus indexing uses electrode labels and physical time:

.. code-block:: python

    stim['A1']
    stim['A1', 10]

The second index is a time, not a column number. If the exact time is not
stored, pulse2percept interpolates the waveform there. Use ``stim.data`` for
ordinary NumPy indexing.

Stimulus state is read-only. Pulse classes retain their defining parameters and
generate waveform samples only when needed:

.. code-block:: python

    pt = p2p.stimuli.BiphasicPulseTrain(20, 50, 0.45)

    pt.freq, pt.amp, pt.phase_dur
    pt.data  # generate and cache the waveform

Operations preserve the structured form when that remains truthful. For
example, ``pt * 2`` is still a pulse train, while ``pt + 5`` becomes a plain
Stimulus because a DC offset is no longer a pulse train.

Most operations return a new object. The older
:py:meth:`~pulse2percept.stimuli.Stimulus.compress` and
:py:meth:`~pulse2percept.stimuli.Stimulus.remove` methods still modify the
Stimulus in place.

:py:meth:`~pulse2percept.stimuli.Stimulus.plot` shows a heatmap for
multi-electrode stimuli and waveform traces for a single or explicitly selected
electrode:

.. code-block:: python

    stim.plot()
    stim.plot(electrodes=['A1', 'A2'])

:py:meth:`~pulse2percept.stimuli.Stimulus.shift` moves a stimulus in time, and
:py:meth:`~pulse2percept.stimuli.Stimulus.pad` adds zero-valued endpoints to a
requested end time. ``stim >> dt`` and ``stim << dt`` are shorthand for
positive and negative shifts.

Visual sources
--------------

:py:class:`~pulse2percept.stimuli.ImageStimulus` and
:py:class:`~pulse2percept.stimuli.VideoStimulus` are visual sources, not
currents. Their values are dimensionless gray levels, and an
:py:class:`~pulse2percept.stimuli.Encoder` is what turns them into
stimulation.

An image handed to an implant is *device-relative*: its pixels are stretched
across the implant's electrodes, and the picture means nothing beyond "this is
what the device was shown". To place a picture in the visual field instead --
so that each electrode sees the part of it that electrode actually looks at --
wrap it in a :py:class:`~pulse2percept.vision.Scene`; see
:ref:`topics-vision`.

A video can also be processed one frame at a time. Iterating over a
:py:class:`~pulse2percept.stimuli.VideoStimulus` yields each frame as an
:py:class:`~pulse2percept.stimuli.ImageStimulus`:

.. code-block:: python

    video = p2p.stimuli.VideoStimulus('movie.mp4')

    for frame in video:
        percept = model.predict_percept(frame)

Each call treats the frame as an independent still image. Pass the complete
video to ``predict_percept`` instead when temporal dynamics across frames
matter.

:py:mod:`pulse2percept.stimuli.samples` provides bundled images and videos for
examples and tests:

.. code-block:: python

    from pulse2percept.stimuli import samples

    image = samples.ucsb_bike()
    video = samples.big_buck_bunny(resize=(60, 80))

See the :py:mod:`~pulse2percept.stimuli.samples` API for the available assets
and ``pulse2percept/stimuli/data/samples/README.rst`` for their provenance and
licensing.

Psychophysical stimuli
~~~~~~~~~~~~~~~~~~~~~~

:py:mod:`pulse2percept.stimuli.psychophysics` generates calibrated visual
patterns in degrees of visual angle and physical time. The generators return
a :py:class:`~pulse2percept.vision.Scene`; ``shape`` controls raster resolution
without changing the stimulus geometry.

.. code-block:: python

    from pulse2percept.stimuli import psychophysics
    from pulse2percept.units import deg, dva

    c = psychophysics.landolt_c(
        gap=0.5 * dva, position=(5, 0) * dva,
        orientation=90 * deg, fov=15 * dva)

    e = psychophysics.tumbling_e(
        stroke=0.5 * dva, position=(5, 0) * dva,
        orientation=90 * deg, fov=15 * dva)

For a Landolt C, ``gap`` is the critical feature; for a Tumbling E it is
``stroke``. Both use standard optotype proportions and are supersampled before
being area-averaged onto the requested raster. The critical feature must span
at least three output pixels.

Gratings and bars use the same visual-field coordinates:

.. code-block:: python

    import numpy as np
    from pulse2percept.units import Hz, s

    grating = psychophysics.grating(
        spatial_freq=0.5 / dva, temporal_freq=2 * Hz,
        fov=20 * dva, time=np.arange(0, 1000, 20))

    bar = psychophysics.bar(
        width=2 * dva, speed=20 * dva / s, offset=-10 * dva,
        fov=20 * dva, time=np.arange(0, 1000, 20))

``spatial_freq`` is measured in cycles/dva, ``temporal_freq`` in Hz, and bar
width, position, and speed in dva or dva/s. ``direction`` is measured
counterclockwise from the positive x axis.

With ``time=None`` the result contains an ``ImageStimulus``. Moving stimuli
require explicit sample times and contain a ``VideoStimulus``; no frame rate
is assumed. Gratings that exceed the spatial or temporal Nyquist limit are
rejected rather than silently aliased.

``GratingStimulus`` and ``BarStimulus`` use the legacy pixel/frame API and are
deprecated until v0.12.

.. _topics-encoders:

Encoding gray levels as stimulation
-----------------------------------

An :py:class:`~pulse2percept.stimuli.Encoder` defines the mapping from gray
level to stimulation.
:py:class:`~pulse2percept.stimuli.StimulusEncoder` covers devices driven by a
current source; :py:class:`~pulse2percept.stimuli.PhotovoltaicEncoder` covers
subretinal photovoltaic arrays, which are driven by light.

Attach an encoder to an implant, then hand it an image or video:

.. code-block:: python

    implant = p2p.implants.retina.ArgusII()
    implant.encoder = p2p.stimuli.AmplitudeEncoder(
        amp_range=(0, 50),
        freq=20,
    )

    model = p2p.models.retina.ScoreboardModel(implant=implant)
    percept = model.predict_percept(p2p.stimuli.VideoStimulus('movie.mp4'))

Dimensionless input is encoded when the implant prepares it. Electrical stimuli
bypass the encoder. Encoding can also be explicit:

.. code-block:: python

    source = p2p.stimuli.VideoStimulus('movie.mp4')
    stim = implant.encoder.encode(source, implant=implant)

Passing the implant samples the source at its electrode locations before pulse
trains are constructed, so the resulting Stimulus has one row per implant
electrode. That sampling is device-relative: the source is stretched across the
implant's bounding box. Registering a picture against the visual field instead
is a model's job, not an encoder's; see :ref:`topics-vision`.

Amplitude and frequency encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Encoder
     - Gray level controls
   * - :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`
     - Pulse amplitude
   * - :py:class:`~pulse2percept.stimuli.FrequencyEncoder`
     - Pulse frequency

Amplitude encoding keeps frequency fixed. Frequency encoding keeps amplitude
fixed:

.. code-block:: python

    implant.encoder = p2p.stimuli.FrequencyEncoder(
        amp=50,
        freq_range=(0, 60),
    )

For video, pulse timing is continuous across frame boundaries. The video frame
rate determines when requested modulation changes; it does not restart the
pulse train.

An encoded Stimulus retains both the requested frame-level modulation and the
delivered pulse schedule. Spatial-only models use the frame-level modulation;
temporal models use the delivered electrical pulses. The result is the same
whether encoding happened explicitly or inside ``prepare_stim``.

Waveform samples are generated lazily, so encoding a large image or video does
not allocate the full electrical waveform until something needs it.

Optical encoding
~~~~~~~~~~~~~~~~

.. versionadded:: 0.11.0

Photovoltaic arrays are illuminated by pulsed near-infrared light rather than
driven by a current source.
:py:class:`~pulse2percept.stimuli.PhotovoltaicEncoder` maps image intensity to
ON duration at fixed peak irradiance and returns ``mW/mm^2``:

.. code-block:: python

    implant = p2p.implants.retina.PRIMAPivotal()
    stim = implant.prepare_stim(p2p.stimuli.samples.logo_bvl())
    stim.unit  # mW/mm^2

The implant class describes the array; the encoder describes an optical
stimulation protocol. Each photovoltaic implant therefore defaults to an
encoder configured for its own experimental system (see
:ref:`topics-implants`), and any array accepts any compatible encoder:

.. code-block:: python

    implant.encoder = p2p.stimuli.PhotovoltaicEncoder(
        irradiance=4,      # mW/mm^2
        freq=40,           # Hz
        pulse_dur=4,       # ms
        wavelength=915,    # nm
    )

:py:class:`~pulse2percept.stimuli.PRIMAEncoder` is the PRIMA specialization.
Its defaults are the pivotal projector: 30 Hz, 3.5 mW/mm^2, and 14 nonzero ON
durations from 0.7 to 9.8 ms. It quantizes duration onto that 0.7 ms grid; the
generic encoder does not, so it can express protocols such as 4 ms at 40 Hz or
10 ms at 2 Hz.

Where a source paper does not specify a natural-image grayscale transfer
function -- which is the case for all of these systems -- the mapping from
gray level to ON duration is an explicit pulse2percept simulation convention
(linear), not a reconstruction of the original camera pipeline. Set
``grayscale=False`` for binary encoding.

The clinical PRIMA system also applies ambient-light adaptation, contrast
enhancement, zoom, and, in some tests, contrast inversion. These operations
remain explicit preprocessing steps in pulse2percept.

For videos, source frames are sampled on the projector clock using zero-order
hold. Spatial-only models can read the resulting *normalized optical drive*.
Drive of 1.0 means a fully lit pixel at the encoder's settings; for
:py:class:`~pulse2percept.stimuli.PRIMAEncoder` it means the projector's
documented maximum instead, so lowering any setting lowers the drive.
:py:class:`~pulse2percept.models.retina.ScoreboardModel` visualizes that drive
and models no retinal response;
:py:class:`~pulse2percept.models.retina.Ho2018Model` predicts one (see
:ref:`topics-models`).

.. _topics-rasters:

Raster scheduling
-----------------

Some stimulators cannot drive every electrode simultaneously. A
:py:class:`~pulse2percept.implants.Raster` divides an array into groups that
take turns. It is a scheduling constraint used by an encoder, not a stimulus by
itself, and is attached to the implant:

.. code-block:: python

    implant = p2p.implants.retina.ArgusII()
    implant.raster = p2p.implants.CheckerboardRaster(n_groups=5)
    implant.encoder = p2p.stimuli.AmplitudeEncoder(amp_range=(0, 50), freq=20)

    delivered = implant.prepare_stim(p2p.stimuli.VideoStimulus('movie.mp4'))

Electrodes in one group may pulse together; different groups occupy different
time slots.

.. list-table::
   :header-rows: 1

   * - Raster
     - Grouping
   * - :py:class:`~pulse2percept.implants.SequentialRaster`
     - Sequential or interleaved groups
   * - :py:class:`~pulse2percept.implants.CheckerboardRaster`
     - Spatially distributed grid groups
   * - :py:class:`~pulse2percept.implants.CustomRaster`
     - Explicit user-defined groups

.. code-block:: python

    implant.raster = p2p.implants.SequentialRaster(n_groups=6)
    implant.raster = p2p.implants.SequentialRaster(n_groups=6, interleave=True)

A CustomRaster is useful when the hardware already defines the groups. Every
electrode must belong to exactly one group. Use ``raster.plot()`` to inspect
the pattern and ``raster.members(...)`` to retrieve the electrodes in a group.

Groups fire in order. ``group_dur`` sets the spacing between group starts. If
it is ``None``, the encoder spreads the groups across the pulse period. An
explicit value fixes the raster sweep duration:

.. code-block:: python

    implant.raster = p2p.implants.SequentialRaster(n_groups=6, group_dur=1)

The slot must be long enough for a pulse, and the full sweep must fit within the
relevant pulse period.

With amplitude encoding, all electrodes share a pulse period, so rastering
only offsets the groups. With frequency encoding, electrodes may request
different periods; those schedules are constrained to whole raster sweeps and
may therefore run more slowly than requested, never faster.

If rastering is not part of the device or question being modeled, leave
``implant.raster`` unset. PRIMA uses no raster; all 378 pixels may be
illuminated at once.

Device constraints
------------------

Requested values are not always deliverable. Encoders can quantize timing with
``clock`` and gray levels with ``n_levels``. These constraints are
conservative: quantization may lower a requested pulse rate, but never
increases it.

With ``safe_mode=True``,
:py:class:`~pulse2percept.implants.retina.PRIMAPivotal` checks the documented
projector settings (880 nm, 3.5 mW/mm^2, 30 Hz, 0.7--9.8 ms ON durations, and
duty cycle <= 0.294). This is not a biological safety check or a demonstrated
hardware maximum. That envelope is specific to the pivotal projector: research
arrays with no published envelope of their own raise on ``safe_mode=True``
rather than borrow it.

Pulses expressed in ``xTh`` (multiples of perceptual threshold) are calibrated
to microamps using ``implant.thresholds``; see :ref:`topics-units`.
