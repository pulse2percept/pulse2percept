.. _topics-encoders:

=================
Stimulus Encoders
=================

.. versionadded:: 0.10.0

A visual source contains gray levels; an implant delivers stimulation. An
:py:class:`~pulse2percept.stimuli.Encoder` defines the mapping between the two.
:py:class:`~pulse2percept.stimuli.StimulusEncoder` covers devices driven by a
current source; :py:class:`~pulse2percept.stimuli.PhotovoltaicEncoder` covers
subretinal photovoltaic arrays, which are driven by light.

Basic usage
-----------

Attach an encoder to an implant, then hand it an image or video:

.. code-block:: python

    import pulse2percept as p2p

    implant = p2p.implants.retina.ArgusII()
    implant.encoder = p2p.stimuli.AmplitudeEncoder(
        amp_range=(0, 50),
        freq=20,
    )

    model = p2p.models.retina.ScoreboardModel(implant=implant)
    percept = model.predict_percept(p2p.stimuli.VideoStimulus('movie.mp4'))

Dimensionless input is encoded when the implant prepares it. Electrical stimuli
bypass the encoder. To see the pulses themselves:

.. code-block:: python

    delivered = implant.prepare_stim(p2p.stimuli.VideoStimulus('movie.mp4'))

Encoding can also be explicit:

.. code-block:: python

    source = p2p.stimuli.VideoStimulus('movie.mp4')
    stim = implant.encoder.encode(source, implant=implant)

Passing the implant samples the source at its electrode locations before pulse
trains are constructed, so the resulting Stimulus has one row per implant
electrode. That sampling is device-relative: the source is stretched across the
implant's bounding box. Registering a picture against the visual field instead
is a model's job, not an encoder's; see :ref:`topics-models-scene`.

Amplitude and frequency encoding
--------------------------------

pulse2percept provides two basic encoders:

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

What an encoded stimulus contains
---------------------------------

An encoded Stimulus retains both the requested frame-level modulation and the
delivered pulse schedule. Spatial-only models use the frame-level modulation;
temporal models use the delivered electrical pulses. The result is the same
whether encoding happened explicitly or inside ``prepare_stim``.

Waveform samples are generated lazily, so encoding a large image or video does
not allocate the full electrical waveform until something needs it.

Optical encoding
----------------

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
For example:

.. code-block:: python

    model = p2p.models.retina.ScoreboardModel(implant=implant)
    percept = model.predict_percept(p2p.stimuli.samples.logo_bvl())

Here ``ScoreboardModel`` visualizes implant geometry and optical drive. It does
not model photovoltaic transduction or retinal activation. Drive of 1.0 means a
fully lit pixel at the encoder's settings; for
:py:class:`~pulse2percept.stimuli.PRIMAEncoder` it means the projector's
documented maximum instead, so lowering any setting lowers the drive.

Device constraints
------------------

An implant's :py:class:`~pulse2percept.implants.Raster` determines which
electrodes may pulse together; see :ref:`topics-rasters`. PRIMA uses no raster;
all 378 pixels may be illuminated at once. With ``safe_mode=True``,
:py:class:`~pulse2percept.implants.retina.PRIMAPivotal` checks the documented projector
settings (880 nm, 3.5 mW/mm^2, 30 Hz, 0.7--9.8 ms ON durations, and duty cycle
<= 0.294).
This is not a biological safety check or a demonstrated hardware maximum. That
envelope is specific to the pivotal projector: research arrays with no
published envelope of their own raise on ``safe_mode=True`` rather than borrow
it.

Encoders can also quantize timing with ``clock`` and gray levels with
``n_levels``. These constraints are conservative: quantization may lower a
requested pulse rate, but never increases it.

Encoder amplitudes, frequencies, and durations follow the physical-unit
conventions in :ref:`topics-units`.
