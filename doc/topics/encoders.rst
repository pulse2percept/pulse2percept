.. _topics-encoders:

=================
Stimulus Encoders
=================

.. versionadded:: 0.10.0

A visual source contains gray levels; an implant delivers electrical pulses. A
:py:class:`~pulse2percept.stimuli.StimulusEncoder` defines the mapping between
the two.

Basic usage
-----------

Attach an encoder to an implant, then hand it an image or video:

.. code-block:: python

    import pulse2percept as p2p

    implant = p2p.implants.ArgusII()
    implant.encoder = p2p.stimuli.AmplitudeEncoder(
        amp_range=(0, 50),
        freq=20,
    )

    model = p2p.models.ScoreboardModel(implant=implant)
    percept = model.predict_percept(p2p.stimuli.BostonTrain())

Dimensionless input is encoded when the implant prepares it. Electrical stimuli
bypass the encoder. To see the pulses themselves:

.. code-block:: python

    delivered = implant.prepare_stim(p2p.stimuli.BostonTrain())

Encoding can also be explicit:

.. code-block:: python

    source = p2p.stimuli.BostonTrain()
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

Device constraints
------------------

An implant's :py:class:`~pulse2percept.implants.Raster` determines which
electrodes may pulse together; see :ref:`topics-rasters`.

Encoders can also quantize timing with ``clock`` and gray levels with
``n_levels``. These constraints are conservative: quantization may lower a
requested pulse rate, but never increases it.

Encoder amplitudes, frequencies, and durations follow the physical-unit
conventions in :ref:`topics-units`.
