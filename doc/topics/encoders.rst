.. _topics-encoders:

=================
Stimulus Encoders
=================

.. versionadded:: 0.10.0

A visual source contains gray levels; an implant delivers stimulation. An
:py:class:`~pulse2percept.stimuli.Encoder` defines the mapping between the two.
:py:class:`~pulse2percept.stimuli.StimulusEncoder` covers devices driven by a
current source; :py:class:`~pulse2percept.stimuli.PRIMAEncoder` covers the
photovoltaic PRIMA system, which is driven by light.

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

Optical encoding
----------------

.. versionadded:: 0.11.0

:py:class:`~pulse2percept.implants.PRIMAPivotal` is photovoltaic: a projector
paints an 880 nm image onto the implant, and each pixel turns the light it
receives into local electrical stimulation. Intensity is set by *how long* a
pixel is lit rather than by how brightly, so
:py:class:`~pulse2percept.stimuli.PRIMAEncoder` returns absolute irradiance in
``mW/mm^2``, not microamps:

.. code-block:: python

    implant = p2p.implants.PRIMAPivotal()
    stim = implant.prepare_stim(p2p.stimuli.LogoBVL())
    stim.unit  # mW/mm^2

Encoding is binary by default (a pixel is off, or on for ``pulse_dur``), which
matches current clinical operation; ``grayscale=True`` pulse-width modulates
gray levels onto the projector's 0.7 ms duration grid instead. Peak irradiance
is fixed either way. Contrast inversion and edge enhancement are not part of
the device, so they are ordinary ``ImageStimulus`` operations applied first
(``image.invert()``, ``image.filter('canny')``).

The projector clock samples a video; it does not re-time it. Every ``1 / freq``
the device looks at whatever frame the source is showing (zero-order hold), so
a 15 fps source has its frames re-sent and a 60 fps source has frames skipped,
and either way the content keeps its own duration.

Converting light into retinal current is the job of a photovoltaic model, which
pulse2percept does not have yet. Until it does, the encoded stimulus offers
spatial-only models a *normalized optical drive* -- peak irradiance times duty
cycle, divided by the largest drive the pivotal device is documented to
produce -- and
:py:class:`~pulse2percept.models.ScoreboardModel` reads it:

.. code-block:: python

    model = p2p.models.ScoreboardModel(implant=implant)
    percept = model.predict_percept(p2p.stimuli.LogoBVL())

That is a visualization of implant geometry and stimulation pattern. It does
not model photodiode transduction, retinal electric fields, bipolar-cell
activation, electrode-retina distance, or temporal retinal dynamics.

Device constraints
------------------

An implant's :py:class:`~pulse2percept.implants.Raster` determines which
electrodes may pulse together; see :ref:`topics-rasters`. PRIMA has none: all
378 photovoltaic pixels may be illuminated at once. With ``safe_mode=True`` it
instead checks the projector envelope -- at most 3.5 mW/mm^2, at most 30 Hz,
ON durations on the 0.7 ms grid and no longer than 9.8 ms, and a duty cycle of
at most 0.294. That is the *device* envelope, not a biological safety limit,
and it is read off the stimulus' own schedule: a stimulus that has been reduced
to samples cannot be verified and is refused.

Encoders can also quantize timing with ``clock`` and gray levels with
``n_levels``. These constraints are conservative: quantization may lower a
requested pulse rate, but never increases it.

Encoder amplitudes, frequencies, and durations follow the physical-unit
conventions in :ref:`topics-units`.
