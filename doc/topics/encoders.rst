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

Attach an encoder to an implant, then assign an image or video:

.. code-block:: python

    import pulse2percept as p2p

    implant = p2p.implants.ArgusII()
    implant.encoder = p2p.stimuli.AmplitudeEncoder(
        amp_range=(0, 50),
        freq=20,
    )
    implant.stim = p2p.stimuli.BostonTrain()

Dimensionless input is encoded on assignment. Electrical stimuli bypass the
encoder.

Encoding can also be explicit:

.. code-block:: python

    source = p2p.stimuli.BostonTrain()
    stim = implant.encoder.encode(source, implant=implant)
    implant.stim = stim

Passing the implant samples the source at its electrode locations before pulse
trains are constructed, so the resulting Stimulus has one row per implant
electrode.

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
whether encoding happened explicitly or during assignment to the implant.

Waveform samples are generated lazily, so encoding a large image or video does
not allocate the full electrical waveform until something needs it.

Where the electrodes look
-------------------------

By default an image is *device-relative*: it is stretched across the implant's
electrodes, and the picture means nothing beyond "this is what the device was
shown". An image that states a ``fov`` (see :ref:`topics-stimuli`) is instead a
scene in the visual field, and registering it needs two more things: the
retinotopy that says where each electrode looks, and where the eye is pointing.

.. code-block:: python

    from pulse2percept.units import dva

    scene = p2p.stimuli.ImageStimulus('scene.png', fov=30 * dva)
    model = p2p.models.ScoreboardModel(rho=200).build()

    implant.stim = implant.encoder.encode(
        scene, implant=implant, vfmap=model.vfmap, gaze=(0, 0) * dva,
    )

Each electrode is followed out along this chain::

    retinal coordinate (um)
      -> vfmap.ret_to_dva -> eye-centered visual field (dva)
      -> + gaze            -> scene coordinate (dva)
      -> sample the image

``gaze`` is the scene location that currently falls on the fovea, so
``scene = visual field + gaze``. The implant does not move when gaze does, and
neither does an eye-centered :py:class:`~pulse2percept.vision.Scotoma`: the two
hold their positions relative to each other while the scene moves past them.
Pass one ``(x, y)`` to fixate, or one per video frame to move the eye between
frames.

A scene is never silently stretched. Encoding an image that states a ``fov``
without a ``vfmap`` raises, rather than producing a spatially wrong stimulus,
and passing a ``vfmap`` for an image that has no ``fov`` raises for the same
reason.

Spatial sampling preserves the source's color channels; today's encoders are
luminance encoders and reduce them to one number per electrode before
modulating.

Device constraints
------------------

An implant's :py:class:`~pulse2percept.implants.Raster` determines which
electrodes may pulse together; see :ref:`topics-rasters`.

Encoders can also quantize timing with ``clock`` and gray levels with
``n_levels``. These constraints are conservative: quantization may lower a
requested pulse rate, but never increases it.

Encoder amplitudes, frequencies, and durations follow the physical-unit
conventions in :ref:`topics-units`.
