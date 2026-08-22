.. _topics-encoders:

=================
Stimulus Encoders
=================

.. versionadded:: 0.10.0

A visual prosthesis does not stimulate with pixels. It stimulates with
electrical pulses. An :py:class:`~pulse2percept.stimuli.StimulusEncoder` defines how
the gray levels of an image or video are turned into those pulses:

::

    image / video -> Encoder -> electrical Stimulus -> implant -> model -> Percept

The encoder is therefore **not a perceptual model**. It produces the electrical
:py:class:`~pulse2percept.stimuli.Stimulus` that a model receives through an
implant.

The usual workflow
------------------

For example, an :py:class:`~pulse2percept.stimuli.AmplitudeEncoder` maps image
brightness onto pulse amplitude while keeping pulse frequency fixed. Give the
implant an encoder, and assigning an image or a video encodes it:

.. code-block:: python

    import pulse2percept as p2p
    from pulse2percept.units import uA, Hz

    implant = p2p.implants.ArgusII()
    implant.encoder = p2p.stimuli.AmplitudeEncoder(
        amp_range=(0, 50 * uA),
        freq=20 * Hz,
    )

    source = p2p.stimuli.BostonTrain()
    implant.stim = source

    model = p2p.models.ScoreboardModel().build()
    percept = model.predict_percept(implant)

Here a gray level of 0 maps to 0 uA, a gray level of 1 maps to 50 uA, and every
electrode is driven at 20 Hz. ``implant.stim`` is the actual biphasic pulse
trains that result.

Encoding explicitly
-------------------

You can also encode a source yourself and assign the result:

.. code-block:: python

    encoder = p2p.stimuli.AmplitudeEncoder(
        amp_range=(0, 50 * uA),
        freq=20 * Hz,
    )
    stim = encoder.encode(source, implant=implant)
    implant.stim = stim

Passing ``implant`` samples the image or video at the implant's electrode
locations before pulse trains are constructed. This is usually what you want:
the resulting stimulus has one row per electrode, with electrode names that
match the implant. Leave it out and every source pixel is treated as its own
electrode instead, which can be useful for custom workflows but can also
produce unnecessarily large stimuli.

Amplitude or frequency?
-----------------------

pulse2percept currently provides two basic encoders:

.. list-table::
   :header-rows: 1

   * - Encoder
     - Gray level controls
   * - :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`
     - Pulse amplitude; frequency is fixed.
   * - :py:class:`~pulse2percept.stimuli.FrequencyEncoder`
     - Pulse frequency; amplitude is fixed.

Amplitude encoding is the simplest place to start. Frequency encoding works the
same way from the user's point of view:

.. code-block:: python

    implant.encoder = p2p.stimuli.FrequencyEncoder(
        amp=50 * uA,
        freq_range=(0, 60 * Hz),
    )
    implant.stim = source

A gray level of 0 then maps to 0 Hz and a gray level of 1 to 60 Hz, with pulse
amplitude fixed at 50 uA. How high the range can go depends on the implant's
raster: a sweep through its groups has to fit inside the shortest pulse period
requested. See :ref:`topics-rasters`.

Frames and pulses have separate clocks
--------------------------------------

For video, a new frame changes the gray level seen by each electrode and hence
the amplitude or frequency being requested. It does **not** restart the pulse
train. Pulses run on their own continuous clock, so video frame rate and pulse
frequency are independent quantities.

This means, for example, that a 30 fps video can be encoded at 20 Hz or 100 Hz.
If the pulse rate is lower than the frame rate, some frames may never coincide
with a pulse and therefore contribute no stimulation; the encoder warns when
that happens.

What the encoded stimulus is
----------------------------

:py:meth:`~pulse2percept.stimuli.StimulusEncoder.encode` returns the **delivered
electrical stimulus**: the biphasic pulse trains the device would actually
emit, on the device's own timing. Its waveform is generated only when
something asks for samples, so encoding a video onto a large implant is cheap
until a model needs the pulses.

The stimulus also remembers what was *asked* of each electrode, which is what
lets each kind of model read the half it can express:

* A purely spatial model (:py:class:`~pulse2percept.models.ScoreboardModel`,
  :py:class:`~pulse2percept.models.AxonMapModel`) has no clock, so it reads the
  frame-level modulation: one amplitude per electrode per frame of the source.
  It never sees the pulse timing, and never generates the waveform.
* A temporal or combined model integrates the delivered pulses, because that
  is what such a model is for.

This does not depend on how the stimulus got to the implant. Encoding by hand
and assigning the result is the same thing as letting the implant encode:

.. code-block:: python

    implant.stim = encoder.encode(source, implant=implant)   # these two

    implant.encoder = encoder                                # ... are the same
    implant.stim = source

.. versionchanged:: 0.10.0
    A spatial model reads the frame-level modulation of any encoded stimulus.
    Previously only stimuli encoded during assignment to an implant were read
    that way, and a hand-encoded one came out as a sequence of raster slots.

Hardware constraints, when you need them
----------------------------------------

The defaults describe the requested stimulation without imposing a particular
stimulator. Optional parameters let you add hardware constraints later:

``clock``
    Quantizes pulse timing to the stimulator's time base.

``n_levels``
    Quantizes input gray levels before they are mapped onto stimulation.

The raster -- which groups of electrodes may stimulate at the same time -- is
a property of the device rather than of the encoder, so it is set on the
implant (``implant.raster``) and picked up from there. See
:ref:`topics-rasters`.

These constraints never make an electrode pulse faster than requested. Timing
is rounded conservatively so that an unrealizable pulse period becomes a
slower realizable one rather than a faster one.

Physical units
--------------

Encoder parameters accept both the usual bare numbers and unitful quantities:

.. code-block:: python

    import pulse2percept.units as u

    encoder = p2p.stimuli.AmplitudeEncoder(
        amp_range=(0 * u.uA, 0.05 * u.mA),
        freq=20 * u.Hz,
        phase_dur=460 * u.us,
    )

The encoded stimulus itself uses pulse2percept's canonical stimulus units:
current in microamps and time in milliseconds. See :ref:`topics-units` for the
full units convention.

.. seealso::

    * :py:class:`pulse2percept.stimuli.StimulusEncoder` for the common encoder options
    * :py:class:`pulse2percept.stimuli.AmplitudeEncoder`
    * :py:class:`pulse2percept.stimuli.FrequencyEncoder`
    * :ref:`Electrical Stimuli <topics-stimuli>`
    * :ref:`Physical Units <topics-units>`
    * :ref:`Computational Models <topics-models>`
