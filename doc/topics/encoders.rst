.. _topics-encoders:

=================
Stimulus Encoders
=================

.. versionadded:: 0.10.0

A visual prosthesis does not stimulate with pixels. It stimulates with
electrical pulses. An :py:class:`~pulse2percept.stimuli.Encoder` defines how
the gray levels of an image or video are turned into those pulses:

::

    image / video -> Encoder -> electrical Stimulus -> implant -> model -> Percept

The encoder is therefore **not a perceptual model**. It produces the electrical
:py:class:`~pulse2percept.stimuli.Stimulus` that a model receives through an
implant.

The usual workflow
------------------

For example, an :py:class:`~pulse2percept.stimuli.AmplitudeEncoder` maps image
brightness onto pulse amplitude while keeping pulse frequency fixed:

.. code-block:: python

    import pulse2percept as p2p
    from pulse2percept.units import uA, Hz

    implant = p2p.implants.ArgusII()
    encoder = p2p.stimuli.AmplitudeEncoder(
        implant, amp_range=(0, 50 * uA), freq=20 * Hz
    )

    source = p2p.stimuli.BostonTrain()
    implant.stim = encoder.encode(source)

    model = p2p.models.ScoreboardModel().build()
    percept = model.predict_percept(implant)

Here a gray level of 0 maps to 0 uA, a gray level of 255 maps to 50 uA, and every
electrode is driven at 20 Hz. The returned stimulus contains the actual
biphasic pulse trains, ready to assign to ``implant.stim``.

Passing the implant matters
---------------------------

When an implant is supplied to the encoder, the image or video is sampled at
the implant's electrode locations before pulse trains are constructed. This is
usually what you want:

.. code-block:: python

    encoder = p2p.stimuli.AmplitudeEncoder(implant)
    stim = encoder.encode(source)

The resulting stimulus has one row per electrode, with electrode names that
match the implant. If ``implant=None``, every source pixel is treated as its
own electrode instead. That can be useful for custom workflows, but can also
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

    encoder = p2p.stimuli.FrequencyEncoder(
        implant, amp=50 * uA, freq_range=(0, 100 * Hz)
    )
    implant.stim = encoder.encode(source)

A gray level of 0 then maps to 0 Hz and a gray level of 255 to 100 Hz, with pulse
amplitude fixed at 50 uA.

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

Hardware constraints, when you need them
----------------------------------------

The defaults describe the requested stimulation without imposing a particular
stimulator. Optional parameters let you add hardware constraints later:

``clock``
    Quantizes pulse timing to the stimulator's time base.

``n_levels``
    Quantizes input gray levels before they are mapped onto stimulation.

``raster``
    Describes which groups of electrodes may stimulate at the same time. If
    omitted, the implant's own raster is used when it has one.

These constraints never make an electrode pulse faster than requested. Timing
is rounded conservatively so that an unrealizable pulse period becomes a
slower realizable one rather than a faster one.

Physical units
--------------

Encoder parameters accept both the usual bare numbers and unitful quantities:

.. code-block:: python

    import pulse2percept.units as u

    encoder = p2p.stimuli.AmplitudeEncoder(
        implant,
        amp_range=(0 * u.uA, 0.05 * u.mA),
        freq=20 * u.Hz,
        phase_dur=460 * u.us,
    )

The encoded stimulus itself uses pulse2percept's canonical stimulus units:
current in microamps and time in milliseconds. See :ref:`topics-units` for the
full units convention.

.. seealso::

    * :py:class:`pulse2percept.stimuli.Encoder` for the common encoder options
    * :py:class:`pulse2percept.stimuli.AmplitudeEncoder`
    * :py:class:`pulse2percept.stimuli.FrequencyEncoder`
    * :ref:`Electrical Stimuli <topics-stimuli>`
    * :ref:`Physical Units <topics-units>`
    * :ref:`Computational Models <topics-models>`
