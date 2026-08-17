.. _topics-stimuli:

==================
Electrical Stimuli
==================

A :py:class:`~pulse2percept.stimuli.Stimulus` describes what is delivered to
one or more electrodes. For electrical stimulation, its rows are electrodes,
its columns are points in time, and its values are current amplitudes:

::

    electrical Stimulus -> implant -> model -> Percept

Most users do not need to build this array by hand. pulse2percept provides
pulse and pulse-train classes for common stimulation waveforms, and
:ref:`stimulus encoders <topics-encoders>` for turning images and videos into
multi-electrode stimulation.

The usual workflow
------------------

A :py:class:`~pulse2percept.stimuli.BiphasicPulseTrain` is a good place to
start:

.. code-block:: python

    import pulse2percept as p2p
    from pulse2percept.units import Hz, ms, uA

    implant = p2p.implants.ArgusII()

    pulse_train = p2p.stimuli.BiphasicPulseTrain(
        freq=20 * Hz,
        amp=50 * uA,
        phase_dur=0.45 * ms,
        stim_dur=500 * ms,
    )

    implant.stim = {'A5': pulse_train}

    model = p2p.models.AxonMapModel().build()
    percept = model.predict_percept(implant)

The dictionary key identifies the electrode. Electrodes not listed receive no
stimulation.

Pulses and pulse trains
-----------------------

pulse2percept provides a small set of common electrical waveforms:

.. list-table::
   :header-rows: 1

   * - Stimulus
     - Description
   * - :py:class:`~pulse2percept.stimuli.BiphasicPulse`
     - One symmetric, charge-balanced biphasic pulse.
   * - :py:class:`~pulse2percept.stimuli.BiphasicPulseTrain`
     - Repeated symmetric biphasic pulses. The usual starting point.
   * - :py:class:`~pulse2percept.stimuli.MonophasicPulse`
     - One cathodic or anodic phase; generally not charge-balanced.
   * - :py:class:`~pulse2percept.stimuli.AsymmetricBiphasicPulse`
     - A biphasic pulse whose two phases may differ in amplitude or duration.
   * - :py:class:`~pulse2percept.stimuli.AsymmetricBiphasicPulseTrain`
     - A train of asymmetric biphasic pulses.
   * - :py:class:`~pulse2percept.stimuli.BiphasicTripletTrain`
     - Repeated triplets of biphasic pulses.
   * - :py:class:`~pulse2percept.stimuli.PulseTrain`
     - Repeat an arbitrary single-pulse Stimulus.

For example, a single cathodic-first biphasic pulse is:

.. code-block:: python

    pulse = p2p.stimuli.BiphasicPulse(
        amp=50 * uA,
        phase_dur=0.45 * ms,
        interphase_dur=0.1 * ms,
    )

The amplitude specifies the magnitude; ``cathodic_first=True`` by default
determines the sign of the first phase.

A generic :py:class:`~pulse2percept.stimuli.PulseTrain` can repeat any
single-pulse stimulus:

.. code-block:: python

    train = p2p.stimuli.PulseTrain(
        freq=20 * Hz,
        pulse=pulse,
        stim_dur=500 * ms,
    )

Only whole pulses are delivered. If the final pulse would extend beyond
``stim_dur``, it is omitted rather than cut in half.

Stimulating multiple electrodes
-------------------------------

The easiest way to specify different stimulation on different electrodes is a
dictionary:

.. code-block:: python

    implant.stim = {
        'A5': p2p.stimuli.BiphasicPulseTrain(
            20 * Hz, 50 * uA, 0.45 * ms, stim_dur=500 * ms
        ),
        'B5': p2p.stimuli.BiphasicPulseTrain(
            20 * Hz, 25 * uA, 0.45 * ms, stim_dur=500 * ms
        ),
    }

pulse2percept combines the individual waveforms into one
:py:class:`~pulse2percept.stimuli.Stimulus`, merging their time axes as
needed.

You can also construct a Stimulus directly from scalar values, arrays, lists,
or dictionaries:

.. code-block:: python

    stim = p2p.stimuli.Stimulus({
        'A1': 10 * uA,
        'A2': 20 * uA,
        'A3': 30 * uA,
    })

A one-dimensional sequence means one value per electrode and has no time
component. A two-dimensional array has shape ``(n_electrodes, n_times)``.

The Stimulus container
----------------------

Every Stimulus exposes the same basic pieces:

``stim.data``
    A 2D NumPy array with shape ``(n_electrodes, n_times)``.

``stim.electrodes``
    The electrode names corresponding to the rows.

``stim.time``
    The time axis, or ``None`` for a stimulus without a time component.

For a time-varying stimulus, the easiest way to see what is being delivered is
often:

.. code-block:: python

    stim.plot()

Stimuli can also be indexed by electrode name and time:

.. code-block:: python

    stim['A5']
    stim['A5', 10 * ms]

The second index is a **time**, not a column number. If that exact time is not
stored, pulse2percept interpolates the waveform there. Use ``stim.data`` when
you want ordinary NumPy indexing by row and column.

The time axis does not need to be uniformly sampled. Pulse classes store the
important transition points rather than a dense sample at every simulation
step, which keeps long pulse trains compact.

Images and videos are different
-------------------------------

:py:class:`~pulse2percept.stimuli.ImageStimulus` and
:py:class:`~pulse2percept.stimuli.VideoStimulus` are also Stimulus objects, but
their values are **dimensionless gray levels, not electrical current**.

For example:

.. code-block:: python

    source = p2p.stimuli.BostonTrain()

That object is a visual source. It should be passed through an
:py:class:`~pulse2percept.stimuli.Encoder` before it is used as electrical
stimulation:

.. code-block:: python

    encoder = p2p.stimuli.AmplitudeEncoder(
        implant,
        amp_range=(0, 50 * uA),
        freq=20 * Hz,
    )

    implant.stim = encoder.encode(source)

This distinction matters. A gray level of 0.5 is not 0.5 uA; the encoder is
what defines how image intensity maps onto stimulation. See
:ref:`topics-encoders` for the full workflow.

Physical units
--------------

Electrical stimulus amplitudes are stored in microamps and time in
milliseconds. Pulse-train frequency is measured in hertz. You can use those
canonical units directly or pass compatible quantities:

.. code-block:: python

    from pulse2percept.units import mA, us

    pulse = p2p.stimuli.BiphasicPulse(
        amp=0.05 * mA,
        phase_dur=450 * us,
    )

This is equivalent to ``50 * uA`` and ``0.45 * ms``. Bare numbers continue to
mean the documented canonical units. See :ref:`topics-units` for the full
convention.

Charge balance and safety
-------------------------

For implanted stimulation, charge balance matters. Symmetric
:py:class:`~pulse2percept.stimuli.BiphasicPulse` and
:py:class:`~pulse2percept.stimuli.BiphasicPulseTrain` objects are
charge-balanced by construction.

The :py:class:`~pulse2percept.stimuli.Stimulus` API also exposes charge-balance
information for arbitrary waveforms, and implants can apply additional safety
checks when ``safe_mode=True``. These checks are useful guardrails, but they
are not a substitute for the safety limits of a particular experimental or
clinical device.

.. seealso::

    * :py:class:`pulse2percept.stimuli.Stimulus`
    * :py:class:`pulse2percept.stimuli.BiphasicPulse`
    * :py:class:`pulse2percept.stimuli.BiphasicPulseTrain`
    * :ref:`Stimulus Encoders <topics-encoders>`
    * :ref:`Visual Prostheses <topics-implants>`
    * :ref:`Raster Strategies <topics-rasters>`
    * :ref:`Physical Units <topics-units>`
    * :ref:`Computational Models <topics-models>`
