.. _topics-stimuli:

==================
Electrical Stimuli
==================

A :py:class:`~pulse2percept.stimuli.Stimulus` is labeled two-dimensional data:
rows are electrodes and columns are points in time. Electrical stimuli contain
current amplitudes; images and videos contain dimensionless visual intensity.

Electrical waveforms
--------------------

For most electrical stimulation, start with a
:py:class:`~pulse2percept.stimuli.BiphasicPulseTrain`:

.. code-block:: python

    import pulse2percept as p2p

    pulse_train = p2p.stimuli.BiphasicPulseTrain(
        freq=20,
        amp=50,
        phase_dur=0.45,
        stim_dur=500,
    )

    implant = p2p.implants.ArgusII()
    model = p2p.models.ScoreboardModel(implant=implant)

    percept = model.predict_percept({'A5': pulse_train})

The dictionary key selects the electrode; unlisted electrodes receive no
stimulation. A stimulus is trial input, not implant state: the implant turns it
into the current its electrodes deliver
(``implant.prepare_stim({'A5': pulse_train})``) and keeps nothing.

Common waveform classes include:

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

The Stimulus container
----------------------

Every Stimulus exposes:

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

Structured and read-only stimuli
--------------------------------

.. versionchanged:: 0.10.0

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

Images and videos
-----------------

:py:class:`~pulse2percept.stimuli.ImageStimulus` and
:py:class:`~pulse2percept.stimuli.VideoStimulus` are visual sources, not
currents. Their values are dimensionless gray levels. A
:py:class:`~pulse2percept.stimuli.StimulusEncoder` defines how those gray
levels become electrical stimulation; see :ref:`topics-encoders`.

An image is *device-relative*: its pixels are stretched across the implant's
electrodes, and the picture means nothing beyond "this is what the device was
shown". To place a picture in the visual field instead -- so that each
electrode sees the part of it that electrode actually looks at -- wrap it in a
:py:class:`~pulse2percept.vision.Scene` and give that to a model; see
:ref:`topics-models-scene`.

Plotting and time operations
----------------------------

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

Electrical amplitudes, time, and frequency use the unit conventions described
in :ref:`topics-units`.
