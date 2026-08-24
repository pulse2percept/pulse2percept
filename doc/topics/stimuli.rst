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

    implant = p2p.implants.ArgusII(stim={'A5': pulse_train})

The dictionary key selects the electrode; unlisted electrodes receive no
stimulation.

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

An image or video can also state the field of view it subtends, which places
its pixels in the visual field rather than merely in a pixel grid:

.. code-block:: python

    from pulse2percept.stimuli import ImageStimulus
    from pulse2percept.units import dva

    scene = ImageStimulus('scene.png', resize=(200, 400), fov=30 * dva)
    scene.fov                      # (30.0, 15.0) degrees, width x height
    scene.pixel_to_dva(0, 0)       # (-14.9625, 7.4625): top-left pixel center
    scene.dva_to_pixel(0, 0)       # (199.5, 99.5): the center of the image

``fov`` is the *outer* extent of the image, centered on it, and a scalar gives
the horizontal FOV with square angular pixels. Pixel coordinates address pixel
centers, so they lie half an angular pixel inside that extent, and row 0 sits
at positive ``y``.

A :py:meth:`~pulse2percept.stimuli.ImageStimulus.resize` keeps the field of
view and resamples the pixels, while a
:py:meth:`~pulse2percept.stimuli.ImageStimulus.crop` keeps the angular pixel
size and shrinks the field of view. A cropped stimulus is centered on its new
frame; it does not retain an offset relative to the original image.

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
