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

    implant = p2p.implants.retina.ArgusII()
    model = p2p.models.retina.ScoreboardModel(implant=implant)

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

Sample stimuli
--------------

:py:mod:`pulse2percept.stimuli.samples` bundles a few ready-made stimuli for
demos, docs, and tests. They are ordinary ``ImageStimulus`` or
``VideoStimulus`` objects (or, for procedural optotypes, a
:py:class:`~pulse2percept.vision.Scene`), and are reached through the module
rather than the top-level namespace:

.. code-block:: python

    from pulse2percept.stimuli import samples

    logo = samples.logo_bvl()
    logo = samples.logo_ucsb()
    cake = samples.bvl_cake()
    surf = samples.ucsb_surf()

The two photographs are naturalistic RGB stills: a cake decorated with the lab
logo (495x435), and a frame of the UCSB coastline (476x845) from the National
Library of Medicine video *Towards a Smart Bionic Eye*. See
``pulse2percept/stimuli/data/samples/README.rst`` for the licensing of each
bundled asset; it differs from file to file.

:py:func:`~pulse2percept.stimuli.samples.big_buck_bunny` is the bundled
naturalistic video, a 115-frame excerpt of *Big Buck Bunny* at 24 fps:

.. code-block:: python

    video = samples.big_buck_bunny()
    video.play()

At full resolution it is 359x640 RGB, i.e. 689,280 electrodes; pass
``resize`` and/or ``as_gray=True`` before handing it to a model. The clip is
© 2008 Blender Foundation and licensed CC BY 3.0, not under pulse2percept's
BSD license; its attribution rides along in ``video.metadata``.

:py:func:`~pulse2percept.stimuli.samples.landolt_c` draws a Landolt C at
standard proportions (stroke width and inner/outer diameters of 1, 3, and 5
gaps) and places it in the visual field:

.. code-block:: python

    from pulse2percept.units import deg, dva

    scene = samples.landolt_c(gap=0.5 * dva, position=(5, 0) * dva,
                              orientation=90 * deg, fov=15 * dva)

``gap`` is the angular size of the critical feature, which is what an acuity
task varies; ``position`` sets where the optotype sits in the visual field,
and therefore its eccentricity, without changing that size. ``orientation``
says where the opening points (0 right, 90 up, 180 left, 270 down), and
``polarity`` chooses a black C on white (``'dark'``) or the reverse.

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
