.. _topics-stimulation:
.. _topics-stimuli:

===========
Stimulation
===========

pulse2percept accepts electrical pulse trains, images and videos, and
calibrated visual stimuli such as optotypes, gratings, and moving bars.

Visual input is converted to device stimulation by the implant. Electrical
stimulation can be supplied directly. In both cases, the delivered stimulation
is represented by a :py:class:`~pulse2percept.stimuli.Stimulus`.


Visual Stimuli
==============

Images and videos are represented by
:py:class:`~pulse2percept.stimuli.ImageStimulus` and
:py:class:`~pulse2percept.stimuli.VideoStimulus`.

They may be loaded from files or arrays, generated procedurally, or taken from
the sample stimuli bundled with pulse2percept.


Sample Images and Videos
------------------------

:mod:`pulse2percept.stimuli.samples` includes images and short videos for
examples, experiments, and testing:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    from pulse2percept.stimuli import ImageStimulus, samples

    def thumbnail(stim, shape=(120, 160), pad=False):
        """Center-crop (or pad) to the thumbnail's aspect ratio, then resize"""
        rows, cols = stim.img_shape[:2]
        aspect = shape[1] / shape[0]
        if pad:
            # Letterbox with black instead of cropping away the wordmark:
            trim = int(round(cols / aspect)) - rows
            img = stim.data.reshape(stim.img_shape)
            stim = ImageStimulus(np.pad(img, ((trim // 2, trim - trim // 2),
                                              (0, 0))))
        elif cols / rows > aspect:
            trim = cols - int(round(rows * aspect))
            stim = stim.crop(left=trim // 2, right=trim - trim // 2)
        else:
            trim = rows - int(round(cols / aspect))
            stim = stim.crop(top=trim // 2, bottom=trim - trim // 2)
        return stim.resize(shape)

    assets = [
        ('BVL logo', samples.logo_bvl(), False),
        ('UCSB bike path', samples.ucsb_bike(), False),
        ('UCSB coast', samples.ucsb_surf(), False),
        ('Cajal retina', samples.cajal_retina(), False),
        ('Zebrafish retina', samples.zebrafish_retina(), False),
        ('UCSB flyover (video)', next(iter(samples.ucsb_flyover())), False),
        ('UCSB pedestrians (video)', next(iter(samples.ucsb_pedestrians())),
         False),
        ('Big Buck Bunny (video)', next(iter(samples.big_buck_bunny())),
         False),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(12, 5))

    for ax, (title, stim, pad) in zip(axes.flat, assets):
        thumbnail(stim, pad=pad).plot(ax=ax)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()

The sample module currently includes:

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Loader
     - Type
     - Content
   * - ``logo_bvl()``
     - image
     - Bionic Vision Lab logo
   * - ``logo_ucsb()``
     - image
     - UC Santa Barbara logo
   * - ``bvl_cake()``
     - image
     - Bionic Vision Lab cake
   * - ``cajal_retina()``
     - image
     - Cajal's drawing of the retina
   * - ``zebrafish_retina()``
     - image
     - Fluorescence micrograph of zebrafish retina
   * - ``ucsb_bike()``
     - image
     - Cyclist, pedestrian, crosswalk, and stop sign
   * - ``ucsb_surf()``
     - image
     - UCSB coastline
   * - ``ucsb_flyover()``
     - video
     - Aerial sweep over UCSB and the coastline
   * - ``ucsb_pedestrians()``
     - video
     - Pedestrians walking on the UCSB campus
   * - ``big_buck_bunny()``
     - video
     - Short clip from *Big Buck Bunny*

The loaders return ordinary ``ImageStimulus`` or ``VideoStimulus`` objects, so
they can be resized, converted to grayscale, filtered, or passed directly to
the rest of pulse2percept:

.. code-block:: python

    from pulse2percept.stimuli import samples

    image = samples.ucsb_bike(as_gray=True).resize((60, 80))
    video = samples.ucsb_flyover(resize=(60, 80), as_gray=True)

See the :mod:`pulse2percept.stimuli.samples` API for details.
``pulse2percept/stimuli/data/samples/README.rst`` records provenance and
licensing for the bundled assets.


Psychophysical Stimuli
----------------------

:mod:`pulse2percept.stimuli.psychophysics` generates visual stimuli directly
in degrees of visual angle and physical time.

Current generators include Landolt Cs, Tumbling Es, sinusoidal gratings, and
moving bars:

.. plot::

    import matplotlib.pyplot as plt

    from pulse2percept.stimuli import psychophysics
    from pulse2percept.units import deg, dva

    stimuli = [
        ('Landolt C',
         psychophysics.landolt_c(
             gap=0.5 * dva,
             orientation=90 * deg,
             fov=15 * dva)),
        ('Tumbling E',
         psychophysics.tumbling_e(
             stroke=0.5 * dva,
             orientation=90 * deg,
             fov=15 * dva)),
        ('Grating',
         psychophysics.grating(
             spatial_freq=0.5 / dva,
             fov=15 * dva)),
        ('Bar',
         psychophysics.bar(
             width=2 * dva,
             offset=-3 * dva,
             fov=15 * dva)),
    ]

    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(12, 3))

    for ax, (title, scene) in zip(axes, stimuli):
        scene.plot(ax=ax)
        ax.set_title(title, fontsize=9)

    fig.tight_layout()

Each generator returns a :py:class:`~pulse2percept.vision.Scene` (also see
`Visual Scenes <topics-vision>`), so stimulus
geometry is independent of raster resolution.

.. code-block:: python

    import numpy as np

    from pulse2percept.stimuli import psychophysics
    from pulse2percept.units import Hz, deg, dva, s

    c = psychophysics.landolt_c(
        gap=0.5 * dva,
        position=(5, 0) * dva,
        orientation=90 * deg,
        fov=15 * dva,
    )

    e = psychophysics.tumbling_e(
        stroke=0.5 * dva,
        position=(5, 0) * dva,
        orientation=90 * deg,
        fov=15 * dva,
    )

    grating = psychophysics.grating(
        spatial_freq=0.5 / dva,
        temporal_freq=2 * Hz,
        fov=20 * dva,
        time=np.arange(0, 1000, 20),
    )

    bar = psychophysics.bar(
        width=2 * dva,
        speed=20 * dva / s,
        offset=-10 * dva,
        fov=20 * dva,
        time=np.arange(0, 1000, 20),
    )

.. list-table::
   :header-rows: 1
   :widths: 27 73

   * - Quantity
     - Convention
   * - Landolt C
     - ``gap`` sets the critical feature in dva
   * - Tumbling E
     - ``stroke`` sets the critical feature in dva
   * - ``spatial_freq``
     - cycles/dva
   * - ``temporal_freq``
     - Hz
   * - ``speed``
     - dva/s
   * - ``direction``
     - counterclockwise from the positive x axis
   * - ``time=None``
     - static stimulus
   * - ``time``
     - explicit sample times in ms for moving stimuli

Optotypes use standard proportions and are supersampled before being
area-averaged onto the requested raster. The critical feature must span at
least three output pixels.

Spatial and temporal frequencies above the corresponding Nyquist limit are
rejected rather than aliased.


Loading and Processing Images
-----------------------------

Custom images and videos can be loaded from files or NumPy arrays:

.. code-block:: python

    import pulse2percept as p2p

    image = p2p.stimuli.ImageStimulus('image.png')
    video = p2p.stimuli.VideoStimulus('movie.mp4')

Common image operations return a new stimulus:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Operation
     - Available for
   * - ``rgb2gray``
     - images and videos
   * - ``invert``
     - images and videos
   * - ``resize``
     - images and videos
   * - ``crop``
     - images and videos
   * - ``crop_square``
     - images and videos
   * - ``rotate``
     - images and videos
   * - ``filter``
     - images and videos; e.g. ``'sobel'``
   * - ``threshold``
     - images

For example:

.. code-block:: python

    image = (
        p2p.stimuli.samples.ucsb_bike(as_gray=True)
        .crop_square()
        .resize((60, 60))
        .filter('sobel')
    )

Images passed directly to an implant are device-relative: the image is mapped
across the electrode array.

To give an image a size and position in the visual field, place it in a
:py:class:`~pulse2percept.vision.Scene`; see :ref:`topics-vision`.

.. note::

   Pass a complete video to ``predict_percept`` when temporal dynamics across
   frames matter.

   Iterating over a ``VideoStimulus`` yields individual ``ImageStimulus``
   frames. Predicting those frames separately treats each one as an
   independent image.


.. _topics-encoders:

Encoders
========

An encoder converts visual input into stimulation that a device can deliver.

For an image or video, the basic path is::

    image / video
          |
          v
      preprocessing
          |
          v
        encoder
          |
          v
    electrode values
          |
          v
    raster scheduling
          |
          v
    delivered Stimulus

:py:class:`~pulse2percept.stimuli.ImplantEncoder` subclasses are attached to
an implant. Electrical encoders generate pulse trains; photovoltaic encoders
generate optical stimulation.

:py:class:`~pulse2percept.stimuli.TraceEncoder` instead uses a model to map a
trajectory in the visual field to a sequence of physical electrodes. It is
called directly through ``encode`` rather than attached to an implant.


Amplitude and Frequency Encoding
--------------------------------

Electrical image encoding can modulate either pulse amplitude or pulse
frequency:

.. list-table::
   :header-rows: 1
   :widths: 36 34 30

   * - Encoder
     - Image intensity controls
     - Held fixed
   * - :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`
     - pulse amplitude (``amp_range``)
     - ``freq``
   * - :py:class:`~pulse2percept.stimuli.FrequencyEncoder`
     - pulse frequency (``freq_range``)
     - ``amp``

For example:

.. code-block:: python

    implant = p2p.implants.retina.ArgusII()

    implant.encoder = p2p.stimuli.AmplitudeEncoder(
        amp_range=(0, 50),   # uA
        freq=20,             # Hz
    )

    source = p2p.stimuli.samples.ucsb_bike(
        resize=(60, 80),
        as_gray=True,
    )

    delivered = implant.prepare_stim(source)

A frequency encoder uses the same interface:

.. code-block:: python

    implant.encoder = p2p.stimuli.FrequencyEncoder(
        amp=50,              # uA
        freq_range=(0, 60),  # Hz
    )

Electrical ``Stimulus`` objects bypass the encoder.

For videos, pulse trains continue across frame boundaries. The video frame
times determine when the requested modulation changes. If the pulse period is
longer than a frame, some frames contain no pulse and ``prepare_stim`` issues
a warning.

An encoded ``Stimulus`` retains both the frame-level values and the delivered
pulse schedule. Spatial-only models use the frame-level representation;
temporal models use the pulse schedule.

Waveform samples are generated only when needed, so encoding a long video does
not allocate the complete electrical waveform up front.


Optical Encoding
----------------

.. versionadded:: 0.11.0

:py:class:`~pulse2percept.stimuli.PhotovoltaicEncoder` maps image intensity to
the ON duration of near-infrared pulses at fixed peak irradiance.

The resulting ``Stimulus`` is expressed in mW/mm^2.

Photovoltaic implants default to the optical protocol used by their
experimental system:

.. code-block:: python

    implant = p2p.implants.retina.PRIMAPivotal()

    stim = implant.prepare_stim(
        p2p.stimuli.samples.logo_bvl()
    )

    stim.unit    # mW/mm^2

A different protocol can be supplied explicitly:

.. code-block:: python

    implant.encoder = p2p.stimuli.PhotovoltaicEncoder(
        irradiance=4,      # mW/mm^2
        freq=40,           # Hz
        pulse_dur=4,       # ms
        wavelength=915,    # nm
    )

:py:class:`~pulse2percept.stimuli.PRIMAEncoder` uses the pivotal projector
settings: 30 Hz and 3.5 mW/mm^2, with ON duration quantized to 14 nonzero
steps of 0.7 ms from 0.7 to 9.8 ms.

The generic ``PhotovoltaicEncoder`` does not quantize ON duration.

.. note::

   **Grayscale mapping.** No source paper specifies the grayscale transfer
   function for natural images. pulse2percept therefore maps gray level
   linearly to ON duration. ``grayscale=False`` uses binary encoding.

   **Clinical processing.** Ambient-light adaptation, contrast enhancement,
   zoom, and contrast inversion are not applied automatically. They can be
   added through preprocessing.

   **Video timing.** Video frames are sampled on the projector clock using
   zero-order hold.

   **Spatial models.** Spatial-only models receive normalized optical drive.
   A value of 1 corresponds to a fully illuminated pixel at the encoder
   settings, or to the documented projector maximum for ``PRIMAEncoder``.


Electrical Waveforms
====================

Electrical stimulation can also be specified directly, without an image or
encoder.

Most examples use a
:py:class:`~pulse2percept.stimuli.BiphasicPulseTrain`:

.. code-block:: python

    pulse_train = p2p.stimuli.BiphasicPulseTrain(
        freq=20,          # Hz
        amp=50,           # uA
        phase_dur=0.45,   # ms
        stim_dur=500,     # ms
    )

    model = p2p.models.retina.ScoreboardModel(
        implant=implant
    )

    percept = model.predict_percept({
        'A5': pulse_train,
    })

The dictionary key selects the electrode. Electrodes omitted from the
dictionary receive no stimulation.

Bare numeric values use uA, ms, and Hz; see :ref:`topics-units`.


Pulse Classes
-------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Stimulus
     - Description
   * - :py:class:`~pulse2percept.stimuli.BiphasicPulse`
     - one symmetric biphasic pulse
   * - :py:class:`~pulse2percept.stimuli.BiphasicPulseTrain`
     - repeated biphasic pulses
   * - :py:class:`~pulse2percept.stimuli.MonophasicPulse`
     - one cathodic or anodic phase
   * - :py:class:`~pulse2percept.stimuli.AsymmetricBiphasicPulse`
     - biphasic pulse with unequal phases
   * - :py:class:`~pulse2percept.stimuli.AsymmetricBiphasicPulseTrain`
     - repeated asymmetric biphasic pulses
   * - :py:class:`~pulse2percept.stimuli.BiphasicTripletTrain`
     - repeated triplets of biphasic pulses
   * - :py:class:`~pulse2percept.stimuli.PulseTrain`
     - repetition of an arbitrary pulse

.. plot::

    import matplotlib.pyplot as plt

    from pulse2percept.stimuli import (AsymmetricBiphasicPulseTrain,
                                       BiphasicPulseTrain,
                                       BiphasicTripletTrain)
    from pulse2percept.units import Hz, ms, uA

    # High rates and short trains so individual phases stay visible
    trains = [
        ('BiphasicPulseTrain',
         BiphasicPulseTrain(100 * Hz, 20 * uA, 1 * ms, stim_dur=30 * ms)),
        ('AsymmetricBiphasicPulseTrain',
         AsymmetricBiphasicPulseTrain(100 * Hz, 20 * uA, 5 * uA, 1 * ms,
                                      4 * ms, stim_dur=30 * ms)),
        ('BiphasicTripletTrain',
         BiphasicTripletTrain(50 * Hz, 20 * uA, 1 * ms,
                              interpulse_dur=1 * ms, stim_dur=30 * ms)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3), sharey=True)

    for ax, (title, stim) in zip(axes, trains):
        stim.plot(ax=ax)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel('')
    axes[0].set_ylabel('Current (uA)')

    fig.tight_layout()

Pulses are cathodic-first by default.

Pulse trains contain only complete pulses. A pulse that would extend beyond
``stim_dur`` is omitted rather than truncated.


.. _topics-rasters:

Raster Scheduling
=================

Some devices stimulate electrodes in groups rather than simultaneously. A
:py:class:`~pulse2percept.implants.Raster` defines those groups and their
timing.

.. code-block:: python

    implant = p2p.implants.retina.ArgusII()

    implant.raster = p2p.implants.CheckerboardRaster(
        n_groups=5
    )

    implant.encoder = p2p.stimuli.AmplitudeEncoder(
        amp_range=(0, 50),
        freq=20,
    )

    delivered = implant.prepare_stim(
        p2p.stimuli.samples.ucsb_pedestrians(
            resize=(60, 80),
            as_gray=True,
        )
    )

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Raster
     - Grouping
   * - :py:class:`~pulse2percept.implants.SequentialRaster`
     - consecutive electrodes, or interleaved electrodes with
       ``interleave=True``
   * - :py:class:`~pulse2percept.implants.CheckerboardRaster`
     - spatially distributed grid positions
   * - :py:class:`~pulse2percept.implants.CustomRaster`
     - user-defined groups; every electrode belongs to exactly one group

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    from pulse2percept.implants import (CheckerboardRaster, CustomRaster,
                                        ElectrodeGrid, Implant,
                                        SequentialRaster)

    implant = Implant(ElectrodeGrid((6, 6), 400))  # 400 um spacing
    names = implant.electrode_names
    # Four equal groups in random order:
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(names)

    rasters = [
        ('SequentialRaster(6)', SequentialRaster(6)),
        ('CheckerboardRaster(4)', CheckerboardRaster(4)),
        ('CustomRaster (random)', CustomRaster(np.split(shuffled, 4))),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    for ax, (title, raster) in zip(axes, rasters):
        implant.raster = raster
        raster.plot(ax=ax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    fig.tight_layout()

``raster.plot()`` displays the grouping. ``raster.members(...)`` returns the
electrodes in a group.

Groups are activated in order and separated by ``group_dur`` milliseconds.
With ``group_dur=None``, they are distributed across the pulse period. Each
slot must fit its pulse, and the complete raster sweep must fit within the
pulse period.

Amplitude and frequency encoding interact with rastering differently:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Encoder
     - Raster behavior
   * - Amplitude encoding
     - All electrodes share one pulse period; rastering offsets the groups
       within it
   * - Frequency encoding
     - Each electrode period is rounded upward to a whole number of raster
       sweeps, so delivered frequency may be lower than requested but never
       higher

Leave ``implant.raster`` unset when raster scheduling is not part of the
device.

PRIMA has no raster; all pixels may be illuminated simultaneously.


Device Constraints
==================

``prepare_stim`` also applies device-specific constraints where available:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Constraint
     - Behavior
   * - ``clock``
     - quantizes supported timing
   * - ``n_levels``
     - quantizes encoded intensity levels
   * - ``thresholds``
     - converts amplitudes expressed in ``xTh`` to uA
   * - ``safe_mode``
     - applies device-specific safety checks

Timing quantization may reduce a requested pulse rate but never increase it.

Amplitudes in ``xTh`` are converted using ``implant.thresholds``; see
:ref:`topics-units`.

For the PRIMA projector envelope enforced by ``safe_mode``, see
:ref:`topics-implants`.


Delivered Stimulation
======================

All paths eventually produce a
:py:class:`~pulse2percept.stimuli.Stimulus`::

    source
      |
      v
    implant.prepare_stim()
      |
      +-- preprocessing
      +-- encoding
      +-- raster scheduling
      +-- threshold conversion
      +-- safety checks
      |
      v
    Stimulus

Models call :py:meth:`~pulse2percept.implants.Implant.prepare_stim`
automatically. Call it directly when you want to inspect the result:

.. code-block:: python

    source = p2p.stimuli.samples.ucsb_bike(
        resize=(60, 80),
        as_gray=True,
    )

    delivered = implant.prepare_stim(source)

    delivered.plot()
    implant.plot(stim=source, stim_cmap=True)


The Stimulus Container
----------------------

A :py:class:`~pulse2percept.stimuli.Stimulus` stores labeled stimulation over
time:

.. list-table::
   :header-rows: 1
   :widths: 24 46 30

   * - Attribute
     - Meaning
     - Shape / units
   * - ``data``
     - stimulation values
     - ``(n_electrodes, n_times)``
   * - ``electrodes``
     - row labels
     - electrode names or indices
   * - ``time``
     - physical time axis
     - ``time_unit``; ms by default
   * - ``time_unit``
     - unit used by ``time``
     - ms by default

``time`` is ``None`` for a timeless stimulus.

A ``Stimulus`` can also be created directly:

.. code-block:: python

    stim = p2p.stimuli.Stimulus({
        'A1': 10,
        'A2': 20,
        'A3': 30,
    })

    stim['A1']
    stim['A1', 10]    # value at t=10 ms; interpolated if needed

Use ``stim.data`` for ordinary NumPy indexing.


Stimulus Operations
-------------------

Most operations return a new object:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Operation
     - Behavior
   * - ``stim.plot()``
     - heatmap for many electrodes or traces for selected electrodes
   * - ``stim.shift(dt)``
     - shift stimulation in time
   * - ``stim >> dt``
     - shorthand for ``shift``
   * - ``stim.pad(...)``
     - extend with zeros
   * - ``stim.compress()``
     - compress redundant samples; modifies in place
   * - ``stim.remove(...)``
     - remove electrodes or samples; modifies in place

Pulse classes retain parameters such as ``freq`` and ``amp`` and generate
waveform samples only when their ``data`` are needed.

Arithmetic preserves the pulse class when the result is still that waveform:

.. code-block:: python

    pt * 2      # still a pulse train
    pt + 5      # generic Stimulus
