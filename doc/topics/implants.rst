.. _topics-implants:

=================
Visual Prostheses
=================

An implant describes the device: its electrodes, their geometry, the tissue it
sits on, and the device-specific rules that turn a source into the stimulation
those electrodes deliver.

An implant holds no stimulus. What it delivers is derived from a source, on
demand, by :py:meth:`~pulse2percept.implants.Implant.prepare_stim`; see
:ref:`topics-stimulation`. Where the device sits in tissue is not device
geometry either: electrode coordinates are device-local, and
``implant_position``, ``implant_rotation`` and ``implant_depth`` are model
parameters (see :ref:`topics-coordinates`).

Generic device machinery lives at the root of
:py:mod:`pulse2percept.implants`; devices live under the anatomical target
they stimulate, which is also where laterality lives:

.. code-block:: text

    Electrode / ElectrodeArray
            |
    generic Implant pipeline        (p2p.implants)
            |
    retina.RetinalImplant  -> eye          ('left', 'right')
    cortex.CorticalImplant -> hemisphere   ('left', 'right', None)

So a device is constructed from its own namespace,
``p2p.implants.retina.ArgusII()`` or ``p2p.implants.cortex.Orion()``, while
electrodes, arrays, rasters and
:py:class:`~pulse2percept.implants.EnsembleImplant` stay at the root.

All implants derive from :py:class:`~pulse2percept.implants.Implant`. The
attributes used most often are:

``electrode_array``
    The :py:class:`~pulse2percept.implants.ElectrodeArray`.

``placement``
    Where the device sits relative to the tissue it stimulates
    (``'epiretinal'``, ``'subretinal'``, ``'suprachoroidal'``,
    ``'epicortical'``, ``'intracortical'``), or ``None`` for a generic array.

``technology``
    Stimulation technology, such as ``'photovoltaic'``, where specified.

``family``
    Named device family, where applicable.

``scene_input_frame``
    Whether gaze moves the device's input: ``'eye'`` if input passes through
    the eye's optics, ``'head'`` for systems driven by a head-fixed camera.
    See :ref:`topics-vision`.

``encoder`` and ``raster``
    Optional device behavior used when visual input is converted to electrical
    stimulation; see :ref:`topics-stimulation`.

Basic use
---------

Electrodes can be accessed by name or index, and the array can be plotted
directly:

.. code-block:: python

    import pulse2percept as p2p

    implant = p2p.implants.retina.ArgusII()

    implant['A8']
    implant[0]
    len(implant)
    implant.electrode_names
    implant.electrode_array.coordinates()
    implant.plot()

Retinal implants
----------------

Retinal implants derive from
:py:class:`~pulse2percept.implants.retina.RetinalImplant` and use device-local
electrode coordinates in microns.

``eye`` (``'left'`` or ``'right'``, default ``'right'``) records the implanted
eye and is read by the models:
:py:class:`~pulse2percept.models.retina.AxonMapModel` puts the optic disc on
the side the eye calls for. Some devices also reverse their column names in
the left eye; see each class's API documentation.

.. list-table::
   :header-rows: 1
   :widths: 22 14 10 26 28

   * - Object
     - Placement
     - Electrodes
     - Device
     - Typical model
   * - :py:class:`~pulse2percept.implants.retina.ArgusI`
     - epiretinal
     - 16
     - Argus I, 4 x 4 array
     - ``AxonMapModel``, ``Nanduri2012Model``
   * - :py:class:`~pulse2percept.implants.retina.ArgusII`
     - epiretinal
     - 60
     - Argus II, 6 x 10 array
     - ``AxonMapModel``, ``BiphasicAxonMapModel``
   * - :py:class:`~pulse2percept.implants.retina.IMIE`
     - epiretinal
     - 256
     - IMIE array
     - ``AxonMapModel``
   * - :py:class:`~pulse2percept.implants.retina.AlphaIMS`
     - subretinal
     - 1500
     - Alpha IMS microphotodiode array
     - ``ScoreboardModel``
   * - :py:class:`~pulse2percept.implants.retina.AlphaAMS`
     - subretinal
     - 1600
     - Alpha AMS microphotodiode array
     - ``ScoreboardModel``
   * - :py:class:`~pulse2percept.implants.retina.BVT24`
     - suprachoroidal
     - 35
     - First-generation suprachoroidal array
     - ``ScoreboardModel``
   * - :py:class:`~pulse2percept.implants.retina.BVT44`
     - suprachoroidal
     - 46
     - Second-generation suprachoroidal array
     - ``ScoreboardModel``
   * - :py:class:`~pulse2percept.implants.retina.PRIMAPivotal`
     - subretinal
     - 378
     - PRIMA photovoltaic array [Holz2026]_
     - ``Ho2018Model``, ``ScoreboardModel``

``BVT24`` and ``BVT44`` are pulse2percept identifiers rather than official
product names. These classes are research-software representations based on
published device descriptions, not manufacturer-validated simulators; see each
class's API documentation for device-specific geometry and assumptions.

Electrode coordinates are device-local microns, on very different physical
scales:

.. plot::

    import matplotlib.pyplot as plt
    import pulse2percept as p2p

    devices = [
        ('ArgusII()', p2p.implants.retina.ArgusII()),
        ('AlphaAMS()', p2p.implants.retina.AlphaAMS()),
        ('BVT44()', p2p.implants.retina.BVT44()),
        ('IMIE()', p2p.implants.retina.IMIE()),
        ('PRIMAPivotal()', p2p.implants.retina.PRIMAPivotal()),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for ax, (title, implant) in zip(axes.flat, devices):
        implant.plot(ax=ax)
        ax.set_title(title, fontsize=9)
        ax.set_aspect('equal')
        ax.set_xlabel('x (um)', fontsize=8)
        ax.set_ylabel('y (um)', fontsize=8)
        ax.tick_params(labelsize=7)
    axes.flat[-1].axis('off')
    fig.tight_layout()

Argus
^^^^^

Argus II includes device-specific defaults for converting visual input to
stimulation. Images and videos are encoded with an
:py:class:`~pulse2percept.stimuli.AmplitudeEncoder` at 6 Hz, and stimulation is
rastered one row at a time using a
:py:class:`~pulse2percept.implants.SequentialRaster` with six groups separated
by 2 ms. Visual stimuli can therefore be passed directly to
:py:meth:`~pulse2percept.implants.Implant.prepare_stim`:

.. code-block:: python

    implant = p2p.implants.retina.ArgusII()
    stim = implant.prepare_stim(image)

Both defaults can be overridden. Passing ``encoder=None`` disables automatic
image/video encoding; passing ``raster=None`` drives electrodes without the
default sequential raster:

.. code-block:: python

    implant = p2p.implants.retina.ArgusII(encoder=None, raster=None)

Photovoltaic arrays
^^^^^^^^^^^^^^^^^^^

PRIMA is a subretinal photovoltaic prosthesis developed at Stanford. Pixium
Vision developed the clinical system; Science Corporation acquired Pixium's
PRIMA assets and intellectual property in 2024.

:py:class:`~pulse2percept.implants.retina.PRIMAPivotal` models the 378-pixel
device used in the pivotal PRIMAvera trial [Holz2026]_. The same 100 um
configuration was used in the earlier first-in-human study [Palanker2020]_.
For a hexagonal array, the row spacing is ``spacing * sqrt(3) / 2``.

pulse2percept also includes several photovoltaic research arrays described in
the literature: :py:class:`~pulse2percept.implants.retina.Lorach2015Array`,
:py:class:`~pulse2percept.implants.retina.Ho2019FlatArray`, and
:py:class:`~pulse2percept.implants.retina.Huang2021Array` model the arrays of
[Lorach2015]_, [Ho2019]_, and [Huang2021]_. The plots below use the same
physical scale:

.. plot::

    import matplotlib.pyplot as plt
    import pulse2percept as p2p

    implants = [
        ('PRIMAPivotal()', p2p.implants.retina.PRIMAPivotal()),
        ('Lorach2015Array()', p2p.implants.retina.Lorach2015Array()),
        ('Ho2019FlatArray(55)', p2p.implants.retina.Ho2019FlatArray(55)),
        ('Ho2019FlatArray(40)', p2p.implants.retina.Ho2019FlatArray(40)),
        ('Huang2021Array(55)', p2p.implants.retina.Huang2021Array(55)),
        ('Huang2021Array(40)', p2p.implants.retina.Huang2021Array(40)),
        ('Huang2021Array(30)', p2p.implants.retina.Huang2021Array(30)),
        ('Huang2021Array(20)', p2p.implants.retina.Huang2021Array(20)),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)

    for ax, (title, implant) in zip(axes.flat, implants):
        implant.plot(ax=ax)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-1100, 1100)
        ax.set_ylim(-1100, 1100)
        ax.set_aspect('equal')
        ax.set_xlabel('')
        ax.set_ylabel('')

    fig.tight_layout()

.. list-table::
   :header-rows: 1
   :widths: 30 12 38 20

   * - Object
     - Pixels
     - Pixel geometry
     - Substrate
   * - ``PRIMAPivotal()``
     - 378
     - 100 um wide, 100 um spacing, 28 um active
     - 2 x 2 mm
   * - ``Lorach2015Array()``
     - 142
     - 70 um wide, 75 um spacing, 20 um active
     - 1 mm
   * - ``Ho2019FlatArray(55)``
     - 250
     - 55 um wide/spacing, 14 um active
     - 1 mm
   * - ``Ho2019FlatArray(40)``
     - 502
     - 40 um wide/spacing, 10 um active
     - 1 mm
   * - ``Huang2021Array(55)``
     - 421
     - 55 um wide/spacing, 22 um active
     - 1.5 mm
   * - ``Huang2021Array(40)``
     - 821
     - 40 um wide/spacing, 16 um active
     - 1.5 mm
   * - ``Huang2021Array(30)``
     - 1388
     - 30 um wide/spacing, 12 um active
     - 1.5 mm
   * - ``Huang2021Array(20)``
     - 2806
     - 20 um wide/spacing, 8 um active
     - 1.5 mm

The F55 layout of :py:class:`~pulse2percept.implants.retina.Ho2019FlatArray` is
reconstructed from Fig. 2(a) of [Ho2019]_. The F40 outline was not published,
so ``Ho2019FlatArray(40)`` uses the 502 lattice sites nearest the substrate
center.

For :py:class:`~pulse2percept.implants.retina.Huang2021Array`, the photovoltaic
cell ("pixel") count includes only exposed, stimulating pixels.
The fabricated arrays included more cells than were exposed for
stimulation, which were used for the common return electrode.
The total number of fabricated cells was therefore 526 for F55,
1027 for F40, 1735 for F30, and 3508 for F20.
However, these peripheral cells covered by the common return are not
independently stimulating and are therefore not exposed in pulse2percept.

All four classes are driven by pulsed near-infrared illumination rather than
injected current: ``prepare_stim`` returns irradiance in ``mW/mm^2``.

An implant class describes the photovoltaic *array*; its default encoder
describes a documented optical stimulation protocol for that experimental
system. The two are separate, so each array gets its own encoder:

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - Object
     - Default encoder
     - Optical protocol
   * - ``PRIMAPivotal()``
     - ``PRIMAEncoder``
     - 880 nm, 3.5 mW/mm^2, 30 Hz, ON durations on a 0.7 ms grid up to 9.8 ms
   * - ``Lorach2015Array()``
     - ``PhotovoltaicEncoder``
     - 915 nm, 4 mW/mm^2, 4 ms pulses at 40 Hz [Lorach2015]_
   * - ``Ho2019FlatArray()``
     - ``PhotovoltaicEncoder``
     - 915 nm, 8 mW/mm^2, 4 ms pulses at 40 Hz [Ho2019]_
   * - ``Huang2021Array()``
     - ``PhotovoltaicEncoder``
     - 880 nm, 4.7 mW/mm^2, 10 ms pulses at 2 Hz [Huang2021]_

[Huang2021]_ swept irradiance from 0.002 to 4.7 mW/mm^2 while measuring VEP
thresholds rather than running a video system, so 4.7 mW/mm^2 is its brightest
measured condition, not a device or safety maximum.

None of these papers specifies a natural-image grayscale transfer function, so
gray level maps to ON duration by an explicit pulse2percept simulation
convention (linear, and additionally quantized onto the 0.7 ms grid for
``PRIMAEncoder``) rather than by a reconstruction of the original camera
pipeline. Pass a configured encoder or ``encoder=None`` to opt out.

``safe_mode=True`` checks the documented PRIMA projector envelope on
:py:class:`~pulse2percept.implants.retina.PRIMAPivotal`. No comparable
envelope has been published for the research arrays, so they raise rather than
borrow PRIMA's limits. Negative or non-finite irradiance is rejected on every
array, with or without ``safe_mode``.

Photovoltaic conversion to tissue current, and retinal transduction, are not
modeled.

``PRIMA``, ``PRIMA75``, ``PRIMA55`` and ``PRIMA40`` are deprecated aliases;
see the v0.11 release notes for the corresponding canonical names.

Cortical implants
-----------------

Cortical implants derive from
:py:class:`~pulse2percept.implants.cortex.CorticalImplant` and use physical
cortical coordinates. A cortical model combines those coordinates with a
:py:class:`~pulse2percept.topography.VisualFieldMap` to place stimulation in
the visual field; see :ref:`topics-coordinates`.

.. list-table::
   :header-rows: 1
   :widths: 26 18 14 42

   * - Object
     - Placement
     - Electrodes
     - Description
   * - :py:class:`~pulse2percept.implants.cortex.Orion`
     - epicortical
     - 60
     - Orion cortical visual prosthesis
   * - :py:class:`~pulse2percept.implants.cortex.Cortivis`
     - intracortical
     - 96
     - CORTIVIS Utah-style array
   * - :py:class:`~pulse2percept.implants.cortex.ICVP`
     - intracortical
     - 18
     - Intracortical Visual Prosthesis
   * - :py:class:`~pulse2percept.implants.cortex.Neuralink`
     - intracortical
     - per thread
     - Ensemble of Neuralink-style threads

Both cortical models, :py:class:`~pulse2percept.models.cortex.ScoreboardModel`
and :py:class:`~pulse2percept.models.cortex.DynaphosModel`, accept any of them.

.. plot::

    import matplotlib.pyplot as plt
    import pulse2percept as p2p

    devices = [
        ('Orion()', p2p.implants.cortex.Orion()),
        ('Cortivis()', p2p.implants.cortex.Cortivis()),
        ('ICVP()', p2p.implants.cortex.ICVP()),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (title, implant) in zip(axes, devices):
        implant.plot(ax=ax)
        ax.set_title(title, fontsize=9)
        ax.set_aspect('equal')
        ax.set_xlabel('x (um)', fontsize=8)
        ax.set_ylabel('y (um)', fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()

``hemisphere`` (``'left'``, ``'right'``, or ``None`` if unspecified) is device
metadata only: the electrode coordinates and the model's ``implant_position``
remain what places the array, and recording a hemisphere neither reflects the
geometry nor overrides it.
:py:class:`~pulse2percept.implants.cortex.Neuralink` is an ensemble of threads
and offers the same attribute.

Custom arrays
-------------

A custom array usually does not need a new implant class. For a regular grid,
use :py:class:`~pulse2percept.implants.GridImplant`:

.. code-block:: python

    implant = p2p.implants.GridImplant(shape=(10, 10), spacing=500)

    implant = p2p.implants.GridImplant(
        shape=(20, 20), spacing=400, grid_type='hex',
        electrode_type=p2p.implants.DiskElectrode, radius=75)

Electrodes are point sources unless an ``electrode_type`` and its arguments
give them a physical extent.
:py:class:`~pulse2percept.implants.GridImplant` is a convenience only:
:py:class:`~pulse2percept.implants.ElectrodeGrid` describes the geometry and
:py:class:`~pulse2percept.implants.Implant` describes the device, so an
irregular array can be built from individual electrodes and wrapped by hand.

:py:class:`~pulse2percept.implants.GridImplant` and
:py:class:`~pulse2percept.implants.Implant` are anatomy-neutral and carry
neither ``eye`` nor ``hemisphere``. A custom array that a retinal or cortical
model should read as sitting on one side goes to the target-specific class
instead:

.. code-block:: python

    from pulse2percept.implants import ElectrodeGrid
    from pulse2percept.implants.retina import RetinalImplant
    from pulse2percept.implants.cortex import CorticalImplant

    array = ElectrodeGrid(shape=(10, 10), spacing=500)
    retinal = RetinalImplant(array, eye='right')
    cortical = CorticalImplant(array, hemisphere='right')

:py:class:`~pulse2percept.implants.EnsembleImplant` combines multiple implants
into one system. Its
:py:meth:`~pulse2percept.implants.EnsembleImplant.from_visual_field_map`
places one constituent per visual field location, through any 2D
:py:class:`~pulse2percept.topography.VisualFieldMap`.
