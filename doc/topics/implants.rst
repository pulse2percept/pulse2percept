.. _topics-implants:

=================
Visual Prostheses
=================

An implant describes the device: its electrodes, their geometry and location,
and how a source stimulus becomes the current those electrodes deliver.

All implants derive from
:py:class:`~pulse2percept.implants.ProsthesisSystem`. The attributes used most
often are:

``earray``
    The :py:class:`~pulse2percept.implants.ElectrodeArray`.

``eye``
    for retinal systems: the implanted eye

``placement``
    Where the device sits relative to the tissue it stimulates
    (``'epiretinal'``, ``'subretinal'``, ``'suprachoroidal'``,
    ``'epicortical'``, ``'intracortical'``), or ``None`` for a generic array.

``technology``
    Stimulation technology, such as ``'photovoltaic'``, where specified.

``family``
    Named device family, where applicable.

``encoder`` and ``raster``
    Optional device behavior used when visual input is converted to electrical
    stimulation. These are covered later in :ref:`topics-encoders` and
    :ref:`topics-rasters`.

New in v0.11.0: An implant holds no stimulus. What it delivers is derived from
a source, on demand, by
:py:meth:`~pulse2percept.implants.ProsthesisSystem.prepare_stim`.

Basic use
---------

Electrodes can be accessed by name or index, and the array can be plotted
directly:

.. code-block:: python

    import pulse2percept as p2p

    implant = p2p.implants.ArgusII()

    implant['A8']
    implant[0]
    implant.electrode_names
    implant.earray.coordinates()
    implant.plot()

Preparing a stimulus
--------------------

:py:meth:`~pulse2percept.implants.ProsthesisSystem.prepare_stim` converts a
source into stimulation the device can deliver:

.. code-block:: python

    delivered = implant.prepare_stim({'A8': 30})
    delivered = implant.prepare_stim(p2p.stimuli.BostonTrain())

Preparation includes preprocessing, image/video encoding, resampling onto the
electrode array, raster scheduling, threshold calibration, and safety checks.
The result is returned as a :py:class:`~pulse2percept.stimuli.Stimulus` object.

Models call ``prepare_stim`` internally. Call it directly when the delivered
stimulation itself is of interest:

.. code-block:: python

    implant.prepare_stim(source).plot()
    implant.plot(stim=source, stim_cmap=True)

Retinal implants
----------------

Retinal implants are centered on the fovea and store distances to the
neuronal targets in microns.
Positive ``x`` points toward nasal retina, positive ``y`` toward superior
retina, and positive ``z`` away from the retina into the vitreous. ``eye``
handles left- versus right-eye geometry where needed.

PRIMA
^^^^^

PRIMA is a subretinal photovoltaic prosthesis developed at Stanford. Pixium
Vision developed the clinical system; Science Corporation acquired Pixium's
PRIMA assets and intellectual property in 2024.

:py:class:`~pulse2percept.implants.PRIMAPivotal` models the 378-pixel device
used in the pivotal PRIMAvera trial [Holz2026]_. The same 100 um configuration
was used in the earlier first-in-human study [Palanker2020]_.

pulse2percept also includes several photovoltaic research arrays described in
the literature. The plots below use the same physical scale:

.. plot::

    import matplotlib.pyplot as plt
    import pulse2percept as p2p

    implants = [
        ('PRIMAPivotal()', p2p.implants.PRIMAPivotal()),
        ('Lorach2015Array()', p2p.implants.Lorach2015Array()),
        ('Ho2019FlatArray(55)', p2p.implants.Ho2019FlatArray(55)),
        ('Ho2019FlatArray(40)', p2p.implants.Ho2019FlatArray(40)),
        ('Huang2021Array(55)', p2p.implants.Huang2021Array(55)),
        ('Huang2021Array(40)', p2p.implants.Huang2021Array(40)),
        ('Huang2021Array(30)', p2p.implants.Huang2021Array(30)),
        ('Huang2021Array(20)', p2p.implants.Huang2021Array(20)),
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

For a hexagonal array, the row spacing is
``spacing * sqrt(3) / 2``.

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
     - 421 (526 total)
     - 55 um wide/spacing, 22 um active
     - 1.5 mm
   * - ``Huang2021Array(40)``
     - 821 (1027 total)
     - 40 um wide/spacing, 16 um active
     - 1.5 mm
   * - ``Huang2021Array(30)``
     - 1388 (1735 total)
     - 30 um wide/spacing, 12 um active
     - 1.5 mm
   * - ``Huang2021Array(20)``
     - 2806 (3508 total)
     - 20 um wide/spacing, 8 um active
     - 1.5 mm

:py:class:`~pulse2percept.implants.PRIMAPivotal` is based on the pivotal
PRIMAvera device [Holz2026]_, also used in the earlier first-in-human study
[Palanker2020]_. :py:class:`~pulse2percept.implants.Lorach2015Array`,
:py:class:`~pulse2percept.implants.Ho2019FlatArray`, and
:py:class:`~pulse2percept.implants.Huang2021Array` model the research arrays
described in [Lorach2015]_, [Ho2019]_, and [Huang2021]_, respectively.

For :py:class:`~pulse2percept.implants.Huang2021Array`, ``n_electrodes`` is
the number of exposed, stimulating pixels. ``n_total_pixels`` includes the
peripheral pixels covered by the common return electrode.

The F55 layout of :py:class:`~pulse2percept.implants.Ho2019FlatArray` is
reconstructed from Fig. 2(a) of [Ho2019]_. The F40 outline was not published,
so ``Ho2019FlatArray(40)`` uses the 502 lattice sites nearest the substrate
center.

``PRIMA``, ``PRIMA75``, ``PRIMA55`` and ``PRIMA40`` are deprecated aliases;
see the v0.11 release notes for the corresponding canonical names.

Argus
^^^^^

:py:class:`~pulse2percept.implants.ArgusI` and
:py:class:`~pulse2percept.implants.ArgusII` model the epiretinal Argus
prostheses. Argus I has 16 electrodes in a 4 x 4 array; Argus II has 60
electrodes in a 6 x 10 array.

Argus II also defines its device-specific image encoder and sequential raster,
so image and video stimuli can be passed directly to
:py:meth:`~pulse2percept.implants.ProsthesisSystem.prepare_stim`.

Alpha IMS and AMS
^^^^^^^^^^^^^^^^^

:py:class:`~pulse2percept.implants.AlphaIMS` and
:py:class:`~pulse2percept.implants.AlphaAMS` model the subretinal Alpha
microphotodiode arrays.

Suprachoroidal implants
^^^^^^^^^^^^^^^^^^^^^^^

:py:class:`~pulse2percept.implants.BVT24` and
:py:class:`~pulse2percept.implants.BVT44` model first- and second-generation
suprachoroidal arrays. The class names are pulse2percept identifiers rather
than official product names.

Other retinal implants
^^^^^^^^^^^^^^^^^^^^^^

:py:class:`~pulse2percept.implants.IMIE` models the epiretinal IMIE array.

These classes are research-software representations based on published device
descriptions, not manufacturer-validated simulators. See each class's API
documentation for device-specific geometry and assumptions.

Cortical implants
-----------------

Cortical implants are available under :mod:`pulse2percept.implants.cortex`:

.. list-table::
   :header-rows: 1

   * - Object
     - Description
   * - :py:class:`~pulse2percept.implants.cortex.Orion`
     - Orion cortical visual prosthesis
   * - :py:class:`~pulse2percept.implants.cortex.Cortivis`
     - CORTIVIS cortical array
   * - :py:class:`~pulse2percept.implants.cortex.ICVP`
     - Intracortical Visual Prosthesis
   * - :py:class:`~pulse2percept.implants.cortex.Neuralink`
     - Neuralink-style cortical array

Cortical implants use physical cortical coordinates. A cortical model combines
those coordinates with a
:py:class:`~pulse2percept.topography.VisualFieldMap` to place stimulation in
the visual field.

Custom implants
---------------

A custom array usually does not need a new implant class. For a regular grid,
use :py:class:`~pulse2percept.implants.GridImplant`:

.. code-block:: python

    import pulse2percept as p2p

    implant = p2p.implants.GridImplant(shape=(10, 10), spacing=500)

Grids can also be hexagonal:

.. code-block:: python

    implant = p2p.implants.GridImplant(shape=(20, 20), spacing=400, type='hex')

By default the electrodes are point sources. Pass an ``etype`` and its
arguments for electrodes with a physical extent:

.. code-block:: python

    implant = p2p.implants.GridImplant(shape=(20, 20), spacing=400, type='hex',
                                       etype=p2p.implants.DiskElectrode, r=75)

:py:class:`~pulse2percept.implants.GridImplant` is a convenience only:
:py:class:`~pulse2percept.implants.ElectrodeGrid` describes the geometry,
:py:class:`~pulse2percept.implants.ProsthesisSystem` describes the device, and
the two can still be combined by hand. Do that for an irregular array, built
from individual electrodes:

.. code-block:: python

    from pulse2percept.implants import ElectrodeArray, ProsthesisSystem

    earray = ElectrodeArray(...)
    implant = ProsthesisSystem(earray)

:py:class:`~pulse2percept.implants.EnsembleImplant` combines multiple implants
into one system.
