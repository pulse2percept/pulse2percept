.. _topics-implants:

========
Implants
========

An implant describes a device: its electrodes, their device-local geometry,
and the rules that convert a source (pulses, an image, a video) into the
stimulation those electrodes deliver. It stores no stimulus; see
:ref:`topics-stimulation`. Where the device sits in tissue is a model
parameter; see :ref:`topics-coordinates`.

Devices live under the tissue they stimulate, ``p2p.implants.retina`` and
``p2p.implants.cortex``. Electrodes, arrays, rasters, and
:py:class:`~pulse2percept.implants.EnsembleImplant` live at the root of
:py:mod:`pulse2percept.implants`.

.. code-block:: python

    import pulse2percept as p2p

    implant = p2p.implants.retina.ArgusII()

    implant['A8']                  # electrode by name
    implant[0]                     # electrode by index
    implant.electrode_names
    implant.electrode_array.coordinates()   # (x, y, z) in um
    implant.plot()

All implants derive from :py:class:`~pulse2percept.implants.Implant`. Its
main attributes:

``electrode_array``
    The :py:class:`~pulse2percept.implants.ElectrodeArray`.

``placement``
    ``'epiretinal'``, ``'subretinal'``, ``'suprachoroidal'``,
    ``'epicortical'``, ``'intracortical'``, or ``None`` for a generic array.

``encoder``, ``raster``, ``preprocess``
    How image and video input becomes stimulation; see
    :ref:`topics-stimulation`.

``thresholds``
    Perceptual threshold current (uA), used to convert ``xTh`` amplitudes to
    current. A scalar applies to every electrode; a dict to the named ones.

``safe_mode``, ``max_current``
    Checks applied when stimulation is prepared: charge balance, and the
    total instantaneous current (uA) summed over electrodes.

``scene_input_frame``
    ``'eye'`` if the input passes through the eye's optics, ``'head'`` for a
    head-mounted camera. Sets whether gaze moves the input; see
    :ref:`topics-vision`.

Retinal implants
----------------

Retinal implants derive from
:py:class:`~pulse2percept.implants.retina.RetinalImplant`. ``eye``
(``'left'`` or ``'right'``, default ``'right'``) records the implanted eye;
axon map models use it to place the optic disc. Some devices also reverse
their column names in the left eye (see each class's API documentation).

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

These classes are built from published device descriptions; they are not
manufacturer-validated simulators. ``BVT24`` and ``BVT44`` are pulse2percept
identifiers, not product names.

The arrays differ widely in physical scale (device-local microns):

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

Argus II
^^^^^^^^

``ArgusII()`` includes the device's video defaults: an
:py:class:`~pulse2percept.stimuli.AmplitudeEncoder` at 6 Hz and a
:py:class:`~pulse2percept.implants.SequentialRaster` that pulses one of six
rows every 2 ms. Images and videos can therefore be passed to it directly.
``encoder=None`` disables encoding; ``raster=None`` lets all electrodes pulse
at once:

.. code-block:: python

    implant = p2p.implants.retina.ArgusII(encoder=None, raster=None)

Photovoltaic arrays
^^^^^^^^^^^^^^^^^^^

:py:class:`~pulse2percept.implants.retina.PRIMAPivotal` models the 378-pixel,
100 um subretinal array of the pivotal PRIMAvera trial [Holz2026]_, which
matches the first-in-human device [Palanker2020]_. Three research arrays
model [Lorach2015]_, [Ho2019]_, and [Huang2021]_. All are shown at the same
scale:

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
   :widths: 24 9 33 10 24

   * - Object
     - Pixels
     - Pixel geometry
     - Substrate
     - Default optical protocol
   * - ``PRIMAPivotal()``
     - 378
     - 100 um wide, 100 um spacing, 28 um active
     - 2 x 2 mm
     - ``PRIMAEncoder``: 880 nm, 3.5 mW/mm^2, 30 Hz, 0.7-9.8 ms ON
   * - ``Lorach2015Array()``
     - 142
     - 70 um wide, 75 um spacing, 20 um active
     - 1 mm
     - ``PhotovoltaicEncoder``: 915 nm, 4 mW/mm^2, 4 ms at 40 Hz
   * - ``Ho2019FlatArray(55)``
     - 250
     - 55 um wide/spacing, 14 um active
     - 1 mm
     - ``PhotovoltaicEncoder``: 915 nm, 8 mW/mm^2, 4 ms at 40 Hz
   * - ``Ho2019FlatArray(40)``
     - 502
     - 40 um wide/spacing, 10 um active
     - 1 mm
     - same as above
   * - ``Huang2021Array(55)``
     - 421
     - 55 um wide/spacing, 22 um active
     - 1.5 mm
     - ``PhotovoltaicEncoder``: 880 nm, 4.7 mW/mm^2, 10 ms at 2 Hz
   * - ``Huang2021Array(40)``
     - 821
     - 40 um wide/spacing, 16 um active
     - 1.5 mm
     - same as above
   * - ``Huang2021Array(30)``
     - 1388
     - 30 um wide/spacing, 12 um active
     - 1.5 mm
     - same as above
   * - ``Huang2021Array(20)``
     - 2806
     - 20 um wide/spacing, 8 um active
     - 1.5 mm
     - same as above

Hexagonal rows are ``spacing * sqrt(3) / 2`` apart. Notes on the geometry:

*  ``Ho2019FlatArray(55)`` is reconstructed from Fig. 2(a) of [Ho2019]_. The
   F40 outline was not published, so ``Ho2019FlatArray(40)`` uses the 502
   lattice sites nearest the substrate center.
*  ``Huang2021Array`` counts only exposed, stimulating pixels. The fabricated
   arrays had 526 (F55), 1027 (F40), 1735 (F30), and 3508 (F20) cells; the
   peripheral ones formed the common return and are not modeled.
*  The Huang2021 irradiance, 4.7 mW/mm^2, is the brightest condition of a VEP
   threshold sweep (0.002-4.7 mW/mm^2), not a device or safety maximum.

These arrays are driven by pulsed near-infrared light, so ``prepare_stim``
returns irradiance in mW/mm^2 (see :ref:`topics-encoders`). Photovoltaic
conversion to tissue current is not modeled. Negative or non-finite
irradiance is rejected. ``safe_mode=True`` checks the documented PRIMA
projector envelope on ``PRIMAPivotal``; the research arrays have no
published envelope, so they reject ``safe_mode=True`` with a
``NotImplementedError``.

``PRIMA``, ``PRIMA75``, ``PRIMA55`` and ``PRIMA40`` are deprecated aliases;
the v0.11 release notes list their replacements.

Cortical implants
-----------------

Cortical implants derive from
:py:class:`~pulse2percept.implants.cortex.CorticalImplant`. Both cortical
models, :py:class:`~pulse2percept.models.cortex.ScoreboardModel` and
:py:class:`~pulse2percept.models.cortex.DynaphosModel`, accept any of them.

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
   * - :py:class:`~pulse2percept.implants.cortex.NeuroPortArray`
     - intracortical
     - 96
     - CORTIVIS NeuroPort array
   * - :py:class:`~pulse2percept.implants.cortex.ICVP`
     - intracortical
     - 18
     - Intracortical Visual Prosthesis (ICVP)
   * - :py:class:`~pulse2percept.implants.cortex.Neuralink`
     - intracortical
     - per thread
     - Ensemble of Neuralink-style threads

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

``hemisphere`` (``'left'``, ``'right'``, or ``None``) is metadata only. The
model's ``implant_position`` places the array; setting ``hemisphere`` neither
moves nor mirrors it. Cortical implants have no default encoder.

Custom arrays
-------------

A regular grid needs no new class:

.. code-block:: python

    implant = p2p.implants.GridImplant(shape=(10, 10), spacing=500)

    implant = p2p.implants.GridImplant(
        shape=(20, 20), spacing=400, grid_type='hex',
        electrode_type=p2p.implants.DiskElectrode, radius=75)

Electrodes are point sources unless ``electrode_type`` gives them an extent.
For an irregular layout, build an
:py:class:`~pulse2percept.implants.ElectrodeArray` from individual electrodes
and wrap it in an :py:class:`~pulse2percept.implants.Implant`.

``GridImplant`` and ``Implant`` carry neither ``eye`` nor ``hemisphere``. To
state laterality, wrap the array in the tissue-specific class:

.. code-block:: python

    from pulse2percept.implants import ElectrodeGrid
    from pulse2percept.implants.retina import RetinalImplant
    from pulse2percept.implants.cortex import CorticalImplant

    array = ElectrodeGrid(shape=(10, 10), spacing=500)
    retinal = RetinalImplant(array, eye='right')
    cortical = CorticalImplant(array, hemisphere='right')

:py:class:`~pulse2percept.implants.EnsembleImplant` combines several implants
into one system.
:py:meth:`~pulse2percept.implants.EnsembleImplant.from_visual_field_map`
places one copy per visual-field location through a 2D
:py:class:`~pulse2percept.topography.VisualFieldMap`.
