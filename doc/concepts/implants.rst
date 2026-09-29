.. _topics-implants:

========
Implants
========

An implant defines the hardware that delivers stimulation: the electrodes,
their geometry, and, when applicable, the encoding and rastering rules used to
turn images or video into electrode stimuli.

The stimulus itself is separate from the implant; see
:ref:`topics-stimulation`. Placement of the device in tissue is handled by the
model; see :ref:`topics-coordinates`.

Retinal devices are available under :mod:`pulse2percept.implants.retina`;
cortical devices under :mod:`pulse2percept.implants.cortex`. Generic
electrodes, arrays, grids, and
:py:class:`~pulse2percept.implants.EnsembleImplant` live directly in
:mod:`pulse2percept.implants`.


Quick Start
===========

.. code-block:: python

    import pulse2percept as p2p

    implant = p2p.implants.retina.ArgusII()

    implant['A8']                        # electrode by name
    implant[0]                           # electrode by index
    implant.electrode_names
    implant.electrode_array.coordinates()
    implant.plot()


Implant Interface
=================

All implants derive from :py:class:`~pulse2percept.implants.Implant`.

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Attribute
     - Meaning
     - Typical values
   * - ``electrode_array``
     - Electrode geometry and device-local coordinates
     - :py:class:`~pulse2percept.implants.ElectrodeArray`
   * - ``placement``
     - Tissue interface represented by the device
     - ``'epiretinal'``, ``'subretinal'``, ``'suprachoroidal'``,
       ``'epicortical'``, ``'intracortical'``
   * - ``encoder``
     - Converts image or video input into electrode values
     - Device-specific encoder or ``None``
   * - ``raster``
     - Controls when electrodes are stimulated
     - Sequential raster, simultaneous stimulation, or ``None``
   * - ``preprocess``
     - Optional preprocessing applied before encoding
     - Callable or ``None``
   * - ``thresholds``
     - Perceptual threshold current used to convert ``xTh`` amplitudes
       to current
     - Scalar or per-electrode dictionary
   * - ``safe_mode``
     - Enables device-specific stimulation checks
     - ``True`` or ``False``
   * - ``max_current``
     - Maximum instantaneous current summed across electrodes
     - Current in uA
   * - ``scene_input_frame``
     - Reference frame of image or video input
     - ``'eye'`` or ``'head'``

``encoder``, ``raster``, and ``preprocess`` are described in
:ref:`topics-stimulation`. ``scene_input_frame`` determines whether gaze moves
the input and is described in :ref:`topics-vision`.


Retinal Implants
================

Retinal implant classes derive from
:py:class:`~pulse2percept.implants.retina.RetinalImplant`.

The ``eye`` argument records which eye is implanted (``'left'`` or
``'right'``; default ``'right'``). Axon map models use it to locate the optic
disc. Some devices also reverse electrode column names for the left eye; see
the individual API documentation for details.

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
   * - :py:class:`~pulse2percept.implants.retina.Suprachoroidal24`
     - suprachoroidal
     - 35
     - First-generation suprachoroidal array
     - ``ScoreboardModel``
   * - :py:class:`~pulse2percept.implants.retina.Suprachoroidal44`
     - suprachoroidal
     - 46
     - Second-generation suprachoroidal array
     - ``ScoreboardModel``
   * - :py:class:`~pulse2percept.implants.retina.PRIMAPivotal`
     - subretinal
     - 378
     - PRIMA photovoltaic array [Holz2026]_
     - ``Ho2018Model``, ``ScoreboardModel``

These classes are reconstructed from published device descriptions rather than
manufacturer-validated simulators. ``Suprachoroidal24`` and
``Suprachoroidal44`` are pulse2percept identifiers, not product names.


Physical Scale
--------------

Retinal arrays differ substantially in size. The plots below use the same
device-local units (um):

.. plot::

    import matplotlib.pyplot as plt
    import pulse2percept as p2p

    devices = [
        ('ArgusII()', p2p.implants.retina.ArgusII()),
        ('AlphaAMS()', p2p.implants.retina.AlphaAMS()),
        ('Suprachoroidal44()', p2p.implants.retina.Suprachoroidal44()),
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
--------

``ArgusII()`` includes the device's default video pipeline: an
:py:class:`~pulse2percept.stimuli.AmplitudeEncoder` operating at 6 Hz and a
:py:class:`~pulse2percept.implants.SequentialRaster` that stimulates one of
six rows every 2 ms. Images and videos can therefore be passed to the implant
directly.

Both stages can be disabled:

.. code-block:: python

    implant = p2p.implants.retina.ArgusII(
        encoder=None,
        raster=None,
    )


Photovoltaic Arrays
-------------------

:py:class:`~pulse2percept.implants.retina.PRIMAPivotal` represents the
378-pixel, 100 um subretinal array used in the pivotal PRIMAvera trial
[Holz2026]_, matching the first-in-human device [Palanker2020]_.

Additional research arrays reproduce geometries reported by [Lorach2015]_,
[Ho2019]_, and [Huang2021]_:

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

    fig, axes = plt.subplots(
        2, 4, figsize=(12, 6), sharex=True, sharey=True
    )

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

Hexagonal rows are separated by ``spacing * sqrt(3) / 2``.

.. note::

   **Geometry.** ``Ho2019FlatArray(55)`` is reconstructed from Fig. 2(a) of
   [Ho2019]_. The F40 outline was not published, so
   ``Ho2019FlatArray(40)`` uses the 502 lattice sites nearest the substrate
   center.

   ``Huang2021Array`` includes only exposed stimulating pixels. The fabricated
   arrays contained 526 (F55), 1027 (F40), 1735 (F30), and 3508 (F20) cells;
   peripheral cells forming the common return are not modeled.

   **Optical stimulation.** These arrays are driven by pulsed near-infrared
   light, so ``prepare_stim`` returns irradiance in mW/mm^2; see
   :ref:`topics-encoders`. Conversion from optical power to tissue current is
   not modeled.

   The 4.7 mW/mm^2 value used by ``Huang2021Array`` is the highest condition in
   the published VEP threshold sweep (0.002-4.7 mW/mm^2), not a device or
   safety maximum.

   **Safety checks.** Negative or non-finite irradiance is rejected.
   ``safe_mode=True`` checks the documented PRIMA projector envelope for
   ``PRIMAPivotal``. No corresponding published envelope is available for the
   research arrays, so they raise ``NotImplementedError`` when ``safe_mode`` is
   requested.

.. note::

   ``PRIMA``, ``PRIMA75``, ``PRIMA55``, and ``PRIMA40`` are deprecated aliases.
   See the v0.11 release notes for their replacements.


Cortical Implants
=================

Cortical implant classes derive from
:py:class:`~pulse2percept.implants.cortex.CorticalImplant`.

Both cortical models,
:py:class:`~pulse2percept.models.cortex.ScoreboardModel` and
:py:class:`~pulse2percept.models.cortex.DynaphosModel`, accept any cortical
implant.

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
        ('NeuroPortArray()', p2p.implants.cortex.NeuroPortArray()),
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

The ``hemisphere`` argument (``'left'``, ``'right'``, or ``None``) is metadata.
The model's ``implant_position`` determines where the array is placed. Setting
``hemisphere`` does not move or mirror it.

Cortical implants do not define a default encoder.


Custom Arrays
=============

Regular layouts can be created directly with
:py:class:`~pulse2percept.implants.GridImplant`:

.. code-block:: python

    implant = p2p.implants.GridImplant(
        shape=(10, 10),
        spacing=500,
    )

    implant = p2p.implants.GridImplant(
        shape=(20, 20),
        spacing=400,
        grid_type='hex',
        electrode_type=p2p.implants.DiskElectrode,
        radius=75,
    )

Electrodes are point sources unless ``electrode_type`` gives them a finite
extent.

For irregular layouts, construct an
:py:class:`~pulse2percept.implants.ElectrodeArray` from individual electrodes
and wrap it in :py:class:`~pulse2percept.implants.Implant`.


Laterality
----------

``GridImplant`` and ``Implant`` do not carry ``eye`` or ``hemisphere``
metadata. Wrap the array in the corresponding tissue-specific class when
laterality is needed:

.. code-block:: python

    from pulse2percept.implants import ElectrodeGrid
    from pulse2percept.implants.retina import RetinalImplant
    from pulse2percept.implants.cortex import CorticalImplant

    array = ElectrodeGrid(shape=(10, 10), spacing=500)

    retinal = RetinalImplant(array, eye='right')
    cortical = CorticalImplant(array, hemisphere='right')


Multiple Implants
-----------------

:py:class:`~pulse2percept.implants.EnsembleImplant` combines several implants
into one system.

:py:meth:`~pulse2percept.implants.EnsembleImplant.from_visual_field_map`
places one copy per visual-field location using a two-dimensional
:py:class:`~pulse2percept.topography.VisualFieldMap`.