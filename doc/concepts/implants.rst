.. _topics-implants:

========
Implants
========

.. _topics-implants-retina:

Retinal Implants
================

Retinal implants span epiretinal, subretinal, and suprachoroidal devices with
very different electrode counts, geometries, and physical scales.

Retinal implant classes derive from
:py:class:`~pulse2percept.implants.retina.RetinalImplant`.

Their ``eye`` argument records the implanted eye (``'left'`` or ``'right'``;
default ``'right'``). Axon map models use it to locate the optic disc. Some
devices also reverse electrode column names in the left eye; see the
individual API documentation.

Several representative arrays are shown below at the same physical scale:

.. plot::

    import matplotlib.pyplot as plt
    import pulse2percept as p2p

    devices = [
        ('Argus II', p2p.implants.retina.ArgusII()),
        ('Alpha AMS', p2p.implants.retina.AlphaAMS()),
        ('Suprachoroidal 44', p2p.implants.retina.Suprachoroidal44()),
        ('IMIE', p2p.implants.retina.IMIE()),
        ('PRIMA', p2p.implants.retina.PRIMAPivotal()),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))

    for ax, (title, implant) in zip(axes.flat, devices):
        implant.plot(ax=ax)
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlabel('x (um)', fontsize=8)
        ax.set_ylabel('y (um)', fontsize=8)
        ax.tick_params(labelsize=7)

    axes.flat[-1].axis('off')
    fig.tight_layout()


.. _topics-implants-human:

Human Implant Systems
---------------------

pulse2percept includes geometries for several retinal prosthesis systems that
have been implanted in people:

.. list-table::
   :header-rows: 1
   :widths: 25 16 12 25 22

   * - Implant
     - Placement
     - Electrodes
     - Device
     - Typical Model
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

:py:class:`~pulse2percept.implants.retina.PRIMAPivotal` represents the
378-pixel, 100 um subretinal array used in the pivotal PRIMAvera trial
[Holz2026]_, matching the first-in-human device [Palanker2020]_.

.. note::

   These classes reproduce published implant geometries and device properties;
   they are not manufacturer-validated device simulators.

   ``Suprachoroidal24`` and ``Suprachoroidal44`` are pulse2percept names for
   the first- and second-generation suprachoroidal systems, not product names.


.. _topics-implants-research:

Research Array Geometries
-------------------------

pulse2percept also includes photovoltaic array geometries reconstructed from
preclinical studies by [Lorach2015]_, [Ho2019]_, and [Huang2021]_. These
classes represent the experimental arrays used in those studies rather than
complete clinical implant systems.

.. plot::

    import matplotlib.pyplot as plt
    import pulse2percept as p2p

    implants = [
        ('Lorach 2015', p2p.implants.retina.Lorach2015Array()),
        ('Ho 2019: 55 um', p2p.implants.retina.Ho2019FlatArray(55)),
        ('Ho 2019: 40 um', p2p.implants.retina.Ho2019FlatArray(40)),
        ('Huang 2021: 55 um', p2p.implants.retina.Huang2021Array(55)),
        ('Huang 2021: 40 um', p2p.implants.retina.Huang2021Array(40)),
        ('Huang 2021: 30 um', p2p.implants.retina.Huang2021Array(30)),
        ('Huang 2021: 20 um', p2p.implants.retina.Huang2021Array(20)),
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

    axes.flat[-1].axis('off')
    fig.tight_layout()

.. list-table::
   :header-rows: 1
   :widths: 24 9 33 10 24

   * - Array
     - Pixels
     - Pixel Geometry
     - Substrate
     - Default Optical Protocol
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

   ``Ho2019FlatArray(55)`` is reconstructed from Fig. 2(a) of [Ho2019]_.
   Because the F40 outline was not published, ``Ho2019FlatArray(40)`` uses
   the 502 lattice sites nearest the substrate center.

   ``Huang2021Array`` includes only exposed stimulating pixels. Peripheral
   cells forming the common return are not modeled.

These arrays use pulsed near-infrared stimulation; their optical protocols are
described in :ref:`topics-stimulation`.


.. _topics-implants-cortex:

Cortical Implants
=================

pulse2percept includes both surface and penetrating cortical arrays.

.. plot::

    import matplotlib.pyplot as plt
    import pulse2percept as p2p

    devices = [
        ('Orion', p2p.implants.cortex.Orion()),
        ('CORTIVIS NeuroPort', p2p.implants.cortex.NeuroPortArray()),
        ('ICVP', p2p.implants.cortex.ICVP()),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))

    for ax, (title, implant) in zip(axes, devices):
        implant.plot(ax=ax)
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlabel('x (um)', fontsize=8)
        ax.set_ylabel('y (um)', fontsize=8)
        ax.tick_params(labelsize=7)

    fig.tight_layout()

.. list-table::
   :header-rows: 1
   :widths: 28 18 14 40

   * - Implant
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

Cortical implant classes derive from
:py:class:`~pulse2percept.implants.cortex.CorticalImplant`.

Both cortical models,
:py:class:`~pulse2percept.models.cortex.ScoreboardModel` and
:py:class:`~pulse2percept.models.cortex.DynaphosModel`, accept cortical
implants.

The ``hemisphere`` argument (``'left'``, ``'right'``, or ``None``) records
laterality but does not place or mirror the array. The model's
``implant_position`` determines its location in cortex.

Cortical implants do not define a default encoder.