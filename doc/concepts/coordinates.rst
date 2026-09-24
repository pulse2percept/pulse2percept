.. _topics-coordinates:

=================================
Coordinates and Visual Field Maps
=================================

A simulation uses four coordinate systems, so device geometry, anatomy, and
the visual world can be specified independently.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Frame
     - Unit
     - Origin and meaning
   * - Device
     - um
     - Device-local. The array's own origin; identical for every copy of a
       device, wherever it is implanted.
   * - Tissue
     - um
     - Retinal or cortical position, measured from the fovea or its cortical
       representation.
   * - Visual field
     - dva
     - Eye-centered. The fovea is ``(0, 0)``; this is where phosphenes and a
       :py:class:`~pulse2percept.vision.Scotoma` live.
   * - Scene
     - dva
     - Fixed to the world in front of the eye; related to the visual field by
       gaze.

``dva`` and ``um`` convert only through a
:py:class:`~pulse2percept.topography.VisualFieldMap`, never directly (see
:ref:`topics-units`).

Device to tissue: placement
---------------------------

Electrode coordinates are device-local. The model places the device:

.. code-block:: python

    import pulse2percept as p2p
    from pulse2percept.units import dva, mm

    model = p2p.models.retina.AxonMapModel(
        p2p.implants.retina.ArgusII(),
        implant_position=(2, -1) * dva,
        implant_rotation=15,
    )

``implant_position``
    Position of the device-local origin, in tissue coordinates or dva.

``implant_rotation``
    In-plane rotation (deg), positive counter-clockwise.

``implant_depth``
    Signed offset (um) along the normal of a 2D tissue map.

Retinal positions are measured from the fovea. Cortical positions are
measured from the foveal representation of the right hemisphere; the left
hemisphere is offset by ``left_offset`` (default -20 mm) along x.

Tissue to visual field: the maps
--------------------------------

A :py:class:`~pulse2percept.topography.VisualFieldMap` converts between
visual field and tissue: ``dva_to_ret`` / ``ret_to_dva`` for retinal maps,
``dva_to_v1`` / ``v1_to_dva`` (and V2, V3) for cortical maps. The inverse
exists only where the mapping is invertible.

.. code-block:: python

    from pulse2percept.topography.retina import Watson2014Map

    x_um, y_um = Watson2014Map().dva_to_ret(2 * dva, 3 * dva)

A model holds one map (``visual_field_map``), samples the visual field on a
:py:class:`~pulse2percept.topography.Grid2D`, and maps that grid onto tissue
during ``build``.

Retinal maps
~~~~~~~~~~~~

Retinal maps derive from
:py:class:`~pulse2percept.topography.retina.RetinalMap`.

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Map
     - Reference
     - Retinal magnification
   * - :py:class:`~pulse2percept.topography.retina.Curcio1990Map`
     - [Curcio1990]_
     - Linear, 280 um per dva
   * - :py:class:`~pulse2percept.topography.retina.Watson2014Map`
     - [Watson2014]_
     - Nonlinear; the default for retinal models
   * - :py:class:`~pulse2percept.topography.retina.Montesano2020Map`
     - [Montesano2020]_
     - Watson2014 plus a 2D, meridian-dependent RGC displacement field

A regular grid in the visual field therefore lands differently on the retina
under each map:

.. plot::

    import matplotlib.pyplot as plt
    import pulse2percept as p2p

    grid = p2p.topography.Grid2D((-50, 50), (-50, 50), step=5)
    maps = [
        p2p.topography.retina.Curcio1990Map(),
        p2p.topography.retina.Watson2014Map(),
        p2p.topography.retina.Montesano2020Map(eye='right'),
    ]

    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
    for ax, vfmap in zip(axes, maps):
        grid.build(vfmap)
        grid.plot(style='cell', ax=ax)
        ax.set_title(type(vfmap).__name__, fontsize=9)
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        ax.axis('equal')
    fig.tight_layout()

:py:class:`~pulse2percept.topography.retina.Montesano2020Map` also
separates the position of a ganglion cell body from that of its receptive
field, which differ within about 15 dva of the fovea:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    import pulse2percept as p2p

    vfmap = p2p.topography.retina.Montesano2020Map(eye='right')
    plain = p2p.topography.retina.Watson2014Map()
    radius = np.linspace(0, 20, 400)

    fig, ax = plt.subplots(figsize=(6, 4))
    for angle, label in [(0, 'nasal'), (90, 'superior'),
                         (180, 'temporal'), (270, 'inferior')]:
        # The visual field mirrors the retina: in a right eye, an anatomical
        # meridian is at the negated visual-field polar angle.
        theta = np.deg2rad(-angle)
        x, y = radius * np.cos(theta), radius * np.sin(theta)
        displaced = np.hypot(*vfmap.dva_to_ret(x, y))
        ax.plot(radius, displaced - np.hypot(*plain.dva_to_ret(x, y)),
                label=label)
    ax.set_xlabel('receptive-field eccentricity (dva)')
    ax.set_ylabel('RGC displacement (microns, Watson2014 scale)')
    ax.legend(title='retinal meridian')
    fig.tight_layout()

*  The displacement zone reaches 14.1 dva temporally and superiorly,
   10.5 dva inferiorly, and 9.5 dva nasally.
*  Displacement is reconstructed from [Montesano2020]_ in dva, then converted
   to microns with ``Watson2014Map`` for consistency with the other retinal
   maps.
*  It is population-average anatomy ([Montesano2020]_, [Curcio1990]_), not
   subject-specific.
*  The deprecated ``Watson2014DisplaceMap`` uses horizontal-meridian fits
   across entire hemifields and has no inverse.

Laterality
~~~~~~~~~~

The visual field is mirrored on the retina. In a right eye, a retinal
meridian lies at the negated visual-field polar angle: nasal retina maps to
the temporal visual field. ``eye`` on a
:py:class:`~pulse2percept.implants.retina.RetinalImplant` sets where models
place the optic disc.
:py:class:`~pulse2percept.topography.retina.Montesano2020Map` takes its own
``eye``, because its displacement field depends on the meridian; the left-eye
map is the horizontal mirror of the right-eye map.

Cortical maps
~~~~~~~~~~~~~

Cortical maps derive from
:py:class:`~pulse2percept.topography.cortex.CorticalMap` and can cover ``v1``,
``v2`` and ``v3``.

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Map
     - Reference
     - Description
   * - :py:class:`~pulse2percept.topography.cortex.Polimeni2006Map`
     - [Polimeni2006]_
     - Wedge-dipole model of V1-V3; the default for cortical models
   * - :py:class:`~pulse2percept.topography.cortex.NeuropythyMap`
     - [Benson2018]_
     - Subject-specific retinotopy estimated from MRI

.. plot::

    import matplotlib.pyplot as plt
    import pulse2percept as p2p
    from pulse2percept.units import mm

    visual_field_map = p2p.topography.cortex.Polimeni2006Map(
        regions=['v1', 'v2', 'v3'])
    model = p2p.models.cortex.ScoreboardModel(
        implant=p2p.implants.cortex.Orion(),
        implant_position=(15, 0) * mm,
        visual_field_map=visual_field_map,
    )
    model.build()

    fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
    visual_field_map.plot(ax=axes[0])
    axes[0].set_title('Polimeni2006Map')
    model.plot(ax=axes[1])
    axes[1].set_title('Model grid')
    fig.tight_layout()

:py:class:`~pulse2percept.topography.cortex.Polimeni2006Map` has six
parameters: global scale ``k``, wedge-dipole parameters ``a`` and ``b``, and
azimuthal shear ``alpha1``, ``alpha2``, ``alpha3`` for V1-V3. Defaults are from
[Polimeni2006]_; individual retinotopy varies substantially.

:py:class:`~pulse2percept.topography.cortex.NeuropythyMap` gives
subject-specific retinotopy. It requires the optional ``neuropythy`` package
(``pip install neuropythy``) and a subject: ``'fsaverage'``, a Benson-Winawer
2018 subject (``'S1201'``-``'S1208'``), or a FreeSurfer subject directory.
Subject data is downloaded on first use. It also takes a cortical surface
(``'midgray'`` by default, ``'white'``, or ``'pial'``); its points are 3D, so
plot them with :py:meth:`~pulse2percept.topography.Grid2D.plot3d`.

.. code-block:: python

    nmap = p2p.topography.cortex.NeuropythyMap(subject='fsaverage',
                                               regions=['v1'])
    model = p2p.models.cortex.ScoreboardModel(
        implant=p2p.implants.cortex.Cortivis(),
        visual_field_map=nmap, regions=['v1'])

Subject-specific phosphene locations
------------------------------------

A map gives the canonical visual-field location of each electrode; measured
phosphenes scatter around it. ``location_noise`` (dva) adds a fixed random
offset per electrode:

.. math::

   \mathbf{p}'_i = \mathbf{p}_i + \boldsymbol{\epsilon}_i,
   \qquad
   \boldsymbol{\epsilon}_i \sim \mathcal{N}(0, \sigma^2 I),

where :math:`\mathbf{p}_i` is the canonical location and :math:`\sigma` is
``location_noise``. Offsets are drawn once per model instance.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    import pulse2percept as p2p

    implant = p2p.implants.retina.ArgusII(raster=None)
    stim = {e: 50 for e in ['A1', 'A3', 'A5', 'C1', 'C3', 'C5',
                            'E1', 'E3', 'E5']}

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True,
                             figsize=(9, 4))
    for ax, noise, title in zip(
            axes, [None, 1],
            ['Canonical locations', 'Subject-specific locations']):
        np.random.seed(1)
        model = p2p.models.retina.ScoreboardModel(
            implant=implant,
            xrange=(-12, 2), yrange=(-7, 7),
            visual_field_map=p2p.topography.retina.Curcio1990Map(),
            location_noise=noise,
        )
        model.predict_percept(stim).plot(ax=ax)
        ax.set_title(title)
    fig.tight_layout()

Retinal phosphenes move without changing shape. On cortex, a displaced
electrode samples a different cortical magnification, so its phosphene size
changes too. Electrode positions and the map are unchanged. ``location_noise``
requires an invertible map.

Visual field to scene: gaze
---------------------------

Scene coordinates are eye-centered coordinates plus gaze. Each electrode
follows this chain::

    device coordinate (um)
      -> implant_position / rotation / depth -> tissue coordinate (um)
      -> visual_field_map.ret_to_dva -> eye-centered visual field (dva)
      -> + gaze, for eye-coupled input only -> scene coordinate (dva)

See :ref:`topics-vision` for what gaze moves.

Custom maps
-----------

A custom retinal map subclasses
:py:class:`~pulse2percept.topography.retina.RetinalMap` and implements
``dva_to_ret``. ``location_noise`` and scenes also require ``ret_to_dva``.

.. code-block:: python

    class MyVisualFieldMap(p2p.topography.retina.RetinalMap):

        def dva_to_ret(self, xdva, ydva):
            return xdva, ydva

        def ret_to_dva(self, xret, yret):
            return xret, yret

Pass it to a model with ``visual_field_map=MyVisualFieldMap()``.
