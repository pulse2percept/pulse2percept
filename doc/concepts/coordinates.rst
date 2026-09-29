.. _topics-coordinates:

==========================
Retinotopy and Coordinates
==========================

pulse2percept maps stimulation from physical electrodes into visual space.
Retinal and cortical maps describe how positions in the visual field correspond
to locations in neural tissue, while implant placement determines where a
device sits within that tissue.

A regular grid in the visual field becomes highly nonuniform when mapped onto
the retina:

.. plot::

    import matplotlib.pyplot as plt
    import pulse2percept as p2p

    maps = [
        ('Curcio 1990',
         p2p.topography.retina.Curcio1990Map()),
        ('Watson 2014',
         p2p.topography.retina.Watson2014Map()),
        ('Montesano 2020',
         p2p.topography.retina.Montesano2020Map(eye='right')),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, (title, vfmap) in zip(axes, maps):
        grid = p2p.topography.Grid2D(
            (-50, 50),
            (-50, 50),
            step=5,
        )
        grid.build(vfmap)
        grid.plot(style='cell', ax=ax)
        ax.set_title(title)
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        ax.axis('equal')

    fig.tight_layout()

These transformations determine where an implanted electrode appears in the
visual field and, for location-dependent models, can also affect the predicted
phosphene shape and size.


Visual-Field Maps
=================

A :py:class:`~pulse2percept.topography.VisualFieldMap` converts between
eye-centered visual-field coordinates and physical tissue coordinates.

Retinal maps provide:

.. code-block:: text

    dva_to_ret
    ret_to_dva

Cortical maps provide the corresponding transformations for visual areas:

.. code-block:: text

    dva_to_v1       v1_to_dva
    dva_to_v2       v2_to_dva
    dva_to_v3       v3_to_dva

Inverse mappings are available only where the transformation is invertible.

For example:

.. code-block:: python

    from pulse2percept.topography.retina import Watson2014Map
    from pulse2percept.units import dva

    vfmap = Watson2014Map()

    x_um, y_um = vfmap.dva_to_ret(
        2 * dva,
        3 * dva,
    )

    x_dva, y_dva = vfmap.ret_to_dva(
        x_um,
        y_um,
    )

Each spatial model owns a ``visual_field_map``. During ``build()``, the model
samples its visual field on a
:py:class:`~pulse2percept.topography.Grid2D` and maps that grid onto the
relevant tissue.


Retinal Retinotopy
==================

Retinal maps derive from
:py:class:`~pulse2percept.topography.retina.RetinalMap`.

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Map
     - Reference
     - Retinal Magnification
   * - :py:class:`~pulse2percept.topography.retina.Curcio1990Map`
     - [Curcio1990]_
     - Linear, 280 um per dva
   * - :py:class:`~pulse2percept.topography.retina.Watson2014Map`
     - [Watson2014]_
     - Nonlinear; default for retinal models
   * - :py:class:`~pulse2percept.topography.retina.Montesano2020Map`
     - [Montesano2020]_
     - Watson2014 magnification plus meridian-dependent RGC displacement

:py:class:`~pulse2percept.topography.retina.Watson2014Map` is the default for
retinal models. It captures the nonlinear relationship between retinal
distance and visual eccentricity.

:py:class:`~pulse2percept.topography.retina.Montesano2020Map` additionally
accounts for displacement between retinal ganglion cell bodies and the
visual-field locations of their receptive fields.


RGC Displacement
----------------

Near the fovea, a retinal ganglion cell body can lie substantially farther from
the fovea than the photoreceptors supplying its receptive field.

The displacement depends on retinal meridian:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    import pulse2percept as p2p

    vfmap = p2p.topography.retina.Montesano2020Map(
        eye='right'
    )
    plain = p2p.topography.retina.Watson2014Map()

    radius = np.linspace(0, 20, 400)

    fig, ax = plt.subplots(figsize=(6, 4))

    for angle, label in [
        (0, 'nasal'),
        (90, 'superior'),
        (180, 'temporal'),
        (270, 'inferior'),
    ]:
        # Visual field and retina are mirrored. In a right eye, the
        # anatomical retinal meridian lies at the negative visual-field angle.
        theta = np.deg2rad(-angle)

        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        displaced = np.hypot(
            *vfmap.dva_to_ret(x, y)
        )
        undisplaced = np.hypot(
            *plain.dva_to_ret(x, y)
        )

        ax.plot(
            radius,
            displaced - undisplaced,
            label=label,
        )

    ax.set_xlabel('receptive-field eccentricity (dva)')
    ax.set_ylabel('RGC displacement (microns, Watson2014 scale)')
    ax.legend(title='retinal meridian')

    fig.tight_layout()

.. note::

   The displacement zone extends to approximately 14.1 dva temporally and
   superiorly, 10.5 dva inferiorly, and 9.5 dva nasally.

   Displacement is reconstructed from [Montesano2020]_ in dva and converted to
   microns using
   :py:class:`~pulse2percept.topography.retina.Watson2014Map`, for consistency
   with the other retinal maps.

   These maps describe population-average anatomy, not subject-specific retinal
   geometry.

   ``Watson2014DisplaceMap`` is deprecated. It uses horizontal-meridian fits
   across entire hemifields and has no inverse mapping.


Retinal Laterality
------------------

The visual field is mirrored on the retina.

For the right eye, retinal polar angle is the negative of the corresponding
visual-field angle. Nasal retina therefore represents temporal visual field,
and vice versa.

The ``eye`` attribute of
:py:class:`~pulse2percept.implants.retina.RetinalImplant` determines where
retinal models place structures such as the optic disc.

:py:class:`~pulse2percept.topography.retina.Montesano2020Map` also takes an
``eye`` argument because its RGC displacement field depends on retinal
meridian. The left-eye field is the horizontal mirror of the right-eye field.


Cortical Retinotopy
===================

Cortical maps transform the visual field onto the surface of visual cortex.

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Map
     - Reference
     - Description
   * - :py:class:`~pulse2percept.topography.cortex.Polimeni2006Map`
     - [Polimeni2006]_
     - Analytic wedge-dipole model of V1-V3; default for cortical models
   * - :py:class:`~pulse2percept.topography.cortex.NeuropythyMap`
     - [Benson2018]_
     - Subject-specific retinotopy estimated from cortical anatomy

Cortical maps derive from
:py:class:`~pulse2percept.topography.cortex.CorticalMap`.


Polimeni2006Map
---------------

:py:class:`~pulse2percept.topography.cortex.Polimeni2006Map` provides an
analytic model of visual-field organization across V1, V2, and V3:

.. plot::

    import matplotlib.pyplot as plt
    import pulse2percept as p2p

    from pulse2percept.units import mm

    visual_field_map = p2p.topography.cortex.Polimeni2006Map(
        regions=['v1', 'v2', 'v3'],
    )

    model = p2p.models.cortex.ScoreboardModel(
        implant=p2p.implants.cortex.Orion(),
        implant_position=(15, 0) * mm,
        visual_field_map=visual_field_map,
    )

    model.build()

    fig, axes = plt.subplots(
        ncols=2,
        figsize=(10, 4),
    )

    visual_field_map.plot(ax=axes[0])
    axes[0].set_title('Visual-field map')

    model.plot(ax=axes[1])
    axes[1].set_title('Model grid')

    fig.tight_layout()

The map has six main parameters:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Parameter
     - Meaning
   * - ``k``
     - Global cortical scale
   * - ``a``, ``b``
     - Wedge-dipole parameters
   * - ``alpha1``
     - Azimuthal shear for V1
   * - ``alpha2``
     - Azimuthal shear for V2
   * - ``alpha3``
     - Azimuthal shear for V3

Defaults follow [Polimeni2006]_. Individual human retinotopy can differ
substantially from this population-level analytic map.


NeuropythyMap
-------------

:py:class:`~pulse2percept.topography.cortex.NeuropythyMap` provides
subject-specific cortical retinotopy based on cortical surface anatomy.

It requires the optional ``neuropythy`` package:

.. code-block:: bash

    pip install neuropythy

Supported subjects include:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Subject
     - Source
   * - ``'fsaverage'``
     - FreeSurfer average anatomy
   * - ``'S1201'`` - ``'S1208'``
     - Benson-Winawer 2018 subjects
   * - FreeSurfer directory
     - User-supplied subject anatomy

Subject data are downloaded on first use.

The map also accepts a cortical surface: ``'midgray'`` by default, or
``'white'`` / ``'pial'``.

Coordinates are three-dimensional, so use
:py:meth:`~pulse2percept.topography.Grid2D.plot3d` for visualization.

.. code-block:: python

    nmap = p2p.topography.cortex.NeuropythyMap(
        subject='fsaverage',
        regions=['v1'],
    )

    model = p2p.models.cortex.ScoreboardModel(
        implant=p2p.implants.cortex.NeuroPortArray(),
        visual_field_map=nmap,
        regions=['v1'],
    )


Implant Placement
=================

Electrode coordinates are stored relative to the implant itself. A spatial
model places that geometry onto retinal or cortical tissue.

For example:

.. code-block:: python

    import pulse2percept as p2p

    from pulse2percept.units import dva

    implant = p2p.implants.retina.ArgusII()

    model = p2p.models.retina.AxonMapModel(
        implant,
        implant_position=(2, -1) * dva,
        implant_rotation=15,
    )

The same device geometry can therefore be translated, rotated, or placed at a
different depth without modifying the implant object.


Placement Parameters
--------------------

.. list-table::
   :header-rows: 1
   :widths: 26 48 26

   * - Parameter
     - Meaning
     - Units
   * - ``implant_position``
     - Position of the device-local origin in tissue
     - tissue length or dva
   * - ``implant_rotation``
     - In-plane rotation, positive counterclockwise
     - deg
   * - ``implant_depth``
     - Signed displacement along the normal of a 2D tissue map
     - um

Retinal positions are measured relative to the fovea.

Cortical positions are measured relative to the foveal representation of the
right hemisphere. The left hemisphere is translated along x by
``left_offset`` (default -20 mm).


Subject-Specific Phosphene Locations
====================================

A visual-field map gives the canonical visual-field location associated with
each electrode. Measured phosphene locations often differ from those
predictions.

``location_noise`` adds one fixed random offset per electrode:

.. math::

    \mathbf{p}'_i
    =
    \mathbf{p}_i + \boldsymbol{\epsilon}_i,

    \qquad

    \boldsymbol{\epsilon}_i
    \sim
    \mathcal{N}(0, \sigma^2 I),

where :math:`\mathbf{p}_i` is the canonical phosphene location and
:math:`\sigma` is ``location_noise`` in dva.

Offsets are sampled once for each model instance:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    import pulse2percept as p2p

    implant = p2p.implants.retina.ArgusII(
        raster=None
    )

    stim = {
        e: 50
        for e in [
            'A1', 'A3', 'A5',
            'C1', 'C3', 'C5',
            'E1', 'E3', 'E5',
        ]
    }

    fig, axes = plt.subplots(
        ncols=2,
        sharex=True,
        sharey=True,
        figsize=(9, 4),
    )

    for ax, noise, title in zip(
        axes,
        [None, 1],
        ['Canonical locations', 'Subject-specific locations'],
    ):
        np.random.seed(1)

        model = p2p.models.retina.ScoreboardModel(
            implant=implant,
            xrange=(-12, 2),
            yrange=(-7, 7),
            visual_field_map=p2p.topography.retina.Curcio1990Map(),
            location_noise=noise,
        )

        model.predict_percept(stim).plot(ax=ax)
        ax.set_title(title)

    fig.tight_layout()

For retinal models, location noise shifts a phosphene without changing its
shape.

For cortical models, moving an electrode to a different retinotopic location
also changes the local cortical magnification, so phosphene size may change.

``location_noise`` does not alter the electrode coordinates or
``visual_field_map`` and requires an invertible mapping.


Coordinate Systems
==================

A stimulation site can pass through four coordinate systems:

.. code-block:: text

    device coordinates
            |
            | implant placement
            v
    tissue coordinates
            |
            | visual_field_map
            v
    visual-field coordinates
            |
            | gaze
            v
    scene coordinates

.. list-table::
   :header-rows: 1
   :widths: 20 14 66

   * - Frame
     - Unit
     - Meaning
   * - Device
     - um
     - Coordinates local to the hardware. The geometry is unchanged when the
       same implant is placed elsewhere.
   * - Tissue
     - um
     - Retinal or cortical position relative to the fovea or its cortical
       representation.
   * - Visual Field
     - dva
     - Eye-centered coordinates. The fovea is ``(0, 0)``. Predicted
       phosphenes and :py:class:`~pulse2percept.vision.Scotoma` objects live
       here.
   * - Scene
     - dva
     - Fixed world coordinates. Gaze relates these to eye-centered visual-field
       coordinates.

``dva`` and ``um`` are different physical dimensions. Conversion between them
requires a :py:class:`~pulse2percept.topography.VisualFieldMap`; see
:ref:`topics-units`.


Visual Field and Scene Coordinates
==================================

Visual-field coordinates are centered on the eye. Scene coordinates are fixed
to the visual world.

Gaze relates the two:

.. code-block:: text

    device coordinate (um)
            |
            | implant_position
            | implant_rotation
            | implant_depth
            v
    tissue coordinate (um)
            |
            | visual_field_map
            v
    eye-centered visual field (dva)
            |
            | + gaze
            v
    scene coordinate (dva)

For an eye-centered coordinate :math:`\mathbf{x}` and gaze
:math:`\mathbf{g}`,

.. math::

    \mathbf{x}_{scene}
    =
    \mathbf{x}_{eye}
    +
    \mathbf{g}.

The visual-field location of an electrode remains fixed relative to the eye.
Changing gaze changes which part of the world occupies that location.

See :ref:`topics-vision` for scenes, gaze, and the distinction between
eye-coupled and head-mounted device input.


Custom Maps
===========

Custom retinal retinotopy can be implemented by subclassing
:py:class:`~pulse2percept.topography.retina.RetinalMap`.

At minimum, implement ``dva_to_ret``:

.. code-block:: python

    class MyVisualFieldMap(
            p2p.topography.retina.RetinalMap):

        def dva_to_ret(self, xdva, ydva):
            return xdva, ydva

Implement ``ret_to_dva`` as well when inverse mapping is needed, including for
scene registration and ``location_noise``:

.. code-block:: python

    class MyVisualFieldMap(
            p2p.topography.retina.RetinalMap):

        def dva_to_ret(self, xdva, ydva):
            return xdva, ydva

        def ret_to_dva(self, xret, yret):
            return xret, yret

Pass the map to a model through ``visual_field_map``:

.. code-block:: python

    model = p2p.models.retina.ScoreboardModel(
        implant=implant,
        visual_field_map=MyVisualFieldMap(),
    )