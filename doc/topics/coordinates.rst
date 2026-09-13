.. _topics-coordinates:

=================================
Coordinates and Visual Field Maps
=================================

A simulation moves between four coordinate systems. Keeping them apart is what
lets device geometry, anatomy, and the visual world be specified independently.

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
     - Retinal or cortical position. The fovea for the retina; V1 anatomy for
       cortex.
   * - Visual field
     - dva
     - Eye-centered. The fovea is ``(0, 0)``; this is where phosphenes and a
       :py:class:`~pulse2percept.vision.Scotoma` live.
   * - Scene
     - dva
     - Fixed to the world in front of the eye; related to the visual field by
       gaze.

Degrees of visual angle are not a length, and a length is not an angle:
``dva`` does not convert to ``um`` without a
:py:class:`~pulse2percept.topography.VisualFieldMap` (see
:ref:`topics-units`).

Device to tissue: placement
---------------------------

Electrode coordinates are device-local, so placing a device is a model
parameter rather than a device attribute:

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

The same array can therefore be placed in different eyes, different subjects,
or different cortical positions without rebuilding the device.

Tissue to visual field: the maps
--------------------------------

A :py:class:`~pulse2percept.topography.VisualFieldMap` converts between visual
field and tissue. Every map provides ``from_dva`` and, where the mapping is
invertible, ``to_dva``; retinal maps expose them as ``dva_to_ret`` and
``ret_to_dva``, and cortical maps as ``dva_to_v1`` / ``v1_to_dva`` and the
corresponding V2 and V3 pairs.

.. code-block:: python

    from pulse2percept.topography.retina import Watson2014Map

    x_um, y_um = Watson2014Map().dva_to_ret(2 * dva, 3 * dva)

A model holds one map, samples the visual field on a
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

:py:class:`~pulse2percept.topography.retina.Montesano2020Map` additionally
separates where a ganglion cell's *body* sits from where its receptive field
looks, which matters within roughly 15 dva of the fovea:

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

The displacement zone reaches 14.1 dva temporally and superiorly, 10.5 dva
inferiorly, and 9.5 dva nasally. Displacement is reconstructed from
[Montesano2020]_ in degrees of visual angle, then converted to retinal microns
with ``Watson2014Map``, which keeps the tissue coordinates consistent with the
other retinal maps. The field represents population-average anatomy from
[Montesano2020]_ and [Curcio1990]_, not subject-specific retinal anatomy. The
deprecated ``Watson2014DisplaceMap`` uses horizontal-meridian fits across
entire hemifields and provides no reverse mapping.

Laterality
~~~~~~~~~~

The visual field mirrors the retina, so anatomical laterality has to be
stated. ``eye`` on a
:py:class:`~pulse2percept.implants.retina.RetinalImplant` records the
implanted eye; models read it to place the optic disc, and
:py:class:`~pulse2percept.topography.retina.Montesano2020Map` takes its own
``eye`` because its displacement field is meridian-dependent. The left-eye map
is the horizontal mirror of the right-eye map.

In a right eye, a retinal meridian sits at the negated visual-field polar
angle: nasal retina is the temporal visual field.

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
parameters: the global scale ``k``, wedge-dipole parameters ``a`` and ``b``,
and azimuthal shear parameters ``alpha1``, ``alpha2`` and ``alpha3`` for
V1-V3. The defaults come from [Polimeni2006]_, but cortical retinotopy varies
substantially across people.

:py:class:`~pulse2percept.topography.cortex.NeuropythyMap` provides an
individualized mapping where subject anatomy is available. It requires the
optional ``neuropythy`` package (``pip install neuropythy``) and a subject:
``'fsaverage'``, a subject from the Benson-Winawer 2018 dataset
(``'S1201'``-``'S1208'``), or a FreeSurfer subject directory. Unless the data
is already cached, the first use downloads it. Unlike the other maps it also
takes a cortical surface (``'midgray'`` by default, or ``'white'``,
``'pial'``), and its points are genuinely three-dimensional, so
:py:meth:`~pulse2percept.topography.Grid2D.plot3d` is the honest way to
look at them.

.. code-block:: python

    nmap = p2p.topography.cortex.NeuropythyMap(subject='fsaverage',
                                               regions=['v1'])
    model = p2p.models.cortex.ScoreboardModel(
        implant=p2p.implants.cortex.Cortivis(),
        visual_field_map=nmap, regions=['v1'])

Subject-specific phosphene locations
------------------------------------

A map describes a canonical relationship between tissue and visual-field
location. An individual phosphene may appear somewhere else. ``location_noise``
models that variability without changing the underlying ``visual_field_map``.

For electrode :math:`i`, pulse2percept draws a fixed visual-field offset:

.. math::

   \mathbf{p}'_i = \mathbf{p}_i + \boldsymbol{\epsilon}_i,
   \qquad
   \boldsymbol{\epsilon}_i \sim \mathcal{N}(0, \sigma^2 I),

where :math:`\mathbf{p}_i` is the canonical phosphene location and
:math:`\sigma` is ``location_noise`` in dva. The offsets remain fixed for a
model instance.

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

The phosphenes move; they do not change shape. On cortex the same is true of
coherence but not of size: a displaced electrode samples a different cortical
magnification, so its phosphene covers a different extent of the visual field.

``location_noise`` changes the predicted percept, not the physical electrode
locations or the canonical visual-field map. It requires an invertible map, so
one without an inverse (such as the deprecated ``Watson2014DisplaceMap``) is
not supported.

Visual field to scene: gaze
---------------------------

Visual-field coordinates are eye-centered and scene coordinates are fixed to
the world, so the two differ by gaze::

    (x_scene, y_scene) = (x_eye, y_eye) + (x_gaze, y_gaze)

Each electrode therefore follows this chain::

    device coordinate (um)
      -> implant_position / rotation / depth -> tissue coordinate (um)
      -> visual_field_map.ret_to_dva -> eye-centered visual field (dva)
      -> + gaze, for eye-coupled input only -> scene coordinate (dva)

What ``gaze`` does and does not move is covered in :ref:`topics-vision`.

Custom maps
-----------

A custom retinal map subclasses
:py:class:`~pulse2percept.topography.retina.RetinalMap` and implements
``dva_to_ret``. Provide ``ret_to_dva`` as well when the mapping can be
inverted; features such as ``location_noise`` and scene registration require
it.

.. code-block:: python

    class MyVisualFieldMap(p2p.topography.retina.RetinalMap):

        def dva_to_ret(self, xdva, ydva):
            return xdva, ydva

        def ret_to_dva(self, xret, yret):
            return xret, yret

Pass it to a model with ``visual_field_map=MyVisualFieldMap()``.
