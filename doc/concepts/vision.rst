.. _topics-vision:

=================================
Visual Input and Simulated Vision
=================================

.. versionadded:: 0.11.0

Visual input can enter a simulation in two ways:

.. list-table::
   :header-rows: 1
   :widths: 26 38 36

   * - Input
     - Coordinates
     - Behavior
   * - :py:class:`~pulse2percept.stimuli.ImageStimulus` or
       :py:class:`~pulse2percept.stimuli.VideoStimulus`
     - Device-relative
     - Stretched across the implant's electrodes; has no size or position in
       the visual field
   * - :py:class:`~pulse2percept.vision.Scene`
     - Visual-field-relative
     - Places the source in visual space so each electrode samples its own
       visual-field location

The distinction matters. The same image, implant, and model can produce
different percepts:

.. plot::

    import matplotlib.pyplot as plt
    import pulse2percept as p2p

    from pulse2percept.units import dva

    image = p2p.stimuli.samples.logo_bvl(resize=(240, 300))
    implant = p2p.implants.retina.ArgusII()
    model = p2p.models.retina.ScoreboardModel(
        implant=implant,
        rho=200,
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    model.predict_percept(image).plot(ax=axes[0])
    axes[0].set_title('ImageStimulus: stretched over the array')

    scene = p2p.vision.Scene(image, fov=40 * dva)
    model.predict_percept(
        scene,
        gaze=(0, 0) * dva,
    ).plot(ax=axes[1])
    axes[1].set_title('Scene: 40 dva of visual field')

    fig.tight_layout()

On the left, the image fills the implant. On the right, the implant samples
only the part of a 40 dva visual field that falls at its electrode locations.


Scenes
======

A :py:class:`~pulse2percept.vision.Scene` places visual content in world
coordinates and views it through an eye-centered field of view.


Quick Start
-----------

.. code-block:: python

    import pulse2percept as p2p

    from pulse2percept.units import dva

    scene = p2p.vision.Scene(
        p2p.stimuli.samples.logo_bvl(),
        fov=40 * dva,
    )

    implant = p2p.implants.retina.ArgusII()
    model = p2p.models.retina.ScoreboardModel(
        implant=implant,
        rho=200,
    )

    percept = model.predict_percept(
        scene,
        gaze=(0, 0) * dva,
    )


Scene Geometry
--------------

Scenes use two coordinate frames related by gaze:

.. math::

    \mathrm{scene}_{xy} = \mathrm{eye}_{xy} + \mathrm{gaze}_{xy}.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Frame
     - Meaning
   * - Scene coordinates
     - Fixed world coordinates. The source is placed here by ``extent``.
   * - Eye coordinates
     - Coordinates centered on the fovea. ``fov``, the aperture, scotoma,
       prosthetic percept, and visual-field grid live here.

The main scene parameters are:

.. list-table::
   :header-rows: 1
   :widths: 20 52 28

   * - Parameter
     - Meaning
     - Units / form
   * - ``extent``
     - Bounds of the source in scene coordinates
     - ``(left, right, bottom, top)`` in dva, or a scalar
   * - ``fov``
     - Eye-centered field of view
     - Scalar for square FOV, or ``(width, height)``
   * - ``source``
     - Image or video placed in the scene
     - Image or video stimulus
   * - ``scotoma``
     - Region of lost native vision
     - :py:class:`~pulse2percept.vision.Scotoma`
   * - ``aperture``
     - Shape of the visible field
     - ``'rectangular'`` or ``'round'``

A scalar ``extent`` sets the span of the source's shorter dimension, centered
at the origin, while preserving square pixels. For example, ``45 * dva`` on a
173 x 320 source produces an extent of approximately 83.2 x 45 dva.

If ``extent`` is omitted, the source is centered with square pixels using the
smallest extent that contains ``fov``. ``extent`` does not need to contain the
entire field of view.

Pixel coordinates refer to pixel centers. Row 0 is the top of the image and
therefore has the largest ``y`` coordinate.

A scene is immutable prediction input: its source, ``extent``, and ``fov`` are
fixed after construction and are not stored on the model or implant.


How the Pieces Fit Together
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Object
     - Role
   * - Scene
     - Describes what is visually present and where native vision is lost
   * - Implant
     - Describes device geometry and encoding constraints
   * - Model
     - Supplies retinotopy and maps implant electrodes into visual space
   * - Percept
     - Stores the predicted visual output

Scene registration is therefore a spatial-model capability. Only retinal
spatial models
(:py:class:`~pulse2percept.models.retina.RetinalSpatial`) currently accept
scenes because they provide the required retinotopic mapping.

Other models raise ``NotImplementedError``. A retinal model with a
non-retinotopic ``visual_field_map``, or an implant without an ``encoder``,
raises ``ValueError``.


Blank Scenes
------------

:py:meth:`~pulse2percept.vision.Scene.blank` creates a black visual-field
canvas when no source image is needed:

.. code-block:: python

    scene = p2p.vision.Scene.blank(fov=45 * dva)

The default 512 x 512 raster is used only when rendering. Model prediction
still uses the model's own grid.

Black is visual content, not blindness. Use a
:py:class:`~pulse2percept.vision.Scotoma` to represent lost vision.


Gaze
====

``gaze`` specifies the scene location currently falling on the fovea.

A single ``(x, y)`` pair gives a fixed gaze position. For time-varying gaze,
:py:class:`~pulse2percept.vision.Gaze` stores fixation locations and onset
times; each fixation is held until the next begins.

.. code-block:: python

    from pulse2percept.units import dva, ms

    scene = p2p.vision.Scene(
        p2p.stimuli.samples.ucsb_flyover(),
        fov=45 * dva,
        scotoma=p2p.vision.Scotoma.circle(5 * dva),
        scotoma_fill=0.5,
    )

    gaze = p2p.vision.Gaze(
        [(0, 0), (6, 2), (-4, 3), (5, -3)] * dva,
        time=[0, 400, 900, 1400] * ms,
    )

    scene.play(gaze=gaze)
    percept = model.predict_percept(scene, gaze=gaze)


Scene and Eye Views
-------------------

``plot`` and ``play`` provide two views of the same scene. The view affects
visualization only; prediction and sampling are unchanged.

.. code-block:: python

    video = p2p.stimuli.samples.ucsb_pedestrians(
        resize=(173, 320)
    )

    scene = p2p.vision.Scene(
        video,
        extent=45 * dva,
        fov=40 * dva,
        scotoma=p2p.vision.Scotoma.circle(5 * dva),
        aperture='round',
    )

    gaze = p2p.vision.Gaze(
        [(0, 0), (-15.5, -6), (12, -6)] * dva,
        time=[0, 635, 1370] * ms,
    )

    scene.play(gaze=gaze)
    scene.play(gaze=gaze, view='eye')

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - View
     - Display
   * - ``'scene'``
     - Fixed world coordinates spanning ``extent``. The source remains fixed
       while the FOV, scotoma, percept, and visual-field rings move with gaze.
   * - ``'eye'``
     - Fixed eye coordinates spanning ``[-fov / 2, fov / 2]``. The source
       moves through the eye-centered window with gaze.

``'scene'`` is the default. Source content outside the FOV is dimmed by
``context_alpha`` (default 0.25). Set it to 0 for black context or 1 for an
undimmed source.

:py:meth:`~pulse2percept.vision.Scene.render` is always eye-centered.


Gaze Timing
-----------

Times are interpreted as ms unless units are supplied.

A fixation beginning exactly on a frame time applies to that frame. For video
scenes, gaze is matched to the video's frame times rather than ``t_percept``.
For a still scene rendered together with a timed percept, the percept times are
used.


Eye- and Head-Centered Input
----------------------------

Gaze always determines where the prosthetic percept appears in scene
coordinates. Whether gaze also changes what reaches the electrodes depends on
:py:attr:`~pulse2percept.implants.Implant.scene_input_frame`.

.. list-table::
   :header-rows: 1
   :widths: 18 34 48

   * - Value
     - Typical device
     - Behavior
   * - ``'eye'``
     - Alpha, PRIMA
     - Input passes through the eye's optics. Gaze moves the scene across the
       implant. This is the default.
   * - ``'head'``
     - Argus, BVT, IMIE
     - Input comes from a head-mounted camera. Gaze moves the percept but does
       not change the electrode input.

The implant and scotoma remain fixed in eye coordinates as gaze changes.

For ``'head'`` systems, each electrode samples its own visual-field position;
patient-specific camera-to-electrode registration is not modeled.

To simulate an eye-tracked camera system, set:

.. code-block:: python

    implant.scene_input_frame = 'eye'


Device Preprocessing
====================

``implant.preprocess`` transforms the prosthetic input before it is sampled at
the electrodes. Native vision continues to use the original scene.

For example:

.. code-block:: python

    implant.preprocess = lambda stim: stim.filter('sobel')

For scenes, preprocessing occurs at the source raster resolution and must
return an :py:class:`~pulse2percept.stimuli.ImageStimulus` or
:py:class:`~pulse2percept.stimuli.VideoStimulus` with the same spatial shape
and frame times.

Pixel values and channels may change. Geometry and timing may not, because
``extent`` and the scene frame clock refer to the original source.


Residual Vision
===============

A :py:class:`~pulse2percept.vision.Scotoma` marks regions in eye-centered
visual space where native vision is lost.

.. code-block:: python

    scotoma = p2p.vision.Scotoma.circle(8 * dva)

    scotoma = p2p.vision.Scotoma.ellipse(
        5 * dva,
        4 * dva,
        center=(-9, 4) * dva,
    )

The scotoma does not affect ``predict_percept``. Prosthetic encoding samples
the original unmasked scene, and the model still predicts brightness over its
full grid.

Native and prosthetic vision are combined only for visualization or rendering.


Combining Native and Prosthetic Vision
--------------------------------------

.. plot::

    import matplotlib.pyplot as plt
    import pulse2percept as p2p

    from pulse2percept.units import dva

    center = (-9, 4) * dva

    image = p2p.stimuli.samples.logo_bvl(
        resize=(240, 300)
    )

    scotoma = p2p.vision.Scotoma.ellipse(
        5 * dva,
        4 * dva,
        center=center,
    )

    scene = p2p.vision.Scene(
        image,
        fov=40 * dva,
        scotoma=scotoma,
        background=1,
    )

    implant = p2p.implants.retina.PRIMAPivotal()

    model = p2p.models.retina.ScoreboardModel(
        implant,
        implant_position=center,
        rho=50,
        xrange=(-15, -3),
        yrange=(-2, 10),
        step=0.1,
    )

    percept = model.predict_percept(
        scene,
        gaze=(0, 0) * dva,
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    scene.plot(
        ax=axes[0],
        rings=True,
    )
    axes[0].set_title('Residual vision with a scotoma')

    scene.plot(
        percept=percept,
        gaze=(0, 0) * dva,
        vmax=2,
        ax=axes[1],
    )
    axes[1].set_title('Prosthetic percept inside the loss')

    fig.tight_layout()

Within the region of visual loss, the displayed value is

.. math::

    (1 - L)N + L\max(S, P),

where :math:`L` is scotoma loss, :math:`N` is native vision,
:math:`S` is ``scotoma_fill``, and :math:`P` is the prosthetic percept.

Without a scotoma, the prosthetic percept is displayed on black rather than
over intact vision. Overlaying it on intact vision would imply an interaction
that the model does not represent.


Plotting and Rendering
----------------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Method
     - Behavior
   * - ``scene.plot()``
     - Draws the source at its own resolution and the percept on the model
       grid; neither is resampled
   * - ``scene.play()``
     - Animated version of the same display
   * - ``scene.render()``
     - Combines the result into a single eye-centered RGB
       :py:class:`~pulse2percept.percepts.Percept`

Model brightness is in arbitrary units, so ``vmax`` is required when a
prosthetic percept is composited with native vision. Keep it fixed when
comparing conditions.

.. code-block:: python

    rgb = scene.render(
        percept=percept,
        gaze=(0, 0) * dva,
        vmax=50,
    )

    fine = scene.render(
        percept=percept,
        vmax=50,
        step=0.02 * dva,
    )

``render`` uses the source raster by default. Set either ``step`` (dva per
pixel) or ``shape`` to choose another output raster; the two are mutually
exclusive.

When ``step`` is supplied, it is rounded so that pixels are never coarser than
requested. Fine sampling over a wide field can be expensive.


Field Aperture
==============

``aperture`` sets the shape of the visible field within ``fov``.

.. code-block:: python

    disc = p2p.vision.Scene(
        image,
        fov=40 * dva,
        aperture='round',
    )

    wide = p2p.vision.Scene(
        image,
        fov=(60, 40) * dva,
        aperture='round',
    )

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Value
     - Behavior
   * - ``'rectangular'``
     - Uses the complete rectangular field of view; the default
   * - ``'round'``
     - Uses the ellipse inscribed within the field of view

For ``fov=(60, 40) * dva``, the round aperture extends 30 dva horizontally and
20 dva vertically.

The aperture is eye-centered, like the scotoma, and affects display only.
``plot`` and ``play`` clip the full-intensity field, percept, and grid to the
aperture while retaining the dimmed scene-view context outside it.
``render`` writes black outside the aperture.

Scene sampling and stimulation are unchanged.


Both Eyes
=========

:py:class:`~pulse2percept.vision.BinocularScene` groups one
:py:class:`~pulse2percept.vision.Scene` for each eye.

.. code-block:: python

    binocular = p2p.vision.BinocularScene(
        left=p2p.vision.Scene(
            image,
            fov=40 * dva,
            scotoma=scotoma,
        ),
        right=p2p.vision.Scene(
            image,
            fov=40 * dva,
        ),
    )

    ax_left, ax_right = binocular.plot(
        left_percept=percept,
        vmax=2,
    )

Models remain monocular. Predict one eye at a time:

.. code-block:: python

    left_percept = model.predict_percept(binocular.left)

A ``Scene`` has no ``eye`` attribute. Its position inside the
``BinocularScene`` identifies the eye.


Creating Binocular Scenes
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Method
     - Behavior
   * - :py:meth:`~pulse2percept.vision.BinocularScene.from_side_by_side`
     - Splits left-right packed imagery into two scenes. ``fov`` describes one
       half. The images are not flipped or resampled, and no disparity or
       depth is inferred.
   * - :py:meth:`~pulse2percept.vision.Scene.fellow_eye`
     - Creates the fellow-eye scene and mirrors eye-specific geometry such as
       a scotoma across the vertical meridian. The source image itself is not
       flipped.
   * - :py:meth:`~pulse2percept.vision.Scotoma.mirror`
     - Mirrors a scotoma alone, such that
       ``mirrored(x, y) == original(-x, y)``.

For example:

.. code-block:: python

    left = p2p.vision.Scene(
        image,
        fov=40 * dva,
        scotoma=scotoma,
    )

    binocular = p2p.vision.BinocularScene(
        left,
        left.fellow_eye(),
    )

    stereo = p2p.vision.BinocularScene.from_side_by_side(
        stereo_image,
        fov=(60, 40) * dva,
    )