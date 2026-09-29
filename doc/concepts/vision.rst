.. _topics-vision:

=============
Visual Scenes
=============

.. versionadded:: 0.11.0

A :py:class:`~pulse2percept.vision.Scene` places images and videos in the
visual field. It provides a common space for gaze, residual vision, and
prosthetic percepts.

For example, start with a natural scene, add a central scotoma, and move gaze
across the image:

.. code-block:: python

    import numpy as np
    import pulse2percept as p2p

    from pulse2percept.units import dva, ms

    image = p2p.stimuli.samples.ucsb_bike(resize=(240, 360))

    # Give the still image a short time axis so gaze can be animated:
    frame = image.data.reshape(image.img_shape)
    time = np.arange(0, 2400, 40)
    video = p2p.stimuli.VideoStimulus(
        np.repeat(frame[..., np.newaxis], len(time), axis=-1),
        time=time * ms,
    )

    scene = p2p.vision.Scene(
        video,
        extent=60 * dva,
        fov=40 * dva,
        scotoma=p2p.vision.Scotoma.circle(5 * dva),
        scotoma_fill='inpaint',
    )

    gaze = p2p.vision.Gaze(
        [(0, 0), (-12, 0), (12, 0), (-8, 2)] * dva,
        time=[0, 600, 1200, 1800] * ms,
    )

    scene.play(gaze=gaze)

The source remains fixed in the world while the eye-centered field of view and
scotoma move with gaze:

.. plot::

    import matplotlib.pyplot as plt
    import pulse2percept as p2p

    from pulse2percept.units import dva

    image = p2p.stimuli.samples.ucsb_bike(resize=(240, 360))

    scene = p2p.vision.Scene(
        image,
        extent=60 * dva,
        fov=45 * dva,
        aperture='round',
        scotoma=p2p.vision.Scotoma.circle(10 * dva),
        scotoma_fill='inpaint',
    )

    fixations = [
        ('Looking left', (-12, 0) * dva),
        ('Straight ahead', (0, 0) * dva),
        ('Looking right', (12, 0) * dva),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, (title, gaze) in zip(axes, fixations):
        scene.plot(
            gaze=gaze,
            ax=ax,
            context_alpha=0.15,
        )
        ax.set_title(title)

    fig.tight_layout()


Scene Geometry
==============

A scene separates the location of the visual world from the location of the
eye.

Scene coordinates are fixed to the world. Eye coordinates are centered on the
fovea. Gaze relates the two:

.. math::

    \mathrm{scene}_{xy} = \mathrm{eye}_{xy} + \mathrm{gaze}_{xy}.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Frame
     - Meaning
   * - Scene Coordinates
     - Fixed world coordinates. ``extent`` places the source here.
   * - Eye Coordinates
     - Coordinates centered on the fovea. The FOV, scotoma, prosthetic
       percept, implant, and visual-field grid live here.

The main scene parameters are:

.. list-table::
   :header-rows: 1
   :widths: 20 52 28

   * - Parameter
     - Meaning
     - Units / Form
   * - ``source``
     - Image or video placed in the visual field
     - image or video stimulus
   * - ``extent``
     - Bounds of the source in scene coordinates
     - ``(left, right, bottom, top)`` in dva, or a scalar
   * - ``fov``
     - Eye-centered viewing window
     - scalar or ``(width, height)`` in dva
   * - ``scotoma``
     - Region of impaired native vision
     - :py:class:`~pulse2percept.vision.Scotoma`
   * - ``aperture``
     - Shape of the visible field
     - ``'rectangle'`` or ``'round'``

A scalar ``extent`` sets the span of the source's shorter dimension while
preserving square pixels. For example, ``45 * dva`` on a 173 x 320 image
produces an extent of approximately 83.2 x 45 dva.

If ``extent`` is omitted, the source is centered and scaled with square pixels
to the smallest extent that contains ``fov``.

Pixel coordinates refer to pixel centers. Row 0 is at the top of the image and
therefore has the largest ``y`` coordinate.

A scene is immutable prediction input: its source, ``extent``, and ``fov`` do
not change after construction.


Gaze
====

``gaze`` specifies the scene location currently falling on the fovea.

A single ``(x, y)`` pair gives one fixation:

.. code-block:: python

    scene.plot(
        gaze=(8, -2) * dva,
    )

For multiple fixations,
:py:class:`~pulse2percept.vision.Gaze` stores their locations and onset times:

.. code-block:: python

    gaze = p2p.vision.Gaze(
        [(0, 0), (-12, 0), (12, 0)] * dva,
        time=[0, 600, 1200] * ms,
    )

Each fixation is held until the next begins. Saccades are instantaneous; gaze
is not interpolated between fixations.


Scene and Eye Views
-------------------

:py:meth:`~pulse2percept.vision.Scene.plot` and
:py:meth:`~pulse2percept.vision.Scene.play` can show the same scene in two
coordinate frames:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - View
     - Display
   * - ``'scene'``
     - Fixed world coordinates spanning ``extent``. The source remains fixed
       while the FOV, scotoma, percept, and visual-field grid move with gaze.
   * - ``'eye'``
     - Fixed eye coordinates spanning the FOV. The source moves through the
       eye-centered window as gaze changes.

``'scene'`` is the default:

.. code-block:: python

    scene.play(gaze=gaze)

The corresponding eye-centered view is:

.. code-block:: python

    scene.play(
        gaze=gaze,
        view='eye',
    )

In the scene view, source content outside the FOV is dimmed by
``context_alpha``. The default is 0.25; use 0 for black context or 1 for the
undimmed source.

:py:meth:`~pulse2percept.vision.Scene.render` always returns an eye-centered
view.


Gaze Timing
-----------

Gaze times are interpreted as ms unless units are supplied.

A fixation beginning exactly on a frame time applies to that frame. For video
scenes, gaze is resolved against the video's frame times. For a still scene
shown together with a timed percept, the percept provides the clock.


Residual Vision
===============

A :py:class:`~pulse2percept.vision.Scotoma` describes where native vision is
lost in eye-centered visual space.

For example:

.. code-block:: python

    central = p2p.vision.Scotoma.circle(
        5 * dva,
    )

    eccentric = p2p.vision.Scotoma.ellipse(
        5 * dva,
        4 * dva,
        center=(-9, 4) * dva,
    )

A scotoma represents the **amount and location of visual loss**, not its visual
appearance. ``scotoma_fill`` controls how complete loss is displayed:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - ``scotoma_fill``
     - Display
   * - ``0``
     - Black; the default
   * - gray level
     - Constant gray in ``[0, 1]``
   * - color
     - RGB value or Matplotlib color
   * - ``'inpaint'``
     - Fill from the surrounding image using biharmonic inpainting

For numeric or color fills, ``scotoma_blend`` controls the softness of the
boundary in degrees of visual angle.

``'inpaint'`` ignores ``scotoma_blend``.

.. note::

   Inpainting is a visualization of missing native vision, not a model of
   perceptual filling-in. It cannot be used when a prosthetic percept is
   composited into the same scotoma.

The scotoma does not affect what the implant receives. Prosthetic encoding
samples the original scene, including regions where native vision is lost.


Combining Native and Prosthetic Vision
--------------------------------------

A predicted percept can be placed back into its visual context:

.. plot::

    import matplotlib.pyplot as plt
    import pulse2percept as p2p

    from pulse2percept.units import dva

    image = p2p.stimuli.samples.ucsb_bike(
        resize=(240, 360),
        as_gray=True,
    )

    scene = p2p.vision.Scene(
        image,
        extent=60 * dva,
        fov=40 * dva,
        scotoma=p2p.vision.Scotoma.circle(6 * dva),
        scotoma_fill=0,
    )

    implant = p2p.implants.retina.PRIMAPivotal()

    model = p2p.models.retina.ScoreboardModel(
        implant,
        rho=50,
        xrange=(-8, 8),
        yrange=(-8, 8),
        step=0.1,
    )

    percept = model.predict_percept(
        scene,
        gaze=(0, 0) * dva,
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    scene.plot(
        ax=axes[0],
        gaze=(0, 0) * dva,
    )
    axes[0].set_title('Residual vision')

    scene.plot(
        percept=percept,
        gaze=(0, 0) * dva,
        ax=axes[1],
    )
    axes[1].set_title('Residual + prosthetic vision')

    fig.tight_layout()

Within visual loss, native and prosthetic vision are composed as

.. math::

    (1 - L)N + L\max(S, P),

where :math:`L` is the fraction of native vision lost, :math:`N` is native
vision, :math:`S` is ``scotoma_fill``, and :math:`P` is the displayed
prosthetic percept.

Without a scotoma, a prosthetic percept is displayed on black rather than
superimposed on intact native vision. The model does not specify how
prosthetic and intact natural vision would interact.


Scenes and Models
=================

A scene, implant, model, and percept each describe a different part of the
simulation:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Object
     - Role
   * - Scene
     - Describes what is visually present, where it is located, where native
       vision is lost, and where the eye is looking
   * - Implant
     - Describes device geometry and how visual input becomes stimulation
   * - Model
     - Places the implant in tissue and predicts the resulting visual percept
   * - Percept
     - Stores the predicted visual output

A retinal spatial model can sample a scene through its retinotopic map:

.. code-block:: python

    implant = p2p.implants.retina.ArgusII()

    model = p2p.models.retina.ScoreboardModel(
        implant,
        rho=200,
    )

    percept = model.predict_percept(
        scene,
        gaze=(0, 0) * dva,
    )

Scene registration currently requires a retinal spatial model
(:py:class:`~pulse2percept.models.retina.RetinalSpatial`) with an invertible
retinotopic map.

Models that do not support scene registration raise ``NotImplementedError``.
A retinal model without the required retinotopy, or an implant without an
encoder when one is required, raises ``ValueError``.


Visual Input to the Implant
===========================

Images and videos can also be passed directly to an implant or model without
placing them in a scene.

The two forms have different spatial meanings:

.. list-table::
   :header-rows: 1
   :widths: 27 32 41

   * - Input
     - Coordinates
     - Interpretation
   * - :py:class:`~pulse2percept.stimuli.ImageStimulus` or
       :py:class:`~pulse2percept.stimuli.VideoStimulus`
     - device-relative
     - The image is mapped across the implant; it has no intrinsic size or
       location in the visual field
   * - :py:class:`~pulse2percept.vision.Scene`
     - visual-field-relative
     - The source has a defined angular extent and each electrode samples its
       own location in visual space

For example, the same image can either fill an implant or occupy a much larger
visual field:

.. plot::

    import matplotlib.pyplot as plt
    import pulse2percept as p2p

    from pulse2percept.units import dva

    image = p2p.stimuli.samples.ucsb_bike(
        resize=(240, 360),
        as_gray=True,
    )

    implant = p2p.implants.retina.ArgusII()

    model = p2p.models.retina.ScoreboardModel(
        implant=implant,
        rho=200,
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    model.predict_percept(image).plot(ax=axes[0])
    axes[0].set_title('ImageStimulus: fills the implant')

    scene = p2p.vision.Scene(
        image,
        extent=60 * dva,
        fov=40 * dva,
    )

    model.predict_percept(
        scene,
        gaze=(0, 0) * dva,
    ).plot(ax=axes[1])
    axes[1].set_title('Scene: positioned in visual space')

    fig.tight_layout()

Use a direct image or video when only the device input matters. Use a
``Scene`` when angular size, gaze, visual-field loss, or the surrounding
visual context matters.


Eye- and Head-Centered Input
----------------------------

Gaze always determines where a predicted prosthetic percept appears in scene
coordinates.

Whether gaze also changes the image reaching the electrodes depends on
:py:attr:`~pulse2percept.implants.Implant.scene_input_frame`:

.. list-table::
   :header-rows: 1
   :widths: 18 34 48

   * - Value
     - Typical Device
     - Behavior
   * - ``'eye'``
     - Alpha, PRIMA
     - Input moves across the implant as gaze changes; the default
   * - ``'head'``
     - Argus, BVT, IMIE
     - A head-mounted camera supplies the input, so eye movements move the
       percept in the scene but do not change electrode input

The implant and scotoma remain fixed in eye coordinates as gaze changes.

For head-centered systems, each electrode samples its own registered
visual-field position. Patient-specific camera-to-electrode registration is
not modeled.

An eye-tracked camera system can be represented with:

.. code-block:: python

    implant.scene_input_frame = 'eye'


Device Preprocessing
--------------------

``implant.preprocess`` transforms the prosthetic input before the implant
samples it. Native vision continues to use the original scene.

For example, an edge-enhancement pipeline can be applied with:

.. code-block:: python

    implant.preprocess = lambda stim: stim.filter('sobel')

For scenes, preprocessing occurs at the source raster resolution and must
return an :py:class:`~pulse2percept.stimuli.ImageStimulus` or
:py:class:`~pulse2percept.stimuli.VideoStimulus` with the same spatial shape
and frame times.

Pixel values and channels may change. Geometry and timing may not, because
``extent`` and the scene clock still refer to the original source.


Plotting, Playing, and Rendering
================================

Scenes provide three complementary display methods:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Method
     - Behavior
   * - ``scene.plot()``
     - Draw one frame, preserving the source and percept at their own spatial
       resolutions
   * - ``scene.play()``
     - Animate a video scene, gaze, and optionally a prosthetic percept
   * - ``scene.render()``
     - Resample all layers into one eye-centered RGB
       :py:class:`~pulse2percept.percepts.Percept`

For example:

.. code-block:: python

    scene.plot(
        gaze=(5, 0) * dva,
    )

    scene.play(
        gaze=gaze,
    )

    rgb = scene.render(
        percept=percept,
        gaze=(0, 0) * dva,
        vmax=50,
    )

Model brightness is in arbitrary units, so use a fixed ``vmax`` when
comparing rendered prosthetic percepts across conditions.

``render`` uses the source's angular pixel pitch by default. Set ``step`` or
``shape`` to choose another output raster:

.. code-block:: python

    fine = scene.render(
        percept=percept,
        vmax=50,
        step=0.02 * dva,
    )

``step`` and ``shape`` are mutually exclusive. When ``step`` is supplied, the
raster is chosen so that pixels are never coarser than requested.


Field Aperture
--------------

``aperture`` controls the visible shape within the FOV:

.. code-block:: python

    disc = p2p.vision.Scene(
        image,
        extent=60 * dva,
        fov=40 * dva,
        aperture='round',
    )

    wide = p2p.vision.Scene(
        image,
        extent=60 * dva,
        fov=(60, 40) * dva,
        aperture='round',
    )

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Value
     - Behavior
   * - ``'rectangle'``
     - Use the complete rectangular FOV; the default
   * - ``'round'``
     - Use the ellipse inscribed within the FOV

For ``fov=(60, 40) * dva``, the round (elliptical) aperture extends 30 dva
horizontally and 20 dva vertically.

The aperture is eye-centered and affects display only. It does not change
scene sampling, stimulation, or model prediction.


Blank Scenes
------------

:py:meth:`~pulse2percept.vision.Scene.blank` creates a black visual field when
no source image is needed:

.. code-block:: python

    scene = p2p.vision.Scene.blank(
        fov=45 * dva,
    )

Black is visual content, not blindness. Use a
:py:class:`~pulse2percept.vision.Scotoma` when native vision is absent.

The default 512 x 512 blank raster is used only for display. Model prediction
still uses the model's own spatial grid.


Both Eyes
=========

:py:class:`~pulse2percept.vision.BinocularScene` groups one scene for each eye:

.. code-block:: python

    left = p2p.vision.Scene(
        image,
        extent=60 * dva,
        fov=40 * dva,
        scotoma=p2p.vision.Scotoma.ellipse(
            5 * dva,
            4 * dva,
            center=(-9, 4) * dva,
        ),
    )

    binocular = p2p.vision.BinocularScene(
        left,
        left.fellow_eye(),
    )

    binocular.plot()

Prosthesis models remain monocular. Predict each eye separately:

.. code-block:: python

    left_percept = model.predict_percept(
        binocular.left,
    )


Creating Binocular Scenes
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Method
     - Behavior
   * - :py:meth:`~pulse2percept.vision.BinocularScene.from_side_by_side`
     - Split left-right packed imagery into two scenes. ``fov`` describes one
       half. Images are not flipped or resampled, and no disparity or depth is
       inferred.
   * - :py:meth:`~pulse2percept.vision.Scene.fellow_eye`
     - Create the fellow-eye scene and mirror eye-specific geometry such as a
       scotoma across the vertical meridian. The source itself is not flipped.
   * - :py:meth:`~pulse2percept.vision.Scotoma.mirror`
     - Mirror a scotoma across the vertical meridian

For side-by-side stereo input:

.. code-block:: python

    stereo = p2p.vision.BinocularScene.from_side_by_side(
        stereo_image,
        fov=(60, 40) * dva,
    )

A ``Scene`` itself has no ``eye`` attribute. Its position inside a
``BinocularScene`` identifies which eye it represents.