.. _topics-vision:

=================================
Visual Input and Simulated Vision
=================================

.. versionadded:: 0.11.0

A picture can be given to a simulation in two ways:

**Device-relative.** An :py:class:`~pulse2percept.stimuli.ImageStimulus` or
:py:class:`~pulse2percept.stimuli.VideoStimulus` passed to an implant is
stretched across its electrodes. It has no size or location in the visual
field.

**Visual-field-relative.** A :py:class:`~pulse2percept.vision.Scene` is one
eye's visual field: what is in front of the eye, its angular extent, and
where native vision is lost. Each electrode receives the part of the scene at
its own visual-field location.

The same image, implant, and model give different percepts:

.. plot::

    import matplotlib.pyplot as plt
    import pulse2percept as p2p
    from pulse2percept.units import dva

    image = p2p.stimuli.samples.logo_bvl(resize=(240, 300))
    implant = p2p.implants.retina.ArgusII()
    model = p2p.models.retina.ScoreboardModel(implant=implant, rho=200)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    model.predict_percept(image).plot(ax=axes[0])
    axes[0].set_title('ImageStimulus: stretched over the array')

    scene = p2p.vision.Scene(image, fov=40 * dva)
    model.predict_percept(scene, gaze=(0, 0) * dva).plot(ax=axes[1])
    axes[1].set_title('Scene: 40 dva of visual field')

    fig.tight_layout()

On the left the logo fills the array. On the right the array covers only its
own part of a 40 dva field, so most of the logo falls outside it.

Scenes
------

A scene wraps a source with its angular extent:

.. code-block:: python

    from pulse2percept.units import dva

    scene = p2p.vision.Scene(p2p.stimuli.samples.logo_bvl(), fov=40 * dva)

    implant = p2p.implants.retina.ArgusII()
    model = p2p.models.retina.ScoreboardModel(implant=implant, rho=200)
    percept = model.predict_percept(scene, gaze=(0, 0) * dva)

*  ``fov`` is the outer angular extent of the frame, centered on it. A scalar
   is the horizontal extent; the vertical extent follows from the aspect
   ratio.
*  Pixel coordinates address pixel centers. Row 0 is the top of the frame
   (largest ``y``).
*  A scene is per-prediction input; it is not stored on the model or
   implant, and its source and geometry are fixed after construction.

Only retinal spatial models
(:py:class:`~pulse2percept.models.retina.RetinalSpatial`) accept scenes,
because placing electrodes in the visual field requires a retinotopic map.
Other models raise ``NotImplementedError``. A retinal model with a
non-retinotopic ``visual_field_map``, or an implant without an ``encoder``,
raises ``ValueError``.

:py:meth:`~pulse2percept.vision.Scene.blank` gives a black scene when no
image is needed:

.. code-block:: python

    scene = p2p.vision.Scene.blank(fov=45 * dva)

Its 512 x 512 raster is used only as the ``render`` default; prediction uses
the model grid. Black is scene content, not blindness; use a
:py:class:`~pulse2percept.vision.Scotoma` for lost vision.

Gaze
----

``gaze`` is the scene location on the fovea (dva), so
``scene = eye-centered visual field + gaze``. Pass one ``(x, y)`` to fixate,
or one per video frame. A :py:class:`~pulse2percept.vision.Gaze` gives
fixations with onset times; each is held until the next:

.. code-block:: python

    from pulse2percept.units import dva, ms

    scene = p2p.vision.Scene(p2p.stimuli.samples.ucsb_flyover(),
                             fov=45 * dva,
                             scotoma=p2p.vision.Scotoma.circle(5 * dva),
                             scotoma_fill=0.5)

    gaze = p2p.vision.Gaze([(0, 0), (6, 2), (-4, 3), (5, -3)] * dva,
                           time=[0, 400, 900, 1400] * ms)

    scene.play(gaze=gaze)
    percept = model.predict_percept(scene, gaze=gaze)

Times are ms unless unitful. A fixation starting exactly on a frame time
applies to that frame. For a video scene, fixations are matched to the
scene's frame times, not ``t_percept``; for a still scene rendered with a
timed percept, the percept's times are used.

Gaze always sets where the percept lands in the scene. Whether it also moves
the electrodes' input depends on
:py:attr:`~pulse2percept.implants.Implant.scene_input_frame`:

==========  =================================================================
``'eye'``   Input passes through the eye's optics (Alpha, PRIMA). Gaze moves
            the scene across the implant. This is the default.
``'head'``  Input comes from a head-mounted camera (Argus, BVT, IMIE). The
            electrodes receive the same input at any gaze; only the percept
            moves.
==========  =================================================================

The implant and the :py:class:`~pulse2percept.vision.Scotoma` stay fixed in
eye-centered coordinates as gaze changes. For ``'head'`` systems, each
electrode samples its own
visual-field position; the per-patient camera-to-electrode mapping is not
modeled. To simulate eye tracking on a camera-based device, set
``implant.scene_input_frame = 'eye'``.

Device preprocessing
--------------------

``implant.preprocess`` (e.g. an edge filter) is applied to the prosthetic
input only, at the scene source's resolution, before the scene is sampled at
the electrodes. Native vision always uses the original scene.

.. code-block:: python

    implant.preprocess = lambda stim: stim.filter('sobel')

For scenes, ``preprocess`` must return an
:py:class:`~pulse2percept.stimuli.ImageStimulus` or
:py:class:`~pulse2percept.stimuli.VideoStimulus` with unchanged spatial shape
and frame times, because ``fov`` and the frame clock refer to the original.
Pixel values and channels may change.

Residual vision
---------------

A :py:class:`~pulse2percept.vision.Scotoma` describes where native vision is
lost, in eye-centered dva:

.. code-block:: python

    scotoma = p2p.vision.Scotoma.circle(8 * dva)
    scotoma = p2p.vision.Scotoma.ellipse(5 * dva, 4 * dva, center=(-9, 4) * dva)

The scotoma does not change ``predict_percept``, which always returns
brightness on the model grid, and prosthetic encoding samples the unmasked
scene. The two are combined only for display:

.. plot::

    import matplotlib.pyplot as plt
    import pulse2percept as p2p
    from pulse2percept.units import dva

    center = (-9, 4) * dva
    image = p2p.stimuli.samples.logo_bvl(resize=(240, 300))
    scotoma = p2p.vision.Scotoma.ellipse(5 * dva, 4 * dva, center=center)
    scene = p2p.vision.Scene(image, fov=40 * dva, scotoma=scotoma, background=1)

    implant = p2p.implants.retina.PRIMAPivotal()
    model = p2p.models.retina.ScoreboardModel(
        implant, implant_position=center, rho=50,
        xrange=(-15, -3), yrange=(-2, 10), step=0.1)
    percept = model.predict_percept(scene, gaze=(0, 0) * dva)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    scene.plot(ax=axes[0], rings=True)
    axes[0].set_title('Residual vision with a scotoma')
    scene.plot(percept=percept, gaze=(0, 0) * dva, vmax=2, ax=axes[1])
    axes[1].set_title('Prosthetic percept inside the loss')
    fig.tight_layout()

Inside the loss, the display is
``(1 - loss) * native + loss * max(scotoma_fill, phosphene)``. With no
scotoma, the percept is drawn alone on black; overlaying it on intact vision
would imply an interaction that is not modeled.

*  ``vmax`` is required, because model brightness has arbitrary units. Keep
   it fixed to compare plots.
*  ``plot`` draws the source at its own resolution and the percept on the
   model grid, so neither is resampled.
*  :py:meth:`~pulse2percept.vision.Scene.render` returns the composite as
   one RGB percept, for saving or further processing:

.. code-block:: python

    rgb = scene.render(percept=percept, gaze=(0, 0) * dva, vmax=50)
    fine = scene.render(percept=percept, vmax=50, step=0.02 * dva)

``render`` uses the source's raster by default. ``step`` (dva per pixel,
rounded so pixels are never coarser) or ``shape`` sets another; they are
mutually exclusive. A fine step over a wide field is expensive.

The field boundary
------------------

``aperture`` sets the field's shape within ``fov``: ``'rectangular'``
(default) uses the whole frame, ``'round'`` the inscribed ellipse:

.. code-block:: python

    disc = p2p.vision.Scene(image, fov=40 * dva, aperture='round')
    wide = p2p.vision.Scene(image, fov=(60, 40) * dva, aperture='round')

The ellipse of ``wide`` reaches 30 dva sideways and 20 dva up. The aperture
is eye-centered, like the scotoma, and affects display only: ``plot`` clips
to it, ``render`` writes black outside it, and scene sampling and
stimulation are unchanged.

Both eyes
---------

:py:class:`~pulse2percept.vision.BinocularScene` holds a left and a right
:py:class:`~pulse2percept.vision.Scene`:

.. code-block:: python

    binocular = p2p.vision.BinocularScene(
        left=p2p.vision.Scene(image, fov=40 * dva, scotoma=scotoma),
        right=p2p.vision.Scene(image, fov=40 * dva),
    )

    ax_left, ax_right = binocular.plot(left_percept=percept, vmax=2)

Models are monocular, so predict for one eye and assign the percept to that
side: ``model.predict_percept(binocular.left)``. A scene has no ``eye``
attribute; its side of the ``BinocularScene`` identifies the eye.

*  :py:meth:`~pulse2percept.vision.BinocularScene.from_side_by_side` splits
   left-right packed imagery into two scenes. ``fov`` describes one half.
   Halves are not flipped or resampled, and no disparity or depth is
   inferred.
*  :py:meth:`~pulse2percept.vision.Scene.fellow_eye` builds the other eye's
   scene with eye-specific geometry (such as the scotoma) mirrored across the
   vertical meridian. The image is not flipped.
*  :py:meth:`~pulse2percept.vision.Scotoma.mirror` mirrors a scotoma alone:
   ``mirrored(x, y) == original(-x, y)``.

.. code-block:: python

    left = p2p.vision.Scene(image, fov=40 * dva, scotoma=scotoma)
    binocular = p2p.vision.BinocularScene(left, left.fellow_eye())

    stereo = p2p.vision.BinocularScene.from_side_by_side(
        stereo_image, fov=(60, 40) * dva)
