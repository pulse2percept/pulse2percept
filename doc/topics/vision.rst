.. _topics-vision:

=================================
Visual Input and Simulated Vision
=================================

.. versionadded:: 0.11.0

There are two different things a picture can mean, and pulse2percept keeps
them apart.

**Device-relative.** An :py:class:`~pulse2percept.stimuli.ImageStimulus` or
:py:class:`~pulse2percept.stimuli.VideoStimulus` handed straight to an implant
is stretched across that implant's electrodes. The picture means "this is what
the device was shown"; it has no location in the visual field, and nothing
says how large it is.

**Visual-field-relative.** A :py:class:`~pulse2percept.vision.Scene` is one
monocular visual field: what is present in front of one eye, how much of the
field it subtends, and where that eye's native vision is lost. Each electrode
then sees the part of the picture that electrode actually looks at.

The same image, the same implant, and the same model give different percepts:

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

On the left the logo fills the array by construction. On the right the array
covers only the part of a 40-degree field it anatomically reaches, so most of
the picture falls outside the device.

Scenes
------

A scene wraps a source with its angular extent:

.. code-block:: python

    from pulse2percept.units import dva

    scene = p2p.vision.Scene(p2p.stimuli.samples.logo_bvl(), fov=40 * dva)

    implant = p2p.implants.retina.ArgusII()
    model = p2p.models.retina.ScoreboardModel(implant=implant, rho=200)
    percept = model.predict_percept(scene, gaze=(0, 0) * dva)

Geometry follows one convention: ``fov`` is the *outer* angular extent of the
frame, centered on it; pixel coordinates address pixel centers; row 0 is the
top of the frame and therefore the largest ``y``. A scalar ``fov`` is the
horizontal extent, with the vertical one following from the aspect ratio.

Scene prediction separates four responsibilities:

=========  ==================================================================
Scene      What is visually present, and where native vision is lost.
Implant    Device geometry and encoding constraints.
Model      Knows the retinotopy, and so connects Scene to Implant.
Percept    What the simulated observer sees.
=========  ==================================================================

Scene registration is a spatial-model capability: a model has to say where in
the visual field each of its electrodes lands. Only retinal models
(:py:class:`~pulse2percept.models.retina.RetinalSpatial`) implement it, through
their retinotopy; any other spatial model raises ``NotImplementedError``. A
retinal model given a non-retinotopic ``visual_field_map``, or an implant
without an ``encoder``, raises ``ValueError``.

A scene is per-prediction input and is not stored on the model or implant. Its
source and FOV geometry are fixed after construction.

:py:meth:`~pulse2percept.vision.Scene.blank` provides a black visual-field
canvas when no image is needed:

.. code-block:: python

    scene = p2p.vision.Scene.blank(fov=45 * dva)

Its fixed 512 x 512 backing raster is only the raster ``render`` falls back
to; prediction still happens on the model grid regardless. Black is scene
content, not blindness; use a :py:class:`~pulse2percept.vision.Scotoma` for
vision that is lost.

Gaze
----

``gaze`` is the scene location that currently falls on the fovea, so
``scene = eye-centered visual field + gaze``. Pass one ``(x, y)`` to fixate,
or one per video frame to move the eye between frames.

:py:class:`~pulse2percept.vision.Gaze` records *when* fixation changes, and
pulse2percept holds each fixation until the next timestamp:

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

Timestamps are milliseconds unless given as a unitful time, and a fixation
starting exactly on a frame time already applies to that frame.
For a video scene, events resolve against the scene's frame times, not
``t_percept``. When rendering a still scene with a timed percept, the percept
provides the clock.

Gaze always decides where the percept lands in scene coordinates. Whether it
also decides what the electrodes are given depends on the implant's
:py:attr:`~pulse2percept.implants.Implant.scene_input_frame`:

==========  =================================================================
``'eye'``   Input passes through the eye's optics (Alpha, PRIMA), so gaze
            moves the scene across the implant as well as moving the percept
            across the scene. This is the default.
``'head'``  Input comes from a head-fixed camera (Argus, BVT, IMIE) that the
            eye cannot move: the electrodes are handed the same scene
            whatever the gaze, and only the percept moves.
==========  =================================================================

Neither the implant nor an eye-centered
:py:class:`~pulse2percept.vision.Scotoma` moves when gaze does.

For a ``'head'`` system, the electrodes' sampling locations are still their
own visual-field positions, which assumes the device's camera-to-electrode
registration is aligned with them. Real systems configure that mapping
separately and it is not modeled here.

``scene_input_frame`` follows the device class but is a property of the
system, so one implant can be run the other way -- an Argus II with eye
tracking, which shifts the camera ROI with gaze:

.. code-block:: python

    implant.scene_input_frame = 'eye'

Device preprocessing
--------------------

An implant's ``preprocess`` -- an edge filter, an inversion, a contrast
stretch -- is applied to the **prosthetic input branch only**, before the
scene is sampled at the electrode locations, because an image operation needs
an image and by sampling time there is one number per electrode. Native and
residual vision always use the original scene: what the device does to its own
input is not something the eye goes through. Spatial preprocessing operates at
the scene source's pixel resolution.

.. code-block:: python

    implant.preprocess = lambda stim: stim.filter('sobel')

For scene input, ``preprocess`` must return an
:py:class:`~pulse2percept.stimuli.ImageStimulus` or
:py:class:`~pulse2percept.stimuli.VideoStimulus`; conversion to electrical
stimulation belongs to the encoder. Pixel values and channels may change, but
spatial shape and frame timing must remain unchanged because ``fov`` and the
frame clock refer to the original scene.

Residual vision
---------------

A :py:class:`~pulse2percept.vision.Scotoma` describes where native vision is
lost:

.. code-block:: python

    scotoma = p2p.vision.Scotoma.circle(8 * dva)
    scotoma = p2p.vision.Scotoma.ellipse(5 * dva, 4 * dva, center=(-9, 4) * dva)

``predict_percept`` always returns perceived brightness on the model grid,
with or without a scotoma. A scotoma describes residual *native* vision, so it
does not enter the model response, and prosthetic encoding samples the
unmasked scene including locations inside it -- a camera does not go blind
where its wearer has. Drawing the two together is a separate step:

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

Inside the loss the two compose as
``(1 - loss) * native + loss * max(scotoma_fill, phosphene)``. With no
scotoma the percept is drawn alone on black, because superimposing it on
intact native vision would assert an unmodeled interaction.

``vmax`` is required: model brightness is in arbitrary units, so which
brightness counts as white is a display choice. Hold it fixed to keep separate
calls comparable.

Source and percept need not share a resolution. ``plot`` draws the source at
its own raster and the percept as a local patch on the model grid, so a fine
simulation in a wide field costs the model grid's pixels rather than the whole
field at the model's step.

:py:meth:`~pulse2percept.vision.Scene.render` is the explicit dense
composition, for when one RGB raster is needed (saving, downstream image
processing):

.. code-block:: python

    rgb = scene.render(percept=percept, gaze=(0, 0) * dva, vmax=50)
    fine = scene.render(percept=percept, vmax=50, step=0.02 * dva)

The raster defaults to the source's own, which resamples nothing. ``step``
(dva per pixel, rounded up so pixels are never coarser than asked) or
``shape`` chooses another; the two are mutually exclusive. A fine step over a
wide field is expensive by construction.

The field boundary
------------------

A scene's source, pixel grid and sampling are rectangular. ``fov`` gives the
field its outer dimensions ``(width, height)``, and ``aperture`` gives it its
shape: the default ``'rectangular'`` uses the whole frame, while ``'round'``
inscribes an ellipse in those dimensions:

.. code-block:: python

    disc = p2p.vision.Scene(image, fov=40 * dva, aperture='round')
    wide = p2p.vision.Scene(image, fov=(60, 40) * dva, aperture='round')

A square ``fov`` therefore renders as a disc, and a 60 x 40 one as an ellipse
reaching 30 degrees sideways and 20 degrees up. Like the scotoma and the
eccentricity rings, it is eye-centered, so gaze moves it through the scene.

The aperture is support, not scene content: ``plot`` clips its artists to it,
``render`` writes black outside it, and the source arrays are never modified.
Scene sampling, device input and stimulation are untouched either way -- a
camera does not go blind at the edge of an eye-shaped display aperture.

Both eyes
---------

:py:class:`~pulse2percept.vision.BinocularScene` holds the left and right
monocular views:

.. code-block:: python

    binocular = p2p.vision.BinocularScene(
        left=p2p.vision.Scene(image, fov=40 * dva, scotoma=scotoma),
        right=p2p.vision.Scene(image, fov=40 * dva),
    )

    ax_left, ax_right = binocular.plot(left_percept=percept, vmax=2)

Imagery already packed left-right side by side splits into the two eyes with
:py:meth:`~pulse2percept.vision.BinocularScene.from_side_by_side`:

.. code-block:: python

    binocular = p2p.vision.BinocularScene.from_side_by_side(
        stereo_image,
        fov=(60, 40) * dva,
    )

The left half becomes the left eye and the right half the right eye. ``fov``
describes one eye's half, splitting does not flip, resample, or interpolate
either half, and no disparity or depth is inferred.

A bilateral loss is often symmetric about the vertical meridian.
:py:meth:`~pulse2percept.vision.Scene.fellow_eye` builds the homologous scene
for the other eye:

.. code-block:: python

    left = p2p.vision.Scene(image, fov=40 * dva, scotoma=scotoma)
    binocular = p2p.vision.BinocularScene(left, left.fellow_eye())

It reflects eye-specific geometry, such as a scotoma, across the vertical
meridian. The image itself is not flipped, since both eyes look at the same
world in the same orientation.

:py:meth:`~pulse2percept.vision.Scotoma.mirror` is the same reflection on a
scotoma alone (``mirrored(x, y) == original(-x, y)``):

.. code-block:: python

    left_scotoma = p2p.vision.Scotoma.circle(3 * dva, center=(6, 0) * dva)
    right_scotoma = left_scotoma.mirror()

Models are monocular in v0.11, so a prediction names the eye it is about:

.. code-block:: python

    percept = model.predict_percept(binocular.left)

A scene carries no ``eye`` of its own; eye identity for a visual field comes
from its side of a :py:class:`~pulse2percept.vision.BinocularScene`.
