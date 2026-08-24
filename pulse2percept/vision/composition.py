""":py:func:`~pulse2percept.vision.compose_amd`"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .scotoma import Scotoma
from ..percepts import Percept
from ..stimuli import ImageStimulus, VideoStimulus
from ..topography import Grid2D
from ..units import Quantity, as_value, dva


def _frame_shape(scene):
    """The (rows, cols) of one scene frame"""
    if isinstance(scene, ImageStimulus):
        return scene.img_shape[:2]
    return scene.vid_shape[:2]


def _scene_rgb(scene):
    """A scene as a dense (rows, cols, 3, n_frames) array of intensities

    One layout for an image and a video, and one channel count: grayscale is
    replicated across the three channels and an alpha channel is blended
    against black, as it is everywhere else in p2p.
    """
    n_rows, n_cols = _frame_shape(scene)
    if isinstance(scene, ImageStimulus):
        frames = scene.data.reshape(scene.img_shape)[..., np.newaxis]
    else:
        frames = scene.data.reshape(scene.vid_shape)
    if frames.ndim == 3:
        frames = frames[:, :, np.newaxis, :]
    n_channels = frames.shape[2]
    if n_channels == 4:
        frames = frames[:, :, :3] * frames[:, :, 3:4]
    elif n_channels == 1:
        frames = np.repeat(frames, 3, axis=2)
    elif n_channels != 3:
        raise ValueError(f"A scene must be grayscale, RGB or RGBA, not "
                         f"{n_channels}-channel.")
    frames = np.asarray(frames, dtype=np.float32)
    if frames.min() < 0 or frames.max() > 1:
        raise ValueError(f"Scene values are display intensities and must lie "
                         f"in [0, 1], but this one spans "
                         f"[{frames.min():g}, {frames.max():g}].")
    return frames.reshape((n_rows, n_cols, 3, -1))


def _gaze_frames(gaze, n_frames):
    """Gaze as one (x, y) in dva, or one per output frame"""
    if gaze is None:
        return np.zeros((1, 2))
    gaze = np.atleast_2d(np.asarray(as_value(gaze, dva, 'gaze'), dtype=float))
    if gaze.shape not in {(1, 2), (n_frames, 2)}:
        raise ValueError(f"'gaze' must be an (x, y) pair in dva, or one per "
                         f"output frame ({n_frames} of them), not an array of "
                         f"shape {gaze.shape}.")
    if not np.all(np.isfinite(gaze)):
        raise ValueError(f"'gaze' must be finite, not {gaze.tolist()}.")
    return gaze


def _prosthetic_frames(scene, prosthetic):
    """Line the percept up with the output frames, and say when they happen

    An image scene has no clock of its own, so a temporal percept sets the
    output timing. A video does have one, and keeps it: the percept is read at
    the video's frame times instead. Whichever clock wins brings its own unit
    along, so a percept counted in seconds does not come back in milliseconds.
    Neither input can arrive with frames but no clock -- ``Percept`` and
    ``VideoStimulus`` both number their frames when nothing better is given.
    """
    if isinstance(scene, ImageStimulus):
        return prosthetic.data, prosthetic.time, prosthetic.time_unit
    n_out = scene.vid_shape[-1]
    if prosthetic.data.shape[-1] == 1:
        # A still percept stands behind every frame of the video:
        return (np.repeat(prosthetic.data, n_out, axis=-1), scene.time,
                scene.time_unit)
    # `Percept.__getitem__` owns time interpolation; the quantity is what
    # carries the video's clock across to the percept's own time unit:
    frames = prosthetic[..., Quantity(np.asarray(scene.time),
                                      scene.time_unit)]
    return frames, scene.time, scene.time_unit


def _scene_grid(scene):
    """A Grid2D on the scene's pixel centers, in scene coordinates

    Gaze moves where the eye-centered scotoma and percept land on the scene; it
    does not move the scene, so this is the one coordinate system the result is
    reported in.
    """
    n_rows, n_cols = _frame_shape(scene)
    x_left, y_top = scene.pixel_to_dva(0, 0)
    x_right, y_bottom = scene.pixel_to_dva(n_cols - 1, n_rows - 1)
    step = (float(x_right - x_left) / (n_cols - 1) if n_cols > 1 else 1.0,
            float(y_top - y_bottom) / (n_rows - 1) if n_rows > 1 else 1.0)
    return Grid2D((float(x_left), float(x_right)),
                  (float(y_bottom), float(y_top)), step=step)


def _prosthetic_sampler(prosthetic, frames):
    """Read the percept at arbitrary eye-centered (y, x) coordinates in dva

    Outside the model's own grid there is no percept, and nothing is
    extrapolated into that space.
    """
    ys = np.asarray(prosthetic.ydva, dtype=float)
    xs = np.asarray(prosthetic.xdva, dtype=float)
    if ys.size < 2 or xs.size < 2:
        raise ValueError(f"A percept needs extent in both directions to be "
                         f"placed in a scene, but this one's grid is "
                         f"{ys.size} x {xs.size}.")
    # `Grid2D` meshes its y axis reversed, so row 0 of the data holds the
    # largest y while `ydva` ascends. Flipping the rows puts the two back in
    # the same order, which is also the ascending one the interpolator wants.
    return RegularGridInterpolator((ys, xs), frames[::-1], method='linear',
                                   bounds_error=False, fill_value=0)


def _check_range(vmin, vmax, scotoma_fill):
    """Reject a brightness mapping or a fill that cannot be displayed"""
    vmin, vmax = float(vmin), float(vmax)
    if not np.isfinite([vmin, vmax]).all():
        raise ValueError(f"'vmin' ({vmin}) and 'vmax' ({vmax}) must be "
                         f"finite.")
    if vmax <= vmin:
        raise ValueError(f"'vmax' ({vmax}) must be greater than 'vmin' "
                         f"({vmin}); the percept is in arbitrary brightness "
                         f"units, and this is what says which of them is "
                         f"white.")
    fill = float(scotoma_fill)
    if not np.isfinite(fill) or fill < 0 or fill > 1:
        raise ValueError(f"'scotoma_fill' is a display intensity and must lie "
                         f"in [0, 1], not {scotoma_fill}.")
    return vmin, vmax, fill


def compose_amd(scene, prosthetic, scotoma, vmax, vmin=0, gaze=None,
                scotoma_fill=0):
    """Compose native and prosthetic vision into one RGB percept

    What someone with an eye-centered scotoma and a retinal implant sees:
    intact vision outside the scotoma, and the modeled percept inside it.

    The three pieces meet in eye-centered visual-field coordinates::

        scene coordinates
            | subtract gaze
        eye-centered visual field
            |- scotoma
            \\- prosthetic percept

    The result is reported on the scene's own pixel grid, so intact vision
    passes through untouched rather than being resampled onto the model's grid.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    scene : ImageStimulus or VideoStimulus
        What is out there to be seen; see
        :py:class:`~pulse2percept.stimuli.ImageStimulus`. Must state a
        ``fov``, since a picture with no visual-field geometry cannot be
        placed against an eye-centered scotoma. Grayscale is replicated into
        RGB and an alpha channel is blended against black.
    prosthetic : :py:class:`~pulse2percept.percepts.Percept`
        The modeled percept, in arbitrary brightness units. Must be grayscale
        and must have been given the ``space`` it was predicted on, which is
        what says where in the visual field it belongs.
    scotoma : :py:class:`~pulse2percept.vision.Scotoma`
        Where native vision is lost, and how much of it.
    vmax : float
        The brightness that displays as white. Required: a percept is in
        arbitrary units, so nothing here can guess it.
    vmin : float, optional
        The brightness that displays as black. Brightness maps linearly onto
        [0, 1] between the two, and is clipped outside them.
    gaze : (x, y) or (n_frames, 2), optional
        Where the eye is pointing: the scene location that falls on the fovea,
        in degrees of visual angle. Defaults to the origin. One pair fixates
        for the whole result; one pair per output frame moves the eye between
        frames.
    scotoma_fill : float, optional
        What complete loss looks like where there is no percept, as a display
        intensity in [0, 1]. Defaults to black.

    Returns
    -------
    percept : :py:class:`~pulse2percept.percepts.Percept`
        An RGB percept of shape ``(Y, X, 3, T)`` on the scene's pixel grid, in
        scene coordinates.

    Notes
    -----
    Each pixel is composed as::

        lost   = maximum(scotoma_fill, prosthetic_rgb)
        output = (1 - loss) * scene_rgb + loss * lost

    That ``maximum`` is a **display composition rule**, not a physiological
    model: it puts a luminous phosphene over whatever the lost view looks like,
    which is what the intact periphery, a complete scotoma, and a phosphene
    inside one each need. It is deliberately the simplest rule that gets those
    three right, and it is expected to be replaced when there is science to
    replace it with.

    Timing follows whichever input has a clock. An image scene takes the
    percept's time axis and time unit; a video keeps its own and reads the
    percept at its frame times.

    """
    if not isinstance(scene, (ImageStimulus, VideoStimulus)):
        raise TypeError(f"'scene' must be an ImageStimulus or a "
                        f"VideoStimulus, not {type(scene)}.")
    if scene.fov is None:
        raise ValueError("A scene must state the 'fov' it subtends before it "
                         "can be composed with an eye-centered scotoma.")
    if not isinstance(prosthetic, Percept):
        raise TypeError(f"'prosthetic' must be a Percept, not "
                        f"{type(prosthetic)}.")
    if prosthetic.is_rgb:
        raise ValueError("'prosthetic' must be a brightness percept: models "
                         "produce brightness in arbitrary units, and this "
                         "function is what turns it into display intensity.")
    if not prosthetic._has_space:
        # Without a `space`, `xdva`/`ydva` are the pixel indices `Data` fills
        # an omitted axis with. Reading those as degrees would place the
        # percept somewhere plausible-looking and wrong:
        raise ValueError("'prosthetic' has no visual-field coordinates, so "
                         "there is nowhere in the scene to put it. Predict it "
                         "on a model grid, or pass 'space' when building it.")
    if not isinstance(scotoma, Scotoma):
        raise TypeError(f"'scotoma' must be a Scotoma, not {type(scotoma)}.")
    vmin, vmax, fill = _check_range(vmin, vmax, scotoma_fill)

    scene_rgb = _scene_rgb(scene)
    pframes, out_time, out_time_unit = _prosthetic_frames(scene, prosthetic)
    n_out = pframes.shape[-1]
    gaze = _gaze_frames(gaze, n_out)
    n_rows, n_cols = _frame_shape(scene)

    # Where every scene pixel sits, in the scene's own coordinates:
    cols, rows = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    x_scene, y_scene = scene.pixel_to_dva(cols, rows)

    static = len(gaze) == 1
    if static:
        # One gaze looks at the same place in every frame, so the scotoma is
        # evaluated once and every frame's brightness comes back from a single
        # call. `scene = visual field + gaze`, run backwards:
        x_vf, y_vf = x_scene - gaze[0, 0], y_scene - gaze[0, 1]
        points = np.column_stack((y_vf.ravel(), x_vf.ravel()))
        brightness = _prosthetic_sampler(prosthetic, pframes)(points)
        brightness = brightness.reshape((n_rows, n_cols, n_out))
        loss = scotoma(x_vf, y_vf)[..., np.newaxis]

    out = np.empty((n_rows, n_cols, 3, n_out), dtype=np.float32)
    for f in range(n_out):
        if static:
            frame = brightness[..., f]
        else:
            # A gaze per frame: the eye was somewhere else when this frame came
            # up, so both the scotoma and the percept land somewhere else too.
            x_vf, y_vf = x_scene - gaze[f, 0], y_scene - gaze[f, 1]
            points = np.column_stack((y_vf.ravel(), x_vf.ravel()))
            sample = _prosthetic_sampler(prosthetic, pframes[..., f:f + 1])
            frame = sample(points).reshape((n_rows, n_cols))
            loss = scotoma(x_vf, y_vf)[..., np.newaxis]
        phosphene = np.clip((frame - vmin) / (vmax - vmin), 0, 1)
        lost = np.maximum(fill, phosphene)[..., np.newaxis]
        native = scene_rgb[..., 0 if scene_rgb.shape[-1] == 1 else f]
        out[..., f] = (1 - loss) * native + loss * lost
    return Percept(out, space=_scene_grid(scene), time=out_time,
                   time_unit=out_time_unit)
