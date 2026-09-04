""":py:class:`~pulse2percept.vision.Scene`"""
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from skimage.color import rgb2gray
from skimage.restoration import inpaint_biharmonic

from .scotoma import Scotoma
from ..percepts import Percept
from ..stimuli import ImageStimulus, VideoStimulus
from ..topography import Grid2D
from ..units import Quantity, as_value, dimensionless, dva
from ..utils import PrettyPrint

# How many sigmas of the blur kernel are kept; also how far off-frame the loss
# map is rasterized, so the cropped result is free of edge effects:
_TRUNCATE = 4.0

# Eccentricity spacing, in dva, that `rings=True` asks for
_RING_STEP = 5.0

# Ring color: dark enough to read on a light scene, gray enough to stay
# annotation rather than content
_RING_COLOR = '0.3'

# The only non-numeric `scotoma_fill`; see `_inpaint_rgb` for what it does.
_INPAINT = 'inpaint'


def _resolve_fov(fov, n_rows, n_cols):
    """Normalize a user-supplied ``fov`` to ``(width, height)`` in dva"""
    fov = np.asarray(as_value(fov, dva, 'fov'), dtype=float)
    if fov.ndim == 0:
        width = float(fov)
        height = width * n_rows / n_cols
    elif fov.shape == (2,):
        width, height = (float(f) for f in fov)
    else:
        raise ValueError(f"'fov' must be a scalar (horizontal FOV) or a "
                         f"(width, height) pair, not {fov.tolist()}.")
    for name, f in (('width', width), ('height', height)):
        if not np.isfinite(f) or f <= 0:
            raise ValueError(f"'fov' {name} must be a finite positive number "
                             f"of degrees, not {f}.")
    return (width, height)


def _gaze_points(gaze, n_frames):
    """Gaze as one (x, y) in dva, or one per frame"""
    if gaze is None:
        return np.zeros((1, 2))
    gaze = np.atleast_2d(np.asarray(as_value(gaze, dva, 'gaze'), dtype=float))
    if gaze.shape not in {(1, 2), (n_frames, 2)}:
        raise ValueError(f"'gaze' must be an (x, y) pair in dva, or one per "
                         f"frame ({n_frames} of them), not an array of shape "
                         f"{gaze.shape}.")
    if not np.all(np.isfinite(gaze)):
        # Left to reach the interpolator, this would come back as a blank
        # percept rather than as a question about where the eye was pointing:
        raise ValueError(f"'gaze' must be finite, not {gaze.tolist()}.")
    return gaze


def _clip_to_frame(points, shape):
    """Clip pixel coordinates onto the frame, and say which were on it"""
    points = np.asarray(points, dtype=float)
    edges = np.asarray(shape[:2], dtype=float) - 0.5
    inside = np.all((points >= -0.5) & (points <= edges), axis=1)
    # A point off the frame is not interpolated at all, so anything goes here
    # as long as it is on the grid:
    on_grid = np.where(inside[:, np.newaxis], points, 0.0)
    return np.clip(on_grid, 0.0, edges - 0.5), inside


def _interpolate(grid, frames, points):
    """Sample ``frames`` at ``points``, carrying the trailing axes along"""
    interpolator = RegularGridInterpolator(grid, frames, method='linear',
                                           bounds_error=False, fill_value=0)
    return interpolator(points)


def _drop_gray_axis(values):
    """Give back the (n_points, n_frames) a grayscale scene samples to"""
    return values[:, 0] if values.shape[1] == 1 else values


def _percept_sampler(prosthetic, frames):
    """Read a percept at arbitrary eye-centered ``(y, x)`` coordinates in dva"""
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


def _resolve_fill(scotoma_fill):
    """Normalize ``scotoma_fill`` to a display intensity or ``_INPAINT``"""
    if isinstance(scotoma_fill, str):
        if scotoma_fill != _INPAINT:
            raise ValueError(f"'scotoma_fill' is either a display intensity "
                             f"in [0, 1] or {_INPAINT!r}, not "
                             f"{scotoma_fill!r}.")
        return _INPAINT
    fill = float(as_value(scotoma_fill, dimensionless, 'scotoma_fill'))
    if not np.isfinite(fill) or fill < 0 or fill > 1:
        raise ValueError(f"'scotoma_fill' is a display intensity and must "
                         f"lie in [0, 1], not {scotoma_fill}.")
    return fill


def _resolve_background(background):
    """Normalize ``background`` to an ``(r, g, b)`` triple in [0, 1]"""
    bg = np.asarray(as_value(background, dimensionless, 'background'),
                    dtype=float)
    if bg.ndim == 0:
        bg = np.repeat(bg, 3)
    if bg.shape != (3,):
        raise ValueError(f"'background' must be a gray level or an (r, g, b) "
                         f"triple, not {background!r}.")
    if not np.all(np.isfinite(bg)) or bg.min() < 0 or bg.max() > 1:
        raise ValueError(f"'background' is a display intensity and must lie "
                         f"in [0, 1], not {background!r}.")
    return bg


def _ring_radii(rings, fov):
    """Eccentricities, in dva, that a ``rings`` argument asks for

    True is ``_RING_STEP``-degree spacing, a number is that spacing, and a
    sequence is the eccentricities themselves. False or None is none.
    """
    if rings is None or rings is False:
        return np.zeros(0)
    if rings is True:
        rings = _RING_STEP
    rings = np.asarray(as_value(rings, dva, 'rings'), dtype=float)
    if rings.ndim == 0:
        step = float(rings)
        if not np.isfinite(step) or step <= 0:
            raise ValueError(f"'rings' is a spacing in degrees and must be "
                             f"finite and positive, not {step}.")
        # The largest ring wholly inside a rectangular FOV is set by its
        # shorter half-axis; 1e-9 keeps one that lands exactly on it:
        return step * np.arange(1, int(min(fov) / 2 / step + 1e-9) + 1)
    radii = np.sort(rings.ravel())
    if radii.size == 0 or not np.all(np.isfinite(radii)) or radii.min() <= 0:
        raise ValueError(f"'rings' must be finite positive eccentricities in "
                         f"degrees, not {rings.tolist()}.")
    return radii


def _identity(x, y):
    """Axes already in degrees need no conversion"""
    return x, y


def _draw_rings(ax, radii, center, to_axes=_identity,
                color=_RING_COLOR):
    """Thin dashed eccentricity rings about ``center``, labelled at the top"""
    cx, cy = center
    theta = np.linspace(0, 2 * np.pi, 181)
    for radius in radii:
        # maps scene degrees onto whatever the axes are drawn in
        ax.plot(*to_axes(cx + radius * np.cos(theta),
                         cy + radius * np.sin(theta)),
                color=color, linestyle='--', linewidth=0.8, alpha=0.9)
        # `va='bottom'` keeps the label above the ring on screen either way:
        ax.text(*to_axes(cx, cy + radius), f'{radius:g}\N{DEGREE SIGN} ecc',
                color=color, fontsize=8, alpha=0.95, ha='center',
                va='bottom')


def _rings_overlay(shape, radii, center, to_pixel, color=_RING_COLOR):
    """The same rings, rasterized into a transparent ``(rows, cols, 4)`` RGBA

    The HTML player lays its frame canvas over the figure, so an annotation
    left as a Matplotlib artist would be covered. Drawing it offscreen through
    `_draw_rings` keeps one definition of the style.
    """
    n_rows, n_cols = shape
    dpi = 100.0
    fig = Figure(figsize=(n_cols / dpi, n_rows / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    fig.patch.set_alpha(0)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.patch.set_alpha(0)
    ax.set_axis_off()
    # One axes unit per pixel, y running down, as `imshow` draws a frame:
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    _draw_rings(ax, radii, center, to_pixel, color=color)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba(), dtype=np.float32) / 255.0


def _over(frames, overlay):
    """Alpha-composite an RGBA ``overlay`` onto every RGB frame"""
    alpha = overlay[..., 3][..., np.newaxis, np.newaxis]
    color = overlay[..., :3][..., np.newaxis]
    return np.clip(frames * (1 - alpha) + color * alpha, 0, 1)


def _inpaint_rgb(image, mask):
    """Fill ``image`` where ``mask`` is True from the pixels where it is not"""
    if mask.all():
        raise ValueError("scotoma_fill='inpaint' fills a scotoma in from the "
                         "vision around it, and here there is none: every "
                         "pixel of the frame is lost. Use a numeric fill, or "
                         "a smaller scotoma.")
    if not mask.any():
        return np.asarray(image, dtype=np.float32)
    holed = np.where(mask[..., np.newaxis], 0.0, image)
    filled = inpaint_biharmonic(holed, mask, channel_axis=-1)
    return np.clip(filled, 0, 1).astype(np.float32)


def _check_range(vmin, vmax):
    """Reject a brightness-to-display mapping that cannot be drawn"""
    if vmax is None:
        raise ValueError("'vmax' is required: a percept is in arbitrary "
                         "brightness units, so nothing here can guess which "
                         "of them displays as white.")
    vmin, vmax = float(vmin), float(vmax)
    if not np.isfinite([vmin, vmax]).all():
        raise ValueError(f"'vmin' ({vmin}) and 'vmax' ({vmax}) must be "
                         f"finite.")
    if vmax <= vmin:
        raise ValueError(f"'vmax' ({vmax}) must be greater than 'vmin' "
                         f"({vmin}); the percept is in arbitrary brightness "
                         f"units, and this is what says which of them is "
                         f"white.")
    return vmin, vmax


class Scene(PrettyPrint):
    """What is visually present, and where native vision is lost

    A scene places a picture in the visual field: it says how much of the
    field the picture subtends, and optionally where in that field native
    vision is missing. That is enough for a model to work out what an
    implanted eye sees, without the caller converting anything by hand.

    Geometry follows one convention:

    *  ``fov`` is the *outer* angular extent of the frame, centered on it, so
       it reaches half an angular pixel past the outermost pixel centers.
    *  Pixel coordinates address pixel *centers*.
    *  Row 0 is the top of the frame and therefore the largest ``y``.

    The scotoma is *eye-centered*: it is fixed relative to the fovea, and so
    is an implant, which sits on the retina. Gaze moves the scene past both of
    them rather than moving either::

        (x_scene, y_scene) = (x_eye, y_eye) + (x_gaze, y_gaze)

    Gaze always decides where an eye-centered percept lands in the scene. It
    also decides what the device is given to encode unless the implant's
    :py:attr:`~pulse2percept.implants.Implant.scene_input_frame` is
    ``'head'``, which says a head-fixed camera supplies the input and the eye
    cannot move it.

    A scene's source and FOV geometry are fixed after construction: ``fov`` is
    resolved against the source's frame shape, so swapping one out without the
    other would leave the geometry describing a picture that is no longer
    there.

    The scotoma is native vision's business only. What an implant is given to
    encode is sampled from the source itself, inside the scotoma as well as
    outside it: a camera does not go blind where its wearer has.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    source : ImageStimulus, VideoStimulus, or image
        The background scene itself. Anything that is not already a
        :py:class:`~pulse2percept.stimuli.ImageStimulus` or a
        :py:class:`~pulse2percept.stimuli.VideoStimulus`, such as a file name
        or a NumPy array, is handed to ``ImageStimulus``.
    fov : float or (width, height)
        How much of the visual field the source covers, in degrees of visual
        angle (e.g. ``40 * dva``). A scalar is the horizontal extent, and the
        vertical one follows from the frame's aspect ratio.
    scotoma : :py:class:`~pulse2percept.vision.Scotoma`, optional
        The region where native vision is lost. If None, native vision is
        intact everywhere and the scene is simply what is out there.
    scotoma_fill : float or 'inpaint', optional
        Gray level to fill the scotoma with (in [0, 1]). Default (0)  black.
        ``'inpaint'`` instead fills the scotoma in from the vision around it
        using :py:func:`skimage.restoration.inpaint_biharmonic` (ignoring
        ``scotoma_blend``). ``'inpaint'`` describes native vision only: it is
        refused when a prosthetic percept is composed into the scotoma, where
        it would act as a brightness floor.
    background : float or (r, g, b), optional
        Gray level or RGB value to use for transparent pixels. Defaults to
        black.
    scotoma_blend : float, optional
        Standard deviation, in scene pixels, of a Gaussian blur applied to the
        rasterized loss map before it is drawn, softening the boundary from
        both sides. Defaults to 2. Rendering only: the scotoma's geometry is
        unchanged.

    Examples
    --------
    A logo covering 40 degrees, seen with a central 16-degree scotoma:

    >>> from pulse2percept.stimuli import LogoBVL
    >>> from pulse2percept.units import dva
    >>> from pulse2percept.vision import Scene, Scotoma
    >>> scene = Scene(LogoBVL(), fov=40 * dva, scotoma=Scotoma.circle(8 * dva))
    >>> scene.fov
    (40.0, 32.0)

    """

    def __init__(self, source, fov, scotoma=None, scotoma_fill=0,
                 scotoma_blend=2, background=0):
        if not isinstance(source, (ImageStimulus, VideoStimulus)):
            # A picture is the common case:
            source = ImageStimulus(source)
        if scotoma is not None and not isinstance(scotoma, Scotoma):
            raise TypeError(f"'scotoma' must be a Scotoma object, not "
                            f"{type(scotoma)}.")
        fill = _resolve_fill(scotoma_fill)
        blend = float(as_value(scotoma_blend, dimensionless,
                               'scotoma_blend'))
        if not np.isfinite(blend) or blend < 0:
            raise ValueError(f"'scotoma_blend' is a Gaussian sigma in scene "
                             f"pixels and must be finite and non-negative, "
                             f"not {scotoma_blend}.")
        self._source = source
        self._background = _resolve_background(background)
        self._scotoma = scotoma
        self._scotoma_fill = fill
        self._scotoma_blend = blend
        n_rows, n_cols = self._frame_shape
        self._fov = _resolve_fov(fov, n_rows, n_cols)
        self._cached_frames = None

    def _pprint_params(self):
        """Return a dict of class attributes to pretty-print"""
        return {'source': type(self.source).__name__, 'fov': self.fov,
                'shape': self.shape, 'scotoma': self.scotoma,
                'background': self.background,
                'scotoma_fill': self.scotoma_fill,
                'scotoma_blend': self.scotoma_blend}

    @property
    def source(self):
        """The picture itself, as an ImageStimulus or a VideoStimulus"""
        return self._source

    @property
    def scotoma(self):
        """Where native vision is lost, or None if it is intact"""
        return self._scotoma

    @property
    def background(self):
        """What shows through a transparent source, as ``(r, g, b)``"""
        return tuple(self._background)

    @property
    def scotoma_fill(self):
        """The display intensity complete loss shows as, or ``'inpaint'``

        ``'inpaint'`` is native vision only; prosthetic composition requires a
        numeric fill.
        """
        return self._scotoma_fill

    @property
    def scotoma_blend(self):
        """Gaussian sigma, in scene pixels, softening the drawn scotoma"""
        return self._scotoma_blend

    @property
    def _frame_shape(self):
        """The (rows, cols) of one frame of the source"""
        if isinstance(self.source, ImageStimulus):
            return tuple(self.source.img_shape[:2])
        return tuple(self.source.vid_shape[:2])

    @property
    def fov(self):
        """Field of view ``(width, height)``, in degrees of visual angle"""
        return self._fov

    @property
    def shape(self):
        """The ``(rows, cols)`` of one frame"""
        return self._frame_shape

    @property
    def time(self):
        """Frame times of the source, or None for a still scene"""
        return self.source.time

    @property
    def time_unit(self):
        """The unit ``time`` is counted in"""
        return self.source.time_unit

    @property
    def _angular_pixel(self):
        """Angular size ``(width, height)`` of one pixel, in dva"""
        n_rows, n_cols = self._frame_shape
        return (self._fov[0] / n_cols, self._fov[1] / n_rows)

    def pixel_to_dva(self, col, row):
        """Visual-field coordinates of a pixel center

        Parameters
        ----------
        col, row : float or array_like
            Pixel coordinates, where ``(0, 0)`` is the center of the top-left
            pixel. Fractional values address points between pixel centers.

        Returns
        -------
        x, y : np.ndarray
            Scene coordinates in degrees of visual angle, relative to the
            center of the frame. ``y`` grows upwards, so row 0 has the largest
            ``y``.

        """
        dx, dy = self._angular_pixel
        col = np.asarray(col, dtype=float)
        row = np.asarray(row, dtype=float)
        x = (col + 0.5) * dx - self._fov[0] / 2
        y = self._fov[1] / 2 - (row + 0.5) * dy
        return x, y

    def dva_to_pixel(self, x, y):
        """Pixel coordinates of a point in the scene

        The inverse of :py:meth:`~pulse2percept.vision.Scene.pixel_to_dva`.

        Parameters
        ----------
        x, y : float or array_like
            Scene coordinates in degrees of visual angle, relative to the
            center of the frame.

        Returns
        -------
        col, row : np.ndarray
            Continuous pixel coordinates, where ``(0, 0)`` is the center of the
            top-left pixel. They are not rounded and not clipped to the frame:
            a point outside the FOV maps outside the pixel grid.

        """
        dx, dy = self._angular_pixel
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        col = (x + self._fov[0] / 2) / dx - 0.5
        row = (self._fov[1] / 2 - y) / dy - 0.5
        return col, row

    def _frames(self):
        """The source as a dense ``(rows, cols, channels, n_frames)`` array"""
        if self._cached_frames is not None:
            return self._cached_frames
        source = self.source
        if isinstance(source, ImageStimulus):
            frames = source.data.reshape(source.img_shape)[..., np.newaxis]
        else:
            frames = source.data.reshape(source.vid_shape)
        if frames.ndim == 3:
            # Grayscale: give it the channel axis the color path already has
            frames = frames[:, :, np.newaxis, :]
        if frames.shape[2] == 4:  # includes alpha channel
            rgb, alpha = frames[:, :, :3], frames[:, :, 3:4]
            bg = self._background.reshape((1, 1, 3, 1))
            frames = np.clip(rgb * alpha + bg * (1 - alpha), 0, 1)
        elif frames.shape[2] not in (1, 3):
            raise ValueError(f"A scene must be grayscale, RGB or RGBA, not "
                             f"{frames.shape[2]}-channel.")
        frames = np.asarray(frames, dtype=np.float32)
        if frames.min() < 0 or frames.max() > 1:
            raise ValueError(f"Scene values are display intensities and must "
                             f"lie in [0, 1], but this one spans "
                             f"[{frames.min():g}, {frames.max():g}].")
        self._cached_frames = frames
        return frames

    @property
    def n_frames(self):
        """How many frames the source has; 1 for a still scene"""
        return self._frames().shape[-1]

    def _sample_at(self, x, y, gaze=None):
        """What the scene shows at eye-centered visual-field positions"""
        frames = self._frames()
        gaze = _gaze_points(gaze, frames.shape[-1])
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        # The pixel grid is the only geometry the interpolator needs; the rest
        # of it lives in `dva_to_pixel`:
        grid = (np.arange(frames.shape[0], dtype=float),
                np.arange(frames.shape[1], dtype=float))
        sampled = []
        for f, (gx, gy) in enumerate(gaze):
            col, row = self.dva_to_pixel(x + gx, y + gy)
            points, inside = _clip_to_frame(np.column_stack((row, col)),
                                            frames.shape)
            values = _interpolate(grid, frames if len(gaze) == 1
                                  else frames[..., f], points)
            values[~inside] = 0
            sampled.append(values)
        if len(sampled) == 1:
            return _drop_gray_axis(sampled[0])
        return _drop_gray_axis(np.stack(sampled, axis=-1))

    def _device_input(self, x, y, gaze=None):
        """One number per position per frame, for a device to encode"""
        values = self._sample_at(x, y, gaze=gaze)
        if values.ndim == 2:
            return values
        # `rgb2gray` wants the channels last:
        return rgb2gray(values.transpose((0, 2, 1)))

    def _rgb_frames(self):
        """The source as ``(rows, cols, 3, n_frames)``, scotoma not applied"""
        frames = self._frames()
        if frames.shape[2] == 1:
            frames = np.repeat(frames, 3, axis=2)
        return frames

    def _loss_at(self, gaze_xy, pad=0):
        """Geometric loss at each scene pixel, in [0, 1]"""
        n_rows, n_cols = self._frame_shape
        if self.scotoma is None:
            return np.zeros((n_rows + 2 * pad, n_cols + 2 * pad))
        # scene = visual field + gaze:
        gx, gy = gaze_xy
        x_scene, y_scene = self._pixel_centers(pad=pad)
        return self.scotoma(x_scene - gx, y_scene - gy)

    def _rendered_loss_at(self, gaze_xy):
        """The loss map as drawn: `_loss_at` softened by `scotoma_blend`"""
        sigma = self._scotoma_blend
        # An inpainted fill ignores the hard boundary:
        hard = self.scotoma is None or self._scotoma_fill == _INPAINT
        if hard or sigma == 0:
            return self._loss_at(gaze_xy)
        # Blur the loss field, not a frame-sized crop of it:
        pad = int(np.ceil(_TRUNCATE * sigma)) + 1
        loss = self._loss_at(gaze_xy, pad=pad)
        blurred = gaussian_filter(loss, sigma, mode='nearest',
                                  truncate=_TRUNCATE)
        return np.clip(blurred[pad:-pad, pad:-pad], 0, 1)

    def _fill_rgb(self, frame_rgb, loss):
        """What complete loss shows for one ``(rows, cols, 3)`` frame"""
        if self._scotoma_fill != _INPAINT:
            return self._scotoma_fill
        return _inpaint_rgb(frame_rgb, loss > 0)

    def _native_rgb(self, gaze=None):
        """What is left of native vision, as ``(rows, cols, 3, n_frames)``"""
        frames = self._rgb_frames()
        if self.scotoma is None:
            return frames
        n_frames = frames.shape[-1]
        gaze = _gaze_points(gaze, n_frames)
        static = len(gaze) == 1
        if static:
            loss = self._rendered_loss_at(gaze[0])
        out = np.empty(frames.shape, dtype=np.float32)
        for f in range(n_frames):
            if not static:
                loss = self._rendered_loss_at(gaze[f])
            frame = frames[..., f]
            # An inpainted fill reads this frame, so it is per-frame work:
            fill = self._fill_rgb(frame, loss)
            alpha = loss[..., np.newaxis]
            out[..., f] = (1 - alpha) * frame + alpha * fill
        return out

    def _pixel_centers(self, pad=0):
        """Scene coordinates of every pixel center, as ``(x, y)`` meshes"""
        n_rows, n_cols = self._frame_shape
        cols, rows = np.meshgrid(np.arange(-pad, n_cols + pad),
                                 np.arange(-pad, n_rows + pad))
        return self.pixel_to_dva(cols, rows)

    def _compose(self, prosthetic, vmax, vmin=0, gaze=None):
        """Native vision with a prosthetic percept painted into the loss"""
        if self._scotoma_fill == _INPAINT:
            raise ValueError(
                f"scotoma_fill={_INPAINT!r} cannot be combined with a "
                f"prosthetic percept: filling-in and prosthetic vision would "
                f"have to be added up, and their interaction is not modeled. "
                f"Use a numeric 'scotoma_fill' (e.g. 0) for prosthetic "
                f"composition; {_INPAINT!r} stays available for native "
                f"vision, e.g. Scene.plot.")
        if not isinstance(prosthetic, Percept):
            raise TypeError(f"'prosthetic' must be a Percept, not "
                            f"{type(prosthetic)}.")
        if prosthetic.is_rgb:
            raise ValueError("'prosthetic' must be a brightness percept: "
                             "models produce brightness in arbitrary units, "
                             "and composing it is what turns that into "
                             "display intensity.")
        if not prosthetic._has_space:
            raise ValueError("'prosthetic' has no visual-field coordinates, "
                             "so there is nowhere in the scene to put it. "
                             "Predict it on a model grid, or pass 'space' "
                             "when building it.")
        vmin, vmax = _check_range(vmin, vmax)
        scene_rgb = self._rgb_frames()
        pframes, out_time, out_unit = self._prosthetic_frames(prosthetic)
        n_out = pframes.shape[-1]
        gaze = _gaze_points(gaze, n_out)
        n_rows, n_cols = self._frame_shape

        x_scene, y_scene = self._pixel_centers()
        static = len(gaze) == 1
        if static:
            gx, gy = gaze[0]
            points = np.column_stack(((y_scene - gy).ravel(),
                                      (x_scene - gx).ravel()))
            brightness = _percept_sampler(prosthetic, pframes)(points)
            brightness = brightness.reshape((n_rows, n_cols, n_out))
            loss = self._rendered_loss_at(gaze[0])

        out = np.empty((n_rows, n_cols, 3, n_out), dtype=np.float32)
        for f in range(n_out):
            if static:
                frame = brightness[..., f]
            else:
                gx, gy = gaze[f]
                points = np.column_stack(((y_scene - gy).ravel(),
                                          (x_scene - gx).ravel()))
                sample = _percept_sampler(prosthetic, pframes[..., f:f + 1])
                frame = sample(points).reshape((n_rows, n_cols))
                loss = self._rendered_loss_at(gaze[f])
            phosphene = np.clip((frame - vmin) / (vmax - vmin), 0, 1)
            native = scene_rgb[..., 0 if scene_rgb.shape[-1] == 1 else f]
            fill = self._fill_rgb(native, loss)
            lost = np.maximum(fill, phosphene[..., np.newaxis])
            alpha = loss[..., np.newaxis]
            out[..., f] = (1 - alpha) * native + alpha * lost
        return Percept(out, space=self._grid(), time=out_time,
                       time_unit=out_unit)

    def _prosthetic_frames(self, prosthetic):
        """Line a percept up with the output frames, and say when they happen"""
        if self.time is None:
            return prosthetic.data, prosthetic.time, prosthetic.time_unit
        n_out = self.n_frames
        n_pros = prosthetic.data.shape[-1]
        if n_pros == 1 and prosthetic.time is None:
            return (np.repeat(prosthetic.data, n_out, axis=-1), self.time,
                    self.time_unit)
        if n_pros == n_out:
            return prosthetic.data, prosthetic.time, prosthetic.time_unit
        unit = prosthetic.time_unit
        asked = np.asarray(self.source.times(unit), dtype=float)
        lo, hi = float(prosthetic.time[0]), float(prosthetic.time[-1])
        slack = 1e-9 * max(abs(lo), abs(hi), 1.0)
        if asked.min() < lo - slack or asked.max() > hi + slack:
            raise ValueError(
                f"The percept covers {lo:g}-{hi:g} {unit}, but the scene runs "
                f"{asked.min():g}-{asked.max():g} {unit}. Nothing was modeled "
                f"outside that interval, and holding the nearest predicted "
                f"frame there would show a phosphene that was never "
                f"simulated. Predict the percept over the whole video, or "
                f"trim the video to the percept.")
        frames = prosthetic[..., Quantity(np.asarray(self.time),
                                          self.time_unit)]
        return frames, self.time, self.time_unit

    def _grid(self):
        """A Grid2D on the scene's pixel centers, in scene coordinates"""
        n_rows, n_cols = self._frame_shape
        x_left, y_top = self.pixel_to_dva(0, 0)
        x_right, y_bottom = self.pixel_to_dva(n_cols - 1, n_rows - 1)
        step = (float(x_right - x_left) / (n_cols - 1) if n_cols > 1 else 1.0,
                float(y_top - y_bottom) / (n_rows - 1) if n_rows > 1 else 1.0)
        return Grid2D((float(x_left), float(x_right)),
                      (float(y_bottom), float(y_top)), step=step)

    def _native_percept(self, gaze=None):
        """Residual native vision as an ordinary RGB percept"""
        return Percept(self._native_rgb(gaze=gaze), space=self._grid(),
                       time=self.time, time_unit=self.time_unit)

    def plot(self, gaze=None, frame=0, ax=None, rings=False,
             ring_color=_RING_COLOR, **kwargs):
        """Plot what is left of native vision

        The scene unchanged where vision is intact, and ``scotoma_fill`` where
        it is lost. A scotoma is eye-centered, so ``gaze`` decides where in the
        scene it falls.

        Parameters
        ----------
        gaze : (x, y), optional
            Where the eye is pointing: the scene location that falls on the
            fovea, in dva. Defaults to the origin.
        frame : int, optional
            Which frame of a video scene to draw. Ignored for a still scene.
        ax : matplotlib.axes.Axes, optional
            The axes to draw on. If None, uses the current axes.
        rings : bool, float, or sequence, optional
            Eccentricity rings about the fovea, which ``gaze`` places in the
            scene. True draws them every 5 degrees out to the edge of the
            field, a number is that spacing instead, and a sequence is the
            eccentricities themselves. Decoration only: the scene data is
            untouched.
        ring_color : color, optional
            Any Matplotlib color for those rings and their labels. Defaults to
            a mid-gray that reads on a light scene.
        **kwargs :
            Passed on to :py:meth:`~pulse2percept.percepts.Percept.plot`.

        Returns
        -------
        ax : matplotlib.axes.Axes

        """
        rgb = self._native_rgb(gaze=gaze)
        if not 0 <= frame < rgb.shape[-1]:
            raise ValueError(f"'frame' must be in 0..{rgb.shape[-1] - 1}, not "
                             f"{frame}.")
        still = Percept(rgb[..., frame:frame + 1], space=self._grid())
        radii = _ring_radii(rings, self.fov)
        ax = still.plot(ax=ax, **kwargs)
        if radii.size:
            # The fovea sits wherever gaze points, which is where the scotoma
            # is drawn too; at the default gaze that is the scene's center.
            _draw_rings(ax, radii, _gaze_points(gaze, rgb.shape[-1])[0],
                        color=ring_color)
        return ax

    def play(self, gaze=None, rings=False, ring_color=_RING_COLOR, ax=None,
             **kwargs):
        """Animate a video scene as it is natively seen

        Parameters
        ----------
        gaze : (x, y) or (n_frames, 2), optional
            Where the eye is pointing, in dva. One pair fixates throughout;
            one pair per frame moves the eye between frames.
        rings : bool, float, or sequence, optional
            Eccentricity rings, as in
            :py:meth:`~pulse2percept.vision.Scene.plot`, painted into the
            displayed frames. Drawn once, so this needs a gaze that holds
            still; the scene's own data is not touched.
        ring_color : color, optional
            Any Matplotlib color for those rings and their labels.
        ax : matplotlib.axes.Axes, optional
            Axes to animate on. If None, the player makes its own.
        **kwargs :
            Passed on to :py:meth:`~pulse2percept.percepts.Percept.play`.

        Returns
        -------
        ani : :py:class:`~pulse2percept.utils.HTMLAnimation`

        """
        if self.time is None:
            raise ValueError("A still scene has nothing to play. Use plot().")
        radii = _ring_radii(rings, self.fov)
        if not radii.size:
            return self._native_percept(gaze=gaze).play(ax=ax, **kwargs)
        points = _gaze_points(gaze, self.n_frames)
        if len(points) > 1:
            raise ValueError(
                "Rings mark eccentricity from the fovea, so a gaze that moves "
                "between frames would have to move them too, and the player "
                "draws them once into the frames. Pass a single gaze, or "
                "rings=False.")
        # Painted into the displayed frames rather than left as an artist
        # behind the player's canvas, which would hide them:
        frames = self._native_rgb(gaze=gaze)
        overlay = _rings_overlay(self._frame_shape, radii, points[0],
                                 self.dva_to_pixel, color=ring_color)
        decorated = Percept(_over(frames, overlay), space=self._grid(),
                            time=self.time, time_unit=self.time_unit)
        return decorated.play(ax=ax, **kwargs)
