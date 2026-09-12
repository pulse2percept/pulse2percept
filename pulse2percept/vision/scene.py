""":py:class:`~pulse2percept.vision.Scene`"""
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, Rectangle
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

# Roughly how many raster nodes are interpolated at once. The interpolator
# works in float64, so a fine raster sampled in one go costs several times the
# float32 result it is written into.
_SAMPLE_BLOCK = 1 << 20

# Eccentricity spacing, in dva, that `rings=True` asks for
_RING_STEP = 5.0

# Ring color: dark enough to read on a light scene, gray enough to stay
# annotation rather than content
_RING_COLOR = '0.3'

# The only non-numeric `scotoma_fill`; see `_inpaint_rgb` for what it does.
_INPAINT = 'inpaint'

# The two `aperture` shapes; see `Scene._aperture_mask`. Both take their
# dimensions from `fov`, so the shape is all that is named here.
_RECTANGLE = 'rectangle'
_ELLIPSE = 'ellipse'

# Backing raster of a blank scene. Fixed: it is a display raster only, and
# making it configurable would let it set the aspect ratio a scalar `fov`
# resolves against. `Scene.render` chooses a render raster of its own.
_BLANK_SHAPE = (512, 512)


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


def _raster_axes(fov, shape):
    """Pixel-center coordinates of a raster spanning ``fov``

    ``fov`` is the raster's *outer* extent, so the outermost centers sit half
    an angular pixel inside it. ``xs`` ascends left to right and ``ys``
    descends, so ``ys[0]`` is row 0 at the top of the field.
    """
    n_rows, n_cols = shape
    width, height = fov
    dx, dy = width / n_cols, height / n_rows
    xs = (np.arange(n_cols, dtype=float) + 0.5) * dx - width / 2
    ys = height / 2 - (np.arange(n_rows, dtype=float) + 0.5) * dy
    return xs, ys


def _raster_step(xs, ys):
    """Angular pixel pitch ``(dx, dy)`` of a raster, as positive numbers"""
    def pitch(axis):
        # A single row or column has no spacing to read off:
        return abs(float(axis[1] - axis[0])) if axis.size > 1 else 1.0
    return pitch(xs), pitch(ys)


def _raster_extent(xs, ys):
    """Outer edges ``(left, right, bottom, top)`` of a raster, for `imshow`"""
    dx, dy = _raster_step(xs, ys)
    return (float(xs[0]) - dx / 2, float(xs[-1]) + dx / 2,
            float(ys[-1]) - dy / 2, float(ys[0]) + dy / 2)


def _raster_grid(xs, ys):
    """A Grid2D on the nodes of a raster, ``ys`` descending"""
    return Grid2D((float(xs[0]), float(xs[-1])),
                  (float(ys[-1]), float(ys[0])), step=_raster_step(xs, ys))


def _pad_axis(axis, step, pad):
    """Extend a regular axis by ``pad`` samples of signed ``step`` each end"""
    if pad == 0:
        return axis
    lead = axis[0] + step * np.arange(-pad, 0)
    tail = axis[-1] + step * np.arange(1, pad + 1)
    return np.concatenate((lead, axis, tail))


def _pixel_count(extent, step):
    """How many pixels of at most ``step`` degrees ``extent`` takes"""
    n = extent / step
    # A ratio that is only integral up to rounding must not buy a pixel:
    return max(int(np.ceil(n - 1e-9 * max(n, 1.0))), 1)


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


def _as_rgb(frame):
    """One ``(rows, cols, channels)`` frame as RGB, without copying gray"""
    if frame.shape[2] == 3:
        return frame
    return np.broadcast_to(frame, frame.shape[:2] + (3,))


def _take_frame(frames, frame):
    """One frame of a stack, kept as a stack; ``None`` keeps all of them"""
    return frames if frame is None else frames[..., frame:frame + 1]


def _take_time(time, frame):
    """The matching slice of a frame clock, which may be absent"""
    if frame is None or time is None:
        return time
    return np.asarray(time)[frame:frame + 1]


def _percept_axes(prosthetic):
    """The percept's eye-centered ``(xs, ys)`` axes in dva, both ascending"""
    ys = np.asarray(prosthetic.ydva, dtype=float)
    xs = np.asarray(prosthetic.xdva, dtype=float)
    if ys.size < 2 or xs.size < 2:
        raise ValueError(f"A percept needs extent in both directions to be "
                         f"placed in a scene, but this one's grid is "
                         f"{ys.size} x {xs.size}.")
    return xs, ys


def _percept_on(prosthetic, frames, xs, ys, gaze_xy):
    """Percept brightness at every node of a scene-coordinate raster

    Returns ``(rows, cols, n)``, one trailing entry per frame carried.
    """
    pxs, pys = _percept_axes(prosthetic)
    # `Grid2D` meshes its y axis reversed, so row 0 of the data holds the
    # largest y while `ydva` ascends. Flipping the rows puts the two back in
    # the same order, which is also the ascending one the interpolator wants.
    sample = RegularGridInterpolator((pys, pxs), frames[::-1],
                                     method='linear', bounds_error=False,
                                     fill_value=0)
    # The percept is eye-centered; the raster is in scene coordinates:
    gx, gy = gaze_xy
    x, y = np.meshgrid(xs - gx, ys - gy)
    points = np.column_stack((y.ravel(), x.ravel()))
    return sample(points).reshape((ys.size, xs.size, -1))


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


def _resolve_aperture(aperture):
    """Normalize ``aperture`` to ``_RECTANGLE`` or ``_ELLIPSE``"""
    for shape in (_RECTANGLE, _ELLIPSE):
        if aperture == shape:
            return shape
    raise ValueError(f"'aperture' is either {_RECTANGLE!r} or {_ELLIPSE!r}, "
                     f"not {aperture!r}.")


def _check_prosthetic(prosthetic):
    """Reject a percept that cannot be placed in a scene as brightness"""
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
        ``scotoma_blend``). ``'inpaint'`` is unavailable when composing a
        prosthetic percept because that interaction is not modeled.
    background : float or (r, g, b), optional
        Gray level or RGB value to use for transparent pixels. Defaults to
        black.
    scotoma_blend : float, optional
        Standard deviation, in degrees of visual angle, of a Gaussian blur
        applied to the rasterized loss map before it is drawn, softening the
        boundary from both sides. Defaults to 0.5 dva; 0 leaves it as sharp as
        the scotoma itself. Converted to pixels against whichever raster is
        being drawn, so the angular softness does not depend on resolution.
        Rendering only: the scotoma's geometry is unchanged.

        .. versionchanged:: 0.11.0
            Measured in degrees of visual angle rather than in scene pixels.
    aperture : {'rectangle', 'ellipse'}, optional
        Shape of the field's support. ``fov`` gives the aperture its
        dimensions and this gives it its shape: the default ``'rectangle'``
        fills the whole frame, while ``'ellipse'`` inscribes an eye-centered
        ellipse of semi-axes ``fov / 2`` in it, so a square ``fov`` renders as
        a disc. Support is a display boundary:
        :py:meth:`~pulse2percept.vision.Scene.plot` clips its artists to it and
        :py:meth:`~pulse2percept.vision.Scene.render` writes black outside it,
        while scene sampling, device input, stimulation and the prosthetic
        model response are untouched.

    Examples
    --------
    A logo covering 40 degrees, seen with a central 16-degree scotoma:

    >>> from pulse2percept.stimuli import samples
    >>> from pulse2percept.units import dva
    >>> from pulse2percept.vision import Scene, Scotoma
    >>> scene = Scene(samples.logo_bvl(), fov=40 * dva,
    ...               scotoma=Scotoma.circle(8 * dva))
    >>> scene.fov
    (40.0, 32.0)

    :py:meth:`~pulse2percept.vision.Scene.blank` gives a black field instead
    of a picture -- darkness, not blindness:

    >>> blank = Scene.blank()
    >>> blank.plot()                                    # doctest: +SKIP

    """

    def __init__(self, source, fov, scotoma=None, scotoma_fill=0,
                 scotoma_blend=0.5, background=0, aperture=_RECTANGLE):
        if not isinstance(source, (ImageStimulus, VideoStimulus)):
            # A picture is the common case:
            source = ImageStimulus(source)
        if scotoma is not None and not isinstance(scotoma, Scotoma):
            raise TypeError(f"'scotoma' must be a Scotoma object, not "
                            f"{type(scotoma)}.")
        fill = _resolve_fill(scotoma_fill)
        blend = float(as_value(scotoma_blend, dva, 'scotoma_blend'))
        if not np.isfinite(blend) or blend < 0:
            raise ValueError(f"'scotoma_blend' is a Gaussian sigma in degrees "
                             f"of visual angle and must be finite and "
                             f"non-negative, not {scotoma_blend}.")
        self._source = source
        self._background = _resolve_background(background)
        self._scotoma = scotoma
        self._scotoma_fill = fill
        self._scotoma_blend = blend
        self._aperture = _resolve_aperture(aperture)
        n_rows, n_cols = self._frame_shape
        self._fov = _resolve_fov(fov, n_rows, n_cols)
        self._cached_frames = None
        self._axes_cache = None
        self._pixel_centers_cache = None

    @classmethod
    def blank(cls, fov=45, **kwargs):
        """A uniformly black visual field

        Black is scene content -- a dark world -- not blindness: a device
        sampling it is given black. Use a
        :py:class:`~pulse2percept.vision.Scotoma` for vision that is lost.

        The black source sits on a fixed 512 x 512 raster, which is what
        :py:meth:`~pulse2percept.vision.Scene.render` falls back to; ask
        ``render`` for ``step`` or ``shape`` to get another output
        resolution. Neither sets the grid a prosthetic model predicts on.

        .. versionadded:: 0.11.0

        Parameters
        ----------
        fov : float or (width, height), optional
            Angular extent of the field, in dva. The backing raster is square,
            so the default 45 dva is a 45 x 45 dva disc.
        **kwargs :
            Any other :py:class:`~pulse2percept.vision.Scene` argument.
            ``aperture`` defaults to ``'ellipse'`` rather than
            ``'rectangle'``.

        Examples
        --------
        >>> from pulse2percept.units import dva
        >>> from pulse2percept.vision import Scene
        >>> blank = Scene.blank(fov=60 * dva)
        >>> blank.fov
        (60.0, 60.0)

        """
        kwargs.setdefault('aperture', _ELLIPSE)
        return cls(np.zeros(_BLANK_SHAPE, dtype=np.float32), fov=fov,
                   **kwargs)

    def _pprint_params(self):
        """Return a dict of class attributes to pretty-print"""
        params = {'source': type(self.source).__name__, 'fov': self.fov,
                  'shape': self.shape, 'scotoma': self.scotoma,
                  'background': self.background,
                  'scotoma_fill': self.scotoma_fill,
                  'scotoma_blend': self.scotoma_blend}
        # Omitted when rectangular, which is the default:
        if self.aperture != _RECTANGLE:
            params['aperture'] = self.aperture
        return params

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

        Prosthetic composition requires a numeric fill.
        """
        return self._scotoma_fill

    @property
    def scotoma_blend(self):
        """Gaussian sigma, in dva, softening the drawn scotoma boundary"""
        return self._scotoma_blend

    @property
    def aperture(self):
        """Shape of the field's support: ``'rectangle'`` or ``'ellipse'``"""
        return self._aperture

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

    def fellow_eye(self):
        """The homologous scene for the other eye

        Reflects eye-specific geometry (e.g., scotoma) across the vertical
        meridian. The scene image is not flipped.

        .. versionadded:: 0.11.0

        Returns
        -------
        scene : :py:class:`~pulse2percept.vision.Scene`
            A new scene. The original is left unchanged.

        Examples
        --------
        A loss 6 degrees into one eye's right hemifield sits 6 degrees into
        the other eye's left hemifield:

        >>> import numpy as np
        >>> from pulse2percept.units import dva
        >>> from pulse2percept.vision import Scene, Scotoma
        >>> left = Scene(np.zeros((8, 8)), fov=40 * dva,
        ...              scotoma=Scotoma.circle(3 * dva, center=(6, 0) * dva))
        >>> right = left.fellow_eye()
        >>> float(right.scotoma(-6, 0)), float(right.scotoma(6, 0))
        (1.0, 0.0)

        """
        scotoma = None if self.scotoma is None else self.scotoma.mirror()
        return Scene(self.source, self.fov, scotoma=scotoma,
                     scotoma_fill=self.scotoma_fill,
                     scotoma_blend=self.scotoma_blend,
                     background=self.background,
                     aperture=self.aperture)

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
        return self._sample_frames(self._frames(), x, y, gaze=gaze)

    def _sample_frames(self, frames, x, y, gaze=None):
        """`_sample_at` against a chosen stack of the source's frames"""
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

    @property
    def _axes(self):
        """Pixel-center axes ``(xs, ys)`` of the source raster, in scene dva"""
        if self._axes_cache is None:
            self._axes_cache = _raster_axes(self._fov, self._frame_shape)
        return self._axes_cache

    def _pixel_centers(self):
        """Scene coordinates of every pixel center, as ``(x, y)`` meshes"""
        if self._pixel_centers_cache is None:
            centers = np.meshgrid(*self._axes)
            for mesh in centers:
                mesh.flags.writeable = False
            self._pixel_centers_cache = centers
        return self._pixel_centers_cache

    def _source_frame(self, frame):
        """Which source frame an output frame reads; None means all of them"""
        if frame is None:
            return None
        # A still source stands behind every output frame:
        return frame if self.n_frames > 1 else 0

    def _source_on(self, xs, ys, frame=None):
        """The source at every node of a scene-coordinate raster

        ``(rows, cols, channels, n_frames)`` with 1 or 3 channels, as
        `_frames`. The source lives in scene coordinates, so this does not
        depend on gaze. ``frame`` restricts both the work and the result to
        that one source frame.
        """
        frames = _take_frame(self._frames(), frame)
        same_x = np.array_equal(xs, self._axes[0])
        if same_x and np.array_equal(ys, self._axes[1]):
            # The raster the source already sits on, so nothing is resampled
            # and the intact periphery comes through bit for bit:
            return frames
        n_rows, n_cols = ys.size, xs.size
        # Interpolated in row blocks: the interpolator works in float64, so a
        # fine raster done in one go costs several times the float32 result.
        block = max(1, _SAMPLE_BLOCK // max(n_cols, 1))
        out = None
        for lo in range(0, n_rows, block):
            hi = min(lo + block, n_rows)
            x, y = np.meshgrid(xs, ys[lo:hi])
            values = self._sample_frames(frames, x, y)
            if values.ndim == 2:
                # Grayscale: give it the channel axis `_frames` has
                values = values[:, np.newaxis, :]
            if out is None:
                out = np.empty((n_rows, n_cols) + values.shape[1:],
                               dtype=np.float32)
            out[lo:hi] = values.reshape((hi - lo, n_cols) + values.shape[1:])
        return out

    def _loss_on(self, xs, ys, gaze_xy):
        """Geometric loss at every node of a raster, in [0, 1], as float32"""
        if self.scotoma is None:
            return np.zeros((ys.size, xs.size), dtype=np.float32)
        # scene = visual field + gaze:
        gx, gy = gaze_xy
        x, y = np.meshgrid(xs - gx, ys - gy)
        return np.asarray(self.scotoma(x, y), dtype=np.float32)

    def _rendered_loss_on(self, xs, ys, gaze_xy):
        """The loss map as drawn: `_loss_on` softened by `scotoma_blend`

        The sigma is angular, so it is converted against this raster's own
        pixel pitch; anisotropic pixels get separate row and column sigmas.
        """
        sigma = self._scotoma_blend
        # An inpainted fill ignores the hard boundary:
        hard = self.scotoma is None or self._scotoma_fill == _INPAINT
        if hard or sigma == 0:
            return self._loss_on(xs, ys, gaze_xy)
        dx, dy = _raster_step(xs, ys)
        sigmas = (sigma / dy, sigma / dx)
        pads = tuple(int(np.ceil(_TRUNCATE * s)) + 1 for s in sigmas)
        # Blur the loss field, not a raster-sized crop of it:
        loss = self._loss_on(_pad_axis(xs, dx, pads[1]),
                             _pad_axis(ys, -dy, pads[0]), gaze_xy)
        blurred = gaussian_filter(loss, sigmas, mode='nearest',
                                  truncate=_TRUNCATE)
        return np.clip(blurred[pads[0]:-pads[0], pads[1]:-pads[1]], 0, 1)

    def _fill_rgb(self, frame_rgb, loss):
        """What complete loss shows for one ``(rows, cols, 3)`` frame"""
        if self._scotoma_fill != _INPAINT:
            return self._scotoma_fill
        return _inpaint_rgb(frame_rgb, loss > 0)

    def _aperture_mask(self, xs, ys, gaze_xy):
        """Raster nodes outside the eye-centered aperture

        The ellipse spans the full Scene FOV, with semi-axes `fov / 2`.
        `gaze_xy` sets its center in scene coordinates.
        """
        gx, gy = gaze_xy
        a, b = self._fov[0] / 2, self._fov[1] / 2
        across = ((xs - gx) / a) ** 2
        down = (((ys - gy) / b) ** 2)[:, np.newaxis]
        return across + down > 1

    def _apply_aperture(self, frames, xs, ys, gaze=None):
        """Black out ``(rows, cols, 3, n_frames)`` outside the aperture

        A display decision taken at the boundary of a finished raster: outside
        the aperture is undefined visual-field support, not black content.
        """
        if self._aperture == _RECTANGLE:
            return frames
        points = _gaze_points(gaze, frames.shape[-1])
        static = len(points) == 1
        mask = self._aperture_mask(xs, ys, points[0]) if static else None
        out = np.array(frames, dtype=np.float32)
        for f in range(frames.shape[-1]):
            outside = mask if static else self._aperture_mask(xs, ys,
                                                              points[f])
            out[outside, :, f] = 0
        return out

    def _support_patch(self, gaze_xy, transform):
        """The field's support as a patch, for clipping drawn artists"""
        width, height = self._fov
        if self._aperture == _ELLIPSE:
            # Eye-centered, so it sits wherever gaze points:
            return Ellipse(tuple(gaze_xy), width, height, transform=transform)
        # A rectangular aperture is the frame itself, which does not move:
        return Rectangle((-width / 2, -height / 2), width, height,
                         transform=transform)

    def _clip_to_support(self, artists, gaze_xy, transform):
        """Clip drawn artists to the field's support

        Outside the aperture is undefined visual-field support, so the
        boundary belongs to the artists rather than to their arrays. The
        source layer already covers exactly the rectangle, so only a local
        patch, which may reach past the field, needs clipping to that.
        """
        if self._aperture == _RECTANGLE:
            artists = artists[1:]
        if not artists:
            return
        clip = self._support_patch(gaze_xy, transform)
        for artist in artists:
            artist.set_clip_path(clip)

    def _native_on(self, xs, ys, gaze=None, frame=None):
        """Residual native vision on a raster, ``(rows, cols, 3, n_frames)``

        ``frame`` restricts the work to that one source frame, in which case
        ``gaze`` is the single pair that frame is seen with.
        """
        frames = self._source_on(xs, ys, frame=frame)
        if self.scotoma is None:
            return (frames if frames.shape[2] == 3
                    else np.repeat(frames, 3, axis=2))
        n_frames = frames.shape[-1]
        points = _gaze_points(gaze, n_frames)
        static = len(points) == 1
        if static:
            loss = self._rendered_loss_on(xs, ys, points[0])
        out = np.empty((ys.size, xs.size, 3, n_frames), dtype=np.float32)
        for f in range(n_frames):
            if not static:
                loss = self._rendered_loss_on(xs, ys, points[f])
            frame = _as_rgb(frames[..., f])
            # An inpainted fill reads this frame, so it is per-frame work:
            fill = self._fill_rgb(frame, loss)
            alpha = loss[..., np.newaxis]
            out[..., f] = (1 - alpha) * frame + alpha * fill
        return out

    def _native_rgb(self, gaze=None):
        """Residual native vision on the source raster, aperture not applied"""
        return self._native_on(*self._axes, gaze=gaze)

    def _composed_on(self, xs, ys, prosthetic, vmax, vmin=0, gaze=None,
                     frame=None):
        """Native vision on a raster with a prosthetic percept in the loss

        ``out = (1 - loss) * native + loss * max(fill, phosphene)``. Returns
        ``(frames, time, time_unit)``. ``frame`` restricts the work to that
        one output frame, in which case ``gaze`` is the single pair for it.
        """
        if self._scotoma_fill == _INPAINT:
            raise ValueError(
                f"scotoma_fill={_INPAINT!r} cannot be combined with a "
                f"prosthetic percept because their interaction is not "
                f"modeled. Use a numeric 'scotoma_fill' to compose one.")
        _check_prosthetic(prosthetic)
        vmin, vmax = _check_range(vmin, vmax)
        pframes, out_time, out_unit = self._prosthetic_frames(prosthetic,
                                                              frame=frame)
        n_out = pframes.shape[-1]
        points = _gaze_points(gaze, n_out)
        # Not `_native_on`: a grayscale source is broadcast to RGB per frame
        # below rather than copied into a second full-size array.
        source = self._source_on(xs, ys, frame=self._source_frame(frame))
        n_scene = source.shape[-1]
        n_rows, n_cols = ys.size, xs.size

        static = len(points) == 1
        if static:
            brightness = _percept_on(prosthetic, pframes, xs, ys, points[0])
            loss = self._rendered_loss_on(xs, ys, points[0])
        # Frame-major while composing: writing a whole frame at a time is
        # contiguous here:
        out = np.empty((n_out, n_rows, n_cols, 3), dtype=np.float32)
        for f in range(n_out):
            if static:
                frame = brightness[..., f]
            else:
                frame = _percept_on(prosthetic, pframes[..., f:f + 1],
                                    xs, ys, points[f])[..., 0]
                loss = self._rendered_loss_on(xs, ys, points[f])
            phosphene = np.clip((frame - vmin) / (vmax - vmin), 0, 1)
            native = _as_rgb(source[..., 0 if n_scene == 1 else f])
            fill = self._fill_rgb(native, loss)
            lost = np.maximum(fill, phosphene[..., np.newaxis])
            alpha = loss[..., np.newaxis]
            out[f] = (1 - alpha) * native + alpha * lost
        return (np.ascontiguousarray(np.moveaxis(out, 0, -1)), out_time,
                out_unit)

    def _prosthetic_on(self, xs, ys, prosthetic, vmax, vmin=0, gaze=None,
                       frame=None):
        """A prosthetic percept alone on black, on a scene-coordinate raster

        Places a percept where and at what size this field sees it. Not a
        composition: with no scotoma there is nothing to paint the percept
        into, and superimposing it on intact native vision would assert an
        interaction that is not modeled. ``frame`` restricts the work to that
        one output frame, in which case ``gaze`` is the single pair for it.
        """
        _check_prosthetic(prosthetic)
        vmin, vmax = _check_range(vmin, vmax)
        pframes, out_time, out_unit = self._prosthetic_frames(prosthetic,
                                                              frame=frame)
        n_out = pframes.shape[-1]
        points = _gaze_points(gaze, n_out)
        if len(points) == 1:
            brightness = _percept_on(prosthetic, pframes, xs, ys, points[0])
        else:
            brightness = np.concatenate(
                [_percept_on(prosthetic, pframes[..., f:f + 1], xs, ys,
                             points[f]) for f in range(n_out)], axis=-1)
        scaled = np.clip((brightness - vmin) / (vmax - vmin), 0, 1)
        rgb = np.repeat(scaled[:, :, np.newaxis, :], 3, axis=2)
        return np.asarray(rgb, dtype=np.float32), out_time, out_unit

    def _display_on(self, xs, ys, percept=None, vmax=None, vmin=0, gaze=None,
                    frame=None):
        """Display-ready RGB on a scene-coordinate raster, and its clock

        Residual native vision, or that with a prosthetic percept composed
        into the loss. The aperture is left to whatever draws the result.
        Returns ``(frames, time, time_unit)``. ``frame`` restricts the work to
        that one output frame, in which case ``gaze`` is the single pair for
        it.
        """
        if percept is None:
            if vmax is not None or vmin != 0:
                raise ValueError("'vmin' and 'vmax' map percept brightness "
                                 "onto a display, and there is no percept "
                                 "here. Pass 'percept'.")
            # Without a percept the output frames are the source's own:
            return (self._native_on(xs, ys, gaze=gaze, frame=frame),
                    _take_time(self.time, frame), self.time_unit)
        if self.scotoma is None:
            return self._prosthetic_on(xs, ys, percept, vmax, vmin=vmin,
                                       gaze=gaze, frame=frame)
        return self._composed_on(xs, ys, percept, vmax, vmin=vmin, gaze=gaze,
                                 frame=frame)

    def _n_display_frames(self, percept):
        """How many frames a drawn or rendered result has

        A video scene sets the clock, so a percept is read at its frame times;
        a still scene has whatever frames the percept brought.
        """
        if percept is None or self.time is not None:
            return self.n_frames
        return percept.data.shape[-1]

    def _prosthetic_frames(self, prosthetic, frame=None):
        """Line a percept up with the output frames, and say when they happen

        ``frame`` narrows the result to that one output frame and aligns it
        alone; the timing checks are made against the whole video either way.
        """
        if self.time is None:
            # A still scene has no clock of its own, so the percept's frames
            # are the output frames:
            return (_take_frame(prosthetic.data, frame),
                    _take_time(prosthetic.time, frame), prosthetic.time_unit)
        n_out = self.n_frames
        n_pros = prosthetic.data.shape[-1]
        out_time = _take_time(self.time, frame)
        if n_pros == 1 and prosthetic.time is None:
            # An untimed still percept stands behind every frame:
            return (np.repeat(prosthetic.data, 1 if frame is not None
                              else n_out, axis=-1), out_time, self.time_unit)
        if n_pros == n_out:
            return (_take_frame(prosthetic.data, frame),
                    _take_time(prosthetic.time, frame), prosthetic.time_unit)
        unit = prosthetic.time_unit
        # Checked against the whole video, not just the frame being drawn:
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
        asked_time = np.asarray(out_time, dtype=float)
        if frame is None:
            frames = prosthetic[..., Quantity(asked_time, self.time_unit)]
        else:
            # A scalar time index drops the frame axis, which is what a
            # one-element array index is read as too; put the axis back:
            frames = prosthetic[..., Quantity(float(asked_time[0]),
                                              self.time_unit)]
            frames = frames[..., np.newaxis]
        return frames, out_time, self.time_unit

    def _grid(self):
        """A Grid2D on the scene's pixel centers, in scene coordinates"""
        return _raster_grid(*self._axes)

    def _render_shape(self, step, shape):
        """The ``(rows, cols)`` a requested render raster comes out at"""
        if step is not None and shape is not None:
            raise ValueError("'step' and 'shape' both choose the render "
                             "raster; pass one or the other.")
        if shape is not None:
            shape = np.asarray(shape)
            bad = shape.shape != (2,) or shape.dtype.kind not in 'iu'
            if bad or shape.min() < 1:
                raise ValueError(f"'shape' must be a (rows, cols) pair of "
                                 f"positive integers, not {np.ravel(shape)}.")
            return (int(shape[0]), int(shape[1]))
        if step is None:
            # The source's own raster, so rendering resamples nothing unless
            # it is asked to:
            return self._frame_shape
        step = np.asarray(as_value(step, dva, 'step'), dtype=float)
        if step.ndim == 0:
            step = np.repeat(step, 2)
        bad = step.shape != (2,) or not np.all(np.isfinite(step))
        if bad or step.min() <= 0:
            raise ValueError(f"'step' is an angular sampling in degrees and "
                             f"must be a positive number or a (dx, dy) pair, "
                             f"not {np.ravel(step)}.")
        # Rounded up, so the rendered pixels are never coarser than asked:
        return (_pixel_count(self._fov[1], step[1]),
                _pixel_count(self._fov[0], step[0]))

    def render(self, percept=None, gaze=None, vmax=None, vmin=0, step=None,
               shape=None):
        """Rasterize this field onto one dense RGB percept

        Residual native vision, with a prosthetic ``percept`` composed into the
        loss where there is a scotoma, on one common raster. Use it when a
        single RGB image is needed, for saving or for downstream image
        processing; :py:meth:`~pulse2percept.vision.Scene.plot` draws the same
        content without forcing a shared resolution.

        The raster defaults to the source's own, which resamples nothing.
        ``step`` or ``shape`` chooses another; a fine ``step`` over a wide
        field is expensive by construction.

        .. versionadded:: 0.11.0

        Parameters
        ----------
        percept : :py:class:`~pulse2percept.percepts.Percept`, optional
            A brightness percept to place in this field, positioned by
            ``gaze``. With a scotoma it is composed into the loss as
            ``(1 - loss) * native + loss * max(scotoma_fill, phosphene)``;
            with none it is rendered alone on black, because superimposing it
            on intact native vision would assert an unmodeled interaction.
            ``scotoma_fill='inpaint'`` cannot be composed with one.
        gaze : (x, y) or (n_frames, 2), optional
            Where the eye is pointing: the scene location that falls on the
            fovea, in dva. Defaults to the origin.
        vmax : float, optional
            The percept brightness that displays as white. Required whenever
            ``percept`` is given: brightness is in arbitrary units.
        vmin : float, optional
            The percept brightness that displays as black. Defaults to 0.
        step : float or (dx, dy), optional
            Angular sampling of the render raster, in dva. Rounded up, so the
            rendered pixels are never coarser than this. Mutually exclusive
            with ``shape``.
        shape : (rows, cols), optional
            The render raster itself. Mutually exclusive with ``step``.

        Returns
        -------
        percept : :py:class:`~pulse2percept.percepts.Percept`
            An RGB percept in ``[0, 1]`` on the render raster, black outside
            the aperture. The source is left unchanged.

        Examples
        --------
        >>> import numpy as np
        >>> from pulse2percept.units import dva
        >>> from pulse2percept.vision import Scene
        >>> scene = Scene(np.zeros((60, 80)), fov=40 * dva)
        >>> scene.render().shape
        (60, 80, 3, 1)
        >>> scene.render(step=0.25 * dva).shape
        (120, 160, 3, 1)

        """
        xs, ys = _raster_axes(self._fov, self._render_shape(step, shape))
        frames, time, unit = self._display_on(xs, ys, percept=percept,
                                              vmax=vmax, vmin=vmin, gaze=gaze)
        return Percept(self._apply_aperture(frames, xs, ys, gaze=gaze),
                       space=_raster_grid(xs, ys), time=time, time_unit=unit)

    def plot(self, gaze=None, frame=0, ax=None, rings=False,
             ring_color=_RING_COLOR, percept=None, vmax=None, vmin=0,
             **kwargs):
        """Plot what is left of native vision

        The scene unchanged where vision is intact, and ``scotoma_fill`` where
        it is lost. A scotoma is eye-centered, so ``gaze`` decides where in the
        scene it falls.

        Passing a ``percept`` draws it in this field as well, so its size and
        place can be read against the FOV. Each layer keeps its own
        resolution: the source on the source raster, the percept as a local
        patch on its own visual-field grid. Neither is resampled onto a common
        raster; see :py:meth:`~pulse2percept.vision.Scene.render` for that.

        Inside the patch the layers compose as
        ``(1 - loss) * native + loss * max(scotoma_fill, phosphene)``, so
        where the percept is dark the patch is ordinary residual vision and
        its boundary does not show. With no scotoma the percept is drawn alone
        on black, because superimposing it on intact native vision would
        assert an unmodeled interaction.

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
        percept : :py:class:`~pulse2percept.percepts.Percept`, optional
            A brightness percept to draw in this field, placed by ``gaze`` and
            drawn at its own resolution over the source.
        vmax : float, optional
            The percept brightness that displays as white. Required whenever
            ``percept`` is given: brightness is in arbitrary units.
        vmin : float, optional
            The percept brightness that displays as black. Defaults to 0.
        **kwargs :
            Passed on to :py:meth:`~pulse2percept.percepts.Percept.plot`.

        Returns
        -------
        ax : matplotlib.axes.Axes

        """
        if percept is not None:
            _check_prosthetic(percept)
        n_out = self._n_display_frames(percept)
        if not 0 <= frame < n_out:
            raise ValueError(f"'frame' must be in 0..{n_out - 1}, not "
                             f"{frame}.")
        points = _gaze_points(gaze, n_out)
        # One frame is drawn, so one gaze and one frame of each layer is all
        # the work there is; the others are never evaluated.
        gaze_xy = points[0] if len(points) == 1 else points[frame]
        xs, ys = self._axes
        src_frame = self._source_frame(frame)
        patch = None
        if percept is None:
            # `_display_on` rejects a display range with nothing to map:
            wide = self._display_on(xs, ys, vmax=vmax, vmin=vmin,
                                    gaze=gaze_xy, frame=frame)[0][..., 0]
        else:
            pxs, pys = _percept_axes(percept)
            # Eye-centered percept coordinates, moved into the scene; `pys`
            # descends so that row 0 of the patch is its top, as drawn:
            pxs = pxs + gaze_xy[0]
            pys = pys[::-1] + gaze_xy[1]
            patch = self._display_on(pxs, pys, percept=percept, vmax=vmax,
                                     vmin=vmin, gaze=gaze_xy,
                                     frame=frame)[0][..., 0]
            if self.scotoma is None:
                # Nothing is lost, so there is no residual vision to draw the
                # percept into; only the field's extent is left to show.
                wide = np.zeros((ys.size, xs.size, 3), dtype=np.float32)
            else:
                wide = self._native_on(xs, ys, gaze=gaze_xy,
                                       frame=src_frame)[..., 0]
        still = Percept(wide[..., np.newaxis], space=self._grid())
        ax = still.plot(ax=ax, **kwargs)
        artists = [ax.images[-1]]
        if patch is not None:
            # `imshow` must not renegotiate the limits the wide layer set:
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            artists.append(ax.imshow(patch, origin='upper',
                                     extent=_raster_extent(pxs, pys),
                                     zorder=artists[0].get_zorder() + 1))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        self._clip_to_support(artists, gaze_xy, ax.transData)
        radii = _ring_radii(rings, self.fov)
        if radii.size:
            # The fovea sits wherever gaze points, which is where the scotoma
            # is drawn too; at the default gaze that is the scene's center.
            _draw_rings(ax, radii, gaze_xy, color=ring_color)
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
        # The player rasterizes its own frames, so this is display output:
        native = self.render(gaze=gaze)
        if not radii.size:
            return native.play(ax=ax, **kwargs)
        points = _gaze_points(gaze, self.n_frames)
        if len(points) > 1:
            raise ValueError(
                "Rings mark eccentricity from the fovea, so a gaze that moves "
                "between frames would have to move them too, and the player "
                "draws them once into the frames. Pass a single gaze, or "
                "rings=False.")
        # Painted into the displayed frames rather than left as an artist
        # behind the player's canvas, which would hide them:
        overlay = _rings_overlay(self._frame_shape, radii, points[0],
                                 self.dva_to_pixel, color=ring_color)
        decorated = Percept(_over(native.data, overlay), space=self._grid(),
                            time=self.time, time_unit=self.time_unit)
        return decorated.play(ax=ax, **kwargs)
