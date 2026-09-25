""":py:class:`~pulse2percept.vision.Scene`"""
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Ellipse, Rectangle
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from skimage.color import rgb2gray
from skimage.restoration import inpaint_biharmonic

from .gaze import Gaze, _gaze_points
from .scotoma import Scotoma
from ..percepts import Percept
from ..percepts.base import _resolve_clim
from ..stimuli import ImageStimulus, VideoStimulus
from ..topography import Grid2D
from ..units import Quantity, as_value, dimensionless, dva, ms
from ..utils import PrettyPrint
from ..utils import _visual_field as vf

# How many sigmas of the blur kernel are kept; also how far off-frame the loss
# map is rasterized, so the cropped result is free of edge effects:
_TRUNCATE = 4.0

# Roughly how many raster nodes are interpolated at once. The interpolator
# works in float64, so a fine raster sampled in one go costs several times the
# float32 result it is written into.
_SAMPLE_BLOCK = 1 << 20

# The one `scotoma_fill` string that is not a color; see `_inpaint_rgb`.
_INPAINT = 'inpaint'

# The two `aperture` shapes; see `Scene._aperture_mask`. Both take their
# dimensions from `fov`, so the shape is all that is named here.
_RECTANGLE = 'rectangle'
_ELLIPSE = 'ellipse'

# Backing raster of a blank scene. Fixed: it is a display raster only, and
# making it configurable would let it set the aspect ratio the inferred
# `extent` follows. `Scene.render` chooses a render raster of its own.
_BLANK_SHAPE = (512, 512)


def _resolve_fov(fov):
    """Normalize ``fov`` to ``(width, height)`` in dva; a scalar is square"""
    fov = np.asarray(as_value(fov, dva, 'fov'), dtype=float)
    if fov.ndim == 0:
        fov = np.repeat(fov, 2)
    elif fov.shape != (2,):
        raise ValueError(f"'fov' must be a scalar (a square window) or a "
                         f"(width, height) pair, not {fov.tolist()}.")
    for name, f in zip(('width', 'height'), fov):
        if not np.isfinite(f) or f <= 0:
            raise ValueError(f"'fov' {name} must be a finite positive number "
                             f"of degrees, not {f}.")
    return (float(fov[0]), float(fov[1]))


def _resolve_extent(extent, fov, n_rows, n_cols):
    """Scene extent ``(left, right, bottom, top)`` in dva

    If omitted: centered, square pixels, and the smallest such extent that
    contains ``fov``.
    """
    if extent is None:
        width, height = fov
        # The binding dimension is copied, not recomputed, so it stays exact:
        if width / n_cols >= height / n_rows:
            height = width * n_rows / n_cols
        else:
            width = height * n_cols / n_rows
        return (-width / 2, width / 2, -height / 2, height / 2)
    values = np.asarray(as_value(extent, dva, 'extent'), dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError(f"'extent' must be four finite numbers (left, right, "
                         f"bottom, top) in dva, not {np.ravel(values)}.")
    left, right, bottom, top = (float(v) for v in values)
    if right <= left or top <= bottom:
        raise ValueError(f"'extent' requires left < right and bottom < top, "
                         f"not {(left, right, bottom, top)}.")
    return (left, right, bottom, top)


def _centered(size):
    """The ``(left, right, bottom, top)`` of a ``(width, height)`` centered
    on the origin"""
    width, height = size
    return (-width / 2, width / 2, -height / 2, height / 2)


def _raster_axes(extent, shape):
    """Pixel-center coordinates of a raster spanning ``extent``

    ``extent`` is the raster's *outer* ``(left, right, bottom, top)``, so the
    outermost centers sit half an angular pixel inside it. ``xs`` ascends left
    to right and ``ys`` descends, so ``ys[0]`` is row 0 at the top.
    """
    n_rows, n_cols = shape
    left, right, bottom, top = extent
    dx, dy = (right - left) / n_cols, (top - bottom) / n_rows
    xs = left + (np.arange(n_cols, dtype=float) + 0.5) * dx
    ys = top - (np.arange(n_rows, dtype=float) + 0.5) * dy
    return xs, ys


def _same_axis(axis, other):
    """Whether two regular axes have the same nodes, to 1e-9 of a pixel"""
    if axis.shape != other.shape:
        return False
    tol = 1e-9 * (abs(float(other[1] - other[0])) if other.size > 1 else 1.0)
    return bool(np.allclose(axis, other, rtol=0, atol=tol))


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


def _percept_on(prosthetic, frames, xs, ys):
    """Percept brightness at every node of an eye-centered raster

    Returns ``(rows, cols, n)``, one trailing entry per frame carried.
    """
    pxs, pys = _percept_axes(prosthetic)
    # `Grid2D` meshes its y axis reversed, so row 0 of the data holds the
    # largest y while `ydva` ascends. Flipping the rows puts the two back in
    # the same order, which is also the ascending one the interpolator wants.
    sample = RegularGridInterpolator((pys, pxs), frames[::-1],
                                     method='linear', bounds_error=False,
                                     fill_value=0)
    x, y = np.meshgrid(xs, ys)
    points = np.column_stack((y.ravel(), x.ravel()))
    return sample(points).reshape((ys.size, xs.size, -1))


def _resolve_fill(scotoma_fill):
    """Normalize scotoma_fill to a gray level, RGB tuple, or _INPAINT."""
    if isinstance(scotoma_fill, str):
        if scotoma_fill == _INPAINT:
            return _INPAINT
        try:
            return tuple(float(c) for c in to_rgb(scotoma_fill))
        except ValueError:
            raise ValueError(f"'scotoma_fill' is a display intensity in "
                             f"[0, 1], a Matplotlib color, or {_INPAINT!r}, "
                             f"not {scotoma_fill!r}.") from None

    fill = np.asarray(as_value(scotoma_fill, dimensionless, 'scotoma_fill'),
                      dtype=float)
    if fill.ndim != 0 and fill.shape != (3,):
        raise ValueError(f"'scotoma_fill' is a display intensity in [0, 1], "
                         f"an (r, g, b) triple, a Matplotlib color, or "
                         f"{_INPAINT!r}, not {scotoma_fill!r}.")
    if not np.all(np.isfinite(fill)) or fill.min() < 0 or fill.max() > 1:
        raise ValueError(f"'scotoma_fill' is a display intensity and must "
                         f"lie in [0, 1], not {scotoma_fill}.")
    return float(fill) if fill.ndim == 0 else tuple(fill.tolist())

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


def _check_range(data, vmin, vmax):
    """Brightness limits for displaying ``data``, omitted ones filled in

    Omitted limits are 0 and the maximum over all of ``data``, so every frame
    shares one scale.
    """
    auto = vmax is None
    vmin, vmax = _resolve_clim(data, vmin, vmax, auto_vmin=0)
    if auto and vmax == vmin:
        # A constant percept at vmin. Any positive span maps it to black, as
        # Matplotlib does for vmin == vmax:
        return vmin, vmin + 1.0
    if vmax <= vmin:
        raise ValueError(f"'vmax' ({vmax}) must be greater than 'vmin' "
                         f"({vmin}); the percept is in arbitrary brightness "
                         f"units, and this is what says which of them is "
                         f"white.")
    return vmin, vmax


class Scene(PrettyPrint):
    """What is visually present, and where native vision is lost

    A scene places a picture in the world at a fixed angular ``extent``, and
    views it through a field of view (``fov``) centered on the fovea,
    optionally with a region where native vision is missing.

    Geometry follows one convention:

    *  ``extent`` is the *outer* ``(left, right, bottom, top)`` of the source
       in scene coordinates, so it reaches half an angular pixel past the
       outermost pixel centers.
    *  Pixel coordinates address pixel *centers*.
    *  Row 0 is the top of the frame and therefore the largest ``y``.

    The FOV, the aperture, the scotoma and an implant are *eye-centered*:
    fixed relative to the fovea. Gaze moves the viewing window through the
    scene; the source does not move or rescale::

        (x_scene, y_scene) = (x_eye, y_eye) + (x_gaze, y_gaze)

    :py:meth:`~pulse2percept.vision.Scene.render`,
    :py:meth:`~pulse2percept.vision.Scene.plot` and
    :py:meth:`~pulse2percept.vision.Scene.play` show the FOV in eye-centered
    coordinates, spanning ``[-fov / 2, fov / 2]``. Parts of the FOV beyond
    ``extent`` are black. Gaze also sets what the device is given to encode,
    unless the implant's
    :py:attr:`~pulse2percept.implants.Implant.scene_input_frame` is
    ``'head'`` (a head-fixed camera the eye cannot move). Device input is
    sampled from the whole ``extent``, not only the FOV.

    Source, ``extent`` and ``fov`` are fixed after construction.

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
        Size of the viewing window, in degrees of visual angle, centered on
        the fovea (e.g. ``40 * dva``). A scalar is a square window.
    extent : (left, right, bottom, top), optional
        Where the source sits in scene coordinates, in dva. If None, the
        source is centered with square pixels and scaled to the smallest
        extent that contains ``fov``: with a scalar ``fov``, the shorter
        source dimension spans ``fov``.
    scotoma : :py:class:`~pulse2percept.vision.Scotoma`, optional
        The region where native vision is lost. If None, native vision is
        intact everywhere and the scene is simply what is out there.
    scotoma_fill : float, color, or 'inpaint', optional
        Scotoma fill: a gray level in [0, 1] (default 0, black), an
        `(r, g, b)` triple in [0, 1], or any Matplotlib color string.
        `'inpaint'` fills from the surrounding image using
        :py:func:`skimage.restoration.inpaint_biharmonic` and ignores
        `scotoma_blend`. It cannot be used when composing a prosthetic percept.
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
        Shape of the FOV. ``fov`` sets its size and this sets its shape: the
        default ``'rectangle'`` shows the whole window, while ``'ellipse'``
        inscribes an eye-centered ellipse of semi-axes ``fov / 2`` in it, so a
        square ``fov`` renders as a disc. The aperture is a display boundary:
        :py:meth:`~pulse2percept.vision.Scene.plot` clips its artists to it and
        :py:meth:`~pulse2percept.vision.Scene.render` writes black outside it,
        while scene sampling, device input, stimulation and the prosthetic
        model response are untouched.

    Examples
    --------
    A 40-degree window onto a logo, seen with a central 16-degree scotoma.
    The logo is landscape, so its height spans the window:

    >>> from pulse2percept.stimuli import samples
    >>> from pulse2percept.units import dva
    >>> from pulse2percept.vision import Scene, Scotoma
    >>> scene = Scene(samples.logo_bvl(), fov=40 * dva,
    ...               scotoma=Scotoma.circle(8 * dva))
    >>> scene.fov, scene.extent
    ((40.0, 40.0), (-25.0, 25.0, -20.0, 20.0))

    A 40-degree disc onto a wider world. ``render`` covers the FOV at the
    source's angular pixel pitch:

    >>> world = Scene(samples.logo_bvl(), extent=(-50, 50, -40, 40) * dva,
    ...               fov=40 * dva, aperture='ellipse')
    >>> world.render().shape
    (288, 288, 3, 1)

    :py:meth:`~pulse2percept.vision.Scene.blank` gives a black field instead
    of a picture -- darkness, not blindness:

    >>> blank = Scene.blank()
    >>> blank.plot()                                    # doctest: +SKIP

    """

    def __init__(self, source, fov, extent=None, scotoma=None, scotoma_fill=0,
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
        self._fov = _resolve_fov(fov)
        self._extent = _resolve_extent(extent, self._fov, n_rows, n_cols)
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
            Viewing window, in dva. The default 45 dva is a 45 x 45 dva disc.
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
                  'extent': self.extent, 'shape': self.shape,
                  'scotoma': self.scotoma,
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
        """Scotoma fill as a gray level, `(r, g, b)` triple, or `'inpaint'`.

        Matplotlib color strings are stored as RGB triples.
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
        """Viewing window ``(width, height)``, in dva, centered on the fovea"""
        return self._fov

    @property
    def extent(self):
        """Source bounds ``(left, right, bottom, top)``, in scene dva"""
        return self._extent

    @property
    def _view_extent(self):
        """The FOV as ``(left, right, bottom, top)``, in eye-centered dva"""
        return _centered(self._fov)

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
        left, right, bottom, top = self._extent
        return ((right - left) / n_cols, (top - bottom) / n_rows)

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
            Scene coordinates in degrees of visual angle, as set by
            ``extent``. ``y`` grows upwards, so row 0 has the largest ``y``.

        """
        dx, dy = self._angular_pixel
        left, _, _, top = self._extent
        col = np.asarray(col, dtype=float)
        row = np.asarray(row, dtype=float)
        x = left + (col + 0.5) * dx
        y = top - (row + 0.5) * dy
        return x, y

    def dva_to_pixel(self, x, y):
        """Pixel coordinates of a point in the scene

        The inverse of :py:meth:`~pulse2percept.vision.Scene.pixel_to_dva`.

        Parameters
        ----------
        x, y : float or array_like
            Scene coordinates in degrees of visual angle, as set by
            ``extent``.

        Returns
        -------
        col, row : np.ndarray
            Continuous pixel coordinates, where ``(0, 0)`` is the center of the
            top-left pixel. They are not rounded and not clipped to the frame:
            a point outside ``extent`` maps outside the pixel grid.

        """
        dx, dy = self._angular_pixel
        left, _, _, top = self._extent
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        col = (x - left) / dx - 0.5
        row = (top - y) / dy - 0.5
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
        return Scene(self.source, self.fov, extent=self.extent,
                     scotoma=scotoma,
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
            self._axes_cache = _raster_axes(self._extent, self._frame_shape)
        return self._axes_cache

    def _view_axes(self):
        """Eye-centered pixel-center axes of the default render raster"""
        return _raster_axes(self._view_extent, self._render_shape(None, None))

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

    def _source_on(self, xs, ys, gaze_xy=(0.0, 0.0), frame=None):
        """The source at every node of an eye-centered raster, seen with one
        gaze

        ``(rows, cols, channels, n_frames)`` with 1 or 3 channels, as
        `_frames`. ``frame`` restricts both the work and the result to that
        one source frame.
        """
        frames = _take_frame(self._frames(), frame)
        gx, gy = gaze_xy
        src_xs, src_ys = self._axes
        if _same_axis(xs + gx, src_xs) and _same_axis(ys + gy, src_ys):
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
            values = self._sample_frames(frames, x, y, gaze=(gx, gy))
            if values.ndim == 2:
                # Grayscale: give it the channel axis `_frames` has
                values = values[:, np.newaxis, :]
            if out is None:
                out = np.empty((n_rows, n_cols) + values.shape[1:],
                               dtype=np.float32)
            out[lo:hi] = values.reshape((hi - lo, n_cols) + values.shape[1:])
        return out

    def _loss_on(self, xs, ys):
        """Geometric loss at every node of an eye-centered raster, in [0, 1],
        as float32"""
        if self.scotoma is None:
            return np.zeros((ys.size, xs.size), dtype=np.float32)
        x, y = np.meshgrid(xs, ys)
        return np.asarray(self.scotoma(x, y), dtype=np.float32)

    def _rendered_loss_on(self, xs, ys):
        """The loss map as drawn: `_loss_on` softened by `scotoma_blend`

        The sigma is angular, so it is converted against this raster's own
        pixel pitch; anisotropic pixels get separate row and column sigmas.
        """
        sigma = self._scotoma_blend
        # An inpainted fill ignores the hard boundary:
        hard = self.scotoma is None or self._scotoma_fill == _INPAINT
        if hard or sigma == 0:
            return self._loss_on(xs, ys)
        dx, dy = _raster_step(xs, ys)
        sigmas = (sigma / dy, sigma / dx)
        pads = tuple(int(np.ceil(_TRUNCATE * s)) + 1 for s in sigmas)
        # Blur the loss field, not a raster-sized crop of it:
        loss = self._loss_on(_pad_axis(xs, dx, pads[1]),
                             _pad_axis(ys, -dy, pads[0]))
        blurred = gaussian_filter(loss, sigmas, mode='nearest',
                                  truncate=_TRUNCATE)
        return np.clip(blurred[pads[0]:-pads[0], pads[1]:-pads[1]], 0, 1)

    def _fill_rgb(self, frame_rgb, loss):
        """Scotoma fill for one ``(rows, cols, 3)`` frame"""
        if self._scotoma_fill != _INPAINT:
            return self._scotoma_fill
        return _inpaint_rgb(frame_rgb, loss > 0)

    def _aperture_mask(self, xs, ys):
        """Nodes of an eye-centered raster outside the aperture

        The ellipse is inscribed in the FOV, with semi-axes `fov / 2`.
        """
        a, b = self._fov[0] / 2, self._fov[1] / 2
        return (xs / a) ** 2 + ((ys / b) ** 2)[:, np.newaxis] > 1

    def _apply_aperture(self, frames, xs, ys):
        """Black out ``(rows, cols, 3, n_frames)`` outside the aperture

        A display decision taken at the boundary of a finished raster: outside
        the aperture is undefined visual-field support, not black content.
        """
        if self._aperture == _RECTANGLE:
            return frames
        out = np.array(frames, dtype=np.float32)
        out[self._aperture_mask(xs, ys)] = 0
        return out

    def _support_patch(self, transform):
        """The FOV's support as an eye-centered patch, for clipping artists"""
        width, height = self._fov
        if self._aperture == _ELLIPSE:
            return Ellipse((0, 0), width, height, transform=transform)
        return Rectangle((-width / 2, -height / 2), width, height,
                         transform=transform)

    def _clip_to_support(self, artists, transform):
        """Clip drawn artists to the FOV's support

        Outside the aperture is undefined visual-field support, so the
        boundary belongs to the artists rather than to their arrays. The
        FOV layer already covers exactly the rectangle, so only a local
        patch, which may reach past the FOV, needs clipping to that.
        """
        if self._aperture == _RECTANGLE:
            artists = artists[1:]
        if not artists:
            return
        clip = self._support_patch(transform)
        for artist in artists:
            artist.set_clip_path(clip)

    def _native_on(self, xs, ys, gaze=None, frame=None):
        """Residual native vision on an eye-centered raster,
        ``(rows, cols, 3, n_frames)``

        ``frame`` restricts the work to that one source frame, in which case
        ``gaze`` is the single pair that frame is seen with.
        """
        n_frames = self.n_frames if frame is None else 1
        points = _gaze_points(gaze, n_frames)
        if len(points) == 1:
            frames = self._source_on(xs, ys, points[0], frame=frame)
        else:
            # Gaze moves the window through the source, frame by frame:
            frames = np.concatenate([self._source_on(xs, ys, points[f],
                                                     frame=f)
                                     for f in range(n_frames)], axis=-1)
        if self.scotoma is None:
            return (frames if frames.shape[2] == 3
                    else np.repeat(frames, 3, axis=2))
        loss = self._rendered_loss_on(xs, ys)
        alpha = loss[..., np.newaxis]
        out = np.empty((ys.size, xs.size, 3, n_frames), dtype=np.float32)
        for f in range(n_frames):
            rgb = _as_rgb(frames[..., f])
            # An inpainted fill reads this frame, so it is per-frame work:
            fill = self._fill_rgb(rgb, loss)
            out[..., f] = (1 - alpha) * rgb + alpha * fill
        return out

    def _native_rgb(self, gaze=None):
        """Residual native vision on the default FOV raster, aperture not
        applied"""
        return self._native_on(*self._view_axes(), gaze=gaze)

    def _composed_on(self, xs, ys, prosthetic, vmax, vmin=None, gaze=None,
                     frame=None):
        """Native vision on an eye-centered raster with a prosthetic percept in
        the loss

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
        vmin, vmax = _check_range(prosthetic.data, vmin, vmax)
        pframes, out_time, out_unit = self._prosthetic_frames(prosthetic,
                                                              frame=frame)
        n_out = pframes.shape[-1]
        points = _gaze_points(gaze, n_out)
        static = len(points) == 1
        if static:
            # Not `_native_on`: a grayscale source is broadcast to RGB per
            # frame below rather than copied into a second full-size array.
            source = self._source_on(xs, ys, points[0],
                                     frame=self._source_frame(frame))
            n_scene = source.shape[-1]
        n_rows, n_cols = ys.size, xs.size
        # Both eye-centered, so neither depends on gaze:
        brightness = _percept_on(prosthetic, pframes, xs, ys)
        loss = self._rendered_loss_on(xs, ys)
        alpha = loss[..., np.newaxis]
        # Frame-major while composing: writing a whole frame at a time is
        # contiguous here:
        out = np.empty((n_out, n_rows, n_cols, 3), dtype=np.float32)
        for f in range(n_out):
            phosphene = np.clip((brightness[..., f] - vmin) / (vmax - vmin),
                                0, 1)
            if static:
                native = source[..., 0 if n_scene == 1 else f]
            else:
                native = self._source_on(xs, ys, points[f],
                                         frame=self._source_frame(f))[..., 0]
            native = _as_rgb(native)
            fill = self._fill_rgb(native, loss)
            lost = np.maximum(fill, phosphene[..., np.newaxis])
            out[f] = (1 - alpha) * native + alpha * lost
        return (np.ascontiguousarray(np.moveaxis(out, 0, -1)), out_time,
                out_unit)

    def _prosthetic_on(self, xs, ys, prosthetic, vmax, vmin=None, gaze=None,
                       frame=None):
        """A prosthetic percept alone on black, on an eye-centered raster

        Places a percept where and at what size this field sees it. Not a
        composition: with no scotoma there is nothing to paint the percept
        into, and superimposing it on intact native vision would assert an
        interaction that is not modeled. ``frame`` restricts the work to that
        one output frame, in which case ``gaze`` is the single pair for it.
        """
        _check_prosthetic(prosthetic)
        vmin, vmax = _check_range(prosthetic.data, vmin, vmax)
        pframes, out_time, out_unit = self._prosthetic_frames(prosthetic,
                                                              frame=frame)
        # Validated only: percept and raster are both eye-centered.
        _gaze_points(gaze, pframes.shape[-1])
        brightness = _percept_on(prosthetic, pframes, xs, ys)
        scaled = np.clip((brightness - vmin) / (vmax - vmin), 0, 1)
        rgb = np.repeat(scaled[:, :, np.newaxis, :], 3, axis=2)
        return np.asarray(rgb, dtype=np.float32), out_time, out_unit

    def _display_on(self, xs, ys, percept=None, vmax=None, vmin=None, gaze=None,
                    frame=None):
        """Display-ready RGB on an eye-centered raster, and its clock

        Residual native vision, or that with a prosthetic percept composed
        into the loss. The aperture is left to whatever draws the result.
        Returns ``(frames, time, time_unit)``. ``frame`` restricts the work to
        that one output frame, in which case ``gaze`` is the single pair for
        it.
        """
        if percept is None:
            if vmax is not None or vmin is not None:
                raise ValueError("'vmin' and 'vmax' map percept brightness "
                                 "onto a display, and there is no percept "
                                 "here. Pass 'percept'.")
            # Without a percept the output frames are the source's own:
            out_time, out_unit, _ = self._output_clock()
            return (self._native_on(xs, ys, gaze=gaze, frame=frame),
                    _take_time(out_time, frame), out_unit)
        if self.scotoma is None:
            return self._prosthetic_on(xs, ys, percept, vmax, vmin=vmin,
                                       gaze=gaze, frame=frame)
        return self._composed_on(xs, ys, percept, vmax, vmin=vmin, gaze=gaze,
                                 frame=frame)

    def _output_clock(self, percept=None):
        """``(time, time_unit, n_frames)`` the output frames happen on

        The one place the display clock is decided: a video scene owns it, so
        a percept is read at the scene's frame times; a still scene has no
        clock of its own and takes the percept's, which may be ``None``.
        These are the instants gaze resolves against. What a returned Percept
        is *labeled* with can differ: see `_prosthetic_frames`.
        """
        if self.time is not None:
            return self.time, self.time_unit, self.n_frames
        if percept is None:
            # No clock, but the source still names the unit one would count in:
            return None, self.time_unit, self.n_frames
        return percept.time, percept.time_unit, percept.data.shape[-1]

    def _n_display_frames(self, percept):
        """How many frames a drawn or rendered result has"""
        return self._output_clock(percept)[2]

    def _resolve_gaze(self, gaze, percept=None):
        """A `Gaze` as one (x, y) per output frame; other forms pass through"""
        if not isinstance(gaze, Gaze):
            return gaze
        time, unit, n_out = self._output_clock(percept)
        return _gaze_points(gaze, n_out, time=time, time_unit=unit)

    def _source_aligned(self, prosthetic):
        """Whether percept frame k was predicted from this scene's frame k

        Reads only the top-level ``metadata['source_frame_time']`` (ms) that
        automatic temporal output records; equal frame counts are not enough.
        """
        meta = prosthetic.metadata
        source = (meta.get('source_frame_time') if isinstance(meta, dict)
                  else None)
        if source is None:
            return False
        source = np.asarray(source, dtype=float).ravel()
        mine = np.asarray(self.source.times(ms), dtype=float)
        return (source.size == mine.size == prosthetic.data.shape[-1] and
                np.allclose(source, mine, rtol=1e-9, atol=1e-6))

    def _prosthetic_frames(self, prosthetic, frame=None):
        """Line a percept up with the output frames, and say when they happen

        ``frame`` narrows the result to that one output frame and aligns it
        alone; the timing checks are made against the whole video either way.
        """
        out_time, out_unit, n_out = self._output_clock(prosthetic)
        out_time = _take_time(out_time, frame)
        if self.time is None:
            # A still scene has no clock of its own, so the percept's frames
            # are the output frames:
            return _take_frame(prosthetic.data, frame), out_time, out_unit
        n_pros = prosthetic.data.shape[-1]
        if n_pros == 1 and prosthetic.time is None:
            # An untimed still percept stands behind every frame:
            return (np.repeat(prosthetic.data, 1 if frame is not None
                              else n_out, axis=-1), out_time, out_unit)
        if self._source_aligned(prosthetic):
            # Frame for frame, but labeled with the percept's own times: a
            # temporal model may report each frame at its end, not its onset.
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
        return frames, out_time, out_unit

    def _grid(self):
        """A Grid2D on the default FOV raster, in eye-centered coordinates"""
        return _raster_grid(*self._view_axes())

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
            # The source's own angular pitch across the FOV:
            step = self._angular_pixel
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

    def render(self, percept=None, gaze=None, vmax=None, vmin=None, step=None,
               shape=None):
        """Rasterize this field onto one dense RGB percept

        Residual native vision, with a prosthetic ``percept`` composed into the
        loss where there is a scotoma, on one common raster. Use it when a
        single RGB image is needed, for saving or for downstream image
        processing; :py:meth:`~pulse2percept.vision.Scene.plot` draws the same
        content without forcing a shared resolution.

        The raster spans the FOV in eye-centered coordinates,
        ``[-fov / 2, fov / 2]``, whatever the gaze; gaze selects which part of
        the source fills it, and parts beyond ``extent`` are black. Its pitch
        defaults to the source's angular pixel pitch. The source is resampled
        unless the raster lands on its pixel centers (e.g. ``fov`` matching
        ``extent``, at zero gaze). ``step`` or ``shape`` chooses another
        raster; a fine ``step`` over a wide field is expensive by
        construction.

        .. versionadded:: 0.11.0

        Parameters
        ----------
        percept : :py:class:`~pulse2percept.percepts.Percept`, optional
            An eye-centered brightness percept, drawn at its visual-field
            coordinates within the FOV. With a scotoma it is composed into
            the loss as
            ``(1 - loss) * native + loss * max(scotoma_fill, phosphene)``;
            with none it is rendered alone on black, because superimposing it
            on intact native vision would assert an unmodeled interaction.
            ``scotoma_fill='inpaint'`` cannot be composed with one.
        gaze : (x, y), (n_frames, 2), or :py:class:`~pulse2percept.vision.Gaze`, optional
            Where the eye is pointing: the scene location that falls on the
            fovea, in dva. Defaults to the origin. A
            :py:class:`~pulse2percept.vision.Gaze` is resolved against the
            output clock: a video scene's frame times, or a timed percept's
            for a still scene.
        vmax : float, optional
            The percept brightness that displays as white. Defaults to the
            maximum brightness across the whole ``percept``.
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
        >>> scene = Scene(np.zeros((60, 80)), fov=(40, 30) * dva)
        >>> scene.render().shape
        (60, 80, 3, 1)
        >>> scene.render(step=0.25 * dva).shape
        (120, 160, 3, 1)

        """
        gaze = self._resolve_gaze(gaze, percept)
        xs, ys = _raster_axes(self._view_extent,
                              self._render_shape(step, shape))
        frames, time, unit = self._display_on(xs, ys, percept=percept,
                                              vmax=vmax, vmin=vmin, gaze=gaze)
        return Percept(self._apply_aperture(frames, xs, ys),
                       space=_raster_grid(xs, ys), time=time, time_unit=unit)

    def plot(self, gaze=None, frame=0, ax=None, rings=False, meridians=False,
             grid_color=vf.GRID_COLOR, percept=None, vmax=None, vmin=None,
             **kwargs):
        """Plot what is left of native vision

        The FOV in eye-centered coordinates, as
        :py:meth:`~pulse2percept.vision.Scene.render` shows it: the scene
        where vision is intact and ``scotoma_fill`` where it is lost. ``gaze``
        selects which part of the scene fills the FOV.

        Passing a ``percept`` draws it in this field as well, so its size and
        place can be read against the FOV. Each layer keeps its own
        resolution: the source at its own angular pitch, the percept as a
        local patch on its own visual-field grid. Neither is resampled onto a
        common raster; see :py:meth:`~pulse2percept.vision.Scene.render` for
        that.

        Inside the patch the layers compose as
        ``(1 - loss) * native + loss * max(scotoma_fill, phosphene)``, so
        where the percept is dark the patch is ordinary residual vision and
        its boundary does not show. With no scotoma the percept is drawn alone
        on black, because superimposing it on intact native vision would
        assert an unmodeled interaction.

        Parameters
        ----------
        gaze : (x, y) or :py:class:`~pulse2percept.vision.Gaze`, optional
            Where the eye is pointing: the scene location that falls on the
            fovea, in dva. Defaults to the origin. A
            :py:class:`~pulse2percept.vision.Gaze` is resolved against the
            output clock, and the fixation held at ``frame`` is drawn.
        frame : int, optional
            Which frame of a video scene to draw. Ignored for a still scene.
        ax : matplotlib.axes.Axes, optional
            The axes to draw on. If None, uses the current axes.
        rings : bool, float, or sequence, optional
            Eccentricity rings (dva) about the fovea, at the center of the
            FOV. True draws 1.25, 2.5, 5, 10, 20, ... dva, a number is a
            spacing, and a sequence is the eccentricities themselves.
            Automatic rings stop at the nearest FOV edge.
        meridians : bool, float, or sequence, optional
            Polar-angle meridians (geometric deg) from the fovea to the FOV
            edge: 0 is +x, 90 is +y, counterclockwise. True is every 45 deg,
            a number is a spacing from 0, and a sequence is the angles
            themselves. Rings and meridians are display annotations only.
        grid_color : color, optional
            Matplotlib color of rings, meridians, and ring labels.
        percept : :py:class:`~pulse2percept.percepts.Percept`, optional
            An eye-centered brightness percept, drawn at its visual-field
            coordinates at its own resolution over the source.
        vmax : float, optional
            The percept brightness that displays as white. Defaults to the
            maximum brightness across the whole ``percept``.
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
        points = _gaze_points(self._resolve_gaze(gaze, percept), n_out)
        # One frame is drawn, so one gaze and one frame of each layer is all
        # the work there is; the others are never evaluated.
        gaze_xy = points[0] if len(points) == 1 else points[frame]
        radii, angles, extent = self._grid_geometry(rings, meridians)
        xs, ys = self._view_axes()
        src_frame = self._source_frame(frame)
        patch = None
        if percept is None:
            # `_display_on` rejects a display range with nothing to map:
            wide = self._display_on(xs, ys, vmax=vmax, vmin=vmin,
                                    gaze=gaze_xy, frame=frame)[0][..., 0]
        else:
            pxs, pys = _percept_axes(percept)
            # `pys` descends so that row 0 of the patch is its top, as drawn:
            pys = pys[::-1]
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
        ax = self._label_fov(still.plot(ax=ax, **kwargs))
        artists = [ax.images[-1]]
        if patch is not None:
            # `imshow` must not renegotiate the limits the wide layer set:
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            artists.append(ax.imshow(patch, origin='upper',
                                     extent=_raster_extent(pxs, pys),
                                     zorder=artists[0].get_zorder() + 1))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        artists += vf.draw(ax, radii, angles, (0, 0), extent,
                           color=grid_color)
        self._clip_to_support(artists, ax.transData)
        return ax

    def _label_fov(self, ax):
        """Set limits and ticks to the FOV's outer edges

        `Percept` limits its axes to the outermost pixel centers, which would
        clip half of each edge pixel.
        """
        left, right, bottom, top = self._view_extent
        ax.set_xlim(left, right)
        ax.set_xticks(np.linspace(left, right, num=5))
        ax.set_ylim(bottom, top)
        ax.set_yticks(np.linspace(bottom, top, num=5))
        return ax

    def _grid_geometry(self, rings, meridians):
        """Ring radii (dva), meridian angles (deg), and the FOV extent"""
        extent = self._view_extent
        # The nearest FOV edge also bounds an elliptical aperture:
        r_min, r_max = vf.visible_band((0, 0), extent)
        return (vf.ring_radii(rings, r_max, r_min=r_min),
                vf.meridian_angles(meridians), extent)

    def play(self, gaze=None, rings=False, meridians=False,
             grid_color=vf.GRID_COLOR, ax=None, *, percept=None, vmax=None,
             vmin=None, fps=None, repeat=True, annotate_time=True,
             fmt='png', title=None):
        """Animate a video scene, optionally with a prosthetic percept

        Shows the frames :py:meth:`~pulse2percept.vision.Scene.render`
        returns for the same ``percept``, ``gaze``, ``vmax`` and ``vmin``,
        on that result's clock.

        Parameters
        ----------
        gaze : (x, y), (n_frames, 2), or :py:class:`~pulse2percept.vision.Gaze`, optional
            Where the eye is pointing, in dva. One pair fixates throughout;
            one pair per frame moves the eye between frames. A
            :py:class:`~pulse2percept.vision.Gaze` is resolved against the
            scene's frame times.
        rings, meridians, grid_color : optional
            Visual-field grid, as in
            :py:meth:`~pulse2percept.vision.Scene.plot`, painted into the
            displayed frames. The scene's own data is not touched.
        ax : matplotlib.axes.Axes, optional
            Axes to animate on. If None, the player makes its own.
        percept : :py:class:`~pulse2percept.percepts.Percept`, optional
            A brightness percept composed into the scene as in
            :py:meth:`~pulse2percept.vision.Scene.render`.
        vmax : float, optional
            The percept brightness that displays as white. Defaults to the
            maximum brightness across the whole ``percept``.
        vmin : float, optional
            The percept brightness that displays as black. Defaults to 0.
        fps, repeat, annotate_time, fmt, title : optional
            Player options, as in
            :py:meth:`~pulse2percept.percepts.Percept.play`.

        Returns
        -------
        ani : :py:class:`~pulse2percept.utils.HTMLAnimation`

        """
        if self.time is None:
            raise ValueError("A still scene has nothing to play. Use plot().")
        gaze = self._resolve_gaze(gaze, percept)
        radii, angles, extent = self._grid_geometry(rings, meridians)
        # The player rasterizes its own frames, so this is display output:
        display = self.render(percept=percept, gaze=gaze, vmax=vmax,
                              vmin=vmin)
        # Brightness scaling is done by `render`; the player gets RGB:
        player = dict(fps=fps, repeat=repeat, annotate_time=annotate_time,
                      ax=ax, fmt=fmt, title=title)
        if not radii.size and not angles.size:
            return self._fov_player(display.play(**player))
        # Painted into the displayed frames rather than left as an artist
        # behind the player's canvas, which would hide them. Eye-centered, so
        # one overlay holds for any gaze:
        xs, ys = self._view_axes()
        dx, dy = _raster_step(xs, ys)

        def to_pixel(x, y):
            return ((np.asarray(x) - xs[0]) / dx, (ys[0] - np.asarray(y)) / dy)

        overlay = vf.rasterize((ys.size, xs.size), radii, angles, (0, 0),
                               extent, to_pixel, color=grid_color)
        if self._aperture == _ELLIPSE:
            overlay[self._aperture_mask(xs, ys), 3] = 0
        # The rendered clock: a temporal percept may label frame ends.
        decorated = Percept(_over(display.data, overlay),
                            space=_raster_grid(xs, ys),
                            time=display.time, time_unit=display.time_unit)
        return self._fov_player(decorated.play(**player))

    def _fov_player(self, ani):
        """Show the full FOV in a player; its crop is read from the axes when
        the HTML is built"""
        self._label_fov(ani._image.axes)
        return ani
