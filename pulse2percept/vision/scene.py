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

# Blur kernel truncation in sigmas; also the off-frame margin of the loss map,
# to avoid edge effects after cropping:
_TRUNCATE = 4.0

# Approximate number of raster nodes interpolated per block, to bound the
# float64 interpolator's memory use:
_SAMPLE_BLOCK = 1 << 20

# The only `scotoma_fill` string that is not a color; see `_inpaint_rgb`:
_INPAINT = 'inpaint'

# `aperture` shapes; size comes from `fov`. See `Scene._aperture_mask`:
_RECTANGULAR = 'rectangular'
_ROUND = 'round'

# `view` of `Scene.plot`/`Scene.play`: fixed axes spanning `extent`, or
# eye-centered axes spanning `fov`:
_SCENE_VIEW = 'scene'
_EYE_VIEW = 'eye'

# Default opacity of the source outside the FOV in the scene view:
_CONTEXT_ALPHA = 0.25

# Fixed raster of a blank scene; a configurable shape would change the aspect
# ratio of the inferred `extent`:
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
    """Returns scene extent ``(left, right, bottom, top)`` in dva

    None: centered, square pixels, smallest extent containing ``fov``.
    Scalar: span of the shorter source dimension, centered, square pixels.
    """
    if extent is None:
        width, height = fov
        # Copy the limiting dimension so it stays exact:
        if width / n_cols >= height / n_rows:
            height = width * n_rows / n_cols
        else:
            width = height * n_cols / n_rows
        return (-width / 2, width / 2, -height / 2, height / 2)
    values = np.asarray(as_value(extent, dva, 'extent'), dtype=float)
    if values.ndim == 0:
        short = float(values)
        if not np.isfinite(short) or short <= 0:
            raise ValueError(f"A scalar 'extent' is the span of the shorter "
                             f"source dimension and must be a finite "
                             f"positive number of degrees, not {short}.")
        if n_cols >= n_rows:
            width, height = short * n_cols / n_rows, short
        else:
            width, height = short, short * n_rows / n_cols
        return (-width / 2, width / 2, -height / 2, height / 2)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError(f"'extent' must be a scalar or four finite numbers "
                         f"(left, right, bottom, top) in dva, not "
                         f"{np.ravel(values)}.")
    left, right, bottom, top = (float(v) for v in values)
    if right <= left or top <= bottom:
        raise ValueError(f"'extent' requires left < right and bottom < top, "
                         f"not {(left, right, bottom, top)}.")
    return (left, right, bottom, top)


def _centered(size):
    """Returns ``(left, right, bottom, top)`` of a ``(width, height)`` box
    centered on the origin"""
    width, height = size
    return (-width / 2, width / 2, -height / 2, height / 2)


def _raster_axes(extent, shape):
    """Returns pixel-center axes ``(xs, ys)`` of a raster spanning ``extent``

    ``extent`` is the *outer* ``(left, right, bottom, top)``; outermost
    centers sit half a pixel inside it. ``xs`` ascends, ``ys`` descends
    (``ys[0]`` is row 0, the top).
    """
    n_rows, n_cols = shape
    left, right, bottom, top = extent
    dx, dy = (right - left) / n_cols, (top - bottom) / n_rows
    xs = left + (np.arange(n_cols, dtype=float) + 0.5) * dx
    ys = top - (np.arange(n_rows, dtype=float) + 0.5) * dy
    return xs, ys


def _same_axis(axis, other):
    """Returns True if two regular axes match to within 1e-9 pixel"""
    if axis.shape != other.shape:
        return False
    tol = 1e-9 * (abs(float(other[1] - other[0])) if other.size > 1 else 1.0)
    return bool(np.allclose(axis, other, rtol=0, atol=tol))


def _raster_step(xs, ys):
    """Returns positive pixel pitch ``(dx, dy)`` of a raster in dva"""
    def pitch(axis):
        # A single row or column has no spacing; use 1:
        return abs(float(axis[1] - axis[0])) if axis.size > 1 else 1.0
    return pitch(xs), pitch(ys)


def _raster_extent(xs, ys):
    """Returns outer edges ``(left, right, bottom, top)`` for `imshow`"""
    dx, dy = _raster_step(xs, ys)
    return (float(xs[0]) - dx / 2, float(xs[-1]) + dx / 2,
            float(ys[-1]) - dy / 2, float(ys[0]) + dy / 2)


def _to_pixel(xs, ys):
    """Returns a function mapping dva to continuous pixel coordinates, with
    (0, 0) at the top-left pixel center"""
    dx, dy = _raster_step(xs, ys)

    def to_pixel(x, y):
        return ((np.asarray(x) - xs[0]) / dx, (ys[0] - np.asarray(y)) / dy)
    return to_pixel


def _regrid(xs, ys, rgb, to_xs, to_ys):
    """Linearly resamples a ``(rows, cols, 3)`` raster onto new nodes

    ``ys`` descend. Nodes past the outermost centers take the edge value, so
    the outer half pixel is not darkened.
    """
    sample = RegularGridInterpolator((ys[::-1], xs), rgb[::-1],
                                     method='linear')
    x, y = np.meshgrid(np.clip(to_xs, xs.min(), xs.max()),
                       np.clip(to_ys, ys.min(), ys.max()))
    points = np.column_stack((y.ravel(), x.ravel()))
    return sample(points).reshape((to_ys.size, to_xs.size, 3)).astype(
        np.float32)


def _imshow_within(ax, image, extent, zorder):
    """Draws an RGB layer at ``extent`` without changing the axis limits"""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    artist = ax.imshow(image, origin='upper', extent=extent, zorder=zorder)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return artist


def _label_limits(ax, extent):
    """Sets axis limits to ``extent`` with five ticks per axis"""
    left, right, bottom, top = extent
    ax.set_xlim(left, right)
    ax.set_xticks(np.linspace(left, right, num=5))
    ax.set_ylim(bottom, top)
    ax.set_yticks(np.linspace(bottom, top, num=5))
    return ax


def _raster_grid(xs, ys):
    """Returns a Grid2D on the raster nodes (``ys`` descending)"""
    return Grid2D((float(xs[0]), float(xs[-1])),
                  (float(ys[-1]), float(ys[0])), step=_raster_step(xs, ys))


def _pad_axis(axis, step, pad):
    """Extends a regular axis by ``pad`` samples of signed ``step`` per end"""
    if pad == 0:
        return axis
    lead = axis[0] + step * np.arange(-pad, 0)
    tail = axis[-1] + step * np.arange(1, pad + 1)
    return np.concatenate((lead, axis, tail))


def _pixel_count(extent, step):
    """Returns the number of pixels of at most ``step`` dva spanning
    ``extent``"""
    n = extent / step
    # Tolerance keeps a ratio that is integral up to rounding from adding a
    # pixel:
    return max(int(np.ceil(n - 1e-9 * max(n, 1.0))), 1)


def _clip_to_frame(points, shape):
    """Returns pixel coordinates clipped onto the frame, and an inside mask"""
    points = np.asarray(points, dtype=float)
    edges = np.asarray(shape[:2], dtype=float) - 0.5
    inside = np.all((points >= -0.5) & (points <= edges), axis=1)
    # Off-frame points are not used, so any on-grid placeholder works:
    on_grid = np.where(inside[:, np.newaxis], points, 0.0)
    return np.clip(on_grid, 0.0, edges - 0.5), inside


def _interpolate(grid, frames, points):
    """Samples ``frames`` at ``points``, keeping trailing axes"""
    interpolator = RegularGridInterpolator(grid, frames, method='linear',
                                           bounds_error=False, fill_value=0)
    return interpolator(points)


def _drop_gray_axis(values):
    """Drops the channel axis of grayscale samples: (n_points, n_frames)"""
    return values[:, 0] if values.shape[1] == 1 else values


def _as_rgb(frame):
    """Returns a ``(rows, cols, channels)`` frame as RGB (gray is
    broadcast, not copied)"""
    if frame.shape[2] == 3:
        return frame
    return np.broadcast_to(frame, frame.shape[:2] + (3,))


def _take_frame(frames, frame):
    """Returns one frame as a 1-frame stack; ``None`` returns all frames"""
    return frames if frame is None else frames[..., frame:frame + 1]


def _take_time(time, frame):
    """Returns the matching slice of ``time`` (may be None)"""
    if frame is None or time is None:
        return time
    return np.asarray(time)[frame:frame + 1]


def _percept_axes(prosthetic):
    """Returns the percept's eye-centered ``(xs, ys)`` in dva, ascending"""
    ys = np.asarray(prosthetic.ydva, dtype=float)
    xs = np.asarray(prosthetic.xdva, dtype=float)
    if ys.size < 2 or xs.size < 2:
        raise ValueError(f"A percept needs extent in both directions to be "
                         f"placed in a scene, but this one's grid is "
                         f"{ys.size} x {xs.size}.")
    return xs, ys


def _percept_on(prosthetic, frames, xs, ys):
    """Returns percept brightness on an eye-centered raster,
    ``(rows, cols, n_frames)``"""
    pxs, pys = _percept_axes(prosthetic)
    # Row 0 of Grid2D data is the largest y, but `ydva` ascends; flip rows to
    # match (the interpolator requires ascending axes):
    sample = RegularGridInterpolator((pys, pxs), frames[::-1],
                                     method='linear', bounds_error=False,
                                     fill_value=0)
    x, y = np.meshgrid(xs, ys)
    points = np.column_stack((y.ravel(), x.ravel()))
    return sample(points).reshape((ys.size, xs.size, -1))


def _resolve_fill(scotoma_fill):
    """Normalizes scotoma_fill to a gray level, RGB tuple, or _INPAINT"""
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
    """Normalizes ``background`` to an ``(r, g, b)`` triple in [0, 1]"""
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


def _resolve_view(view):
    """Normalizes ``view`` to ``_SCENE_VIEW`` or ``_EYE_VIEW``"""
    for name in (_SCENE_VIEW, _EYE_VIEW):
        if view == name:
            return name
    raise ValueError(f"'view' is either {_SCENE_VIEW!r} or {_EYE_VIEW!r}, "
                     f"not {view!r}.")


def _resolve_context_alpha(context_alpha):
    """Normalizes ``context_alpha`` to a float in [0, 1]"""
    alpha = float(as_value(context_alpha, dimensionless, 'context_alpha'))
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError(f"'context_alpha' is the opacity of the scene "
                         f"outside the FOV and must lie in [0, 1], not "
                         f"{context_alpha}.")
    return alpha


def _resolve_aperture(aperture):
    """Normalizes ``aperture`` to ``_RECTANGULAR`` or ``_ROUND``"""
    for shape in (_RECTANGULAR, _ROUND):
        if aperture == shape:
            return shape
    raise ValueError(f"'aperture' is either {_RECTANGULAR!r} or {_ROUND!r}, "
                     f"not {aperture!r}.")


def _check_prosthetic(prosthetic):
    """Checks that ``prosthetic`` is a brightness Percept with a grid"""
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
    """Alpha-composites an RGBA ``overlay`` onto every RGB frame"""
    alpha = overlay[..., 3][..., np.newaxis, np.newaxis]
    color = overlay[..., :3][..., np.newaxis]
    return np.clip(frames * (1 - alpha) + color * alpha, 0, 1)


def _inpaint_rgb(image, mask):
    """Inpaints ``image`` where ``mask`` is True from the unmasked pixels"""
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
    """Returns display limits ``(vmin, vmax)`` for ``data``

    Defaults are 0 and the maximum over all frames, so frames share one scale.
    """
    auto = vmax is None
    vmin, vmax = _resolve_clim(data, vmin, vmax, auto_vmin=0)
    if auto and vmax == vmin:
        # Constant percept at vmin: any positive span maps it to black, as
        # Matplotlib does for vmin == vmax:
        return vmin, vmin + 1.0
    if vmax <= vmin:
        raise ValueError(f"'vmax' ({vmax}) must be greater than 'vmin' "
                         f"({vmin}); the percept is in arbitrary brightness "
                         f"units, and this is what says which of them is "
                         f"white.")
    return vmin, vmax


class Scene(PrettyPrint):
    """Visual scene with optional loss of native vision

    A scene places a source image at a fixed angular ``extent`` and views it
    through a field of view (``fov``) centered on the fovea, optionally with
    a scotoma.

    Geometry conventions:

    *  ``extent`` is the *outer* ``(left, right, bottom, top)`` of the source
       in scene coordinates, half a pixel past the outermost pixel centers.
    *  Pixel coordinates address pixel *centers*.
    *  Row 0 is the top of the frame (largest ``y``).

    Coordinate frames:

    *  **Scene coordinates** are fixed world coordinates; ``extent`` places
       the source in them.
    *  **Eye coordinates** are centered on the fovea. The FOV, aperture,
       scotoma, prosthetic percept, visual-field grid, and implant use them.

    Gaze moves the FOV through the scene; the source does not move or
    rescale::

        (x_scene, y_scene) = (x_eye, y_eye) + (x_gaze, y_gaze)

    :py:meth:`~pulse2percept.vision.Scene.plot` and
    :py:meth:`~pulse2percept.vision.Scene.play` default to
    ``view='scene'``: fixed axes spanning ``extent``, the FOV moving with
    gaze, and the source outside it dimmed. ``view='eye'`` shows the FOV in
    eye coordinates, spanning ``[-fov / 2, fov / 2]``, as
    :py:meth:`~pulse2percept.vision.Scene.render` always does; parts of the
    FOV beyond ``extent`` are black. Gaze also sets the device input, unless
    the implant's
    :py:attr:`~pulse2percept.implants.Implant.scene_input_frame` is
    ``'head'`` (head-fixed camera). Device input is sampled from the whole
    ``extent``, not only the FOV.

    Source, ``extent`` and ``fov`` are fixed after construction.

    The scotoma affects native vision only. Device input is sampled from the
    source inside and outside the scotoma.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    source : ImageStimulus, VideoStimulus, or image
        The scene content. Anything that is not an
        :py:class:`~pulse2percept.stimuli.ImageStimulus` or
        :py:class:`~pulse2percept.stimuli.VideoStimulus` (e.g., a file name
        or NumPy array) is passed to ``ImageStimulus``.
    fov : float or ``(width, height)``
        Size of the viewing window, in degrees of visual angle, centered on
        the fovea (e.g. ``40 * dva``). A scalar is a square window.
    extent : float or ``(left, right, bottom, top)``, optional
        Where the source sits in scene coordinates, in dva. A scalar is the
        span of the shorter source dimension, centered with square pixels
        (e.g., ``45 * dva`` on a 173 x 320 source is 83.2 x 45 dva). If None,
        the source is centered with square pixels and scaled to the smallest
        extent that contains ``fov``: with a scalar ``fov``, the shorter
        source dimension spans ``fov``. ``extent`` need not contain ``fov``.
    scotoma : :py:class:`~pulse2percept.vision.Scotoma`, optional
        Region where native vision is lost. If None, native vision is intact
        everywhere.
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
        Standard deviation (dva) of a Gaussian blur applied to the rasterized
        loss map before drawing, softening the scotoma boundary. Defaults to
        0.5 dva; 0 keeps the scotoma's own edge. Converted to pixels per
        raster, so the softness in dva does not depend on resolution.
        Rendering only: the scotoma's geometry is unchanged.

        .. versionchanged:: 0.11.0
            Measured in degrees of visual angle rather than in scene pixels.
    aperture : {'rectangular', 'round'}, optional
        Shape of the FOV (``fov`` sets its size). ``'rectangular'`` (default)
        shows the whole window; ``'round'`` inscribes an eye-centered ellipse
        with semi-axes ``fov / 2`` (a disc for a square ``fov``).
        Display only: :py:meth:`~pulse2percept.vision.Scene.plot` clips the
        full-intensity FOV, percept, and grid to it (scene-view context stays
        visible outside), and :py:meth:`~pulse2percept.vision.Scene.render`
        writes black outside it. Scene sampling, device input, stimulation,
        and the model response are unaffected.

    Examples
    --------
    A 40-dva window onto a logo with a central 16-dva scotoma. The logo is
    landscape, so its height spans the window:

    >>> from pulse2percept.stimuli import samples
    >>> from pulse2percept.units import dva
    >>> from pulse2percept.vision import Scene, Scotoma
    >>> scene = Scene(samples.logo_bvl(), fov=40 * dva,
    ...               scotoma=Scotoma.circle(8 * dva))
    >>> scene.fov, scene.extent
    ((40.0, 40.0), (-25.0, 25.0, -20.0, 20.0))

    A 40-dva disc onto a wider extent. ``render`` covers the FOV at the
    source's pixel pitch:

    >>> world = Scene(samples.logo_bvl(), extent=(-50, 50, -40, 40) * dva,
    ...               fov=40 * dva, aperture='round')
    >>> world.render().shape
    (288, 288, 3, 1)

    :py:meth:`~pulse2percept.vision.Scene.blank` gives a black field (no
    scotoma):

    >>> blank = Scene.blank()
    >>> blank.plot()                                    # doctest: +SKIP

    """

    def __init__(self, source, fov, extent=None, scotoma=None, scotoma_fill=0,
                 scotoma_blend=0.5, background=0, aperture=_RECTANGULAR):
        if not isinstance(source, (ImageStimulus, VideoStimulus)):
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

        Black is scene content, not vision loss: device input is black. Use a
        :py:class:`~pulse2percept.vision.Scotoma` for lost vision.

        The source is a fixed 512 x 512 raster, the default resolution of
        :py:meth:`~pulse2percept.vision.Scene.render`; pass ``step`` or
        ``shape`` to ``render`` for another. Neither sets the grid a
        prosthetic model predicts on.

        .. versionadded:: 0.11.0

        Parameters
        ----------
        fov : float or ``(width, height)``, optional
            Viewing window, in dva. Defaults to a 45 x 45 dva disc.
        **kwargs :
            Any other :py:class:`~pulse2percept.vision.Scene` argument.
            ``aperture`` defaults to ``'round'`` rather than
            ``'rectangular'``.

        Examples
        --------
        >>> from pulse2percept.units import dva
        >>> from pulse2percept.vision import Scene
        >>> blank = Scene.blank(fov=60 * dva)
        >>> blank.fov
        (60.0, 60.0)

        """
        kwargs.setdefault('aperture', _ROUND)
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
        # Omit the default:
        if self.aperture != _RECTANGULAR:
            params['aperture'] = self.aperture
        return params

    @property
    def source(self):
        """Scene content, as an ImageStimulus or VideoStimulus"""
        return self._source

    @property
    def scotoma(self):
        """Region of lost native vision, or None if intact"""
        return self._scotoma

    @property
    def background(self):
        """Color behind transparent source pixels, as ``(r, g, b)``"""
        return tuple(self._background)

    @property
    def scotoma_fill(self):
        """Scotoma fill as a gray level, `(r, g, b)` triple, or `'inpaint'`

        Matplotlib color strings are stored as RGB triples.
        """
        return self._scotoma_fill


    @property
    def scotoma_blend(self):
        """Gaussian sigma (dva) of the drawn scotoma boundary"""
        return self._scotoma_blend

    @property
    def aperture(self):
        """FOV shape: ``'rectangular'`` or ``'round'``"""
        return self._aperture

    @property
    def _frame_shape(self):
        """(rows, cols) of one source frame"""
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
        """FOV as ``(left, right, bottom, top)``, in eye-centered dva"""
        return _centered(self._fov)

    @property
    def shape(self):
        """``(rows, cols)`` of one frame"""
        return self._frame_shape

    @property
    def time(self):
        """Frame times of the source, or None for a still scene"""
        return self.source.time

    @property
    def time_unit(self):
        """Unit of ``time``"""
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
        A scotoma 6 dva into one eye's right hemifield is 6 dva into the
        other eye's left hemifield:

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
        """Returns the source as ``(rows, cols, channels, n_frames)``"""
        if self._cached_frames is not None:
            return self._cached_frames
        source = self.source
        if isinstance(source, ImageStimulus):
            frames = source.data.reshape(source.img_shape)[..., np.newaxis]
        else:
            frames = source.data.reshape(source.vid_shape)
        if frames.ndim == 3:
            # Grayscale: add a channel axis:
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
        """Number of source frames; 1 for a still scene"""
        return self._frames().shape[-1]

    def _sample_at(self, x, y, gaze=None):
        """Samples the scene at eye-centered positions (dva)"""
        return self._sample_frames(self._frames(), x, y, gaze=gaze)

    def _sample_frames(self, frames, x, y, gaze=None):
        """`_sample_at` on a given stack of source frames"""
        gaze = _gaze_points(gaze, frames.shape[-1])
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        # Interpolate in pixel coordinates; `dva_to_pixel` does the mapping:
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
        """Returns grayscale device input, (n_points, n_frames)"""
        values = self._sample_at(x, y, gaze=gaze)
        if values.ndim == 2:
            return values
        # `rgb2gray` requires channels last:
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
        """Returns scene coordinates of pixel centers as ``(x, y)`` meshes"""
        if self._pixel_centers_cache is None:
            centers = np.meshgrid(*self._axes)
            for mesh in centers:
                mesh.flags.writeable = False
            self._pixel_centers_cache = centers
        return self._pixel_centers_cache

    def _source_frame(self, frame):
        """Returns the source frame for an output frame; None means all"""
        if frame is None:
            return None
        # A still source has one frame for all output frames:
        return frame if self.n_frames > 1 else 0

    def _source_on(self, xs, ys, gaze_xy=(0.0, 0.0), frame=None):
        """Samples the source on an eye-centered raster at one gaze

        Returns ``(rows, cols, channels, n_frames)`` with 1 or 3 channels.
        ``frame`` restricts the result to that source frame.
        """
        frames = _take_frame(self._frames(), frame)
        gx, gy = gaze_xy
        src_xs, src_ys = self._axes
        if _same_axis(xs + gx, src_xs) and _same_axis(ys + gy, src_ys):
            # Raster matches the source, so skip resampling (exact values):
            return frames
        n_rows, n_cols = ys.size, xs.size
        # Interpolate in row blocks to bound float64 memory use:
        block = max(1, _SAMPLE_BLOCK // max(n_cols, 1))
        out = None
        for lo in range(0, n_rows, block):
            hi = min(lo + block, n_rows)
            x, y = np.meshgrid(xs, ys[lo:hi])
            values = self._sample_frames(frames, x, y, gaze=(gx, gy))
            if values.ndim == 2:
                # Grayscale: add a channel axis:
                values = values[:, np.newaxis, :]
            if out is None:
                out = np.empty((n_rows, n_cols) + values.shape[1:],
                               dtype=np.float32)
            out[lo:hi] = values.reshape((hi - lo, n_cols) + values.shape[1:])
        return out

    def _loss_on(self, xs, ys):
        """Returns float32 scotoma loss in [0, 1] on an eye-centered raster"""
        if self.scotoma is None:
            return np.zeros((ys.size, xs.size), dtype=np.float32)
        x, y = np.meshgrid(xs, ys)
        return np.asarray(self.scotoma(x, y), dtype=np.float32)

    def _rendered_loss_on(self, xs, ys):
        """Returns `_loss_on` blurred by `scotoma_blend`

        Sigma (dva) is converted to pixels per axis, so anisotropic pixels get
        separate row and column sigmas.
        """
        sigma = self._scotoma_blend
        # Inpainting uses the hard boundary:
        hard = self.scotoma is None or self._scotoma_fill == _INPAINT
        if hard or sigma == 0:
            return self._loss_on(xs, ys)
        dx, dy = _raster_step(xs, ys)
        sigmas = (sigma / dy, sigma / dx)
        pads = tuple(int(np.ceil(_TRUNCATE * s)) + 1 for s in sigmas)
        # Blur a padded loss map, so loss outside the raster is included:
        loss = self._loss_on(_pad_axis(xs, dx, pads[1]),
                             _pad_axis(ys, -dy, pads[0]))
        blurred = gaussian_filter(loss, sigmas, mode='nearest',
                                  truncate=_TRUNCATE)
        return np.clip(blurred[pads[0]:-pads[0], pads[1]:-pads[1]], 0, 1)

    def _fill_rgb(self, frame_rgb, loss):
        """Returns the scotoma fill for one ``(rows, cols, 3)`` frame"""
        if self._scotoma_fill != _INPAINT:
            return self._scotoma_fill
        return _inpaint_rgb(frame_rgb, loss > 0)

    def _aperture_mask(self, xs, ys):
        """Returns a mask of raster nodes outside the round aperture

        The ellipse is inscribed in the FOV, with semi-axes `fov / 2`.
        """
        a, b = self._fov[0] / 2, self._fov[1] / 2
        return (xs / a) ** 2 + ((ys / b) ** 2)[:, np.newaxis] > 1

    def _apply_aperture(self, frames, xs, ys):
        """Sets ``(rows, cols, 3, n_frames)`` to black outside the aperture"""
        if self._aperture == _RECTANGULAR:
            return frames
        out = np.array(frames, dtype=np.float32)
        out[self._aperture_mask(xs, ys)] = 0
        return out

    def _outside_support(self, xs, ys):
        """Returns a mask of raster nodes outside the FOV (either aperture)"""
        if self._aperture == _ROUND:
            return self._aperture_mask(xs, ys)
        a, b = self._fov[0] / 2, self._fov[1] / 2
        return (np.abs(xs) > a) | (np.abs(ys) > b)[:, np.newaxis]

    def _support_patch(self, transform, center=(0.0, 0.0)):
        """Returns the FOV shape as a patch centered on ``center``, for
        clipping"""
        width, height = self._fov
        cx, cy = center
        if self._aperture == _ROUND:
            return Ellipse((cx, cy), width, height, transform=transform)
        return Rectangle((cx - width / 2, cy - height / 2), width, height,
                         transform=transform)

    def _clip_to_support(self, artists, transform):
        """Clips drawn artists to the FOV

        With a rectangular aperture, the first artist (the FOV layer) already
        matches the FOV and is skipped.
        """
        if self._aperture == _RECTANGULAR:
            artists = artists[1:]
        if not artists:
            return
        clip = self._support_patch(transform)
        for artist in artists:
            artist.set_clip_path(clip)

    def _native_on(self, xs, ys, gaze=None, frame=None):
        """Returns residual native vision on an eye-centered raster,
        ``(rows, cols, 3, n_frames)``

        With ``frame``, only that source frame is used and ``gaze`` is a
        single (x, y).
        """
        n_frames = self.n_frames if frame is None else 1
        points = _gaze_points(gaze, n_frames)
        if len(points) == 1:
            frames = self._source_on(xs, ys, points[0], frame=frame)
        else:
            # One gaze per frame:
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
            # Inpainting depends on the frame content:
            fill = self._fill_rgb(rgb, loss)
            out[..., f] = (1 - alpha) * rgb + alpha * fill
        return out

    def _native_rgb(self, gaze=None):
        """Returns residual native vision on the default FOV raster, without
        the aperture"""
        return self._native_on(*self._view_axes(), gaze=gaze)

    def _composed_on(self, xs, ys, prosthetic, vmax, vmin=None, gaze=None,
                     frame=None):
        """Composes a prosthetic percept into the scotoma on an eye-centered
        raster

        ``out = (1 - loss) * native + loss * max(fill, phosphene)``. Returns
        ``(frames, time, time_unit)``. With ``frame``, only that output frame
        is computed and ``gaze`` is a single (x, y).
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
            # Not `_native_on`: gray is broadcast to RGB per frame below
            # instead of copied:
            source = self._source_on(xs, ys, points[0],
                                     frame=self._source_frame(frame))
            n_scene = source.shape[-1]
        n_rows, n_cols = ys.size, xs.size
        # Both eye-centered, so independent of gaze:
        brightness = _percept_on(prosthetic, pframes, xs, ys)
        loss = self._rendered_loss_on(xs, ys)
        alpha = loss[..., np.newaxis]
        # Frame-major, so each frame write is contiguous:
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
        """Returns a prosthetic percept on black, on an eye-centered raster

        Used without a scotoma: overlaying the percept on intact native vision
        would imply an interaction that is not modeled. With ``frame``, only
        that output frame is computed and ``gaze`` is a single (x, y).
        """
        _check_prosthetic(prosthetic)
        vmin, vmax = _check_range(prosthetic.data, vmin, vmax)
        pframes, out_time, out_unit = self._prosthetic_frames(prosthetic,
                                                              frame=frame)
        # Validate only; percept and raster are both eye-centered:
        _gaze_points(gaze, pframes.shape[-1])
        brightness = _percept_on(prosthetic, pframes, xs, ys)
        scaled = np.clip((brightness - vmin) / (vmax - vmin), 0, 1)
        rgb = np.repeat(scaled[:, :, np.newaxis, :], 3, axis=2)
        return np.asarray(rgb, dtype=np.float32), out_time, out_unit

    def _display_on(self, xs, ys, percept=None, vmax=None, vmin=None, gaze=None,
                    frame=None):
        """Returns display RGB on an eye-centered raster

        Residual native vision, with the percept composed in if given.
        The aperture is not applied. Returns ``(frames, time, time_unit)``.
        With ``frame``, only that output frame is computed and ``gaze`` is a
        single (x, y).
        """
        if percept is None:
            if vmax is not None or vmin is not None:
                raise ValueError("'vmin' and 'vmax' map percept brightness "
                                 "onto a display, and there is no percept "
                                 "here. Pass 'percept'.")
            # Without a percept, output frames are the source frames:
            out_time, out_unit, _ = self._output_clock()
            return (self._native_on(xs, ys, gaze=gaze, frame=frame),
                    _take_time(out_time, frame), out_unit)
        if self.scotoma is None:
            return self._prosthetic_on(xs, ys, percept, vmax, vmin=vmin,
                                       gaze=gaze, frame=frame)
        return self._composed_on(xs, ys, percept, vmax, vmin=vmin, gaze=gaze,
                                 frame=frame)

    def _output_clock(self, percept=None):
        """Returns ``(time, time_unit, n_frames)`` of the output frames

        A video scene uses its own frame times (the percept is resampled at
        them); a still scene uses the percept's, which may be ``None``.
        Gaze is resolved at these times. Time stamps of a returned Percept
        can differ; see `_prosthetic_frames`.
        """
        if self.time is not None:
            return self.time, self.time_unit, self.n_frames
        if percept is None:
            # No frame times, but keep the source's time unit:
            return None, self.time_unit, self.n_frames
        return percept.time, percept.time_unit, percept.data.shape[-1]

    def _n_display_frames(self, percept):
        """Returns the number of output frames"""
        return self._output_clock(percept)[2]

    def _resolve_gaze(self, gaze, percept=None):
        """Converts a `Gaze` to one (x, y) per output frame; returns other
        inputs unchanged"""
        if not isinstance(gaze, Gaze):
            return gaze
        time, unit, n_out = self._output_clock(percept)
        return _gaze_points(gaze, n_out, time=time, time_unit=unit)

    def _source_aligned(self, prosthetic):
        """Returns True if percept frame k was predicted from scene frame k

        Uses only the top-level ``metadata['source_frame_time']`` (ms) set by
        temporal models; equal frame counts are not sufficient.
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
        """Aligns a percept with the output frames

        Returns ``(frames, time, time_unit)``. ``frame`` restricts the result
        to one output frame; timing is still checked against the whole video.
        """
        out_time, out_unit, n_out = self._output_clock(prosthetic)
        out_time = _take_time(out_time, frame)
        if self.time is None:
            # Still scene: the percept's frames are the output frames:
            return _take_frame(prosthetic.data, frame), out_time, out_unit
        n_pros = prosthetic.data.shape[-1]
        if n_pros == 1 and prosthetic.time is None:
            # Untimed single-frame percept is repeated for every frame:
            return (np.repeat(prosthetic.data, 1 if frame is not None
                              else n_out, axis=-1), out_time, out_unit)
        if self._source_aligned(prosthetic):
            # Frame for frame, but keep the percept's time stamps (a temporal
            # model may report each frame at its end, not its onset):
            return (_take_frame(prosthetic.data, frame),
                    _take_time(prosthetic.time, frame), prosthetic.time_unit)
        unit = prosthetic.time_unit
        # Check against the whole video, not just the drawn frame:
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
            # A scalar time index drops the frame axis; restore it:
            frames = prosthetic[..., Quantity(float(asked_time[0]),
                                              self.time_unit)]
            frames = frames[..., np.newaxis]
        return frames, out_time, out_unit

    def _grid(self):
        """Returns a Grid2D on the default FOV raster (eye-centered)"""
        return _raster_grid(*self._view_axes())

    def _render_shape(self, step, shape):
        """Returns ``(rows, cols)`` of the render raster"""
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
            # Default to the source's pixel pitch:
            step = self._angular_pixel
        step = np.asarray(as_value(step, dva, 'step'), dtype=float)
        if step.ndim == 0:
            step = np.repeat(step, 2)
        bad = step.shape != (2,) or not np.all(np.isfinite(step))
        if bad or step.min() <= 0:
            raise ValueError(f"'step' is an angular sampling in degrees and "
                             f"must be a positive number or a (dx, dy) pair, "
                             f"not {np.ravel(step)}.")
        # Round up, so pixels are never coarser than ``step``:
        return (_pixel_count(self._fov[1], step[1]),
                _pixel_count(self._fov[0], step[0]))

    def render(self, percept=None, gaze=None, vmax=None, vmin=None, step=None,
               shape=None):
        """Rasterize the FOV into one RGB percept

        Residual native vision, with a prosthetic ``percept`` composed into
        the scotoma, on one common raster (e.g., for saving or image
        processing). :py:meth:`~pulse2percept.vision.Scene.plot` draws the
        same content with each layer at its own resolution.

        The raster spans the FOV in eye-centered coordinates,
        ``[-fov / 2, fov / 2]``, regardless of gaze; gaze selects the part of
        the source shown, and parts beyond ``extent`` are black. Pixel pitch
        defaults to the source's. The source is resampled unless the raster
        lands on its pixel centers (e.g., ``fov`` matching ``extent`` at zero
        gaze). ``step`` or ``shape`` sets another raster; a fine ``step``
        over a wide field is expensive.

        .. versionadded:: 0.11.0

        Parameters
        ----------
        percept : :py:class:`~pulse2percept.percepts.Percept`, optional
            Eye-centered brightness percept, drawn at its visual-field
            coordinates within the FOV. With a scotoma, it is composed as
            ``(1 - loss) * native + loss * max(scotoma_fill, phosphene)``;
            without one, it is rendered alone on black (overlaying it on
            intact native vision would imply an unmodeled interaction).
            Not supported with ``scotoma_fill='inpaint'``.
        gaze : (x, y), (n_frames, 2), or :py:class:`~pulse2percept.vision.Gaze`, optional
            Scene location (dva) on the fovea. Defaults to the origin. A
            :py:class:`~pulse2percept.vision.Gaze` is resolved at the output
            frame times: a video scene's, or a timed percept's for a still
            scene.
        vmax : float, optional
            Percept brightness shown as white. Defaults to the maximum over
            the whole ``percept``.
        vmin : float, optional
            Percept brightness shown as black. Defaults to 0.
        step : float or (dx, dy), optional
            Pixel size of the render raster, in dva. Rounded up, so pixels are
            never coarser than this. Mutually exclusive with ``shape``.
        shape : (rows, cols), optional
            Render raster shape. Mutually exclusive with ``step``.

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
             view=_SCENE_VIEW, context_alpha=_CONTEXT_ALPHA, **kwargs):
        """Plot residual native vision

        The scene where vision is intact and ``scotoma_fill`` where it is
        lost, inside the FOV at the given ``gaze``.

        With ``view='scene'`` (default), the axes span ``extent`` in scene
        coordinates. The source stays fixed; the FOV, aperture, scotoma,
        percept, and visual-field grid are centered on ``gaze``. The source
        outside the FOV is dimmed to ``context_alpha * source``.

        With ``view='eye'``, the axes span ``[-fov / 2, fov / 2]`` in
        eye-centered coordinates, as in
        :py:meth:`~pulse2percept.vision.Scene.render`, and gaze moves the
        source through the fixed window. Parts of the FOV beyond ``extent``
        are black.

        A ``percept`` is drawn in the same field, so its size and position
        can be compared with the FOV. Each layer keeps its own resolution:
        the source at its pixel pitch, the percept as a patch on its own
        grid. For a single common raster, use
        :py:meth:`~pulse2percept.vision.Scene.render`.

        Inside the patch, layers compose as
        ``(1 - loss) * native + loss * max(scotoma_fill, phosphene)``, so
        where the percept is dark the patch matches residual vision and its
        border is invisible. Without a scotoma, the percept is drawn alone on
        black inside the FOV (overlaying it on intact native vision would
        imply an unmodeled interaction).

        Parameters
        ----------
        gaze : (x, y) or :py:class:`~pulse2percept.vision.Gaze`, optional
            Scene location (dva) on the fovea. Defaults to the origin. A
            :py:class:`~pulse2percept.vision.Gaze` is resolved at the output
            frame times, and the position at ``frame`` is drawn.
        frame : int, optional
            Frame of a video scene to draw. Ignored for a still scene.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If None, uses the current axes.
        rings : bool, float, or sequence, optional
            Eccentricity rings (dva) centered on the fovea. True draws 1.25,
            2.5, 5, 10, 20, ... dva; a number sets the spacing; a sequence
            gives the eccentricities. Automatic rings stop at the nearest FOV
            edge.
        meridians : bool, float, or sequence, optional
            Polar-angle meridians (geometric deg) from the fovea to the FOV
            edge: 0 is +x, 90 is +y, counterclockwise. True is every 45 deg;
            a number sets the spacing from 0; a sequence gives the angles.
            Rings and meridians are display annotations only.
        grid_color : color, optional
            Matplotlib color of rings, meridians, and ring labels.
        percept : :py:class:`~pulse2percept.percepts.Percept`, optional
            Eye-centered brightness percept, drawn at its visual-field
            coordinates and own resolution over the source.
        vmax : float, optional
            Percept brightness shown as white. Defaults to the maximum over
            the whole ``percept``.
        vmin : float, optional
            Percept brightness shown as black. Defaults to 0.
        view : {'scene', 'eye'}, optional
            Scene-centered axes spanning ``extent`` (default), or eye-centered
            axes spanning the FOV.
        context_alpha : float, optional
            Opacity in [0, 1] of the source outside the FOV (over black) in
            the scene view: 0 is black, 1 is undimmed. Display only; ignored
            for ``view='eye'``.
        **kwargs :
            Passed on to :py:meth:`~pulse2percept.percepts.Percept.plot`.

        Returns
        -------
        ax : matplotlib.axes.Axes

        """
        view = _resolve_view(view)
        if percept is not None:
            _check_prosthetic(percept)
        n_out = self._n_display_frames(percept)
        if not 0 <= frame < n_out:
            raise ValueError(f"'frame' must be in 0..{n_out - 1}, not "
                             f"{frame}.")
        points = _gaze_points(self._resolve_gaze(gaze, percept), n_out)
        # Only the drawn frame is evaluated:
        gaze_xy = points[0] if len(points) == 1 else points[frame]
        if view == _EYE_VIEW:
            return self._plot_eye(gaze_xy, frame, ax, rings, meridians,
                                  grid_color, percept, vmax, vmin, **kwargs)
        return self._plot_scene(gaze_xy, frame, ax, rings, meridians,
                                grid_color, percept, vmax, vmin,
                                _resolve_context_alpha(context_alpha),
                                **kwargs)

    def _fov_layer(self, xs, ys, gaze_xy, frame, percept, vmax, vmin):
        """Returns the wide `plot` layer on an eye-centered raster: native
        vision, residual vision behind a percept, or black"""
        if percept is None:
            # `_display_on` checks that vmin/vmax are None:
            return self._display_on(xs, ys, vmax=vmax, vmin=vmin,
                                    gaze=gaze_xy, frame=frame)[0][..., 0]
        if self.scotoma is None:
            # Percept without a scotoma is drawn on black:
            return np.zeros((ys.size, xs.size, 3), dtype=np.float32)
        return self._native_on(xs, ys, gaze=gaze_xy,
                               frame=self._source_frame(frame))[..., 0]

    def _percept_patch(self, percept, vmax, vmin, gaze_xy, frame,
                       offset=(0.0, 0.0)):
        """Returns the percept composed on its own grid, and its `imshow`
        extent shifted by ``offset`` (dva)"""
        pxs, pys = _percept_axes(percept)
        # Descending, so row 0 of the patch is the top:
        pys = pys[::-1]
        patch = self._display_on(pxs, pys, percept=percept, vmax=vmax,
                                 vmin=vmin, gaze=gaze_xy,
                                 frame=frame)[0][..., 0]
        left, right, bottom, top = _raster_extent(pxs, pys)
        dx, dy = offset
        return patch, (left + dx, right + dx, bottom + dy, top + dy)

    def _plot_eye(self, gaze_xy, frame, ax, rings, meridians, grid_color,
                  percept, vmax, vmin, **kwargs):
        """`plot` on eye-centered axes spanning the FOV"""
        radii, angles, extent = self._grid_geometry(rings, meridians)
        xs, ys = self._view_axes()
        patch = None
        if percept is not None:
            patch = self._percept_patch(percept, vmax, vmin, gaze_xy, frame)
        wide = self._fov_layer(xs, ys, gaze_xy, frame, percept, vmax, vmin)
        still = Percept(wide[..., np.newaxis], space=self._grid())
        ax = self._label_fov(still.plot(ax=ax, **kwargs))
        artists = [ax.images[-1]]
        if patch is not None:
            artists.append(_imshow_within(ax, *patch,
                                          artists[0].get_zorder() + 1))
        artists += vf.draw(ax, radii, angles, (0, 0), extent,
                           color=grid_color)
        self._clip_to_support(artists, ax.transData)
        return ax

    def _inpaints(self):
        """Returns True if the scotoma is inpainted

        Inpainting depends on the raster, so it must use the `render` raster.
        """
        return self.scotoma is not None and self._scotoma_fill == _INPAINT

    def _plot_scene(self, gaze_xy, frame, ax, rings, meridians, grid_color,
                    percept, vmax, vmin, context_alpha, **kwargs):
        """`plot` on scene-centered axes spanning ``extent``"""
        radii, angles, extent = self._grid_geometry(rings, meridians,
                                                    center=gaze_xy)
        xs, ys = self._axes
        gx, gy = gaze_xy
        patch = None
        if percept is not None:
            patch = self._percept_patch(percept, vmax, vmin, gaze_xy, frame,
                                        offset=gaze_xy)
        if self._inpaints():
            # Eye-view FOV raster, shifted by gaze:
            view_xs, view_ys = self._view_axes()
            wide = self._fov_layer(view_xs, view_ys, gaze_xy, frame, percept,
                                   vmax, vmin)
            wide_xs, wide_ys = view_xs + gx, view_ys + gy
        else:
            # Source pixel centers, so `_source_on` does not resample:
            wide = self._fov_layer(xs - gx, ys - gy, gaze_xy, frame, percept,
                                   vmax, vmin)
            wide_xs, wide_ys = xs, ys
        source = _as_rgb(self._frames()[..., self._source_frame(frame)])
        context = Percept((context_alpha * source)[..., np.newaxis],
                          space=_raster_grid(xs, ys))
        # Only `figsize` is passed; styling comes from `context_alpha`:
        ax = context.plot(ax=ax, **{key: kwargs[key] for key in ('figsize',)
                                    if key in kwargs})
        fov = Percept(wide[..., np.newaxis],
                      space=_raster_grid(wide_xs, wide_ys))
        ax = self._label_scene(fov.plot(ax=ax, **kwargs))
        artists = [ax.images[-1]]
        if patch is not None:
            artists.append(_imshow_within(ax, *patch,
                                          artists[0].get_zorder() + 1))
        artists += vf.draw(ax, radii, angles, gaze_xy, extent,
                           color=grid_color)
        clip = self._support_patch(ax.transData, center=gaze_xy)
        for artist in artists:
            artist.set_clip_path(clip)
        return ax

    def _label_fov(self, ax):
        """Sets limits and ticks to the FOV's outer edges

        `Percept` limits axes to the outermost pixel centers, which clips
        half of each edge pixel.
        """
        return _label_limits(ax, self._view_extent)

    def _label_scene(self, ax):
        """Sets limits and ticks to the outer edges of ``extent``"""
        return _label_limits(ax, self._extent)

    def _grid_geometry(self, rings, meridians, center=(0.0, 0.0)):
        """Returns ring radii (dva), meridian angles (deg), and the FOV extent
        centered on ``center``"""
        cx, cy = center
        left, right, bottom, top = self._view_extent
        extent = (left + cx, right + cx, bottom + cy, top + cy)
        # The nearest FOV edge also bounds a round aperture:
        r_min, r_max = vf.visible_band((0, 0), self._view_extent)
        return (vf.ring_radii(rings, r_max, r_min=r_min),
                vf.meridian_angles(meridians), extent)

    def play(self, gaze=None, rings=False, meridians=False,
             grid_color=vf.GRID_COLOR, ax=None, *, percept=None, vmax=None,
             vmin=None, fps=None, repeat=True, annotate_time=True,
             fmt='png', title=None, view=_SCENE_VIEW,
             context_alpha=_CONTEXT_ALPHA):
        """Animate a video scene, optionally with a prosthetic percept

        Shows the content :py:meth:`~pulse2percept.vision.Scene.render`
        returns for the same ``percept``, ``gaze``, ``vmax``, and ``vmin``,
        at its frame times.

        With ``view='scene'`` (default), the axes span ``extent``: the source
        stays fixed, the FOV moves with gaze, and the source outside the FOV
        is dimmed to ``context_alpha * source``. With ``view='eye'``, the
        frames are those ``render`` returns: a fixed eye-centered FOV with
        the source moving through it.

        Parameters
        ----------
        gaze : (x, y), (n_frames, 2), or :py:class:`~pulse2percept.vision.Gaze`, optional
            Scene location (dva) on the fovea. One (x, y) for fixed gaze, or
            one per frame. A :py:class:`~pulse2percept.vision.Gaze` is
            resolved at the scene's frame times.
        rings, meridians, grid_color : optional
            Visual-field grid, as in
            :py:meth:`~pulse2percept.vision.Scene.plot`, drawn into the
            displayed frames. Scene data is unchanged.
        ax : matplotlib.axes.Axes, optional
            Axes to animate on. If None, creates new axes.
        percept : :py:class:`~pulse2percept.percepts.Percept`, optional
            Brightness percept composed into the scene as in
            :py:meth:`~pulse2percept.vision.Scene.render`.
        vmax : float, optional
            Percept brightness shown as white. Defaults to the maximum over
            the whole ``percept``.
        vmin : float, optional
            Percept brightness shown as black. Defaults to 0.
        fps, repeat, annotate_time, fmt, title : optional
            Player options, as in
            :py:meth:`~pulse2percept.percepts.Percept.play`.
        view : {'scene', 'eye'}, optional
            Scene-centered frames spanning ``extent`` (default), or
            eye-centered frames spanning the FOV.
        context_alpha : float, optional
            Opacity in [0, 1] of the source outside the FOV, as in
            :py:meth:`~pulse2percept.vision.Scene.plot`.

        Returns
        -------
        ani : :py:class:`~pulse2percept.utils.HTMLAnimation`

        """
        view = _resolve_view(view)
        if self.time is None:
            raise ValueError("A still scene has nothing to play. Use plot().")
        gaze = self._resolve_gaze(gaze, percept)
        # Brightness is scaled here; the player receives RGB:
        player = dict(fps=fps, repeat=repeat, annotate_time=annotate_time,
                      ax=ax, fmt=fmt, title=title)
        if view == _EYE_VIEW:
            return self._play_eye(gaze, rings, meridians, grid_color, percept,
                                  vmax, vmin, player)
        return self._play_scene(gaze, rings, meridians, grid_color, percept,
                                vmax, vmin,
                                _resolve_context_alpha(context_alpha), player)

    def _play_eye(self, gaze, rings, meridians, grid_color, percept, vmax,
                  vmin, player):
        """`play` on eye-centered frames spanning the FOV"""
        radii, angles, extent = self._grid_geometry(rings, meridians)
        display = self.render(percept=percept, gaze=gaze, vmax=vmax,
                              vmin=vmin)
        if not radii.size and not angles.size:
            return self._fov_player(display.play(**player))
        # Draw the grid into the frames (the player canvas would hide an
        # artist). Eye-centered, so one overlay works for any gaze:
        xs, ys = self._view_axes()
        overlay = vf.rasterize((ys.size, xs.size), radii, angles, (0, 0),
                               extent, _to_pixel(xs, ys), color=grid_color)
        if self._aperture == _ROUND:
            overlay[self._aperture_mask(xs, ys), 3] = 0
        # Keep rendered time stamps (may be frame ends for temporal percepts):
        decorated = Percept(_over(display.data, overlay),
                            space=_raster_grid(xs, ys),
                            time=display.time, time_unit=display.time_unit)
        return self._fov_player(decorated.play(**player))

    def _play_scene(self, gaze, rings, meridians, grid_color, percept, vmax,
                    vmin, context_alpha, player):
        """`play` on scene-centered frames spanning ``extent``"""
        xs, ys = self._axes
        source = self._frames()
        n_out = self._n_display_frames(percept)
        points = _gaze_points(gaze, n_out)
        data = np.empty((ys.size, xs.size, 3, n_out), dtype=np.float32)
        times, unit, overlays = [], None, {}
        for f in range(n_out):
            gx, gy = points[0] if len(points) == 1 else points[f]
            # Eye coordinates of the source pixel centers:
            eye_xs, eye_ys = xs - gx, ys - gy
            if self._inpaints():
                # Compose on the eye-view raster, then resample onto the
                # source raster:
                view_xs, view_ys = self._view_axes()
                shown, time, unit = self._display_on(view_xs, view_ys,
                                                     percept=percept,
                                                     vmax=vmax, vmin=vmin,
                                                     gaze=(gx, gy), frame=f)
                shown = _regrid(view_xs, view_ys, shown[..., 0], eye_xs,
                                eye_ys)
            else:
                shown, time, unit = self._display_on(eye_xs, eye_ys,
                                                     percept=percept,
                                                     vmax=vmax, vmin=vmin,
                                                     gaze=(gx, gy), frame=f)
                shown = shown[..., 0]
            outside = self._outside_support(eye_xs, eye_ys)
            context = context_alpha * _as_rgb(
                source[..., self._source_frame(f)])
            rgb = np.where(outside[..., np.newaxis], context, shown)
            radii, angles, extent = self._grid_geometry(rings, meridians,
                                                        center=(gx, gy))
            if radii.size or angles.size:
                # Rasterizing requires a figure draw; cache per gaze:
                key = (float(gx), float(gy))
                if key not in overlays:
                    overlay = vf.rasterize((ys.size, xs.size), radii, angles,
                                           key, extent, _to_pixel(xs, ys),
                                           color=grid_color)
                    overlay[outside, 3] = 0
                    overlays[key] = overlay
                rgb = _over(rgb[..., np.newaxis], overlays[key])[..., 0]
            data[..., f] = rgb
            times.append(time)
        # Per-frame time stamps (may be frame ends for temporal percepts):
        display = Percept(data, space=_raster_grid(xs, ys),
                          time=np.concatenate(times), time_unit=unit)
        ani = display.play(**player)
        self._label_scene(ani._image.axes)
        return ani

    def _fov_player(self, ani):
        """Sets player axes to the full FOV (read when the HTML is built)"""
        self._label_fov(ani._image.axes)
        return ani
