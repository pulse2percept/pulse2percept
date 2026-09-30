""":py:class:`~pulse2percept.percepts.Percept`"""
import numpy as np
import os
import re
import warnings
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.axes import Subplot
from scipy.cluster.vq import kmeans2
import imageio
import imageio.v3 as iio
import logging
from skimage import img_as_float32, img_as_ubyte
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import resize

from ..units import DimensionMismatchError, Hz, Quantity, Unit, as_value, ms
from ..utils import Data, HTMLAnimation, frame_interval
from ..utils import _visual_field as vf
from ..utils.animation import _frame_timeline
from ..utils.array import _interp_rows, _slice_times
from ..utils.constants import VIDEO_BLOCK_SIZE

# A number in a brightness-range tag:
_NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'

# Saved brightness range in media metadata or file name, e.g.
# 'foo__p2p_vmin=0.0_vmax=20.0.png'. The ``__p2p_`` prefix avoids matching
# tags written by other tools:
_P2P_RANGE_RE = re.compile(rf'__p2p_(?:vmin=(?P<vmin>{_NUM}))?_?'
                           rf'(?:vmax=(?P<vmax>{_NUM}))?')


def _range_tag(vmin, vmax):
    """Return the brightness-range tag for ``vmin``, ``vmax``"""
    return f'__p2p_vmin={float(vmin)!r}_vmax={float(vmax)!r}'


def _parse_range_tag(text):
    """Return (vmin, vmax) from a range tag; missing values are None"""
    if not isinstance(text, str):
        return None, None
    match = _P2P_RANGE_RE.search(text)
    if match is None:
        return None, None
    return tuple(None if val is None else float(val)
                 for val in match.group('vmin', 'vmax'))


def _range_tagged_path(fname, vmin, vmax):
    """Return ``fname`` with its stored brightness range in the filename."""
    head, tail = os.path.split(os.fspath(fname))
    root, ext = os.path.splitext(tail)
    tag = _range_tag(vmin, vmax)
    root, n_replaced = _P2P_RANGE_RE.subn(lambda match: tag, root, count=1)
    return os.path.join(head, (root if n_replaced else root + tag) + ext)


def _media_metadata(fname):
    """Yield metadata dictionaries from imageio backends that can read ``fname``."""
    for kwargs in ({'plugin': 'pillow'}, {}):
        try:
            yield dict(iio.immeta(fname, **kwargs))
        except Exception:
            # Try the next backend:
            continue


def _media_range(fname):
    """Return (vmin, vmax) stored in a media file's metadata"""
    for meta in _media_metadata(fname):
        for key, value in meta.items():
            if str(key).lower() != 'comment':
                continue
            if isinstance(value, bytes):
                value = value.decode('utf-8', 'replace')
            vmin, vmax = _parse_range_tag(value)
            if vmin is not None or vmax is not None:
                return vmin, vmax
    return None, None


def _frame_durations(fname, n_frames):
    """Return per-frame durations (ms) of a Pillow-readable file, or None"""
    durations = []
    for index in range(n_frames):
        try:
            meta = iio.immeta(fname, plugin='pillow', index=index)
        except Exception:
            return None
        if 'duration' not in meta:
            return None
        durations.append(float(meta['duration']))
    return durations


def _media_fps(fname, n_frames):
    """Return the frame rate (Hz) stored in a media file, or None"""
    for meta in _media_metadata(fname):
        if meta.get('fps'):
            # FFMPEG reports the rate directly:
            return float(meta['fps'])
        if 'fps' in meta or not meta.get('duration'):
            continue
        # Pillow reports ms per frame:
        durations = _frame_durations(fname, n_frames)
        if durations is None:
            return 1000.0 / float(meta['duration'])
        if max(durations) - min(durations) > 1e-6:
            raise ValueError(
                f"The frames of '{fname}' are not all the same length "
                f"({min(durations):g} to {max(durations):g} ms), so it "
                f"has no one frame rate. Pass 'time' instead.")
        return 1000.0 / durations[0]
    return None


def _metadata_kwargs(fname, tag):
    """Return writer arguments that store ``tag`` in supported media metadata."""
    ext = os.path.splitext(os.fspath(fname))[1].lower()
    if ext == '.png':
        try:
            from PIL.PngImagePlugin import PngInfo
        except ImportError:
            return {}
        info = PngInfo()
        info.add_text('Comment', tag)
        return {'pnginfo': info}
    if ext == '.gif':
        return {'comment': tag.encode('utf-8')}
    return {}


# Single-image formats (cannot store a time axis):
_STILL_EXTENSIONS = ('.jpg', '.jpeg', '.bmp', '.png', '.tif', '.tiff', '.jif',
                     '.jfif')


def _check_clim(vmin, vmax):
    """Raise ValueError unless vmin, vmax are finite and vmin <= vmax"""
    if not np.all(np.isfinite([vmin, vmax])) or vmax < vmin:
        raise ValueError(f"'vmin' ({vmin}) and 'vmax' ({vmax}) must be finite "
                         f"with 'vmin' <= 'vmax'.")


def _resolve_clim(data, vmin, vmax, auto_vmin):
    """Fill omitted display limits from the full percept."""
    vmin = auto_vmin if vmin is None else vmin
    vmax = np.max(data) if vmax is None else vmax
    vmin, vmax = float(vmin), float(vmax)
    _check_clim(vmin, vmax)
    return vmin, vmax


def _is_rgb(data):
    """Return True for RGB, False for gray; raise ValueError if invalid"""
    shape = np.shape(data)
    if len(shape) == 3:
        return False
    if not (len(shape) == 4 and shape[2] == 3):
        raise ValueError(f"Percept data must have shape (Y, X, T) for "
                         f"brightness or (Y, X, 3, T) for RGB, not "
                         f"{tuple(shape)}.")
    values = np.asarray(data)
    if not np.all(np.isfinite(values)):
        raise ValueError("RGB percept data must be finite.")
    if values.min() < 0 or values.max() > 1:
        raise ValueError(f"RGB percept data are display intensities and must "
                         f"lie in [0, 1], but this one spans "
                         f"[{values.min():g}, {values.max():g}]. Scale it "
                         f"explicitly if it is in some other unit.")
    return True


def _reject_rgb(name, extra=''):
    """Return the ValueError for a brightness-only operation on RGB"""
    return ValueError(f"'{name}' is defined on perceived brightness, and has "
                      f"no unambiguous meaning for an RGB percept.{extra}")


def _quantize_gray(data, n_gray):
    """Return float32 ``data`` reduced to ``n_gray`` k-means levels.

    Cluster initialization uses NumPy's global RNG.
    """
    n_gray = int(n_gray)
    if n_gray <= 1:
        raise ValueError(f'"n_gray" must be greater than 1, not {n_gray}.')
    data = np.asarray(data, dtype=np.float32)
    centroids, labels = kmeans2(data.ravel(), n_gray, minit='points')
    return centroids[labels].reshape(data.shape)


def _pixel_extent(xdva, ydva):
    """Return (left, right, bottom, top) edges of a pixel-center grid"""
    def edges(centers):
        centers = np.asarray(centers, dtype=float)
        # A single row or column uses half-width 0.5:
        half = (centers[1] - centers[0]) / 2 if centers.size > 1 else 0.5
        return centers[0] - half, centers[-1] + half
    return (*edges(xdva), *edges(ydva))


class Percept(Data):
    """Visual percept in space and time.

    Percepts are typically produced by computational models. A percept has one
    of two layouts, with time as the last axis in both::

        (Y, X, T)     perceived brightness in arbitrary units
        (Y, X, 3, T)  RGB intensities in [0, 1]

    Models produce brightness percepts. RGB percepts are used to display a
    scene alongside a modeled percept. RGB values are display intensities and
    must be finite and lie in [0, 1]. Brightness-only operations raise
    ValueError for RGB percepts (see Notes).

    .. versionadded:: 0.6

    .. versionchanged:: 0.11.0

        Added RGB percepts.

    Parameters
    ----------
    data : 3D or 4D array_like
        Percept data in (Y, X, T) or (Y, X, 3, T) dimensions. RGB data must be
        finite and lie in [0, 1].
    space : :py:class:`~pulse2percept.topography.Grid2D`, optional
        Spatial coordinates of the percept. If None, ``xdva`` and ``ydva``
        hold pixel indices, not dva; see Notes.
    time : 1D array_like, optional
        Time points corresponding to the frames. Bare values are expressed in
        ``time_unit``; unitful values are converted to it.
    metadata : dict, optional
        Additional percept metadata.
    n_gray : int, optional
        Number of gray levels. If specified, k-means clustering is used to
        reduce the percept to ``n_gray`` levels. Not available for RGB.
    time_unit : :py:class:`~pulse2percept.units.Unit`, optional
        Unit in which ``time`` is stored.

        .. versionadded:: 0.10.0

    Notes
    -----
    Spatial dimensions use standard NumPy indexing. When a time axis exists,
    values indexing the last dimension are interpreted as time points and may
    be interpolated; see :py:meth:`Percept.__getitem__`. The RGB axis is not a
    spatial dimension: ``space`` describes ``(Y, X)``, and a frame is
    ``(Y, X)`` or ``(Y, X, 3)``.

    ``n_gray``, ``argmax``, ``max``, and the ``vmin``/``vmax`` display range
    are defined on perceived brightness and raise ``ValueError`` for an RGB
    percept, since ranking RGB values requires a color metric. Use
    ``percept.data`` for plain numerical operations.

    A percept built without ``space`` still reports ``xdva`` and ``ydva``, but
    these are pixel indices, not visual-field coordinates.

    Examples
    --------
    A one-frame RGB percept, and the frame it displays:

    >>> import numpy as np
    >>> from pulse2percept.percepts import Percept
    >>> rgb = Percept(np.zeros((4, 6, 3, 1)))
    >>> rgb.is_rgb
    True
    >>> rgb[..., 0].shape
    (4, 6, 3)

    """

    def __init__(self, data, space=None, time=None, metadata=None, n_gray=None,
                 time_unit=ms):
        # import at runtime to avoid circular import
        from ..topography import Grid2D
        if not isinstance(time_unit, Unit):
            raise TypeError(f"'time_unit' must be a Unit object, not "
                            f"{type(time_unit)}.")
        if time_unit.dimension != ms.dimension:
            raise DimensionMismatchError(
                f"'time_unit' must be a unit of time (e.g. ms, s), not "
                f"{time_unit.dimension.name} ({time_unit}).")
        self._time_unit = time_unit
        data = deepcopy(data)
        is_rgb = _is_rgb(data)
        # Distinguishes real coordinates from pixel-index placeholders:
        self._has_space = space is not None
        xdva = None
        ydva = None
        if space is not None:
            if not isinstance(space, Grid2D):
                raise TypeError(f"'space' must be a Grid2D object, not "
                                f"{type(space)}.")
            xdva = space._xflat
            ydva = space._yflat
        # Reduce number of gray levels if requested:
        if n_gray is not None:
            if is_rgb:
                raise _reject_rgb('n_gray', ' Quantize the color channels '
                                            'yourself if that is what you '
                                            'want.')
            data = _quantize_gray(data, n_gray)
        time = as_value(time, self._time_unit, 'time')
        if time is not None:
            time = np.array([time]).flatten()
        # `Data` requires one axis label per dimension (RGB: channel index):
        axes = [('ydva', ydva), ('xdva', xdva), ('time', time)]
        if is_rgb:
            axes.insert(2, ('channel', np.arange(3)))
        self._internal = {
            'data': data,
            'axes': axes,
            'metadata': metadata
        }

    def __getitem__(self, item):
        """Return percept data, interpolating requested time points as needed.

        Spatial dimensions use normal NumPy indexing. A numeric index that
        reaches the final axis is interpreted as time rather than a frame
        number. Returns an array or scalar, not a new :class:`Percept`.

        .. versionadded:: 0.10.0
        """
        # Only an index that reaches the last axis names a time point
        # (``percept[0, 1]`` returns a pixel's time series):
        space, time = item, None
        if self.time is not None and isinstance(item, tuple) and len(item) > 1:
            head = item[:-1]
            if (any(idx is Ellipsis for idx in head) or
                    len(head) == self.data.ndim - 1):
                space, time = head, item[-1]
        # Distinguish time values from ordinary NumPy frame indices.
        scalar_time = mask_time = False
        if isinstance(time, slice):
            sliced = _slice_times(time, self.time, self.time_unit)
            if sliced is not None:
                time = sliced
            # Otherwise NumPy treats the slice as frame indices below.
        elif time is not None and time is not Ellipsis:
            time = as_value(time, self.time_unit, 'time')
            if np.asarray(time).dtype == bool:
                # A boolean mask selects stored frames:
                mask_time = True
            else:
                # Convert to float so time is not mistaken for a frame index:
                time = np.float64(time)
                scalar_time = time.ndim == 0
        # Let NumPy handle ordinary indexing first.
        try:
            return self.data[space if time is None else (*space, time)]
        except IndexError:
            # A float index fails in NumPy, so interpolate it as a time below.
            # A mask of the wrong length re-raises IndexError:
            if time is None or mask_time:
                raise
        # Interpolate explicit time values.
        frames = self.data[space]
        times = np.array([time], dtype=np.float64).ravel()
        # One row per pixel:
        data = _interp_rows(times, self.time,
                            frames.reshape((-1, len(self.time))))
        data = data.reshape(frames.shape[:-1] + times.shape)
        if scalar_time:
            # A scalar index drops the axis it indexes:
            data = data[..., 0]
        if data.ndim == 0:
            return data.item()
        return data

    @property
    def is_rgb(self):
        """Whether this percept is RGB (Y, X, 3, T) rather than (Y, X, T)

        .. versionadded:: 0.11.0
        """
        return self.data.ndim == 4

    def _inherit_space(self, other):
        """Copy visual-field coordinates from ``other`` if the grid matches"""
        if not getattr(other, '_has_space', False):
            return self
        coords = {name: getattr(other, name) for name in ('ydva', 'xdva')}
        for dim, name in enumerate(('ydva', 'xdva')):
            # One-point axes store None; others must match in size:
            values = coords[name]
            if values is not None and np.size(values) != self.data.shape[dim]:
                return self
        axes = self._internal['axes']
        for name, values in coords.items():
            if values is not None:
                axes[name] = np.asarray(values)
        self._has_space = True
        return self

    @property
    def time_unit(self):
        """Unit in which ``time`` is stored.

        The property is read-only; use :meth:`times` to request another
        unit.

        .. versionadded:: 0.10.0
        """
        return self._time_unit

    @property
    def time_quantity(self):
        """Time axis with its unit attached, or None.

        .. versionadded:: 0.10.0
        """
        if self.time is None:
            return None
        return Quantity(self.time, self.time_unit)

    def times(self, unit=None):
        """Return the time axis in ``unit``

        .. versionadded:: 0.10.0

        Parameters
        ----------
        unit : :py:class:`~pulse2percept.units.Unit`, optional
            The unit to express the time axis in. If None, ``time`` is
            returned as it is stored.

        Returns
        -------
        times : np.ndarray or None
            Plain NumPy array (not a
            :py:class:`~pulse2percept.units.Quantity`), or None if the percept
            has no time component.

        Examples
        --------
        >>> import numpy as np
        >>> from pulse2percept.percepts import Percept
        >>> from pulse2percept.units import s
        >>> Percept(np.zeros((3, 3, 2)), time=[0, 20.0]).times(s)
        array([0.  , 0.02])

        """
        if self.time is None:
            return None
        if unit is None:
            return self.time
        return self.time_quantity.to_value(unit)

    def argmax(self, axis=None):
        """Return the indices of the maximum values along an axis

        Parameters
        ----------
        axis : None or 'frames'
            Axis along which to operate.
            By default, the index of the brightest pixel is returned.
            Set ``axis='frames'`` to get the index of the brightest frame.

        Returns
        -------
        argmax : ndarray or scalar
            Indices at which the maxima of ``percept.data`` along an axis occur.
            If `axis` is None, the result is a scalar value.
            If `axis` is 'frames', the result is the time of the brightest
            frame.

        Raises
        ------
        ValueError
            For an RGB percept.
        """
        if axis is not None and not isinstance(axis, str):
            raise TypeError('"axis" must be a string or None.')
        if self.is_rgb:
            raise _reject_rgb('argmax', ' Use percept.data.argmax() for the '
                                        'largest number it holds.')
        if axis is None:
            return self.data.argmax()
        elif axis.lower() == 'frames':
            return np.argmax(np.max(self.data, axis=(0, 1)))
        raise ValueError(f'Unknown axis value "{axis}". Use "frames" or '
                         f'None.')

    def max(self, axis=None):
        """Brightest pixel or frame

        Parameters
        ----------
        axis : None or 'frames'
            Axis along which to operate.
            By default, the value of the brightest pixel is returned.
            Set ``axis='frames'`` to get the brightest frame.

        Returns
        -------
        pmax : ndarray or scalar
            Maximum of ``percept.data``.
            If `axis` is None, the result is a scalar value.
            If `axis` is 'frames', the result is the brightest frame.

        Raises
        ------
        ValueError
            For an RGB percept.
        """
        if axis is not None and not isinstance(axis, str):
            raise TypeError('"axis" must be a string or None.')
        if self.is_rgb:
            raise _reject_rgb('max', ' Use percept.data.max() for the largest '
                                     'number it holds.')
        if axis is None:
            return self.data.max()
        elif axis.lower() == 'frames':
            return self.data[..., self.argmax(axis='frames')]
        raise ValueError(f'Unknown axis value "{axis}". Use "frames" or '
                         f'None.')

    def measure(self, threshold=0.5):
        """Measure the brightness and geometry of the phosphenes in a percept

        Measures integrated and peak brightness, and the position, size, and
        shape of the suprathreshold support, for each frame of a brightness
        percept. Results are not cached.

        See :py:func:`~pulse2percept.percepts.metrics.measure_percept` for
        definitions, units, and invalid inputs.

        .. versionadded:: 0.11.0

        Parameters
        ----------
        threshold : float, optional
            Fraction of each frame's own positive maximum at or above which a
            pixel belongs to the support. Must lie in (0, 1].

        Returns
        -------
        metrics : :py:class:`~pulse2percept.percepts.metrics.PerceptMetrics`
            One :py:class:`~pulse2percept.percepts.metrics.FrameMetrics` per
            frame, plus the framewise arrays and the brightest frame.

        """
        # Lazy import, so predicting a percept does not load metrics:
        from .metrics import measure_percept
        return measure_percept(self, threshold=threshold)

    def rewind(self):
        """Rewind the iterator"""
        self._next_frame = 0

    def __iter__(self):
        """Iterate over all frames in self.data"""
        self.rewind()
        return self

    def __next__(self):
        """Returns the next frame when iterating over all frames"""
        this_frame = self._next_frame
        if this_frame >= self.data.shape[-1]:
            raise StopIteration
        self._next_frame += 1
        return self.data[..., this_frame]

    def plot(self, kind='pcolor', ax=None, rings=False, meridians=False,
             grid_color=vf.GRID_COLOR, **kwargs):
        """Plot the percept

        For a spatial percept, will plot the perceived brightness across the
        x, y grid.
        For a temporal percept, will plot the evolution of perceived brightness
        over time.
        For a spatiotemporal percept, will plot the brightest frame.
        Use ``percept.play()`` to animate the percept across time points.

        An RGB percept is drawn without a colormap. A multi-frame RGB percept
        raises ValueError (no brightest frame); use ``play()``.

        Parameters
        ----------
        kind : { 'pcolor', 'hex' }, optional
            Kind of plot to draw:

            *  'pcolor': using Matplotlib's ``pcolor``. Additional parameters
               (e.g., ``vmin``, ``vmax``) can be passed as keyword arguments.
               By default, ``vmin`` is the drawn frame's minimum and ``vmax``
               the maximum brightness across the percept.
            *  'hex': using Matplotlib's ``hexbin``. Additional parameters
               (e.g., ``gridsize``) can be passed as keyword arguments.
        ax : matplotlib.axes.AxesSubplot, optional
            A Matplotlib axes object. If None, will either use the current axes
            (if exists) or create a new Axes object
        rings : bool, float, or sequence, optional
            Eccentricity rings (dva) about the fovea at visual-field (0, 0).
            True draws 1.25, 2.5, 5, 10, 20, ... dva, a number is a spacing,
            and a sequence is the eccentricities themselves. Automatic rings
            stop at the field edge nearest the fovea, or, for a field that
            excludes the fovea, span the eccentricities it covers.
        meridians : bool, float, or sequence, optional
            Polar-angle meridians (geometric deg) from the fovea to the field
            edge: 0 is +x, 90 is +y, counterclockwise. True is every 45 deg,
            a number is a spacing from 0, and a sequence is the angles
            themselves.
        grid_color : color, optional
            Matplotlib color of rings, meridians, and ring labels.
        **kwargs :
            Other optional arguments passed down to the Matplotlib function

        Returns
        -------
        ax : matplotlib.axes.Axes
            Returns the axes with the plot on it

        Notes
        -----
        Rings and meridians are display annotations. They require a percept
        built with ``space`` (``ValueError`` otherwise), since pixel indices
        are not visual angle.

        """
        grid = self._grid_geometry(rings, meridians)
        if ax is None:
            ax = plt.gca()
            if 'figsize' in kwargs:
                ax.figure.set_size_inches(kwargs['figsize'])
        else:
            if not isinstance(ax, Subplot):
                raise TypeError(f"'ax' must be a Matplotlib axis, not "
                                f"{type(ax)}.")
        if self.xdva is None and self.ydva is None and self.time is not None:
            # Special case of a purely temporal percept:
            trace = self.data.squeeze()
            if self.is_rgb:
                # One line (column) per channel:
                trace = trace.T
            ax.plot(self.time, trace, linewidth=2, **kwargs)
            ax.set_xlabel(f'time ({self.time_unit})')
            ax.set_ylabel('RGB intensity' if self.is_rgb
                          else 'Perceived brightness (a.u.)')
            return ax

        if self.is_rgb:
            for name in ('vmin', 'vmax', 'cmap'):
                if name in kwargs:
                    raise _reject_rgb(name, ' Its RGB values are drawn as '
                                            'they are; scale the data if you '
                                            'want a different range.')
            if kind != 'pcolor':
                raise ValueError(f"kind='{kind}' needs one number per pixel "
                                 f"and cannot draw an RGB percept. Use "
                                 f"kind='pcolor'.")
            if self.data.shape[-1] > 1:
                raise ValueError("RGB percepts do not define a brightest "
                                 "frame. Use play() to view a temporal "
                                 "percept.")
            # `pcolormesh` requires one value per pixel, so use imshow for RGB:
            drop = ['figsize', 'shading']
            other_kwargs = {key: kwargs[key]
                            for key in (kwargs.keys() - drop)}
            ax.imshow(self.data[..., 0], origin='upper',
                      extent=_pixel_extent(self.xdva, self.ydva),
                      **other_kwargs)
            return self._draw_grid(self._label_axes(ax), grid, grid_color)

        # A spatial or spatiotemporal percept: Find the brightest frame
        idx = np.argmax(np.max(self.data, axis=(0, 1)))
        frame = self.data[..., idx]

        vmin = kwargs['vmin'] if 'vmin' in kwargs.keys() else frame.min()
        vmax = kwargs['vmax'] if 'vmax' in kwargs.keys() else self.data.max()
        cmap = kwargs['cmap'] if 'cmap' in kwargs.keys() else 'gray'
        shading = kwargs['shading'] if 'shading' in kwargs.keys() else 'nearest'
        X, Y = np.meshgrid(self.xdva, self.ydva, indexing='xy')
        if kind == 'pcolor':
            # Create a pseudocolor plot. Make sure to pass additional keyword
            # arguments that have not already been extracted:
            other_kwargs = {key: kwargs[key]
                            for key in (kwargs.keys() - ['figsize', 'cmap',
                                                         'vmin', 'vmax'])}
            ax.pcolormesh(X, Y, np.flipud(frame), cmap=cmap, vmin=vmin,
                          vmax=vmax, shading=shading, **other_kwargs)
        elif kind == 'hex':
            # Create a hexbin plot:
            gridsize = kwargs['gridsize'] if 'gridsize' in kwargs else 80
            # X, Y = np.meshgrid(self.xdva, self.ydva, indexing='xy')
            # Make sure to pass additional keyword arguments that have not
            # already been extracted:
            other_kwargs = {key: kwargs[key]
                            for key in (kwargs.keys() - ['figsize', 'cmap',
                                                         'gridsize', 'vmin',
                                                         'vmax'])}
            ax.hexbin(X.ravel(), Y.ravel()[::-1], frame.ravel(),
                      cmap=cmap, gridsize=gridsize, vmin=vmin, vmax=vmax,
                      **other_kwargs)
        else:
            raise ValueError(f"Unknown plot option '{kind}'. Choose either "
                             f"'pcolor' or 'hex'.")
        return self._draw_grid(self._label_axes(ax), grid, grid_color)

    def _grid_geometry(self, rings, meridians):
        """Ring radii (dva), meridian angles (deg), and visible extent, or
        None if no grid is requested"""
        if vf._is_off(rings) and vf._is_off(meridians):
            return None
        if not self._has_space or self.xdva is None or self.ydva is None:
            raise ValueError("Rings and meridians require visual-field "
                             "coordinates, and this percept has none: its "
                             "xdva/ydva are pixel indices or absent. Pass "
                             "'space' when building it, or predict it on a "
                             "model grid.")
        extent = (float(np.min(self.xdva)), float(np.max(self.xdva)),
                  float(np.min(self.ydva)), float(np.max(self.ydva)))
        r_min, r_max = vf.visible_band((0, 0), extent)
        return (vf.ring_radii(rings, r_max, r_min=r_min),
                vf.meridian_angles(meridians), extent)

    @staticmethod
    def _draw_grid(ax, grid, color):
        """Draw a `_grid_geometry` result on ``ax``, centered at (0, 0)"""
        if grid is not None:
            radii, angles, extent = grid
            vf.draw(ax, radii, angles, (0, 0), extent, color=color)
        return ax

    @staticmethod
    def _grid_layer(ax, grid, color, zorder):
        """Rasterize a `_grid_geometry` result over the visible axes, at the
        axes' on-screen size, and show it as an RGBA image"""
        radii, angles, extent = grid
        left, right, bottom, top = extent
        ax.apply_aspect()
        bbox = ax.get_window_extent()
        shape = (max(int(round(bbox.height)), 1),
                 max(int(round(bbox.width)), 1))

        def to_pixel(x, y):
            col = (np.asarray(x) - left) / (right - left) * shape[1] - 0.5
            row = (top - np.asarray(y)) / (top - bottom) * shape[0] - 0.5
            return col, row

        rgba = vf.rasterize(shape, radii, angles, (0, 0), extent, to_pixel,
                            color=color, dpi=ax.figure.dpi)
        return ax.imshow(rgba, origin='upper', extent=extent, zorder=zorder)

    def _label_axes(self, ax):
        """Set equal aspect, limits, ticks, and dva labels on ``ax``"""
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(self.xdva[0], self.xdva[-1])
        ax.set_xticks(np.linspace(self.xdva[0], self.xdva[-1], num=5))
        ax.set_xlabel('x (degrees of visual angle)')
        ax.set_ylim(self.ydva[0], self.ydva[-1])
        ax.set_yticks(np.linspace(self.ydva[0], self.ydva[-1], num=5))
        ax.set_ylabel('y (degrees of visual angle)')
        return ax

    def play(self, fps=None, repeat=True, annotate_time=True, ax=None,
            colorbar=True, fmt='png', vmin=None, vmax=None, rings=False,
            meridians=False, grid_color=vf.GRID_COLOR, title=None):
        """Animate the percept in an interactive HTML player.

        Parameters
        ----------
        fps : float, optional
            Display frame rate in Hz. If None, use the percept's recorded timing.
        repeat : bool, optional
            Whether to repeat the animation.
        annotate_time : bool, optional
            Whether to show the current time above each frame.
        ax : matplotlib.axes.Axes, optional
            Axes on which to draw the animation.
        colorbar : bool, optional
            Whether to show a colorbar. Ignored for RGB percepts.
        fmt : {'png', 'jpg'}, optional
            Image format used to encode animation frames.

            .. versionadded:: 0.10.0
        vmin, vmax : float, optional
            Brightness limits. By default, ``vmin=0`` and ``vmax`` is the maximum
            brightness across the percept. Not available for an RGB percept,
            whose values are shown as they are (clipped to [0, 1]).

            .. versionadded:: 0.10.0
        rings, meridians, grid_color : optional
            Visual-field grid about (0, 0), as in :py:meth:`plot`, drawn as a
            static layer over every frame.

            .. versionadded:: 0.11.0
        title : str, optional
            Figure title (``fig.suptitle``), shown independently of
            ``annotate_time``.

            .. versionadded:: 0.11.0

        Returns
        -------
        pulse2percept.utils.HTMLAnimation
            The animation.

        Notes
        -----
        ``fps`` controls display sampling, not interpolation. Use
        ``percept[..., t]`` to interpolate the percept at an arbitrary time.

        .. versionchanged:: 0.10.0
            Added support for irregular timing, ``fps`` display sampling, and
            ``vmin``/``vmax``.
        """
        if self.time is None:
            raise ValueError("Cannot animate a percept with time=None. Use "
                             "percept.plot() instead.")
        grid = self._grid_geometry(rings, meridians)
        # Convert percept times to wall-clock milliseconds:
        timeline = _frame_timeline(self.times(ms), fps=fps)
        idx = timeline.indices
        def update(i):
            if annotate_time:
                t = self.time[idx[i]]
                mat.axes.set_title(f't = {t:.2f} {self.time_unit}')
            mat.set_data(self.data[..., idx[i]])
            return mat

        def data_gen():
            yield from range(idx.size)

        # There are several options to animate a percept in Jupyter/IPython
        # (see https://stackoverflow.com/a/46878531). Displaying the animation
        # as HTML with JavaScript is compatible with most browsers and even
        # %matplotlib inline (although it can be kind of slow):
        plt.rcParams["animation.html"] = 'jshtml'
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure
        # Show an empty frame. The color scale spans the whole percept, so it
        # does not depend on `fps`. Use dva extent, as in `plot()`:
        spatial = self.xdva is not None and self.ydva is not None
        extent = _pixel_extent(self.xdva, self.ydva) if spatial else None
        if self.is_rgb:
            if vmin is not None or vmax is not None:
                raise _reject_rgb('vmin/vmax', ' Its RGB values are shown as '
                                               'they are.')
            # No colormap or colorbar for RGB:
            mat = ax.imshow(np.zeros_like(self.data[..., 0]), origin='upper',
                            extent=extent)
        else:
            vmin, vmax = _resolve_clim(self.data, vmin, vmax, auto_vmin=0)
            mat = ax.imshow(np.zeros_like(self.data[..., 0]), cmap='gray',
                            vmin=vmin, vmax=vmax, origin='upper',
                            extent=extent)
            if colorbar:
                cbar = fig.colorbar(mat)
                cbar.ax.set_ylabel('Phosphene brightness (a.u.)', rotation=-90,
                                   va='center')
        if spatial:
            self._label_axes(ax)
        images, frames, index = [mat], [self.data], [idx]
        if grid is not None:
            # The player canvas hides ordinary artists, so draw the grid as a
            # static image layer:
            images.append(self._grid_layer(ax, grid, grid_color,
                                           mat.get_zorder() + 1))
            frames.append(np.asarray(images[-1].get_array())[..., np.newaxis])
            index.append(np.zeros_like(idx))
        if title is not None:
            fig.suptitle(title)
        plt.close(fig)
        # HTMLAnimation renders frames from `frame_data` directly (no
        # Matplotlib), selecting frames via `frame_index`:
        labels = None
        if annotate_time:
            labels = [f't = {t:.2f} {self.time_unit}' for t in self.time[idx]]
        return HTMLAnimation(fig, update, data_gen, repeat=repeat,
                             intervals=timeline.intervals,
                             save_count=idx.size, image=images,
                             frame_data=frames, frame_index=index,
                             labels=labels, fmt=fmt)

    def save(self, fname, shape=None, fps=None, vmin=None, vmax=None):
        """Save the percept to an image or video file.

        Parameters
        ----------
        fname : str
            Output filename. The extension determines the file format.
        shape : ``(height, width)``, optional
            Output size in pixels. Either dimension may be ``None`` to preserve
            the percept's aspect ratio.
        fps : float, optional
            Movie frame rate in Hz. If None, use the percept's recorded timing.
        vmin, vmax : float, optional
            Brightness limits mapped to the file's gray levels. Values outside
            this range are clipped. If either limit is given, the resolved range
            is stored so that :meth:`Percept.load` can restore it. Not
            available for an RGB percept, whose values are written as they are
            (clipped to [0, 1]) and read back by ``load(..., as_gray=False)``.

            .. versionadded:: 0.10.0

        Returns
        -------
        str
            Path of the file that was written.

        Notes
        -----
        If the output format cannot store the brightness range in metadata,
        ``save`` adds it to the filename.

        Movie dimensions may be adjusted for codec compatibility.

        .. versionchanged:: 0.10.0
            Added ``vmin``/``vmax`` and return of the output filename.
        """
        fname = os.fspath(fname)
        # imageio takes a plain number in Hz:
        fps = as_value(fps, Hz, 'fps')
        if self.time is not None:
            if os.path.splitext(fname)[1].lower() in _STILL_EXTENSIONS:
                raise ValueError(f"Cannot save multi-frame percept as a "
                                 f"static image: {fname}")
        if self.is_rgb:
            if vmin is not None or vmax is not None:
                raise _reject_rgb('vmin/vmax', ' Its RGB values are '
                                               'written as they are.')
            # RGB is always stored on [0, 1]:
            fixed_clim = False
            vmin, vmax = 0.0, 1.0
            data = self.data
        else:
            # A user-given limit allows the file name to be tagged below:
            fixed_clim = vmin is not None or vmax is not None
            if not fixed_clim:
                warnings.warn("Normalizing the percept to its own brightness "
                              "range, so percepts saved separately do not "
                              "share a scale. Pass 'vmin' and 'vmax' to fix "
                              "the range.", stacklevel=2)
            # Resolve the range before resampling frames, so `fps` does not
            # change the brightness scale:
            vmin, vmax = _resolve_clim(self.data, vmin, vmax,
                                       auto_vmin=self.data.min())
            span = vmax - vmin
            if span > 0:
                data = np.clip((self.data - vmin) / span, 0, 1)
            else:
                # Constant percept: save as black (avoid dividing by zero):
                data = np.zeros(self.data.shape, dtype=np.float64)
        data = img_as_ubyte(data)

        if shape is None:
            # Use 320px width and infer height from aspect ratio:
            shape = (None, 320)
        height, width = shape
        if height is None and width is None:
            raise ValueError('If shape is a tuple, must specify either height '
                             'or width or both.')
        # Infer height or width if necessary:
        if height is None and width is not None:
            height = width / self.data.shape[1] * self.data.shape[0]
        elif height is not None and width is None:
            width = height / self.data.shape[0] * self.data.shape[1]
        # Rescale percept to desired shape. `resize` requires the trailing
        # (RGB, time) axes explicitly:
        data = resize(data, (np.int32(height), np.int32(width),
                             *data.shape[2:]))

        # Store the range for `load`: in metadata if the format supports it,
        # otherwise in the file name (only if vmin/vmax was given, or the name
        # already has a stale tag):
        meta_kwargs = _metadata_kwargs(fname, _range_tag(vmin, vmax))
        if (fixed_clim and not meta_kwargs) or _P2P_RANGE_RE.search(
                os.path.basename(fname)) is not None:
            fname = _range_tagged_path(fname, vmin, vmax)
        if self.time is None:
            # No time component, store as an image. imwrite will automatically
            # scale the gray levels:
            imageio.imwrite(fname, img_as_ubyte(data)[..., 0], **meta_kwargs)
        else:
            # With time component, store as a movie:
            if fps is None:
                # Movies require a fixed rate; `frame_interval` raises
                # NotImplementedError for irregular times. Convert ms to Hz:
                fps = 1000.0 / frame_interval(self.times(ms), tol=1e-6)
            else:
                # Same display clock as `play`: resampling changes the number
                # of frames, not the movie duration.
                timeline = _frame_timeline(self.times(ms), fps=fps)
                data = data[..., timeline.indices]
            # Most codecs require dimensions divisible by VIDEO_BLOCK_SIZE
            # (default: 16), so upsize to the next multiple:
            h, w = data.shape[:2]
            if VIDEO_BLOCK_SIZE > 1:
                if h % VIDEO_BLOCK_SIZE > 0 or w % VIDEO_BLOCK_SIZE > 0:
                    out_h, out_w = h, w
                    if w % VIDEO_BLOCK_SIZE > 0:
                        out_w += VIDEO_BLOCK_SIZE - (w % VIDEO_BLOCK_SIZE)
                    if h % VIDEO_BLOCK_SIZE > 0:
                        out_h += VIDEO_BLOCK_SIZE - (h % VIDEO_BLOCK_SIZE)
                    data = resize(data, (out_h, out_w, *data.shape[2:]))
            data = img_as_ubyte(data)
            # (Y, X[, C], T) -> (T, Y, X[, C]):
            frames = np.moveaxis(data, -1, 0)
            try:
                imageio.mimwrite(fname, frames, fps=float(fps), **meta_kwargs)
            except TypeError:
                imageio.mimwrite(fname, frames, duration=1000/fps,
                                 **meta_kwargs)
        logging.getLogger(__name__).info(f'Created {fname}.')
        return fname

    @classmethod
    def load(cls, fname, space=None, time=None, fps=None, vmin=None,
            vmax=None, as_gray=True):
        """Load a percept from an image or video file.

        .. versionadded:: 0.10.0

        Parameters
        ----------
        fname : str
            File to load.
        space : :py:class:`~pulse2percept.topography.Grid2D`, optional
            Spatial coordinates of the percept.
        time : 1D array_like, optional
            Frame times. Overrides ``fps`` and timing stored in the file.
        fps : float, optional
            Frame rate in Hz. Overrides the frame rate stored in the file.
        vmin, vmax : float, optional
            Brightness limits represented by the file. Explicit values override
            any range stored in the file metadata or filename.
        as_gray : bool, optional
            Whether to convert a color file to a brightness percept. Pass False
            to load it as an RGB percept instead, keeping its three channels.

            .. versionadded:: 0.11.0

        Returns
        -------
        Percept
            Loaded percept.

        Notes
        -----
        Color images are converted to grayscale unless ``as_gray=False``. If
        the brightness range cannot be recovered, values remain on the encoded
        [0, 1] scale and a warning is issued.
        """
        # `index=...` always returns a frame stack, so RGB channels are not
        # read as frames:
        frames = iio.imread(fname, index=...)
        if frames.ndim == 4:
            if frames.shape[-1] == 4:
                # Blend alpha against black:
                frames = rgba2rgb(frames, background=(0, 0, 0))
            if frames.shape[-1] != 3:
                frames = frames[..., 0]
            elif as_gray:
                frames = rgb2gray(frames)
        if frames.ndim not in (3, 4):
            raise ValueError(f"Expected a 2-D image or a stack of them in "
                             f"'{fname}', not an array of shape "
                             f"{frames.shape}.")
        # Float in [0, 1]; (T, Y, X[, 3]) -> (Y, X[, 3], T):
        data = np.moveaxis(img_as_float32(frames), 0, -1)

        # Frame times:
        if time is None:
            fps = as_value(fps, Hz, 'fps')
            if fps is None and data.shape[-1] > 1:
                fps = _media_fps(fname, data.shape[-1])
                if fps is None:
                    raise ValueError(f"Cannot infer the frame rate of "
                                     f"'{fname}'. Pass 'fps' or 'time'.")
            if fps is not None:
                if not np.isfinite(fps) or fps <= 0:
                    raise ValueError(f"'fps' must be a finite number greater "
                                     f"than zero, not {fps}.")
                # Hz -> ms:
                time = np.arange(data.shape[-1]) * 1000.0 / fps

        # Brightness range:
        if data.ndim == 4:
            # RGB is always on [0, 1]:
            if vmin is not None or vmax is not None:
                raise _reject_rgb('vmin/vmax', ' Load it with as_gray=True to '
                                               'put its gray levels back on a '
                                               'brightness scale.')
            vmin, vmax = 0.0, 1.0
            return cls(data, space=space, time=time,
                       metadata={'source': os.fspath(fname), 'vmin': vmin,
                                 'vmax': vmax})
        file_vmin, file_vmax = _media_range(fname)
        name_vmin, name_vmax = _parse_range_tag(
            os.path.splitext(os.path.basename(os.fspath(fname)))[0])
        if vmin is None:
            vmin = file_vmin if file_vmin is not None else name_vmin
        if vmax is None:
            vmax = file_vmax if file_vmax is not None else name_vmax
        if vmin is None and vmax is None:
            warnings.warn(f"The brightness range of '{fname}' is unknown, so "
                          f"the data is left on the encoded [0, 1] scale. "
                          f"Pass 'vmin' and 'vmax' if you know it.",
                          stacklevel=2)
        elif vmin is None or vmax is None:
            # One limit alone cannot restore the scale:
            missing, known = (('vmin', f'vmax={vmax}') if vmin is None
                              else ('vmax', f'vmin={vmin}'))
            raise ValueError(f"Cannot restore the brightness scale of "
                             f"'{fname}' from {known} alone, because "
                             f"'{missing}' is unknown and the file does not "
                             f"record it. Pass '{missing}' as well.")
        else:
            _check_clim(vmin, vmax)
            data = vmin + data * (vmax - vmin)
        return cls(data, space=space, time=time,
                   metadata={'source': os.fspath(fname), 'vmin': vmin,
                             'vmax': vmax})
