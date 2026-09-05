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
from ..utils.animation import _frame_timeline
from ..utils.array import _interp_rows, _slice_times
from ..utils.constants import VIDEO_BLOCK_SIZE

# A number as it may appear in a brightness-range tag:
_NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'

# The brightness range a percept was saved with, as it appears both in media
# metadata and in a file name ('foo__p2p_vmin=0.0_vmax=20.0.png'). The
# ``__p2p_`` prefix is what keeps metadata and file names written by other
# tools from being read as pulse2percept information.
_P2P_RANGE_RE = re.compile(rf'__p2p_(?:vmin=(?P<vmin>{_NUM}))?_?'
                           rf'(?:vmax=(?P<vmax>{_NUM}))?')


def _range_tag(vmin, vmax):
    """The namespaced tag that records a brightness range"""
    return f'__p2p_vmin={float(vmin)!r}_vmax={float(vmax)!r}'


def _parse_range_tag(text):
    """The (vmin, vmax) a namespaced tag records, either one None if absent"""
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
            # Any backend may decline any file; the caller falls back on the
            # next one, and ultimately on knowing nothing about the file:
            continue


def _media_range(fname):
    """The (vmin, vmax) a media file records in its own metadata"""
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
    """How long each frame of a Pillow-readable file stays up (in ms)"""
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
    """The frame rate (in Hz) a media file records, or None"""
    for meta in _media_metadata(fname):
        if meta.get('fps'):
            # FFMPEG reports the rate directly:
            return float(meta['fps'])
        if 'fps' in meta or not meta.get('duration'):
            continue
        # Pillow counts milliseconds per frame:
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


# The containers that hold one image and no more, so that a percept with a
# time axis has to go somewhere else:
_STILL_EXTENSIONS = ('.jpg', '.jpeg', '.bmp', '.png', '.tif', '.tiff', '.jif',
                     '.jfif')


def _check_clim(vmin, vmax):
    """Reject a brightness range that cannot be mapped onto a color scale"""
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
    """Whether ``data`` is an RGB percept, having rejected anything else"""
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
    """The error for an operation that only brightness percepts have"""
    return ValueError(f"'{name}' is defined on perceived brightness, and has "
                      f"no unambiguous meaning for an RGB percept.{extra}")


def _pixel_extent(xdva, ydva):
    """Outer edges of a pixel grid whose centers sit at ``xdva``/``ydva``"""
    def edges(centers):
        centers = np.asarray(centers, dtype=float)
        # A single row or column has no spacing to halve:
        half = (centers[1] - centers[0]) / 2 if centers.size > 1 else 0.5
        return centers[0] - half, centers[-1] + half
    return (*edges(xdva), *edges(ydva))


class Percept(Data):
    """Visual percept in space and time.

    Percepts are typically produced by computational models. A percept has one
    of two layouts, with time as the last axis in both::

        (Y, X, T)     perceived brightness in arbitrary units
        (Y, X, 3, T)  RGB intensities in [0, 1]

    Models produce brightness percepts. RGB percepts exist to display a scene
    alongside a modeled percept. Their values are display intensities rather
    than brightness in arbitrary units, so they must be finite and lie in
    [0, 1]; anything else is rejected at construction. That is also why the
    operations defined on brightness reject them (see Notes).

    .. versionadded:: 0.6

    .. versionchanged:: 0.11.0

        Added RGB percepts.

    Parameters
    ----------
    data : 3D or 4D array_like
        Percept data in (Y, X, T) or (Y, X, 3, T) dimensions. RGB data must be
        finite and lie in [0, 1].
    space : :py:class:`~pulse2percept.topography.Grid2D`, optional
        Spatial coordinates of the percept. Without one, ``xdva`` and ``ydva``
        are filled in with pixel indices, which are not degrees of visual
        angle; see Notes.
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
    spatial dimension: ``space`` still describes ``(Y, X)``, and a frame comes
    out as ``(Y, X)`` or ``(Y, X, 3)``.

    ``n_gray``, ``argmax``, ``max``, and the ``vmin``/``vmax`` display range
    are defined on perceived brightness and raise a ``ValueError`` for an RGB
    percept, which already carries its own display values. Reducing three channels to one number -- to rank pixels or pick a
    brightest frame -- would have to choose a color metric, and a metric the
    models never produced is a decision for the caller, not for this class.
    ``percept.data`` is always there for the plain numerical answer.

    A percept built without a ``space`` still reports ``xdva`` and ``ydva``:
    they are the pixel indices ``Data`` fills an omitted axis with, and they
    are indistinguishable from coordinates once stored. Code that needs real
    visual-field positions -- placing a percept in a scene, say -- must ask
    whether one was given rather than trust the numbers.

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
        # An omitted spatial axis is filled in with pixel indices, which read
        # exactly like coordinates; this is what tells the two apart:
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
            n_gray = int(n_gray)
            if n_gray <= 1:
                raise ValueError(f'"n_gray" must be greater than 1, not '
                                 f'{n_gray}.')
            data = np.asarray(data, dtype=np.float32)
            centroids, labels = kmeans2(data.ravel(), n_gray, minit='points')
            data = centroids[labels].reshape(data.shape)
        time = as_value(time, self._time_unit, 'time')
        if time is not None:
            time = np.array([time]).flatten()
        # `Data` wants one axis label per dimension, so the RGB axis needs a
        # name of its own; it is a channel index, not a coordinate:
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
        # Determine whether the index reaches the time axis.
        # ``percept[0, 1]`` asks for the time series of a pixel, so only an
        # index that reaches the last axis can be naming a time point:
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
            # Otherwise the slice stays what it is, and NumPy takes the frames
            # it names below.
        elif time is not None and time is not Ellipsis:
            # A requested time point (or a list of them) may be unitful; after
            # this it is an ordinary number:
            time = as_value(time, self.time_unit, 'time')
            if np.asarray(time).dtype == bool:
                # A mask selects stored frames and is not a time at all:
                mask_time = True
            else:
                # Convert to float so time is not mistaken for a frame index:
                time = np.float64(time)
                scalar_time = time.ndim == 0
        # Let NumPy handle ordinary indexing first.
        try:
            return self.data[space if time is None else (*space, time)]
        except IndexError:
            # NumPy refusing a float is how we find out that the index named a
            # time point. A mask it refuses is the wrong length, and reading
            # its True/False as times t=1 and t=0 would answer a broken
            # question instead of raising:
            if time is None or mask_time:
                raise
        # Interpolate explicit time values.
        frames = self.data[space]
        times = np.array([time], dtype=np.float64).ravel()
        # ``_interp_rows`` interpolates rows, and a percept's rows are its
        # pixels rather than a stimulus's electrodes:
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
        """Take the visual-field coordinates of the percept this came from"""
        if not getattr(other, '_has_space', False):
            return self
        coords = {name: getattr(other, name) for name in ('ydva', 'xdva')}
        for dim, name in enumerate(('ydva', 'xdva')):
            # A one-point axis stores no coordinates at all, so there is
            # nothing to copy for it; anything else has to line up exactly, or
            # this was not a stage that merely rewrote the same grid.
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
        """The time axis, expressed in ``unit``

        .. versionadded:: 0.10.0

        Parameters
        ----------
        unit : :py:class:`~pulse2percept.units.Unit`, optional
            The unit to express the time axis in. If None, ``time`` is
            returned as it is stored.

        Returns
        -------
        times : np.ndarray or None
            An ordinary NumPy array, never a
            :py:class:`~pulse2percept.units.Quantity`, or None if the percept
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
            For an RGB percept, which has no brightest pixel or frame.
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
            For an RGB percept, which has no brightest pixel or frame.
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

    def plot(self, kind='pcolor', ax=None, **kwargs):
        """Plot the percept

        For a spatial percept, will plot the perceived brightness across the
        x, y grid.
        For a temporal percept, will plot the evolution of perceived brightness
        over time.
        For a spatiotemporal percept, will plot the brightest frame.
        Use ``percept.play()`` to animate the percept across time points.

        An RGB percept is drawn as RGB rather than through a colormap. It has
        no brightest frame to single out, so a multi-frame one raises; use
        ``play()``.

        Parameters
        ----------
        kind : { 'pcolor', 'hex' }, optional
            Kind of plot to draw:

            *  'pcolor': using Matplotlib's ``pcolor``. Additional parameters
               (e.g., ``vmin``, ``vmax``) can be passed as keyword arguments.
            *  'hex': using Matplotlib's ``hexbin``. Additional parameters
               (e.g., ``gridsize``) can be passed as keyword arguments.
        ax : matplotlib.axes.AxesSubplot, optional
            A Matplotlib axes object. If None, will either use the current axes
            (if exists) or create a new Axes object
        **kwargs :
            Other optional arguments passed down to the Matplotlib function

        Returns
        -------
        ax : matplotlib.axes.Axes
            Returns the axes with the plot on it

        """
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
                # One line per channel, and `plot` wants them in columns:
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
            # `pcolormesh` maps one number per pixel through a colormap, so RGB
            drop = ['figsize', 'shading']
            other_kwargs = {key: kwargs[key]
                            for key in (kwargs.keys() - drop)}
            ax.imshow(self.data[..., 0], origin='upper',
                      extent=_pixel_extent(self.xdva, self.ydva),
                      **other_kwargs)
            return self._label_axes(ax)

        # A spatial or spatiotemporal percept: Find the brightest frame
        idx = np.argmax(np.max(self.data, axis=(0, 1)))
        frame = self.data[..., idx]

        vmin = kwargs['vmin'] if 'vmin' in kwargs.keys() else frame.min()
        vmax = kwargs['vmax'] if 'vmax' in kwargs.keys() else frame.max()
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
            raise ValueError(f"Unknown plot option '%s'. Choose either 'pcolor'"
                             f"or '{kind}'.")
        return self._label_axes(ax)

    def _label_axes(self, ax):
        """Put a drawn percept on the visual-field axes it belongs on"""
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(self.xdva[0], self.xdva[-1])
        ax.set_xticks(np.linspace(self.xdva[0], self.xdva[-1], num=5))
        ax.set_xlabel('x (degrees of visual angle)')
        ax.set_ylim(self.ydva[0], self.ydva[-1])
        ax.set_yticks(np.linspace(self.ydva[0], self.ydva[-1], num=5))
        ax.set_ylabel('y (degrees of visual angle)')
        return ax

    def play(self, fps=None, repeat=True, annotate_time=True, ax=None,
            colorbar=True, fmt='png', vmin=None, vmax=None):
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
            Whether to show a colorbar. An RGB percept never gets one: it
            carries its own colors, and there is no one brightness scale to put
            next to them.
        fmt : {'png', 'jpg'}, optional
            Image format used to encode animation frames.

            .. versionadded:: 0.10.0
        vmin, vmax : float, optional
            Brightness limits. By default, ``vmin=0`` and ``vmax`` is the maximum
            brightness across the percept. Not available for an RGB percept,
            whose values are shown as they are (clipped to [0, 1]).

            .. versionadded:: 0.10.0

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
        # Show an empty frame. The color scale spans the whole percept, so
        # that it does not shift with the display rate:
        if self.is_rgb:
            if vmin is not None or vmax is not None:
                raise _reject_rgb('vmin/vmax', ' Its RGB values are shown as '
                                               'they are.')
            # No colormap and no brightness colorbar: an RGB frame carries its
            # own colors, and there is no one scale to put next to it.
            mat = ax.imshow(np.zeros_like(self.data[..., 0]))
        else:
            vmin, vmax = _resolve_clim(self.data, vmin, vmax, auto_vmin=0)
            mat = ax.imshow(np.zeros_like(self.data[..., 0]), cmap='gray',
                            vmin=vmin, vmax=vmax)
            if colorbar:
                cbar = fig.colorbar(mat)
                cbar.ax.set_ylabel('Phosphene brightness (a.u.)', rotation=-90,
                                   va='center')
        plt.close(fig)
        # Create the animation. The frame data is handed to HTMLAnimation so
        # that it can render the HTML player without going through Matplotlib:
        labels = None
        if annotate_time:
            labels = [f't = {t:.2f} {self.time_unit}' for t in self.time[idx]]
        return HTMLAnimation(fig, update, data_gen, repeat=repeat,
                             intervals=timeline.intervals,
                             save_count=idx.size, image=mat,
                             frame_data=self.data[..., idx], labels=labels,
                             fmt=fmt)

    def save(self, fname, shape=None, fps=None, vmin=None, vmax=None):
        """Save the percept to an image or video file.

        Parameters
        ----------
        fname : str
            Output filename. The extension determines the file format.
        shape : (height, width), optional
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
        # This path hands `fps` to imageio rather than to `frame_timeline`, so
        # it is its own boundary too: a frame rate is a frequency, and imageio
        # takes a plain number of hertz.
        fps = as_value(fps, Hz, 'fps')
        if self.time is not None:
            # A movie needs a container that can hold more than one frame:
            if os.path.splitext(fname)[1].lower() in _STILL_EXTENSIONS:
                raise ValueError(f"Cannot save multi-frame percept as a "
                                 f"static image: {fname}")
        if self.is_rgb:
            if vmin is not None or vmax is not None:
                raise _reject_rgb('vmin/vmax', ' Its RGB values are '
                                               'written as they are.')
            # An RGB percept already is what a file stores, so there is no
            # scale to pick and nothing to warn about; the range recorded below
            # is simply the one RGB always has:
            fixed_clim = False
            vmin, vmax = 0.0, 1.0
            data = self.data
        else:
            # Either limit says that the scale matters, which is what allows
            # the file name to be changed below:
            fixed_clim = vmin is not None or vmax is not None
            if not fixed_clim:
                warnings.warn("Normalizing the percept to its own brightness "
                              "range, so percepts saved separately do not "
                              "share a scale. Pass 'vmin' and 'vmax' to fix "
                              "the range.", stacklevel=2)
            # Resolve the range before any frame is dropped below, so that the
            # export rate cannot change how bright the movie comes out:
            vmin, vmax = _resolve_clim(self.data, vmin, vmax,
                                       auto_vmin=self.data.min())
            span = vmax - vmin
            if span > 0:
                data = np.clip((self.data - vmin) / span, 0, 1)
            else:
                # A constant percept spans no range to stretch, and comes out
                # uniformly black rather than dividing by zero:
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
        # Rescale percept to desired shape. The trailing axes (an RGB axis,
        # if any, and time) have to be spelled out; `resize` will not infer
        # them for a 4-D array:
        data = resize(data, (np.int32(height), np.int32(width),
                             *data.shape[2:]))

        # Record the resolved range so that `load` can put the gray levels
        # back on this scale. A container with metadata of its own always gets
        # it; one without has only its name, which is the caller's to choose
        # and so is rewritten only when the caller asked for a range -- or
        # when the name already claims one, which it must not go on claiming.
        meta_kwargs = _metadata_kwargs(fname, _range_tag(vmin, vmax))
        if (fixed_clim and not meta_kwargs) or _P2P_RANGE_RE.search(
                os.path.basename(fname)) is not None:
            fname = _range_tagged_path(fname, vmin, vmax)
        if self.time is None:
            # No time component, store as an image. imwrite will automatically
            # scale the gray levels:
            imageio.imwrite(fname, img_as_ubyte(data)[..., 0], **meta_kwargs)
        else:
            # With time component, store as a movie. A single-frame percept
            # has no frame rate of its own, but can still be written out:
            if fps is None:
                # A movie file runs at one fixed rate, so there is nothing to
                # write a ragged time axis to; `frame_interval` says so. In
                # milliseconds, whatever the percept counts in: frames per
                # second is a wall-clock rate, not a number in the percept's
                # own unit.
                fps = 1000.0 / frame_interval(self.times(ms), tol=1e-6)
            else:
                # Same display clock as `play`: resampling changes the number
                # of frames, not how long the movie runs.
                timeline = _frame_timeline(self.times(ms), fps=fps)
                data = data[..., timeline.indices]
            # Note, for most codecs, the image dimensions must be divisible by
            # 16 the default for the VIDEO_BLOCK_SIZE is 16. Check if image is
            # divisible, if not have ffmpeg upsize to nearest size and warn
            # user they should correct input image if this is not desired.
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
            # (Y, X[, C], T) -> (T, Y, X[, C]), which is how a writer reads
            # a stack of frames:
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
        # `index=...` reads every format the same way, as a stack of frames,
        # so a three-channel image is never mistaken for three frames:
        frames = iio.imread(fname, index=...)
        if frames.ndim == 4:
            if frames.shape[-1] == 4:
                # As elsewhere in p2p, alpha is blended against black:
                frames = rgba2rgb(frames, background=(0, 0, 0))
            if frames.shape[-1] != 3:
                frames = frames[..., 0]
            elif as_gray:
                frames = rgb2gray(frames)
        if frames.ndim not in (3, 4):
            raise ValueError(f"Expected a 2-D image or a stack of them in "
                             f"'{fname}', not an array of shape "
                             f"{frames.shape}.")
        # Encoded pixels become floating-point data in [0, 1], and the frame
        # axis moves to the back, where a percept keeps it. A color stack
        # (T, Y, X, 3) lands on (Y, X, 3, T) by the same move:
        data = np.moveaxis(img_as_float32(frames), 0, -1)

        # STEP 1: WHEN THE FRAMES HAPPEN
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
                # Frames per second is a wall-clock rate; a percept counts
                # milliseconds:
                time = np.arange(data.shape[-1]) * 1000.0 / fps

        # STEP 2: WHAT THE GRAY LEVELS MEAN
        if data.ndim == 4:
            # An RGB percept is already on the only scale RGB has, so there is
            # nothing to recover and nothing to choose:
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
            # Half a range is no range: the other end cannot be read off the
            # encoded pixels without inventing it, and silently dropping what
            # the caller did pass would be worse than saying so.
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
