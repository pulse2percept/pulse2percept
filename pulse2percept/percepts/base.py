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
from ..utils import Data, HTMLAnimation, frame_interval, sample
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


def _media_metadata(fname):
    """Whatever metadata imageio can read off a media file

    Pillow carries the text chunks and comments that
    :py:meth:`~pulse2percept.percepts.Percept.save` writes, but cannot open a
    video; FFMPEG can, and reports the frame rate. Whichever opens the file
    wins, so this yields at most one dictionary per backend that can read it.
    """
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
    """Writer arguments that record ``tag`` in the file's own metadata

    Only for the containers whose writer already has a comment field to put it
    in. A percept saved in any other format can still carry its range in the
    file name, which is the caller's to choose.
    """
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


def _check_clim(vmin, vmax):
    """Reject a brightness range that cannot be mapped onto a color scale"""
    if not np.all(np.isfinite([vmin, vmax])) or vmax < vmin:
        raise ValueError(f"'vmin' ({vmin}) and 'vmax' ({vmax}) must be finite "
                         f"with 'vmin' <= 'vmax'.")


def _resolve_clim(data, vmin, vmax, auto_vmin):
    """Fill omitted display limits in from the whole percept

    Resolving against all of ``data`` rather than against the frames a display
    or export clock happens to sample is what keeps the brightness scale
    independent of ``fps``.
    """
    vmin = auto_vmin if vmin is None else vmin
    vmax = np.max(data) if vmax is None else vmax
    vmin, vmax = float(vmin), float(vmax)
    _check_clim(vmin, vmax)
    return vmin, vmax


class Percept(Data):
    """Visual percept

    A visual percept in space and time (optional). Typically the output of a
    computational model.

    .. versionadded:: 0.6

    Parameters
    ----------
    data : 3D NumPy array
        A NumPy array specifying the percept in (Y, X, T) dimensions
    space : :py:class:`~pulse2percept.topography.Grid2D`, optional
        A grid object specifying the (x,y) coordinates in space
    time : 1D array, optional
        A list of time points, expressed in ``time_unit``. May be given as a
        unitful quantity (e.g. ``[0, 0.01] * s``), which is converted into
        ``time_unit`` rather than changing it.
    metadata : dict, optional
        Additional stimulus metadata can be stored in a dictionary.
    n_gray : int, optional
        The number of gray levels to use. If an integer is given, k-means
        clustering is used to compress the color space of the percept into
        ``n_gray`` bins. If None, no compression is performed.
    noise : float or int, optional
        Adds salt-and-pepper noise to each percept frame. An integer will be
        interpreted as the number of pixels to subject to noise in each frame.
        A float between 0 and 1 will be interpreted as a ratio of pixels to
        subject to noise in each frame.
    time_unit : :py:class:`~pulse2percept.units.Unit`, optional
        The unit ``time`` is stored in. Bare numbers passed as ``time`` are
        assumed to already be expressed in this unit; unitful ones are
        converted into it. A model-created percept records the model's own
        :py:attr:`~pulse2percept.models.BaseModel.time_unit` here, which is
        what lets its time axis cross into another model correctly.

        .. versionadded:: 0.10.0

    Notes
    -----
    Space is indexed the NumPy way, but a number that reaches the time axis is
    a *time*, not a frame number, and is linearly interpolated between the two
    frames on either side of it:

    *  ``percept[0, 1]``: the time series of pixel (0, 1)
    *  ``percept[..., 12.5]``: the frame at t=12.5, interpolated
    *  ``percept[..., 0.02 * s]``: that same frame in another unit
    *  ``percept[..., [0, 10]]``, ``percept[..., 0:50:10]``: several frames
    *  ``percept[..., percept.time < 20]``: the stored frames before t=20

    One time point selects a frame and drops the time axis; a list or slice of
    them keeps it. The ``fps`` of :py:meth:`~pulse2percept.percepts.Percept.play` and
    :py:meth:`~pulse2percept.percepts.Percept.save` is a rendering clock
    rather than an interpolation: it chooses which stored frame to show when,
    and never invents one in between.

    .. versionchanged:: 0.10.0

        The time axis is unit-aware (see ``time_unit`` above). ``data`` is
        not: a percept is perceived brightness in arbitrary units, which is
        model output rather than a physical quantity.

    Examples
    --------
    A time axis given in seconds is stored in the percept's own unit, so
    these two are the same percept:

    >>> import numpy as np
    >>> from pulse2percept.percepts import Percept
    >>> from pulse2percept.units import s
    >>> data = np.zeros((3, 3, 2))
    >>> Percept(data, time=[0.0, 10.0]).time
    array([ 0., 10.])
    >>> Percept(data, time=[0, 0.01] * s).time
    array([ 0., 10.])

    Read the percept halfway between its two frames:

    >>> Percept(np.dstack([np.zeros((3, 3)), np.ones((3, 3))]),
    ...         time=[0.0, 10.0])[0, 0, 5.0]
    0.5

    """

    def __init__(self, data, space=None, time=None, metadata=None, n_gray=None,
                 noise=None, time_unit=ms):
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
            n_gray = int(n_gray)
            if n_gray <= 1:
                raise ValueError(f'"n_gray" must be greater than 1, not '
                                 f'{n_gray}.')
            data = np.asarray(data, dtype=np.float32)
            centroids, labels = kmeans2(data.ravel(), n_gray, minit='points')
            data = centroids[labels].reshape(data.shape)
        # Add salt-and-pepper noise if requested:
        if noise is not None:
            n_pixels = np.prod(data.shape[:2])
            vmin, vmax = data.min(), data.max()
            for t in range(data.shape[2]):
                idx_noise = sample(np.arange(n_pixels), k=noise)
                n_noise = len(idx_noise)
                xi, yi = np.unravel_index(idx_noise[:n_noise//2],
                                          data.shape[:2])
                data[xi, yi, t] = vmin
                xi, yi = np.unravel_index(idx_noise[n_noise//2:n_noise],
                                          data.shape[:2])
                data[xi, yi, t] = vmax
        time = as_value(time, self._time_unit, 'time')
        if time is not None:
            time = np.array([time]).flatten()
        self._internal = {
            'data': data,
            'axes': [('ydva', ydva), ('xdva', xdva), ('time', time)],
            'metadata': metadata
        }

    def __getitem__(self, item):
        """Return percept data, interpolated in time where necessary

        Space is indexed the NumPy way, but -- as in
        :py:class:`~pulse2percept.stimuli.Stimulus` -- a number that reaches
        the time axis is a *time*, not a frame index, and is linearly
        interpolated between the two frames on either side of it. Bare numbers
        are read in the percept's
        :py:attr:`~pulse2percept.percepts.Percept.time_unit`; unitful ones are
        converted into it.

        *  ``percept[0, 1]``: the time series of pixel (0, 1)
        *  ``percept[..., 12.5]``: the frame at t=12.5, interpolated
        *  ``percept[..., 20 * ms]``, ``percept[..., 0.02 * s]``: that same
           frame, asked for in another unit
        *  ``percept[:, :, [0, 10]]``: the frames at t=0 and t=10
        *  ``percept[..., 0:50:10]``: every 10 time units from t=0 to t=50
        *  ``percept[..., percept.time < 20]``: the stored frames before t=20
        *  ``percept[0, 1, 12.5]``: one interpolated pixel, as a scalar

        One time point selects a frame and drops the time axis, the way a
        scalar index does on any other axis; a list or slice of them keeps it.

        With ``time=None`` there is no time axis to name, and indexing is
        ordinary NumPy indexing throughout.

        Returns a NumPy array or a scalar, never a new
        :py:class:`~pulse2percept.percepts.Percept`.

        .. versionadded:: 0.10.0

        """
        # STEP 1: DOES THE INDEX REACH THE TIME AXIS?
        # ``percept[0, 1]`` asks for the time series of a pixel, so only an
        # index that reaches the last axis can be naming a time point:
        space, time = item, None
        if self.time is not None and isinstance(item, tuple) and len(item) > 1:
            head = item[:-1]
            if (any(idx is Ellipsis for idx in head) or
                    len(head) == self.data.ndim - 1):
                space, time = head, item[-1]
        # STEP 2: AVOID CONFUSING TIME POINTS WITH FRAME INDICES
        scalar_time = False
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
            # Convert to float so time is not mistaken for a frame index (a
            # boolean mask selects stored frames and must stay boolean):
            if np.asarray(time).dtype != bool:
                time = np.float64(time)
                scalar_time = time.ndim == 0
        # STEP 3: NUMPY HANDLES MOST INDEXING AND SLICING
        try:
            return self.data[space if time is None else (*space, time)]
        except IndexError:
            # An IndexError must still be thrown unless the index named a time
            # point, in which case NumPy refused a float and we interpolate:
            if time is None:
                raise
        # STEP 4: INTERPOLATE TIME
        frames = self.data[space]
        times = np.array([time], dtype=np.float64).ravel()
        # ``_interp_rows`` works on rows; a percept has one time series per
        # pixel rather than per electrode, so flatten space and put it back:
        data = _interp_rows(times, self.time,
                            frames.reshape((-1, len(self.time))))
        data = data.reshape(frames.shape[:-1] + times.shape)
        if scalar_time:
            # One time point picks a frame out rather than slicing a one-frame
            # stack out, exactly as a scalar index does on any other axis:
            data = data[..., 0]
        if data.ndim == 0:
            return data.item()
        return data

    @property
    def time_unit(self):
        """The unit ``time`` is expressed in

        Milliseconds unless the percept was built with a different
        ``time_unit``. Read-only: the stored numbers mean what they meant when
        they were written down. Ask for another unit with
        :py:meth:`~pulse2percept.percepts.Percept.times`.

        .. versionadded:: 0.10.0

        """
        return self._time_unit

    @property
    def time_quantity(self):
        """The time axis with its unit attached, or None

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
        """
        if axis is not None and not isinstance(axis, str):
            raise TypeError('"axis" must be a string or None.')
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
        """
        if axis is not None and not isinstance(axis, str):
            raise TypeError('"axis" must be a string or None.')
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
            ax.plot(self.time, self.data.squeeze(), linewidth=2, **kwargs)
            ax.set_xlabel(f'time ({self.time_unit})')
            ax.set_ylabel('Perceived brightness (a.u.)')
            return ax

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
        """Animate the percept as HTML with JavaScript

        The percept will be played in an interactive player in IPython or
        Jupyter Notebook.

        Parameters
        ----------
        fps : float or None
            Display sampling rate in Hz. If None, show every percept frame using
            its recorded timing. Otherwise resample onto a regular display clock
            using zero-order hold. Playback duration is preserved to within one
            display frame. May also be given as a unitful frequency.
        repeat : bool, optional
            Whether the animation should repeat when the sequence of frames is
            completed.
        annotate_time : bool, optional
            If True, the time of the frame will be shown as t = X in the title
            of the panel, in the percept's
            :py:attr:`~pulse2percept.percepts.Percept.time_unit`.
        ax : matplotlib.axes.AxesSubplot, optional
            A Matplotlib axes object. If None, will create a new Axes object
        colorbar : {True, False}
            Whether to show the colorbar
        fmt : {'png', 'jpg'}, optional
            The image format used to embed the frames. Prefer 'jpg' only if
            size matters more than pixel-exact frames.

            .. versionadded:: 0.10.0
        vmin, vmax : float, optional
            The brightness range the color scale spans, in the percept's own
            arbitrary units. Omitted limits are resolved from the whole
            percept (0 and its brightest pixel), never from the frames the
            display clock happens to sample, so ``fps`` cannot change how
            bright the percept looks. Pass both to put two percepts on a
            common scale.

            .. versionadded:: 0.10.0

        Returns
        -------
        ani : pulse2percept.utils.HTMLAnimation
            A Matplotlib animation object that will play the percept
            frame-by-frame.

        Notes
        -----
        .. versionchanged:: 0.10.0

            The HTML player is now generated by
            :py:class:`~pulse2percept.utils.HTMLAnimation`, which renders the
            figure once and ships all frames as a single sprite sheet.

            ``fps`` now controls display sampling rather than playback speed, and
            nonuniform time axes are supported. It is a display clock, not an
            interpolation: use ``percept[..., t]`` to read the percept at a
            time point that was not recorded.

            ``vmin`` and ``vmax`` were added.
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
        """Save the percept to an image or video file

        Parameters
        ----------
        fname : str
            The filename to be created, with the file extension indicating the
            file type. Percepts with time=None can be saved as images (e.g.,
            '.jpg', '.png', '.gif'). Multi-frame percepts can be saved as
            movies (e.g., '.mp4', '.avi', '.mov') or '.gif'.
        shape : (height, width) or None, optional
            The desired width x height of the resulting image/video.
            Use (h, None) to use a specified height and automatically infer the
            width from the percept's aspect ratio.
            Analogously, use (None, w) to use a specified width.
            If shape is None, width will be set to 320px and height will be
            inferred accordingly.
        fps : float or None
            Movie frame rate in Hz. If None, use the percept's native rate,
            which requires a homogeneous time axis. Otherwise resample using
            zero-order hold. Movie duration is preserved to within one frame.
            May also be given as a unitful frequency.
        vmin, vmax : float, optional
            The brightness range that is mapped onto the file's gray levels,
            in the percept's own arbitrary units. Values outside it are
            clipped. Omitted limits are resolved from the whole percept (its
            darkest and brightest pixel), never from the frames the export
            clock happens to sample, so ``fps`` cannot change how bright the
            movie looks. Pass both to put two percepts on a common scale;
            leaving both out normalizes this percept alone and warns.

            .. versionadded:: 0.10.0

        Notes
        -----
        *  ``shape`` will be adjusted so that width and height are multiples
            of 16 to ensure compatibility with most codecs and players.
        *  PNG and GIF files record the range in their own metadata, so that
            :py:meth:`~pulse2percept.percepts.Percept.load` can undo the
            scaling. Other containers have nowhere to put it; naming the file
            ``'foo__p2p_vmin=0.0_vmax=20.0.mp4'`` tells ``load`` the same
            thing.

        .. versionchanged:: 0.10.0

            ``fps`` now changes the export sampling rate rather than movie
            duration. ``vmin`` and ``vmax`` were added.

        """
        # This path hands `fps` to imageio rather than to `frame_timeline`, so
        # it is its own boundary too: a frame rate is a frequency, and imageio
        # takes a plain number of hertz.
        fps = as_value(fps, Hz, 'fps')
        if vmin is None and vmax is None:
            warnings.warn("Normalizing the percept to its own brightness "
                          "range, so percepts saved separately do not share a "
                          "scale. Pass 'vmin' and 'vmax' to fix the range.",
                          stacklevel=2)
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
        # Rescale percept to desired shape:
        data = resize(data, (np.int32(height), np.int32(width)))

        # Record the range in the file itself where the container has room
        # for it, so that `load` can put the gray levels back on this scale:
        meta_kwargs = _metadata_kwargs(fname, _range_tag(vmin, vmax))
        if self.time is None:
            # No time component, store as an image. imwrite will automatically
            # scale the gray levels:
            imageio.imwrite(fname, img_as_ubyte(data).squeeze(2),
                            **meta_kwargs)
        else:
            # Throw error if we try to save as a static image
            for ext in ['.jpg','.jpeg','.bmp','.png','.tif','.tiff','.jif','.jfif']:
                if fname.endswith(ext):
                    raise ValueError(f"Cannot save multi-frame percept as a static image: {fname}")
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
                    data = resize(data, (out_h, out_w))
            data = img_as_ubyte(data)
            try:
                imageio.mimwrite(fname, data.transpose((2, 0, 1)),
                                 fps=float(fps), **meta_kwargs)
            except TypeError:
                imageio.mimwrite(fname, data.transpose((2, 0, 1)),
                                 duration=1000/fps, **meta_kwargs)
        logging.getLogger(__name__).info(f'Created {fname}.')

    @classmethod
    def load(cls, fname, space=None, time=None, fps=None, vmin=None,
             vmax=None):
        """Load a percept from an image, GIF, or movie file

        The counterpart to :py:meth:`~pulse2percept.percepts.Percept.save`,
        for the image and video formats imageio can read. A static image
        becomes a (Y, X, 1) percept with ``time=None``; a GIF or movie becomes
        a (Y, X, T) one. Color input is converted to grayscale, because a
        percept is a single perceived brightness per pixel.

        .. versionadded:: 0.10.0

        Parameters
        ----------
        fname : str
            The file to read. The file type is inferred from its extension.
        space : :py:class:`~pulse2percept.topography.Grid2D`, optional
            The (x, y) coordinates the pixels sit at. A media file does not
            record them; without a grid the percept is indexed in pixels.
        time : 1D array, optional
            The time points of the frames, in milliseconds or as a unitful
            quantity. Overrides both ``fps`` and the file's own timing.
        fps : float or None
            The frame rate to read the file at, in Hz, overriding the rate the
            file records. May be given as a unitful frequency. A GIF may hold
            a different duration for every frame, and one that does has no
            single rate to be read at: pass ``time`` for it.
        vmin, vmax : float, optional
            The brightness range the file's gray levels stand for. See the
            notes below.

        Returns
        -------
        percept : :py:class:`~pulse2percept.percepts.Percept`

        Notes
        -----
        A media file holds quantized pixel values, not brightness, so the
        scale the percept was saved on has to be recovered separately.
        ``vmin`` and ``vmax`` are resolved independently, in this order:

        1.  the arguments given here;
        2.  the pulse2percept metadata
            :py:meth:`~pulse2percept.percepts.Percept.save` writes into a PNG
            or GIF;
        3.  a ``'foo__p2p_vmin=0.0_vmax=20.0.png'`` file name.

        With both bounds known, the decoded intensities are mapped back onto
        that range. Otherwise the data is left normalized to [0, 1] and a
        warning says so.

        Recovering the range restores the brightness *scale* that was encoded,
        not the original percept: clipping, quantization to 256 gray levels,
        resizing, and lossy video compression all happened on the way out and
        cannot be undone.

        """
        # `index=...` reads every format the same way, as a stack of frames,
        # so a three-channel image is never mistaken for three frames:
        frames = iio.imread(fname, index=...)
        if frames.ndim == 4:
            if frames.shape[-1] == 4:
                # As elsewhere in p2p, alpha is blended against black:
                frames = rgba2rgb(frames, background=(0, 0, 0))
            if frames.shape[-1] == 3:
                frames = rgb2gray(frames)
            else:
                frames = frames[..., 0]
        if frames.ndim != 3:
            raise ValueError(f"Expected a 2-D image or a stack of them in "
                             f"'{fname}', not an array of shape "
                             f"{frames.shape}.")
        # Encoded pixels become floating-point data in [0, 1], and the frame
        # axis moves to the back, where a percept keeps it:
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
                if fps <= 0:
                    raise ValueError(f"'fps' must be greater than zero, not "
                                     f"{fps}.")
                # Frames per second is a wall-clock rate; a percept counts
                # milliseconds:
                time = np.arange(data.shape[-1]) * 1000.0 / fps

        # STEP 2: WHAT THE GRAY LEVELS MEAN
        file_vmin, file_vmax = _media_range(fname)
        name_vmin, name_vmax = _parse_range_tag(
            os.path.splitext(os.path.basename(os.fspath(fname)))[0])
        if vmin is None:
            vmin = file_vmin if file_vmin is not None else name_vmin
        if vmax is None:
            vmax = file_vmax if file_vmax is not None else name_vmax
        if vmin is None or vmax is None:
            warnings.warn(f"The brightness range of '{fname}' is unknown, so "
                          f"the data is left normalized to [0, 1]. Pass "
                          f"'vmin' and 'vmax' if you know it.", stacklevel=2)
        else:
            _check_clim(vmin, vmax)
            data = vmin + data * (vmax - vmin)
        return cls(data, space=space, time=time,
                   metadata={'source': os.fspath(fname), 'vmin': vmin,
                             'vmax': vmax})
