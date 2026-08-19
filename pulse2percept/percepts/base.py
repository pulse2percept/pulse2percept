""":py:class:`~pulse2percept.percepts.Percept`"""
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.axes import Subplot
from math import isclose
from scipy.cluster.vq import kmeans2
import imageio
import logging
from skimage import img_as_ubyte
from skimage.transform import resize

from ..units import DimensionMismatchError, Hz, Quantity, Unit, as_value, ms
from ..utils import Data, HTMLAnimation, frame_interval, sample
from ..utils.animation import _frame_timeline
from ..utils.constants import VIDEO_BLOCK_SIZE


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

    def __get_item__(self, key):
        return self.data[key]

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
             colorbar=True, fmt='png'):
        """Animate the percept as HTML with JavaScript

        The percept will be played in an interactive player in IPython or
        Jupyter Notebook.

        Parameters
        ----------
        fps : float or None
            The rate at which the percept is sampled for display. The percept
            plays for as long as its
            :py:attr:`~pulse2percept.percepts.Percept.time` axis says it
            lasts, whatever ``fps`` is: a three-second percept takes three
            seconds to play at 15, 30, or 60 fps.

            If None, every frame is shown for its own time step, however
            irregular the time axis is. Otherwise the percept is resampled
            onto a regular display clock with a zero-order hold: each display
            frame shows the most recent frame that was due, and brief events
            in between two samples are missed rather than smeared out.

            May be given as a plain number of hertz or as a unitful frequency
            (e.g. ``30 * Hz``, ``0.03 * kHz``); see
            :py:mod:`pulse2percept.units`.
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
            The image format used to embed the frames. A percept is scalar, so
            'png' ships it as a palettized image of one byte per pixel, which
            is already compact; 'jpg' roughly halves that again, but a
            phosphene is exactly the kind of high-contrast blob against black
            that JPEG rings around. Prefer 'jpg' only if size matters more than
            pixel-exact frames.

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
            figure once and ships all frames as a single sprite sheet. This is
            roughly two orders of magnitude faster than Matplotlib's
            ``to_jshtml`` and produces much smaller notebooks and doc pages.

            ``fps`` no longer changes how fast the percept plays; the time
            axis owns that. A non-homogeneous time axis (e.g., the short
            phases and long gaps of a pulse train) is now played back as it
            is, rather than rejected. Browsers do not schedule timeouts more
            finely than a few milliseconds, so a time step shorter than that
            is skipped rather than stretched -- the same rule as a low
            ``fps``, and for the same reason: a display that cannot resolve a
            brief event should miss it, not distort time.

        """
        if self.time is None:
            raise ValueError("Cannot animate a percept with time=None. Use "
                             "percept.plot() instead.")
        # How long a frame stays up is wall-clock time, counted in
        # milliseconds whatever the percept counts in: a percept at
        # [0, 20, 40] ms and one at [0, .02, .04] s play at the same real
        # speed. Only the labels below are in the percept's own unit.
        timeline = _frame_timeline(self.times(ms), fps=fps)
        idx = timeline.indices

        def update(i):
            if annotate_time:
                t = self.time[idx[i]]
                mat.axes.set_title(f't = {t:.2f} {self.time_unit}')
            mat.set_data(self.data[..., idx[i]])
            return mat

        def data_gen():
            # Which frame to show at each tick of the display clock. With
            # `fps=None` that is simply every frame, in order:
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
        mat = ax.imshow(np.zeros_like(self.data[..., 0]), cmap='gray',
                        vmin=0, vmax=self.data.max())
        if colorbar:
            cbar = fig.colorbar(mat)
            cbar.ax.set_ylabel('Phosphene brightness (a.u.)', rotation=-90,
                               va='center')
        plt.close(fig)
        # Create the animation. The frame data is handed to HTMLAnimation so
        # that it can render the HTML player without going through Matplotlib:
        labels = None
        if annotate_time:
            # A held frame keeps the time stamp of the frame it holds: that is
            # when the data it shows was computed.
            labels = [f't = {t:.2f} {self.time_unit}' for t in self.time[idx]]
        return HTMLAnimation(fig, update, data_gen, repeat=repeat,
                             intervals=timeline.intervals,
                             save_count=idx.size, image=mat,
                             frame_data=self.data[..., idx], labels=labels,
                             fmt=fmt)

    def save(self, fname, shape=None, fps=None):
        """Save the percept as an MP4 or GIF

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
            The rate at which the percept is sampled for the movie. The movie
            runs for as long as the percept's
            :py:attr:`~pulse2percept.percepts.Percept.time` axis says it
            lasts, whatever ``fps`` is: a higher rate buys more frames, not a
            longer movie.

            If None, every frame of the percept is written out at the rate its
            time axis implies, which a movie file can only express if that
            axis is homogeneous. Otherwise the percept is resampled onto a
            regular clock with a zero-order hold, exactly as in
            :py:meth:`~pulse2percept.percepts.Percept.play`.

            May be given as a plain number of hertz or as a unitful frequency
            (e.g. ``30 * Hz``, ``0.03 * kHz``); see
            :py:mod:`pulse2percept.units`.

        Notes
        -----
        *  ``shape`` will be adjusted so that width and height are multiples
            of 16 to ensure compatibility with most codecs and players.

        .. versionchanged:: 0.10.0

            ``fps`` no longer changes how long the movie runs; the time axis
            owns that.

        """
        # This path hands `fps` to imageio rather than to `frame_timeline`, so
        # it is its own boundary too: a frame rate is a frequency, and imageio
        # takes a plain number of hertz.
        fps = as_value(fps, Hz, 'fps')
        data = self.data - self.data.min()
        if not isclose(np.max(data), 0):
            data = data / np.max(data)
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

        if self.time is None:
            # No time component, store as an image. imwrite will automatically
            # scale the gray levels:
            imageio.imwrite(fname, img_as_ubyte(data).squeeze(2))
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
                imageio.mimwrite(fname, data.transpose((2, 0, 1)), fps=float(fps))
            except TypeError:
                imageio.mimwrite(fname, data.transpose((2, 0, 1)), duration=1000/fps)
        logging.getLogger(__name__).info(f'Created {fname}.')
