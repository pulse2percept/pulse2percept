""":py:class:`~pulse2percept.stimuli.VideoStimulus`, 
   :py:class:`~pulse2percept.stimuli.BostonTrain`, 
   :py:class:`~pulse2percept.stimuli.GirlPool`"""
from os.path import dirname, join
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage.transform import resize as vid_resize, rotate as vid_rotate
from skimage.filters import scharr, sobel, median
from skimage.feature import canny

from skimage import img_as_float32
from imageio import get_reader as video_reader

from .base import Stimulus, _adoptable
from ._geometry import HasFieldOfView, resolve_fov
from ..units import as_value, dimensionless, ms
from .names import ElectrodeNames
from ..utils import (center_image, shift_image, scale_image, trim_image,
                     frame_interval, HTMLAnimation)
from ..utils.images import _as_writable
from ..utils.constants import MS_PER_S


#: Anything this close to a frame boundary is treated as being on it
_FRAME_TOL = 1e-6


def _read_video(source, format, start_time, stop_time):
    """Decode the frames of a video file that start in [start, stop) ms"""
    start_time = as_value(start_time, ms, 'start_time')
    stop_time = as_value(stop_time, ms, 'stop_time')
    clipped = start_time is not None or stop_time is not None
    for name, t in (('start_time', start_time), ('stop_time', stop_time)):
        if t is not None and not np.isfinite(t):
            raise ValueError(f'"{name}" must be a finite time in ms, not {t}.')
    if start_time is not None and start_time < 0:
        raise ValueError(f'"start_time" cannot be negative, but is '
                         f'{start_time} ms.')
    if (start_time is not None and stop_time is not None and
            stop_time <= start_time):
        raise ValueError(f'"stop_time" ({stop_time} ms) must be greater than '
                         f'"start_time" ({start_time} ms).')
    with video_reader(source, format=format) as reader:
        meta = reader.get_meta_data()
        fps = meta.get('fps') if meta is not None else None
        if clipped:
            if not fps:
                raise ValueError(f'"{source}" does not report a frame rate, '
                                 f'so "start_time"/"stop_time" cannot be '
                                 f'mapped onto frames.')
            first = 0 if start_time is None else _frame_index(start_time, fps)
            last = None if stop_time is None else _frame_index(stop_time, fps)
        else:
            first, last = 0, None
        if last is not None and last <= first:
            raise ValueError(f'No video frame starts in [{start_time}, '
                             f'{stop_time}) ms.')
        if first:
            reader.set_image_index(first)
        frames = []
        while last is None or first + len(frames) < last:
            try:
                frames.append(reader.get_next_data())
            except (IndexError, StopIteration, EOFError):
                break  # End of file
    if clipped and not frames:
        raise ValueError(f'No video frame starts in [{start_time}, '
                         f'{stop_time}) ms.')
    return np.array(frames), meta


def _frame_index(t, fps):
    """Index of the first frame that starts at or after ``t`` ms"""
    return int(np.ceil(t * fps / MS_PER_S - _FRAME_TOL))


class VideoStimulus(HasFieldOfView, Stimulus):
    """VideoStimulus

    A stimulus made from a movie file, where each pixel gets assigned to an
    electrode, and grayscale values in the range [0, 255] get assigned to
    activation values in the range [0, 1].

    The frame rate of the movie is used to infer the time points at which to
    stimulate.

    .. seealso ::

        *  `Basic Concepts > Electrical Stimuli <topics-stimuli>`
        *  :py:class:`~pulse2percept.stimuli.ImageStimulus`

    .. versionadded:: 0.7

    Parameters
    ----------
    source : str
        Path to video file. Supported file types include MP4, AVI, MOV, and
        GIF; and are inferred from the file ending. If the file does not have
        a proper file ending, specify the file type via ``format``.

        Alternatively, pass a <rows x columns x channels x frames> NumPy array
        or another :py:class:`~pulse2percept.stimuli.VideoStimulus` object.

    format : str
        A video format string supported by imageio, such as 'MP4', 'AVI', or
        'MOV'. Use if the file type cannot be inferred from ``source``.
        For a full list of supported formats, see
        https://imageio.readthedocs.io/en/stable/formats.html.

    resize : (height, width) or None, optional, default: None
        A tuple specifying the desired height and the width of each video frame

    as_gray : bool, optional
        Flag whether to convert the image to grayscale.
        A four-channel image is interpreted as RGBA (e.g., a PNG), and the
        alpha channel will be blended with the color black.

    electrodes : int, string or list thereof; optional, default: None
        Optionally, you can provide your own electrode names. If none are
        given, each pixel is named after its place in the image: a letter for
        the row, a number for the column, and a suffix for the color channel
        (e.g. 'A1', 'C12', 'A1_R'). See
        :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional, default: None
        Additional stimulus metadata can be stored in a dictionary.

    compress : bool, optional, default: False
        If True, will compress the source data in two ways:
        * Remove electrodes with all-zero activation.
        * Retain only the time points at which the stimulus changes.

    start_time, stop_time : float or Quantity, optional, default: None
        Load only the frames that start in the half-open interval
        ``[start_time, stop_time)`` of the source video, in milliseconds.
        Time-based clipping requires the video reader to report a frame rate.

        .. note::
           The clip starts at ``time[0] == 0`` no matter where it was cut
           from. To shorten a video that is already in memory, and keep its
           original time stamps, use
           :py:meth:`~pulse2percept.stimuli.VideoStimulus.crop` instead.

        .. versionadded:: 0.10.0

    fov : float, (width, height), or None; optional
        The field of view each frame subtends, in degrees of visual angle
        (e.g., ``30 * dva``). A scalar gives the horizontal FOV, and the
        vertical one follows from the frame's aspect ratio. The FOV is the
        outer extent of the frame, centered on it; pixel coordinates address
        pixel centers, half an angular pixel inside that extent, and row 0 lies
        at positive ``y``. See
        :py:meth:`~pulse2percept.stimuli.VideoStimulus.pixel_to_dva`.

        Without a FOV, the video's pixels have no visual-field geometry.

        .. versionadded:: 0.11.0

    """
    __slots__ = ('vid_shape', '_next_frame', '_fov')

    #: Pixel intensities are gray levels in [0, 1], not currents; see
    #: :py:class:`~pulse2percept.stimuli.ImageStimulus`.
    _default_unit = dimensionless

    def __init__(self, source, format=None, resize=None, as_gray=False,
                 electrodes=None, time=None, metadata=None, compress=False,
                 start_time=None, stop_time=None, fov=None):
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            metadata = {'user': metadata}
        # The buffer the caller still holds, if any (see below):
        borrowed = None
        # The video whose pixel names this one may inherit; decided once the
        # final frame shape is known, because `as_gray` and `resize` build a
        # different grid:
        parent = None
        if isinstance(source, str):
            vid, meta = _read_video(source, format, start_time, stop_time)
            # Move frame index to the last dimension:
            if vid.ndim == 4:
                vid = np.ascontiguousarray(vid.transpose((1, 2, 3, 0)))
            elif vid.ndim == 3:
                vid = np.ascontiguousarray(vid.transpose((1, 2, 0)))
            # Combine video metadata with user-specified metadata:
            if meta is not None:
                metadata.update(meta)
            metadata['source'] = source
            metadata['source_shape'] = vid.shape
            # Infer the time points from the video frame rate:
            time = np.arange(vid.shape[-1]) * MS_PER_S / meta['fps']
        elif isinstance(source, VideoStimulus):
            vid = source.data.reshape(source.vid_shape)
            borrowed = source.data
            metadata.update(source.metadata)
            parent = source
            if time is None:
                time = source.time
            if fov is None:
                # A resize keeps the angular extent of a frame, so the FOV
                # survives every way of building one video from another:
                fov = source.fov
        elif isinstance(source, np.ndarray):
            vid = source
            borrowed = source
            if time is None and 'fps' in metadata:
                # Infer the time points from the video frame rate:
                time = np.arange(vid.shape[-1]) * MS_PER_S / metadata['fps']
        else:
            raise TypeError(f"Source must be a filename, a 3D NumPy array or "
                            f"another VideoStimulus, not {type(source)}.")
        if not isinstance(source, str) and (start_time is not None or
                                            stop_time is not None):
            raise ValueError('"start_time"/"stop_time" only apply to a video '
                             'read from a file. Use crop(idx_time=...) to '
                             'shorten an array or another VideoStimulus.')
        if vid.ndim < 3 or vid.ndim > 4:
            raise ValueError(f"Videos must have 3 or 4 dimensions, not "
                             f"{vid.ndim}.")
        # Convert to grayscale if necessary:
        if as_gray:
            if vid.ndim == 4:
                vid = rgb2gray(vid.transpose((0, 1, 3, 2)))
        # Convert to float array in [0, 1] and call the Stimulus constructor:
        vid = img_as_float32(vid)
        # Resize if necessary:
        if resize is not None:
            height, width = resize
            if height < 0 and width < 0:
                raise ValueError('"height" and "width" cannot both be -1.')
            if height < 0:
                height = int(vid.shape[0] * width / vid.shape[1])
            if width < 0:
                width = int(vid.shape[1] * height / vid.shape[0])
            vid = vid_resize(vid, (height, width, *vid.shape[2:]))
        # Store the original image shape for resizing and color conversion:
        self.vid_shape = vid.shape
        self._fov = resolve_fov(fov, vid.shape[0], vid.shape[1])
        if electrodes is None:
            grid = self.vid_shape[:-1]
            if parent is not None and parent.vid_shape[:-1] == grid:
                # A pixel keeps its name only for as long as it is the same
                # pixel:
                electrodes = parent.electrodes
            else:
                # One electrode per pixel, named after its place in the frame
                # ('A1', 'C12', 'A1_R' for a color video). The last axis holds
                # the frames, which are the time component and not electrodes:
                electrodes = ElectrodeNames(grid)
        if borrowed is not None and np.may_share_memory(vid, borrowed):
            vid = vid.copy()
        super().__init__(_adoptable(vid.reshape((-1, vid.shape[-1]))),
                                            time=time, electrodes=electrodes,
                                            metadata=metadata,
                                            compress=compress)
        self.metadata = metadata
        self.rewind()

    def compress(self):
        """Compress the source data

        Also brings ``vid_shape`` back in line with the compressed data:
        compression drops the time points at which the video does not change,
        so the frame count of the source is no longer the frame count of the
        stimulus. Every ``data.reshape(vid_shape)`` in this module relies on
        that invariant. (Compression can also drop all-zero pixels, in which
        case no shape describes the data any more; see ``_frames``.)

        Returns
        -------
        compressed : :py:class:`~pulse2percept.stimuli.VideoStimulus`
        """
        super().compress()
        # ``Stimulus.__init__`` calls this method for ``compress=True``, which
        # is why ``vid_shape`` is set before the constructor runs: one
        # implementation then covers both that and an explicit ``compress()``.
        self.vid_shape = (*self.vid_shape[:-1], self.data.shape[-1])

    def _frames(self):
        """The stimulus as a dense <rows x columns [x channels] x frames> array

        Raises a ``ValueError`` if the video has been compressed in space,
        which removes all-zero pixels and therefore leaves nothing that can be
        reshaped back into a frame.
        """
        n_px = int(np.prod(self.vid_shape[:-1]))
        if self.data.shape[0] != n_px:
            raise ValueError(
                f"This video was compressed in space: {self.data.shape[0]} of "
                f"its {n_px} pixels are left, so its frames cannot be "
                f"reconstructed. Pass 'compress=False' to keep the video "
                f"dense.")
        return self.data.reshape(self.vid_shape)

    def _pprint_params(self):
        params = super()._pprint_params()
        params.update({'vid_shape': self.vid_shape, 'fov': self.fov})
        return params

    @property
    def _frame_shape(self):
        return self.vid_shape[:2]

    def _names_for(self, vid, electrodes):
        """Electrode names for a video derived from this one

        A pixel keeps its name across an operation that leaves the pixel grid
        alone, which is what makes 'A1' refer to the same thing before and
        after. An operation that resamples the grid (a resize, a rotation that
        grows the canvas) has no such correspondence to preserve, so the result
        is named afresh rather than inheriting names that no longer describe
        it. Only the frame layout is compared; the number of frames is the time
        axis, not an electrode count.
        """
        if electrodes is not None:
            return electrodes
        same = np.shape(vid)[:-1] == self.vid_shape[:-1]
        return self.electrodes if same else None

    def apply(self, func, *args, electrodes=None, **kwargs):
        """Apply a function to each frame of the video

        .. versionchanged:: 0.10.0

            ``func`` may now change the shape of a frame, and ``electrodes``
            can name the result.

        Parameters
        ----------
        func : function
            The function to apply to each frame in the video. Must accept a 2D
            or 3D image and return a 2D or 3D image. The returned frames need
            not have the same shape as the originals (but must all have the
            same shape as each other); see ``electrodes``.
        *args :
            Additional positional arguments passed to the function
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, the original names are carried over whenever ``func`` leaves
            the shape of a frame alone, and the result is named after its place
            in the new frame otherwise (e.g. for
            ``skimage.transform.resize``). See
            :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

            .. note::
               The number of electrode names provided must match the number of
               pixels in a returned frame.
        **kwargs :
            Additional keyword arguments passed to the function

        Returns
        -------
        stim : `VideoStimulus`
            A copy of the stimulus object with the new video

        Notes
        -----
        *  ``func`` can reshape a frame in any way it likes, so a result
           whose pixel grid differs from the original is given no field of
           view rather than an unverifiable one. If you know it, pass the
           result to ``VideoStimulus(..., fov=...)``.
        """
        # `func` gets a frame of its own: several of the scikit-image
        # transforms this exists to reach cannot take a read-only one.
        shape = self.vid_shape[:-1]
        vid = np.array([func(_as_writable(frame.reshape(shape)),
                             *args, **kwargs)
                        for frame in self])
        # Move first axis (frames) to last:
        vid = np.moveaxis(vid, 0, -1)
        fov = self.fov if np.shape(vid)[:2] == self._frame_shape else None
        return VideoStimulus(vid, electrodes=self._names_for(vid, electrodes),
                             time=self.time, metadata=self.metadata, fov=fov)

    def invert(self):
        """Invert the gray levels of the video

        Returns
        -------
        stim : `VideoStimulus`
            A copy of the stimulus object with all grayscale values inverted
            in the range [0, 1].

        """
        return VideoStimulus(1.0 - self.data.reshape(self.vid_shape),
                             electrodes=self.electrodes, time=self.time,
                             metadata=self.metadata, fov=self.fov)

    def rgb2gray(self, electrodes=None):
        """Convert the video to grayscale

        Parameters
        ----------
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, each pixel is named after its place in the image (e.g.
            'A1', 'C12', 'A1_R'). See
            :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

            .. note::
               The number of electrode names provided must match the number of
               pixels in the grayscale video.

        Returns
        -------
        stim : `VideoStimulus`
            A copy of the stimulus object with all RGB values converted to
            grayscale in the range [0, 1].

        """
        vid = self.data.reshape(self.vid_shape)
        if len(self.vid_shape) == 4:
            vid = rgb2gray(vid.transpose((0, 1, 3, 2)))
        return VideoStimulus(vid, electrodes=electrodes, time=self.time,
                             metadata=self.metadata, fov=self.fov)

    def resize(self, shape, electrodes=None, **kwargs):
        """Resize the video

        .. versionchanged:: 0.10.0

            Keyword arguments are passed on to scikit-image.

        .. _skimage.transform.resize: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize

        Parameters
        ----------
        shape : (rows, cols)
            Shape of each frame in the resized video. If one of the dimensions
            is set to -1, its value will be inferred by keeping a constant
            aspect ratio.
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, each pixel is named after its place in the image (e.g.
            'A1', 'C12', 'A1_R'). See
            :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

            .. note::
               The number of electrode names provided must match the number of
               pixels in the resized video.
        **kwargs :
            Additional keyword arguments passed to `skimage.transform.resize`_,
            such as ``order=0`` for nearest-neighbor interpolation (which keeps
            a binary video binary).

        Returns
        -------
        stim : `VideoStimulus`
            A copy of the stimulus object containing the resized video

        """
        height, width = shape
        if height < 0 and width < 0:
            raise ValueError('"height" and "width" cannot both be -1.')
        if height < 0:
            height = int(self.vid_shape[0] * width / self.vid_shape[1])
        if width < 0:
            width = int(self.vid_shape[1] * height / self.vid_shape[0])
        vid = vid_resize(self.data.reshape(self.vid_shape),
                         (height, width, *self.vid_shape[2:]), **kwargs)
        return VideoStimulus(vid, electrodes=electrodes, time=self.time,
                             metadata=self.metadata, fov=self.fov)

    def crop(self, idx_space=None, idx_time=None, left=0, right=0, top=0,
             bottom=0, front=0, back=0, electrodes=None):
        """Crop the video

        This method maps a rectangle (defined by two corners) from each video
        frame to a rectangle of the given size. Similarly, the video can be
        shortened to a specified range of frames.

        Alternatively, this method can be used to crop a number of columns
        either from the left or the right of the video frame, or a number of
        rows either from the top or the bottom, or a number of frames from the
        front (beginning) or back (end) of the video.

        .. versionadded:: 0.8

        Parameters
        ----------
        idx_space : 4-tuple (y0, x0, y1, x1)
            Image indices of the top-left corner ``[y0, x0]`` and bottom-right
            corner ``[y1, x1]`` (exclusive) of the rectangle to crop.
        idx_time : tuple (t0, t1)
            Frame indices defining the start ``t0`` and end ``t1`` of the
            cropped video.
        left : int
            Number of columns to crop from the left of each video frame
        right: int
            Number of columns to crop from the right of each video frame
        top: int
            Number of rows to crop from the top of each video frame
        bottom : int
            Number of rows to crop from the bottom of each video frame
        front : int
            Number of frames to crop from the front (beginning) of the video
        back : int
            Number of frames to crop from the back (end) of the video
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, each pixel is named after its place in the image (e.g.
            'A1', 'C12', 'A1_R'). See
            :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

            .. note::

               The number of electrode names provided must match the number of
               pixels in the cropped image.

        Returns
        -------
        stim : `VideoStimulus`
            A copy of the stimulus object containing the video

        """
        if idx_space is not None:
            if left > 0 or right > 0 or top > 0 or bottom > 0:
                raise ValueError('Crop window "idx_space" cannot be given at '
                                 'the same time as "left"/"right"/"top"/'
                                 '"bottom".')
            # Crop window is given by a rectangle (ignore left, right, etc.):
            try:
                y0, x0, y1, x1 = idx_space
            except (ValueError, TypeError):
                raise TypeError('"idx_space" must be a 4-tuple (y0,x0,y1,x1)')
        else:
            # Crop window not given, use left/right/top/bottom:
            y0, x0 = top, left
            y1, x1 = self.vid_shape[0] - bottom, self.vid_shape[1] - right
        if idx_time is not None:
            if front > 0 or back > 0:
                raise ValueError('Crop window "idx_time" cannot be given at '
                                 'the same times as "front"/"back".')
            try:
                t0, t1 = idx_time
            except (ValueError, TypeError):
                raise TypeError('"idx_time" must be a tuple (t0, t1).')
        else:
            t0, t1 = front, self.vid_shape[-1] - back
        # Safety checks:
        if y1 <= y0 or x1 <= x0:
            raise ValueError(f"The corners do not define a valid rectangle:"
                             f"(y0,x0)=({y0},{x0}), (y1,x1)=({y1},{x1}).")
        if y0 < 0 or x0 < 0:
            raise ValueError(f"Top-left corner (y0,x0)=({y0},{x0}) lies "
                             f"outside the video frame.")
        if y1 >= self.vid_shape[0] or x1 >= self.vid_shape[1]:
            raise ValueError(f"Bottom-right corner (y1,x1)=({y1},{x1}) lies "
                             f"outside the video frame.")
        if t1 <= t0:
            raise ValueError(f"Start and stop frame do not form a valid range: "
                             f"t0={t0}, t1={t1}.")
        if t0 < 0 or t1 > self.vid_shape[-1]:
            raise ValueError(f"Start/stop frames lie outside the valid range: "
                             f"t0={t0}, t1={t1}")
        # Crop the video:
        vid = self.data.reshape(self.vid_shape)
        cropped_vid = vid[y0:y1, x0:x1, ..., t0:t1]  # could be RGB or gray
        time = self.time[t0:t1]
        if electrodes is None:
            # Carry the cropped pixels' original names over, so that a pixel
            # keeps the same name before and after cropping:
            electrodes = self.electrodes.reshape(self.vid_shape[:-1])
            electrodes = electrodes[y0:y1, x0:x1, ...].ravel()
        return VideoStimulus(cropped_vid, electrodes=electrodes, time=time,
                             metadata=self.metadata,
                             fov=self._fov_for_shape(cropped_vid.shape))

    def trim(self, tol=0, electrodes=None):
        """Remove any black border around the video

        .. versionadded:: 0.7

        Parameters
        ----------
        tol : float
            Any pixels with gray levels > tol will be trimmed.
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, each pixel is named after its place in the image (e.g.
            'A1', 'C12', 'A1_R'). See
            :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

            .. note::
               The number of electrode names provided must match the number of
               pixels in each frame of the trimmed video.

        Returns
        -------
        stim : `VideoStimulus`
            A copy of the stimulus object with trimmed borders.

        """
        vid = self.data.reshape(self.vid_shape)
        # First we trim each frame individually and record the start and stop
        # indices for rows and columns:
        rows, cols = [], []
        for i in range(vid.shape[-1]):
            _, r, c = trim_image(vid[..., i], return_coords=True)
            rows.append(r)
            cols.append(c)
        rows, cols = np.array(rows), np.array(cols)
        # Then we
        col_start, col_end = cols[:, 0].min(), cols[:, 1].max()
        row_start, row_end = rows[:, 0].min(), rows[:, 1].max()
        vid = vid[row_start:row_end, col_start:col_end, ...]
        return VideoStimulus(vid, electrodes=electrodes, metadata=self.metadata,
                             time=self.time,
                             fov=self._fov_for_shape(vid.shape))

    def rotate(self, angle, mode='constant', electrodes=None, **kwargs):
        """Rotate each frame of the video

        .. versionchanged:: 0.10.0

            Keyword arguments are passed on to scikit-image.

        .. _skimage.transform.rotate: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.rotate

        Parameters
        ----------
        angle : float
            Angle by which to rotate each video frame (degrees).
            Positive: counter-clockwise, negative: clockwise
        mode : str, optional
            How to fill in the corners the rotation leaves empty; see
            `skimage.transform.rotate`_.
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, each pixel keeps the name it had before the rotation, unless
            ``resize=True`` grew the frame, in which case the enlarged video is
            named after its own pixel grid. See
            :py:class:`~pulse2percept.stimuli.ElectrodeNames`.
        **kwargs :
            Additional keyword arguments passed to `skimage.transform.rotate`_,
            such as ``order``, ``cval``, or ``resize=True`` to grow each frame
            so that it contains every rotated pixel.

        Returns
        -------
        stim : `VideoStimulus`
            A copy of the stimulus object containing the rotated video

        """
        # Rotating in place is the common case, and keeps the pixel names
        # meaningful; ``resize=True`` is available through kwargs:
        kwargs.setdefault('resize', False)
        data = self.data.reshape(self.vid_shape)
        if len(self.vid_shape) == 3:
            # A grayscale video can be fed to `rotate` in one go, with its
            # frames standing in for the color channels it expects:
            data = vid_rotate(_as_writable(data), angle, mode=mode,
                              **kwargs)
            # A rotation resamples a frame but not its angular pixel size, so a
            # grown canvas (``resize=True``) subtends a larger FOV:
            return VideoStimulus(data,
                                 electrodes=self._names_for(data, electrodes),
                                 metadata=self.metadata, time=self.time,
                                 fov=self._fov_for_shape(data.shape))
        # Else need to feed in each frame individually:
        return self.apply(vid_rotate, angle, mode=mode, electrodes=electrodes,
                          **kwargs)

    def shift(self, shift_cols, shift_rows):
        """Shift the image foreground

        This function shifts the center of mass (CoM) of the image by the
        specified number of rows and columns.

        Parameters
        ----------
        shift_cols : float
            Number of columns by which to shift the CoM.
            Positive: to the right, negative: to the left
        shift_rows : float
            Number of rows by which to shift the CoM.
            Positive: downward, negative: upward

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object containing the shifted image

        """
        return self.apply(shift_image, shift_cols, shift_rows)

    def center(self, loc=None):
        """Center the image foreground

        This function shifts the center of mass (CoM) to the image center.

        Parameters
        ----------
        loc : (col, row), optional
            The pixel location at which to center the CoM. By default, shifts
            the CoM to the image center.

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object containing the centered image

        """
        return self.apply(center_image, loc=loc)

    def scale(self, scaling_factor):
        """Scale the image foreground

        This function scales the image foreground (excluding black pixels)
        by a factor.

        Parameters
        ----------
        scaling_factor : float
            Factory by which to scale the image

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object containing the scaled image

        """
        return self.apply(scale_image, scaling_factor)

    def filter(self, filt, **kwargs):
        """Filter each frame of the video

        Parameters
        ----------
        filt : str
            Image filter that will be applied to every frame of the video.
            Additional parameters can be passed as keyword arguments.
            The following filters are supported:

            *  'sobel': Edge filter the image using the `Sobel filter
               <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel>`_.
            *  'scharr': Edge filter the image using the `Scarr filter
               <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.scharr>`_.
            *  'canny': Edge filter the image using the `Canny algorithm
               <https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.canny>`_.
               You can also specify ``sigma``, ``low_threshold``,
               ``high_threshold``, ``mask``, and ``use_quantiles``.
            *  'median': Return local median of the image.
        **kwargs :
            Additional parameters passed to the filter

        Returns
        -------
        stim : `VideoStimulus`
            A copy of the stimulus object with the filtered image
        """
        if not isinstance(filt, str):
            raise TypeError(f"'filt' must be a string, not {type(filt)}.")
        if len(self.vid_shape) == 4:
            raise ValueError('Cannot apply filter to RGB video. Convert to '
                             'grayscale first.')
        filters = {'sobel': sobel, 'scharr': scharr, 'canny': canny,
                   'median': median}
        try:
            filt = filters[filt.lower()]
        except KeyError:
            raise ValueError(f"Unknown filter '{filt}'.")
        return self.apply(filt, **kwargs)

    def encode(self, amp_range=(0, 50), freq=20, implant=None, **kwargs):
        """Encode the video using amplitude modulation

        Encodes every frame of the video as a train of biphasic pulses, where
        the gray level of a pixel sets the amplitude of its pulses. Each train
        lasts one frame period.

        This is a shorthand for
        :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`; use that directly
        for the full set of options.

        .. versionchanged:: 0.10.0

            Gray levels now map onto ``amp_range`` absolutely rather than being
            stretched to fill it (pass ``stretch=True`` for the old behavior),
            each frame receives a pulse *train* rather than a single pulse, and
            ``implant`` encodes at electrode rather than pixel resolution.

        Parameters
        ----------
        amp_range : (min_amp, max_amp), optional
            Range of pulse amplitudes (uA). A gray level of 0 maps onto
            ``min_amp``, a gray level of 1 onto ``max_amp``.
        freq : float, optional
            Pulse train frequency (Hz).
        implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`, optional
            If given, the video is first sampled at the implant's electrode
            locations, so that the pulse trains are built at electrode rather
            than pixel resolution. Strongly recommended: a video has orders of
            magnitude more pixels than an implant has electrodes.
        **kwargs :
            Additional arguments passed to
            :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`.

        Returns
        -------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus`
            Encoded stimulus

        """
        # Imported here because `encoders` imports this module:
        from .encoders import AmplitudeEncoder
        return AmplitudeEncoder(amp_range=amp_range, freq=freq,
                                **kwargs).encode(self, implant=implant)

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

    def rewind(self):
        """Rewind the iterator"""
        self._next_frame = 0

    def play(self, fps=None, repeat=True, annotate_time=True, ax=None,
             fmt='jpg'):
        """Animate the video as HTML with JavaScript

        The video will be played in an interactive player in IPython or
        Jupyter Notebook.

        Parameters
        ----------
        fps : float or None
            If None, uses the video's time axis. Not supported for
            non-homogeneous time axis. May be given as a plain number of hertz
            or as a unitful frequency (e.g. ``30 * Hz``, ``0.03 * kHz``); see
            :py:mod:`pulse2percept.units`.
        repeat : bool, optional
            Whether the animation should repeat when the sequence of frames is
            completed.
        annotate_time : bool, optional
            If True, the time of the frame will be shown as t = X ms in the
            title of the panel.
        ax : matplotlib.axes.AxesSubplot, optional
            A Matplotlib axes object. If None, will create a new Axes object
        fmt : {'jpg', 'png'}, optional
            The image format used to embed the frames. 'jpg' keeps notebooks
            and doc pages an order of magnitude smaller; use 'png' if you need
            the frames to be pixel-exact.

            .. versionadded:: 0.10.0

        Returns
        -------
        ani : pulse2percept.utils.HTMLAnimation
            A Matplotlib animation object that will play the video
            frame-by-frame.

        Notes
        -----
        .. versionchanged:: 0.10.0

            The HTML player is now generated by
            :py:class:`~pulse2percept.utils.HTMLAnimation`, which renders the
            figure once and ships all frames as a single sprite sheet. This is
            roughly two orders of magnitude faster than Matplotlib's
            ``to_jshtml`` and produces much smaller notebooks and doc pages.
        """
        def update(data):
            if annotate_time:
                mat.axes.set_title(f't = {self.time[self._next_frame - 1]:.2f} ms')
            mat.set_data(data.reshape(self.vid_shape[:-1]))
            return mat

        def data_gen():
            try:
                self.rewind()
                # Advance to the next frame:
                while True:
                    yield next(self)
            except StopIteration:
                # End of the sequence, exit:
                pass

        if self.time is None:
            raise ValueError("Cannot animate a percept with time=None.")
        # Raises if the video was compressed in space, in which case there is
        # no dense frame left to display:
        frames = self._frames()

        # There are several options to animate a percept in Jupyter/IPython
        # (see https://stackoverflow.com/a/46878531). Displaying the animation
        # as HTML with JavaScript is compatible with most browsers and even
        # %matplotlib inline (although it can be kind of slow):
        plt.rcParams["animation.html"] = 'jshtml'
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure
        # Rewind the percept and show an empty frame:
        self.rewind()
        mat = ax.imshow(np.zeros(self.vid_shape[:-1]), cmap='gray',
                        vmin=0, vmax=self.data.max())
        plt.close(fig)
        # Create the animation. The frame data is handed to HTMLAnimation so
        # that it can render the HTML player without going through Matplotlib:
        labels = None
        if annotate_time:
            labels = [f't = {t:.2f} ms' for t in self.time]
        return HTMLAnimation(fig, update, data_gen, repeat=repeat,
                             interval=frame_interval(self.time, fps=fps),
                             save_count=len(self.time), image=mat,
                             labels=labels, fmt=fmt, frame_data=frames)


class BostonTrain(VideoStimulus):
    """Boston Train sequence

    Load the Boston subway sequence, consisting of 94 frames of 240x426x3
    pixels each.

    .. versionadded:: 0.7

    Parameters
    ----------
    resize : (height, width) or None
        A tuple specifying the desired height and the width of the video
        stimulus.

    electrodes : int, string or list thereof; optional, default: None
        Optionally, you can provide your own electrode names. If none are
        given, each pixel is named after its place in the image: a letter for
        the row, a number for the column, and a suffix for the color channel
        (e.g. 'A1', 'C12', 'A1_R'). See
        :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) video frame.

    as_gray : bool, optional
        Flag whether to convert the image to grayscale.
        A four-channel image is interpreted as RGBA (e.g., a PNG), and the
        alpha channel will be blended with the color black.

    metadata : dict, optional, default: None
        Additional stimulus metadata can be stored in a dictionary.

    fov : float, (width, height), or None; optional
        Field of view in degrees of visual angle; see
        :py:class:`~pulse2percept.stimuli.VideoStimulus`.

        .. versionadded:: 0.11.0

    """
    __slots__ = ()

    def __init__(self, resize=None, electrodes=None, as_gray=False,
                 metadata=None, fov=None):
        # Load logo from data dir:
        module_path = dirname(__file__)
        source = join(module_path, 'data', 'boston-train.mp4')
        # Call VideoStimulus constructor:
        super().__init__(source, format="MP4",
                                          resize=resize,
                                          as_gray=as_gray,
                                          electrodes=electrodes,
                                          metadata=metadata,
                                          compress=False,
                                          fov=fov)


class GirlPool(VideoStimulus):
    """A girl jumping into a swimming pool

    Load the "girl jumping in a pool" sequence, consisting of 91 frames of
    240x426x3 pixels each.

    .. versionadded:: 0.9

    Parameters
    ----------
    resize : (height, width) or None
        A tuple specifying the desired height and the width of the video
        stimulus.

    electrodes : int, string or list thereof; optional, default: None
        Optionally, you can provide your own electrode names. If none are
        given, each pixel is named after its place in the image: a letter for
        the row, a number for the column, and a suffix for the color channel
        (e.g. 'A1', 'C12', 'A1_R'). See
        :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) video frame.

    as_gray : bool, optional
        Flag whether to convert the image to grayscale.
        A four-channel image is interpreted as RGBA (e.g., a PNG), and the
        alpha channel will be blended with the color black.

    metadata : dict, optional, default: None
        Additional stimulus metadata can be stored in a dictionary.

    fov : float, (width, height), or None; optional
        Field of view in degrees of visual angle; see
        :py:class:`~pulse2percept.stimuli.VideoStimulus`.

        .. versionadded:: 0.11.0

    """
    __slots__ = ()

    def __init__(self, resize=None, electrodes=None, as_gray=False,
                 metadata=None, fov=None):
        # Load logo from data dir:
        module_path = dirname(__file__)
        source = join(module_path, 'data', 'girl-pool.mp4')
        # Call VideoStimulus constructor:
        super().__init__(source, format="MP4",
                                       resize=resize,
                                       as_gray=as_gray,
                                       electrodes=electrodes,
                                       metadata=metadata,
                                       compress=False,
                                       fov=fov)
