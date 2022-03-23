"""`VideoStimulus`, `BostonTrain`"""
from os.path import dirname, join
import numpy as np
from math import isclose
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from skimage.color import rgb2gray
from skimage.transform import (resize as vid_resize, rotate as vid_rotate,
                               SimilarityTransform)
from skimage.filters import scharr, sobel, median
from skimage.feature import canny

from skimage import img_as_float32
from imageio import get_reader as video_reader

from .base import Stimulus
from .pulses import BiphasicPulse
from ..utils import center_image, shift_image, scale_image, trim_image, unique
from ..utils.constants import DT


class VideoStimulus(Stimulus):
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
        given, electrode names will be numbered 0..N.

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional, default: None
        Additional stimulus metadata can be stored in a dictionary.

    compress : bool, optional, default: False
        If True, will compress the source data in two ways:
        * Remove electrodes with all-zero activation.
        * Retain only the time points at which the stimulus changes.

    """

    def __init__(self, source, format=None, resize=None, as_gray=False,
                 electrodes=None, time=None, metadata=None, compress=False):
        if metadata is None:
            metadata = {}
        elif type(metadata) != dict:
            metadata = {'user': metadata}
        if isinstance(source, str):
            # Filename provided, read the video:
            reader = video_reader(source, format=format)
            vid = np.array([frame for frame in reader])
            # Move frame index to the last dimension:
            if vid.ndim == 4:
                vid = vid.transpose((1, 2, 3, 0))
            elif vid.ndim == 3:
                vid = vid.transpose((1, 2, 0))
            # Combine video metadata with user-specified metadata:
            meta = reader.get_meta_data()
            if meta is not None:
                metadata.update(meta)
            metadata['source'] = source
            metadata['source_shape'] = vid.shape
            # Infer the time points from the video frame rate:
            time = np.arange(vid.shape[-1]) * 1000.0 / meta['fps']
        elif isinstance(source, VideoStimulus):
            vid = source.data.reshape(source.vid_shape)
            metadata.update(source.metadata)
            if electrodes is None:
                electrodes = source.electrodes
            if time is None:
                time = source.time
        elif isinstance(source, np.ndarray):
            vid = source
        else:
            raise TypeError(f"Source must be a filename, a 3D NumPy array or "
                            f"another VideoStimulus, not {type(source)}.")
        if vid.ndim < 3 or vid.ndim > 4:
            raise ValueError(f"Videos must have 3 or 4 dimensions, not "
                             f"{vid.ndim}.")
        # Convert to grayscale if necessary:
        if as_gray:
            if vid.ndim == 4:
                vid = rgb2gray(vid.transpose((0, 1, 3, 2)))
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
        # Convert to float array in [0, 1] and call the Stimulus constructor:
        vid = img_as_float32(vid)
        super(VideoStimulus, self).__init__(vid.reshape((-1, vid.shape[-1])),
                                            time=time, electrodes=electrodes,
                                            metadata=metadata,
                                            compress=compress)
        self.metadata = metadata
        self.rewind()

    def _pprint_params(self):
        params = super(VideoStimulus, self)._pprint_params()
        params.update({'vid_shape': self.vid_shape})
        return params

    def apply(self, func, *args, **kwargs):
        """Apply a function to each frame of the video

        Parameters
        ----------
        func : function
            The function to apply to each frame in the video. Must accept a 2D
            or 3D image and return an image with the same dimensions
        *args :
            Additional positional arguments passed to the function
        **kwargs :
            Additional keyword arguments passed to the function

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object with the new image
        """
        vid = np.array([func(frame.reshape(self.vid_shape[:-1]), *args,
                             **kwargs)
                        for frame in self])
        # Move first axis (frames) to last:
        vid = np.moveaxis(vid, 0, -1)
        return VideoStimulus(vid, electrodes=self.electrodes, time=self.time,
                             metadata=self.metadata)

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
                             metadata=self.metadata)

    def rgb2gray(self, electrodes=None):
        """Convert the video to grayscale

        Parameters
        ----------
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, electrode names will be numbered 0..N.

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
                             metadata=self.metadata)

    def resize(self, shape, electrodes=None):
        """Resize the video

        Parameters
        ----------
        shape : (rows, cols)
            Shape of each frame in the resized video. If one of the dimensions
            is set to -1, its value will be inferred by keeping a constant
            aspect ratio.
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, electrode names will be numbered 0..N.

            .. note::
               The number of electrode names provided must match the number of
               pixels in the resized video.

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
                         (height, width, *self.vid_shape[2:]))
        return VideoStimulus(vid, electrodes=electrodes, time=self.time,
                             metadata=self.metadata)

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
            given, electrode names will be numbered 0..N.

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
        electrodes = self.electrodes
        if electrodes is not None:
            electrodes = electrodes.reshape(self.vid_shape[:-1])
            electrodes = electrodes[y0:y1, x0:x1, ...].ravel()
        return VideoStimulus(cropped_vid, electrodes=electrodes, time=time,
                             metadata=self.metadata)

    def trim(self, tol=0, electrodes=None):
        """Remove any black border around the video

        .. versionadded:: 0.7

        Parameters
        ----------
        tol : float
            Any pixels with gray levels > tol will be trimmed.
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, electrode names will be numbered 0..N.

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
                             time=self.time)

    def rotate(self, angle, mode='constant'):
        """Rotate each frame of the video

        Parameters
        ----------
        angle : float
            Angle by which to rotate each video frame (degrees).
            Positive: counter-clockwise, negative: clockwise

        Returns
        -------
        stim : `VideoStimulus`
            A copy of the stimulus object containing the rotated video

        """
        data = self.data.reshape(self.vid_shape)
        if len(self.vid_shape) == 3:
            # Grayscale videos can be fed directly into rotate:
            data = vid_rotate(data, angle, mode=mode, resize=False)
            return VideoStimulus(data, electrodes=self.electrodes,
                                 metadata=self.metadata, time=self.time)
        # Else need to feed in each frame individually:
        return self.apply(vid_rotate, angle, mode=mode, resize=False)

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

    def encode(self, amp_range=(0, 50), pulse=None):
        """Encode image using amplitude modulation

        Encodes the image as a series of pulses, where the gray levels of the
        image are interpreted as the amplitude of a pulse with values in
        ``amp_range``.

        By default, a single biphasic pulse is used for each pixel, with 0.46ms
        phase duration, and 500ms total stimulus duration.

        Parameters
        ----------
        amp_range : (min_amp, max_amp)
            Range of amplitude values to use for the encoding. The image's
            gray levels will be scaled such that the smallest value is mapped
            onto ``min_amp`` and the largest onto ``max_amp``.
        pulse : :py:class:`~pulse2percept.stimuli.Stimulus`, optional
            A valid pulse or pulse train to be used for the encoding.
            If None given, a :py:class:`~pulse2percept.stimuli.BiphasicPulse`
            (0.46 ms phase duration, stimulus duration inferred from the movie
            frame rate) will be used.

        Returns
        -------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus`
            Encoded stimulus

        """
        # Set frame rate, either from metadata or inferred from stim.time:
        try:
            frame_dur = 1000.0 / self.metadata['fps']
        except KeyError:
            t_diff = unique(np.diff(self.time))
            if len(t_diff) > 1:
                raise NotImplementedError
            frame_dur = 1000.0 / t_diff[0]
        # If no user-defined pulse is given, use a standard one:
        if pulse is None:
            pulse = BiphasicPulse(1, 0.46, stim_dur=frame_dur-DT)
        else:
            if not isinstance(pulse, Stimulus):
                raise TypeError('"pulse" must be a Stimulus object.')
            if pulse.time is None:
                raise ValueError('"pulse" must have a time component.')
            if pulse.time[-1] > frame_dur:
                raise ValueError(f"'pulse.time[-1]={pulse.time[-1]}' exceeds "
                                 f"frame_dur={frame_dur}.")
            if isclose(pulse.time[-1], frame_dur):
                pulse.time[-1] -= DT
        # Scale the pixel values to the desired amplitude range:
        vid_data = self.data.copy() - self.data.min()
        if not isclose(np.abs(vid_data).max(), 0):
            vid_data /= np.abs(vid_data).max()
        vid_data = vid_data * (amp_range[1] - amp_range[0]) + amp_range[0]
        # Normalize pulse data:
        enc_data = pulse.data.copy()
        if not isclose(np.abs(enc_data).max(), 0):
            enc_data /= np.abs(enc_data).max()
        # Amplitude encoding: Every pixel value serves as an amplitude, so must
        # be multiplied with the pulse data. This is easily done with an outer
        # product, then we just need to concatenate all frames in time:
        vid_data = np.outer(vid_data.ravel(),
                            enc_data).reshape((self.shape[0], -1))
        # The time points are given by the pulse data, we just need to shift:
        vid_time = np.array([pulse.time + s * frame_dur
                             for s in range(self.shape[1])]).flatten()
        # Cosmetics: Make sure resulting stimulus has the desired length up to
        # DT precision:
        if not isclose(vid_time[-1], self.shape[1] * frame_dur):
            vid_time[-1] = self.shape[1] * frame_dur
        return Stimulus(vid_data, electrodes=self.electrodes, time=vid_time)

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

    def play(self, fps=None, repeat=True, annotate_time=True, ax=None):
        """Animate the percept as HTML with JavaScript

        The percept will be played in an interactive player in IPython or
        Jupyter Notebook.

        Parameters
        ----------
        fps : float or None
            If None, uses the percept's time axis. Not supported for
            non-homogeneous time axis.
        repeat : bool, optional
            Whether the animation should repeat when the sequence of frames is
            completed.
        annotate_time : bool, optional
            If True, the time of the frame will be shown as t = X ms in the
            title of the panel.
        ax : matplotlib.axes.AxesSubplot, optional
            A Matplotlib axes object. If None, will create a new Axes object

        Returns
        -------
        ani : matplotlib.animation.FuncAnimation
            A Matplotlib animation object that will play the percept
            frame-by-frame.
        """
        def update(data):
            if annotate_time:
                mat.axes.set_title(f't = {self.time[self._next_frame - 1]} ms')
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
                        vmax=self.data.max())
        cbar = fig.colorbar(mat)
        cbar.ax.set_ylabel('Brightness (a.u.)', rotation=-90, va='center')
        plt.close(fig)
        if fps is None:
            interval = unique(np.diff(self.time))
            if len(interval) > 1:
                raise NotImplementedError
            interval = interval[0]
        else:
            interval = 1000.0 / fps
        # Create the animation:
        return FuncAnimation(fig, update, data_gen, interval=interval,
                             save_count=len(self.time), repeat=repeat)


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
        given, electrode names will be numbered 0..N.

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) video frame.

    as_gray : bool, optional
        Flag whether to convert the image to grayscale.
        A four-channel image is interpreted as RGBA (e.g., a PNG), and the
        alpha channel will be blended with the color black.

    metadata : dict, optional, default: None
        Additional stimulus metadata can be stored in a dictionary.

    """

    def __init__(self, resize=None, electrodes=None, as_gray=False,
                 metadata=None):
        # Load logo from data dir:
        module_path = dirname(__file__)
        source = join(module_path, 'data', 'boston-train.mp4')
        # Call VideoStimulus constructor:
        super(BostonTrain, self).__init__(source, format="MP4",
                                          resize=resize,
                                          as_gray=as_gray,
                                          electrodes=electrodes,
                                          metadata=metadata,
                                          compress=False)
