"""`VideoStimulus`, `BostonTrain`"""
from os.path import dirname, join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from skimage.color import rgb2gray
from skimage.transform import resize as vid_resize
from skimage.filters import scharr, sobel, median
from skimage.feature import canny

from skimage import img_as_float32
from imageio import get_reader as video_reader

from .base import Stimulus


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
            time = np.arange(vid.shape[-1]) * meta['fps']
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
            raise TypeError("Source must be a filename, a 3D NumPy array or "
                            "another VideoStimulus, not %s." % type(source))
        if vid.ndim < 3 or vid.ndim > 4:
            raise ValueError("Videos must have 3 or 4 dimensions, not "
                             "%d." % vid.ndim)
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
        self.rewind()

    def _pprint_params(self):
        params = super(VideoStimulus, self)._pprint_params()
        params.update({'vid_shape': self.vid_shape})
        return params

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
            raise TypeError("'filt' must be a string, not %s." % type(filt))
        if len(self.vid_shape) == 4:
            raise ValueError('Cannot apply filter to RGB video. Convert to '
                             'grayscale first.')
        filters = {'sobel': sobel, 'scharr': scharr, 'canny': canny,
                   'median': median}
        try:
            filt = filters[filt.lower()]
        except KeyError:
            raise ValueError("Unknown filter '%s'." % filt)
        vid = np.array([filt(frame.reshape(self.vid_shape[:-1]), **kwargs)
                        for frame in self]).transpose((1, 2, 0))
        return VideoStimulus(vid, electrodes=self.electrodes, time=self.time,
                             metadata=self.metadata)

    def apply(self, func, **kwargs):
        """Apply a function to each frame of the video

        Parameters
        ----------
        func : function
            The function to apply to each frame in the video. Must accept a 2D
            or 3D image and return an image with the same dimensions
        **kwargs :
            Additional parameters passed to the function

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object with the new image
        """
        vid = np.array([func(frame.reshape(self.vid_shape[:-1]), **kwargs)
                        for frame in self]).transpose((1, 2, 0))
        return VideoStimulus(vid, electrodes=self.electrodes, time=self.time,
                             metadata=self.metadata)

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

    def _get_interval(self):
        # Determine the frame rate from the time axis. Problem is that
        # np.unique doesn't work well with floats, so we need to specify a
        # tolerance `TOL`:
        interval = np.diff(self.time)
        TOL = interval.min()
        # Two time points are the same if they are within `TOL` from each
        # other:
        interval = np.unique(np.floor(interval / TOL).astype(int)) * TOL
        return interval

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
                mat.axes.set_title('t = %d ms' %
                                   self.time[self._next_frame - 1])
            mat.set_data(data.reshape(self.vid_shape[:-1]))
            return mat

        def data_gen():
            try:
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
            interval = self._get_interval()
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
