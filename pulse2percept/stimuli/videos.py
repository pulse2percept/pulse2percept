"""`VideoStimulus`"""
from os.path import dirname, join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from skimage.color import rgb2gray
from skimage.transform import resize as img_resize
from skimage import img_as_float
from imageio import get_reader as video_reader

from .base import Stimulus
from ..utils import parfor


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
    fname : str
        Path to video file. Supported file types include MP4, AVI, MOV, and
        GIF; and are inferred from the file ending. If the file does not have
        a proper file ending, specify the file type via ``format``.

    format : str
        An image format string supported by imageio, such as 'JPG', 'PNG', or
        'TIFF'. Use if the file type cannot be inferred from ``fname``.
        For a full list of supported formats, see
        https://imageio.readthedocs.io/en/stable/formats.html.

    resize : (height, width) or None, optional, default: None
        A tuple specifying the desired height and the width of the image
        stimulus.

    anti_aliasing : bool, optional, default: False
        Whether to apply a Gaussian filter to smooth the image prior to
        resizing. It is crucial to filter when down-sampling the image to
        avoid aliasing artifacts.

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

    interp_method : str or int, optional, default: 'linear'
        For SciPy's ``interp1`` method, specifies the kind of interpolation as
        a string ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
        'previous', 'next') or as an integer specifying the order of the spline
        interpolator to use.
        Here, 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline
        interpolation of zeroth, first, second or third order; 'previous' and
        'next' simply return the previous or next value of the point.

    extrapolate : bool, optional, default: False
        Whether to extrapolate data points outside the given range.

    """

    def __init__(self, fname, format=None, resize=None, anti_aliasing=False,
                 electrodes=None, metadata=None, compress=False,
                 interp_method='linear', extrapolate=False):
        # Open the video reader:
        reader = video_reader(fname, format=format)
        # Combine video metadata with user-specified metadata:
        meta = reader.get_meta_data()
        if metadata is not None:
            meta.update(metadata)
        meta['source'] = fname
        # Read the video:
        vid = [frame for frame in reader]
        # Consider downscaling before doing anything else (with anti-aliasing,
        # this can take a while):
        if resize is not None:
            vid = parfor(img_resize, vid, func_args=[resize],
                         func_kwargs={'anti_aliasing': anti_aliasing})
        if vid[0].ndim == 3 and vid[0].shape[-1] == 3:
            vid = parfor(rgb2gray, vid)
        vid = np.array(parfor(img_as_float, vid)).transpose((1, 2, 0))
        self.vid_shape = vid.shape
        # Infer the time points from the video frame rate:
        n_frames = vid.shape[-1]
        time = np.arange(n_frames) * meta['fps']
        # Call the Stimulus constructor:
        super(VideoStimulus, self).__init__(vid.reshape((-1, n_frames)),
                                            time=time, electrodes=electrodes,
                                            metadata=meta, compress=compress,
                                            interp_method=interp_method,
                                            extrapolate=extrapolate)
        self.rewind()

    def _pprint_params(self):
        params = super(VideoStimulus, self)._pprint_params()
        params.update({'vid_shape': self.vid_shape})
        return params

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

    def play(self, fps=None, repeat=True, ax=None):
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
        ax : matplotlib.axes.AxesSubplot, optional
            A Matplotlib axes object. If None, will create a new Axes object

        Returns
        -------
        ani : matplotlib.animation.FuncAnimation
            A Matplotlib animation object that will play the percept
            frame-by-frame.

        """
        def update(data):
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
        cbar.ax.set_ylabel('Phosphene brightness (a.u.)', rotation=-90,
                           va='center')
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
                             repeat=repeat)
