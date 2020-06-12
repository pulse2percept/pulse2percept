"""`ImageStimulus`, `BionicVisionLabLogo`"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Subplot

from skimage import img_as_float
from skimage.io import imread
from skimage.color import rgba2rgb, rgb2gray
from skimage.transform import resize as img_resize

from .base import Stimulus


class ImageStimulus(Stimulus):
    """ImageStimulus

    .. versionadded:: 0.7

    A stimulus made from an image, where each pixel gets assigned to an
    electrode, and grayscale values in the range [0, 255] get converted to
    activation values in the range [0, 1].

    .. seealso ::

        *  `Basic Concepts > Electrical Stimuli <topics-stimuli>`
        *  :py:class:`~pulse2percept.stimuli.VideoStimulus`

    Parameters
    ----------
    fname : str
        Path to image file. Supported image types include JPG, PNG, and TIF;
        and are inferred from the file ending. If the file does not have a
        proper file ending, specify the file type via ``format``.
        Use :py:class:`~pulse2percept.stimuli.VideoStimulus` for GIFs.

    format : str
        An image format string supported by imageio, such as 'JPG', 'PNG', or
        'TIFF'. Use if the file type cannot be inferred from ``fname``.
        For a full list of supported formats, see
        https://imageio.readthedocs.io/en/stable/formats.html.

    resize : (height, width) or None
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional, default: None
        Optionally, you can provide your own electrode names. If none are
        given, electrode names will be numbered 0..N.

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional, default: None
        Additional stimulus metadata can be stored in a dictionary.

    compress : bool, optional, default: False
        If True, will remove pixels with 0 grayscale value.

    """
    __slots__ = ('img_shape',)

    def __init__(self, fname, format=None, resize=None, as_gray=False,
                 electrodes=None, metadata=None, compress=False):
        img = imread(fname, format=format)
        # Build the metadata container:
        if metadata is None:
            metadata = {}
        metadata['source'] = fname
        metadata['source_shape'] = img.shape
        # Convert to grayscale if necessary:
        if as_gray:
            if img.shape[-1] == 4:
                # Convert the transparent background to black:
                img = rgba2rgb(img, background=(0, 0, 0))
            img = rgb2gray(img)
        # Resize if necessary:
        if resize is not None:
            img = img_resize(img, resize)
        # Store the original image shape for resizing and color conversion:
        self.img_shape = img.shape
        # Convert to float array in [0, 1] and call the Stimulus constructor:
        super(ImageStimulus, self).__init__(img_as_float(img).ravel(),
                                            time=None, electrodes=electrodes,
                                            metadata=metadata,
                                            compress=compress)

    def resize(self, shape, electrodes=None):
        img = img_resize(self.data.reshape(self.img_shape), shape)
        if electrodes is None:
            electrodes = np.arange(img.size)
        self._stim = {
            'data': img.reshape((-1, 1)),
            'electrodes': electrodes,
            'time': None
        }
        self.img_shape = img.shape
        return self

    def rgb2gray(self, electrodes=None):
        img = rgb2gray(self.data.reshape(self.img_shape))
        if electrodes is None:
            electrodes = np.arange(img.size)
        self._stim = {
            'data': img.reshape((-1, 1)),
            'electrodes': electrodes,
            'time': None
        }
        self.img_shape = img.shape
        return self

    def invert(self):
        self._stim = {
            'data': 1.0 - self.data,
            'electrodes': self.electrodes,
            'time': None
        }
        return self

    def plot(self, kind='pcolor', ax=None, **kwargs):
        """Plot the percept

        Parameters
        ----------
        kind : { 'pcolor' | 'hex' }, optional, default: 'pcolor'
            Kind of plot to draw:
            *  'pcolor': using Matplotlib's ``pcolor``. Additional parameters
               (e.g., ``vmin``, ``vmax``) can be passed as keyword arguments.
            *  'hex': using Matplotlib's ``hexbin``. Additional parameters
               (e.g., ``gridsize``) can be passed as keyword arguments.
        ax : matplotlib.axes.Axes; optional, default: None
            A Matplotlib Axes object. If None, a new Axes object will be
            created.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Returns the axes with the plot on it

        """
        frame = self.data.reshape(self.img_shape)
        print(frame.shape)
        if ax is None:
            if 'figsize' in kwargs:
                figsize = kwargs['figsize']
            else:
                figsize = (12, 8)
                # figsize = np.int32(np.array(self.shape[:2][::-1]) / 15)
                # figsize = np.maximum(figsize, 1)
            _, ax = plt.subplots(figsize=figsize)
        else:
            if not isinstance(ax, Subplot):
                raise TypeError("'ax' must be a Matplotlib axis, not "
                                "%s." % type(ax))

        vmin, vmax = frame.min(), frame.max()
        cmap = kwargs['cmap'] if 'cmap' in kwargs else 'gray'
        xdva = np.arange(frame.shape[1])
        ydva = np.arange(frame.shape[0])
        X, Y = np.meshgrid(xdva, ydva, indexing='xy')
        if kind == 'pcolor':
            # Create a pseudocolor plot. Make sure to pass additional keyword
            # arguments that have not already been extracted:
            other_kwargs = {key: kwargs[key]
                            for key in (kwargs.keys() - ['figsize', 'cmap',
                                                         'vmin', 'vmax'])}
            ax.pcolormesh(X, Y, np.flipud(frame), cmap=cmap, vmin=vmin,
                          vmax=vmax, **other_kwargs)
        elif kind == 'hex':
            # Create a hexbin plot:
            if 'gridsize' in kwargs:
                gridsize = kwargs['gridsize']
            else:
                gridsize = np.min(frame.shape[:2]) // 2
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
            raise ValueError("Unknown plot option '%s'. Choose either 'pcolor'"
                             "or 'hex'." % kind)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(xdva[0], xdva[-1])
        ax.set_xticks(np.linspace(xdva[0], xdva[-1], num=5))
        ax.set_xlabel('x (dva)')
        ax.set_ylim(ydva[0], ydva[-1])
        ax.set_yticks(np.linspace(ydva[0], ydva[-1], num=5))
        ax.set_ylabel('y (dva)')
        return ax
