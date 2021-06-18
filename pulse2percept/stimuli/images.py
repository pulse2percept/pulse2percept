"""`ImageStimulus`, `LogoBVL`, `LogoUCSB`, `SnellenChart`"""
from os.path import dirname, join
import numpy as np
from math import isclose
from copy import deepcopy
import matplotlib.pyplot as plt

from skimage import img_as_float32
from skimage.io import imread, imsave
from skimage.color import rgba2rgb, rgb2gray
from skimage.transform import (resize as img_resize, rotate as img_rotate,
                               warp as img_warp, SimilarityTransform)
from skimage.filters import (threshold_mean, threshold_minimum, threshold_otsu,
                             threshold_local, threshold_isodata, scharr, sobel,
                             median)
from skimage.feature import canny

from .base import Stimulus
from .pulses import BiphasicPulse
from ..utils import center_image, shift_image, scale_image, trim_image


class ImageStimulus(Stimulus):
    """ImageStimulus

    A stimulus made from an image, where each pixel gets assigned to an
    electrode, and grayscale values in the range [0, 255] get converted to
    activation values in the range [0, 1].

    .. seealso ::

        *  `Basic Concepts > Electrical Stimuli <topics-stimuli>`
        *  :py:class:`~pulse2percept.stimuli.VideoStimulus`

    .. versionadded:: 0.7

    Parameters
    ----------
    source : str
        Path to image file. Supported image types include JPG, PNG, and TIF;
        and are inferred from the file ending. If the file does not have a
        proper file ending, specify the file type via ``format``.
        Use :py:class:`~pulse2percept.stimuli.VideoStimulus` for GIFs.

    format : str, optional
        An image format string supported by imageio, such as 'JPG', 'PNG', or
        'TIFF'. Use if the file type cannot be inferred from ``source``.
        For a full list of supported formats, see
        https://imageio.readthedocs.io/en/stable/formats.html.

    resize : (height, width) or None, optional
        Shape of the resized image. If one of the dimensions is set to -1,
        its value will be inferred by keeping a constant aspect ratio.

    as_gray : bool, optional
        Flag whether to convert the image to grayscale.
        A four-channel image is interpreted as RGBA (e.g., a PNG), and the
        alpha channel will be blended with the color black.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. If none are
        given, electrode names will be numbered 0..N.

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional
        Additional stimulus metadata can be stored in a dictionary.

    compress : bool, optional
        If True, will remove pixels with 0 grayscale value.

    """
    __slots__ = ('img_shape',)

    def __init__(self, source, format=None, resize=None, as_gray=False,
                 electrodes=None, metadata=None, compress=False):
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            metadata = {'user': metadata}
        if isinstance(source, str):
            # Filename provided:
            img = imread(source, format=format)
            metadata['source'] = source
            metadata['source_shape'] = img.shape
        elif isinstance(source, ImageStimulus):
            img = source.data.reshape(source.img_shape)
            metadata.update(source.metadata)
            if electrodes is None:
                electrodes = source.electrodes
        elif isinstance(source, np.ndarray):
            img = source
        else:
            raise TypeError("Source must be a filename or another "
                            "ImageStimulus, not %s." % type(source))
        if img.ndim < 2 or img.ndim > 3:
            raise ValueError("Images must have 2 or 3 dimensions, not "
                             "%d." % img.ndim)
        # Convert to grayscale if necessary:
        if as_gray:
            if img.ndim == 3 and img.shape[2] == 4:
                # Blend the background with black:
                img = rgba2rgb(img, background=(0, 0, 0))
            if img.ndim == 3:
                img = rgb2gray(img)
        # Resize if necessary:
        if resize is not None:
            height, width = resize
            if height < 0 and width < 0:
                raise ValueError('"height" and "width" cannot both be -1.')
            if height < 0:
                height = int(img.shape[0] * width / img.shape[1])
            if width < 0:
                width = int(img.shape[1] * height / img.shape[0])
            img = img_resize(img, (height, width))
        # Store the original image shape for resizing and color conversion:
        self.img_shape = img.shape
        # Convert to float array in [0, 1] and call the Stimulus constructor:
        super(ImageStimulus, self).__init__(img_as_float32(img).ravel(),
                                            time=None, electrodes=electrodes,
                                            metadata=metadata,
                                            compress=compress)
        self.metadata = metadata

    def _pprint_params(self):
        params = super(ImageStimulus, self)._pprint_params()
        params.update({'img_shape': self.img_shape})
        return params

    def apply(self, func, *args, **kwargs):
        """Apply a function to the image

        Parameters
        ----------
        func : function
            The function to apply to the image. Must accept a 2D or 3D image
            and return an image with the same dimensions
        * args :
            Additional positional arguments passed to the function
        **kwargs :
            Additional keyword arguments passed to the function

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object with the new image
        """
        img = func(self.data.reshape(self.img_shape), *args, **kwargs)
        return ImageStimulus(img, electrodes=self.electrodes,
                             metadata=self.metadata)

    def invert(self):
        """Invert the gray levels of the image

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object with all grayscale values inverted
            in the range [0, 1].

        """
        img = deepcopy(self.data.reshape(self.img_shape))
        if len(self.img_shape) > 2:
            img[..., :3] = 1.0 - img[..., :3]
        else:
            img = 1.0 - img
        return ImageStimulus(img, electrodes=self.electrodes,
                             metadata=self.metadata)

    def rgb2gray(self, electrodes=None):
        """Convert the image to grayscale

        Parameters
        ----------
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, electrode names will be numbered 0..N.

            .. note::
               The number of electrode names provided must match the number of
               pixels in the grayscale image.

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object with all RGB values converted to
            grayscale in the range [0, 1].

        Notes
        -----
        *  A four-channel image is interpreted as RGBA (e.g., a PNG), and the
           alpha channel will be blended with the color black.

        """
        img = self.data.reshape(self.img_shape)
        if img.ndim == 3 and img.shape[2] == 4:
            # Blend the background with black:
            img = rgba2rgb(img, background=(0, 0, 0))
        return ImageStimulus(rgb2gray(img), electrodes=electrodes,
                             metadata=self.metadata)

    def resize(self, shape, electrodes=None):
        """Resize the image

        Parameters
        ----------
        shape : (rows, cols)
            Shape of the resized image
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, electrode names will be numbered 0..N.

            .. note::
               The number of electrode names provided must match the number of
               pixels in the grayscale image.

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object containing the resized image

        """
        height, width = shape
        if height < 0 and width < 0:
            raise ValueError('"height" and "width" cannot both be -1.')
        if height < 0:
            height = int(self.img_shape[0] * width / self.img_shape[1])
        if width < 0:
            width = int(self.img_shape[1] * height / self.img_shape[0])
        img = img_resize(self.data.reshape(self.img_shape), (height, width))

        return ImageStimulus(img, electrodes=electrodes,
                             metadata=self.metadata)

    def trim(self, tol=0, electrodes=None):
        """Remove any black border around the image

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
               pixels in the trimmed image.

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object with trimmed borders.

        """
        img = self.data.reshape(self.img_shape)
        return ImageStimulus(trim_image(img, tol=tol), electrodes=electrodes,
                             metadata=self.metadata)

    def threshold(self, thresh, **kwargs):
        """Threshold the image

        Parameters
        ----------
        thresh : str or float
            If a float in [0,1] is provided, pixels whose grayscale value is
            above said threshold will be white, others black.

            A number of additional methods are supported:

            *  'mean': Threshold image based on the mean of grayscale values.
            *  'minimum': Threshold image based on the minimum method, where
                          the histogram of the input image is computed and
                          smoothed until there are only two maxima.
            *  'local': Threshold image based on `local pixel neighborhood
                        <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_local>_.
                        Requires ``block_size``: odd number of pixels in the
                        neighborhood.
            *  'otsu': `Otsu's method
                       <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu>_
            *  'isodata': `ISODATA method
                          <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_isodata>`_,
                          also known as the Ridler-Calvard method or
                          intermeans.

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object with two gray levels 0.0 and 1.0
        """
        if len(self.img_shape) > 2:
            raise ValueError("Thresholding is only supported for grayscale "
                             "(i.e., single-channel) images. Use `rgb2gray` "
                             "first.")
        img = self.data.reshape(self.img_shape)
        if isinstance(thresh, str):
            if thresh.lower() == 'mean':
                img = img > threshold_mean(img)
            elif thresh.lower() == 'minimum':
                img = img > threshold_minimum(img, **kwargs)
            elif thresh.lower() == 'local':
                img = img > threshold_local(img, **kwargs)
            elif thresh.lower() == 'otsu':
                img = img > threshold_otsu(img, **kwargs)
            elif thresh.lower() == 'isodata':
                img = img > threshold_isodata(img, **kwargs)
            else:
                raise ValueError("Unknown threshold method '%s'." % thresh)
        elif np.isscalar(thresh):
            img = self.data.reshape(self.img_shape) > thresh
        else:
            raise TypeError("Threshold type must be str or float, not "
                            "%s." % type(thresh))
        return ImageStimulus(img, electrodes=self.electrodes,
                             metadata=self.metadata)

    def rotate(self, angle, mode='constant'):
        """Rotate the image

        Parameters
        ----------
        angle : float
            Angle by which to rotate the image (degrees).
            Positive: counter-clockwise, negative: clockwise

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object containing the rotated image

        """
        img = img_rotate(self.data.reshape(self.img_shape), angle, mode=mode,
                         resize=False)
        return ImageStimulus(img, electrodes=self.electrodes,
                             metadata=self.metadata)

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
        # Calculate center of mass:
        img = self.data.reshape(self.img_shape)
        return ImageStimulus(center_image(img, loc=None),
                             electrodes=self.electrodes,
                             metadata=self.metadata)

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
        img = self.data.reshape(self.img_shape)
        return ImageStimulus(scale_image(img, scaling_factor),
                             electrodes=self.electrodes,
                             metadata=self.metadata)

    def filter(self, filt, **kwargs):
        """Filter the image

        Parameters
        ----------
        filt : str
            Image filter. Additional parameters can be passed as keyword
            arguments. The following filters are supported:

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
        stim : `ImageStimulus`
            A copy of the stimulus object with the filtered image
        """
        if not isinstance(filt, str):
            raise TypeError("'filt' must be a string, not %s." % type(filt))
        filters = {'sobel': sobel, 'scharr': scharr, 'canny': canny,
                   'median': median}
        try:
            filt = filters[filt.lower()]
        except KeyError:
            raise ValueError("Unknown filter '%s'." % filt)
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
            (0.46 ms phase duration, 500 ms total duration) will be used.

        Returns
        -------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus`
            Encoded stimulus

        """
        if pulse is None:
            pulse = BiphasicPulse(1, 0.46, stim_dur=500)
        else:
            if not isinstance(pulse, Stimulus):
                raise TypeError("'pulse' must be a Stimulus object.")
            if pulse.time is None:
                raise ValueError("'pulse' must have a time component.")
        # Make sure the provided pulse has max amp 1:
        enc_data = pulse.data
        if not isclose(np.abs(enc_data).max(), 0):
            enc_data /= np.abs(enc_data).max()
        # Normalize the range of pixel values:
        px_data = self.data - self.data.min()
        if not isclose(np.abs(px_data).max(), 0):
            px_data /= np.abs(px_data).max()
        # Amplitude modulation:
        stim = {}
        for px, e in zip(px_data.ravel(), self.electrodes):
            amp = px * (amp_range[1] - amp_range[0]) + amp_range[0]
            s = Stimulus(amp * enc_data, time=pulse.time, electrodes=e)
            stim.update({e: s})
        return Stimulus(stim)

    def plot(self, ax=None, **kwargs):
        """Plot the stimulus

        Parameters
        ----------
        ax : matplotlib.axes.Axes or list thereof; optional, default: None
            A Matplotlib Axes object or a list thereof (one per electrode to
            plot). If None, a new Axes object will be created.

        Returns
        -------
        ax: matplotlib.axes.Axes
            Returns the axes with the plot on it

        """
        if ax is None:
            ax = plt.gca()
        if 'figsize' in kwargs:
            ax.figure.set_size_inches(kwargs.pop('figsize'))
        if 'vmin' in kwargs:
            vmin = kwargs.pop('vmin')
        else:
            vmin = 0

        cmap = None
        if len(self.img_shape) == 2:
            cmap = 'gray'
        if 'cmap' in kwargs:
            cmap = kwargs.pop('cmap')
        ax.imshow(self.data.reshape(self.img_shape), cmap=cmap, vmin=vmin,
                  **kwargs)
        return ax

    def save(self, fname):
        """Save the stimulus as an image

        Parameters
        ----------
        fname : str
            The name of the image file to be created. Image type will be
            inferred from the file extension.

        """
        imsave(fname, self.data.reshape(self.img_shape))


class SnellenChart(ImageStimulus):
    """Snellen chart

    Load the 1348x840 Snellen chart commonly used to measure visual acuity.

    .. versionadded:: 0.7

    Parameters
    ----------
    resize : (height, width) or None, optional
        A tuple specifying the desired height and the width of the image
        stimulus.

    show_annotations : {True, False}, optional
        If True, show the full Snellen chart including annotations of the rows
        and corresponding acuity measures.

    row : None, optional
        Select a single row (between 1 and 11) from the Snellen chart.
        For example, row 1 corresponds to 20/200, row 2 to 20/100.

    electrodes : int, string or list thereof; optional, default: None
        Optionally, you can provide your own electrode names. If none are
        given, electrode names will be numbered 0..N.

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional, default: None
        Additional stimulus metadata can be stored in a dictionary.

    """

    def __init__(self, resize=None, show_annotations=True, row=None,
                 electrodes=None, metadata=None):
        # Load image from data dir:
        module_path = dirname(__file__)
        source = join(module_path, 'data', 'snellen.png')
        if row is not None or show_annotations is False:
            # Need to crop the image before passing it on:
            source = imread(source)
            if show_annotations is False:
                # Crop the line numbers and acuity annotations:
                source = source[:, :444]
            if row is not None:
                # Select a single row of the chart, using the following as
                # start and stop indices:
                row_bounds = (
                    [5, 260],  # line 1
                    [310, 450],  # line 2
                    [505, 600],
                    [645, 715],
                    [755, 810],
                    [840, 883],
                    [965, 1003],
                    [1057, 1088],
                    [1170, 1193],
                    [1243, 1263],
                    [1317, 1335]  # line 11
                )
                try:
                    # It's 1-indexed, so make sure row=0 does not return the
                    # last row:
                    idx = row - 1
                    if idx < 0:
                        idx += 12
                    source = source[row_bounds[idx][0]:row_bounds[idx][1]]
                except (IndexError, TypeError):
                    raise ValueError('Invalid value for "row": %s. Choose '
                                     'an int between 1 and 11.' % row)
        # Call ImageStimulus constructor:
        super(SnellenChart, self).__init__(source, format="PNG",
                                           resize=resize,
                                           as_gray=True,
                                           electrodes=electrodes,
                                           metadata=metadata,
                                           compress=False)


class LogoBVL(ImageStimulus):
    """Bionic Vision Lab (BVL) logo

    Load the 576x720x4 Bionic Vision Lab (BVL) logo.

    .. versionadded:: 0.7

    Parameters
    ----------
    resize : (height, width) or None, optional
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. If none are
        given, electrode names will be numbered 0..N.

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional
        Additional stimulus metadata can be stored in a dictionary.

    """

    def __init__(self, resize=None, electrodes=None, metadata=None,
                 as_gray=False):
        # Load logo from data dir:
        module_path = dirname(__file__)
        source = join(module_path, 'data', 'bionic-vision-lab.png')
        # Call ImageStimulus constructor:
        super(LogoBVL, self).__init__(source, format="PNG",
                                      resize=resize,
                                      as_gray=as_gray,
                                      electrodes=electrodes,
                                      metadata=metadata,
                                      compress=False)


class LogoUCSB(ImageStimulus):
    """UCSB logo

    Load a 324x727 white-on-black logo of the University of California, Santa
    Barbara.

    .. versionadded:: 0.7

    Parameters
    ----------
    resize : (height, width) or None, optional
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. If none are
        given, electrode names will be numbered 0..N.

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional
        Additional stimulus metadata can be stored in a dictionary.

    """

    def __init__(self, resize=None, electrodes=None, metadata=None):
        # Load logo from data dir:
        module_path = dirname(__file__)
        source = join(module_path, 'data', 'ucsb.png')
        # Call ImageStimulus constructor:
        super(LogoUCSB, self).__init__(source, format="PNG",
                                       resize=resize,
                                       as_gray=True,
                                       electrodes=electrodes,
                                       metadata=metadata,
                                       compress=False)
