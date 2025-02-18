""":py:class:`~pulse2percept.stimuli.ImageStimulus`, 
   :py:class:`~pulse2percept.stimuli.LogoBVL`, 
   :py:class:`~pulse2percept.stimuli.LogoUCSB`, 
   :py:class:`~pulse2percept.stimuli.SnellenChart`"""
from os.path import dirname, join
import numpy as np
import warnings
from math import isclose
from copy import deepcopy
import matplotlib.pyplot as plt

from skimage import img_as_float32, img_as_ubyte
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
        and are inferred from the file ending.
        Use :py:class:`~pulse2percept.stimuli.VideoStimulus` for GIFs.

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

    def __init__(self, source, resize=None, as_gray=False,
                 electrodes=None, metadata=None, compress=False):
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            metadata = {'user': metadata}
        if isinstance(source, str):
            # Filename provided:
            img = imread(source)
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
            raise TypeError(f"Source must be a filename or another "
                            f"ImageStimulus, not {type(source)}.")
        if img.ndim < 2 or img.ndim > 3:
            raise ValueError(f"Images must have 2 or 3 dimensions, not "
                             f"{img.ndim}.")
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
        if img.ndim == 3:
            img = rgb2gray(img)
        return ImageStimulus(img, electrodes=electrodes,
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

    def crop(self, idx_rect=None, left=0, right=0, top=0, bottom=0,
             electrodes=None):
        """Crop the image

        This method maps a rectangle (defined by two corners) from the image
        to a rectangle of the given size. Alternatively, this method can be used
        to crop a number of columns either from the left or the right of the
        image, or a number of rows either from the top or the bottom.

        .. versionadded:: 0.8

        Parameters
        ----------
        idx_rect : 4-tuple (y0, x0, y1, x1)
            Image indices of the top-left corner ``[y0, x0]`` and bottom-right
            corner ``[y1, x1]`` (exclusive) of the rectangle to crop.
        left : int
            Number of columns to crop from the left
        right : int
            Number of columns to crop from the right
        top : int
            Number of rows to crop from the top
        bottom : int
            Number of rows to crop from the bottom
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, electrode names will be numbered 0..N.

            .. note::

               The number of electrode names provided must match the number of
               pixels in the cropped image.

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object containing the cropped image

        """
        if idx_rect is not None:
            if left > 0 or right > 0 or top > 0 or bottom > 0:
                raise ValueError('Crop window "idx_rect" cannot be given at '
                                 'the same time as "left"/"right"/"top"/'
                                 '"bottom".')
            # Crop window is given by a rectangle (ignore left, right, etc.):
            try:
                y0, x0, y1, x1 = idx_rect
            except (ValueError, TypeError):
                raise TypeError('"idx_rect" must be a 4-tuple (y0, x0, y1, x1)')
        else:
            y0, x0 = top, left
            y1, x1 = self.img_shape[0] - bottom, self.img_shape[1] - right
        # Safety checks:
        if y1 <= y0 or x1 <= x0:
            raise ValueError(f"The corners do not define a valid rectangle:"
                             f"(y0,x0)=({y0},{x0}), (y1,x1)=({y1},{x1}).")
        if y0 < 0 or x0 < 0:
            raise ValueError(f"Top-left corner (y0,x0)=({y0},{x0}) lies "
                             f"outside the image.")
        if y1 > self.img_shape[0] or x1 > self.img_shape[1]:
            raise ValueError(f"Bottom-right corner (y1-1,x1-1)=({y1-1},{x1-1}) lies "
                             f"outside the image.")
        # Crop the image:
        img = self.data.reshape(self.img_shape)
        # Check if we have color channels & index appropriately
        if len(self.img_shape) == 3:
            cropped_img = img[y0:y1, x0:x1, :3]
        else:
            cropped_img = img[y0:y1, x0:x1]
        electrodes = self.electrodes
        if electrodes is not None:
            electrodes = electrodes.reshape(self.img_shape)
            if len(self.img_shape) == 3:
                electrodes = electrodes[y0:y1, x0:x1, :3].ravel()
            else:
                electrodes = electrodes[y0:y1, x0:x1].ravel()
        return ImageStimulus(cropped_img, electrodes=electrodes,
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
            *  'local': Threshold image based on `local pixel neighborhood`_.
                        Requires ``block_size``: odd number of pixels in the
                        neighborhood.
            *  'otsu': `Otsu's method`_
            *  'isodata': `ISODATA method`_, also known as the Ridler-Calvard 
                          method or intermeans.

        .. _local pixel neighborhood: https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_local
        .. _Otsu's method: https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu
        .. _ISODATA method: https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_isodata

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
                raise ValueError(f"Unknown threshold method '{thresh}'.")
        elif np.isscalar(thresh):
            img = self.data.reshape(self.img_shape) > thresh
        else:
            raise TypeError(f"Threshold type must be str or float, not "
                            f"{type(thresh)}.")
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

            *  'sobel': Edge filter the image using the `Sobel filter`_.
            *  'scharr': Edge filter the image using the `Scharr filter`_.
            *  'canny': Edge filter the image using the `Canny algorithm`_.
               You can also specify ``sigma``, ``low_threshold``,
               ``high_threshold``, ``mask``, and ``use_quantiles``.
            *  'median': Return local median of the image.
        **kwargs :
            Additional parameters passed to the filter

        .. _Sobel filter: https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel
        .. _Scharr filter: https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.scharr
        .. _Canny algorithm: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.canny

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object with the filtered image
        """
        if not isinstance(filt, str):
            raise TypeError(f"'filt' must be a string, not {type(filt)}.")
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

    def save(self, fname, vmin=0, vmax=None):
        """Save the stimulus as an image

        Parameters
        ----------
        fname : str
            The name of the image file to be created. Image type will be
            inferred from the file extension.

        """
        # if vmax is not passed by user
        if vmax is None:
            vmax = self.data.max()
        # clip to vmin, vmax vals
        clipped_data = self.data.clip(vmin,vmax)
        # if not a TIFF file, scale to uint8
        if not fname.endswith(".tif") and not fname.endswith(".tiff"):
            # scale to [0,255] 
            scaled_data = ((clipped_data - vmin) * ( 1 / (vmax - vmin) * 255)).astype('uint8')
            imsave(fname, scaled_data.reshape(self.img_shape))
            warnings.warn(f"Stimulus {fname} has been scaled & compressed to the range [0, 255]. To retain the full precision and scaling of the original stimulus, please save using the TIFF format.", UserWarning)
        else:
            imsave(fname, clipped_data.reshape(self.img_shape))


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
                    raise ValueError(f'Invalid value for "row": {row}. Choose '
                                     f'an int between 1 and 11.')
        # Call ImageStimulus constructor:
        super(SnellenChart, self).__init__(source,
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
        super(LogoBVL, self).__init__(source,
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
        super(LogoUCSB, self).__init__(source,
                                       resize=resize,
                                       as_gray=True,
                                       electrodes=electrodes,
                                       metadata=metadata,
                                       compress=False)
