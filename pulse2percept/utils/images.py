"""`center_image`, `scale_image`, `shift_image`, `trim_image`"""
import numpy as np
from math import isclose
from skimage import img_as_bool, img_as_ubyte, img_as_float32
from skimage.color import rgba2rgb, rgb2gray
from skimage.measure import moments
from skimage.transform import warp, SimilarityTransform


def shift_image(img, shift_cols, shift_rows):
    """Shift the image foreground

    This function shifts the center of mass (CoM) of the image by the
    specified number of rows and columns.
    The background of the image is assumed to be black (0 grayscale).

    .. versionadded:: 0.7

    Parameters
    ----------
    img : ndarray
        A 2D NumPy array representing a (height, width) grayscale image, or a
        3D NumPy array representing a (height, width, channels) RGB image
    shift_cols : float
        Number of columns by which to shift the CoM.
        Positive: to the right, negative: to the left
    shift_rows : float
        Number of rows by which to shift the CoM.
        Positive: downward, negative: upward

    Returns
    -------
    img : ndarray
        A copy of the shifted image

    """
    if img.ndim < 2 or img.ndim > 3:
        raise ValueError("Only 2D and 3D images are allowed, not "
                         "%dD." % img.ndim)
    tf = SimilarityTransform(translation=[shift_cols, shift_rows])
    img_warped = warp(img, tf.inverse)
    # Warp automatically converts to double, so we need to convert the image
    # back to its original format:
    if img.dtype == bool:
        return img_as_bool(img_warped)
    if img.dtype == np.uint8:
        return img_as_ubyte(img_warped)
    if img.dtype == np.float32:
        return img_as_float32(img_warped)
    return img_warped


def center_image(img, loc=None):
    """Center the image foreground

    This function shifts the center of mass (CoM) to the image center.
    The background of the image is assumed to be black (0 grayscale).

    .. versionadded:: 0.7

    Parameters
    ----------
    img : ndarray
        A 2D NumPy array representing a (height, width) grayscale image, or a
        3D NumPy array representing a (height, width, channels) RGB image
    loc : (col, row), optional
        The pixel location at which to center the CoM. By default, shifts
        the CoM to the image center.

    Returns
    -------
    img : ndarray
        A copy of the image centered at ``loc``

    """
    if img.ndim < 2 or img.ndim > 3:
        raise ValueError("Only 2D and 3D images are allowed, not "
                         "%dD." % img.ndim)
    m = moments(img, order=1)
    # No area found:
    if isclose(m[0, 0], 0):
        return img
    # Center location:
    if loc is None:
        loc = np.array(img.shape[::-1]) / 2.0 - 0.5
    # Shift the image by -centroid, +image center:
    transl = (loc[0] - m[0, 1] / m[0, 0], loc[1] - m[1, 0] / m[0, 0])
    return shift_image(img, *transl)


def scale_image(img, scaling_factor):
    """Scale the image foreground

    This function scales the image foreground by a factor.
    The background of the image is assumed to be black (0 grayscale).

    .. versionadded:: 0.7

    Parameters
    ----------
    img : ndarray
        A 2D NumPy array representing a (height, width) grayscale image, or a
        3D NumPy array representing a (height, width, channels) RGB image
    scaling_factor : float
        Factory by which to scale the image

    Returns
    -------
    img : ndarray
        A copy of the scaled image

    """
    if img.ndim < 2 or img.ndim > 3:
        raise ValueError("Only 2D and 3D images are allowed, not "
                         "%dD." % img.ndim)
    if scaling_factor <= 0:
        raise ValueError("Scaling factor must be greater than zero")
    # Calculate center of mass:
    m = moments(img, order=1)
    # No area found:
    if isclose(m[0, 0], 0):
        return img
    # Shift the phosphene to (0, 0):
    center_mass = np.array([m[0, 1] / m[0, 0], m[1, 0] / m[0, 0]])
    tf_shift = SimilarityTransform(translation=-center_mass)
    # Scale the phosphene:
    tf_scale = SimilarityTransform(scale=scaling_factor)
    # Shift the phosphene back to where it was:
    tf_shift_inv = SimilarityTransform(translation=center_mass)
    # Combine all three transforms:
    tf = tf_shift + tf_scale + tf_shift_inv
    img_warped = warp(img, tf.inverse)
    # Warp automatically converts to double, so we need to convert the image
    # back to its original format:
    if img.dtype == bool:
        return img_as_bool(img_warped)
    if img.dtype == np.uint8:
        return img_as_ubyte(img_warped)
    if img.dtype == np.float32:
        return img_as_float32(img_warped)
    return img_warped


def trim_image(img, tol=0, return_coords=False):
    """Remove any black border around the image

    .. versionadded:: 0.7

    Parameters
    ----------
    img : ndarray
        A 2D NumPy array representing a (height, width) grayscale image, or a
        3D NumPy array representing a (height, width, channels) RGB image.
        If an alpha channel is present, the image will first be blended with
        black.
    tol : float, optional
        Any pixels with gray levels > tol will be trimmed.
    return_coords : bool, optional
        If True, will also return the row and column coordinates of the
        retained image

    Returns
    -------
    img : ndarray
        A copy of the image with trimmed borders.
    (row_start, row_end): tuple, optional
        The range of row indices in the trimmed image (returned only if
        ``return_coords`` is True)
    (col_start, col_end): tuple, optional
        The range of column indices in the trimmed image (returned only if
        ``return_coords`` is True)

    """
    if img.ndim < 2 or img.ndim > 3:
        raise ValueError("Only 2D and 3D images are allowed, not "
                         "%dD." % img.ndim)
    if tol < 0:
        raise ValueError("'tol' cannot be negative.")
    # Convert to grayscale if necessary:
    if img.ndim == 3 and img.shape[2] == 4:
        # Blend the background with black:
        img = rgba2rgb(img, background=(0, 0, 0))
    if img.ndim == 3:
        img = rgb2gray(img)
    m, n = img.shape
    mask = img > tol
    if not np.any(mask):
        return np.array([[]])
    # Determine the extent of the non-zero region:
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    # Trim the border by cropping out the relevant part:
    img = img[row_start:row_end, col_start:col_end, ...]
    if return_coords:
        return img, (row_start, row_end), (col_start, col_end)
    return img
