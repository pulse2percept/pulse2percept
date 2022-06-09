"""`center_image`, `scale_image`, `shift_image`, `trim_image`"""
import numpy as np
from math import isclose
from scipy.signal import convolve2d
from skimage import img_as_bool, img_as_ubyte, img_as_float32
from skimage.color import rgba2rgb, rgb2gray
from skimage.measure import moments
from skimage.transform import warp, SimilarityTransform

from ._preprocessing import (temporal_saliency, seam_carving_energy,
                             seam_carvings, importance_matrix, shrink_row)


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
        raise ValueError(f"Only 2D and 3D images are allowed, not "
                         f"{img.ndim}D.")
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
        raise ValueError(f"Only 2D and 3D images are allowed, not "
                         f"{img.ndim}D.")
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
        raise ValueError(f"Only 2D and 3D images are allowed, not "
                         f"{img.ndim}D.")
    if img.ndim == 3 and img.shape[-1] > 3:
        raise ValueError(f"Only RGB and grayscale images are allowed, not "
                         f"{img.shape[-1]}-channel images.")
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
        raise ValueError(f"Only 2D and 3D images are allowed, not "
                         f"{img.ndim}D.")
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


def spatial_saliency(image_gray):
    """Calculates the spatial saliency map

    This function calculates the spatial saliency map based on the algorithm in [Fleck1992]

    Parameters
    ----------
    image_gray : A 2D NumPy array 
       represents a (height, width) grayscale image
    """
    # compute the first derivative in four directions : x (horizontal), y (vertical), d1, and d2
    dx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])*0.125
    dy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])*0.125
    dd_1 = np.array([[0, 1, 2],
                     [-1, 0, 1],
                     [-2, -1, 0]])*0.125
    dd_2 = np.array([[-2, -1, 0],
                     [-1, 0, 1],
                     [0, 1, 2]])*0.125
    img_sobel_x = convolve2d(image_gray, dx, mode='same')
    img_sobel_y = convolve2d(image_gray, dy, mode='same')
    img_sobel_d1 = convolve2d(image_gray, dd_1, mode='same')
    img_sobel_d2 = convolve2d(image_gray, dd_2, mode='same')

    # This is implementing Eq. 1 from the paper:
    X = img_sobel_x+(img_sobel_d1+img_sobel_d2)*0.5

    # This is implementing Eq. 2 from the paper:
    Y = img_sobel_y+(img_sobel_d1-img_sobel_d2)*0.5

    # This is implementing Eq. 3 from the paper:
    Ws = np.sqrt(X*X+Y*Y)
    return Ws


def _spatial_temporal_saliency(image_gray, second_frame, N=4, boundary=0.5):
    """Calculates the spatio-temporal importance matrix

    This function calculates the spatio-temporal importance matrix, which is the combination of the spatial saliency map and the temporal saliency map

    Parameters
    ----------
    image_gray : A 2D NumPy array 
      represents a (height, width) grayscale image
    second_frame : A 2D NumPy array
      represents another (height, width) grayscale image which is the previous or the next frame of image_gray
    N : int, optional
      the image will be divided into blocks of size N*N, and the motion map Wt[x, y] will be determined by if there is motion in the block containing the pixel [x,y]
    boundary : float, optional
      claims that there is a change occurs when the difference in the motion saliency map of two images is higher than the boundary
    """
    Ws = spatial_saliency(image_gray)
    Wt = temporal_saliency(image_gray, second_frame, N, boundary)
    # This is implementing Eq. 4 from the paper:
    Wst = Ws+Wt

    # This is implementing Eq. 5 from the paper:
    Wst = (Wst-Wst.min())/(Wst.max()-Wst.min())

    return Wst


def shrinkability_matrix(W_fin, image_gray_shape, K):
    """Calculates shrinkability matrix

    This function calculates the shrinkability matrix from the final importance matrix, which indicates how much of an input pixel contributes to each successive output pixel.

    Parameters
    ----------
    W_fin : 2d numpy array
      represents the final importance matrix
    image_gray_shape: A tuple of int
      represents the shape of the image
    K: int, optional
      reduces the source image width by K pixels
    """
    # This is implementing Eq. 14 from the paper:
    sum_of_inv_w = 0
    for j in range(image_gray_shape[1]):
        sum_of_inv_w += 1/W_fin[0, j]
    S = 1/(W_fin[0]*sum_of_inv_w)

    # This is implementing Eq. 15 16 and 17 from the paper:
    S_ = S*K
    while True:
        S_ = np.array([min(0.9, S_[j]) for j in range(image_gray_shape[1])])
        S_ = K/np.sum(S_)*S_
        if (np.abs(np.sum(S_)-K) < 0.001):
            break
    return S_


def shrinked(S_, image_gray, K):
    """Calculates the output shrinked image

    This function calculates the shrinked image using the shrinkability matrix

    Parameters
    ----------
    S : 1d numpy array
      repre---
    imsents the shrinkability matrix
    image_gray : A 2D NumPy array 
      represents a (height, width) grayscale image
    K: int
      reduces the source image width by K pixels.
    """
    retain_factor = 1 - S_
    result_rows = []
    for i in range(image_gray.shape[0]):
        sr = shrink_row(image_gray[i], retain_factor, K)
        result_rows.append(sr)
    return result_rows


def image_retargeting(image_gray, second_frame, wid=0, hei=0, N=4, boundary=0.5,
                      L=5, num=15):
    """Calculates the image after content-aware image retargeting

    This function calculates the shrinked image from the image and the next or previous frame of the image

    Parameters
    Image_gray : A NumPy array 
      represents a (height, width) grayscale image. The input size of the image should be larger than 15*15.
    second_frame : A NumPy array
      represents another (height, width) grayscale image which is the previous or the next frame of image_gray,
    wid: int
      reduces the source image width by wid pixels.
    hei: int
      reduces the source image height by hei pixels.
    N : int, optional
      the image will be divided into blocks of size N*N, and the motion map Wt[x, y] will be determined by if there is motion in the block containing the pixel [x,y]. Initialized to 4
    boundary : float, optional
      claims that there is a change occurs when the difference in the motion saliency map of two images is higher than the boundary. Initialized to 0.5.
    L: int, optional
      a moving average window of size L would be applied to the importance matrix to preserve continuity between rows. Initilized to 5.
    num: int, optional
      the number of seams we would like to have. Initialized to 15. num should be smaller than the resolution of your image or video.
    """
    if len(image_gray.shape) > 2:
        image_gray = color.rgb2gray(image_gray)
    if len(second_frame.shape) > 2:
        second_frame = color.rgb2gray(second_frame)
    result = image_gray
    if wid > 0:
        Wst = _spatial_temporal_saliency(image_gray, second_frame, N, boundary)
        W_fin = importance_matrix(Wst, image_gray.shape, L, num)
        S_ = shrinkability_matrix(W_fin, image_gray.shape, wid)
        result = np.array(shrinked(S_, image_gray, wid))
    if hei > 0:
        if wid > 0:
            shrinked_second = np.array(shrinked(S_, second_frame, wid))
        result = np.rot90(result, 1)
        Wst = _spatial_temporal_saliency(
            result, np.rot90(shrinked_second, 1), N, boundary)
        W_fin = importance_matrix(Wst, result.shape, L, num)
        S_ = shrinkability_matrix(W_fin, result.shape, hei)
        result = np.array(shrinked(S_, result, hei))
        result = np.rot90(result, -1)
    return result


def single_image_retargeting(image_gray, wid=0, hei=0, L=5, num=15):
    """Calculates the image after content-aware image retargeting with only one image

    This function calculates the spatio-temporal importance matrix, which is the combination of the spatial saliency map and the temporal saliency map

    Parameters
    ----------
    image_gray : A NumPy array 
      represents a (height, width) grayscale image
    second_frame : A NumPy array
      represents another (height, width) grayscale image which is the previous or the next frame of image_gray
    wid: int
      reduces the source image width by wid pixels.
    hei: int
      reduces the source image height by hei pixels.
    L: int, optional
      a moving average window of size L would be applied to the importance matrix to preserve continuity between rows. Initilized to 5.
    num: int, optional
      the number of seams we would like to have. Initialized to 15. num should be smaller than the resolution of your image or video.
    """
    if len(image_gray.shape) > 2:
        image_gray = color.rgb2gray(image_gray)
    result = image_gray
    if wid > 0:
        Wst = spatial_saliency(image_gray)
        W_fin = importance_matrix(Wst, image_gray.shape, L, num)
        S_ = shrinkability_matrix(W_fin, image_gray.shape, wid)
        result = shrinked(S_, image_gray, wid)
    if hei > 0:
        result = np.rot90(result, 1)
        Wst = spatial_saliency(result)
        W_fin = importance_matrix(Wst, result.shape, L, num)
        S_ = shrinkability_matrix(W_fin, result.shape, hei)
        result = np.array(shrinked(S_, result, hei))
        result = np.rot90(result, -1)
    return result


def video_retargeting_1d(video, K, N=4, boundary=0.5, L=5, num=15):
    """Calculates the retargeted video which is shrinked by only one dimension (width)

    This function calculates the shrinked video which is shrinked by only one dimension

    Parameters
    ----------
    video : A 3D NumPy array
      represents a video in gray scale (array of gray-scale images). The video resolution should be larger than 15*15, and the duration should be longer than 2 frame.
    K: int
      reduces the source image width by K pixels.
    N : int, optional
      the image will be divided into blocks of size N*N, and the motion map Wt[x, y] will be determined by if there is motion in the block containing the pixel [x,y]. Initialized to 4
    boundary : float, optional
      claims that there is a change occurs when the difference in the motion saliency map of two images is higher than the boundary. Initialized to 0.5.
    L: int, optional
      a moving average window of size L would be applied to the importance matrix to preserve continuity between rows. Initilized to 5.
    num: int, optional
      the number of seams we would like to have. Initialized to 15. num should be smaller than the resolution of your image or video.
    """
    result = []
    Wst_ = np.zeros((video.shape[0], video.shape[1], video.shape[2]))
    Wst_[0] = _spatial_temporal_saliency(video[0], video[1], N, boundary)
    for i in range(1, video.shape[0]):
        image_gray = video[i]
        second_frame = video[i-1]
        Wst_[i] = _spatial_temporal_saliency(
            image_gray, second_frame, N, boundary)

    M_v, bt_v = seam_carving_energy(Wst_[0], image_gray.shape)
    lowest = np.argpartition(
        M_v[image_gray.shape[0]-1, 0:image_gray.shape[1]], num)
    old_sum = sum(M_v[image_gray.shape[0]-1, lowest[0:num]])
    for i in range(0, video.shape[0]):
        image_gray = video[i]
        M_v_new, bt_v_new = seam_carving_energy(Wst_[i], image_gray.shape)
        lowest = np.argpartition(
            M_v_new[image_gray.shape[0]-1, 0:image_gray.shape[1]], num)
        if (abs(sum(M_v[image_gray.shape[0]-1, lowest[0:num]])-old_sum) < old_sum*0.2):
            Wst_rescaled = seam_carvings(
                M_v, bt_v, Wst_[i], image_gray.shape, num)
        else:
            old_sum = sum(M_v[image_gray.shape[0]-1, lowest[0:num]])
            Wst_rescaled = seam_carvings(M_v_new, bt_v_new, Wst_[
                                         i], image_gray.shape, num)
        W_fin = importance_matrix(Wst_rescaled, image_gray.shape, L, num)
        S_ = shrinkability_matrix(W_fin, image_gray.shape, K)
        frame = np.array(shrinked(S_, image_gray, K))
        if (result != []):
            if (frame.shape == result[0].shape):
                result.append(frame)
        else:
            result.append(frame)
    return np.array(result)


def video_retargeting(video, wid=0, hei=0, N=4, boundary=0.5, L=5, num=15):
    """Calculates the shrinked video.

    This function calculates the shrinked video.

    Parameters
    ----------
    video : A 3D NumPy array
      represents a video (array of images). The duration of the video should be longer than 2 frame.
    wid: int
      reduces the source image width by wid pixels.
    hei: int
      reduces the source image width by hei pixels.
    N : int, optional
      the image will be divided into blocks of size N*N, and the motion map Wt[x, y] will be determined by if there is motion in the block containing the pixel [x,y]. Initialized to 4
    boundary : float, optional
      claims that there is a change occurs when the difference in the motion saliency map of two images is higher than the boundary. Initialized to 0.5.
    L: int, optional
      a moving average window of size L would be applied to the importance matrix to preserve continuity between rows. Initilized to 5.
    num: int, optional
      the number of seams we would like to have. Initialized to 15. num should be smaller than the resolution of your image or video.
    """
    result = video
    if len(video.shape) == 4:
        result = color.rgb2gray(video)
    if wid > 0:
        result = video_retargeting_1d(result, wid, N, boundary, L, num)
    if hei > 0:
        result = np.rot90(result, 1, (1, 2))
        result = video_retargeting_1d(result, hei, N, boundary, L, num)
        result = np.rot90(result, -1, (1, 2))
    return result
