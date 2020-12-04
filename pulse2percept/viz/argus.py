import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.measure import moments as img_moments
from skimage.transform import (estimate_transform as img_transform,
                               warp as img_warp, SimilarityTransform)

import matplotlib.pyplot as plt
from matplotlib import patches
from pkg_resources import resource_filename

from ..implants import ArgusI, ArgusII
from ..utils import Watson2014Transform

PATH_ARGUS1 = resource_filename('pulse2percept', 'viz/data/argus1.png')
PATH_ARGUS2 = resource_filename('pulse2percept', 'viz/data/argus2.png')
# Pixel locations of electrodes (Argus I: A1-4, B1-4, ...; Argus II: A1-10,
# B1-10, ...) in the above images:
PX_ARGUS1 = np.array([
    [163.12857037, 92.32202802], [208.00952276, 93.7029804],
    [248.74761799, 93.01250421], [297.77142752, 91.63155183],
    [163.12857037, 138.58393279], [213.53333228, 137.8934566],
    [252.89047514, 137.2029804], [297.77142752, 136.51250421],
    [163.12857037, 181.3934566], [207.31904657, 181.3934566],
    [250.81904657, 181.3934566], [297.08095133, 181.3934566],
    [163.81904657, 226.27440898], [210.08095133, 226.27440898],
    [252.89047514, 227.65536136], [297.08095133, 227.65536136]
])
PX_ARGUS2 = np.array([
    [296.94026284, 140.58506571], [328.48148148, 138.4823178],
    [365.27956989, 140.58506571], [397.87216249, 139.53369176],
    [429.41338112, 138.4823178], [463.05734767, 140.58506571],
    [495.64994026, 139.53369176], [528.24253286, 139.53369176],
    [560.83512545, 139.53369176], [593.42771804, 138.4823178],
    [296.94026284, 173.1776583], [329.53285544, 174.22903226],
    [363.17682198, 173.1776583], [396.82078853, 173.1776583],
    [430.46475508, 173.1776583], [463.05734767, 174.22903226],
    [494.59856631, 173.1776583], [529.29390681, 174.22903226],
    [559.78375149, 175.28040621], [593.42771804, 173.1776583],
    [296.94026284, 206.82162485], [329.53285544, 206.82162485],
    [363.17682198, 205.7702509], [395.76941458, 205.7702509],
    [429.41338112, 205.7702509], [463.05734767, 208.92437276],
    [496.70131422, 207.87299881], [529.29390681, 209.97574671],
    [559.78375149, 208.92437276], [592.37634409, 206.82162485],
    [296.94026284, 240.4655914], [330.58422939, 240.4655914],
    [363.17682198, 240.4655914], [396.82078853, 240.4655914],
    [430.46475508, 240.4655914], [460.95459976, 240.4655914],
    [494.59856631, 242.56833931], [528.24253286, 239.41421744],
    [559.78375149, 240.4655914], [593.42771804, 241.51696535],
    [297.9916368, 274.10955795], [328.48148148, 273.05818399],
    [361.07407407, 274.10955795], [395.76941458, 273.05818399],
    [428.36200717, 274.10955795], [463.05734767, 273.05818399],
    [494.59856631, 275.1609319], [526.13978495, 274.10955795],
    [560.83512545, 274.10955795], [591.32497013, 274.10955795],
    [295.88888889, 306.70215054], [329.53285544, 305.65077658],
    [363.17682198, 305.65077658], [393.66666667, 307.75352449],
    [427.31063321, 307.75352449], [459.90322581, 305.65077658],
    [492.4958184, 308.80489845], [527.1911589, 307.75352449],
    [559.78375149, 307.75352449], [590.27359618, 306.70215054]
])


def center_phosphene(img, loc=None):
    """Center the image foreground

    This function shifts the center of mass (CoM) to the image center.

    .. versionadded:: 0.7

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
    m = img_moments(img, order=1)
    # No area found:
    if np.isclose(m[0, 0], 0):
        return img
    # Center location:
    if loc is None:
        loc = np.array(self.img_shape[::-1]) / 2.0 - 0.5
    # Shift the image by -centroid, +image center:
    transl = (loc[0] - m[0, 1] / m[0, 0], loc[1] - m[1, 0] / m[0, 0])
    tf_shift = SimilarityTransform(translation=transl)
    img = img_warp(img, tf_shift.inverse)
    return img


def scale_phosphene(img, scaling_factor):
    """Scale the image foreground

    This function scales the image foreground (excluding black pixels)
    by a factor.

    .. versionadded:: 0.7

    Parameters
    ----------
    scaling_factor : float
        Factory by which to scale the image

    Returns
    -------
    stim : `ImageStimulus`
        A copy of the stimulus object containing the scaled image

    """
    if scaling_factor <= 0:
        raise ValueError("Scaling factor must be greater than zero")
    # Calculate center of mass:
    m = img_moments(img, order=1)
    # No area found:
    if np.isclose(m[0, 0], 0):
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
    img = img_warp(img, tf.inverse)
    return img


def plot_argus_phosphenes(X, argus, scale=1.0, axon_map=None, show_fovea=True,
                          ax=None):
    """Plots phosphenes centered over the corresponding electrodes

    .. versionadded:: 0.7

    Parameters
    ----------
    X : pd.DataFrame
    argus : :py:class:`~pulse2percept.implants.ArgusI` or :py:class:`~pulse2percept.implants.ArgusII`
        Either an Argus I or Argus II implant
    scale : float
        Scaling factor to apply to the phosphenes
    show_fovea : bool
        Whether to indicate the location of the fovea with a square
    ax : axis
        Matplotlib axis
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError('"X" must be a Pandas DataFrame, not %s.' % type(X))
    req_cols = ['subject', 'electrode', 'image', 'img_x_dva', 'img_y_dva']
    if not all(col in X.columns for col in req_cols):
        raise ValueError('"X" must have columns %s.' % req_cols)
    if len(X) == 0:
        raise ValueError('"X" is empty.')
    if len(X.subject.unique()) > 1:
        raise ValueError('"X" cannot contain data from more than one subject.')
    if not isinstance(argus, (ArgusI, ArgusII)):
        raise TypeError('"argus" must be an Argus I or Argus II implant, not '
                        '%s.' % type(argus))
    if ax is None:
        ax = plt.gca()
    alpha_bg = 0.5  # alpha value for the array in the background
    thresh_fg = 0.95  # Grayscale value above which to mask the drawings

    if isinstance(argus, ArgusI):
        px_argus = PX_ARGUS1
        img_argus = imread(PATH_ARGUS1)
    else:
        px_argus = PX_ARGUS2
        img_argus = imread(PATH_ARGUS2)

    # To simulate an implant in a left eye, flip the image left-right (along
    # with the electrode x-coordinates):
    if argus.eye == 'LE':
        img_argus = np.fliplr(img_argus)
        px_argus[:, 0] = img_argus.shape[1] - px_argus[:, 0] - 1

    # Add some padding to the output image so the array is not cut off:
    pad = 2000  # microns
    x_el = [e.x for e in argus.values()]
    y_el = [e.y for e in argus.values()]
    x_min = Watson2014Transform.ret2dva(np.min(x_el) - pad)
    x_max = Watson2014Transform.ret2dva(np.max(x_el) + pad)
    y_min = Watson2014Transform.ret2dva(np.min(y_el) - pad)
    y_max = Watson2014Transform.ret2dva(np.max(y_el) + pad)

    # Coordinate transform from degrees of visual angle to output, and from
    # image coordinates to output image:
    pts_in = []
    pts_dva = []
    pts_out = []
    try:
        out_shape = X.img_shape.values[0]
    except AttributeError:
        # Dataset does not have 'img_shape' column:
        try:
            out_shape = X.image.values[0].shape
        except IndexError:
            out_shape = (768, 1024)
    for xy, e in zip(px_argus, argus.values()):
        x_dva, y_dva = Watson2014Transform.ret2dva([e.x, e.y])
        x_out = (x_dva - x_min) / (x_max - x_min) * (out_shape[1] - 1)
        y_out = (y_dva - y_min) / (y_max - y_min) * (out_shape[0] - 1)
        pts_in.append(xy)
        pts_dva.append([x_dva, y_dva])
        pts_out.append([x_out, y_out])
    pts_in = np.array(pts_in)
    pts_dva = np.array(pts_dva)
    pts_out = np.array(pts_out)
    dva2out = img_transform('similarity', pts_dva, pts_out)
    argus2out = img_transform('similarity', pts_in, pts_out)

    # Top left, top right, bottom left, bottom right corners:
    x_range = X.img_x_dva
    y_range = X.img_y_dva
    pts_dva = [[x_range[0], y_range[0]], [x_range[0], y_range[1]],
               [x_range[1], y_range[0]], [x_range[1], y_range[1]]]

    # Calculate average drawings, but don't binarize:
    all_imgs = np.zeros(out_shape)
    num_imgs = X.groupby('electrode')['image'].count()
    for _, row in X.iterrows():
        e_pos = Watson2014Transform.ret2dva((argus[row['electrode']].x,
                                             argus[row['electrode']].y))
        align_center = dva2out(e_pos)[0]
        img_drawing = scale_phosphene(row['image'], scale)
        img_drawing = center_phosphene(img_drawing, loc=align_center)
        # We normalize by the number of phosphenes per electrode, so that if
        # all phosphenes are the same, their brightness would add up to 1:
        all_imgs += 1.0 / num_imgs[row['electrode']] * img_drawing
    all_imgs = 1 - all_imgs

    # Draw array schematic with specific alpha level:
    img_arr = img_warp(img_argus, argus2out.inverse, cval=1.0,
                       output_shape=out_shape)
    img_arr[:, :, 3] = alpha_bg

    # Replace pixels where drawings are dark enough, set alpha=1:
    rr, cc = np.unravel_index(np.where(all_imgs.ravel() < thresh_fg)[0],
                              all_imgs.shape)
    for channel in range(3):
        img_arr[rr, cc, channel] = all_imgs[rr, cc]
    img_arr[rr, cc, 3] = 1

    ax.imshow(img_arr, cmap='gray', zorder=1)

    if show_fovea:
        fovea = dva2out([0, 0])[0]
        ax.scatter(*fovea, s=100, marker='s', c='w', edgecolors='k', zorder=99)

    if axon_map is not None:
        axon_bundles = axon_map.grow_axon_bundles(n_bundles=100, prune=False)
        # Draw axon pathways:
        for bundle in axon_bundles:
            # Flip y upside down for dva:
            bundle = Watson2014Transform.ret2dva(bundle) * [1, -1]
            # Trim segments outside the drawing window:
            idx = np.logical_and(np.logical_and(bundle[:, 0] >= x_min,
                                                bundle[:, 0] <= x_max),
                                 np.logical_and(bundle[:, 1] >= y_min,
                                                bundle[:, 1] <= y_max))
            bundle = dva2out(bundle)
            ax.plot(bundle[idx, 0], bundle[idx, 1], c=(0.6, 0.6, 0.6),
                    linewidth=2, zorder=1)

    return ax
