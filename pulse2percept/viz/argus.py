"""`plot_argus_phosphenes`, `plot_argus_simulated_phosphenes`"""
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
from ..models import AxonMapModel
from ..utils import Watson2014Transform, scale_image, center_image
from ..utils.constants import ZORDER

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


def plot_argus_phosphenes(data, argus, scale=1.0, axon_map=None,
                          show_fovea=True, ax=None):
    """Plots phosphenes centered over the corresponding electrodes

    .. versionadded:: 0.7

    Parameters
    ----------
    data : pd.DataFrame
        The Beyeler2019 dataset, a subset thereof, or a DataFrame with
        identical organization (i.e., must contain columns 'subject', 'image',
        'xrange', and 'yrange').
    argus : :py:class:`~pulse2percept.implants.ArgusI` or :py:class:`~pulse2percept.implants.ArgusII`
        Either an Argus I or Argus II implant
    scale : float
        Scaling factor to apply to the phosphenes
    axon_map : :py:class:`~pulse2percept.models.AxonMapModel`
        An instance of the axon map model to use for visualization.
    show_fovea : bool
        Whether to indicate the location of the fovea with a square
    ax : axis
        Matplotlib axis
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError('"data" must be a Pandas DataFrame, not '
                        '%s.' % type(data))
    req_cols = ['subject', 'electrode', 'image', 'xrange', 'yrange']
    if not all(col in data.columns for col in req_cols):
        raise ValueError('"data" must have columns %s.' % req_cols)
    if len(data) == 0:
        raise ValueError('"data" is empty.')
    if len(data.subject.unique()) > 1:
        raise ValueError('"data" cannot contain data from more than one '
                         'subject.')
    if not isinstance(argus, (ArgusI, ArgusII)):
        raise TypeError('"argus" must be an Argus I or Argus II implant, not '
                        '%s.' % type(argus))
    if axon_map is not None and not isinstance(axon_map, AxonMapModel):
        raise TypeError('"axon_map" must be an AxonMapModel instance, not '
                        '%s.' % type(axon_map))
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
    x_el = [e.x for e in argus.electrode_objects]
    y_el = [e.y for e in argus.electrode_objects]
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
        out_shape = data.img_shape.values[0]
    except AttributeError:
        # Dataset does not have 'img_shape' column:
        try:
            out_shape = data.image.values[0].shape
        except IndexError:
            out_shape = (768, 1024)
    for xy, e in zip(px_argus, argus.electrode_objects):
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
    x_range = data.xrange
    y_range = data.yrange
    pts_dva = [[x_range[0], y_range[0]], [x_range[0], y_range[1]],
               [x_range[1], y_range[0]], [x_range[1], y_range[1]]]

    # Calculate average drawings, but don't binarize:
    all_imgs = np.zeros(out_shape)
    num_imgs = data.groupby('electrode')['image'].count()
    for _, row in data.iterrows():
        e_pos = Watson2014Transform.ret2dva((argus[row['electrode']].x,
                                             argus[row['electrode']].y))
        align_center = dva2out(e_pos)[0]
        img_drawing = scale_image(row['image'], scale)
        img_drawing = center_image(img_drawing, loc=align_center)
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

    ax.imshow(img_arr, cmap='gray', zorder=ZORDER['background'])

    if show_fovea:
        fovea = dva2out([0, 0])[0]
        ax.scatter(*fovea, s=100, marker='s', c='w', edgecolors='k',
                   zorder=ZORDER['foreground'])

    if axon_map is not None:
        axon_bundles = axon_map.grow_axon_bundles(n_bundles=100, prune=False)
        # Draw axon pathways:
        for bundle in axon_bundles:
            # Flip y upside down for dva:
            bundle = Watson2014Transform.ret2dva(bundle) * [1, -1]
            # Set segments outside the drawing window to NaN:
            x_idx = np.logical_or(bundle[:, 0] < x_min, bundle[:, 0] > x_max)
            bundle[x_idx, 0] = np.nan
            y_idx = np.logical_or(bundle[:, 1] < y_min, bundle[:, 1] > y_max)
            bundle[y_idx, 1] = np.nan
            bundle = dva2out(bundle)
            ax.plot(bundle[:, 0], bundle[:, 1], c=(0.6, 0.6, 0.6),
                    linewidth=2, zorder=ZORDER['background'])

    return ax


def plot_argus_simulated_phosphenes(percepts, argus, scale=1.0, axon_map=None,
                                    show_fovea=True, ax=None):
    """Plots simulated phosphenes centered over the corresponding electrodes

    .. versionadded:: 0.7

    Parameters
    ----------
    percepts : :py:class:`~pulse2percept.percepts.Percept`
        A Percept object containing multiple frames, where each frame is the
        percept produced by activating a single electrode.
    argus : :py:class:`~pulse2percept.implants.ArgusI` or :py:class:`~pulse2percept.implants.ArgusII`
        Either an Argus I or Argus II implant
    scale : float
        Scaling factor to apply to the phosphenes
    axon_map : :py:class:`~pulse2percept.models.AxonMapModel`
        An instance of the axon map model to use for visualization.
    show_fovea : bool
        Whether to indicate the location of the fovea with a square
    ax : axis
        Matplotlib axis

    """
    stim = percepts.metadata['stim']
    if not np.allclose(stim.data.sum(axis=0), 1):
        raise ValueError("This function only works for stimuli where one "
                         "electrodes is active at a time.")
    # Build the missing DataFrame columns from Percept metadata:
    df = []
    for p, s in zip(percepts, stim.data.T):
        df.append({
            'subject': 'S000',
            'electrode': stim.electrodes[np.asarray(s, dtype=bool)][0],
            'image': p,
            'xrange': (percepts.xdva.min(), percepts.xdva.max()),
            'yrange': (percepts.ydva.min(), percepts.ydva.max())
        })
    plot_argus_phosphenes(pd.DataFrame(df), argus, scale=scale, ax=ax,
                          axon_map=axon_map, show_fovea=show_fovea)
