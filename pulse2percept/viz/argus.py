import numpy as np
import scipy.stats as spst
import skimage.io as skio
import skimage.transform as skit

import pulse2percept.implants as p2pi
import pulse2percept.retina as p2pr
import pulse2percept.utils as p2pu

from matplotlib import patches
from pkg_resources import resource_filename

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


def plot_argus_phosphenes(ax, subject, Xymu, subjectdata, alpha_bg=0.5,
                          thresh_fg=0.95, show_fovea=True):
    """Plots phosphenes centered over the corresponding electrodes

    Parameters
    ----------
    ax : axis
        Matplotlib axis
    subject : str
        Subject ID, must be a valid value for column 'subject' in `Xymu` and
        `subjectdata`.
    Xymu : pd.DataFrame
        DataFrame with required columns 'electrode', 'image'. May contain data
        from more than one subject, in which case a column 'subject' must
        exist. May also have a column 'img_shape' indicating the shape of each
        phosphene image.
    subjectdata : pd.DataFrame
        DataFrame with Subject ID as index. Must have columns 'implant_x',
        'implant_y', 'implant_rot', 'implant_type', and 'eye'. May also have a
        column 'scale' containing a scaling factor applied to phosphene size.
    alpha_bg : float
        Alpha value for the array in the background
    thresh_fg : float
        Grayscale value above which to mask the drawings
    show_fovea : bool
        Whether to indicate the location of the fovea with a square
    """
    for col in ['electrode', 'image']:
        if col not in Xymu.columns:
            raise ValueError('Xymu must contain column "%s".' % col)
    # If subject column not present, choose all entries:
    if 'subject' in Xymu.columns:
        Xymu = Xymu[Xymu.subject == subject]
    for col in ['implant_x', 'implant_y', 'implant_rot', 'implant_type',
                'eye']:
        if col not in subjectdata.columns:
            raise ValueError('subjectdata must contain column "%s".' % col)
    if subject not in subjectdata.index:
        raise ValueError('Subject "%s" not an index in subjectdata.' % subject)
    if 'scale' not in subjectdata.columns:
        print("'scale' not in subjectdata, setting scale=1.0")
        subjectdata['scale'] = 1.0

    eye = subjectdata.loc[subject, 'eye']

    # Choose the appropriate image / electrode locations based on implant type:
    implant_type = subjectdata.loc[subject, 'implant_type']
    is_argus2 = isinstance(implant_type(), p2pi.ArgusII)
    if is_argus2:
        px_argus = PX_ARGUS2
        img_argus = skio.imread(IMG_ARGUS2)
    else:
        px_argus = PX_ARGUS1
        img_argus = skio.imread(IMG_ARGUS1)

    # To simulate an implant in a left eye, flip the image left-right (along
    # with the electrode x-coordinates):
    if eye == 'LE':
        img_argus = np.fliplr(img_argus)
        px_argus[:, 0] = img_argus.shape[1] - px_argus[:, 0] - 1

    # Create an instance of the array using p2p:
    argus = implant_type(x_center=subjectdata.loc[subject, 'implant_x'],
                         y_center=subjectdata.loc[subject, 'implant_y'],
                         rot=subjectdata.loc[subject, 'implant_rot'],
                         eye=eye)

    # Add some padding to the output image so the array is not cut off:
    padding = 2000  # microns
    x_range = (p2pr.ret2dva(np.min([e.x_center for e in argus]) - padding),
               p2pr.ret2dva(np.max([e.x_center for e in argus]) + padding))
    y_range = (p2pr.ret2dva(np.min([e.y_center for e in argus]) - padding),
               p2pr.ret2dva(np.max([e.y_center for e in argus]) + padding))

    # If img_shape column not present, choose shape of first entry:
    if 'img_shape' in Xymu.columns:
        out_shape = Xymu.img_shape.unique()[0]
    else:
        out_shape = Xymu.image.values[0].shape

    # Coordinate transform from degrees of visual angle to output, and from
    # image coordinates to output image:
    pts_in = []
    pts_dva = []
    pts_out = []
    for xy, e in zip(px_argus, argus):
        pts_in.append(xy)
        dva = p2pr.ret2dva([e.x_center, e.y_center])
        pts_dva.append(dva)
        xout = (dva[0] - x_range[0]) / \
            (x_range[1] - x_range[0]) * (out_shape[1] - 1)
        yout = (dva[1] - y_range[0]) / \
            (y_range[1] - y_range[0]) * (out_shape[0] - 1)
        pts_out.append([xout, yout])
    dva2out = skit.estimate_transform('similarity', np.array(pts_dva),
                                      np.array(pts_out))
    argus2out = skit.estimate_transform('similarity', np.array(pts_in),
                                        np.array(pts_out))

    # Top left, top right, bottom left, bottom right:
    x_range = subjectdata.loc[subject, 'xrange']
    y_range = subjectdata.loc[subject, 'yrange']
    pts_dva = [[x_range[0], y_range[0]], [x_range[0], y_range[1]],
               [x_range[1], y_range[0]], [x_range[1], y_range[1]]]

    # Calculate average drawings, but don't binarize:
    all_imgs = np.zeros(out_shape)
    for _, row in Xymu.iterrows():
        e_pos = p2pr.ret2dva((argus[row['electrode']].x_center,
                              argus[row['electrode']].y_center))
        align_center = dva2out(e_pos)[0]
        img_drawing = imgproc.scale_phosphene(
            row['image'], subjectdata.loc[subject, 'scale']
        )
        img_drawing = imgproc.center_phosphene(
            img_drawing, center=align_center[::-1]
        )
        all_imgs += img_drawing
    all_imgs = 1 - np.minimum(1, np.maximum(0, all_imgs))

    # Draw array schematic with specific alpha level:
    img_arr = skit.warp(img_argus, argus2out.inverse, cval=1.0,
                        output_shape=out_shape)
    img_arr[:, :, 3] = alpha_bg

    # Replace pixels where drawings are dark enough, set alpha=1:
    rr, cc = np.unravel_index(np.where(all_imgs.ravel() < thresh_fg)[0],
                              all_imgs.shape)
    for channel in range(3):
        img_arr[rr, cc, channel] = all_imgs[rr, cc]
    img_arr[rr, cc, 3] = 1

    ax.imshow(img_arr, cmap='gray')

    if show_fovea:
        fovea = fovea = dva2out([0, 0])[0]
        ax.scatter(fovea[0], fovea[1], s=100,
                   marker='s', c='w', edgecolors='k')
