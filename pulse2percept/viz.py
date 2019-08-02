import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import logging

from pulse2percept.implants import ProsthesisSystem
from pulse2percept.utils import parfor
from pulse2percept.models import AxonMapModel, dva2ret


def plot_fundus(implant, ax=None, loc_od=(15.5, 1.5), n_bundles=100,
                upside_down=False, annot_array=True, annot_quadr=True):
    """Plot an implant on top of the axon map
    This function plots an electrode array on top of the axon map, akin to a
    fundus photograph.
    Parameters
    ----------
    implant : p2p.implants.ProsthesisSystem
        A ProsthesisSystem object
    ax : matplotlib.axes._subplots.AxesSubplot, optional, default: None
        A Matplotlib axes object. If None given, a new one will be created.
    loc_od : (x_od, y_od), optional, default: (15.5, 1.5)
        Location of the optic disc center (deg).
    n_bundles : int, optional, default: 100
        Number of nerve fiber bundles to plot.
    upside_down : bool, optional, default: False
        Flag whether to plot the retina upside-down, such that the upper
        half of the plot corresponds to the upper visual field. In general,
        inferior retina == upper visual field (and superior == lower).
    annot_array : bool, optional, default: True
        Flag whether to label electrodes and the tack.
    annot_quadr : bool, optional, default: True
        Flag whether to annotate the four retinal quadrants
        (inferior/superior x temporal/nasal).
    """
    if not isinstance(implant, ProsthesisSystem):
        e_s = "`implant` must be of type ProsthesisSystem"
        raise TypeError(e_s)
    if n_bundles < 1:
        raise ValueError('Number of nerve fiber bundles must be >= 1.')

    # Make sure x-coord of optic disc has the correct sign for LE/RE:
    if (implant.eye == 'RE' and loc_od[0] <= 0 or
            implant.eye == 'LE' and loc_od[0] > 0):
        logstr = ("For eye==%s, expected opposite sign of x-coordinate of "
                  "the optic disc; changing %.2f to %.2f" % (implant.eye,
                                                             loc_od[0],
                                                             -loc_od[0]))
        print(logstr)
        loc_od = (-loc_od[0], loc_od[1])
    if ax is None:
        # No axes object given: create
        fig, ax = plt.subplots(1, figsize=(10, 8))
    else:
        fig = ax.figure

    # Matplotlib<2 compatibility
    if hasattr(ax, 'set_facecolor'):
        ax.set_facecolor('black')
    elif hasattr(ax, 'set_axis_bgcolor'):
        ax.set_axis_bgcolor('black')

    # Draw axon pathways:
    axon_map = AxonMapModel(n_axons=n_bundles, loc_od_x=loc_od[0],
                            loc_od_y=loc_od[1])
    axon_bundles = axon_map.grow_axon_bundles()
    for bundle in axon_bundles:
        ax.plot(bundle[:, 0], bundle[:, 1], c=(0.5, 1.0, 0.5))

    # Plot all electrodes and label them (optional):
    for name, el in implant.items():
        if annot_array:
            ax.text(el.x + 100, el.y + 50, name, color='white', size='x-large')
        ax.plot(el.x, el.y, 'ow', markersize=np.sqrt(el.r))

    # Plot the location of the array's tack and annotate it (optional):
    if implant.tack:
        tx, ty = implant.tack
        ax.plot(tx, ty, 'ow')
        if annot_array:
            if upside_down:
                offset = 100
            else:
                offset = -100
            ax.text(tx, ty + offset, 'tack',
                    horizontalalignment='center',
                    verticalalignment='top',
                    color='white', size='large')

    # Show circular optic disc:
    ax.add_patch(patches.Circle(dva2ret(loc_od), radius=900, alpha=1,
                                color='black', zorder=10))

    xmin, xmax, ymin, ymax = dva2ret([-20, 20, -15, 15])
    ax.set_aspect('equal')
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel('x (microns)')
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('y (microns)')
    ax.set_title(implant)
    ax.grid(False)

    # Annotate the four retinal quadrants near the corners of the plot:
    # superior/inferior x temporal/nasal
    if annot_quadr:
        if upside_down:
            topbottom = ['bottom', 'top']
        else:
            topbottom = ['top', 'bottom']
        if implant.eye == 'RE':
            temporalnasal = ['temporal', 'nasal']
        else:
            temporalnasal = ['nasal', 'temporal']
        for yy, valign, si in zip([ymax, ymin], topbottom,
                                  ['superior', 'inferior']):
            for xx, halign, tn in zip([xmin, xmax], ['left', 'right'],
                                      temporalnasal):
                ax.text(xx, yy, si + ' ' + tn,
                        color='black', fontsize=14,
                        horizontalalignment=halign,
                        verticalalignment=valign,
                        backgroundcolor=(1, 1, 1, 0.8))

    # Need to flip y axis to have upper half == upper visual field
    if upside_down:
        ax.invert_yaxis()

    return fig, ax
