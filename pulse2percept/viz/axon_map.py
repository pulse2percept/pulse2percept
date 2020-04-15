"""`plot_axon_map`, `plot_implant_on_axon_map`"""
# https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python
from sys import platform
import matplotlib as mpl
if platform == "darwin":  # OS X
    mpl.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import logging

from ..implants import ProsthesisSystem
from ..utils import parfor
from ..models import AxonMapSpatial


def plot_axon_map(eye='RE', loc_od=(15.5, 1.5), n_bundles=100, ax=None,
                  upside_down=False, annotate_quadrants=True):
    """Plot an axon map

    This function generates an axon map for a left/right eye and a given
    optic disc location.

    Parameters
    ----------
    eye : str
        Either 'LE' for left eye or 'RE' for right eye
    loc_od : (x_od, y_od), optional, default: (15.5, 1.5)
        Location of the optic disc center (deg).
    n_bundles : int, optional, default: 100
        Number of nerve fiber bundles to plot.
    ax : matplotlib.axes._subplots.AxesSubplot, optional, default: None
        A Matplotlib axes object. If None given, a new one will be created.
    upside_down : bool, optional, default: False
        Flag whether to plot the retina upside-down, such that the upper
        half of the plot corresponds to the upper visual field. In general,
        inferior retina == upper visual field (and superior == lower).
    annotate_quadrants : bool, optional, default: True
        Flag whether to annotate the four retinal quadrants
        (inferior/superior x temporal/nasal).
    """
    loc_od = np.asarray(loc_od)
    if len(loc_od) != 2:
        raise ValueError("'loc_od' must specify the x and y coordinates of "
                         "the optic disc.")
    if eye not in ['LE', 'RE']:
        raise ValueError("'eye' must be either 'LE' or 'RE', not %s." % eye)
    if n_bundles < 1:
        raise ValueError('Number of nerve fiber bundles must be >= 1.')

    # Make sure x-coord of optic disc has the correct sign for LE/RE:
    if (eye == 'RE' and loc_od[0] <= 0 or eye == 'LE' and loc_od[0] > 0):
        logstr = ("For eye==%s, expected opposite sign of x-coordinate of "
                  "the optic disc; changing %.2f to %.2f" % (eye, loc_od[0],
                                                             -loc_od[0]))
        logging.getLogger(__name__).info(logstr)
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
    axon_map = AxonMapSpatial(n_axons=n_bundles, loc_od_x=loc_od[0],
                              loc_od_y=loc_od[1], eye=eye)
    axon_bundles = axon_map.grow_axon_bundles()
    for bundle in axon_bundles:
        ax.plot(bundle[:, 0], bundle[:, 1], c=(0.5, 1.0, 0.5))

    # Show circular optic disc:
    ax.add_patch(patches.Circle(axon_map.dva2ret(loc_od), radius=900, alpha=1,
                                color='black', zorder=10))

    xmin, xmax, ymin, ymax = axon_map.dva2ret([-20, 20, -15, 15])
    ax.set_aspect('equal')
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel('x (microns)')
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('y (microns)')
    ax.set_title('Axon map: %s, %s' % (eye, loc_od))
    ax.grid(False)

    # Annotate the four retinal quadrants near the corners of the plot:
    # superior/inferior x temporal/nasal
    if annotate_quadrants:
        if upside_down:
            topbottom = ['bottom', 'top']
        else:
            topbottom = ['top', 'bottom']
        if eye == 'RE':
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


def plot_implant_on_axon_map(implant, loc_od=(15.5, 1.5), n_bundles=100,
                             ax=None, upside_down=False, annotate_implant=True,
                             annotate_quadrants=True):
    """Plot an implant on top of the axon map

    This function plots an electrode array on top of an axon map.

    Parameters
    ----------
    implant : p2p.implants.ProsthesisSystem
        A ProsthesisSystem object. If a stimulus is given, stimulating
        electrodes will be highlighted in yellow.
    loc_od : (x_od, y_od), optional, default: (15.5, 1.5)
        Location of the optic disc center (deg).
    n_bundles : int, optional, default: 100
        Number of nerve fiber bundles to plot.
    ax : matplotlib.axes._subplots.AxesSubplot, optional, default: None
        A Matplotlib axes object. If None given, a new one will be created.
    upside_down : bool, optional, default: False
        Flag whether to plot the retina upside-down, such that the upper
        half of the plot corresponds to the upper visual field. In general,
        inferior retina == upper visual field (and superior == lower).
    annotate_implant : bool, optional, default: True
        Flag whether to label electrodes in the implant.
    annotate_quadrants : bool, optional, default: True
        Flag whether to annotate the four retinal quadrants
        (inferior/superior x temporal/nasal).
    """
    if not isinstance(implant, ProsthesisSystem):
        e_s = "`implant` must be of type ProsthesisSystem"
        raise TypeError(e_s)

    fig, ax = plot_axon_map(eye=implant.eye, loc_od=loc_od, ax=ax,
                            n_bundles=n_bundles, upside_down=upside_down,
                            annotate_quadrants=annotate_quadrants)

    # Highlight location of stimulated electrodes:
    if implant.stim is not None:
        for e in implant.stim.electrodes:
            ax.plot(implant[e].x, implant[e].y, 'oy',
                    markersize=np.sqrt(implant[e].r) * 2)

    # Plot all electrodes and label them (optional):
    for name, el in implant.items():
        if annotate_implant:
            ax.text(el.x + 100, el.y + 50, name, color='white', size='x-large')
        ax.plot(el.x, el.y, 'ow', markersize=np.sqrt(el.r))

    ax.set_title(implant)

    return fig, ax
