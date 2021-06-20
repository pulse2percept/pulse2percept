"""`plot_axon_map`, `plot_implant_on_axon_map`"""
# https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python
from sys import platform
import matplotlib as mpl
if platform == "darwin":  # OS X
    mpl.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.collections import PatchCollection
import logging
from copy import deepcopy

from ..implants import ProsthesisSystem
from ..utils import deprecated
from ..utils.constants import ZORDER
from ..models import AxonMapSpatial


@deprecated(deprecated_version='0.7', removed_version='0.8')
def plot_axon_map(eye='RE', loc_od=(15.5, 1.5), n_bundles=100, ax=None,
                  upside_down=False, annotate=False, xlim=None, ylim=None):
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
    ax : ``matplotlib.axes.Axes``, optional, default: None
        A Matplotlib axes object. If None given, a new one will be created.
    upside_down : bool, optional, default: False
        Flag whether to plot the retina upside-down, such that the upper
        half of the plot corresponds to the upper visual field. In general,
        inferior retina == upper visual field (and superior == lower).
    annotate : bool, optional, default: True
        Flag whether to annotate the four retinal quadrants
        (inferior/superior x temporal/nasal).
    xlim: (xmin, xmax), optional, default: (-5000, 5000)
        Range of x coordinates to visualize. If None, the center 10 mm of the
        retina will be shown.
    ylim: (ymin, ymax), optional, default: (-4000, 4000)
        Range of y coordinates to visualize. If None, the center 8 mm of the
        retina will be shown.

    Returns
    -------
    ax : ``matplotlib.axes.Axes``
        Returns the axis object of the plot

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

    # No axes object given: create
    if ax is None:
        _, ax = plt.subplots(1, figsize=(10, 8))
    ax.set_facecolor('white')

    # Draw axon pathways:
    axon_map = AxonMapSpatial(n_axons=n_bundles, loc_od=loc_od, eye=eye)
    axon_bundles = axon_map.grow_axon_bundles()
    for bundle in axon_bundles:
        ax.plot(bundle[:, 0], bundle[:, 1], c=(0.6, 0.6, 0.6), linewidth=2,
                zorder=ZORDER['background'])

    # Show elliptic optic nerve head:
    ax.add_patch(Ellipse(axon_map.dva2ret(loc_od), width=1770, height=1880,
                         alpha=1, color='white',
                         zorder=ZORDER['background'] + 1))

    if xlim is not None:
        xmin, xmax = xlim
    else:
        xmin, xmax = -5000, 5000
    if ylim is not None:
        ymin, ymax = ylim
    else:
        ymin, ymax = -4000, 4000
    ax.set_aspect('equal')
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel('x (microns)')
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('y (microns)')
    ax.set_title('Axon map: %s, %s' % (eye, loc_od))
    ax.grid(False)

    # Annotate the four retinal quadrants near the corners of the plot:
    # superior/inferior x temporal/nasal
    if annotate:
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
                        bbox={'boxstyle': 'square,pad=-0.1', 'ec': 'none',
                              'fc': (1, 1, 1, 0.7)},
                        zorder=ZORDER['annotate'])

    # Need to flip y axis to have upper half == upper visual field
    if upside_down:
        ax.invert_yaxis()

    return ax


@deprecated(deprecated_version='0.7', removed_version='0.8')
def plot_implant_on_axon_map(implant, loc_od=(15.5, 1.5), n_bundles=100,
                             ax=None, upside_down=False,
                             annotate_implant=False, annotate_quadrants=True,
                             xlim=None, ylim=None):
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
    xlim : (xmin, xmax), optional, default: None
        Range of x values to plot. If None, the plot will be centered over the
        implant.
    ylim : (ymin, ymax), optional, default: None
        Range of y values to plot. If None, the plot will be centered over the
        implant.

    Returns
    -------
    ax : ``matplotlib.axes.Axes``
        Returns the axis object of the plot

    """
    if not isinstance(implant, ProsthesisSystem):
        e_s = "`implant` must be of type ProsthesisSystem"
        raise TypeError(e_s)

    pad = 1500
    prec = 500
    if xlim is None:
        xmin = np.ceil(np.min([el.x - pad
                               for el in implant.electrode_objects]) / prec)
        xmax = np.ceil(np.max([el.x + pad
                               for el in implant.electrode_objects]) / prec)
        xlim = (prec * xmin, prec * xmax)
    if ylim is None:
        ymin = np.ceil(np.min([el.y - pad
                               for el in implant.electrode_objects]) / prec)
        ymax = np.ceil(np.max([el.y + pad
                               for el in implant.electrode_objects]) / prec)
        ylim = (prec * ymin, prec * ymax)

    ax = plot_axon_map(eye=implant.eye, loc_od=loc_od, ax=ax,
                       n_bundles=n_bundles, upside_down=upside_down,
                       annotate=annotate_quadrants, xlim=xlim,
                       ylim=ylim)

    # Determine marker size for electrodes:
    radii = []
    for el in implant.electrode_objects:
        # Use electrode radius (if exists), else constant radius:
        if hasattr(el, 'r'):
            radii.append(el.r)
        else:
            radii.append(100)

    # Highlight location of stimulated electrodes:
    if implant.stim is not None:
        _stim = deepcopy(implant.stim)
        _stim.compress()
        circles = [Circle((implant[e].x, implant[e].y), color='red', alpha=1,
                          radius=1.7 * radii[i], linewidth=4,
                          zorder=ZORDER['foreground'])
                   for i, (e, r) in enumerate(zip(_stim.electrodes, radii))]
        ax.add_collection(PatchCollection(circles, match_original=True))

    # Plot all electrodes and label them (optional):
    circles = []
    for i, (name, el) in enumerate(implant.electrodes.items()):
        if annotate_implant:
            ax.text(el.x, el.y, name, horizontalalignment='center',
                    verticalalignment='center',
                    color='black', size='x-large',
                    bbox={'boxstyle': 'square,pad=-0.2', 'ec': 'none',
                          'fc': (1, 1, 1, 0.7)},
                    zorder=ZORDER['foreground'] + 2)
        circles.append(Circle((el.x, el.y), radius=radii[i],
                              edgecolor=(0.3, 0.3, 0.3, 1),
                              facecolor=(1, 1, 1, 0.7),
                              linewidth=4))
    ax.add_collection(PatchCollection(circles, match_original=True,
                                      zorder=ZORDER['foreground'] + 1))

    ax.set_title(implant)

    return ax
