"""`PRIMA`"""

# https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python
from sys import platform
import matplotlib as mpl
if platform == "darwin":  # OS X
    mpl.use('TkAgg')

import numpy as np
from .base import ElectrodeGrid, ProsthesisSystem, DiskElectrode
from collections import OrderedDict


class PRIMA(ProsthesisSystem):
    """Create a PRIMA array on retina

    This function creates a PRIMA array and places it on the retina
    such that the center of the array is located at 3D location (x,y,z),
    given in microns, and the array is rotated by rotation angle ``rot``,
    given in radians.


    Parameters
    ----------
    x : float, optional, default: 0
        x coordinate of the array center (um)
    y : float, optional: default: 0
        y coordinate of the array center (um)
    z: float || array_like, optional, default: 0
        Distance of the array to the retinal surface (um). Either a list
        with 378 entries or a scalar.
    rot : float, optional, default: 0
        Rotation angle of the array (rad). Positive values denote
        counter-clock-wise (CCW) rotations in the retinal coordinate
        system.
    eye : {'LE', 'RE'}, optional, default: 'RE'
        Eye in which array is implanted.

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape',)

    def __init__(self, x=0, y=0, z=0, rot=0, eye='RE', stim=None):
        self.shape = (19, 22) # total number of columns is 22
                              # maximum number of electrodes of each row is 19
        self.eye= eye
        elec_radius = 10 # um
        e_spacing = 75  # um
        self.earray = ElectrodeGrid(self.shape, e_spacing, x=x, y=y, z=z,
                                    rot=rot, type='hex', orientation='vertical',etype=DiskElectrode,
                                    r=elec_radius)
        # Set stimulus if available:
        self.stim = stim
        
        # remove extra electrodes to fit the actual implant
        extra_elecs = ['A1','A2','A3','A4','A14','A16','A17',
                       'A18','A19','A20','A21','A22','B1',
                       'B2','B18','B19','B20','B21','B22',
                       'C1','C20','C21','C22','D22','E22','P1',
                       'Q1','Q22','R1','R2','R21','R22','S1',
                       'S2','S3','S5','S19','S20','S21','S22']

        for elec in extra_elecs:
            self.earray.remove_electrode(elec)
        
        # rename all electrodes
        idx_col = 0
        rows, cols = self.shape
        idx_update_row = chr(ord('A') + rows - 1) # start from the last electrode
        idx_prev_row = 'A'
        orig_earray = self.earray.electrodes
        new_earray = OrderedDict()
        for name in orig_earray:
            if name[0] == idx_prev_row:
                idx_col += 1
            else:
                idx_prev_row = chr(ord(idx_prev_row) + 1)
                idx_update_row = chr(ord(idx_update_row) - 1)
                idx_col = 1
            new_name = idx_update_row + str(idx_col)
            new_earray.update({new_name: orig_earray[name]})
        self.earray.electrodes = new_earray
        
    def plot(self, ax=None, annotate=False, xlim=None, ylim=None):
        """Plot the PRIMA implant
        
        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot, optional, default: None
            A Matplotlib axes object. If None given, a new one will be created.
        annotate : bool, optional, default: False
            Flag whether to label electrodes in the implant.
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

        if ax is None:
            _, ax = plt.subplots(figsize=(15, 8))

        for name, el in self.items():
            # Hexagonal return electrode:
            honeycomb = RegularPolygon((el.x, el.y), numVertices=6, radius=35, 
                                       orientation=np.radians(30), facecolor='k',
                                       alpha=0.2, edgecolor='k', zorder=1)
            ax.add_patch(honeycomb)

            # Circular center electrode:
            circle = Circle((el.x, el.y), radius=el.r, linewidth=0, color='k', 
                            alpha=0.5, zorder=2)
            ax.add_patch(circle)

            if annotate:
                ax.text(el.x, el.y, name, ha='center', va='center',
                                color='black', size='large',
                                bbox={'boxstyle': 'square,pad=-0.2', 'ec': 'none',
                                      'fc': (1, 1, 1, 0.7)},
                                zorder=3)

        # Determine xlim, ylim: Allow for some padding `pad` and round to the
        # nearest `step`:
        pad = 100
        step = 200
        xlim, ylim = None, None
        if xlim is None:
            xmin = np.floor(np.min([el.x - pad for el in self.values()]) / step)
            xmax = np.ceil(np.max([el.x + pad for el in self.values()]) / step)
            xlim = (step * xmin, step * xmax)
        if ylim is None:
            ymin = np.floor(np.min([el.y - pad for el in self.values()]) / step)
            ymax = np.ceil(np.max([el.y + pad for el in self.values()]) / step)
            ylim = (step * ymin, step * ymax)
        ax.set_xlim(xlim)
        ax.set_xticks(np.linspace(*xlim, num=5))
        ax.set_xlabel('x (microns)')
        ax.set_ylim(ylim)
        ax.set_yticks(np.linspace(*ylim, num=5))
        ax.set_ylabel('y (microns)')
        ax.set_aspect('equal')

        return ax
