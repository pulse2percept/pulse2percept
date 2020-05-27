"""`PRIMA`"""

# https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python
from sys import platform
import matplotlib as mpl
if platform == "darwin":  # OS X
    mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon

import numpy as np
from collections import OrderedDict
from .base import ElectrodeGrid, ProsthesisSystem, DiskElectrode


class PRIMAPlotMixin(object):

    # Frozen class: User cannot add more class attributes
    __slots__ = ()

    def plot(self, ax=None, annotate=False, upside_down=False, xlim=None,
             ylim=None):
        """Plot the PRIMA implant

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot, optional, default: None
            A Matplotlib axes object. If None given, a new one will be created.
        annotate : bool, optional, default: False
            Flag whether to label electrodes in the implant.
        upside_down : bool, optional, default: False
            Flag whether to plot the retina upside-down, such that the upper
            half of the plot corresponds to the upper visual field. In general,
            inferior retina == upper visual field (and superior == lower).
        xlim : (xmin, xmax), optional, default: None
            Range of x values to plot. If None, the plot will be centered over
            the implant.
        ylim : (ymin, ymax), optional, default: None
            Range of y values to plot. If None, the plot will be centered over
            the implant.

        Returns
        -------
        ax : ``matplotlib.axes.Axes``
            Returns the axis object of the plot
        """

        if ax is None:
            _, ax = plt.subplots(figsize=(15, 8))

        for name, el in self.items():
            # Hexagonal return electrode:
            honeycomb = RegularPolygon((el.x, el.y), numVertices=6,
                                       radius=(self.spacing -
                                               self.trench) // 2,
                                       orientation=np.radians(30),
                                       facecolor='k', alpha=0.2, edgecolor='k',
                                       zorder=1)
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
            xmin = np.floor(
                np.min([el.x - pad for el in self.values()]) / step)
            xmax = np.ceil(np.max([el.x + pad for el in self.values()]) / step)
            xlim = (step * xmin, step * xmax)
        if ylim is None:
            ymin = np.floor(
                np.min([el.y - pad for el in self.values()]) / step)
            ymax = np.ceil(np.max([el.y + pad for el in self.values()]) / step)
            ylim = (step * ymin, step * ymax)
        ax.set_xlim(xlim)
        ax.set_xticks(np.linspace(*xlim, num=5))
        ax.set_xlabel('x (microns)')
        ax.set_ylim(ylim)
        ax.set_yticks(np.linspace(*ylim, num=5))
        ax.set_ylabel('y (microns)')
        ax.set_aspect('equal')

        # Need to flip y axis to have upper half == upper visual field
        if upside_down:
            ax.invert_yaxis()

        return ax


class PRIMA(PRIMAPlotMixin, ProsthesisSystem):
    """Create a PRIMA-100 array on the retina

    This function creates a PRIMA array with 378 photovoltaic pixels (each
    100um in diameter) as used in the clinical trial [Palanker2020]_, and
    places it in the subretinal space such that the center of the array is
    located at 3D location (x,y,z), given in microns, and the array is rotated
    by rotation angle ``rot``, given in radians.

    The device consists of 378 85um-wide pixels separated by 15um trenches,
    arranged in a 2-mm wide hexagonal pattern.

    This corresponds to a 100um pitch, with adjacent rows separated by 87um.
    The active electrode is a disk with 28um diameter.

    Parameters
    ----------
    x : float, optional, default: 0
        x coordinate of the array center (um)
    y : float, optional: default: 0
        y coordinate of the array center (um)
    z: float || array_like, optional, default: -100
        Distance of the array to the retinal surface (um). Either a list
        with 378 entries or a scalar.
    rot : float, optional, default: 0
        Rotation angle of the array (rad). Positive values denote
        counter-clock-wise (CCW) rotations in the retinal coordinate
        system.
    eye : {'LE', 'RE'}, optional, default: 'RE'
        Eye in which array is implanted.

    Notes
    -----
    *  The diameter of the active electrode and the trench width were estimated
       from Fig.1 in [Palanker2020]_.

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape', 'spacing', 'trench')

    def __init__(self, x=0, y=0, z=-100, rot=0, eye='RE', stim=None):
        # 85 um pixels with 15 um trenches, 28 um active electrode:
        self.trench = 15  # um
        self.spacing = 100  # um
        elec_radius = 14  # um
        # Roughly a 19x22 grid, but edges are trimmed off:
        self.shape = (19, 22)
        self.eye = eye

        # The user might provide a list of z values for each of the
        # 378 resulting electrodes, not for the 22x19 initial ones.
        # In this case, don't pass it to ElectrodeGrid, but overwrite
        # the z values later:
        overwrite_z = isinstance(z, (list, np.ndarray))
        zarr = -100 if overwrite_z else z
        self.earray = ElectrodeGrid(self.shape, self.spacing, x=x, y=y,
                                    z=zarr, rot=rot, type='hex',
                                    orientation='vertical',
                                    etype=DiskElectrode, r=elec_radius)

        # Remove extra electrodes to fit the actual implant:
        extra_elecs = ['A1', 'A2', 'A3', 'A4', 'A14', 'A16', 'A17',
                       'A18', 'A19', 'A20', 'A21', 'A22', 'B1',
                       'B2', 'B18', 'B19', 'B20', 'B21', 'B22',
                       'C1', 'C20', 'C21', 'C22', 'D22', 'E22', 'P1',
                       'Q1', 'Q22', 'R1', 'R2', 'R21', 'R22', 'S1',
                       'S2', 'S3', 'S5', 'S19', 'S20', 'S21', 'S22']
        for elec in extra_elecs:
            self.earray.remove_electrode(elec)

        # Adjust the z values:
        if overwrite_z:
            for elec, z_elec in zip(self.earray.values(), z):
                elec.z = z_elec

        # Beware of race condition: Stim must be set last, because it requires
        # indexing into self.electrodes:
        self.stim = stim


class PRIMA75(PRIMAPlotMixin, ProsthesisSystem):
    """Create a PRIMA-75 array on the retina

    This function creates a PRIMA array with 142 photovoltaic pixels (each 75um
    in diameter) as described in [Lorach2015]_, and places it in the subretinal
    space, such that that the center of the array is located at 3D location
    (x,y,z), given in microns, and the array is rotated by rotation angle
    ``rot``, given in radians.

    The device consists of 142 70um-wide pixels separated by 5um trenches,
    arranged in a 1-mm wide hexagonal pattern.

    This corresponds to a 75um pitch, with adjacent rows separated by 65um.
    The active electrode is a disk with 20um diameter.

    Parameters
    ----------
    x : float, optional, default: 0
        x coordinate of the array center (um)
    y : float, optional: default: 0
        y coordinate of the array center (um)
    z: float || array_like, optional, default: -100
        Distance of the array to the retinal surface (um). Either a list
        with 142 entries or a scalar.
    rot : float, optional, default: 0
        Rotation angle of the array (rad). Positive values denote
        counter-clock-wise (CCW) rotations in the retinal coordinate
        system.
    eye : {'LE', 'RE'}, optional, default: 'RE'
        Eye in which array is implanted.

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape', 'spacing', 'trench')

    def __init__(self, x=0, y=0, z=-100, rot=0, eye='RE', stim=None):
        # 70 um pixels with 5 um trenches, 20 um active electrode:
        self.spacing = 75  # um
        self.trench = 5  # um
        elec_radius = 10  # um
        # Roughly a 12x15 grid, but edges are trimmed off:
        self.shape = (12, 15)
        self.eye = eye

        # The user might provide a list of z values for each of the
        # 378 resulting electrodes, not for the 22x19 initial ones.
        # In this case, don't pass it to ElectrodeGrid, but overwrite
        # the z values later:
        overwrite_z = isinstance(z, (list, np.ndarray))
        zarr = -100 if overwrite_z else z
        self.earray = ElectrodeGrid(self.shape, self.spacing, x=x, y=y,
                                    z=zarr, rot=rot, type='hex',
                                    orientation='vertical',
                                    etype=DiskElectrode, r=elec_radius)

        # Remove extra electrodes to fit the actual implant:
        extra_elecs = ['A1', 'B1', 'C1', 'D1', 'E1', 'I1', 'J1', 'K1', 'L1',
                       'A2', 'B2', 'C2', 'D2', 'K2', 'L2',
                       'A3', 'B3', 'L3',
                       'A4',
                       'A12',
                       'A13', 'K13', 'L13',
                       'A14', 'B14', 'C14', 'J14', 'K14', 'L14',
                       'A15', 'B15', 'C15', 'D15', 'H15', 'I15', 'J15', 'K15',
                       'L15']
        for elec in extra_elecs:
            self.earray.remove_electrode(elec)

        # Adjust the z values:
        if overwrite_z:
            for elec, z_elec in zip(self.earray.values(), z):
                elec.z = z_elec

        # Beware of race condition: Stim must be set last, because it requires
        # indexing into self.electrodes:
        self.stim = stim


class PRIMA55(PRIMAPlotMixin, ProsthesisSystem):
    """Create a PRIMA-55 array on the retina

    This function creates a PRIMA array with 273 photovoltaic pixels (each 55um
    in diameter) as described in [Lorach2015]_, and places it in the subretinal
    space, such that that the center of the array is located at 3D location
    (x,y,z), given in microns, and the array is rotated by rotation angle
    ``rot``, given in radians.

    The device consists of 273 50um-wide pixels separated by 5um trenches,
    arranged in a 1-mm wide hexagonal pattern.

    This corresponds to a 55um pitch, with adjacent rows separated by 48um.
    The active electrode is a disk with 16um diameter.

    .. warning ::

        The exact shape of the device has not been published yet. We assume the
        array fits on a circular 1mm-diameter substrate, which leaves us with
        273 electrodes.

    Parameters
    ----------
    x : float, optional, default: 0
        x coordinate of the array center (um)
    y : float, optional: default: 0
        y coordinate of the array center (um)
    z: float || array_like, optional, default: -100
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
    __slots__ = ('shape', 'spacing', 'trench')

    def __init__(self, x=0, y=0, z=-100, rot=0, eye='RE', stim=None):
        # 50 um pixels with 5 um trenches, 16 um active electrode:
        self.spacing = 55  # um
        self.trench = 5
        elec_radius = 8  # um
        # Roughly a 18x21 grid, but edges are trimmed off:
        self.shape = (18, 21)
        self.eye = eye

        # The user might provide a list of z values for each of the
        # 378 resulting electrodes, not for the 22x19 initial ones.
        # In this case, don't pass it to ElectrodeGrid, but overwrite
        # the z values later:
        overwrite_z = isinstance(z, (list, np.ndarray))
        zarr = -100 if overwrite_z else z
        self.earray = ElectrodeGrid(self.shape, self.spacing, x=x, y=y,
                                    z=zarr, rot=rot, type='hex',
                                    orientation='vertical',
                                    etype=DiskElectrode, r=elec_radius)

        # Note that the exact shape of this implant is not known. We remove
        # all electrodes that don't fit on a circular 1mm x 1mm substrate:
        extra_elec = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9',
                      'A10', 'A12', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19',
                      'A20', 'A21', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B16',
                      'B17', 'B18', 'B19', 'B20', 'B21', 'C1', 'C2', 'C3',
                      'C4', 'C18', 'C19', 'C20', 'C21', 'D1', 'D2', 'D3', 'D4',
                      'D20', 'D21', 'E1', 'E2', 'E20', 'E21', 'F1', 'F2',
                      'F21', 'G1', 'G2', 'G21', 'H1', 'I1', 'J1', 'K1', 'L1',
                      'L21', 'M1', 'M2', 'M21', 'N1', 'N2', 'N21', 'O1', 'O2',
                      'O3', 'O20', 'O21', 'P1', 'P2', 'P3', 'P4', 'P19', 'P20',
                      'P21', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q17', 'Q18', 'Q19',
                      'Q20', 'Q21', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7',
                      'R9', 'R13', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20',
                      'R21']
        for elec in extra_elec:
            self.earray.remove_electrode(elec)

        # Adjust the z values:
        if overwrite_z:
            for elec, z_elec in zip(self.earray.values(), z):
                elec.z = z_elec

        # Beware of race condition: Stim must be set last, because it requires
        # indexing into self.electrodes:
        self.stim = stim


class PRIMA40(PRIMAPlotMixin, ProsthesisSystem):
    """Create a PRIMA-40 array on the retina

    This function creates a PRIMA array with 532 photovoltaic pixels (each 55um
    in diameter) as described in [Lorach2015]_, and places it in the subretinal
    space, such that that the center of the array is located at 3D location
    (x,y,z), given in microns, and the array is rotated by rotation angle
    ``rot``, given in radians.

    The device consists of 532 35um-wide pixels separated by 5um trenches,
    arranged in a 1-mm wide hexagonal pattern.

    This corresponds to a 40um pitch, with adjacent rows separated by 48um.
    The active electrode is a disk with 16um diameter.

    .. important ::

        The exact shape of the device has not been published yet. We assume the
        array fits on a circular 1mm-diameter substrate, which leaves us with
        532 electrodes.

    Parameters
    ----------
    x : float, optional, default: 0
        x coordinate of the array center (um)
    y : float, optional: default: 0
        y coordinate of the array center (um)
    z: float || array_like, optional, default: -100
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
    __slots__ = ('shape', 'spacing', 'trench')

    def __init__(self, x=0, y=0, z=-100, rot=0, eye='RE', stim=None):
        # 35 um pixels with 5 um trenches, 16 um active electrode:
        self.spacing = 40  # um
        self.trench = 5  # um
        elec_radius = 8  # um
        # Roughly a 25x28 grid, but edges are trimmed off:
        self.shape = (25, 28)
        self.eye = eye

        # The user might provide a list of z values for each of the
        # 378 resulting electrodes, not for the 22x19 initial ones.
        # In this case, don't pass it to ElectrodeGrid, but overwrite
        # the z values later:
        overwrite_z = isinstance(z, (list, np.ndarray))
        zarr = -100 if overwrite_z else z
        self.earray = ElectrodeGrid(self.shape, self.spacing, x=x, y=y,
                                    z=zarr, rot=rot, type='hex',
                                    orientation='vertical',
                                    etype=DiskElectrode, r=elec_radius)

        # Note that the exact shape of this implant is not known. We remove
        # all electrodes that don't fit on a circular 1mm x 1mm substrate:
        extra_elec = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9',
                      'A10', 'A11', 'A12', 'A14', 'A16', 'A17', 'A18', 'A19',
                      'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27',
                      'A28', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
                      'B10', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26',
                      'B27', 'B28', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C22',
                      'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'D1', 'D2',
                      'D3', 'D4', 'D5', 'D24', 'D25', 'D26', 'D27', 'D28',
                      'E1', 'E2', 'E3', 'E4', 'E25', 'E26', 'E27', 'E28', 'F1',
                      'F2', 'F26', 'F27', 'F28', 'G1', 'G2', 'G27', 'G28',
                      'H1', 'H27', 'H28', 'I1', 'I28', 'J28', 'K28', 'P28',
                      'Q1', 'Q28', 'R1', 'R27', 'R28', 'S1', 'S27', 'S28',
                      'T1', 'T2', 'T27', 'T28', 'U1', 'U2', 'U3', 'U25', 'U26',
                      'U27', 'U28', 'V1', 'V2', 'V3', 'V4', 'V5', 'V25', 'V26',
                      'V27', 'V28', 'W1', 'W2', 'W3', 'W4', 'W5', 'W23', 'W24',
                      'W25', 'W26', 'W27', 'W28', 'X1', 'X2', 'X3', 'X4', 'X5',
                      'X6', 'X7', 'X21', 'X22', 'X23', 'X24', 'X25', 'X26',
                      'X27', 'X28', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7',
                      'Y8', 'Y9', 'Y10', 'Y11', 'Y17', 'Y19', 'Y20', 'Y21',
                      'Y22', 'Y23', 'Y24', 'Y25', 'Y26', 'Y27', 'Y28']
        for elec in extra_elec:
            self.earray.remove_electrode(elec)

        # Adjust the z values:
        if overwrite_z:
            for elec, z_elec in zip(self.earray.values(), z):
                elec.z = z_elec

        # Beware of race condition: Stim must be set last, because it requires
        # indexing into self.electrodes:
        self.stim = stim
