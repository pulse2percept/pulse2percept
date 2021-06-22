"""`PhotovoltaicPixel`, `PRIMA`, `PRIMA75`, `PRIMA55`, `PRIMA40`"""

from matplotlib.patches import Circle, RegularPolygon

import numpy as np
# Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working:
from collections.abc import Sequence

from .base import ProsthesisSystem
from .electrodes import HexElectrode
from .electrode_arrays import ElectrodeGrid


class PhotovoltaicPixel(HexElectrode):
    """Photovoltaic pixel

    .. versionadded:: 0.7

    Parameters
    ----------
    x/y/z : double
        3D location of the electrode.
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the nasal retina.
        Positive ``y`` values move the electrode into the superior retina.
        Positive ``z`` values move the electrode away from the retina into the
        vitreous humor (sometimes called electrode-retina distance).
    r : double
        Disk radius in the x,y plane
    a : double
        Length of line drawn from the center of the hexagon to the midpoint of
        one of its sides.
    activated : bool
        To deactivate, set to ``False``. Deactivated electrodes cannot receive
        stimuli.

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('r', 'a')

    def __init__(self, x, y, z, r, a, name=None, activated=True):
        super(PhotovoltaicPixel, self).__init__(x, y, z, a, name=name,
                                                activated=activated)
        if isinstance(r, (Sequence, np.ndarray)):
            raise TypeError("Radius of the active electrode must be a scalar.")
        if r <= 0:
            raise ValueError("Radius of the active electrode must be > 0, not "
                             "%f." % r)
        self.r = r
        # Plot two objects: hex honeycomb and circular active electrode
        self.plot_patch = [RegularPolygon, Circle]
        self.plot_kwargs = [{'radius': a, 'numVertices': 6, 'alpha': 0.2,
                             'orientation': np.radians(30),
                             'fc': 'k', 'ec': 'k'},
                            {'radius': r, 'linewidth': 0, 'color': 'k',
                             'alpha': 0.5}]
        self.plot_deactivated_kwargs = [{'radius': a, 'numVertices': 6,
                                         'orientation': np.radians(30),
                                         'fc': 'k', 'ec': 'k', 'alpha': 0.1},
                                        {'radius': r, 'linewidth': 0,
                                         'color': 'k', 'alpha': 0.2}]

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'r': self.r, 'a': self.a})
        return params

    def electric_potential(self, x, y, z, v0):
        raise NotImplementedError


class PRIMA(ProsthesisSystem):
    """Create a PRIMA-100 array on the retina

    This class creates a PRIMA array with 378 photovoltaic pixels (each
    100um in diameter) as used in the clinical trial [Palanker2020]_, and
    places it in the subretinal space such that the center of the array is
    located at 3D location (x,y,z), given in microns, and the array is rotated
    by rotation angle ``rot``, given in degrees.

    The device consists of 378 85um-wide pixels separated by 15um trenches,
    arranged in a 2-mm wide hexagonal pattern.

    This corresponds to a 100um pitch, with adjacent rows separated by 87um.
    The active electrode is a disk with 28um diameter.

    .. versionadded:: 0.7

    Parameters
    ----------
    x/y/z : double
        3D location of the center of the electrode array.
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the nasal retina.
        Positive ``y`` values move the electrode into the superior retina.
        Positive ``z`` values move the electrode away from the retina into the
        vitreous humor (sometimes called electrode-retina distance).
        ``z`` can either be a list with 378 entries or a scalar that is applied
        to all electrodes.
    rot : float, optional
        Rotation angle of the array (deg). Positive values denote
        counter-clock-wise (CCW) rotations in the retinal coordinate
        system.
    eye : {'RE', 'LE'}, optional
        Eye in which array is implanted.
    preprocess : bool or callable, optional
        Either True/False to indicate whether to execute the implant's default
        preprocessing method whenever a new stimulus is assigned, or a custom
        function (callable).
    safe_mode : bool, optional
        If safe mode is enabled, only charge-balanced stimuli are allowed.

    Notes
    -----
    *  The diameter of the active electrode and the trench width were estimated
       from Fig.1 in [Palanker2020]_.

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape', 'spacing', 'trench')

    def __init__(self, x=0, y=0, z=-100, rot=0, eye='RE', stim=None,
                 preprocess=False, safe_mode=False):
        # 85 um pixels with 15 um trenches, 28 um active electrode:
        self.trench = 15  # um
        self.spacing = 100  # um
        elec_radius = 14  # um
        # Roughly a 19x22 grid, but edges are trimmed off:
        self.shape = (19, 22)
        self.eye = eye
        self.preprocess = preprocess
        self.safe_mode = safe_mode

        # The user might provide a list of z values for each of the
        # 378 resulting electrodes, not for the 22x19 initial ones.
        # In this case, don't pass it to ElectrodeGrid, but overwrite
        # the z values later:
        overwrite_z = isinstance(z, (list, np.ndarray))
        zarr = -100 if overwrite_z else z
        self.earray = ElectrodeGrid(self.shape, self.spacing, x=x, y=y,
                                    z=zarr, rot=rot, type='hex',
                                    orientation='vertical',
                                    etype=PhotovoltaicPixel, r=elec_radius,
                                    a=(self.spacing - self.trench) / 2)

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
            # Specify different height for every electrode in a list:
            z_arr = np.asarray(z).flatten()
            if z_arr.size != self.n_electrodes:
                raise ValueError("If `z` is a list, it must have %d entries, "
                                 "not %d." % (self.n_electrodes, z_arr.size))
            for elec, z_elec in zip(self.earray.electrode_objects, z):
                elec.z = z_elec

        # Beware of race condition: Stim must be set last, because it requires
        # indexing into self.electrodes:
        self.stim = stim


class PRIMA75(ProsthesisSystem):
    """Create a PRIMA-75 array on the retina

    This class creates a PRIMA array with 142 photovoltaic pixels (each 75um
    in diameter) as described in [Lorach2015]_, and places it in the subretinal
    space, such that that the center of the array is located at 3D location
    (x,y,z), given in microns, and the array is rotated by rotation angle
    ``rot``, given in degrees.

    The device consists of 142 70um-wide pixels separated by 5um trenches,
    arranged in a 1-mm wide hexagonal pattern.

    This corresponds to a 75um pitch, with adjacent rows separated by 65um.
    The active electrode is a disk with 20um diameter.

    .. versionadded:: 0.7

    Parameters
    ----------
    x/y/z : double
        3D location of the center of the electrode array.
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the nasal retina.
        Positive ``y`` values move the electrode into the superior retina.
        Positive ``z`` values move the electrode away from the retina into the
        vitreous humor (sometimes called electrode-retina distance).
        ``z`` can either be a list with 142 entries or a scalar that is applied
        to all electrodes.
    rot : float, optional
        Rotation angle of the array (deg). Positive values denote
        counter-clock-wise (CCW) rotations in the retinal coordinate
        system.
    eye : {'RE', 'LE'}, optional
        Eye in which array is implanted.
    preprocess : bool or callable, optional
        Either True/False to indicate whether to execute the implant's default
        preprocessing method whenever a new stimulus is assigned, or a custom
        function (callable).
    safe_mode : bool, optional
        If safe mode is enabled, only charge-balanced stimuli are allowed.

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape', 'spacing', 'trench')

    def __init__(self, x=0, y=0, z=-100, rot=0, eye='RE', stim=None,
                 preprocess=False, safe_mode=False):
        # 70 um pixels with 5 um trenches, 20 um active electrode:
        self.spacing = 75  # um
        self.trench = 5  # um
        elec_radius = 10  # um
        # Roughly a 12x15 grid, but edges are trimmed off:
        self.shape = (12, 15)
        self.eye = eye
        self.preprocess = preprocess
        self.safe_mode = safe_mode

        # The user might provide a list of z values for each of the
        # 378 resulting electrodes, not for the 22x19 initial ones.
        # In this case, don't pass it to ElectrodeGrid, but overwrite
        # the z values later:
        overwrite_z = isinstance(z, (list, np.ndarray))
        zarr = -100 if overwrite_z else z
        self.earray = ElectrodeGrid(self.shape, self.spacing, x=x, y=y,
                                    z=zarr, rot=rot, type='hex',
                                    orientation='vertical',
                                    etype=PhotovoltaicPixel, r=elec_radius,
                                    a=(self.spacing - self.trench) / 2)

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
            # Specify different height for every electrode in a list:
            z_arr = np.asarray(z).flatten()
            if z_arr.size != self.n_electrodes:
                raise ValueError("If `z` is a list, it must have %d entries, "
                                 "not %d." % (self.n_electrodes, z_arr.size))
            for elec, z_elec in zip(self.earray.electrode_objects, z):
                elec.z = z_elec

        # Beware of race condition: Stim must be set last, because it requires
        # indexing into self.electrodes:
        self.stim = stim


class PRIMA55(ProsthesisSystem):
    """Create a PRIMA-55 array on the retina

    This class creates a PRIMA array with 273 photovoltaic pixels (each 55um
    in diameter), and places it in the subretinal space, such that that the
    center of the array is located at 3D location (x,y,z), given in microns,
    and the array is rotated by rotation angle ``rot``, given in degrees.

    The device consists of 273 50um-wide pixels separated by 5um trenches,
    arranged in a 1-mm wide hexagonal pattern.

    This corresponds to a 55um pitch, with adjacent rows separated by 48um.
    The active electrode is a disk with 16um diameter.

    .. warning ::

        The exact shape of the device has not been published yet. We assume the
        array fits on a circular 1mm-diameter substrate, which leaves us with
        273 electrodes.

    .. versionadded:: 0.7

    Parameters
    ----------
    x/y/z : double
        3D location of the center of the electrode array.
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the nasal retina.
        Positive ``y`` values move the electrode into the superior retina.
        Positive ``z`` values move the electrode away from the retina into the
        vitreous humor (sometimes called electrode-retina distance).
        ``z`` can either be a list with 378 entries or a scalar that is applied
        to all electrodes.
    rot : float, optional
        Rotation angle of the array (deg). Positive values denote
        counter-clock-wise (CCW) rotations in the retinal coordinate
        system.
    eye : {'RE', 'LE'}, optional
        Eye in which array is implanted.
    preprocess : bool or callable, optional
        Either True/False to indicate whether to execute the implant's default
        preprocessing method whenever a new stimulus is assigned, or a custom
        function (callable).
    safe_mode : bool, optional
        If safe mode is enabled, only charge-balanced stimuli are allowed.

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape', 'spacing', 'trench')

    def __init__(self, x=0, y=0, z=-100, rot=0, eye='RE', stim=None,
                 preprocess=False, safe_mode=False):
        # 50 um pixels with 5 um trenches, 16 um active electrode:
        self.spacing = 55  # um
        self.trench = 5
        elec_radius = 8  # um
        # Roughly a 18x21 grid, but edges are trimmed off:
        self.shape = (18, 21)
        self.eye = eye
        self.preprocess = preprocess
        self.safe_mode = safe_mode

        # The user might provide a list of z values for each of the
        # 378 resulting electrodes, not for the 22x19 initial ones.
        # In this case, don't pass it to ElectrodeGrid, but overwrite
        # the z values later:
        overwrite_z = isinstance(z, (list, np.ndarray))
        zarr = -100 if overwrite_z else z
        self.earray = ElectrodeGrid(self.shape, self.spacing, x=x, y=y,
                                    z=zarr, rot=rot, type='hex',
                                    orientation='vertical',
                                    etype=PhotovoltaicPixel, r=elec_radius,
                                    a=(self.spacing - self.trench) / 2)

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
            # Specify different height for every electrode in a list:
            z_arr = np.asarray(z).flatten()
            if z_arr.size != self.n_electrodes:
                raise ValueError("If `z` is a list, it must have %d entries, "
                                 "not %d." % (self.n_electrodes, z_arr.size))
            for elec, z_elec in zip(self.earray.electrode_objects, z):
                elec.z = z_elec

        # Beware of race condition: Stim must be set last, because it requires
        # indexing into self.electrodes:
        self.stim = stim


class PRIMA40(ProsthesisSystem):
    """Create a PRIMA-40 array on the retina

    This class creates a PRIMA array with 532 photovoltaic pixels (each 40um
    in diameter), and places it in the subretinal space, such that that the
    center of the array is located at 3D location (x,y,z), given in microns,
    and the array is rotated by rotation angle ``rot``, given in degrees.

    The device consists of 532 35um-wide pixels separated by 5um trenches,
    arranged in a 1-mm wide hexagonal pattern.

    This corresponds to a 40um pitch, with adjacent rows separated by 48um.
    The active electrode is a disk with 16um diameter.

    .. important ::

        The exact shape of the device has not been published yet. We assume the
        array fits on a circular 1mm-diameter substrate, which leaves us with
        532 electrodes.

    .. versionadded:: 0.7

    Parameters
    ----------
    x/y/z : double
        3D location of the center of the electrode array.
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the nasal retina.
        Positive ``y`` values move the electrode into the superior retina.
        Positive ``z`` values move the electrode away from the retina into the
        vitreous humor (sometimes called electrode-retina distance).
        ``z`` can either be a list with 532 entries or a scalar that is applied
        to all electrodes.
    rot : float, optional
        Rotation angle of the array (deg). Positive values denote
        counter-clock-wise (CCW) rotations in the retinal coordinate
        system.
    eye : {'LE', 'RE'}, optional
        Eye in which array is implanted.
    preprocess : bool or callable, optional
        Either True/False to indicate whether to execute the implant's default
        preprocessing method whenever a new stimulus is assigned, or a custom
        function (callable).
    safe_mode : bool, optional
        If safe mode is enabled, only charge-balanced stimuli are allowed.

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape', 'spacing', 'trench')

    def __init__(self, x=0, y=0, z=-100, rot=0, eye='RE', stim=None,
                 preprocess=False, safe_mode=False):
        # 35 um pixels with 5 um trenches, 16 um active electrode:
        self.spacing = 40  # um
        self.trench = 5  # um
        elec_radius = 8  # um
        # Roughly a 25x28 grid, but edges are trimmed off:
        self.shape = (25, 28)
        self.eye = eye
        self.preprocess = preprocess
        self.safe_mode = safe_mode

        # The user might provide a list of z values for each of the
        # 378 resulting electrodes, not for the 22x19 initial ones.
        # In this case, don't pass it to ElectrodeGrid, but overwrite
        # the z values later:
        overwrite_z = isinstance(z, (list, np.ndarray))
        zarr = -100 if overwrite_z else z
        self.earray = ElectrodeGrid(self.shape, self.spacing, x=x, y=y,
                                    z=zarr, rot=rot, type='hex',
                                    orientation='vertical',
                                    etype=PhotovoltaicPixel, r=elec_radius,
                                    a=(self.spacing - self.trench) / 2)

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
            # Specify different height for every electrode in a list:
            z_arr = np.asarray(z).flatten()
            if z_arr.size != self.n_electrodes:
                raise ValueError("If `z` is a list, it must have %d entries, "
                                 "not %d." % (self.n_electrodes, z_arr.size))
            for elec, z_elec in zip(self.earray.electrode_objects, z):
                elec.z = z_elec

        # Beware of race condition: Stim must be set last, because it requires
        # indexing into self.electrodes:
        self.stim = stim
