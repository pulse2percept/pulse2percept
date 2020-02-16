"""`ArgusI`, `ArgusII`"""
import numpy as np
import collections as coll
from .base import (DiskElectrode, ElectrodeArray, ElectrodeGrid,
                   ProsthesisSystem)


class ArgusI(ProsthesisSystem):
    """Create an ArgusI array on the retina

    This function creates an ArgusI array and places it on the retina
    such that the center of the array is located at 3D location (x,y,z),
    given in microns, and the array is rotated by rotation angle ``rot``,
    given in radians.

    The array is oriented in the visual field as shown in Fig. 1 of
    Horsager et al. (2009); that is, if placed in (0,0), the top two
    rows will lie in the lower retina (upper visual field):

    .. raw:: html

        <pre>
          -y      A1 B1 C1 D1                     260 520 260 520
          ^       A2 B2 C2 D2   where electrode   520 260 520 260
          |       A3 B3 C3 D3   diameters are:    260 520 260 520
          -->x    A4 B4 C4 D4                     520 260 520 260
        </pre>

    Electrode order is: A1, B1, C1, D1, A2, B2, ..., D4.

    If ``use_legacy_names`` is True, electrode order is: L6, L2, M8, M4, ...

    An electrode can be addressed by name, row/column index, or integer index
    (into the flattened array).

    .. note::

        Column order is reversed in a left-eye implant.

    Parameters
    ----------
    x : float, optional, default: 0
        x coordinate of the array center (um)
    y : float, optional, default: 0
        y coordinate of the array center (um)
    z : float || array_like, optional, default: 0
        Distance of the array to the retinal surface (um). Either a list
        with 16 entries or a scalar.
    rot : float, optional, default: 0
        Rotation angle of the array (rad). Positive values denote
        counter-clock-wise (CCW) rotations in the retinal coordinate
        system.
    eye : {'LE', 'RE'}, optional, default: 'RE'
        Eye in which array is implanted.

    Examples
    --------
    Create an ArgusI array centered on the fovea, at 100um distance from
    the retina:

    >>> from pulse2percept.implants import ArgusI
    >>> ArgusI(x=0, y=0, z=100, rot=0)  # doctest: +NORMALIZE_WHITESPACE
    ArgusI(earray=pulse2percept.implants.base.ElectrodeGrid, eye='RE',
           shape=(4, 4), stim=None)

    Get access to electrode 'B1', either by name or by row/column index:

    >>> argus = ArgusI(x=0, y=0, z=100, rot=0)
    >>> argus['B1']
    DiskElectrode(r=260.0, x=-400.0, y=-1200.0, z=100.0)
    >>> argus[0, 1]
    DiskElectrode(r=260.0, x=-400.0, y=-1200.0, z=100.0)

    """

    def __init__(self, x=0, y=0, z=0, rot=0, eye='RE', stim=None,
                 use_legacy_names=False):
        # Argus I is a 4x4 grid of electrodes with 200um in diamater, spaced
        # 525um apart, with rows labeled alphabetically and columsn
        # numerically:
        self.eye = eye
        self.shape = (4, 4)
        r_arr = np.array([260, 520, 260, 520]) / 2.0
        r_arr = np.concatenate((r_arr, r_arr[::-1], r_arr, r_arr[::-1]),
                               axis=0)
        spacing = 800.0

        # In older papers, Argus I electrodes go by L and M:
        self.old_names = names = ['L6', 'L2', 'M8', 'M4',
                                  'L5', 'L1', 'M7', 'M3',
                                  'L8', 'L4', 'M6', 'M2',
                                  'L7', 'L3', 'M5', 'M1']
        names = self.old_names if use_legacy_names else ('1', 'A')
        self.earray = ElectrodeGrid(self.shape, spacing, x=x, y=y, z=z,
                                    rot=rot, etype=DiskElectrode, r=r_arr,
                                    names=names)

        # Set stimulus if available:
        self.stim = stim

        # Set left/right eye:
        if not isinstance(eye, str):
            raise TypeError("'eye' must be a string, either 'LE' or 'RE'.")
        if eye != 'LE' and eye != 'RE':
            raise ValueError("'eye' must be either 'LE' or 'RE'.")
        self.eye = eye
        # Unfortunately, in the left eye the labeling of columns is reversed...
        if eye == 'LE':
            # FIXME: Would be better to have more flexibility in the naming
            # convention. This is a quick-and-dirty fix:
            names = list(self.earray.keys())
            objects = list(self.earray.values())
            names = np.array(names).reshape(self.earray.shape)
            # Reverse column names:
            for row in range(self.earray.shape[0]):
                names[row] = names[row][::-1]
            # Build a new ordered dict:
            electrodes = coll.OrderedDict([])
            for name, obj in zip(names.ravel(), objects):
                electrodes.update({name: obj})
            # Assign the new ordered dict to earray:
            self.earray.electrodes = electrodes

    def get_params(self):
        params = super().get_params()
        params.update({'shape': self.shape})
        return params


class ArgusII(ProsthesisSystem):
    """Create an ArgusII array on the retina

    This function creates an ArgusII array and places it on the retina
    such that the center of the array is located at (x,y,z), given in
    microns, and the array is rotated by rotation angle ``rot``, given in
    radians.

    The array is oriented upright in the visual field, such that an
    array with center (0,0) has the top three rows lie in the lower
    retina (upper visual field), as shown below:

    .. raw:: html

        <pre>
                  A1 A2 A3 A4 A5 A6 A7 A8 A9 A10
          -y      B1 B2 B3 B4 B5 B6 B7 B8 B9 B10
          ^       C1 C2 C3 C4 C5 C6 C7 C8 C9 C10
          |       D1 D2 D3 D4 D5 D6 D7 D8 D9 D10
          -->x    E1 E2 E3 E4 E5 E6 E7 E8 E9 E10
                  F1 F2 F3 F4 F5 F6 F7 F8 F9 F10
        </pre>

    Electrode order is: A1, A2, ..., A10, B1, B2, ..., F10.

    An electrode can be addressed by name, row/column index, or integer index
    (into the flattened array).

    .. note::

        Column order is reversed in a left-eye implant.

    Parameters
    ----------
    x : float
        x coordinate of the array center (um)
    y : float
        y coordinate of the array center (um)
    z: float || array_like
        Distance of the array to the retinal surface (um). Either a list
        with 60 entries or a scalar.
    rot : float
        Rotation angle of the array (rad). Positive values denote
        counter-clock-wise (CCW) rotations in the retinal coordinate
        system.
    eye : {'LE', 'RE'}, optional, default: 'RE'
        Eye in which array is implanted.

    Examples
    --------
    Create an ArgusII array centered on the fovea, at 100um distance from
    the retina:

    >>> from pulse2percept.implants import ArgusII
    >>> ArgusII(x=0, y=0, z=100, rot=0)  # doctest: +NORMALIZE_WHITESPACE
    ArgusII(earray=pulse2percept.implants.base.ElectrodeGrid, eye='RE',
            shape=(6, 10), stim=None)

    Get access to electrode 'E7', either by name or by row/column index:

    >>> argus = ArgusII(x=0, y=0, z=100, rot=0)
    >>> argus['E7']
    DiskElectrode(r=100.0, x=787.5, y=787.5, z=100.0)
    >>> argus[4, 6]
    DiskElectrode(r=100.0, x=787.5, y=787.5, z=100.0)

    """

    def __init__(self, x=0, y=0, z=0, rot=0, eye='RE', stim=None):
        # Argus II is a 6x10 grid of electrodes with 200um in diamater, spaced
        # 525um apart, with rows labeled alphabetically and columsn
        # numerically:
        self.shape = (6, 10)
        r = 100.0
        spacing = 525.0
        names = ('A', '1')
        self.earray = ElectrodeGrid(self.shape, spacing, x=x, y=y, z=z, r=r,
                                    rot=rot, names=names, etype=DiskElectrode)
        self.shape = self.earray.shape

        # Set stimulus if available:
        self.stim = stim

        # Set left/right eye:
        if not isinstance(eye, str):
            raise TypeError("'eye' must be a string, either 'LE' or 'RE'.")
        if eye != 'LE' and eye != 'RE':
            raise ValueError("'eye' must be either 'LE' or 'RE'.")
        self.eye = eye
        # Unfortunately, in the left eye the labeling of columns is reversed...
        if eye == 'LE':
            # FIXME: Would be better to have more flexibility in the naming
            # convention. This is a quick-and-dirty fix:
            names = list(self.earray.keys())
            objects = list(self.earray.values())
            names = np.array(names).reshape(self.earray.shape)
            # Reverse column names:
            for row in range(self.earray.shape[0]):
                names[row] = names[row][::-1]
            # Build a new ordered dict:
            electrodes = coll.OrderedDict([])
            for name, obj in zip(names.ravel(), objects):
                electrodes.update({name: obj})
            # Assign the new ordered dict to earray:
            self.earray.electrodes = electrodes

    def get_params(self):
        params = super().get_params()
        params.update({'shape': self.shape})
        return params
