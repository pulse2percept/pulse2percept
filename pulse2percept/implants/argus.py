"""`ArgusI`, `ArgusII`"""
import numpy as np
from collections import OrderedDict

from .base import ProsthesisSystem
from .electrodes import DiskElectrode
from .electrode_arrays import ElectrodeGrid


class ArgusI(ProsthesisSystem):
    """Create an Argus I array on the retina

    This function creates an Argus I array and places it on the retina
    such that the center of the array is located at 3D location (x,y,z),
    given in microns, and the array is rotated by rotation angle ``rot``,
    given in degrees.

    Argus I is a modified cochlear implant containing 16 electrodes in a 4x4
    array with a center-to-center separation of 800 um, and two electrode
    diameters (250 um and 500 um) arranged in a checkerboard pattern
    [Yue2020]_.

    The array is oriented in the visual field as shown in Fig. 1 of
    [Horsager2009]_; that is, if placed in (0,0), the top two rows will lie in
    the lower retina (upper visual field):

    .. raw:: html

        <pre>
          -->x    A1 B1 C1 D1                     260 520 260 520
          |       A2 B2 C2 D2   where electrode   520 260 520 260
          v       A3 B3 C3 D3   diameters are:    260 520 260 520
           y      A4 B4 C4 D4                     520 260 520 260
        </pre>

    Electrode order is: A1, B1, C1, D1, A2, B2, ..., D4.

    If ``use_legacy_names`` is True, electrode order is: L6, L2, M8, M4, ...

    An electrode can be addressed by name, row/column index, or integer index
    (into the flattened array).

    .. note::

        Column order is reversed in a left-eye implant.

    Parameters
    ----------
    x/y/z : double
        3D location of the center of the electrode array.
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the nasal retina.
        Positive ``y`` values move the electrode into the superior retina.
        Positive ``z`` values move the electrode away from the retina into the
        vitreous humor (sometimes called electrode-retina distance).
        ``z`` can either be a list with 16 entries or a scalar that is applied
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
    use_legacy_names : bool, optional
        If True, uses L/M based electrode names from older papers (e.g., L6,
        L2) instead of A1-A16.

    Examples
    --------
    Create an Argus I array centered on the fovea, at 100um distance from
    the retina, rotated counter-clockwise by 5 degrees:

    >>> from pulse2percept.implants import ArgusI
    >>> ArgusI(x=0, y=0, z=100, rot=5)  # doctest: +NORMALIZE_WHITESPACE
    ArgusI(earray=ElectrodeGrid, eye='RE', preprocess=True,
           safe_mode=False, shape=(4, 4), stim=None)

    Get access to electrode 'B1', either by name or by row/column index:

    >>> argus = ArgusI(x=0, y=0, z=100, rot=0)
    >>> argus['B1']  # doctest: +NORMALIZE_WHITESPACE
    DiskElectrode(activated=True, name='B1', r=250.0, x=-400.0,
                  y=-1200.0, z=100.0)
    >>> argus[0, 1]  # doctest: +NORMALIZE_WHITESPACE
    DiskElectrode(activated=True, name='B1', r=250.0, x=-400.0,
                  y=-1200.0, z=100.0)

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape',)

    def __init__(self, x=0, y=0, z=0, rot=0, eye='RE', stim=None,
                 preprocess=True, safe_mode=False, use_legacy_names=False):
        self.eye = eye
        self.preprocess = preprocess
        self.safe_mode = safe_mode
        self.shape = (4, 4)
        r_arr = np.array([250, 500, 250, 500]) / 2.0
        r_arr = np.concatenate((r_arr, r_arr[::-1], r_arr, r_arr[::-1]),
                               axis=0)
        spacing = 800.0

        # In older papers, Argus I electrodes go by L and M:
        old_names = names = ['L6', 'L2', 'M8', 'M4',
                             'L5', 'L1', 'M7', 'M3',
                             'L8', 'L4', 'M6', 'M2',
                             'L7', 'L3', 'M5', 'M1']
        names = old_names if use_legacy_names else ('1', 'A')
        self.earray = ElectrodeGrid(self.shape, spacing, x=x, y=y, z=z,
                                    rot=rot, etype=DiskElectrode, r=r_arr,
                                    names=names)

        # Beware of race condition: Stim must be set last, because it requires
        # indexing into self.electrodes:
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
            names = self.earray.electrode_names
            objects = self.earray.electrode_objects
            names = np.array(names).reshape(self.earray.shape)
            # Reverse column names:
            for row in range(self.earray.shape[0]):
                names[row] = names[row][::-1]
            # Build a new ordered dict:
            electrodes = OrderedDict()
            for name, obj in zip(names.ravel(), objects):
                electrodes.update({name: obj})
            # Assign the new ordered dict to earray:
            self.earray._electrodes = electrodes

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'shape': self.shape})
        return params


class ArgusII(ProsthesisSystem):
    """Create an Argus II array on the retina

    This function creates an Argus II array and places it on the retina
    such that the center of the array is located at (x,y,z), given in
    microns, and the array is rotated by rotation angle ``rot``, given in
    degrees.

    Argus II contains 60 electrodes of 225 um diameter arranged in a 6 x 10
    grid (575 um center-to-center separation) [Yue2020]_.

    The array is oriented upright in the visual field, such that an
    array with center (0,0) has the top three rows lie in the lower
    retina (upper visual field), as shown below:

    .. raw:: html

        <pre>
                  A1 A2 A3 A4 A5 A6 A7 A8 A9 A10
          -- x    B1 B2 B3 B4 B5 B6 B7 B8 B9 B10
          |       C1 C2 C3 C4 C5 C6 C7 C8 C9 C10
          v       D1 D2 D3 D4 D5 D6 D7 D8 D9 D10
           y      E1 E2 E3 E4 E5 E6 E7 E8 E9 E10
                  F1 F2 F3 F4 F5 F6 F7 F8 F9 F10
        </pre>

    Electrode order is: A1, A2, ..., A10, B1, B2, ..., F10.

    An electrode can be addressed by name, row/column index, or integer index
    (into the flattened array).

    .. note::

        Column order is reversed in a left-eye implant.

    Parameters
    ----------
    x/y/z : double
        3D location of the center of the electrode array.
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the nasal retina.
        Positive ``y`` values move the electrode into the superior retina.
        Positive ``z`` values move the electrode away from the retina into the
        vitreous humor (sometimes called electrode-retina distance).
        ``z`` can either be a list with 60 entries or a scalar that is applied
        to all electrodes.
    rot : float
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

    Examples
    --------
    Create an ArgusII array centered on the fovea, at 100um distance from
    the retina, rotated counter-clockwise by 5 degrees:

    >>> from pulse2percept.implants import ArgusII
    >>> ArgusII(x=0, y=0, z=100, rot=5)  # doctest: +NORMALIZE_WHITESPACE
    ArgusII(earray=ElectrodeGrid, eye='RE', preprocess=True,
            safe_mode=False, shape=(6, 10), stim=None)

    Get access to electrode 'E7', either by name or by row/column index:

    >>> argus = ArgusII(x=0, y=0, z=100, rot=0)
    >>> argus['E7']  # doctest: +NORMALIZE_WHITESPACE
    DiskElectrode(activated=True, name='E7', r=112.5, x=862.5,
                  y=862.5, z=100.0)
    >>> argus[4, 6]  # doctest: +NORMALIZE_WHITESPACE
    DiskElectrode(activated=True, name='E7', r=112.5, x=862.5,
                  y=862.5, z=100.0)

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape',)

    def __init__(self, x=0, y=0, z=0, rot=0, eye='RE', stim=None,
                 preprocess=True, safe_mode=False):
        self.safe_mode = safe_mode
        self.preprocess = preprocess
        self.shape = (6, 10)
        r = 225.0 / 2.0
        spacing = 575.0
        names = ('A', '1')
        self.earray = ElectrodeGrid(self.shape, spacing, x=x, y=y, z=z, r=r,
                                    rot=rot, names=names, etype=DiskElectrode)

        # Beware of race condition: Stim must be set last, because it requires
        # indexing into self.electrodes:
        self.stim = stim

        # Set left/right eye:
        if not isinstance(eye, str):
            raise TypeError("'eye' must be a string, either 'LE' or 'RE'.")
        if eye != 'LE' and eye != 'RE':
            raise ValueError("'eye' must be either 'LE' or 'RE'.")
        self.eye = eye
        # Unfortunately, in the left eye the labeling of columns is reversed...
        if eye == 'LE':
            # TODO: Would be better to have more flexibility in the naming
            # convention. This is a quick-and-dirty fix:
            names = self.earray.electrode_names
            objects = self.earray.electrode_objects
            names = np.array(names).reshape(self.earray.shape)
            # Reverse column names:
            for row in range(self.earray.shape[0]):
                names[row] = names[row][::-1]
            # Build a new ordered dict:
            electrodes = OrderedDict()
            for name, obj in zip(names.ravel(), objects):
                electrodes.update({name: obj})
            # Assign the new ordered dict to earray:
            self.earray._electrodes = electrodes

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'shape': self.shape, 'safe_mode': self.safe_mode,
                       'preprocess': self.preprocess})
        return params
