"""`AlphaIMS`, `AlphaAMS`"""
import numpy as np
from collections import OrderedDict
from .electrodes import SquareElectrode, DiskElectrode
from .base import ElectrodeArray, ElectrodeGrid, ProsthesisSystem


class AlphaIMS(ProsthesisSystem):
    """Alpha-IMS

    This class creates an Alpha-IMS array with 1500 photovoltaic pixels (each
    50um in diameter) as described in [Stingl2013]_, and places it in the
    subretinal space, such that the center of the array is located at (x,y,z),
    given in microns, and the array is rotated by rotation angle ``rot``,
    given in radians.

    The device consists of 1500 50um-wide square pixels, arranged on a 39x39
    rectangular grid with 72um pixel pitch.

    The array is oriented upright in the visual field, such that an
    array with center (0,0) has the top three rows lie in the lower
    retina (upper visual field).

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
    z: float or array_like
        Distance of the array to the retinal surface (um). Either a list
        with 1500 entries or a scalar.
    rot : float
        Rotation angle of the array (rad). Positive values denote
        counter-clock-wise (CCW) rotations in the retinal coordinate
        system.
    eye : 'LE' or 'RE', optional, default: 'RE'
        Eye in which array is implanted.

    Examples
    --------
    Create an Alpha-IMS array centered on the fovea, at 100um distance from
    the retina:

    >>> from pulse2percept.implants import AlphaIMS
    >>> AlphaIMS(x=0, y=0, z=100, rot=0)  # doctest: +NORMALIZE_WHITESPACE
    AlphaIMS(earray=ElectrodeGrid, eye='RE', shape=(39, 39),
             stim=None)

    Get access to the third electrode in the top row (by name or by row/column
    index):

    >>> alpha_ims = AlphaIMS(x=0, y=0, z=100, rot=0)
    >>> alpha_ims['A3']
    SquareElectrode(a=50.0, x=-1224.0, y=-1368.0, z=100.0)
    >>> alpha_ims[0, 2]
    SquareElectrode(r=50.0, x=-1224.0, y=-1368.0, z=100.0)

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape',)

    def __init__(self, x=0, y=0, z=-100, rot=0, eye='RE', stim=None):
        self.eye = eye
        self.shape = (39, 39)
        elec_width = 50.0  # um
        e_spacing = 72.0  # um

        # The user might provide a list of z values for each of the
        # 378 resulting electrodes, not for the 22x19 initial ones.
        # In this case, don't pass it to ElectrodeGrid, but overwrite
        # the z values later:
        overwrite_z = isinstance(z, (list, np.ndarray))
        zarr = -100.0 if overwrite_z else z
        self.earray = ElectrodeGrid(self.shape, e_spacing, x=x, y=y, z=zarr,
                                    rot=rot, etype=SquareElectrode,
                                    a=elec_width)

        # Set left/right eye:
        if not isinstance(eye, str):
            raise TypeError("'eye' must be a string, either 'LE' or 'RE'.")
        if eye != 'LE' and eye != 'RE':
            raise ValueError("'eye' must be either 'LE' or 'RE'.")
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
            electrodes = OrderedDict([])
            for name, obj in zip(names.ravel(), objects):
                electrodes.update({name: obj})
            # Assign the new ordered dict to earray:
            self.earray.electrodes = electrodes

        # Remove electrodes:
        extra_elecs = ['AM39', 'AL39', 'AK39', 'AJ39', 'AI39', 'AH39', 'AG39',
                       'AF39', 'AE39', 'AD39', 'AC39',
                       'AM38', 'AL38', 'AK38', 'AJ38', 'AI38', 'AH38', 'AG38',
                       'AF38', 'AE38', 'AD38']
        for elec in extra_elecs:
            self.earray.remove_electrode(elec)

        # Now that the superfluous electrodes have been deleted, adjust the
        # z values:
        if overwrite_z:
            # Specify different height for every electrode in a list:
            z_arr = np.asarray(z).flatten()
            if z_arr.size != self.n_electrodes:
                raise ValueError("If `z` is a list, it must have %d entries, "
                                 "not %d." % (self.n_electrodes, z_arr.size))
            for elec, z_elec in zip(self.earray.values(), z):
                elec.z = z_elec

        # Beware of race condition: Stim must be set last, because it requires
        # indexing into self.electrodes:
        self.stim = stim

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'shape': self.shape})
        return params


class AlphaAMS(ProsthesisSystem):
    """Alpha-AMS

    This class creates an Alpha-AMS array with 1600 photovoltaic pixels (each
    30um in diameter) as described in [Stingl2017]_, and places it in the
    subretinal space, such that the center of the array is located at (x,y,z),
    given in microns, and the array is rotated by rotation angle ``rot``,
    given in radians.

    The device consists of 1600 30um-wide round pixels, arranged on a 40x40
    rectangular grid with 70um pixel pitch.

    The array is oriented upright in the visual field, such that an
    array with center (0,0) has the top three rows lie in the lower
    retina (upper visual field), as shown below:

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
    Create an AlphaAMS array centered on the fovea, at 100um distance from
    the retina:

    >>> from pulse2percept.implants import AlphaAMS
    >>> AlphaAMS(x=0, y=0, z=100, rot=0)  # doctest: +NORMALIZE_WHITESPACE
    AlphaAMS(earray=ElectrodeGrid, eye='RE', shape=(40, 40),
             stim=None)

    Get access to the third electrode in the top row (by name or by row/column
    index):

    >>> alpha_ims = AlphaAMS(x=0, y=0, z=100, rot=0)
    >>> alpha_ims['A3']
    DiskElectrode(r=15.0, x=-1225.0, y=-1365.0, z=100.0)
    >>> alpha_ims[0, 2]
    DiskElectrode(r=15.0, x=-1225.0, y=-1365.0, z=100.0)

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape',)

    def __init__(self, x=0, y=0, z=0, rot=0, eye='RE', stim=None):
        self.eye = eye
        self.shape = (40, 40)
        elec_radius = 15.0
        e_spacing = 70.0  # um

        self.earray = ElectrodeGrid(self.shape, e_spacing, x=x, y=y, z=z,
                                    rot=rot, etype=DiskElectrode,
                                    r=elec_radius)

        # Beware of race condition: Stim must be set last, because it requires
        # indexing into self.electrodes:
        self.stim = stim

        # Set left/right eye:
        if not isinstance(eye, str):
            raise TypeError("'eye' must be a string, either 'LE' or 'RE'.")
        if eye != 'LE' and eye != 'RE':
            raise ValueError("'eye' must be either 'LE' or 'RE'.")
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
            electrodes = OrderedDict([])
            for name, obj in zip(names.ravel(), objects):
                electrodes.update({name: obj})
            # Assign the new ordered dict to earray:
            self.earray.electrodes = electrodes

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'shape': self.shape})
        return params
