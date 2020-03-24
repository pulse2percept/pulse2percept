"""`AlphaIMS`, `AlphaAMS`"""
import numpy as np
from collections import OrderedDict
from .base import DiskElectrode, ElectrodeArray, ElectrodeGrid, ProsthesisSystem


class AlphaIMS(ProsthesisSystem):
    """Alpha IMS

    This class creates an AlphaIMS array and places it on the retina
    such that the center of the array is located at (x,y,z), given in
    microns, and the array is rotated by rotation angle ``rot``, given in
    radians.

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
        with 60 entries or a scalar.
    rot : float
        Rotation angle of the array (rad). Positive values denote
        counter-clock-wise (CCW) rotations in the retinal coordinate
        system.
    eye : 'LE' or 'RE', optional, default: 'RE'
        Eye in which array is implanted.

    Examples
    --------
    Create an AlphaIMS array centered on the fovea, at 100um distance from
    the retina:

    >>> from pulse2percept.implants import AlphaIMS
    >>> AlphaIMS(x=0, y=0, z=100, rot=0)  # doctest: +NORMALIZE_WHITESPACE
    AlphaIMS(earray=pulse2percept.implants.base.ElectrodeGrid, eye='RE',
             shape=(37, 37), stim=None)

    Get access to the third electrode in the top row (by name or by row/column
    index):

    >>> alpha_ims = AlphaIMS(x=0, y=0, z=100, rot=0)
    >>> alpha_ims['A3']
    DiskElectrode(r=50.0, x=-1152.0, y=-1296.0, z=100.0)
    >>> alpha_ims[0, 2]
    DiskElectrode(r=50.0, x=-1152.0, y=-1296.0, z=100.0)

    """

    def __init__(self, x=0, y=0, z=0, rot=0, eye='RE', stim=None):
        self.eye = eye
        self.shape = (37, 37)
        elec_radius = 50
        e_spacing = 72  # um
        self.earray = ElectrodeGrid(self.shape, e_spacing, x=x, y=y, z=z,
                                    rot=rot, etype=DiskElectrode,
                                    r=elec_radius)

        # Set stimulus if available:
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


class AlphaAMS(ProsthesisSystem):
    """Alpha AMS

    This class creates an AlphaAMS array and places it below the retina
    such that the center of the array is located at (x,y,z), given in
    microns, and the array is rotated by rotation angle ``rot``, given in
    radians.

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
    AlphaAMS(earray=pulse2percept.implants.base.ElectrodeGrid, eye='RE',
             shape=(40, 40), stim=None)

    Get access to the third electrode in the top row (by name or by row/column
    index):

    >>> alpha_ims = AlphaAMS(x=0, y=0, z=100, rot=0)
    >>> alpha_ims['A3']
    DiskElectrode(r=15.0, x=-1225.0, y=-1365.0, z=100.0)
    >>> alpha_ims[0, 2]
    DiskElectrode(r=15.0, x=-1225.0, y=-1365.0, z=100.0)

    """

    def __init__(self, x=0, y=0, z=0, rot=0, eye='RE', stim=None):
        self.eye = eye
        self.shape = (40, 40)
        elec_radius = 15
        e_spacing = 70  # um

        self.earray = ElectrodeGrid(self.shape, e_spacing, x=x, y=y, z=z,
                                    rot=rot, etype=DiskElectrode,
                                    r=elec_radius)

        # Set stimulus if available:
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
