""":py:class:`~pulse2percept.implants.cortex.ICVP`"""
import numpy as np

from ..base import Implant
from ..electrodes import DiskElectrode
from ..electrode_arrays import ElectrodeGrid
from ...units import as_value, um


class ICVP(Implant):
    """Create an ICVP array

    This function creates a ICVP array and places it on the visual cortex
    such that the center of the base of the array is at 3D location (x,y,z) given
    in microns, and the array is rotated by angle ``rot``, given in degrees.

    ICVP (Intracortical Visual Prosthesis Project) is an electrode array containing 
    16 Parylene-insulated (and 2 uninsulated reference and counter) iridium shaft
    electrodes in a 4 column array with 400 um spacing [Troyk2003]_.
    The electrodes have a diameter of 15 um at the laser cut.
    They are inserted either 650 um or 850 um into the cortex.

    .. note::

        By default the implant is in right hemisphere, use negative x-values to shift it to left hemisphere
    
    Parameters
    ----------
    x/y/z : double
        3D location (um) of the center of the electrode array.
        ``z`` can either be a list with 35 entries or a scalar that is applied
        to all electrodes.
        May be given as unitful quantities (e.g. ``ICVP(x=15 * mm)``); see
        :py:mod:`pulse2percept.units`.
    rot : float or Quantity
        Rotation angle of the array (deg). Positive values denote
        counter-clock-wise (CCW) rotations in the retinal coordinate
        system.
    preprocess : bool or callable, optional
        Either True/False to indicate whether to execute the implant's default
        preprocessing method whenever a stimulus is prepared, or a custom
        function (callable).
    safe_mode : bool, optional
        If safe mode is enabled, only charge-balanced stimuli are allowed.

    Examples
    --------
    Create an ICVP array, by default centered 15mm to the right of fovea in V1:

    >>> from pulse2percept.implants.cortex import Orion
    >>> ICVP() # doctest: +NORMALIZE_WHITESPACE
    ICVP(earray=ElectrodeGrid, preprocess=False, 
         safe_mode=False, shape=(5, 4))

    Get access to electrode '11':

    >>> icvp = ICVP()
    >>> icvp['11'] # doctest: +NORMALIZE_WHITESPACE
    DiskElectrode(activated=True, name='11', r=50.0, 
                  x=15173.205080756888, y=100.0, z=-650.0)
    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape',)

    # 100um diameter at base 
    # (https://iopscience.iop.org/article/10.1088/1741-2552/abb9bf/pdf)

    # 400um spacing, 4x4 + reference (R) and count (C)
    # (https://iopscience.iop.org/article/10.1088/1741-2552/ac2bb8)

    # depth of shanks: 650 or 850 um
    # (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9175335)

    placement = 'intracortical'

    def __init__(self, x=15000, y=0, z=0, rot=0, preprocess=False, safe_mode=False):
        # Inspected, broadcast and offset here, before the grid ever sees it:
        z = as_value(z, um, 'z')
        if not np.isclose(z, 0):
            raise NotImplementedError
        self.preprocess = preprocess
        self.safe_mode = safe_mode
        self.shape = (5, 4)
        spacing = 400
        names = np.array(
            [
                [i for i in range(1, 5)] + ['R'],
                [i for i in range(5, 9)] + ['t1'],
                [i for i in range(9, 14)],
                ['C'] + [i for i in range(14, 17)] + ['t2']
            ]
        )
        names = np.rot90(names).flatten()

        if not isinstance(z, (list, np.ndarray)):
            z = np.full(20, z, dtype=float)

        # These electrodes have a shaft length of 650 microns, the rest are 650 microns
        length_650 = {'9', '2', '6', '11', '15', '4', '8', '13'}

        # account for depth of shanks
        z_offset = [650 if name in length_650 else 850 for name in names]
        z -= z_offset

        self.earray = ElectrodeGrid(
            self.shape, spacing, x=x, y=y, z=z, rot=rot, names=names,
            type='hex', orientation='vertical', r=50, etype=DiskElectrode
        )
        for e in ['t1', 't2']:
            self.earray.remove_electrode(e)

        self.earray.deactivate(['R', 'C'])

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'shape': self.shape, 'safe_mode': self.safe_mode,
                       'preprocess': self.preprocess})
        return params
