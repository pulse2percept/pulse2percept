""":py:class:`~pulse2percept.implants.cortex.ICVP`"""
import numpy as np

from ..base import Implant
from ..electrodes import DiskElectrode
from ..electrode_arrays import ElectrodeGrid


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

        The electrodes describe the device in its own frame, centered on
        ``(0, 0)``. Where it is implanted is the spatial model's
        ``implant_pos``, e.g. ``implant_pos=(20, -5) * mm`` for the right
        hemisphere, or a visual field position in dva.
    
    Parameters
    ----------
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
    Create an ICVP array in its own coordinate frame:

    >>> from pulse2percept.implants.cortex import Orion
    >>> ICVP() # doctest: +NORMALIZE_WHITESPACE
    ICVP(electrode_array=ElectrodeGrid, preprocess=False, 
         safe_mode=False, shape=(5, 4))

    Get access to electrode '11':

    >>> icvp = ICVP()
    >>> icvp['11'] # doctest: +NORMALIZE_WHITESPACE
    DiskElectrode(activated=True, name='11', radius=50.0,
                  x=173.2050807568877, y=100.0, z=-650.0)
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

    def __init__(self, rot=0, preprocess=False, safe_mode=False):
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

        # These electrodes have a shaft length of 650 microns, the rest 850.
        # Shank length is device geometry rather than placement:
        length_650 = {'9', '2', '6', '11', '15', '4', '8', '13'}
        z = -np.array([650 if name in length_650 else 850 for name in names],
                      dtype=float)

        self.electrode_array = ElectrodeGrid(
            self.shape, spacing, z=z, rot=rot, names=names,
            grid_type='hex', orientation='vertical', radius=50,
            electrode_type=DiskElectrode
        )
        for e in ['t1', 't2']:
            self.electrode_array.remove_electrode(e)

        self.electrode_array.deactivate(['R', 'C'])

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'shape': self.shape, 'safe_mode': self.safe_mode,
                       'preprocess': self.preprocess})
        return params
