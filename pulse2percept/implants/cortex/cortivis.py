"""`Cortivis`"""
import numpy as np

from ..base import ProsthesisSystem
from ..electrodes import DiskElectrode
from ..electrode_arrays import ElectrodeGrid

class Cortivis(ProsthesisSystem):
    """Create a Cortivis array
    
    This function creates a Cortivis array and places it on the visual cortex
    such that the center of the base of the array is at 3D location (x,y,z) given
    in microns, and the array is rotated by angle ``rot``, given in degrees.

    Cortivis is a Utah electrode array containing 96 electrodes in a 10x10 array
    with 400 um spacing, and electrode diameter of 80 um at the base.
    
    .. note::

        By default the implant is in right hemisphere, use negative x-values to shift it to left hemisphere
    
    Parameters
    ----------
    x/y/z : double
        3D location of the center of the electrode array.
        ``z`` can either be a list with 35 entries or a scalar that is applied
        to all electrodes.
    rot : float
        Rotation angle of the array (deg). Positive values denote
        counter-clock-wise (CCW) rotations in the retinal coordinate
        system.
    stim : :py:class:`~pulse2percept.stimuli.Stimulus` source type
        A valid source type for the :py:class:`~pulse2percept.stimuli.Stimulus`
        object (e.g., scalar, NumPy array, pulse train).
    preprocess : bool or callable, optional
        Either True/False to indicate whether to execute the implant's default
        preprocessing method whenever a new stimulus is assigned, or a custom
        function (callable).
    safe_mode : bool, optional
        If safe mode is enabled, only charge-balanced stimuli are allowed.

    Examples
    --------
    Create an Cortivis array, by default centered 20mm to the right of fovea in V1:

    >>> from pulse2percept.implants.cortex import Cortivis
    >>> Cortivis() # doctest: +NORMALIZE_WHITESPACE
    Cortivis(earray=ElectrodeGrid, preprocess=False, 
         safe_mode=False, shape=(10, 10), stim=None)

    Get access to electrode '11':

    >>> cortivis = Cortivis()
    >>> cortivis['11'] # doctest: +NORMALIZE_WHITESPACE
    DiskElectrode(activated=True, name='11', r=40.0, x=21400.0, 
                  y=-6000.0, z=-1500.0)
    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape',)

    # 400um spacing, 80um diameter at base, 10x10
    # depth of shanks: 1.5mm
    def __init__(self, x=20000, y=-5000, z=0, rot=0, stim=None,
                 preprocess=False, safe_mode=False):
        if not np.isclose(z, 0):
            raise NotImplementedError
        self.preprocess = preprocess
        self.safe_mode = safe_mode
        self.shape = (10, 10)
        spacing = 400
        names = ['01','1','2','3','4','5','6','7','8','02'] \
                + [str(i) for i in range(9, 89)] \
                + ['03','89','90','91','92','93','94','95','96','04']
        
        names = np.array(names).reshape((10, 10))
        names = np.swapaxes(names, 0, 1)[:, ::-1].reshape(100)

        # Account for depth of shanks
        z -= 1500
        self.earray = ElectrodeGrid(self.shape, spacing, x=x, y=y, z=z, rot=rot,
                                    names=names, type='rect', r=40,
                                    etype=DiskElectrode)
        for e in ['01', '02', '03', '04']:
            self.earray.remove_electrode(e)

        # Beware of race condition: Stim must be set last, because it requires
        # indexing into self.electrodes:
        self.stim = stim

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'shape': self.shape, 'safe_mode': self.safe_mode,
                       'preprocess': self.preprocess})
        return params
