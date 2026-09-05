""":py:class:`~pulse2percept.implants.cortex.Cortivis`"""
import numpy as np

from ..base import Implant
from ..electrodes import DiskElectrode
from ..electrode_arrays import ElectrodeGrid

class Cortivis(Implant):
    """Create a Cortivis array
    
    Electrode coordinates are device-local, with the base centered at
    ``(0, 0)``.

    Cortivis is a Utah electrode array containing 96 electrodes in a 10x10 array
    with 400 um spacing, and electrode diameter of 80 um at the base
    [Fernandez2017]_.
    
    .. note::

        Implant the array with the model's ``implant_position``, e.g.
        ``implant_position=(20, -5) * mm`` for the right hemisphere.
    
    Parameters
    ----------
    preprocess : bool or callable, optional
        Either True/False to indicate whether to execute the implant's default
        preprocessing method whenever a stimulus is prepared, or a custom
        function (callable).
    safe_mode : bool, optional
        If safe mode is enabled, only charge-balanced stimuli are allowed.

    Examples
    --------
    Create a Cortivis array in its own coordinate frame:

    >>> from pulse2percept.implants.cortex import Cortivis
    >>> Cortivis() # doctest: +NORMALIZE_WHITESPACE
    Cortivis(electrode_array=ElectrodeGrid, preprocess=False, 
         safe_mode=False, shape=(10, 10))

    Get access to electrode '11':

    >>> cortivis = Cortivis()
    >>> cortivis['11'] # doctest: +NORMALIZE_WHITESPACE
    DiskElectrode(activated=True, name='11', radius=40.0,
                  x=1400.0, y=-1000.0, z=-1500.0)
    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape',)

    # 400um spacing, 80um diameter at base, 10x10
    # depth of shanks: 1.5mm
    placement = 'intracortical'

    def __init__(self, preprocess=False, safe_mode=False):
        self.preprocess = preprocess
        self.safe_mode = safe_mode
        self.shape = (10, 10)
        spacing = 400
        names = ['01','1','2','3','4','5','6','7','8','02'] \
                + [str(i) for i in range(9, 89)] \
                + ['03','89','90','91','92','93','94','95','96','04']
        
        names = np.array(names).reshape((10, 10))
        names = np.swapaxes(names, 0, 1)[:, ::-1].reshape(100)

        # Shank depth, which is device geometry rather than placement:
        z = -1500
        self.electrode_array = ElectrodeGrid(
            self.shape, spacing, z=z, names=names,
            grid_type='rect', radius=40, electrode_type=DiskElectrode)
        for e in ['01', '02', '03', '04']:
            self.electrode_array.remove_electrode(e)

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'shape': self.shape, 'safe_mode': self.safe_mode,
                       'preprocess': self.preprocess})
        return params
