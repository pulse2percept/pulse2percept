""":py:class:`~pulse2percept.implants.cortex.Orion`"""
import numpy as np
from pulse2percept.implants import ProsthesisSystem

from .. import ProsthesisSystem
from ..electrodes import DiskElectrode
from ..electrode_arrays import ElectrodeGrid


class Orion(ProsthesisSystem):
    """Create a Orion array
    
    This function creates a Orion array and places it on the visual cortex
    such that the center of the base of the array is at 3D location (x,y,z) given
    in microns, and the array is rotated by angle ``rot``, given in degrees.

    Orion contains 60 electrodes in a hex shaped grid inspired by Argus II.
    
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
    Create an Orion array, by default centered 15mm to the right of fovea in V1:

    >>> from pulse2percept.implants.cortex import Orion
    >>> Orion() # doctest: +NORMALIZE_WHITESPACE
    Orion(earray=ElectrodeGrid, preprocess=False, 
          safe_mode=False, shape=(10, 7), stim=None)

    Get access to electrode '96':

    >>> orion = Orion()
    >>> orion['96'] # doctest: +NORMALIZE_WHITESPACE
    DiskElectrode(activated=True, name='96', r=1000.0, 
                  x=3450.0, y=-9640.928378532848, z=0.0)
    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape',)
    def __init__(self, x=15000, y=0, z=0, rot=0, stim=None,
                 preprocess=False, safe_mode=False):

        if not np.isclose(z, 0):
            raise NotImplementedError
        self.preprocess = preprocess
        self.safe_mode = safe_mode
        self.shape = (10, 7)
        spacing = (4200, np.sqrt(3**2-2.1**2)*1000)
        self.earray = ElectrodeGrid(self.shape, spacing, x=x, y=y, z=z, rot=rot,
                                    names=('A', '-1'), type='hex', r=1000,
                                    etype=DiskElectrode)
        for e in ['A1', 'F7', 'G7', 'H6', 'H7', 'I6', 'I7', 'J5', 'J6', 'J7']:
            self.earray.remove_electrode(e)
        # Hacking the naming scheme:
        names = [f'{i:02}' for i in range(96, 36, -1)]
        electrodes = {}
        for ename, eobject in zip(names, self.earray.electrode_objects):
            eobject.name = ename
            electrodes.update({ename: eobject})
        self._earray._electrodes = electrodes

        # Beware of race condition: Stim must be set last, because it requires
        # indexing into self.electrodes:
        self.stim = stim

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'shape': self.shape, 'safe_mode': self.safe_mode,
                       'preprocess': self.preprocess})
        return params
