""":py:class:`~pulse2percept.implants.cortex.Orion`"""
import numpy as np

from .. import Implant
from ..electrodes import DiskElectrode
from ..electrode_arrays import ElectrodeGrid
from ...utils.constants import UM_PER_MM


class Orion(Implant):
    """Create a Orion array
    
    This function creates a Orion array and places it on the visual cortex
    such that the center of the base of the array is at 3D location (x,y,z) given
    in microns, and the array is rotated by angle ``rot``, given in degrees.

    Orion contains 60 electrodes in a hex shaped grid inspired by Argus II.
    
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
    Create an Orion array in its own coordinate frame:

    >>> from pulse2percept.implants.cortex import Orion
    >>> Orion() # doctest: +NORMALIZE_WHITESPACE
    Orion(electrode_array=ElectrodeGrid, preprocess=False, 
          safe_mode=False, shape=(10, 7))

    Get access to electrode '96':

    >>> orion = Orion()
    >>> orion['96'] # doctest: +NORMALIZE_WHITESPACE
    DiskElectrode(activated=True, name='96', radius=1000.0,
                  x=-11550.0, y=-9640.928378532848, z=0.0)
    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape',)
    placement = 'epicortical'

    def __init__(self, rot=0, preprocess=False, safe_mode=False):
        self.preprocess = preprocess
        self.safe_mode = safe_mode
        self.shape = (10, 7)
        # The row offset is published in millimeters; coordinates are microns:
        spacing = (4200, np.sqrt(3**2-2.1**2) * UM_PER_MM)
        self.electrode_array = ElectrodeGrid(
            self.shape, spacing, rot=rot, names=('A', '-1'),
            grid_type='hex', radius=1000, electrode_type=DiskElectrode)
        for e in ['A1', 'F7', 'G7', 'H6', 'H7', 'I6', 'I7', 'J5', 'J6', 'J7']:
            self.electrode_array.remove_electrode(e)
        # Hacking the naming scheme:
        names = [f'{i:02}' for i in range(96, 36, -1)]
        electrodes = {}
        for ename, eobject in zip(names,
                                  self.electrode_array.electrode_objects):
            eobject.name = ename
            electrodes.update({ename: eobject})
        self._electrode_array._electrodes = electrodes

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'shape': self.shape, 'safe_mode': self.safe_mode,
                       'preprocess': self.preprocess})
        return params
