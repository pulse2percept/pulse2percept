""":py:class:`~pulse2percept.implants.cortex.Orion`"""
import numpy as np

from .. import Implant
from ..electrodes import DiskElectrode
from ..electrode_arrays import ElectrodeGrid
from ...utils.constants import UM_PER_MM


class Orion(Implant):
    """Create a Orion array
    
    Electrode coordinates are given in the array's own frame, with the
    center of its base at ``(0, 0)``. Where the array is implanted is set
    by the model's ``implant_position`` and ``implant_rotation``.

    Orion contains 60 electrodes in a hex shaped grid inspired by Argus II.
    
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

    def __init__(self, preprocess=False, safe_mode=False):
        self.preprocess = preprocess
        self.safe_mode = safe_mode
        self.shape = (10, 7)
        # The row offset is published in millimeters; coordinates are microns:
        spacing = (4200, np.sqrt(3**2-2.1**2) * UM_PER_MM)
        self.electrode_array = ElectrodeGrid(
            self.shape, spacing, names=('A', '-1'),
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
