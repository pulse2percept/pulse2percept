import numpy as np
from pulse2percept.implants import ProsthesisSystem

from .. import ProsthesisSystem
from ..electrodes import DiskElectrode
from ..electrode_arrays import ElectrodeGrid


class Orion(ProsthesisSystem):
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape',)
    # TODO: Get rid of eye arg?
    def __init__(self, x=15000, y=0, z=0, rot=0, stim=None,
                 preprocess=False, safe_mode=False):
        """Orion I
        Parameters
        ----------
        x/y/z : double
            3D location (mm) of the center of the electrode array.
            The coordinate system is centered over the fovea on the cortical
            surface.
            Positive ``x`` values move the electrode into the periphery
            (azimuth), whereas positive ``y`` values move the electrode into
            the upper visual field (elevation). ``z`` is ignored for now.
        rot : float, optional
            Rotation angle of the array (deg). Positive values denote
            counter-clock-wise (CCW) rotations in the cortical coordinate
            system.
        preprocess : bool or callable, optional
            Either True/False to indicate whether to execute the implant's
            default preprocessing method whenever a new stimulus is assigned, 
            or a custom function (callable).
        safe_mode : bool, optional
            If safe mode is enabled, only charge-balanced stimuli are allowed.
        """

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
