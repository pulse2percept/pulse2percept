"""`Cortivis`"""
import numpy as np

from ..base import ProsthesisSystem
from ..electrodes import DiskElectrode
from ..electrode_arrays import ElectrodeGrid

class Cortivis(ProsthesisSystem):
    """Create a Cortivis array.
    
    This function creates a Cortivis array and places it on the visual cortex
    such that the center of the base of the array is at 3D location (x,y,z) given
    in microns, and the array is rotated by angle ``rot``, given in degrees.

    Cortivis is a Utah electrode array containing 96 electrodes in a 10x10 array
    with 400 um spacing, and electrode diameter of 80 um at the base.
    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape',)

    # 400um spacing, 80um diameter at base, 10x10
    # depth of shanks: 1.5mm
    def __init__(self, x=15000, y=0, z=0, rot=0, stim=None,
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
        # account for depth of shanks
        z -= 1500
        self.earray = ElectrodeGrid(self.shape, spacing, x=x, y=y, z=z, rot=rot,
                                    names=names, type='rect', r=40,
                                    etype=DiskElectrode)
        for e in ['01', '02', '03', '04']:
            self.earray.remove_electrode(e)

        # Beware of race condition: Stim must be set last, because it requires
        # indexing into self.electrodes:
        self.stim = stim

    def plot(self, annotate=False, autoscale=True, ax=None):
        ax = super(Cortivis, self).plot(annotate=annotate, autoscale=autoscale,
                                     ax=ax)
        return ax
