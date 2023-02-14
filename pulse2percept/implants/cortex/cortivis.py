"""`Cortivis`"""
import numpy as np

from ..base import ProsthesisSystem
from ..electrodes import DiskElectrode
from ..electrode_arrays import ElectrodeGrid

class Cortivis(ProsthesisSystem):
    """

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape',)

    # 400um spacing, 80um diameter at base, 10x10
    # depth of shanks: 1.5mm
    def __init__(self, x=15000, y=0, z=0, rot=0, eye='RE', stim=None,
                 preprocess=False, safe_mode=False):
        self.eye = eye
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