""":py:class:`~pulse2percept.implants.IMIE`"""
import numpy as np
from collections import OrderedDict

from .base import ProsthesisSystem
from .electrodes import DiskElectrode
from .electrode_arrays import ElectrodeGrid

class IMIE(ProsthesisSystem):
    """The 256-channel epiretinal prosthesis system (IMIE 256)

    This class implements a 256-channel Intelligent Micro Implant Eye 
    epiretinal prosthesis system (IMIE 256) [Xu2021]. It was 
    co-developed by Golden Eye Bionic, LLC (Pasadena CA) and 
    IntelliMicro Medical Co., Ltd. (Changsha, Hunan Province, China) 
    and is manufactured by IntelliMicro.

    IMIE contains 248 large electrodes (210 µm in diameter) and 8 
    smaller electrodes (160 µm in diameter) arranged in a 4.75mm×6.50mm
    area.

    The array is oriented upright in the visual field, such that an
    array with center (0,0) has the top three rows lie in the lower
    retina (upper visual field):

    Parameters
    ----------
    x/y/z : double
        3D location of the center of the electrode array.
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the nasal retina.
        Positive ``y`` values move the electrode into the superior retina.
        Positive ``z`` values move the electrode away from the retina into the
        vitreous humor (sometimes called electrode-retina distance).
        ``z`` can either be a list with 35 entries or a scalar that is applied
        to all electrodes.
    rot : float
        Rotation angle of the array (deg). Positive values denote
        counter-clock-wise (CCW) rotations in the retinal coordinate
        system.
    eye : {'RE', 'LE'}, optional
        Eye in which array is implanted.
    preprocess : bool or callable, optional
        Either True/False to indicate whether to execute the implant's default
        preprocessing method whenever a new stimulus is assigned, or a custom
        function (callable).
    safe_mode : bool, optional
        If safe mode is enabled, only charge-balanced stimuli are allowed.
    """

    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape',)

    def __init__(self, x=0, y=0, z=0, rot=0, eye='RE', stim=None,
                 preprocess=True, safe_mode=False):
        self.eye = eye
        self.preprocess = preprocess
        self.safe_mode = safe_mode
        self.shape = (14, 19)
        elec_radius = 210.0 / 2
        e_spacing = 350.0  # um

        self.earray = ElectrodeGrid(self.shape, e_spacing, x=x, y=y, z=z,
                                    rot=rot, etype=DiskElectrode,
                                    r=elec_radius)
        
        # Set left/right eye:
        if not isinstance(eye, str):
            raise TypeError("'eye' must be a string, either 'LE' or 'RE'.")
        if eye != 'LE' and eye != 'RE':
            raise ValueError("'eye' must be either 'LE' or 'RE'.")
        self.eye = eye
        # Unfortunately, in the left eye the labeling of columns is reversed...
        if eye == 'LE':
            # TODO: Would be better to have more flexibility in the naming
            # convention. This is a quick-and-dirty fix:
            names = self.earray.electrode_names
            objects = self.earray.electrode_objects
            names = np.array(names).reshape(self.earray.shape)
            # Reverse column names:
            for row in range(self.earray.shape[0]):
                names[row] = names[row][::-1]
            # Build a new ordered dict:
            electrodes = OrderedDict()
            for name, obj in zip(names.ravel(), objects):
                electrodes.update({name: obj})
            # Assign the new ordered dict to earray:
            self.earray._electrodes = electrodes

        # Remove electrodes:
        extra_elecs = ['N1', 'N2', 'M1', 'B1', 'A1', 'A2', 'N18', 'N19', 
                       'A18', 'A19' ]
        for elec in extra_elecs:
            self.earray.remove_electrode(elec)

        # Change electrodes to smaller ones in place:
        small_elecs = ['N16', 'A16', 'K1', 'D1']
        small_radius = 160.0 / 2
        for elec in small_elecs:
            e = self.earray.electrodes[elec]
            x, y, z = e.x, e.y, e.z
            self.earray.remove_electrode(elec)
            new_e = DiskElectrode(x, y, z, small_radius, name=elec)
            self.earray.add_electrode(elec, new_e)
        
        # Change the rest smaller electrodes according to its neighbor:
        small_elecs_rest = {'N17' : 'N16', 'A17' : 'A16', 'L1' : 'K1', 
                            'C1' : 'D1'}
        for elec in small_elecs_rest:
            e = self.earray.electrodes[elec]
            x, y, z = e.x, e.y, e.z
            neighbor = self.earray.electrodes[small_elecs_rest[elec]]
            nx, ny, nz = neighbor.x, neighbor.y, neighbor.z
            newx, newy, newz = x-(x-nx)/7, y-(y-ny)/7, z-(z-nz)/7
            self.earray.remove_electrode(elec)
            new_e = DiskElectrode(newx, newy, newz, small_radius, name=elec)
            self.earray.add_electrode(elec, new_e)

        # Beware of race condition: Stim must be set last, because it requires
        # indexing into self.electrodes:   
        self.stim = stim

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'shape': self.shape, 'safe_mode': self.safe_mode,
                       'preprocess': self.preprocess})
        return params