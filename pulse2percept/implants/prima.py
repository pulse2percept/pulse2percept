"""`Prima`"""
import numpy as np
from collections import OrderedDict
from pulse2percept.implants import ElectrodeGrid, ProsthesisSystem, DiskElectrode

class Prima(ProsthesisSystem):
    def __init__(self, x=0, y=0, z=0, rot=0, eye='RE', stim=None):
        self.shape = (19, 22) # total number of columns is 22
                              # maximum number of electrodes of each row is 19
        self.eye= eye
        elec_radius = 10 # um
        e_spacing = 70  # um
        self.earray = ElectrodeGrid(self.shape, e_spacing, x=x, y=y, z=z,
                                    rot=rot, type='hex', orientation='vertical',etype=DiskElectrode,
                                    r=elec_radius)
        # Set stimulus if available:
        self.stim = stim
        # remove extra electrodes to fit the actual implant
        extra_elecs = ['A1','A2','A3','A4','A14','A16','A17',
                       'A18','A19','A20','A21','A22','B1',
                       'B2','B18','B19','B20','B21','B22',
                       'C1','C20','C21','C22','D22','E22','P1',
                       'Q1','Q22','R1','R2','R21','R22','S1',
                       'S2','S3','S5','S19','S20','S21','S22']

        # rename all electrodes
        for elec in extra_elecs:
            self.earray.remove_electrode(elec)