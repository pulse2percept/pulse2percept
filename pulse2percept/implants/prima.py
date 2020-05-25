"""`Prima`"""
import numpy as np
from .base import ElectrodeGrid, ProsthesisSystem, DiskElectrode
from collections import OrderedDict

class Prima(ProsthesisSystem):
    """Create an Prima array on retina

    This function creates an Prima array and places it on the retina
    such that the center of the array is located at 3D location (x,y,z),
    given in microns, and the array is rotated by rotation angle ``rot``,
    given in radians.


    Parameters
    ----------
    x : float, optional, default: 0
        x coordinate of the array center (um)
    y : float, optional: default: 0
        y coordinate of the array center (um)
    z: float || array_like, optional, default: 0
        Distance of the array to the retinal surface (um). Either a list
        with 378 entries or a scalar.
    rot : float, optional, default: 0
        Rotation angle of the array (rad). Positive values denote
        counter-clock-wise (CCW) rotations in the retinal coordinate
        system.
    eye : {'LE', 'RE'}, optional, default: 'RE'
        Eye in which array is implanted.

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape',)

    def __init__(self, x=0, y=0, z=0, rot=0, eye='RE', stim=None):
        self.shape = (19, 22) # total number of columns is 22
                              # maximum number of electrodes of each row is 19
        self.eye= eye
        elec_radius = 10 # um
        e_spacing = 75  # um
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

        for elec in extra_elecs:
            self.earray.remove_electrode(elec)
        
        # rename all electrodes
        colIndex = 0
        rows, cols = self.shape
        updateRowIndex = chr(ord('A') + rows - 1) # start from the last electrode
        prevRowIndex = 'A'
        originalEarray = self.earray.electrodes
        newEarray = OrderedDict()
        for name in originalEarray:
            if name[0] == prevRowIndex:
                colIndex += 1
            else:
                prevRowIndex = chr(ord(prevRowIndex) + 1)
                updateRowIndex = chr(ord(updateRowIndex) - 1)
                colIndex = 1
            newName = updateRowIndex + str(colIndex)
            newEarray.update({newName: originalEarray[name]})
        self.earray.electrodes = newEarray