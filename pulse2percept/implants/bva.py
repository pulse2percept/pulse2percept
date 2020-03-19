"""`BVA24`"""
import numpy as np
from pulse2percept.implants.base import (ProsthesisSystem, ElectrodeArray,
                                         DiskElectrode)

class BVA24(ProsthesisSystem):
    """
    33 platinum stimulating electrodes: 
        30 electrodes in 600 microns (Electrodes 1-20(except 9, 17, 19) and Electrodes 21a-m), 
        3 electrodes in 400 microns (Electrodes 9, 17, 19)
    2 return electrodes in 2000 microns (Electrodes 22, 23)
    """
    def __init__(self, x_center=0, y_center=0, rot=0, eye='RE', stim=None):
        self.earray = ElectrodeArray([])
        self.x = x_center
        self.y = y_center
        rot = np.deg2rad(rot)
        self.rot = rot
        self.stim = stim

        # Set left/right eye:
        if not isinstance(eye, str):
            raise TypeError("'eye' must be a string, either 'LE' or 'RE'.")
        if eye != 'LE' and eye != 'RE':
            raise ValueError("'eye' must be either 'LE' or 'RE'.")
        self.eye = eye
        
        # the positions of the electrodes 1-20, 21a-21m, R1-R2
        x_arr = [-1275.0, -850.0, -1275.0, -850.0, -1275.0,
                 -425.0, 0, -425.0, 0, -425.0,
                 425.0, 850.0, 425.0, 850.0, 425.0,
                 1275.0, 1700.0, 1275.0, 1700.0, 1275.0,
                 -850.0, 0, 850.0, 1700.0, 2125.0,
                 2550.0, 2125.0, 2550.0, 2125.0, 1700.0,
                 850.0, 0, -850.0, 7000.0, 9370.0]
        y_arr = [1520.0, 760.0, 0, -760.0, -1520.0,
                 1520.0, 760.0, 0, -760.0, -1520.0,
                 1520.0, 760.0, 0, -760.0, -1520.0,
                 1520.0, 760.0, 0, -760.0, -1520.0,
                 2280.0, 2280.0, 2280.0, 2280.0, 1520.0,
                 760.0, 0.0, -760.0, -1520.0, -2280.0,
                 -2280.0, -2280.0, -2280.0, 0, 0]
        z_arr = np.ones(35, dtype=float)*0

        # the position of the electrodes 1-20, 21a-21m, R1-R2 for left eye
        if eye == 'LE':
            x_arr = np.negative(x_arr)
        
        z_arr = np.ones(35, dtype=float)*0
        
        # the radius of all the electrodes in the implants
        r_arr = [300.0]*35
        # the radius of electrodes 9, 17, 19 is 200.0 um
        r_arr[8] = r_arr[16] = r_arr[18] = 200.0
        # the radius of the return electrodes is 1000.0 um
        r_arr[33] = r_arr[34] = 1000.0
        # the names of the electrodes 1-20, 21a-21m, R1 and R2
        names = [str(name) for name in range(1,21)]
        names.extend(['21a', '21b', '21c', '21d', '21e',
                      '21f', '21g', '21h', '21i', '21j',
                      '21k', '21l', '21m'])
        names.extend(['R1', 'R2'])
        
        # Rotate the grid:
        rotmat = np.array([np.cos(self.rot), -np.sin(self.rot),
                           np.sin(self.rot), np.cos(self.rot)]).reshape((2, 2))
        xy = np.matmul(rotmat, np.vstack((x_arr, y_arr)))
        x_arr = xy[0, :]
        y_arr = xy[1, :]
    
        # Apply offset to make the grid centered at (self.x, self.y):
        x_arr += self.x
        y_arr += self.y
        
        for x, y, z, r, name in zip(x_arr, y_arr, z_arr, r_arr, names):
            self.earray.add_electrode(name, DiskElectrode(x, y, z, r))
        
        