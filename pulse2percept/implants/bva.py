"""`BVA24`"""
import numpy as np

from .base import ProsthesisSystem
from .electrodes import DiskElectrode
from .electrode_arrays import ElectrodeArray


class BVA24(ProsthesisSystem):
    """24-channel suprachoroidal retinal prosthesis

    .. versionadded:: 0.6

    This class creates a 24-channel suprachoroidal retinal prosthesis as
    described in [Layton2014]_, where the center of the array is located
    at (x,y,z), given in microns, and the array is rotated by rotation
    angle ``rot``, given in radians.

    The array consists of:

    -   33 platinum stimulating electrodes:

        -   30 electrodes with 600um diameter (Electrodes 1-20 (except
            9, 17, 19) and Electrodes 21a-m),

        -   3 electrodes with 400um diameter (Electrodes 9, 17, 19)

    -   2 return electrodes with 2000um diameter (Electrodes 22, 23)

    Electrodes 21a-m are typically being ganged to provide an external
    ring for common ground. The center of the array is assumed to lie
    between Electrodes 7, 8, 9, and 13.

    .. note::

        Column order for electrode numbering is reversed in a left-eye
        implant.

    Parameters
    ----------
    x : float
        x coordinate of the array center (um)
    y : float
        y coordinate of the array center (um)
    z: float || array_like
        Distance of the array to the retinal surface (um). Either a list
        with 60 entries or a scalar.
    rot : float
        Rotation angle of the array (rad). Positive values denote
        counter-clock-wise (CCW) rotations in the retinal coordinate
        system.
    eye : {'LE', 'RE'}, optional, default: 'RE'
        Eye in which array is implanted.

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ()

    def __init__(self, x=0, y=0, z=0, rot=0, eye='RE', stim=None):
        self.eye = eye
        self.earray = ElectrodeArray([])
        n_elecs = 35

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
        if isinstance(z, (list, np.ndarray)):
            # Specify different height for every electrode in a list:
            z_arr = np.asarray(self.z).flatten()
            if z_arr.size != n_elecs:
                raise ValueError("If `z` is a list, it must have %d entries, "
                                 "not %d." % (n_elecs, len(z)))
        else:
            # If `z` is a scalar, choose same height for all electrodes:
            z_arr = np.ones(n_elecs, dtype=float) * z

        # the position of the electrodes 1-20, 21a-21m, R1-R2 for left eye
        if eye == 'LE':
            x_arr = np.negative(x_arr)

        # the radius of all the electrodes in the implants
        r_arr = [300.0] * n_elecs
        # the radius of electrodes 9, 17, 19 is 200.0 um
        r_arr[8] = r_arr[16] = r_arr[18] = 200.0
        # the radius of the return electrodes is 1000.0 um
        r_arr[33] = r_arr[34] = 1000.0
        # the names of the electrodes 1-20, 21a-21m, R1 and R2
        names = [str(name) for name in range(1, 21)]
        names.extend(['21a', '21b', '21c', '21d', '21e',
                      '21f', '21g', '21h', '21i', '21j',
                      '21k', '21l', '21m'])
        names.extend(['R1', 'R2'])

        # Rotate the grid:
        rotmat = np.array([np.cos(rot), -np.sin(rot),
                           np.sin(rot), np.cos(rot)]).reshape((2, 2))
        xy = np.matmul(rotmat, np.vstack((x_arr, y_arr)))
        x_arr = xy[0, :]
        y_arr = xy[1, :]

        # Apply offset to make the grid centered at (x, y):
        x_arr += x
        y_arr += y

        for x, y, z, r, name in zip(x_arr, y_arr, z_arr, r_arr, names):
            self.earray.add_electrode(name, DiskElectrode(x, y, z, r))

        # Beware of race condition: Stim must be set last, because it requires
        # indexing into self.electrodes:
        self.stim = stim
