"""`BVT24`"""
import numpy as np

from .base import ProsthesisSystem
from .electrodes import DiskElectrode
from .electrode_arrays import ElectrodeArray


class BVT24(ProsthesisSystem):
    """24-channel suprachoroidal retinal prosthesis

    This class creates a 24-channel suprachoroidal retinal prosthesis
    [Layton2014]_, which was developed by the Bionic Vision Australia
    Consortium and commercialized by Bionic Vision Technologies (BVT).
    The center of the array is located at (x,y,z), given in microns, and the
    array is rotated by rotation angle ``rot``, given in degrees.

    The array consists of:

    -   33 platinum stimulating electrodes:

        -   30 electrodes with 600um diameter (Electrodes C1-20 (except
            C9, C17, C19) and Electrodes C21a-m),

        -   3 electrodes with 400um diameter (Electrodes C9, C17, C19)

    -   2 return electrodes with 2000um diameter (Electrodes R1, R2)

    Electrodes C21a-m are typically being ganged to provide an external
    ring for common ground. The center of the array is assumed to lie
    between Electrodes C7, C8, C9, and C13.

    .. note::

        Column order for electrode numbering is reversed in a left-eye
        implant.

    .. versionadded:: 0.6

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
    __slots__ = ()

    def __init__(self, x=0, y=0, z=0, rot=0, eye='RE', stim=None,
                 preprocess=False, safe_mode=False):
        self.eye = eye
        self.preprocess = preprocess
        self.safe_mode = safe_mode
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
        # the names of the electrodes C1-20, C21a-21m, R1 and R2
        names = ["C%s" % name for name in range(1, 21)]
        names.extend(['C21a', 'C21b', 'C21c', 'C21d', 'C21e',
                      'C21f', 'C21g', 'C21h', 'C21i', 'C21j',
                      'C21k', 'C21l', 'C21m'])
        names.extend(['R1', 'R2'])

        # Rotate the grid:
        rot_rad = np.deg2rad(rot)
        rotmat = np.array([np.cos(rot_rad), -np.sin(rot_rad),
                           np.sin(rot_rad), np.cos(rot_rad)]).reshape((2, 2))
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
