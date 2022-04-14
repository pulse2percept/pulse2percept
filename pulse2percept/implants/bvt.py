"""`BVT24`, `BVT44`"""
import numpy as np
from skimage.transform import SimilarityTransform

from .base import ProsthesisSystem
from .electrodes import DiskElectrode
from .electrode_arrays import ElectrodeArray, ElectrodeGrid


class BVT24(ProsthesisSystem):
    """24-channel suprachoroidal retinal prosthesis

    This class creates a 24-channel suprachoroidal retinal prosthesis
    [Layton2014]_, which was developed by the Bionic Vision Australia
    Consortium and commercialized by Bionic Vision Technologies (BVT).
    The center of the array is located at (x,y,z), given in microns, and the
    array is rotated counter-clockwise by rotation angle ``rot``, given in 
    degrees.

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

    .. versionadded :: 0.6

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
        x_arr = np.array([1275.0, 850.0, 1275.0, 850.0, 1275.0,
                          425.0, 0, 425.0, 0, 425.0,
                          -425.0, -850.0, -425.0, -850.0, -425.0,
                          -1275.0, -1700.0, -1275.0, -1700.0, -1275.0,
                          850.0, 0, -850.0, -1700.0, -2125.0,
                          -2550.0, -2125.0, -2550.0, -2125.0, -1700.0,
                          -850.0, 0, 850.0, -7000.0, -9370.0])
        y_arr = np.array([1520.0, 760.0, 0, -760.0, -1520.0,
                          1520.0, 760.0, 0, -760.0, -1520.0,
                          1520.0, 760.0, 0, -760.0, -1520.0,
                          1520.0, 760.0, 0, -760.0, -1520.0,
                          2280.0, 2280.0, 2280.0, 2280.0, 1520.0,
                          760.0, 0.0, -760.0, -1520.0, -2280.0,
                          -2280.0, -2280.0, -2280.0, 0, 0])
        if isinstance(z, (list, np.ndarray)):
            # Specify different height for every electrode in a list:
            z_arr = np.asarray(self.z).flatten()
            if z_arr.size != n_elecs:
                raise ValueError(f"If `z` is a list, it must have {n_elecs} entries, "
                                 f"not {len(z)}.")
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
        names = [f"C{name}" for name in range(1, 21)]
        names.extend(['C21a', 'C21b', 'C21c', 'C21d', 'C21e',
                      'C21f', 'C21g', 'C21h', 'C21i', 'C21j',
                      'C21k', 'C21l', 'C21m'])
        names.extend(['R1', 'R2'])

        # Rotate the grid and center at (x,y):
        tf = SimilarityTransform(rotation=np.deg2rad(rot), translation=[x, y])
        x_arr, y_arr = tf(np.vstack([x_arr.ravel(), y_arr.ravel()]).T).T

        for x, y, z, r, name in zip(x_arr, y_arr, z_arr, r_arr, names):
            self.earray.add_electrode(name, DiskElectrode(x, y, z, r))

        # Beware of race condition: Stim must be set last, because it requires
        # indexing into self.electrodes:
        self.stim = stim


class BVT44(ProsthesisSystem):
    """    44-channel suprachoroidal retinal prosthesis

    This class creates a 44-channel suprachoroidal retinal prosthesis
    [Petoe2021]_, which was developed by the Bionic Vision Australia
    Consortium and commercialized by Bionic Vision Technologies (BVT).

    The center of the array (x,y,z) is located at the center of electrodes
    D4, D5, C4, and E4, and the  array is rotated counter-clockwise by rotation
    angle ``rot``, given in degrees.

    The array consists of:

    -   44 platinum stimulating electrodes with 1000um exposed diameter

    -   2 return electrodes with 2000um diameter (Electrodes R1, R2)

    The position of each electrode is measured from Figure 7 in [Petoe2021]_.

    .. note::

        Column order for electrode numbering is reversed in a left-eye
        implant.

    .. versionadded :: 0.8

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

    def __init__(self, x=0, y=0, z=0, rot=0, eye='LE', stim=None,
                 preprocess=False, safe_mode=False):
        self.eye = eye
        self.preprocess = preprocess
        self.safe_mode = safe_mode
        self.earray = ElectrodeArray([])
        n_elecs = 46

        # The 44 stimulating electrodes are arranged in a hex grid; two return
        # electrodes are added as well:
        grid = ElectrodeGrid((7, 7), (1320, 925), type='hex', x=-330)
        for e in ['D1', 'A7', 'C7', 'E7', 'G7']:
            grid.remove_electrode(e)
        x_arr = np.array([e.x for e in grid.electrode_objects] + [-7000, -7000])
        y_arr = np.array([e.y for e in grid.electrode_objects] + [1500, -1500])
        r_arr = [500.0] * grid.n_electrodes + [1000.0, 1000.0]
        names = grid.electrode_names + ['R1', 'R2']
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

        # Rotate the grid and center at (x,y):
        tf = SimilarityTransform(rotation=np.deg2rad(rot), translation=[x, y])
        x_arr, y_arr = tf(np.vstack([x_arr.ravel(), y_arr.ravel()]).T).T

        for x, y, z, r, name in zip(x_arr, y_arr, z_arr, r_arr, names):
            self.earray.add_electrode(name, DiskElectrode(x, y, z, r))

        # Beware of race condition: Stim must be set last, because it requires
        # indexing into self.electrodes:
        self.stim = stim
