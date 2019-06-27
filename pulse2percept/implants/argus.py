import numpy as np
from pulse2percept import utils
from pulse2percept.implants import DiskElectrode, ElectrodeArray, ProsthesisSystem


class ArgusI(ProsthesisSystem):

    def __init__(self, x=0, y=0, z=0, rot=0, eye='RE', stim=None,
                 use_legacy_names=False):
        """Create an ArgusI array on the retina
        This function creates an ArgusI array and places it on the retina
        such that the center of the array is located at 3D location (x,y,z),
        given in microns, and the array is rotated by rotation angle `rot`,
        given in radians.

        The array is oriented in the visual field as shown in Fig. 1 of
        Horsager et al. (2009); that is, if placed in (0,0), the top two
        rows will lie in the lower retina (upper visual field):

        .. raw:: html
          <pre>
            y       A1 B1 C1 D1                     260 520 260 520
            ^       A2 B2 C2 D2   where electrode   520 260 520 260
            |       A3 B3 C3 D3   diameters are:    260 520 260 520
            -->x    A4 B4 C4 D4                     520 260 520 260
          </pre>

        Electrode order is: A1, B1, C1, D1, A2, B2, ..., D4.
        If `use_legacy_names` is True, electrode order is: L6, L2, M8, M4, ...
        An electrode can be addressed by index (integer) or name.

        Parameters
        ----------
        x : float, optional, default: 0
            x coordinate of the array center (um)
        y : float, optional, default: 0
            y coordinate of the array center (um)
        z : float || array_like, optional, default: 0
            Distance of the array to the retinal surface (um). Either a list
            with 16 entries or a scalar.
        rot : float, optional, default: 0
            Rotation angle of the array (rad). Positive values denote
            counter-clock-wise (CCW) rotations in the retinal coordinate
            system.
        eye : {'LE', 'RE'}, optional, default: 'RE'
            Eye in which array is implanted.

        Examples
        --------
        Create an ArgusI array centered on the fovea, at 100um distance from
        the retina:
        >>> from pulse2percept import implants
        >>> argus = implants.ArgusI(x=0, y=0, z=100, rot=0)

        Get access to electrode 'B1':
        >>> my_electrode = argus['B1']
        """
        # Alternating electrode sizes, arranged in checkerboard pattern
        r_arr = np.array([260, 520, 260, 520]) / 2.0
        r_arr = np.concatenate((r_arr, r_arr[::-1], r_arr, r_arr[::-1]),
                               axis=0)

        # Set left/right eye
        self.eye = eye

        # In older papers, Argus I electrodes go by L and M
        self.old_names = names = ['L6', 'L2', 'M8', 'M4',
                                  'L5', 'L1', 'M7', 'M3',
                                  'L8', 'L4', 'M6', 'M2',
                                  'L7', 'L3', 'M5', 'M1']
        # In newer papers, they go by A-D: A1, B1, C1, D1, A1, B2, ..., D4
        # Shortcut: Use `chr` to go from int to char
        self.new_names = [chr(i) + str(j) for j in range(1, 5)
                          for i in range(65, 69)]
        names = self.old_names if use_legacy_names else self.new_names

        if isinstance(z, (list, np.ndarray)):
            z_arr = np.asarray(z).flatten()
            if z_arr.size != len(r_arr):
                e_s = "If `z` is a list, it must have 16 entries."
                raise ValueError(e_s)
        else:
            # All electrodes have the same height
            z_arr = np.ones_like(r_arr) * z

        # Equally spaced electrodes: n_rows x n_cols = 16
        e_spacing = 800  # um
        n_cols = 4  # number of electrodes horizontally (same vertically)
        x_arr = np.arange(n_cols) * e_spacing - (n_cols / 2 - 0.5) * e_spacing
        if self.eye == 'LE':
            # Left eye: Need to invert x coordinates and rotation angle
            x_arr = x_arr[::-1]
        x_arr, y_arr = np.meshgrid(x_arr, x_arr, sparse=False)

        # Rotation matrix
        R = np.array([np.cos(rot), -np.sin(rot),
                      np.sin(rot), np.cos(rot)]).reshape((2, 2))

        # Set the x, y location of the tack
        if self.eye == 'RE':
            self.tack = np.matmul(R, [-(n_cols / 2 + 0.5) * e_spacing, 0])
        else:
            self.tack = np.matmul(R, [(n_cols / 2 + 0.5) * e_spacing, 0])
        self.tack = tuple(self.tack + [x, y])

        # Rotate the array
        xy = np.vstack((x_arr.flatten(), y_arr.flatten()))
        xy = np.matmul(R, xy)
        x_arr = xy[0, :]
        y_arr = xy[1, :]

        # Apply offset
        x_arr += x
        y_arr += y

        self.array = ElectrodeArray([])
        for x, y, z, r, name in zip(x_arr, y_arr, z_arr, r_arr, names):
            self.array.add_electrode(name, DiskElectrode(x, y, z, r))
        self.stim = None

    def get_old_name(self, new_name):
        """Look up the legacy name of a standard-named Argus I electrode"""
        return self.old_names[self.new_names.index(new_name)]

    def get_new_name(self, old_name):
        """Look up the standard name of a legacy-named Argus I electrode"""
        return self.new_names[self.old_names.index(old_name)]


class ArgusII(ProsthesisSystem):

    def __init__(self, x=0, y=0, z=0, rot=0, eye='RE'):
        """Create an ArgusII array on the retina

        This function creates an ArgusII array and places it on the retina
        such that the center of the array is located at (x,y,z), given in
        microns, and the array is rotated by rotation angle `rot`, given in
        radians.

        The array is oriented upright in the visual field, such that an
        array with center (0,0) has the top three rows lie in the lower
        retina (upper visual field), as shown below:

        .. raw:: html
          <pre>
                    A1 A2 A3 A4 A5 A6 A7 A8 A9 A10
            y       B1 B2 B3 B4 B5 B6 B7 B8 B9 B10
            ^       C1 C2 C3 C4 C5 C6 C7 C8 C9 C10
            |       D1 D2 D3 D4 D5 D6 D7 D8 D9 D10
            -->x    E1 E2 E3 E4 E5 E6 E7 E8 E9 E10
                    F1 F2 F3 F4 F5 F6 F7 F8 F9 F10
          </pre>

        Electrode order is: A1, A2, ..., A10, B1, B2, ..., F10.
        An electrode can be addressed by index (integer) or name.

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

        Examples
        --------
        Create an ArgusII array centered on the fovea, at 100um distance from
        the retina:
        >>> from pulse2percept import implants
        >>> argus = implants.ArgusII(x=0, y=0, z=100, rot=0)

        Get access to electrode 'E7':
        >>> my_electrode = argus['E7']
        """
        # Electrodes are 200um in diameter
        r_arr = np.ones(60) * 100.0

        # Set left/right eye
        self.eye = eye

        # Standard ArgusII names: A1, A2, ..., A10, B1, ..., F10
        names = [chr(i) + str(j) for i in range(65, 71) for j in range(1, 11)]

        if isinstance(z, (list, np.ndarray)):
            z_arr = np.asarray(z).flatten()
            if z_arr.size != len(r_arr):
                e_s = "If `h` is a list, it must have 60 entries."
                raise ValueError(e_s)
        else:
            # All electrodes have the same height
            z_arr = np.ones_like(r_arr) * z

        # Equally spaced electrodes: n_rows x n_cols = 60
        e_spacing = 525  # um
        n_cols = 10  # number of electrodes horizontally
        n_rows = 6  # number of electrodes vertically
        x_arr = np.arange(n_cols) * e_spacing - (n_cols / 2 - 0.5) * e_spacing
        if self.eye == 'LE':
            # Left eye: Need to invert x coordinates and rotation angle
            x_arr = x_arr[::-1]
        y_arr = np.arange(n_rows) * e_spacing - (n_rows / 2 - 0.5) * e_spacing
        x_arr, y_arr = np.meshgrid(x_arr, y_arr, sparse=False)

        # Rotation matrix
        rotmat = np.array([np.cos(rot), -np.sin(rot),
                           np.sin(rot), np.cos(rot)]).reshape((2, 2))

        # Set the x, y location of the tack
        if self.eye == 'RE':
            self.tack = np.matmul(rotmat, [-(n_cols / 2 + 0.5) * e_spacing, 0])
        else:
            self.tack = np.matmul(rotmat, [(n_cols / 2 + 0.5) * e_spacing, 0])
        self.tack = tuple(self.tack + [x, y])

        # Rotate the array
        xy = np.vstack((x_arr.flatten(), y_arr.flatten()))
        xy = np.matmul(rotmat, xy)
        x_arr = xy[0, :]
        y_arr = xy[1, :]

        # Apply offset
        x_arr += x
        y_arr += y

        self.array = ElectrodeArray([])
        for x, y, z, r, name in zip(x_arr, y_arr, z_arr, r_arr, names):
            self.array.add_electrode(name, DiskElectrode(x, y, z, r))
        self.stim = None
