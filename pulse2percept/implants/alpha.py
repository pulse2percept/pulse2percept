"""Alpha IMS"""
import numpy as np
from .base import DiskElectrode, ElectrodeArray, ProsthesisSystem


class AlphaIMS(ProsthesisSystem):

    def __init__(self, x=0, y=0, z=0, rot=0, eye='RE', stim=None):
        """Alpha IMS

        This function creates an AlphaIMS array and places it on the retina
        such that the center of the array is located at (x,y,z), given in
        microns, and the array is rotated by rotation angle `rot`, given in
        radians.

        The array is oriented upright in the visual field, such that an
        array with center (0,0) has the top three rows lie in the lower
        retina (upper visual field), as shown below:

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
        Create an AlphaIMS array centered on the fovea, at 100um distance from
        the retina:
        >>> from pulse2percept import implants
        >>> alpha_ims = implants.AlphaIMS(x=0, y=0, z=100, rot=0)

        Get access to the third electrode:
        >>> my_electrode = alpha_ims[2]
        """
        self.eye = eye

        # Electrode spacing, radius
        e_spacing = 72  # um
        elec_radius = 50
        # number of electrodes horizontally, vertically, and total
        n_cols = 37
        n_rows = 37
        n_elecs = n_cols * n_rows

        # TODO: look up naming convention
        names = np.arange(n_elecs)

        # array containing electrode radii (uniform)
        r_arr = np.full(shape=n_elecs, fill_value=elec_radius)

        if isinstance(z, (list, np.ndarray)):
            z_arr = np.asarray(z).flatten()
            if z_arr.size != len(r_arr):
                e_s = "If `h` is a list, it must have %d entries." % n_elecs
                raise ValueError(e_s)
        else:
            # All electrodes have the same height
            z_arr = np.ones_like(r_arr) * z

        # arrays of x and y coordinates
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

        self.earray = ElectrodeArray([])
        for x, y, z, r, name in zip(x_arr, y_arr, z_arr, r_arr, names):
            self.earray.add_electrode(name, DiskElectrode(x, y, z, r))
        self.stim = stim
