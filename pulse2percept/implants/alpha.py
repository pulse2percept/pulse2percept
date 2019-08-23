"""Alpha IMS"""
import numpy as np
import collections as coll
from .base import DiskElectrode, ElectrodeArray, ElectrodeGrid, ProsthesisSystem


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

        shape = (37, 37)
        elec_radius = 50
        e_spacing = 72  # um

        #names = ('A', '1')
        self.earray = ElectrodeGrid(shape, x=x, y=y, z=z, rot=rot, r=elec_radius,
                                    spacing=e_spacing)

        # Set stimulus if available:
        self.stim = stim

        # Set left/right eye:
        if not isinstance(eye, str):
            raise TypeError("'eye' must be a string, either 'LE' or 'RE'.")
        if eye != 'LE' and eye != 'RE':
            raise ValueError("'eye' must be either 'LE' or 'RE'.")
        # Unfortunately, in the left eye the labeling of columns is reversed...
        if eye == 'LE':
            # FIXME: Would be better to have more flexibility in the naming
            # convention. This is a quick-and-dirty fix:
            names = list(self.earray.keys())
            objects = list(self.earray.values())
            names = np.array(names).reshape(self.earray.shape)
            # Reverse column names:
            for row in range(self.earray.shape[0]):
                names[row] = names[row][::-1]
            # Build a new ordered dict:
            electrodes = coll.OrderedDict([])
            for name, obj in zip(names.ravel(), objects):
                electrodes.update({name: obj})
            # Assign the new ordered dict to earray:
            self.earray.electrodes = electrodes


class AlphaAMS(ProsthesisSystem):

    def __init__(self, x=0, y=0, z=0, rot=0, eye='RE', stim=None):
        """Alpha AMS

        This function creates an AlphaAMS array and places it below the retina
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
        Create an AlphaAMS array centered on the fovea, at 100um distance from
        the retina:
        >>> from pulse2percept import implants
        >>> alpha_ams = implants.AlphaAMS(x=0, y=0, z=100, rot=0)

        Get access to the third electrode:
        >>> my_electrode = alpha_ams[2]
        """
        self.eye = eye

        # array dimensions
        shape = (40, 40)
        elec_radius = 15
        e_spacing = 70  # um

        self.earray = ElectrodeGrid(shape, x=x, y=y, z=z, rot=rot, r=elec_radius,
                                    spacing=e_spacing)

        # Set stimulus if available:
        self.stim = stim

        # Set left/right eye:
        if not isinstance(eye, str):
            raise TypeError("'eye' must be a string, either 'LE' or 'RE'.")
        if eye != 'LE' and eye != 'RE':
            raise ValueError("'eye' must be either 'LE' or 'RE'.")
        # Unfortunately, in the left eye the labeling of columns is reversed...
        if eye == 'LE':
            # FIXME: Would be better to have more flexibility in the naming
            # convention. This is a quick-and-dirty fix:
            names = list(self.earray.keys())
            objects = list(self.earray.values())
            names = np.array(names).reshape(self.earray.shape)
            # Reverse column names:
            for row in range(self.earray.shape[0]):
                names[row] = names[row][::-1]
            # Build a new ordered dict:
            electrodes = coll.OrderedDict([])
            for name, obj in zip(names.ravel(), objects):
                electrodes.update({name: obj})
            # Assign the new ordered dict to earray:
            self.earray.electrodes = electrodes
