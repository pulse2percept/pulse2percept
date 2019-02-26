# -*implants -*-
"""

Functions for creating retinal implants

"""
import numpy as np
import logging

from pulse2percept import utils


SUPPORTED_IMPLANT_TYPES = ['epiretinal', 'subretinal']


class Electrode(object):

    def __init__(self, etype, radius, x_center, y_center, height=0, name=None):
        """Create an electrode on the retina

        This function creates a disk electrode of type `etype` and places it
        on the retina at location (`xs`, `ys`) in microns. The electrode has
        radius `radius` (microns) and sits a distance `height` away from the
        retinal surface.
        The coordinate system is anchored around the fovea at (0, 0).

        Parameters
        ----------
        etype : str
            Electrode type, {'epiretinal', 'subretinal'}
        radius : float
            The radius of the electrode (in microns).
        x_center : float
            The x coordinate of the electrode center (in microns) from the
            fovea.
        y_center : float
            The y location of the electrode (in microns) from the fovea
        height : float
            The height of the electrode from the retinal surface:

            - epiretinal array: distance to the ganglion layer
            - subretinal array: distance to the bipolar layer
        name : string
            Electrode name

        """
        assert radius >= 0
        assert height >= 0

        if etype.lower() not in SUPPORTED_IMPLANT_TYPES:
            e_s = "Acceptable values for `etype` are: "
            e_s += ", ".join(SUPPORTED_IMPLANT_TYPES) + "."
            raise ValueError(e_s)

        self.etype = etype.lower()
        self.radius = radius
        self.x_center = x_center
        self.y_center = y_center
        self.name = name
        self.height = height

    def __str__(self):
        info_s = "Electrode(%s, r=%.2f um, " % (self.etype, self.radius)
        info_s += "(x,y) = (%.2f, %.2f) um, " % (self.x_center, self.y_center)
        info_s += "h=%.2f um, n=%s" % (self.height, self.name)
        return info_s

    def get_height(self):
        """Returns the electrode-retina distance

        For epiretinal electrodes, this returns the distance to the ganglion
        cell layer.
        For subretinal electrodes, this returns the distance to the bipolar
        layer.
        """
        if self.etype == 'epiretinal':
            return self.h_ofl
        elif self.etype == 'subretinal':
            return self.h_inl
        else:
            raise ValueError("Unknown `etype`: " + self.etype)

    def set_height(self, height):
        """Sets the electrode-to-retina distance

        This function sets the electrode-to-retina distance according to
        `height`. For an epiretinal device, we calculate the distance to
        the ganglion cell layer (layer thickness depends on retinal location).
        For a subretinal device, we calculate the distance to the bipolar
        layer (layer thickness again depends on retinal location).

        Estimates of layer thickness based on:
        LoDuca et al. Am J. Ophthalmology 2011
        Thickness Mapping of Retinal Layers by Spectral Domain Optical
        Coherence Tomography
        Note that this is for normal retinal, so may overestimate thickness.
        Thickness from their paper (averaged across quadrants):
            0-600 um radius (from fovea):

            - Layer 1. (Nerve fiber layer) = 4
            - Layer 2. (Ganglion cell bodies + inner plexiform) = 56
            - Layer 3. (Bipolar bodies, inner nuclear layer) = 23

          600-1550 um radius:

            - Layer 1. 34
            - Layer 2. 87
            - Layer 3. 37.5

          1550-3000 um radius:
            - Layer 1. 45.5
            - Layer 2. 58.2
            - Layer 3. 30.75

        We place our ganglion axon surface on the inner side of the nerve fiber
        layer.
        We place our bipolar surface 1/2 way through the inner nuclear layer.
        So for an epiretinal array the bipolar layer is L1 + L2 + 0.5 * L3.

        """
        fovdist = np.sqrt(self.x_center ** 2 + self.y_center ** 2)
        if fovdist <= 600:
            # Layer thicknesses given for 0-600 um distance (from fovea)
            th_ofl = 4.0  # nerve fiber layer
            th_gc = 56.0  # ganglion cell bodies + inner nuclear layer
            th_bp = 23.0  # bipolar bodies + inner nuclear layer
        elif fovdist <= 1550:
            # Layer thicknesses given for 600-1550 um distance (from fovea)
            th_ofl = 34.0
            th_gc = 87.0
            th_bp = 37.5
        else:
            # Layer thicknesses given for 1550-3000 um distance (from fovea)
            th_ofl = 45.5
            th_gc = 58.2
            th_bp = 30.75
            if fovdist > 3000:
                e_s = "Distance to fovea=%.0f > 3000 um, " % fovdist
                e_s += "assuming same layer thicknesses as for 1550-3000 um "
                e_s += "distance."
                logging.getLogger(__name__).warning(e_s)

        if self.etype == 'epiretinal':
            # This is simply the electrode-retina distance
            self.h_ofl = height

            # All the way through the ganglion cell layer, inner plexiform
            # layer, and halfway through the inner nuclear layer
            self.h_inl = height + th_ofl + th_gc + 0.5 * th_bp
        elif self.etype == 'subretinal':
            # Starting from the outer plexiform layer, go halfway through the
            # inner nuclear layer
            self.h_inl = height + 0.5 * th_bp

            # Starting from the outer plexiform layer, all the way through the
            # inner nuclear layer, inner plexiform layer, and ganglion cell
            # layer
            self.h_ofl = height + th_bp + th_gc + th_ofl
        else:
            raise ValueError("Unknown `etype`: " + self.etype)
    height = property(get_height, set_height)

    def current_spread(self, xg, yg, layer, alpha=14000, n=1.69):
        """

        The current spread due to a current pulse through an electrode,
        reflecting the fall-off of the current as a function of distance from
        the electrode center. This can be calculated for any layer in the
        retina.
        Based on equation 2 in Nanduri et al [1].

        Parameters
        ----------
        xg : array
            x-coordinates of the retinal grid
        yg : array
            y-coordinates of the retinal grid
        layer: str
            Layer for which to calculate the current spread:

            - 'OFL': optic fiber layer, ganglion axons
            - 'INL': inner nuclear layer, containing the bipolars
        alpha : float
            A constant to do with the spatial fall-off.

        n : float
            A constant to do with the spatial fall-off (Default: 1.69, based
            on Ahuja et al. [2]  An In Vitro Model of a Retinal Prosthesis.
            Ashish K. Ahuja, Matthew R. Behrend, Masako Kuroda, Mark S.
            Humayun, and James D. Weiland (2008). IEEE Trans Biomed Eng 55.

        """
        r = np.sqrt((xg - self.x_center) ** 2 + (yg - self.y_center) ** 2)
        # current values on the retina due to array being above the retinal
        # surface
        if 'OFL' in layer:  # optic fiber layer, ganglion axons
            h = np.ones(r.shape) * self.h_ofl
            # actual distance from the electrode edge
            d = ((r - self.radius)**2 + self.h_ofl**2)**.5
        elif 'INL' in layer:  # inner nuclear layer, containing the bipolars
            h = np.ones(r.shape) * self.h_inl
            d = ((r - self.radius)**2 + self.h_inl**2)**.5
        else:
            s = "Layer %s not found. Acceptable values for `layer` are " \
                "'OFL' or 'INL'." % layer
            raise ValueError(s)
        cspread = (alpha / (alpha + h ** n))
        cspread[r > self.radius] = (alpha
                                   / (alpha + d[r > self.radius] ** n))

        return cspread

    def receptive_field(self, xg, yg, rftype='square', size=None):
        """An electrode's receptive field

        Parameters
        ----------
        xg : array_like
            Array of all x coordinates
        yg : array_like
            Array of all y coordinates
        rftype : {'square', 'gaussian'}
            The type of receptive field.
            - 'square': A simple square box receptive field with side length
                        `size`.
            - 'gaussian': A Gaussian receptive field where the weight drops off
                          as a function of distance from the electrode center.
                          The standard deviation of the Gaussian is `size`.
        size : float, optional
            Parameter describing the size of the receptive field. For square
            receptive fields, this corresponds to the side length of the
            square.
            For Gaussian receptive fields, this corresponds to the standard
            deviation of the Gaussian.
            Default: Twice the electrode radius.
        """
        if size is None:
            size = 2 * self.radius

        if rftype == 'square':
            # Create a map of the retina for each electrode
            # where it's 1 under the electrode, 0 elsewhere
            rf = np.zeros(xg.shape).astype(np.float32)
            ind = np.where((xg > self.x_center - (size / 2.0))
                           & (xg < self.x_center + (size / 2.0))
                           & (yg > self.y_center - (size / 2.0))
                           & (yg < self.y_center + (size / 2.0)))
            rf[ind] = 1.0
        elif rftype == 'gaussian':
            # Create a map of the retina where the weight drops of as a
            # function of distance from the electrode center
            dist = (xg - self.x_center) ** 2 + (yg - self.y_center) ** 2
            rf = np.exp(-dist / (2 * size ** 2))
            rf /= np.sum(rf)
        else:
            e_s = "Acceptable values for `rftype` are 'square' or 'gaussian'"
            raise ValueError(e_s)

        return rf


class ElectrodeArray(object):

    def __init__(self, etype, radii, xs, ys, hs=0, names=None, eye='RE'):
        """Create an ElectrodeArray on the retina
        This function creates an electrode array of type `etype` and places it
        on the retina. Lists should specify, for each electrode, its size
        (`radii`), location on the retina (`xs` and `ys`), distance to the
        retina (height, `hs`), and a string identifier (`names`, optional).
        Array location should be given in microns, where the fovea is located
        at (0, 0).
        Single electrodes in the array can be addressed by index (integer)
        or name.
        Parameters
        ----------
        radii : array_like
            List of electrode radii.
        xs : array_like
            List of x-coordinates for the center of the electrodes (microns).
        ys : array_like
            List of y-coordinates for the center of the electrodes (microns).
        hs : float | array_like, optional, default: 0
            List of electrode heights (distance from the retinal surface).
        names : array_like, optional, default: None
            List of names (string identifiers) for each eletrode.
        eye : {'LE', 'RE'}, optional, default: 'RE'
            Eye in which array is implanted.

        Examples
        --------
        A single epiretinal electrode called 'A1', with radius 100um, sitting
        at retinal location (0, 0), 10um away from the retina:
        >>> from pulse2percept import implants
        >>> implant0 = implants.ElectrodeArray('epiretinal', 100, 0, 0, hs=10,
        ...                                    names='A1')

        Get access to the electrode with name 'A1' in the first array:
        >>> my_electrode = implant0['A1']

        An array with two electrodes of size 100um, one sitting at
        (-100, -100), the other sitting at (0, 0), with 0 distance from the
        retina, of type 'subretinal':
        >>> implant1 = implants.ElectrodeArray('subretinal', [100, 100],
        ...                                    [-100, 0], [-100, 0], hs=[0, 0])
        """
        self.etype = etype
        self.eye = eye
        self.electrodes = []
        self.num_electrodes = 0
        self.add_electrodes(radii, xs, ys, hs, names)

    def __str__(self):
        return "ElectrodeArray(%s, num_electrodes=%d)" % (self.etype,
                                                          self.num_electrodes)

    def add_electrode(self, electrode):
        """Adds an electrode to an ElectrodeArray object
        This function adds a single electrode to an existing ElectrodeArray
        object. The electrode must have the same type as the array
        (see implants.SUPPORTED_IMPLANT_TYPES).
        Parameters
        ----------
        electrode : implants.Electrode
            An electrode object specifying type, size, and location of the
            electrode on the retina.
        """
        if not isinstance(electrode, Electrode):
            raise TypeError("`electrode` must be of type retina.Electrode.")

        if electrode.etype != self.etype:
            e_s = "Added electrode must be of same type as the existing"
            e_s = "array (%s)." % self.etype
            raise ValueError(e_s)

        self.num_electrodes += 1
        self.electrodes.append(electrode)

    def add_electrodes(self, radii, xs, ys, hs=0, names=None):
        """Adds electrodes to an ElectrodeArray object
        This function adds one or more electrodes to an existing ElectrodeArray
        object. Lists should specify, for each electrode to be added, the size
        (`radii`), location on the retina (`xs` and `ys`), distance to the
        retina (height, `hs`), and a string identifier (`names`, optional).
        Array location should be given in microns, where the fovea is located
        at (0, 0).
        Single electrodes in the array can be addressed by index (integer)
        or name.

        Parameters
        ----------
        radii : array_like
            List of electrode radii.
        xs : array_like
            List of x-coordinates for the center of the electrodes (microns).
        ys : array_like
            List of y-coordinates for the center of the electrodes (microns).
        hs : float | array_like, optional, default: 0
            List of electrode heights (distance from the retinal surface).
        names : array_like, optional, default: None
            List of names (string identifiers) for each eletrode.

        Examples
        --------
        Adding a single electrode of radius 50um sitting at (0, 0) to an
        existing ElectrodeArray object:
        >>> implant = ElectrodeArray('epiretinal', 100, 100, 100)
        >>> implant.add_electrodes(50, 0, 0)
        """
        # Make it so the method can accept either floats, lists, or
        # numpy arrays, and `zip` works regardless.
        radii = np.array([radii], dtype=np.float32).flatten()
        xs = np.array([xs], dtype=np.float32).flatten()
        ys = np.array([ys], dtype=np.float32).flatten()
        names = np.array([names], dtype=np.str).flatten()

        if isinstance(hs, list):
            hs = np.array(hs).flatten()
        else:
            # All electrodes have the same height
            hs = np.ones_like(radii) * hs

        assert radii.size == xs.size == ys.size == hs.size

        if names.size != radii.size:
            # If not every electrode has a name, replace with None's
            names = np.array([None] * radii.size)

        for r, x, y, h, n in zip(radii, xs, ys, hs, names):
            self.add_electrode(Electrode(self.etype, r, x, y, h, n))

    def __iter__(self):
        return iter(self.electrodes)

    def __getitem__(self, item):
        """Return the electrode specified by `item`
        Parameters
        ----------
        item : int|string
            If `item` is an integer, returns the `item`-th electrode in the
            array. If `item` is a string, returns the electrode with string
            identifier `item`.
        """
        try:
            # Is `item` an integer?
            return self.electrodes[item]
        except (IndexError, TypeError):
            # If `item` is a valid string identifier, return valid index.
            # Else return None
            try:
                return self.electrodes[self.get_index(item)]
            except (IndexError, TypeError):
                return None

    def get_index(self, name):
        """Returns the index of an electrode called `name`
        This function searches the electrode array for an electrode with
        string identifier `name`. If found, the index of that electrode is
        returned, else None.
        Parameters
        ----------
        name : str
            An electrode name (string identifier).
        Returns
        -------
        A valid electrode index or None.
        """
        # Is `name` a valid electrode name?
        # Iterate through electrodes to find a matching name. Shuffle list
        # to reduce time complexity of average lookup.
        for idx, el in utils.traverse_randomly(enumerate(self.electrodes)):
            if el.name == name:
                return idx

        # Worst case O(n): name could not be found.
        return None

    def get_eye(self):
        return self._eye

    def set_eye(self, eye):
        if eye.lower() in ['r', 're', 'right']:
            self._eye = 'RE'
        elif eye.lower() in ['l', 'le', 'left']:
            self._eye = 'LE'
        else:
            raise ValueError("Unknown eye '%s'. Choose from 'LE', 'RE'.")

    eye = property(get_eye, set_eye)


class ArgusI(ElectrodeArray):

    def __init__(self, x_center=0, y_center=0, h=0, rot=0, eye='RE',
                 use_legacy_names=False):
        """Create an ArgusI array on the retina
        This function creates an ArgusI array and places it on the retina
        such that the center of the array is located at
        [`x_center`, `y_center`] (microns) and the array is rotated by
        rotation angle `rot` (radians).
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
        x_center : float, optional, default: 0
            x coordinate of the array center (um)
        y_center : float, optional, default: 0
            y coordinate of the array center (um)
        h : float || array_like, optional, default: 0
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
        >>> argus = implants.ArgusI(x_center=0, y_center=0, h=100, rot=0)

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

        if use_legacy_names:
            names = self.old_names
        else:
            names = self.new_names

        if isinstance(h, list):
            h_arr = np.array(h).flatten()
            if h_arr.size != len(r_arr):
                e_s = "If `h` is a list, it must have 16 entries."
                raise ValueError(e_s)
        else:
            # All electrodes have the same height
            h_arr = np.ones_like(r_arr) * h

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
        self.tack = tuple(self.tack + [x_center, y_center])

        # Rotate the array
        xy = np.vstack((x_arr.flatten(), y_arr.flatten()))
        xy = np.matmul(R, xy)
        x_arr = xy[0, :]
        y_arr = xy[1, :]

        # Apply offset
        x_arr += x_center
        y_arr += y_center

        self.etype = 'epiretinal'
        self.num_electrodes = 0
        self.electrodes = []
        for r, x, y, h, n in zip(r_arr, x_arr, y_arr, h_arr, names):
            self.add_electrode(Electrode(self.etype, r, x, y, h, n))

    def __str__(self):
        return "ArgusI(%s, num_electrodes=%d)" % (self.etype,
                                                  self.num_electrodes)

    def get_old_name(self, new_name):
        """Look up the legacy name of a standard-named Argus I electrode"""
        return self.old_names[self.new_names.index(new_name)]

    def get_new_name(self, old_name):
        """Look up the standard name of a legacy-named Argus I electrode"""
        return self.new_names[self.old_names.index(old_name)]


class ArgusII(ElectrodeArray):

    def __init__(self, x_center=0, y_center=0, h=0, rot=0, eye='RE'):
        """Create an ArgusII array on the retina
        This function creates an ArgusII array and places it on the retina
        such that the center of the array is located at
        [`x_center`, `y_center`] (microns) and the array is rotated by
        rotation angle `rot` (radians).
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
        x_center : float
            x coordinate of the array center (um)
        y_center : float
            y coordinate of the array center (um)
        h : float || array_like
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
        >>> argus = implants.ArgusII(x_center=0, y_center=0, h=100, rot=0)

        Get access to electrode 'E7':
        >>> my_electrode = argus['E7']
        """
        # Electrodes are 200um in diameter
        r_arr = np.ones(60) * 100.0

        # Set left/right eye
        self.eye = eye

        # Standard ArgusII names: A1, A2, ..., A10, B1, ..., F10
        names = [chr(i) + str(j) for i in range(65, 71) for j in range(1, 11)]

        if isinstance(h, list):
            h_arr = np.array(h).flatten()
            if h_arr.size != len(r_arr):
                e_s = "If `h` is a list, it must have 60 entries."
                raise ValueError(e_s)
        else:
            # All electrodes have the same height
            h_arr = np.ones_like(r_arr) * h

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
        self.tack = tuple(self.tack + [x_center, y_center])

        # Rotate the array
        xy = np.vstack((x_arr.flatten(), y_arr.flatten()))
        xy = np.matmul(rotmat, xy)
        x_arr = xy[0, :]
        y_arr = xy[1, :]

        # Apply offset
        x_arr += x_center
        y_arr += y_center

        self.etype = 'epiretinal'
        self.num_electrodes = 0
        self.electrodes = []
        for r, x, y, h, n in zip(r_arr, x_arr, y_arr, h_arr, names):
            self.add_electrode(Electrode(self.etype, r, x, y, h, n))

    def __str__(self):
        return "ArgusII(%s, num_electrodes=%d)" % (self.etype,
                                                   self.num_electrodes)


class AlphaIMS(ElectrodeArray):

    def __init__(self, x_center=0, y_center=0, h=0, rot=0, eye='RE'):
        """Create an Alpha IMS array on the retina and place it on the retina
        such that the center of the array is located at [`x_center`, `y_center`]
        (microns) and the array is rotated by rotation angle `rot` (radians).
        The array is oriented upright in the visual field, such that an array
        with center (0,0) has the top three rows lie in the lower retina
        (upper visual field), as shown below:

        Parameters
        ----------
        x_center : float
            x coordinate of the array center (um)
        y_center : float
            y coordinate of the array center (um)
        h : float || array_like
            Distance of the array to the retinal surface (um). Either a list
            with 60 entries or a scalar.
        rot : float
            Rotation angle of the array (rad). Positive values denote
            counter-clock-wise (CCW) rotations in the retinal coordinate
            system.
        eye : {'LE', 'RE'}, optional, default: 'RE'
            Eye in which array is implanted.
        """

        self.eye = eye
        self.etype = 'subretinal'

        # Electrode spacing, radius
        e_spacing = 72  # um
        elec_radius = 50
        # number of electrodes horizontally, vertically, and total
        n_cols = 37
        n_rows = 37
        n_elecs = n_cols * n_rows

        # TODO: look up naming convention
        names = np.ones(n_elecs)

        # array containing electrode radii (uniform)
        r_arr = np.full(shape=n_elecs, fill_value=elec_radius)

        # array of electrode heights (uniform)
        h_arr = np.ones_like(r_arr) * h

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
        self.tack = tuple(self.tack + [x_center, y_center])

        # Rotate the array
        xy = np.vstack((x_arr.flatten(), y_arr.flatten()))
        xy = np.matmul(rotmat, xy)
        x_arr = xy[0, :]
        y_arr = xy[1, :]

        # Apply offset
        x_arr += x_center
        y_arr += y_center

        # add all electrodes
        self.num_electrodes = 0
        self.electrodes = []
        for r, x, y, h, n in zip(r_arr, x_arr, y_arr, h_arr, names):
            self.add_electrode(Electrode(self.etype, r, x, y, h, n))

    def __str__(self):
        return "AlphaIMS(%s, num_electrodes=%d)" % (self.etype,
                                                    self.num_electrodes)
