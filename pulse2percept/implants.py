# -*implants -*-
"""

Functions for creating retinal implants

"""
import numpy as np
import logging

from pulse2percept import utils


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

        if etype.lower() not in ['epiretinal', 'subretinal']:
            e_s = "Acceptable values for `etype` are: 'epiretinal', "
            e_s += "'subretinal'."
            raise ValueError(e_s)

        self.etype = etype.lower()
        self.radius = radius
        self.x_center = x_center
        self.y_center = y_center
        self.name = name
        self.height = height

    def get_height(self):
        """Returns the electrode-retina distance

        For epiretinal electrodes, this returns the distance to the ganglion
        cell layer.
        For subretinal electrodes, this returns the distance to the bipolar
        layer.
        """
        if self.etype == 'epiretinal':
            return self.h_nfl
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
            th_nfl = 4.0  # nerve fiber layer
            th_gc = 56.0  # ganglion cell bodies + inner nuclear layer
            th_bp = 23.0  # bipolar bodies + inner nuclear layer
        elif fovdist <= 1550:
            # Layer thicknesses given for 600-1550 um distance (from fovea)
            th_nfl = 34.0
            th_gc = 87.0
            th_bp = 37.5
        else:
            # Layer thicknesses given for 1550-3000 um distance (from fovea)
            th_nfl = 45.5
            th_gc = 58.2
            th_bp = 30.75
            if fovdist > 3000:
                e_s = "Distance to fovea=%.0f > 3000 um, " % fovdist
                e_s += "assuming same layer thicknesses as for 1550-3000 um "
                e_s += "distance."
                logging.getLogger(__name__).warning(e_s)

        if self.etype == 'epiretinal':
            # This is simply the electrode-retina distance
            self.h_nfl = height

            # All the way through the ganglion cell layer, inner plexiform
            # layer, and halfway through the inner nuclear layer
            self.h_inl = height + th_nfl + th_gc + 0.5 * th_bp
        elif self.etype == 'subretinal':
            # Starting from the outer plexiform layer, go halfway through the
            # inner nuclear layer
            self.h_inl = height + 0.5 * th_bp

            # Starting from the outer plexiform layer, all the way through the
            # inner nuclear layer, inner plexiform layer, and ganglion cell
            # layer
            self.h_nfl = height + th_bp + th_gc + th_nfl
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

            - 'NFL': nerve fiber layer, ganglion axons
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
        if 'NFL' in layer:  # nerve fiber layer, ganglion axons
            h = np.ones(r.shape) * self.h_nfl
            # actual distance from the electrode edge
            d = ((r - self.radius)**2 + self.h_nfl**2)**.5
        elif 'INL' in layer:  # inner nuclear layer, containing the bipolars
            h = np.ones(r.shape) * self.h_inl
            d = ((r - self.radius)**2 + self.h_inl**2)**.5
        else:
            s = "Layer %s not found. Acceptable values for `layer` are " \
                "'NFL' or 'INL'." % layer
            raise ValueError(s)
        cspread = (alpha / (alpha + h ** n))
        cspread[r > self.radius] = (alpha /
                                    (alpha + d[r > self.radius] ** n))

        return cspread


class ElectrodeArray(object):

    def __init__(self, etype, radii, xs, ys, hs, names=None):
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
        etype : string
            Electrode type, {'epiretinal', 'subretinal'}
        radii : array_like
            List of electrode radii.
        xs : array_like
            List of x-coordinates for the center of the electrodes (microns).
        ys : array_like
            List of y-coordinates for the center of the electrodes (microns).
        hs : array_like
            List of electrode heights (distance from the retinal surface)
        names : array_like, optional
            List of names (string identifiers) for each eletrode.
            Default: None.

        Examples
        --------
        A single epiretinal electrode called 'A1', with radius 100um, sitting
        at retinal location (0, 0), 10um away from the retina:

        >>> from pulse2percept import implants
        >>> implant1 = implants.ElectrodeArray('epiretinal', 100, 0, 0, 10,
        ...                                    'A1')

        An array with two electrodes of size 100um, one sitting at
        (-100, -100), the other sitting at (0, 0), with 0 distance from the
        retina, of type 'subretinal':

        >>> implant2 = implants.ElectrodeArray('subretinal', [100, 100],
        ...                                    [-100, 0], [-100, 0], [0, 0])

        Get access to the electrode with name 'A1' in the array:

        >>> my_electrode = implant1['A1']

        Get access to the second electrode in the array:

        >>> my_electrode = implant2[1]

        """
        # Make it so the constructor can accept either floats, lists, or
        # numpy arrays, and `zip` works regardless.
        radii = np.array([radii], dtype=np.float32).flatten()
        xs = np.array([xs], dtype=np.float32).flatten()
        ys = np.array([ys], dtype=np.float32).flatten()
        hs = np.array([hs], dtype=np.float32).flatten()
        names = np.array([names], dtype=np.str).flatten()
        assert radii.size == xs.size == ys.size == hs.size

        if names.size != radii.size:
            # If not every electrode has a name, replace with None's
            names = np.array([None] * radii.size)

        self.etype = etype
        self.num_electrodes = names.size
        self.electrodes = []
        for r, x, y, h, n in zip(radii, xs, ys, hs, names):
            self.electrodes.append(Electrode(etype, r, x, y, h, n))

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
        except:
            # If `item` is a valid string identifier, return valid index.
            # Else return None
            try:
                return self.electrodes[self.get_index(item)]
            except:
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


class ArgusI(ElectrodeArray):

    def __init__(self, x_center=0, y_center=0, h=0, rot=0 * np.pi / 180,
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
        x_center : float
            x coordinate of the array center (um)
        y_center : float
            y coordinate of the array center (um)
        h : float || array_like
            Distance of the array to the retinal surface (um). Either a list
            with 16 entries or a scalar.
        rot : float
            Rotation angle of the array (rad). Positive values denote
            counter-clock-wise rotations.

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

        if use_legacy_names:
            # Legacy Argus I names
            names = ['L6', 'L2', 'M8', 'M4',
                     'L5', 'L1', 'M7', 'M3',
                     'L8', 'L4', 'M6', 'M2',
                     'L7', 'L3', 'M5', 'M1']
        else:
            # Standard Argus I names: A1, B1, C1, D1, A1, B2, ..., D4
            # Shortcut: Use `chr` to go from int to char
            names = [chr(i) + str(j) for j in range(1, 5)
                     for i in range(65, 69)]

        if isinstance(h, list):
            h_arr = np.array(h).flatten()
            if h_arr.size != len(r_arr):
                e_s = "If `h` is a list, it must have 16 entries."
                raise ValueError(e_s)
        else:
            # All electrodes have the same height
            h_arr = np.ones_like(r_arr) * h

        # Equally spaced electrodes
        e_spacing = 800  # um
        x_arr = np.arange(0, 4) * e_spacing - 1.5 * e_spacing
        x_arr, y_arr = np.meshgrid(x_arr, x_arr, sparse=False)

        # Rotation matrix
        R = np.array([np.cos(rot), np.sin(rot),
                      -np.sin(rot), np.cos(rot)]).reshape((2, 2))

        # Rotate the array
        xy = np.vstack((x_arr.flatten(), y_arr.flatten()))
        xy = np.matmul(R, xy)
        x_arr = xy[0, :]
        y_arr = xy[1, :]

        # Apply offset
        x_arr += x_center
        y_arr += y_center

        self.etype = 'epiretinal'
        self.num_electrodes = len(names)
        self.electrodes = []
        for r, x, y, h, n in zip(r_arr, x_arr, y_arr, h_arr, names):
            self.electrodes.append(Electrode(self.etype, r, x, y, h, n))


class ArgusII(ElectrodeArray):

    def __init__(self, x_center=0, y_center=0, h=0, rot=0 * np.pi / 180):
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
            counter-clock-wise rotations.

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

        # Equally spaced electrodes
        e_spacing = 525  # um
        x_arr = np.arange(10) * e_spacing - 4.5 * e_spacing
        y_arr = np.arange(6) * e_spacing - 2.5 * e_spacing
        x_arr, y_arr = np.meshgrid(x_arr, y_arr, sparse=False)

        # Rotation matrix
        R = np.array([np.cos(rot), np.sin(rot),
                      -np.sin(rot), np.cos(rot)]).reshape((2, 2))

        # Rotate the array
        xy = np.vstack((x_arr.flatten(), y_arr.flatten()))
        xy = np.matmul(R, xy)
        x_arr = xy[0, :]
        y_arr = xy[1, :]

        # Apply offset
        x_arr += x_center
        y_arr += y_center

        self.etype = 'epiretinal'
        self.num_electrodes = len(names)
        self.electrodes = []
        for r, x, y, h, n in zip(r_arr, x_arr, y_arr, h_arr, names):
            self.electrodes.append(Electrode(self.etype, r, x, y, h, n))
