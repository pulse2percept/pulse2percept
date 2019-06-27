"""implants"""
import numpy as np
import abc
import collections as coll

from pulse2percept import utils


class Electrode(utils.PrettyPrint):

    def __init__(self, x, y, z):
        """Electrode

        Abstract base class for all electrodes.
        """
        if isinstance(x, (coll.Sequence, np.ndarray)):
            raise TypeError("x must be a scalar.")
        if isinstance(y, (coll.Sequence, np.ndarray)):
            raise TypeError("y must be a scalar.")
        if isinstance(z, (coll.Sequence, np.ndarray)):
            raise TypeError("z must be a scalar.")
        self.x = x
        self.y = y
        self.z = z

    def get_params(self):
        """Return a dictionary of class attributes"""
        return {'x': self.x, 'y': self.y, 'z': self.z}

    @abc.abstractmethod
    def electric_potential(self, x, y, z):
        raise NotImplementedError


class PointSource(Electrode):
    """Point source"""

    def electric_potential(self, x, y, z, amp, sigma):
        r = np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2 + (z - self.z) ** 2)
        if np.isclose(r, 0):
            return sigma * amp
        return sigma * amp / (4.0 * np.pi * r)


class DiskElectrode(Electrode):
    """Circular disk electrode"""

    def __init__(self, x, y, z, r):
        """Circular disk electrode

        Parameters
        ----------
        x, y, z : double
            3D location that is the center of the disk electrode
        r : double
            Disk radius in the x,y plane
        """
        super(DiskElectrode, self).__init__(x, y, z)
        if isinstance(r, (coll.Sequence, np.ndarray)):
            raise TypeError("Electrode radius must be a scalar.")
        if r <= 0:
            raise ValueError("Electrode radius must be > 0, not %f." % r)
        self.r = r

    def get_params(self):
        """Return a dictionary of class attributes"""
        params = super().get_params()
        params.update({'r': self.r})
        return params

    def electric_potential(self, x, y, z, v0):
        """Calculate electric potential at (x, y, z)

        Parameters
        ----------
        x, y, z : double
            3D location at which to evaluate the electric potential
        v0 : double
            The quasi-static disk potential relative to a ground electrode at
            infinity
        """
        radial_dist = np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)
        axial_dist = z - self.z
        if np.isclose(axial_dist, 0):
            # Potential on the electrode surface:
            if radial_dist > self.r:
                # Outside the electrode:
                return 2.0 * v0 / np.pi * np.arcsin(self.r / radial_dist)
            else:
                # On the electrode:
                return v0
        else:
            # Off the electrode surface:
            numer = 2 * self.r
            denom = np.sqrt((radial_dist - self.r) ** 2 + axial_dist ** 2)
            denom += np.sqrt((radial_dist + self.r) ** 2 + axial_dist ** 2)
            return 2.0 * v0 / np.pi * np.arcsin(numer / denom)


class ElectrodeArray(utils.PrettyPrint):

    def __init__(self, electrodes):
        self.electrodes = coll.OrderedDict()
        self.add_electrodes(electrodes)

    @property
    def n_electrodes(self):
        return len(self.electrodes)

    def get_params(self):
        """Return a dictionary of class attributes"""
        return {'electrodes': self.electrodes,
                'n_electrodes': self.n_electrodes}

    def add_electrode(self, name, electrode):
        """Add an electrode to the array

        Parameters
        ----------
        name : int|str|...
            Electrode name or index
        electrode : implants.Electrode
            An Electrode object, such as a PointSource or a DiskElectrode.
        """
        if not isinstance(electrode, Electrode):
            raise TypeError(("Electrode %s must be an Electrode object, not "
                             "%s.") % (name, type(electrode)))
        if name in self.electrodes.keys():
            raise ValueError(("Cannot add electrode: key '%s' already "
                              "exists.") % name)
        self.electrodes.update({name: electrode})

    def add_electrodes(self, electrodes):
        """
        Note that if you pass a dictionary, keys will automatically be sorted.
        """
        if isinstance(electrodes, dict):
            for name, electrode in electrodes.items():
                self.add_electrode(name, electrode)
        elif isinstance(electrodes, list):
            for electrode in electrodes:
                self.add_electrode(self.n_electrodes, electrode)
        elif isinstance(electrodes, Electrode):
            self.add_electrode(self.n_electrodes, electrodes)
        else:
            raise TypeError(("electrodes must be a list or dict, not "
                             "%s") % type(electrodes))

    def __getitem__(self, item):
        """Return an electrode from the array

        An electrode in the array can be accessed either by its name (the
        key value in the dictionary) or by index (in the list).

        Parameters
        ----------
        item : int|string
            If `item` is an integer, returns the `item`-th electrode in the
            array. If `item` is a string, returns the electrode with string
            identifier `item`.
        """
        if isinstance(item, (list, np.ndarray)):
            # Recursive call for list items:
            return [self.__getitem__(i) for i in item]
        if isinstance(item, str):
            # A string is probably a dict key:
            try:
                return self.electrodes[item]
            except KeyError:
                return None
        try:
            # Else, try indexing in various ways:
            return self.electrodes[item]
        except (KeyError, TypeError):
            # If not a dict key, `item` might be an int index into the list:
            try:
                key = list(self.electrodes.keys())[item]
                return self.electrodes[key]
            except (IndexError, TypeError):
                return None

    def __iter__(self):
        return iter(self.electrodes)

    def keys(self):
        return self.electrodes.keys()

    def values(self):
        return self.electrodes.values()

    def items(self):
        return self.electrodes.items()


class ProsthesisSystem(utils.PrettyPrint):

    def get_params(self):
        return {'array': self.array, 'stim': self.stim, 'eye': self.eye}

    @property
    def n_electrodes(self):
        return self.array.n_electrodes

    def __getitem__(self, item):
        return self.array[item]

    def keys(self):
        return self.array.keys()

    def values(self):
        return self.array.values()

    def items(self):
        return self.array.items()


class ImplantStimulus(object):

    def __init__(self, implant, stim=None):
        if not isinstance(implant, ProsthesisSystem):
            raise TypeError(("'implant' must be of type ProsthesisSystem, "
                             "not %s") % type(implant))
        self.implant = implant
        self.stim = self.parse(stim)

    def parse(self, stim):
        if not isinstance(stim, np.ndarray):
            raise TypeError("stim must be a NumPy array")
        if stim.size != self.implant.n_electrodes:
            raise ValueError(("'stim' must have the same number of elements "
                              "as 'implant' has electrodes (%d), not %d.") %
                             (self.implant.n_electrodes, stim.size))
        return stim

    def update(self, stim):
        self.stim = self.parse(stim)

    def nonzero(self):
        # Find all nonzero entries in the stimulus array:
        idx_nz = np.flatnonzero(self.stim)
        return self.implant[idx_nz], self.stim.ravel()[idx_nz]


# def parse_pulse_trains(stim, implant):
#     """Parse input stimulus and convert to list of pulse trains

#     Parameters
#     ----------
#     stim : utils.TimeSeries|list|dict
#         There are several ways to specify an input stimulus:

#         - For a single-electrode array, pass a single pulse train; i.e., a
#           single utils.TimeSeries object.
#         - For a multi-electrode array, pass a list of pulse trains, where
#           every pulse train is a utils.TimeSeries object; i.e., one pulse
#           train per electrode.
#         - For a multi-electrode array, specify all electrodes that should
#           receive non-zero pulse trains by name in a dictionary. The key
#           of each element is the electrode name, the value is a pulse train.
#           Example: stim = {'E1': pt, 'stim': pt}, where 'E1' and 'stim' are
#           electrode names, and `pt` is a utils.TimeSeries object.
#     implant : p2p.implants.ElectrodeArray
#         A p2p.implants.ElectrodeArray object that describes the implant.

#     Returns
#     -------
#     A list of pulse trains; one pulse train per electrode.
#     """
#     # Parse input stimulus
#     if isinstance(stim, utils.TimeSeries):
#         # `stim` is a single object: This is only allowed if the implant
#         # has only one electrode
#         if implant.n_electrodes > 1:
#             e_s = "More than 1 electrode given, use a list of pulse trains"
#             raise ValueError(e_s)
#         pt = [copy.deepcopy(stim)]
#     elif isinstance(stim, dict):
#         # `stim` is a dictionary: Look up electrode names and assign pulse
#         # trains, fill the rest with zeros

#         # Get right size from first dict element, then generate all zeros
#         idx0 = list(stim.keys())[0]
#         pt_zero = utils.TimeSeries(stim[idx0].tsample,
#                                    np.zeros_like(stim[idx0].data))
#         pt = [pt_zero] * implant.n_electrodes

#         # Iterate over dictionary and assign non-zero pulse trains to
#         # corresponding electrodes
#         for key, value in stim.items():
#             el_idx = implant.get_index(key)
#             if el_idx is not None:
#                 pt[el_idx] = copy.deepcopy(value)
#             else:
#                 e_s = "Could not find electrode with name '%s'" % key
#                 raise ValueError(e_s)
#     else:
#         # Else, `stim` must be a list of pulse trains, one for each electrode
#         if len(stim) != implant.n_electrodes:
#             e_s = "Number of pulse trains must match number of electrodes"
#             raise ValueError(e_s)
#         pt = copy.deepcopy(stim)

#     return pt
