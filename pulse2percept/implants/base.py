"""implants"""
import numpy as np
import abc
import collections as coll
import xarray as xr

from ..utils import PrettyPrint


class Electrode(PrettyPrint):

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


class ElectrodeArray(PrettyPrint):

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
            except IndexError:
                raise StopIteration
            return None

    def __iter__(self):
        return iter(self.electrodes)

    def keys(self):
        return self.electrodes.keys()

    def values(self):
        return self.electrodes.values()

    def items(self):
        return self.electrodes.items()


class ProsthesisSystem(PrettyPrint):

    def __init__(self, earray, stim=None, eye='RE'):
        """ProsthesisSystem

        A visual prosthesis that combines an electrode array and a stimulus.
        """
        if not isinstance(earray, ElectrodeArray):
            raise TypeError("'earray' must be an ElectrodeArray object, not "
                            "%s." % type(earray))
        self.earray = earray
        self.stim = stim
        if eye not in ['LE', 'RE']:
            raise ValueError("'eye' must be either 'LE' or 'RE', not "
                             "%s." % eye)
        self.eye = eye

    def get_params(self):
        return {'earray': self.earray, 'stim': self.stim, 'eye': self.eye}

    @property
    def stim(self):
        """Stimulus"""
        return self._stim

    @stim.setter
    def stim(self, data):
        if data is None:
            self._stim = None
        elif isinstance(data, np.ndarray):
            if data.size != self.earray.n_electrodes:
                raise ValueError(("NumPy array must have the same number of "
                                  "elements as the implant has electrodes "
                                  "(%d), not %d.") % (self.earray.n_electrodes,
                                                      data.size))
            # Convert to double
            data = np.asarray(data, dtype=np.double)
            # Find all nonzero entries:
            idx_nz = np.flatnonzero(data)
            # Make sure these are valid indexes into the electrode array:
            electrodes = self.earray[idx_nz]
            # Set these as coordinates:
            coords = [('electrodes', electrodes)]
            # Retain only nonzero data:
            self._stim = xr.DataArray(data.ravel()[idx_nz], coords=coords,
                                      name='current (uA)')
        else:
            raise NotImplementedError

    @property
    def n_electrodes(self):
        """Number of electrodes in the array"""
        return self.earray.n_electrodes

    def __getitem__(self, item):
        return self.earray[item]

    def keys(self):
        return self.earray.keys()

    def values(self):
        return self.earray.values()

    def items(self):
        return self.earray.items()


# let the implants deal with deciding parameter values, R/L eye / tack,
# instantiating electrode array
# inherit electrode array
class ElectrodeGrid(ElectrodeArray):

    def __init__(self, cols, rows, x, y, z, rot, r, spacing):
        self.add_electrodes(electrodes)
        self.cols = cols
        self.rows = rows
        self.x = x
        self.y = y
        self.z = z
        self.rot = rot
        self.r = r
        self.spacing = spacing
        self.set_grid()

    def get_params(self):
        """Return a dictionary of class attributes"""
        return {'cols': self.cols, 'rows': self.rows,
         'x': self.x, 'y': self.y 'z': self.z, 'rot': self.rot,
          'r': self.r, 'spacing': self.spacing}

    def get_x_arr():
        x_arr = np.arange(cols) * spacing - (cols / 2 - 0.5) * spacing
        return x_arr

    def set_grid(self):
        if isinstance(z, (list, np.ndarray)):
            z_arr = np.asarray(z).flatten()
            if z_arr.size != len(r_arr):
                e_s = "If `h` is a list, it must have %d entries." % n_elecs
                raise ValueError(e_s)
         else:
            # All electrodes have the same height
            z_arr = np.ones_like(r_arr) * z   

        n_elecs = cols*rows
        names = np.arange(n_elecs)

        # array containing electrode radii (uniform)
        r_arr = np.full(shape=n_elecs, fill_value=r)

        x_arr = np.arange(cols) * spacing - (cols / 2 - 0.5) * spacing

        y_arr = np.arange(rows) * spacing - (rows / 2 - 0.5) * spacing

        x_arr, y_arr = np.meshgrid(x_arr, y_arr, sparse=False)

        rotmat = np.array([np.cos(rot), -np.sin(rot),
                           np.sin(rot), np.cos(rot)]).reshape((2, 2))

        # Rotate the array
        xy = np.vstack((x_arr.flatten(), y_arr.flatten()))
        xy = np.matmul(rotmat, xy)
        x_arr = xy[0, :]
        y_arr = xy[1, :]

        # Apply offset
        x_arr += x
        y_arr += y

        # maybe parameterize thee type of electrode later
        for x, y, z, r, name in zip(x_arr, y_arr, z_arr, r_arr, names):
        self.earray.add_electrode(name, DiskElectrode(x, y, z, r))
