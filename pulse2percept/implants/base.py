"""implants"""
import numpy as np
import abc
import collections as coll

from ..stimuli import Stimulus
from ..utils import PrettyPrint


class Electrode(PrettyPrint):
    """Electrode

    Abstract base class for all electrodes.
    """

    def __init__(self, x, y, z):
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
    """Circular disk electrode

    Parameters
    ----------
    x, y, z : double
        3D location that is the center of the disk electrode
    r : double
        Disk radius in the x,y plane
    """

    def __init__(self, x, y, z, r):
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
    """Visual prosthesis system

    A visual prosthesis combines an electrode array and (optionally) a
    stimulus. This is the base class for prosthesis systems such as `ArgusII`
    and `AlphaIMS`.

    Parameters
    ----------
    earray : `ElectrodeArray` or `Electrode`
        The electrode array used to deliver electrical stimuli to the retina.
    stim : `Stimulus` source type
        A valid source type for the `Stimulus` object (e.g., scalar, NumPy
        array, pulse train).
    eye : 'LE' or 'RE'
        A string indicating whether the system is implanted in the left ('LE')
        or right eye ('RE')

    Examples
    --------
    A system in the left eye made from a single DiskElectrode with radius
    r=100um sitting at x=200um, y=-50um, z=10um:

    >>> from pulse2percept.implants import DiskElectrode, ProsthesisSystem
    >>> implant = ProsthesisSystem(DiskElectrode(200, -50, 10, 100), eye='LE')

    .. note::
       A stimulus can also be assigned later (see `stim`).
    """

    def __init__(self, earray, stim=None, eye='RE'):
        self.earray = earray
        self.stim = stim
        self.eye = eye

    def get_params(self):
        return {'earray': self.earray, 'stim': self.stim, 'eye': self.eye}

    def check_stim(self, stim):
        """Quality-check the stimulus

        This method is executed every time a new value is assigned to `stim`.

        No checks are performed by default, but the user can define their own
        checks in implants that inherit from `ProsthesisSystem`.

        Parameters
        ----------
        stim : `Stimulus` source type
            A valid source type for the `Stimulus` object (e.g., scalar, NumPy
            array, pulse train).
        """
        pass

    @property
    def earray(self):
        """Electrode array

        """
        return self._earray

    @earray.setter
    def earray(self, earray):
        """Electrode array setter (called upon `self.earray = earray`)"""
        # Assign the electrode array:
        if isinstance(earray, Electrode):
            # For convenience, build an array froma single electrode:
            earray = ElectrodeArray(earray)
        if not isinstance(earray, ElectrodeArray):
            raise TypeError("'earray' must be an ElectrodeArray object, not "
                            "%s." % type(earray))
        self._earray = earray

    @property
    def stim(self):
        """Stimulus

        A stimulus can be created from many source types, such as scalars,
        NumPy arrays, and dictionaries (see `Stimulus` for a complete list).

        A stimulus can be assigned either in the `ProsthesisSystem` constructor
        or later by assigning a value to `stim`.

        .. note::
           Unless when using dictionary notation, the number of stimuli must
           equal the number of electrodes in `earray`.

        Examples
        --------
        Send a biphasic pulse to an implant made from a single `DiskElectrode`:

        >>> from pulse2percept.implants import DiskElectrode, ProsthesisSystem
        >>> from pulse2percept.stimuli import BiphasicPulse
        >>> implant = ProsthesisSystem(DiskElectrode(0, 0, 0, 100))
        >>> implant.stim = BiphasicPulse('cathodicfirst', 1e-4, 1e-6)

        Stimulate Electrode B7 in Argus II with 13 uA:

        >>> from pulse2percept.implants import ArgusII
        >>> implant = ArgusII(stim={'B7': 13})

        """
        return self._stim

    @stim.setter
    def stim(self, data):
        """Stimulus setter (called upon `self.stim = data`)"""
        if data is None:
            self._stim = None
        else:
            if isinstance(data, dict):
                # Electrode names already provided by keys:
                stim = Stimulus(data)
            else:
                # Use electrode names as stimulus coordinates:
                stim = Stimulus(data, electrodes=list(self.earray.keys()))
            # Perform safety checks, etc.:
            self.check_stim(stim)
            # Store safe stimulus:
            self._stim = stim

    @property
    def eye(self):
        """Implanted eye

        A `ProsthesisSystem` can be implanted either in a left eye ('LE') or
        right eye ('RE'). Models such as `AxonMapModel` will treat left and
        right eyes differently (for example, adjusting the location of the
        optic disc).

        Examples
        --------
        Implant Argus II in a left eye:

        >>> from pulse2percept.implants import ArgusII
        >>> implant = ArgusII(eye='LE')
        """
        return self._eye

    @eye.setter
    def eye(self, eye):
        """Eye setter (called upon `self.eye = eye`)"""
        if eye not in ['LE', 'RE']:
            raise ValueError("'eye' must be either 'LE' or 'RE', not "
                             "%s." % eye)
        self._eye = eye

    @property
    def n_electrodes(self):
        """Number of electrodes in the array

        This is equivalent to calling `earray.n_electrodes`.
        """
        return self.earray.n_electrodes

    def __getitem__(self, item):
        return self.earray[item]

    def __iter__(self):
        return iter(self.earray)

    def keys(self):
        """Return all electrode names in the electrode array"""
        return self.earray.keys()

    def values(self):
        """Return all electrode objects in the electrode array"""
        return self.earray.values()

    def items(self):
        """Return all electrode names and objects in the electrode array

        Internally, electrodes are stored in a dictionary in
        `earray.electrodes`. For convenience, electrodes can also be accessed
        via `items`.

        Examples
        --------
        Save the x-coordinates of all electrodes of Argus I in a dictionary:

        >>> from pulse2percept.implants import ArgusI
        >>> xcoords = {}
        >>> for name, electrode in ArgusI().items():
        ...     xcoords[name] = electrode.x

        """
        return self.earray.items()
