"""`Electrode`, `ElectrodeArray`, `ElectrodeGrid`, `ProsthesisSystem`"""
import numpy as np
from abc import ABCMeta, abstractmethod
import collections as coll

from ..stimuli import Stimulus
from ..utils import PrettyPrint


class Electrode(PrettyPrint, metaclass=ABCMeta):
    """Electrode

    Abstract base class for all electrodes.
    """

    def __init__(self, x, y, z):
        if isinstance(x, (coll.Sequence, np.ndarray)):
            raise TypeError("x must be a scalar, not %s." % (type(x)))
        if isinstance(y, (coll.Sequence, np.ndarray)):
            raise TypeError("y must be a scalar, not %s." % type(y))
        if isinstance(z, (coll.Sequence, np.ndarray)):
            raise TypeError("z must be a scalar, not %s." % type(z))
        self.x = x
        self.y = y
        self.z = z

    def get_params(self):
        """Return a dictionary of class attributes"""
        return {'x': self.x, 'y': self.y, 'z': self.z}

    @abstractmethod
    def electric_potential(self, x, y, z):
        raise NotImplementedError


class PointSource(Electrode):
    """Point source"""

    def electric_potential(self, x, y, z, amp, sigma):
        """Calculate electric potential at (x, y, z)

        Parameters
        ----------
        x/y/z : double
            3D location at which to evaluate the electric potential
        amp : double
            amplitude of the constant current pulse
        sigma : double
            resistivity of the extracellular solution

        Returns
        -------
        pot : double
            The electric potential at (x, y, z)

        The electric potential :math:`V(r)` of a point source is given by:

        .. math::

            V(r) = \\frac{\\sigma I}{4 \\pi r},

        where :math:`\\sigma` is the resistivity of the extracellular solution
        (typically Ames medium, :math:`\\sigma = 110 \\Ohm cm`),
        :math:`I` is the amplitude of the constant current pulse,
        and :math:`r` is the distance from the stimulating electrode to the
        point at which the voltage is being computed.

        """
        r = np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2 + (z - self.z) ** 2)
        if np.isclose(r, 0):
            return sigma * amp
        return sigma * amp / (4.0 * np.pi * r)


class DiskElectrode(Electrode):
    """Circular disk electrode

    Parameters
    ----------
    x/y/z : double
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
        x/y/z : double
            3D location at which to evaluate the electric potential
        v0 : double
            The quasi-static disk potential relative to a ground electrode at
            infinity

        Returns
        -------
        pot : double
            The electric potential at (x, y, z).


        The electric potential :math:`V(r,z)` of a disk electrode is given by
        [WileyWebster1982]_:

        .. math::

            V(r,z) = \\sin^{-1} \\bigg\\{ \\frac{2a}{\\sqrt{(r-a)^2 + z^2} + \\sqrt{(r+a)^2 + z^2}} \\bigg\\} \\times \\frac{2 V_0}{\\pi},

        for :math:`z \\neq 0`, where :math:`r` and :math:`z` are the radial
        and axial distances from the center of the disk, :math:`V_0` is the
        disk potential, :math:`\\sigma` is the medium conductivity,
        and :math:`a` is the disk radius.

        """
        radial_dist = np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)
        axial_dist = z - self.z
        if np.isclose(axial_dist, 0):
            # Potential on the electrode surface (Eq. 9 in Wiley & Webster):
            if radial_dist > self.r:
                # Outside the electrode:
                return 2.0 * v0 / np.pi * np.arcsin(self.r / radial_dist)
            else:
                # On the electrode:
                return v0
        else:
            # Off the electrode surface (Eq. 10):
            numer = 2 * self.r
            denom = np.sqrt((radial_dist - self.r) ** 2 + axial_dist ** 2)
            denom += np.sqrt((radial_dist + self.r) ** 2 + axial_dist ** 2)
            return 2.0 * v0 / np.pi * np.arcsin(numer / denom)


class ElectrodeArray(PrettyPrint):
    """Electrode array

    A collection of :py:class:`~pulse2percept.implants.Electrode` objects.

    Parameters
    ----------
    electrodes : array-like
        Either a single :py:class:`~pulse2percept.implants.Electrode` object
        or a dict, list, or NumPy array thereof. The keys of the dict will
        serve as electrode names. Otherwise electrodes will be indexed 0..N.

    Examples
    --------
    Electrode array made from a single DiskElectrode:

    >>> from pulse2percept.implants import ElectrodeArray, DiskElectrode
    >>> earray = ElectrodeArray(DiskElectrode(0, 0, 0, 100))
    >>> earray.electrodes  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    OrderedDict([(0, DiskElectrode(r=100..., x=0..., y=0..., z=0...))])

    Electrode array made from a single DiskElectrode with name 'A1':

    >>> from pulse2percept.implants import ElectrodeArray, DiskElectrode
    >>> earray = ElectrodeArray({'A1': DiskElectrode(0, 0, 0, 100)})
    >>> earray.electrodes  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    OrderedDict([('A1', DiskElectrode(r=100..., x=0..., y=0..., z=0...))])

    """

    def __init__(self, electrodes):
        self.electrodes = coll.OrderedDict()
        self.add_electrodes(electrodes)

    @property
    def electrodes(self):
        return self._electrodes

    @electrodes.setter
    def electrodes(self, electrodes):
        self._electrodes = electrodes

    @property
    def n_electrodes(self):
        return len(self.electrodes)

    def get_params(self):
        """Return a dictionary of class attributes"""
        return {'electrodes': self.electrodes,
                'n_electrodes': self.n_electrodes}

    def add_electrode(self, name, electrode):
        """Add an electrode to the dictionary

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
        self._electrodes.update({name: electrode})

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


class ElectrodeGrid(ElectrodeArray):
    """Rectangular grid of electrodes

    Parameters
    ----------
    shape : (rows, cols)
        A tuple containing the number of rows x columns in the grid
    x, y, z : double
        3D coordinates of the center of the grid
    rot : double
        Rotation of the grid in radians (positive angle: counter-clockwise
        rotation on the retinal surface)
    r : double
        Electrode radius in microns
    spacing : double
        Electrode-to-electrode spacing in microns. If None, 2x radius is
        chosen.
    names: (name_rows, name_cols), each of which either 'A' or '1'
        Naming convention for rows and columns, respectively.
        If 'A', rows or columns will be labeled alphabetically.
        If '1', rows or columns will be labeled numerically.
        Columns and rows may only be strings and integers.
        For example ('1', 'A') will number rows numerically and columns
        alphabetically.

    .. note::

        For now, all electrodes will be
        :py:class:`~pulse2percept.implants.DiskElectrode` objects.

    Examples
    --------
    An electrode grid with 2 rows and 4 columns, made of electrodes with 10um
    radius spaced 20um apart, centered at (10, 20)um, and located 500um away
    from the retinal surface, with names like this:

        A1 A2 A3 A4
        B1 B2 B3 B4

    >>> from pulse2percept.implants import ElectrodeGrid
    >>> ElectrodeGrid((2, 4), x=10, y=20, z=500, names=('A', '1')) # doctest: +NORMALIZE_WHITESPACE
    ElectrodeGrid(name_cols='1', name_rows='A', r=10, rot=0, shape=(2, 4),
                  spacing=20.0, x=10, y=20, z=500)

    There are three ways to access (e.g.) the last electrode in the grid,
    either by name (``grid['C3']``), by row/column index (``grid[2, 2]``), or
    by index into the flattened array (``grid[8]``):

    >>> from pulse2percept.implants import ElectrodeGrid
    >>> grid = ElectrodeGrid((3, 3), names=('A', '1'))
    >>> grid['C3']  # doctest: +ELLIPSIS
    DiskElectrode(r=10..., x=20.0, y=20.0, z=0...)
    >>> grid['C3'] == grid[8] == grid[2, 2]
    True

    You can also access multiple electrodes at the same time by passing a
    list of indices/names (it's ok to mix-and-match):

    >>> from pulse2percept.implants import ElectrodeGrid
    >>> grid = ElectrodeGrid((3, 3), names=('A', '1'))
    >>> grid[['A1', 1, (0, 2)]]  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [DiskElectrode(r=10..., x=-20.0, y=-20.0, z=0...),
     DiskElectrode(r=10..., x=0.0, y=-20.0, z=0...),
     DiskElectrode(r=10..., x=20.0, y=-20.0, z=0...)]

    Todo
    ----
    * Allow user-specified electrode types (see :issue:`122`).

    """
    
    def __init__(self, shape, x=0, y=0, z=0, rot=0, spacing=None,
                 names=('A', '1'), electrode_type = 'DiskElectrode', electrode_kwargs=None): # the default electrode type is DiskElectrode
        if not isinstance(names, (tuple, list, np.ndarray)):
            raise TypeError("'names' must be a tuple/list of (rows, cols)")
        if not isinstance(shape, (tuple, list, np.ndarray)):
            raise TypeError("'shape' must be a tuple/list of (rows, cols)")
        if len(shape) != 2:
            raise ValueError("'shape' must have two elements: (rows, cols)")
        if np.prod(shape) <= 0:
            raise ValueError("Grid must have all non-zero rows and columns.")
        if electrode_type != 'DiskElectrode' and electrode_type != "PointSource": # check the value of the electrode_type
            raise ValueError("'electrode_type' must be either 'DiskElectrode' or 'PointSource' ")
        if electrode_type == 'DiskElectrode':
            if electrode_kwargs is None:
                raise NotImplementedError
            elif not isinstance(electrode_kwargs, dict):
                raise TypeError("'electrode_kwargs' must be a dict")
            elif 'r' not in electrode_kwargs:
                raise NotImplementedError
            elif 'r' in electrode_kwargs and electrode_kwargs['r'] <= 0:
                raise ValueError("The radius of the DiskElectrode 'r' must be positive")


        # Extract rows and columns from shape:
        self.shape = shape
        self.x = x
        self.y = y
        self.z = z
        if(electrode_kwargs is not None):
            self.r = electrode_kwargs['r']
        self.rot = rot
        self.spacing = spacing
        self.electrode_type = electrode_type # add electrode_type variable under the class
        if len(names) == 2:
            self.name_rows, self.name_cols = names
        self.names = names
        # Instantiate empty collection of electrodes. This dictionary will be
        # populated in a private method ``_set_egrid``:
        self.electrodes = coll.OrderedDict()
        self._set_grid()
    

    def get_params(self):
        """Return a dictionary of class attributes"""
        params = {'shape': self.shape,
                'x': self.x, 'y': self.y, 'z': self.z,
                'rot': self.rot, 'spacing': self.spacing,
                'name_cols': self.name_cols, 'name_rows': self.name_rows}
        if(self.electrode_type == 'DiskElectrode'):
            params.update({'r':self.r})
        return params

    def __getitem__(self, item):
        """Access electrode(s) in the grid

        Parameters
        ----------
        item : index, string, tuple, or list thereof
            An electrode in the grid can be accessed in three ways:

            *  by name, e.g. grid['A1']
            *  by index into the flattened array, e.g. grid[0]
            *  by (row, column) index into the 2D grid, e.g. grid[0, 0]

            You can also pass a list or NumPy array of the above.

        Returns
        -------
        electrode : `~pulse2percept.implants.Electrode`, list thereof, or None
            Returns the corresponding `~pulse2percept.implants.Electrode`
            object or ``None`` if index is not valid.
        """
        if isinstance(item, (list, np.ndarray)):
            # Recursive call for list items:
            return [self.__getitem__(i) for i in item]
        try:
            # Access by key into OrderedDict, e.g. grid['A1']:
            return self.electrodes[item]
        except (KeyError, TypeError):
            # Access by index into flattened array, e.g. grid[0]:
            try:
                return list(self.electrodes.values())[item]
            except (KeyError, TypeError):
                # Access by [r, c] into 2D grid, e.g. grid[0, 3]:
                try:
                    idx = np.ravel_multi_index(item, self.shape)
                    return list(self.electrodes.values())[idx]
                except (KeyError, ValueError):
                    # Index not found:
                    return None

    def _set_grid(self):
        """Private method to build the electrode grid"""
        n_elecs = np.prod(self.shape)
        rows, cols = self.shape

        # The user did not specify a unique naming scheme:
        if len(self.names) == 2:
            # Create electrode names, using either A-Z or 1-n:
            if self.name_rows.isalpha():
                rws = [chr(i) for i in range(
                    ord(self.name_rows), ord(self.name_rows) + rows + 1)]
            elif self.name_rows.isdigit():
                rws = [str(i) for i in range(
                    int(self.name_rows), rows + int(self.name_rows))]
            else:
                raise ValueError("rows must be alphabetic or numeric")

            if self.name_cols.isalpha():
                clms = [chr(i) for i in range(ord(self.name_cols),
                                              ord(self.name_cols) + cols)]
            elif self.name_cols.isdigit():
                clms = [str(i) for i in range(
                    int(self.name_cols), cols + int(self.name_cols))]
            else:
                raise ValueError("Columns must be alphabetic or numeric.")

            # facilitating Argus I naming scheme
            if self.name_cols.isalpha() and not self.name_rows.isalpha():
                names = [clms[j] + rws[i] for i in range(len(rws))
                         for j in range(len(clms))]
            else:
                names = [rws[i] + clms[j] for i in range(len(rws))
                         for j in range(len(clms))]
        else:
            if len(self.names) != n_elecs:
                raise ValueError("If `names` specifies more than row/column "
                                 "names, it must have %d entries, not "
                                 "%d)." % (n_elecs, len(self.names)))
            names = self.names

        if self.electrode_type == 'DiskElectrode':
            if isinstance(self.r, (list, np.ndarray)):
                # Specify different radius for every electrode in a list:
                if len(self.r) != n_elecs:
                    raise ValueError("If `r` is a list, it must have %d entries, "
                                    "not %d)." % (n_elecs, len(self.r)))
                r_arr = self.r
            else:
                # If `r` is a scalar, choose same radius for all electrodes:
                r_arr = np.ones(n_elecs, dtype=float) * self.r

        if isinstance(self.z, (list, np.ndarray)):
            # Specify different height for every electrode in a list:
            z_arr = np.asarray(self.z).flatten()
            if z_arr.size != n_elecs:
                raise ValueError("If `h` is a list, it must have %d entries, "
                                 "not %d." % (n_elecs, len(self.z)))
        else:
            # If `z` is a scalar, choose same height for all electrodes:
            z_arr = np.ones(n_elecs, dtype=float) * self.z

        # If spacing is None:
        if self.spacing is None:
            # If the electrode_type is "DiskElectrode", choose 2 * radius as the spacing
            if self.electrode_type == "DiskElectrode":
                self.spacing = 2.0 * self.r
            # If the electrode_type is "PointSource", choose 20.0 as the spacing
            elif self.electrode_type == "PointSource":
                self.spacing = 20.0

        # Make a 2D meshgrid from x, y coordinates:
        # For example, cols=3 with spacing=100 should give: [-100, 0, 100]
        x_arr = (np.arange(cols) * self.spacing -
                 (cols / 2.0 - 0.5) * self.spacing)
        y_arr = (np.arange(rows) * self.spacing -
                 (rows / 2.0 - 0.5) * self.spacing)
        x_arr, y_arr = np.meshgrid(x_arr, y_arr, sparse=False)

        # Rotate the grid:
        rotmat = np.array([np.cos(self.rot), -np.sin(self.rot),
                           np.sin(self.rot), np.cos(self.rot)]).reshape((2, 2))
        xy = np.matmul(rotmat, np.vstack((x_arr.flatten(), y_arr.flatten())))
        x_arr = xy[0, :]
        y_arr = xy[1, :]

        # Apply offset to make the grid centered at (self.x, self.y):
        x_arr += self.x
        y_arr += self.y

        
        if self.electrode_type == 'DiskElectrode':
            for x, y, z, r, name in zip(x_arr, y_arr, z_arr, r_arr, names):
                self.add_electrode(name, DiskElectrode(x, y, z, r))
        # when the electrode type is PointSource
        elif self.electrode_type == 'PointSource':
            for x, y, z, name in zip(x_arr, y_arr, z_arr, names):
                self.add_electrode(name, PointSource(x, y, z))    


class ProsthesisSystem(PrettyPrint):
    """Visual prosthesis system

    A visual prosthesis combines an electrode array and (optionally) a
    stimulus. This is the base class for prosthesis systems such as
    :py:class:`~pulse2percept.implants.ArgusII` and
    :py:class:`~pulse2percept.implants.AlphaIMS`.

    Parameters
    ----------
    earray : :py:class:`~pulse2percept.implants.ElectrodeArray` or
             :py:class:`~pulse2percept.implants.Electrode`
        The electrode array used to deliver electrical stimuli to the retina.
    stim : :py:class:`~pulse2percept.stimuli.Stimulus` source type
        A valid source type for the :py:class:`~pulse2percept.stimuli.Stimulus`
        object (e.g., scalar, NumPy array, pulse train).
    eye : 'LE' or 'RE'
        A string indicating whether the system is implanted in the left ('LE')
        or right eye ('RE')

    Examples
    --------
    A system in the left eye made from a single
    :py:class:`~pulse2percept.implants.DiskElectrode` with radius
    r=100um sitting at x=200um, y=-50um, z=10um:

    >>> from pulse2percept.implants import DiskElectrode, ProsthesisSystem
    >>> implant = ProsthesisSystem(DiskElectrode(200, -50, 10, 100), eye='LE')

    .. note::

        A stimulus can also be assigned later (see
        :py:attr:`~pulse2percept.implants.ProsthesisSystem.stim`).
    """

    def __init__(self, earray, stim=None, eye='RE'):
        self.earray = earray
        self.stim = stim
        self.eye = eye

    def get_params(self):
        return {'earray': self.earray, 'stim': self.stim, 'eye': self.eye}

    def check_stim(self, stim):
        """Quality-check the stimulus

        This method is executed every time a new value is assigned to ``stim``.

        No checks are performed by default, but the user can define their own
        checks in implants that inherit from
        :py:class:`~pulse2percept.implants.ProsthesisSystem`.

        Parameters
        ----------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus` source type
            A valid source type for the
            :py:class:`~pulse2percept.stimuli.Stimulus` object (e.g., scalar,
            NumPy array, pulse train).
        """
        pass

    @property
    def earray(self):
        """Electrode array

        """
        return self._earray

    @earray.setter
    def earray(self, earray):
        """Electrode array setter (called upon ``self.earray = earray``)"""
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
        NumPy arrays, and dictionaries (see
        :py:class:`~pulse2percept.stimuli.Stimulus` for a complete list).

        A stimulus can be assigned either in the
        :py:class:`~pulse2percept.implants.ProsthesisSystem` constructor
        or later by assigning a value to `stim`.

        .. note::
           Unless when using dictionary notation, the number of stimuli must
           equal the number of electrodes in ``earray``.

        Examples
        --------
        Send a biphasic pulse to an implant made from a single
        :py:class:`~pulse2percept.implants.DiskElectrode`:

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
        """Stimulus setter (called upon ``self.stim = data``)"""
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

        A :py:class:`~pulse2percept.implants.ProsthesisSystem` can be implanted
        either in a left eye ('LE') or right eye ('RE'). Models such as
        :py:class:`~pulse2percept.models.AxonMapModel` will treat left and
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

        This is equivalent to calling ``earray.n_electrodes``.
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
        ``earray.electrodes``. For convenience, electrodes can also be accessed
        via ``items``.

        Examples
        --------
        Save the x-coordinates of all electrodes of Argus I in a dictionary:

        >>> from pulse2percept.implants import ArgusI
        >>> xcoords = {}
        >>> for name, electrode in ArgusI().items():
        ...     xcoords[name] = electrode.x

        """
        return self.earray.items()
