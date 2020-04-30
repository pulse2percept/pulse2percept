"""`Electrode`, `PointSource`, `DiskElectrode`, `ElectrodeArray`,
   `ElectrodeGrid`, `ProsthesisSystem`"""
import numpy as np
from abc import ABCMeta, abstractmethod
from copy import deepcopy
# Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working:
from collections.abc import Sequence
from collections import OrderedDict


from ..stimuli import Stimulus
from ..utils import PrettyPrint


class Electrode(PrettyPrint, metaclass=ABCMeta):
    """Electrode

    Abstract base class for all electrodes.
    """
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x, y, z):
        if isinstance(x, (Sequence, np.ndarray)):
            raise TypeError("x must be a scalar, not %s." % (type(x)))
        if isinstance(y, (Sequence, np.ndarray)):
            raise TypeError("y must be a scalar, not %s." % type(y))
        if isinstance(z, (Sequence, np.ndarray)):
            raise TypeError("z must be a scalar, not %s." % type(z))
        self.x = x
        self.y = y
        self.z = z

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        return {'x': self.x, 'y': self.y, 'z': self.z}

    @abstractmethod
    def electric_potential(self, x, y, z, *args, **kwargs):
        raise NotImplementedError


class PointSource(Electrode):
    """Point source"""
    # Frozen class: User cannot add more class attributes
    __slots__ = ()

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
    # Frozen class: User cannot add more class attributes
    __slots__ = ('r',)

    def __init__(self, x, y, z, r):
        super(DiskElectrode, self).__init__(x, y, z)
        if isinstance(r, (Sequence, np.ndarray)):
            raise TypeError("Electrode radius must be a scalar.")
        if r <= 0:
            raise ValueError("Electrode radius must be > 0, not %f." % r)
        self.r = r

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
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

        .. note::

            If you pass multiple electrodes in a dictionary, the keys of the
            dictionary will automatically be sorted. Thus the original order
            of electrodes might not be preserved.

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
    # Frozen class: User cannot add more class attributes
    __slots__ = ('_electrodes',)

    def __init__(self, electrodes):
        self.electrodes = OrderedDict()
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

    @property
    def electrodes(self):
        return self._electrodes

    @electrodes.setter
    def electrodes(self, electrodes):
        self._electrodes = electrodes

    @property
    def n_electrodes(self):
        return len(self.electrodes)

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
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
        self._electrodes.update({name: electrode})

    def remove_electrode(self, name):
        """Remove an electrode from the array

        Parameter
        ----------
        name: int|str|...
            Electrode name or index
        """
        if name not in self.electrodes.keys():
            raise ValueError(("Cannot remove electrode: key '%s' does not "
                              "exist") % name)
        del self.electrodes[name]

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
    """2D grid of electrodes

    Parameters
    ----------
    shape : (rows, cols)
        A tuple containing the number of rows x columns in the grid
    spacing : double
        Electrode-to-electrode spacing in microns.
    type : 'rect' or 'hex', optional, default: 'rect'
        Grid type ('rect': rectangular, 'hex': hexagonal).
    x, y, z : double, optional, default: (0,0,0)
        3D coordinates of the center of the grid
    rot : double, optional, default: 0rad
        Rotation of the grid in radians (positive angle: counter-clockwise
        rotation on the retinal surface)
    names: (name_rows, name_cols), each of which either 'A' or '1'
        Naming convention for rows and columns, respectively.
        If 'A', rows or columns will be labeled alphabetically.
        If '1', rows or columns will be labeled numerically.
        Columns and rows may only be strings and integers.
        For example ('1', 'A') will number rows numerically and columns
        alphabetically.
    etype : :py:class:`~pulse2percept.implants.Electrode`, optional
        A valid Electrode class. By default,
        :py:class:`~pulse2percept.implants.PointSource` is used.
    **kwargs :
        Any additional arguments that should be passed to the
        :py:class:`~pulse2percept.implants.Electrode` constructor, such as
        radius ``r`` for :py:class:`~pulse2percept.implants.DiskElectrode`.
        See examples below.

    Examples
    --------
    A hexagonal electrode grid with 3 rows and 4 columns, made of disk
    electrodes with 10um radius spaced 20um apart, centered at (10, 20)um, and
    located 500um away from the retinal surface, with names like this:

    .. raw:: html

        A1    A2    A3    A4
           B1    B2    B3    B4
        C1    C2    C3    C4

    >>> from pulse2percept.implants import ElectrodeGrid, DiskElectrode
    >>> ElectrodeGrid((3, 4), 20, x=10, y=20, z=500, names=('A', '1'), r=10,
    ...               type='hex', etype=DiskElectrode) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ElectrodeGrid(shape=(3, 4), type='hex')

    A rectangulr electrode grid with 2 rows and 4 columns, made of disk
    electrodes with 10um radius spaced 20um apart, centered at (10, 20)um, and
    located 500um away from the retinal surface, with names like this:

    .. raw:: html

        A1 A2 A3 A4
        B1 B2 B3 B4

    >>> from pulse2percept.implants import ElectrodeGrid, DiskElectrode
    >>> ElectrodeGrid((2, 4), 20, x=10, y=20, z=500, names=('A', '1'), r=10,
    ...               type='rect', etype=DiskElectrode) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ElectrodeGrid(shape=(2, 4), type='rect')

    There are three ways to access (e.g.) the last electrode in the grid,
    either by name (``grid['C3']``), by row/column index (``grid[2, 2]``), or
    by index into the flattened array (``grid[8]``):

    >>> from pulse2percept.implants import ElectrodeGrid
    >>> grid = ElectrodeGrid((3, 3), 20, names=('A', '1'))
    >>> grid['C3']  # doctest: +ELLIPSIS
    PointSource(x=20..., y=20..., z=0...)
    >>> grid['C3'] == grid[8] == grid[2, 2]
    True

    You can also access multiple electrodes at the same time by passing a
    list of indices/names (it's ok to mix-and-match):

    >>> from pulse2percept.implants import ElectrodeGrid, DiskElectrode
    >>> grid = ElectrodeGrid((3, 3), 20, etype=DiskElectrode, r=10)
    >>> grid[['A1', 1, (0, 2)]]  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [DiskElectrode(r=10..., x=-20.0, y=-20.0, z=0...),
     DiskElectrode(r=10..., x=0.0, y=-20.0, z=0...),
     DiskElectrode(r=10..., x=20.0, y=-20.0, z=0...)]

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape', 'type')

    def __init__(self, shape, spacing, x=0, y=0, z=0, rot=0, names=('A', '1'),
                 type='rect', etype=PointSource, **kwargs):
        if not isinstance(names, (tuple, list, np.ndarray)):
            raise TypeError("'names' must be a tuple/list of (rows, cols)")
        if not isinstance(shape, (tuple, list, np.ndarray)):
            raise TypeError("'shape' must be a tuple/list of (rows, cols)")
        if len(shape) != 2:
            raise ValueError("'shape' must have two elements: (rows, cols)")
        if np.prod(shape) <= 0:
            raise ValueError("Grid must have all non-zero rows and columns.")
        if not isinstance(type, str):
            raise TypeError("'type' must be a string, either 'rect' or 'hex'.")
        if type not in ['rect', 'hex']:
            raise ValueError("'type' must be either 'rect' or 'hex'.")
        if not issubclass(etype, Electrode):
            raise TypeError("'etype' must be a valid Electrode object.")
        if issubclass(etype, DiskElectrode):
            if 'r' not in kwargs.keys():
                raise ValueError("A DiskElectrode needs a radius ``r``.")

        self.shape = shape
        self.type = type
        # Instantiate empty collection of electrodes. This dictionary will be
        # populated in a private method ``_set_egrid``:
        self.electrodes = OrderedDict()
        self._make_grid(spacing, x, y, z, rot, names, etype, **kwargs)

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = {'shape': self.shape, 'type': self.type}
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

    def _make_grid(self, spacing, x, y, z, rot, names, etype, **kwargs):
        """Private method to build the electrode grid"""
        n_elecs = np.prod(self.shape)
        rows, cols = self.shape

        # The user did not specify a unique naming scheme:
        if len(names) == 2:
            name_rows, name_cols = names
            # Create electrode names, using either A-Z or 1-n:
            if name_rows.isalpha():
                rws = [chr(i) for i in range(ord(name_rows),
                                             ord(name_rows) + rows + 1)]
            elif name_rows.isdigit():
                rws = [str(i) for i in range(
                    int(name_rows), rows + int(name_rows))]
            else:
                raise ValueError("rows must be alphabetic or numeric")

            if name_cols.isalpha():
                clms = [chr(i) for i in range(ord(name_cols),
                                              ord(name_cols) + cols)]
            elif name_cols.isdigit():
                clms = [str(i) for i in range(int(name_cols),
                                              cols + int(name_cols))]
            else:
                raise ValueError("Columns must be alphabetic or numeric.")

            # facilitating Argus I naming scheme
            if name_cols.isalpha() and not name_rows.isalpha():
                names = [clms[j] + rws[i] for i in range(len(rws))
                         for j in range(len(clms))]
            else:
                names = [rws[i] + clms[j] for i in range(len(rws))
                         for j in range(len(clms))]
        else:
            if len(names) != n_elecs:
                raise ValueError("If `names` specifies more than row/column "
                                 "names, it must have %d entries, not "
                                 "%d)." % (n_elecs, len(names)))

        if isinstance(z, (list, np.ndarray)):
            # Specify different height for every electrode in a list:
            z_arr = np.asarray(z).flatten()
            if z_arr.size != n_elecs:
                raise ValueError("If `h` is a list, it must have %d entries, "
                                 "not %d." % (n_elecs, len(z)))
        else:
            # If `z` is a scalar, choose same height for all electrodes:
            z_arr = np.ones(n_elecs, dtype=float) * z

        spc = spacing
        if self.type.lower() == 'rect':
            # Rectangular grid from x, y coordinates:
            # For example, cols=3 with spacing=100 should give: [-100, 0, 100]
            x_arr = (np.arange(cols) * spc - (cols / 2.0 - 0.5) * spc)
            y_arr = (np.arange(rows) * spc - (rows / 2.0 - 0.5) * spc)
            x_arr, y_arr = np.meshgrid(x_arr, y_arr, sparse=False)
        elif self.type.lower() == 'hex':
            # Hexagonal grid from x,y coordinates:
            x_arr_lshift = (np.arange(cols) * spc - (cols / 2.0 - 0.5) * spc -
                            spc * 0.25)
            x_arr_rshift = (np.arange(cols) * spc - (cols / 2.0 - 0.5) * spc +
                            spc * 0.25)
            y_arr = (np.arange(rows) * np.sqrt(3) * spc / 2.0 -
                     (rows / 2.0 - 0.5) * spc)
            x_arr_lshift, y_arr_lshift = np.meshgrid(x_arr_lshift, y_arr,
                                                     sparse=False)
            x_arr_rshift, y_arr_rshift = np.meshgrid(x_arr_rshift, y_arr,
                                                     sparse=False)
            # Shift every other row to get an interleaved pattern:
            x_arr = []
            for row in range(rows):
                if row % 2:
                    x_arr.append(x_arr_rshift[row])
                else:
                    x_arr.append(x_arr_lshift[row])
            x_arr = np.array(x_arr)
            y_arr = y_arr_rshift
        else:
            raise NotImplementedError

        # Rotate the grid:
        rotmat = np.array([np.cos(rot), -np.sin(rot),
                           np.sin(rot), np.cos(rot)]).reshape((2, 2))
        xy = np.matmul(rotmat, np.vstack((x_arr.flatten(), y_arr.flatten())))
        x_arr = xy[0, :]
        y_arr = xy[1, :]

        # Apply offset to make the grid centered at (x, y):
        x_arr += x
        y_arr += y

        if issubclass(etype, DiskElectrode):
            if isinstance(kwargs['r'], (list, np.ndarray)):
                # Specify different radius for every electrode in a list:
                if len(kwargs['r']) != n_elecs:
                    err_s = ("If `r` is a list, it must have %d entries, not "
                             "%d)." % (n_elecs, len(kwargs['r'])))
                    raise ValueError(err_s)
                r_arr = kwargs['r']
            else:
                # If `r` is a scalar, choose same radius for all electrodes:
                r_arr = np.ones(n_elecs, dtype=float) * kwargs['r']

            # Create a grid of DiskElectrode objects:
            for x, y, z, r, name in zip(x_arr, y_arr, z_arr, r_arr, names):
                self.add_electrode(name, DiskElectrode(x, y, z, r))
        elif issubclass(etype, PointSource):
            # Create a grid of PointSource objects:
            for x, y, z, name in zip(x_arr, y_arr, z_arr, names):
                self.add_electrode(name, PointSource(x, y, z))
        else:
            raise NotImplementedError


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
    # Frozen class: User cannot add more class attributes
    __slots__ = ('_earray', '_stim', '_eye')

    def __init__(self, earray, stim=None, eye='RE'):
        self.earray = earray
        self.stim = stim
        self.eye = eye

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
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
            # For convenience, build an array from a single electrode:
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
            if isinstance(data, Stimulus):
                # Already a stimulus object:
                stim = Stimulus(data, extrapolate=True)
            elif isinstance(data, dict):
                # Electrode names already provided by keys:
                stim = Stimulus(data, extrapolate=True)
            else:
                # Use electrode names as stimulus coordinates:
                stim = Stimulus(data, electrodes=list(self.earray.keys()),
                                extrapolate=True)

            # Make sure all electrode names are valid:
            for electrode in stim.electrodes:
                # Invalid index will return None:
                if not self.earray[electrode]:
                    raise ValueError("Electrode '%s' not found in "
                                     "implant." % electrode)
            # Perform safety checks, etc.:
            self.check_stim(stim)
            # Store stimulus:
            self._stim = deepcopy(stim)

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
