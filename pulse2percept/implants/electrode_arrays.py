"""`ElectrodeArray`, `ElectrodeGrid`"""
import numpy as np
from collections import OrderedDict
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

from .electrodes import Electrode, PointSource, DiskElectrode
from ..utils import PrettyPrint, bijective26_name
from ..utils.constants import ZORDER


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
    OrderedDict([(0,
                  DiskElectrode(activated=True, name=None, r=100..., x=0..., y=0...,
                  z=0...))])

    Electrode array made from a single DiskElectrode with name 'A1':

    >>> from pulse2percept.implants import ElectrodeArray, DiskElectrode
    >>> earray = ElectrodeArray({'A1': DiskElectrode(0, 0, 0, 100)})
    >>> earray.electrodes  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    OrderedDict([('A1',
                  DiskElectrode(activated=True, name=None, r=100..., x=0..., y=0...,
                  z=0...))])

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('_electrodes',)

    def __init__(self, electrodes):
        self._electrodes = OrderedDict()
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
        if name in self.electrode_names:
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
        if name not in self.electrode_names:
            raise ValueError(("Cannot remove electrode: key '%s' does not "
                              "exist") % name)
        del self.electrodes[name]

    def activate(self, electrodes):
        if np.isscalar(electrodes):
            if electrodes == 'all':
                electrodes = self.electrode_names
            else:
                electrodes = [electrodes]
        for electrode in electrodes:
            self.__getitem__(electrode).activated = True

    def deactivate(self, electrodes):
        if np.isscalar(electrodes):
            if electrodes == 'all':
                electrodes = self.electrode_names
            else:
                electrodes = [electrodes]
        for electrode in electrodes:
            self.__getitem__(electrode).activated = False

    def plot(self, annotate=False, autoscale=True, ax=None):
        """Plot the electrode array

        Parameters
        ----------
        annotate : bool, optional
            Flag whether to label electrodes in the implant.
        autoscale : bool, optional
            Whether to adjust the x,y limits of the plot to fit the implant
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            A Matplotlib axes object. If None, will either use the current axes
            (if exists) or create a new Axes object.

        Returns
        -------
        ax : ``matplotlib.axes.Axes``
            Returns the axis object of the plot
        """
        if ax is None:
            ax = plt.gca()
        ax.set_aspect('equal')
        patches = []
        for name, electrode in self.electrodes.items():
            # Rather than calling electrode.plot(), generate all the patch
            # objects and add them to a collection:
            kwargs = electrode.plot_kwargs
            if not electrode.activated:
                kwargs = electrode.plot_deactivated_kwargs
            if isinstance(electrode.plot_patch, list):
                # Special case: draw multiple objects per electrode
                for p, kw in zip(electrode.plot_patch, kwargs):
                    patches.append(p((electrode.x, electrode.y), **kw))
            else:
                # Regular use case: single object
                patches.append(electrode.plot_patch((electrode.x, electrode.y),
                                                    **kwargs))
            if annotate:
                ax.text(electrode.x, electrode.y, name, ha='center',
                        va='center',  color='black', size='large',
                        bbox={'boxstyle': 'square,pad=-0.2', 'ec': 'none',
                              'fc': (1, 1, 1, 0.7)},
                        zorder=ZORDER['annotate'])
        ax.add_collection(PatchCollection(patches, match_original=True,
                                          zorder=ZORDER['annotate']))
        if autoscale:
            ax.autoscale(True)
        ax.set_xlabel('x (microns)')
        ax.set_ylabel('y (microns)')
        return ax

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
                key = list(self.electrode_names)[item]
                return self.electrodes[key]
            except IndexError:
                raise StopIteration
            return None

    def __iter__(self):
        return iter(self.electrodes)

    @property
    def n_electrodes(self):
        return len(self.electrodes)

    @property
    def electrodes(self):
        """Return all electrode names and objects in the electrode array

        Internally, electrodes are stored in an ordered dictionary.
        You can iterate over different electrodes in the array as follows:

        .. code::

            for name, electrode in earray.electrodes.items():
                print(name, electrode)

        You can access an individual electrode by indexing directly into the
        electrode array object, e.g. ``earray['A1']`` or ``earray[0]``.

        """
        return self._electrodes

    @property
    def electrode_names(self):
        """Return a list of all electrode names in the array"""
        return list(self.electrodes.keys())

    @property
    def electrode_objects(self):
        """Return a list of all electrode objects in the array"""
        return list(self.electrodes.values())


def _get_alphabetic_names(n_electrodes):
    """Create alphabetic electrode names: A-Z, AA-AZ, BA-BZ, etc. """
    return [bijective26_name(i) for i in range(n_electrodes)]


def _get_numeric_names(n_electrodes):
    """Create numeric electrode names: 1-n"""
    return [str(i) for i in range(1, n_electrodes + 1)]


class ElectrodeGrid(ElectrodeArray):
    """2D grid of electrodes

    Parameters
    ----------
    shape : (rows, cols)
        A tuple containing the number of rows x columns in the grid
    spacing : double
        Electrode-to-electrode spacing in microns.
    type : {'rect', 'hex'}, optional
        Grid type ('rect': rectangular, 'hex': hexagonal).
    x/y/z : double
        3D location of the center of the grid.
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the nasal retina.
        Positive ``y`` values move the electrode into the superior retina.
        Positive ``z`` values move the electrode away from the retina into the
        vitreous humor (sometimes called electrode-retina distance).
    rot : double, optional
        Rotation of the grid in degrees (positive angle: counter-clockwise
        rotation on the retinal surface)
    names: (name_rows, name_cols), each of which either 'A' or '1'
        Naming convention for rows and columns, respectively.
        If 'A', rows or columns will be labeled alphabetically: A-Z, AA-AZ,
        BA-BZ, CA-CZ, etc.
        If '1', rows or columns will be labeled numerically.
        Letters will always precede numbers in electrode names.
        For example ('1', 'A') will number rows numerically and columns
        alphabetically; first row: 'A1', 'B1', 'C1', NOT '1A', '1B', '1C'.
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
    ElectrodeGrid(rot=0, shape=(3, 4), spacing=20, type='hex')

    A rectangular electrode grid with 2 rows and 4 columns, made of disk
    electrodes with 10um radius spaced 20um apart, centered at (10, 20)um, and
    located 500um away from the retinal surface, with names like this:

    .. raw:: html

        A1 A2 A3 A4
        B1 B2 B3 B4

    >>> from pulse2percept.implants import ElectrodeGrid, DiskElectrode
    >>> ElectrodeGrid((2, 4), 20, x=10, y=20, z=500, names=('A', '1'), r=10,
    ...               type='rect', etype=DiskElectrode) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ElectrodeGrid(rot=0, shape=(2, 4), spacing=20, type='rect')

    There are three ways to access (e.g.) the last electrode in the grid,
    either by name (``grid['C3']``), by row/column index (``grid[2, 2]``), or
    by index into the flattened array (``grid[8]``):

    >>> from pulse2percept.implants import ElectrodeGrid
    >>> grid = ElectrodeGrid((3, 3), 20, names=('A', '1'))
    >>> grid['C3']  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    PointSource(activated=True, name='C3', x=20..., y=20...,
                z=0...)
    >>> grid['C3'] == grid[8] == grid[2, 2]
    True

    You can also access multiple electrodes at the same time by passing a
    list of indices/names (it's ok to mix-and-match):

    >>> from pulse2percept.implants import ElectrodeGrid, DiskElectrode
    >>> grid = ElectrodeGrid((3, 3), 20, etype=DiskElectrode, r=10)
    >>> grid[['A1', 1, (0, 2)]]  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [DiskElectrode(activated=True, name='A1', r=10..., x=-20.0,
                   y=-20.0, z=0...),
     DiskElectrode(activated=True, name='A2', r=10..., x=0.0,
                   y=-20.0, z=0...),
     DiskElectrode(activated=True, name='A3', r=10..., x=20.0,
                   y=-20.0, z=0...)]

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape', 'type', 'spacing', 'rot')

    def __init__(self, shape, spacing, x=0, y=0, z=0, rot=0, names=('A', '1'),
                 type='rect', orientation='horizontal', etype=PointSource,
                 **kwargs):
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
        if not isinstance(orientation, str):
            raise TypeError("'orientation' must be a string, either "
                            "'horizontal' or 'veritical'.")
        if type not in ['rect', 'hex']:
            raise ValueError("'type' must be either 'rect' or 'hex'.")
        if orientation not in ['horizontal', 'vertical']:
            raise ValueError(
                "'orientation' must be either 'horizontal' or 'vertical'.")
        if not issubclass(etype, Electrode):
            raise TypeError("'etype' must be a valid Electrode object.")
        if issubclass(etype, DiskElectrode):
            if 'r' not in kwargs.keys():
                raise ValueError("A DiskElectrode needs a radius ``r``.")
        if not isinstance(names, (tuple, list, np.ndarray)):
            raise TypeError("'names' must be a tuple or list, not "
                            "%s." % type(names))
        else:
            if len(names) != 2 and len(names) != np.prod(shape):
                raise ValueError("'names' must either have two entries for "
                                 "rows/columns or %d entries, not "
                                 "%d" % (np.prod(shape), len(names)))
        self.shape = shape
        self.type = type
        self.spacing = spacing
        self.rot = rot
        # Instantiate empty collection of electrodes. This dictionary will be
        # populated in a private method ``_set_egrid``:
        self._electrodes = OrderedDict()
        self._make_grid(x, y, z, rot, names, orientation, etype, **kwargs)

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = {'shape': self.shape, 'spacing': self.spacing,
                  'type': self.type, 'rot': self.rot}
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
            except (IndexError, KeyError, TypeError):
                # Access by [r, c] into 2D grid, e.g. grid[0, 3]:
                try:
                    idx = np.ravel_multi_index(item, self.shape)
                    return list(self.electrodes.values())[idx]
                except (KeyError, TypeError, ValueError):
                    # Index not found:
                    return None

    def _make_grid(self, x, y, z, rot, names, orientation, etype, **kwargs):
        """Private method to build the electrode grid"""
        n_elecs = np.prod(self.shape)
        rows, cols = self.shape

        # The user did not specify a unique naming scheme:
        if len(names) == 2 and np.prod(self.shape) != 2:
            name_rows, name_cols = names
            if not isinstance(name_rows, str):
                raise TypeError("Row name must be a string, not "
                                "%s." % type(name_rows))
            if not isinstance(name_cols, str):
                raise TypeError("Column name must be a string, not "
                                "%s." % type(name_cols))
            # Row names:
            if name_rows.isalpha():
                rws = _get_alphabetic_names(rows)
            elif name_rows.isdigit():
                rws = _get_numeric_names(rows)
            else:
                raise ValueError("Row name must be alphabetic or numeric.")
            # Column names:
            if name_cols.isalpha():
                clms = _get_alphabetic_names(cols)
            elif name_cols.isdigit():
                clms = _get_numeric_names(cols)
            else:
                raise ValueError("Column name must be alphabetic or numeric.")
            # Letters before digits:
            if name_cols.isalpha() and not name_rows.isalpha():
                names = [clms[j] + rws[i] for i in range(len(rws))
                         for j in range(len(clms))]
            else:
                names = [rws[i] + clms[j] for i in range(len(rws))
                         for j in range(len(clms))]

        if isinstance(z, (list, np.ndarray)):
            # Specify different height for every electrode in a list:
            z_arr = np.asarray(z).flatten()
            if z_arr.size != n_elecs:
                raise ValueError("If `h` is a list, it must have %d entries, "
                                 "not %d." % (n_elecs, len(z)))
        else:
            # If `z` is a scalar, choose same height for all electrodes:
            z_arr = np.ones(n_elecs, dtype=float) * z

        spc = self.spacing
        if self.type.lower() == 'rect':
            # Rectangular grid from x, y coordinates:
            # For example, cols=3 with spacing=100 should give: [-100, 0, 100]
            x_arr = (np.arange(cols) * spc - (cols / 2.0 - 0.5) * spc)
            y_arr = (np.arange(rows) * spc - (rows / 2.0 - 0.5) * spc)
            x_arr, y_arr = np.meshgrid(x_arr, y_arr, sparse=False)
        elif self.type.lower() == 'hex':
            # When orientation is horizontal, x_arr is shifting to create the
            # hex grid
            if orientation == 'horizontal':
                # Hexagonal grid from x,y coordinates:
                x_arr_lshift = ((np.arange(cols) * spc -
                                 (cols / 2.0 - 0.5) * spc - spc * 0.25))
                x_arr_rshift = ((np.arange(cols) * spc -
                                 (cols / 2.0 - 0.5) * spc + spc * 0.25))
                y_arr = (np.arange(rows) * np.sqrt(3) * spc / 2.0 -
                         (rows / 2.0 - 0.5) * spc)
                x_arr_lshift, _ = np.meshgrid(x_arr_lshift, y_arr)
                x_arr_rshift, y_arr_rshift = np.meshgrid(x_arr_rshift, y_arr)
                # Shift every other row to get an interleaved pattern:
                x_arr = []
                for row in range(rows):
                    if row % 2:
                        x_arr.append(x_arr_rshift[row])
                    else:
                        x_arr.append(x_arr_lshift[row])
                x_arr = np.array(x_arr)
                y_arr = y_arr_rshift
            # When orientation is vertical, y_arr is shifting to create the hex
            # grid
            elif orientation == 'vertical':
                # Hexagonal grid from x,y coordinates:
                x_arr = (np.arange(cols) * np.sqrt(3) * spc / 2.0 -
                         (cols / 2.0 - 0.5) * spc)
                y_arr_downshift = ((np.arange(rows) * spc -
                                    (rows / 2.0 - 0.5) * spc - spc * 0.25))
                y_arr_upshift = ((np.arange(rows) * spc -
                                  (rows / 2.0 - 0.5) * spc + spc * 0.25))
                x_arr_downshift, y_arr_downshift = np.meshgrid(x_arr,
                                                               y_arr_downshift,
                                                               sparse=False)
                x_arr_upshift, y_arr_upshift = np.meshgrid(x_arr,
                                                           y_arr_upshift,
                                                           sparse=False)
                # Shift every other column to get an interleaved pattern:
                y_arr = np.zeros(shape=(rows, cols))
                for row in range(rows):
                    for col in range(cols):
                        if col % 2:
                            y_arr[row][col] = y_arr_downshift[row][col]
                        else:
                            y_arr[row][col] = y_arr_upshift[row][col]
                x_arr = x_arr_upshift
                y_arr = np.array(y_arr)
        else:
            raise NotImplementedError

        # Rotate the grid:
        rot_rad = np.deg2rad(rot)
        rotmat = np.array([np.cos(rot_rad), -np.sin(rot_rad),
                           np.sin(rot_rad), np.cos(rot_rad)]).reshape((2, 2))
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
                self.add_electrode(name, DiskElectrode(x, y, z, r, name=name))
        else:
            # Pass keyword arguments to the electrode constructor:
            for x, y, z, name in zip(x_arr, y_arr, z_arr, names):
                self.add_electrode(name, etype(x, y, z, name=name, **kwargs))
