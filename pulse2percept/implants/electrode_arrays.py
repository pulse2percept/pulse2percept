""":py:class:`~pulse2percept.implants.ElectrodeArray`, 
   :py:class:`~pulse2percept.implants.ElectrodeGrid`"""
from matplotlib.colors import Normalize
import numpy as np
from collections import Counter, OrderedDict
from collections.abc import Iterable
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from skimage.transform import SimilarityTransform
from copy import deepcopy

from .electrodes import Electrode, PointSource, HexElectrode
from ..stimuli.names import ElectrodeNames
from ..units import Quantity, as_value, deg, um
from ..utils import PrettyPrint, bijective26_name
from ..utils.constants import ZORDER


def _is_electrode_collection(selector):
    """Whether an electrode selector names several electrodes or just one

    Three things are one electrode however sequence-like they look: a name, a
    ``(row, col)`` pair on an
    :py:class:`~pulse2percept.implants.ElectrodeGrid`, and an index. Anything
    else that can be iterated is a collection -- a list, an array, or the
    :py:class:`~pulse2percept.stimuli.ElectrodeNames` a stimulus reports.

    The tuple carve-out is what makes ``grid[0, 0]`` and
    ``grid.coordinates(electrodes=(0, 0))`` mean the same thing, and the
    string carve-out keeps ``'A1'`` from being read as the electrodes 'A' and
    '1'.
    """
    if isinstance(selector, (str, bytes, tuple)):
        return False
    if isinstance(selector, np.ndarray):
        # A 0-d array holds one item and cannot be iterated:
        return selector.ndim > 0
    return isinstance(selector, Iterable)


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
    >>> electrode_array = ElectrodeArray(DiskElectrode(0, 0, 0, 100))
    >>> electrode_array.electrodes  # doctest: +SKIP
    OrderedDict([(0,
                  DiskElectrode(activated=True, name=None, radius=100...,
                  x=0..., y=0..., z=0...))])

    Electrode array made from a single DiskElectrode with name 'A1':

    >>> from pulse2percept.implants import ElectrodeArray, DiskElectrode
    >>> electrode_array = ElectrodeArray({'A1': DiskElectrode(0, 0, 0, 100)})
    >>> electrode_array.electrodes  # doctest: +SKIP
    OrderedDict([('A1',
                  DiskElectrode(activated=True, name=None, radius=100...,
                  x=0..., y=0..., z=0...))])

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('_electrodes',)

    #: The unit electrode coordinates are stored in, i.e. what the plain
    #: numbers returned by :py:meth:`coordinates` mean by default.
    coordinate_unit = um

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
            raise TypeError((f"electrodes must be a list or dict, not "
                             f"{type(electrodes)}"))

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        return {'electrodes': self.electrodes,
                'n_electrodes': self.n_electrodes}

    def coordinates(self, unit=None, electrodes=None):
        """Positions of the electrodes in the array

        The one place to ask an implant where its electrodes are. Code that
        needs the coordinates in a particular unit says so here, instead of
        reading ``electrode.x`` and knowing that electrodes happen to store
        microns.

        .. versionadded:: 0.10.0

        Parameters
        ----------
        unit : :py:class:`~pulse2percept.units.Unit`, optional
            Length unit to express the coordinates in. If None, they are
            returned as they are stored (microns).
        electrodes : optional
            Which electrodes to return, either by name, index into the
            flattened array, or a ``(row, col)`` pair on an
            :py:class:`~pulse2percept.implants.ElectrodeGrid`.
            Everything else is treated as a collection.

        Returns
        -------
        coords : (n_electrodes, 3) np.ndarray
            One ``[x, y, z]`` row per electrode

        Examples
        --------
        >>> from pulse2percept.implants import ArgusII
        >>> from pulse2percept.units import mm
        >>> ArgusII().electrode_array.coordinates(mm)[0]
        array([-2.5875, -1.4375,  0.    ])
        >>> ArgusII().electrode_array.coordinates(electrodes=['F10', 'A1'])
        array([[ 2587.5,  1437.5,     0. ],
               [-2587.5, -1437.5,     0. ]])

        """
        if electrodes is None:
            elecs = self.electrode_objects
        else:
            if _is_electrode_collection(electrodes):
                elecs = [self[name] for name in electrodes]
            else:
                # A name, an index, or a grid's `(row, col)` pair:
                elecs = [self[electrodes]]
        xyz = np.array([[e.x, e.y, e.z] for e in elecs],
                       dtype=float).reshape((-1, 3))
        if unit is None:
            return xyz
        return Quantity(xyz, self.coordinate_unit).to_value(unit)

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
            raise TypeError(f"Electrode {name} must be an Electrode object, not "
                            f"{type(electrode)}.")
        if name in self._electrodes:
            raise ValueError(f"Cannot add electrode: key '{name}' already "
                             f"exists.")
        self._electrodes.update({name: electrode})

    def remove_electrode(self, name):
        """Remove an electrode from the array

        Parameter
        ----------
        name: int|str|...
            Electrode name or index
        """
        if name not in self._electrodes:
            raise ValueError(f"Cannot remove electrode: key '{name}' does not "
                             f"exist")
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

    def plot(self, annotate=False, autoscale=True, ax=None, color_stim=None, cmap='OrRd'):
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
        color_stim : ``pulse2percept.stimuli.Stimulus``, or None
            If provided, colors the electrode_array based on the stimulus
            amplitudes
        cmap : str
            Matplotlib colormap to use for stimulus coloring.

        Returns
        -------
        ax : ``matplotlib.axes.Axes``
            Returns the axis object of the plot
        """
        if ax is None:
            ax = plt.gca()
        ax.set_aspect('equal')
        patches = []
        cm = None
        norm = None
        if color_stim is not None:
            cm = plt.get_cmap(cmap)
            norm = Normalize(vmin=0, vmax=np.max(color_stim.data))
        for name, electrode in self.electrodes.items():
            # Rather than calling electrode.plot(), generate all the patch
            # objects and add them to a collection:
            if electrode.activated:
                kwargs = deepcopy(electrode.plot_kwargs)
                if color_stim is not None and name in color_stim.electrodes:
                    amp = np.max(color_stim[name])
                    if amp != 0:
                        kwargs['fc'] = cm(norm(amp), alpha=0.8)
            else:
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
        patch_collection = PatchCollection(patches, match_original=True,
                                          zorder=ZORDER['foreground'], cmap=cm, norm=norm)
        ax.add_collection(patch_collection)
        ax._sci(patch_collection) # enables plt.colormap()
        if autoscale:
            ax.autoscale(True)
        # dont relabel if its already set
        if ax.get_xlabel() == "":
            ax.set_xlabel('x (microns)')
        if ax.get_ylabel() == "":
            ax.set_ylabel('y (microns)')
        return ax

    def __getitem__(self, item):
        """Return an electrode from the array

        Parameters
        ----------
        item : str, int, slice, or list thereof
            An electrode name, or an index into the array (negative indices
            count from the end). A slice, a list, or a NumPy array returns a
            list of electrodes.

        Returns
        -------
        electrode : :py:class:`~pulse2percept.implants.Electrode` or list

        Raises
        ------
        KeyError
            If ``item`` names an electrode the array does not have.
        IndexError
            If ``item`` is an index outside the array.
        TypeError
            If ``item`` is neither a name nor an index.

        Notes
        -----
        *  A name is looked up before an integer is read as a position, so an
           array whose electrodes are *named* 0, 1, 2 answers with the
           electrode of that name rather than the one in that position.

        .. versionchanged:: 0.11.0
            A lookup that fails raises instead of returning ``None``.
        """
        if _is_electrode_collection(item):
            return [self[i] for i in item]
        try:
            return self._electrodes[item]
        except (KeyError, TypeError):
            pass
        return self._by_position(item)

    def _by_position(self, item):
        """Return the ``item``-th electrode, for an ``item`` that is no name"""
        if isinstance(item, slice):
            return self.electrode_objects[item]
        if isinstance(item, (int, np.integer)):
            try:
                return self.electrode_objects[item]
            except IndexError:
                raise IndexError(f"Index {item} is out of range for an array "
                                 f"of {len(self)} electrodes.") from None
        if isinstance(item, (str, bytes)):
            raise KeyError(item)
        raise TypeError(f"An electrode is selected by name or by integer "
                        f"index, not by {type(item).__name__}.")

    def __len__(self):
        """.. versionadded:: 0.11.0"""
        return len(self._electrodes)

    def __iter__(self):
        return iter(self.electrodes)

    def get(self, item, default=None):
        """Return an electrode, or ``default`` if the array has no such one

        .. versionadded:: 0.11.0

        Parameters
        ----------
        item : str, int, or list thereof
            An electrode selector, as ``electrode_array[item]`` takes.
        default : optional
            What to answer when the array has no such electrode.

        """
        try:
            return self[item]
        except (KeyError, IndexError):
            return default

    @property
    def n_electrodes(self):
        return len(self.electrodes)

    @property
    def electrodes(self):
        """Return all electrode names and objects in the electrode array

        Internally, electrodes are stored in an ordered dictionary.
        You can iterate over different electrodes in the array as follows:

        .. code::

            for name, electrode in electrode_array.electrodes.items():
                print(name, electrode)

        You can access an individual electrode by indexing directly into the
        electrode array object, e.g. ``electrode_array['A1']`` or
        ``electrode_array[0]``.

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


def _is_naming_scheme(names):
    """Whether a two-entry ``names`` gives the (rows, cols) naming scheme

    This is only ever in doubt on a grid with exactly two electrodes, where
    the same two entries could just as well be the two electrode names. A
    scheme entry is a token like 'A' or '1' (optionally reversed: '-A', '-1'),
    so ``('A', '1')`` is the scheme, while ``('C1', '4')`` cannot be one and
    must therefore be the names themselves. A list or array is always taken to
    be the names, which is how two electrodes can still be named 'A' and '1'
    if that is really what is wanted.
    """
    if not isinstance(names, tuple) or len(names) != 2:
        return False
    for name in names:
        if not isinstance(name, str):
            return False
        token = name.replace('-', '')
        if not token or not (token.isalpha() or token.isdigit()):
            return False
    return True


class ElectrodeGrid(ElectrodeArray):
    """2D grid of electrodes

    Parameters
    ----------
    shape : (rows, cols)
        A tuple containing the number of rows x columns in the grid
    spacing : double or (x_spacing, y_spacing)
        Electrode-to-electrode spacing in microns.
        Must be either a tuple specifying the spacing in x and y directions or
        a float (assuming the same spacing in x and y).
        If a tuple is specified for a horizontal hex grid, ``x_spacing`` will
        define the electrode-to-electrode distance, and ``y_spacing`` will
        define the vertical distance between adjacent hexagon centers.
        In a vertical hex grid, the order is reversed.
    grid_type : {'rect', 'hex'}, optional
        Grid type ('rect': rectangular, 'hex': hexagonal).

        .. versionchanged:: 0.11.0
            Renamed from ``type``.
    orientation : {'horizontal', 'vertical'}, optional
        Hex-grid orientation. ``'horizontal'`` staggers alternate rows;
        ``'vertical'`` staggers alternate columns. Hexagonal electrode bodies
        follow the grid orientation.
    x/y/z : double
        3D location (um) of the center of the grid.
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the nasal retina.
        Positive ``y`` values move the electrode into the superior retina.
        Positive ``z`` values move the electrode away from the retina into the
        vitreous humor (sometimes called electrode-retina distance).
    rot : double, optional
        Rotation of the grid in degrees (positive angle: counter-clockwise
        rotation on the retinal surface). A plain angle, not a unitful one:
        ``dva`` means visual angle, which is a different thing.
    names: (name_rows, name_cols), each of which either 'A' or '1'
        Naming convention for rows and columns, respectively.
        If 'A', rows or columns will be labeled alphabetically: A-Z, AA-AZ,
        BA-BZ, CA-CZ, etc. '-A' will reverse the order.
        If '1', rows or columns will be labeled numerically. '-1' will reverse.
        Letters will always precede numbers in electrode names.
        For example ('1', 'A') will number rows numerically and columns
        alphabetically; first row: 'A1', 'B1', 'C1', NOT '1A', '1B', '1C'.

        The default, ``('A', '1')``, is the same convention that
        :py:class:`~pulse2percept.stimuli.ElectrodeNames` uses to name the
        pixels of an :py:class:`~pulse2percept.stimuli.ImageStimulus`, and is
        generated by it. The other combinations exist to reproduce the naming
        of specific published implants and are not otherwise recommended.

        Alternatively, pass a list or NumPy array with one name per electrode
        to name them all explicitly. On a grid with exactly two electrodes the
        two readings collide, and only something that could be a scheme is
        read as one: ``names=('A', '1')`` gives 'A1', 'A2', whereas
        ``names=('C1', '4')`` names the two electrodes 'C1' and '4'. Pass a
        list (``names=['A', '1']``) to name two electrodes 'A' and '1'.

        .. versionchanged:: 0.10.0
            On a grid with exactly two electrodes, ``('A', '1')`` now yields
            'A1', 'A2' (was: 'A', '1'), consistent with every other shape.
    electrode_type : :py:class:`~pulse2percept.implants.Electrode`, optional
        A valid Electrode class. By default,
        :py:class:`~pulse2percept.implants.PointSource` is used.

        .. versionchanged:: 0.11.0
            Renamed from ``etype``.
    **electrode_params :
        Keyword arguments passed to the ``electrode_type`` constructor, such
        as ``radius`` for
        :py:class:`~pulse2percept.implants.DiskElectrode`. They are forwarded
        unchanged, except that ``radius`` may be given per electrode (see
        below).

    Notes
    -----
    *  ``z`` and ``radius`` may be given per electrode, as a list or array
       with one entry per grid position. Every other electrode parameter is
       one value shared by all electrodes.
    *  ``spacing``, ``x``, ``y``, ``z`` and ``radius`` may be given as plain
       numbers of microns or as unitful quantities, and may be mixed freely:
       ``spacing=(0.5 * mm, 600 * um)`` and ``z=[0 * um, 0.1 * mm, ...]`` both
       work. Any other electrode keyword is normalized by the electrode class
       it is passed to. See :py:mod:`pulse2percept.units`.

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
    >>> ElectrodeGrid((3, 4), 20, x=10, y=20, z=500, names=('A', '1'),
    ...               radius=10, grid_type='hex',
    ...               electrode_type=DiskElectrode)
    ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ElectrodeGrid(grid_type='hex', rot=0, shape=(3, 4), spacing=20)

    A rectangular electrode grid with 2 rows and 4 columns, made of disk
    electrodes with 10um radius spaced 20um apart, centered at (10, 20)um, and
    located 500um away from the retinal surface, with names like this:

    .. raw:: html

        A1 A2 A3 A4
        B1 B2 B3 B4

    >>> from pulse2percept.implants import ElectrodeGrid, DiskElectrode
    >>> ElectrodeGrid((2, 4), 20, x=10, y=20, z=500, names=('A', '1'),
    ...               radius=10, grid_type='rect',
    ...               electrode_type=DiskElectrode)
    ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ElectrodeGrid(grid_type='rect', rot=0, shape=(2, 4), spacing=20)

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
    >>> grid = ElectrodeGrid((3, 3), 20, electrode_type=DiskElectrode,
    ...                      radius=10)
    >>> grid[['A1', 1, (0, 2)]]  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [DiskElectrode(activated=True, name='A1', radius=10..., x=-20.0,
                   y=-20.0, z=0...),
     DiskElectrode(activated=True, name='A2', radius=10..., x=0.0,
                   y=-20.0, z=0...),
     DiskElectrode(activated=True, name='A3', radius=10..., x=20.0,
                   y=-20.0, z=0...)]

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape', 'grid_type', 'spacing', 'rot')

    def __init__(self, shape, spacing, x=0, y=0, z=0, rot=0, names=('A', '1'),
                 grid_type='rect', orientation='horizontal',
                 electrode_type=PointSource, **electrode_params):
        if not isinstance(names, (tuple, list, np.ndarray)):
            raise TypeError("'names' must be a tuple/list of (rows, cols)")
        if not isinstance(shape, (tuple, list, np.ndarray)):
            raise TypeError("'shape' must be a tuple/list of (rows, cols)")
        if len(shape) != 2:
            raise ValueError("'shape' must have two elements: (rows, cols)")
        if np.prod(shape) <= 0:
            raise ValueError("Grid must have all non-zero rows and columns.")
        if not isinstance(grid_type, str):
            raise TypeError("'grid_type' must be a string, either 'rect' or "
                            "'hex'.")
        if not isinstance(orientation, str):
            raise TypeError("'orientation' must be a string, either "
                            "'horizontal' or 'veritical'.")
        if grid_type not in ['rect', 'hex']:
            raise ValueError("'grid_type' must be either 'rect' or 'hex'.")
        if orientation not in ['horizontal', 'vertical']:
            raise ValueError(
                "'orientation' must be either 'horizontal' or 'vertical'.")
        if not isinstance(electrode_type, type) or \
                not issubclass(electrode_type, Electrode):
            raise TypeError("'electrode_type' must be a valid Electrode "
                            "class.")
        if not isinstance(names, (tuple, list, np.ndarray)):
            raise TypeError(f"'names' must be a tuple or list, not "
                            f"{type(names)}.")
        else:
            if len(names) != 2 and len(names) != np.prod(shape):
                raise ValueError(f"'names' must either have two entries for "
                                 f"rows/columns or {np.prod(shape)} entries, not "
                                 f"{len(names)}")
        # Normalized before anything is built with them: `_make_grid` lays out
        # the pitch from `spacing`, translates by (x, y), broadcasts `z` and
        # `radius` over the electrodes, and stores `spacing` on the grid
        # itself. Every other electrode parameter travels through
        # **electrode_params untouched and is normalized by the electrode
        # class it belongs to.
        spacing = as_value(spacing, um, 'spacing')
        x = as_value(x, um, 'x')
        y = as_value(y, um, 'y')
        z = as_value(z, um, 'z')
        if 'radius' in electrode_params:
            electrode_params['radius'] = as_value(electrode_params['radius'],
                                                  um, 'radius')
        # `deg` is an ordinary geometric angle; `dva` is visual angle, and is
        # rejected here:
        rot = as_value(rot, deg, 'rot')
        self.shape = shape
        self.grid_type = grid_type
        self.spacing = spacing
        self.rot = rot
        # Instantiate empty collection of electrodes. This dictionary will be
        # populated in a private method ``_set_egrid``:
        self._electrodes = OrderedDict()
        self._make_grid(x, y, z, rot, names, orientation, electrode_type,
                        **electrode_params)

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = {'shape': self.shape, 'spacing': self.spacing,
                  'grid_type': self.grid_type, 'rot': self.rot}
        return params

    def __getitem__(self, item):
        """Access electrode(s) in the grid

        Parameters
        ----------
        item : str, int, (row, col), or list thereof
            An electrode in the grid can be accessed in three ways:

            *  by name, e.g. ``grid['A1']``
            *  by index into the flattened array, e.g. ``grid[0]``
            *  by (row, column) index into the 2D grid, e.g. ``grid[0, 0]``

            Indices may be negative, and a slice, list, or NumPy array of
            any of the above returns a list of electrodes.

        Returns
        -------
        electrode : :py:class:`~pulse2percept.implants.Electrode` or list

        Raises
        ------
        KeyError
            If ``item`` names an electrode the grid does not have.
        IndexError
            If ``item`` is a flat index or a (row, col) pair outside the grid.
        TypeError
            If ``item`` is none of the three forms above.

        .. versionchanged:: 0.11.0
            A lookup that fails raises instead of returning ``None``.
        """
        if _is_electrode_collection(item):
            return [self[i] for i in item]
        try:
            return self._electrodes[item]
        except (KeyError, TypeError):
            pass
        if isinstance(item, tuple):
            return self._by_row_col(item)
        return self._by_position(item)

    def _by_row_col(self, item):
        """Return the electrode at a ``(row, col)`` position in the grid"""
        rows, cols = self.shape
        if len(item) != 2 or not all(isinstance(i, (int, np.integer))
                                     for i in item):
            raise TypeError(f"A tuple selects one electrode by (row, col); "
                            f"{item!r} is not one. Pass a list to select "
                            f"several electrodes.")
        row, col = item
        if not -rows <= row < rows or not -cols <= col < cols:
            raise IndexError(f"({row}, {col}) is out of range for a "
                             f"{rows}x{cols} grid.")
        return self.electrode_objects[(row % rows) * cols + col % cols]

    def _make_grid(self, x, y, z, rot, names, orientation, electrode_type,
                   **electrode_params):
        """Private method to build the electrode grid"""
        n_elecs = np.prod(self.shape)
        rows, cols = self.shape

        # A two-entry `names` is the (rows, cols) naming scheme -- except on a
        # grid that happens to have exactly two electrodes, where it could
        # just as well be both electrode names spelled out. There, only
        # something that could actually be a scheme is read as one, so that
        # ('A', '1') means the same thing at every grid size:
        if len(names) == 2 and (n_elecs != 2 or _is_naming_scheme(names)):
            name_rows, name_cols = names
            if not isinstance(name_rows, str):
                raise TypeError(f"Row name must be a string, not "
                                f"{type(name_rows)}.")
            if not isinstance(name_cols, str):
                raise TypeError(f"Column name must be a string, not "
                                f"{type(name_cols)}.")
            # Row names:
            reverse_rows = False
            if '-' in name_rows:
                reverse_rows = True
                name_rows = name_rows.replace('-', '')
            if name_rows.isalpha():
                rws = _get_alphabetic_names(rows)
            elif name_rows.isdigit():
                rws = _get_numeric_names(rows)
            else:
                raise ValueError("Row name must be alphabetic or numeric.")
            if reverse_rows:
                rws = rws[::-1]
            # Column names:
            reverse_cols = False
            if '-' in name_cols:
                reverse_cols = True
                name_cols = name_cols.replace('-', '')
            if name_cols.isalpha():
                clms = _get_alphabetic_names(cols)
            elif name_cols.isdigit():
                clms = _get_numeric_names(cols)
            else:
                raise ValueError("Column name must be alphabetic or numeric.")
            if reverse_cols:
                clms = clms[::-1]
            # Letters before digits:
            if name_cols.isalpha() and not name_rows.isalpha():
                names = [clms[j] + rws[i] for i in range(len(rws))
                         for j in range(len(clms))]
            elif (name_rows.isalpha() and name_cols.isdigit() and
                    not reverse_rows and not reverse_cols):
                # The canonical convention: a letter for the row, a number for
                # the column. This is the same scheme that names the pixels of
                # an ImageStimulus, so both come from the one implementation -
                # a generic grid and an image stimulus cannot drift apart.
                # The remaining schemes above and below exist to reproduce the
                # naming of specific published implants (ArgusI, Orion), and
                # are deliberately left as overrides:
                names = np.asarray(ElectrodeNames((rows, cols))).tolist()
            else:
                names = [rws[i] + clms[j] for i in range(len(rws))
                         for j in range(len(clms))]

        if isinstance(z, (list, np.ndarray)):
            # Specify different height for every electrode in a list:
            z_arr = np.asarray(z).flatten()
            if z_arr.size != n_elecs:
                raise ValueError(f"If `h` is a list, it must have {n_elecs} entries, "
                                 f"not {len(z)}.")
        else:
            # If `z` is a scalar, choose same height for all electrodes:
            z_arr = np.ones(n_elecs, dtype=float) * z

        # Spacing can be different for x and y (tuple) or the same (float):
        if isinstance(self.spacing, (list, np.ndarray, tuple)):
            x_spc, y_spc = self.spacing[:2]
        else:
            x_spc = y_spc = self.spacing
            if self.grid_type.lower() == 'hex':
                # In a hex grid, we need to adjust the spacing so that
                # neighboring electrodes are separated by self.spacing:
                if orientation.lower() == 'horizontal':
                    y_spc = x_spc * np.sqrt(3) / 2
                else:
                    x_spc = y_spc * np.sqrt(3) / 2

        # Start with a rectangular grid, laid out from the origin:
        x_arr = np.arange(cols, dtype=float) * x_spc
        y_arr = np.arange(rows, dtype=float) * y_spc
        x_arr, y_arr = np.meshgrid(x_arr, y_arr, sparse=False)
        if self.grid_type.lower() == 'hex':
            if orientation.lower() == 'horizontal':
                # Shift every other row:
                x_arr[::2] += 0.5 * x_spc
            else:
                # Shift every other column:
                y_arr[:, ::2] += 0.5 * y_spc
        # Center the lattice on (0, 0) once it is built, rather than assuming
        # what the stagger did to its extent. (x, y) is the middle of that
        # extent, not the centroid of the electrode centers: a hex grid with
        # an odd number of rows has one stagger more often than the other, and
        # its centroid sits a fraction of a pitch off center.
        x_arr -= 0.5 * (x_arr.min() + x_arr.max())
        y_arr -= 0.5 * (y_arr.min() + y_arr.max())

        # Rotate the grid and center at (x,y):
        tf = SimilarityTransform(rotation=np.deg2rad(rot), translation=[x, y])
        x_arr, y_arr = tf(np.vstack([x_arr.ravel(), y_arr.ravel()]).T).T

        if issubclass(electrode_type, HexElectrode):
            # Match the hexagonal body to the grid:
            electrode_params.setdefault('orientation', orientation)
            electrode_params.setdefault('rot', rot)
        # `radius` is the one electrode parameter the grid itself interprets,
        # because implants with two electrode sizes exist (e.g. ArgusI): a
        # list gives one radius per grid position. Everything else in
        # `electrode_params` is one value shared by all electrodes.
        radius = electrode_params.pop('radius', None)
        if radius is None:
            elecs = [electrode_type(ex, ey, ez, name=nm, **electrode_params)
                     for ex, ey, ez, nm in zip(x_arr, y_arr, z_arr, names)]
        else:
            if isinstance(radius, (list, np.ndarray)):
                if len(radius) != n_elecs:
                    raise ValueError(f"If `radius` is a list, it must have "
                                     f"{n_elecs} entries, not {len(radius)}.")
                r_arr = radius
            else:
                # Floated like `z`, so that an integer radius gives the same
                # electrodes a float one does:
                r_arr = np.ones(n_elecs, dtype=float) * radius
            elecs = [electrode_type(ex, ey, ez, radius=er, name=nm,
                                    **electrode_params)
                     for ex, ey, ez, er, nm in zip(x_arr, y_arr, z_arr, r_arr,
                                                   names)]
        # Populated in one shot rather than through ``add_electrode``: on a
        # grid every name is known up front, so a duplicate shows up as a
        # short dict instead of costing a lookup per electrode.
        self._electrodes = OrderedDict(zip(names, elecs))
        if len(self._electrodes) != n_elecs:
            dupe = next(nm for nm, n in Counter(names).items() if n > 1)
            raise ValueError(f"Cannot add electrode: key '{dupe}' already "
                             f"exists.")
