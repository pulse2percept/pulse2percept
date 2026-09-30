"""Lazily generated grid labels shared by image/video stimuli and electrode
grids.

Private: users access names through :py:attr:`Stimulus.electrodes` and
:py:attr:`ElectrodeGrid.electrode_names`.
"""
import re

import numpy as np

from ..utils.base import bijective26_name

# Channel suffixes for RGB/RGBA; other channel counts use numeric suffixes:
_CHANNEL_LABELS = {3: ('R', 'G', 'B'), 4: ('R', 'G', 'B', 'A')}

# 'A1', 'BC17', 'A1_R', 'A1_12': letters = row, digits = column, optional
# suffix = color channel:
_NAME_RE = re.compile(r'^([A-Z]+)([0-9]+)(?:_([A-Z0-9]+))?$')


def _bijective26_index(letters):
    """Inverse of :py:func:`~pulse2percept.utils.bijective26_name`

    Returns the integer for a letter code, e.g. 'A' -> 0, 'Z' -> 25,
    'AA' -> 26.
    """
    value = 0
    for char in letters:
        value = value * 26 + (ord(char) - 64)
    return value - 1


def _is_pure_selection(item):
    """Return True if an index expression cannot repeat elements

    Slices, ellipses and boolean masks preserve uniqueness; integer (fancy)
    indexing does not (``names[[0, 0]]``). Lets
    :py:class:`~pulse2percept.stimuli.Stimulus` skip its duplicate-name check.
    """
    if item is Ellipsis or isinstance(item, slice):
        return True
    if isinstance(item, tuple):
        return all(_is_pure_selection(i) for i in item)
    if isinstance(item, np.ndarray):
        return item.dtype == bool
    return False


class _GridNames:
    """Lazily generated names for a grid of pixels or electrodes

    Names each element of a (rows x columns [x channels]) grid by position:
    letters = row, digits = column, optional suffix = color channel. The first
    pixel of an RGB image is ``'A1_R'``; row 3, column 12 of a grayscale image
    is ``'C12'``.

    Only the grid shape (and, for a subset such as a cropped image, the kept
    indices) is stored. Names are generated from indices on demand, and
    indices are recovered by parsing names, so construction, copying and
    lookup do not scale with the number of electrodes (one per pixel: a
    576x720 RGBA image has 1.66 million).

    Behaves like a read-only 1-D array of strings: supports ``len``,
    iteration, indexing, slicing, boolean masking, ``reshape`` and ``ravel``.
    ``np.asarray`` builds the string array; this is the only operation whose
    cost scales with the number of electrodes.

    Parameters
    ----------
    grid_shape : tuple
        Shape of the electrode grid: ``(rows, cols)`` for a single-channel
        image, or ``(rows, cols, channels)`` for a multi-channel one.
    idx : array_like, optional
        Flat indices into the grid, selecting (and ordering) the names to
        expose. The array may have any shape; ``None`` means the whole grid in
        row-major order.
    unique : bool, optional
        Whether ``idx`` is known to be free of duplicates. ``None`` means
        unknown; :py:meth:`check_unique` then computes it.

    Examples
    --------
    >>> from pulse2percept.stimuli._grid_names import _GridNames
    >>> names = _GridNames((3, 4))
    >>> names[0], names[6]
    ('A1', 'B3')
    >>> names.index('B3')
    6

    """
    __slots__ = ('_grid_shape', '_idx', '_unique')

    def __init__(self, grid_shape, idx=None, unique=None):
        grid_shape = tuple(int(s) for s in grid_shape)
        if len(grid_shape) not in (2, 3):
            raise ValueError(f"'grid_shape' must be (rows, cols) or "
                             f"(rows, cols, channels), not {grid_shape}.")
        if any(s < 0 for s in grid_shape):
            raise ValueError(f"'grid_shape' must not be negative, got "
                             f"{grid_shape}.")
        self._grid_shape = grid_shape
        if idx is None:
            self._idx = None
            # The full grid has no duplicates:
            self._unique = True
        else:
            self._idx = np.asarray(idx, dtype=np.intp)
            self._unique = unique

    # -- Grid geometry --------------------------------------------------

    @property
    def grid_shape(self):
        """Shape of the underlying electrode grid"""
        return self._grid_shape

    @property
    def grid_size(self):
        """Total number of electrodes in the underlying grid"""
        return int(np.prod(self._grid_shape))

    @property
    def indices(self):
        """Flat indices into the grid, one per name"""
        if self._idx is None:
            return np.arange(self.grid_size, dtype=np.intp)
        return self._idx

    # -- Array-like interface -------------------------------------------

    @property
    def shape(self):
        """Shape of the name container"""
        if self._idx is None:
            return (self.grid_size,)
        return self._idx.shape

    @property
    def size(self):
        """Total number of names"""
        if self._idx is None:
            return self.grid_size
        return self._idx.size

    @property
    def ndim(self):
        """Number of dimensions of the name container"""
        return len(self.shape)

    @property
    def dtype(self):
        """Dtype the names would have if materialized"""
        return np.dtype(f'<U{self._max_name_len()}')

    @property
    def is_unique(self):
        """Whether the names are known to be free of duplicates

        ``False`` means "not known to be unique"; call
        :py:meth:`check_unique` to find out.
        """
        return bool(self._unique)

    def __len__(self):
        shape = self.shape
        if not shape:
            raise TypeError("len() of unsized _GridNames")
        return shape[0]

    def __getitem__(self, item):
        # Raise KeyError on names, so callers can fall back to `index` (as they
        # do on IndexError from a NumPy array):
        if isinstance(item, str):
            raise KeyError(item)
        idx = self.indices[item]
        if np.ndim(idx) == 0:
            return self._name_at(int(idx))
        # Fancy indexing may repeat names, so uniqueness is unknown (None):
        unique = True if (self._unique and _is_pure_selection(item)) else None
        return _GridNames(self._grid_shape, idx, unique=unique)

    def __iter__(self):
        # Generate names lazily, so early exits don't build the full array:
        for i in self.indices.ravel():
            yield self._name_at(int(i))

    def __contains__(self, name):
        try:
            self.index(name)
        except (ValueError, KeyError):
            return False
        return True

    def __array__(self, dtype=None, copy=None):
        names = self._materialize()
        if dtype is not None:
            names = names.astype(dtype)
        return names

    def __eq__(self, other):
        if isinstance(other, _GridNames):
            # Same grid: compare indices instead of strings:
            if self._grid_shape != other._grid_shape:
                return np.asarray(self) == np.asarray(other)
            if self._idx is None and other._idx is None:
                return np.ones(self.shape, dtype=bool)
            return self.indices == other.indices
        return np.asarray(self) == other

    def __ne__(self, other):
        result = self.__eq__(other)
        return np.logical_not(result)

    def __repr__(self):
        return (f"_GridNames(grid_shape={self._grid_shape}, "
                f"size={self.size})")

    def reshape(self, *shape):
        """Return a view of the names with a new shape"""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, np.ndarray)):
            shape = tuple(shape[0])
        return _GridNames(self._grid_shape, self.indices.reshape(shape),
                              unique=self._unique)

    def ravel(self):
        """Return a flattened view of the names"""
        if self._idx is None or self._idx.ndim == 1:
            return self
        return _GridNames(self._grid_shape, self._idx.ravel(),
                              unique=self._unique)

    def copy(self):
        """Return an independent copy"""
        idx = None if self._idx is None else self._idx.copy()
        return _GridNames(self._grid_shape, idx, unique=self._unique)

    def tolist(self):
        """Return the names as a list of strings"""
        return np.asarray(self).tolist()

    # -- Name <-> index mapping -----------------------------------------

    def index(self, name):
        """Return the position of ``name``

        Parses ``name`` instead of generating the names, so the cost does not
        depend on the number of electrodes.

        Parameters
        ----------
        name : str
            An electrode name, e.g. ``'C12'`` or ``'A1_R'``.

        Returns
        -------
        index : int
            Position of ``name`` in the (flattened) sequence of names.
        """
        flat = self._flat_index_of(name)
        if self._idx is None:
            return int(flat)
        # Subsets (e.g. a cropped image) require locating the grid index with
        # a vectorized integer scan:
        hits = np.flatnonzero(self._idx.ravel() == flat)
        if hits.size == 0:
            raise ValueError(f"'{name}' is not in the list of electrodes.")
        return int(hits[0])

    def check_unique(self):
        """Compute (and cache) whether the names are free of duplicates

        Grid names are unique, so this checks the indices instead of the
        names.

        Returns
        -------
        unique : bool
            True if no name occurs twice.
        """
        if self._unique is None:
            self._unique = bool(
                np.unique(self._idx).size == self._idx.size)
        return bool(self._unique)

    # -- Internals ------------------------------------------------------

    def _channel_labels(self):
        n_channels = self._grid_shape[2]
        labels = _CHANNEL_LABELS.get(n_channels,
                                     tuple(str(c) for c in range(n_channels)))
        return np.array([f'_{label}' for label in labels])

    def _row_labels(self):
        return np.array([bijective26_name(r)
                         for r in range(self._grid_shape[0])])

    def _col_labels(self):
        # Size the string dtype to the largest column number; NumPy's default
        # int-to-str ('<U21') would bloat the materialized array:
        n_cols = self._grid_shape[1]
        width = len(str(n_cols)) if n_cols else 1
        return (np.arange(n_cols) + 1).astype(f'<U{width}')

    def _max_name_len(self):
        if self.grid_size == 0:
            return 1
        length = (len(bijective26_name(self._grid_shape[0] - 1)) +
                  len(str(self._grid_shape[1])))
        if len(self._grid_shape) > 2:
            length += max(len(label) for label in self._channel_labels())
        return length

    def _name_at(self, flat):
        """Generate the name of a single grid index"""
        if flat < 0:
            flat += self.grid_size
        coords = np.unravel_index(flat, self._grid_shape)
        name = f"{bijective26_name(int(coords[0]))}{int(coords[1]) + 1}"
        if len(self._grid_shape) > 2:
            name += self._channel_labels()[int(coords[2])]
        return name

    def _flat_index_of(self, name):
        """Parse a name back into its flat index into the grid"""
        if not isinstance(name, str):
            raise KeyError(name)
        match = _NAME_RE.match(name)
        if match is None:
            raise ValueError(f"'{name}' is not a valid electrode name.")
        letters, digits, suffix = match.groups()
        row = _bijective26_index(letters)
        col = int(digits) - 1
        coords = [row, col]
        if len(self._grid_shape) > 2:
            if suffix is None:
                raise ValueError(f"'{name}' does not name a color channel, "
                                 f"but the electrode grid has "
                                 f"{self._grid_shape[2]} of them.")
            labels = [label[1:] for label in self._channel_labels()]
            try:
                coords.append(labels.index(suffix))
            except ValueError:
                raise ValueError(f"'{name}' names an unknown color channel "
                                 f"'{suffix}'.")
        elif suffix is not None:
            raise ValueError(f"'{name}' names a color channel, but the "
                             f"electrode grid does not have any.")
        if any(c < 0 or c >= s for c, s in zip(coords, self._grid_shape)):
            raise ValueError(f"'{name}' lies outside a {self._grid_shape} "
                             f"electrode grid.")
        return int(np.ravel_multi_index(tuple(coords), self._grid_shape))

    def _materialize(self):
        """Build the array of name strings (cost scales with grid size)"""
        idx = self.indices
        if idx.size == 0:
            return np.empty(idx.shape, dtype=self.dtype)
        coords = np.unravel_index(idx.ravel(), self._grid_shape)
        names = np.char.add(self._row_labels()[coords[0]],
                            self._col_labels()[coords[1]])
        if len(self._grid_shape) > 2:
            names = np.char.add(names, self._channel_labels()[coords[2]])
        return names.reshape(idx.shape)


def _names_equal(a, b):
    """Return True if two containers hold the same electrode names"""
    if isinstance(a, _GridNames) and isinstance(b, _GridNames):
        if a.grid_shape == b.grid_shape:
            return np.array_equal(a.indices, b.indices)
    return np.array_equal(np.asarray(a), np.asarray(b))


def _index_of_name(electrodes, name):
    """Return the position of electrode ``name`` in ``electrodes``"""
    if isinstance(electrodes, _GridNames):
        return electrodes.index(name)
    return list(electrodes).index(name)
