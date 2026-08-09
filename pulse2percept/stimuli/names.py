""":py:class:`~pulse2percept.stimuli.ElectrodeNames`"""
import re
import numpy as np

from ..utils.base import bijective26_name

__all__ = ['ElectrodeNames']

# Channel suffixes for the common color models. Anything else falls back to a
# numeric suffix, so that every channel remains addressable:
_CHANNEL_LABELS = {3: ('R', 'G', 'B'), 4: ('R', 'G', 'B', 'A')}

# 'A1', 'BC17', 'A1_R', 'A1_12' -- letters address the row, digits the column,
# and the optional suffix the color channel:
_NAME_RE = re.compile(r'^([A-Z]+)([0-9]+)(?:_([A-Z0-9]+))?$')


def _bijective26_index(letters):
    """Inverse of :py:func:`~pulse2percept.utils.bijective26_name`

    Translates an "alphabetic number" back into the integer it names, e.g.
    'A' -> 0, 'Z' -> 25, 'AA' -> 26.
    """
    value = 0
    for char in letters:
        value = value * 26 + (ord(char) - 64)
    return value - 1


def _is_pure_selection(item):
    """Whether an index expression can only ever select, never repeat

    Slices, ellipses and boolean masks visit every element at most once, so
    they preserve uniqueness of the names they select. Integer (fancy)
    indexing does not: ``names[[0, 0]]`` repeats an element. Uniqueness
    matters because :py:class:`~pulse2percept.stimuli.Stimulus` can skip its
    duplicate-name check whenever it is guaranteed by construction.
    """
    if item is Ellipsis or isinstance(item, slice):
        return True
    if isinstance(item, tuple):
        return all(_is_pure_selection(i) for i in item)
    if isinstance(item, np.ndarray):
        return item.dtype == bool
    return False


class ElectrodeNames:
    """Lazily generated electrode names for a grid of electrodes

    Names every element of a (rows x columns [x channels]) grid after its
    position in that grid: letters address the row, digits the column, and an
    optional suffix the color channel. The first pixel of an RGB image is
    therefore ``'A1_R'``, and the pixel in the third row and twelfth column of
    a grayscale image is ``'C12'``.

    The names are *not* stored. Only the shape of the grid is, plus (for a
    subset such as a cropped image) the indices that were kept. Both
    directions of the mapping are computed from that: a name is generated from
    its index on demand, and the index of a name is recovered by parsing it.
    That keeps construction, copying and lookup independent of the number of
    electrodes, which matters because an image or video stimulus assigns one
    electrode per pixel -- a 576x720 RGBA image has 1.66 million of them.

    An ``ElectrodeNames`` behaves like a read-only 1-D array of strings: it
    supports ``len``, iteration, indexing, slicing, boolean masking,
    ``reshape`` and ``ravel``, and converts to a NumPy array of strings via
    ``np.asarray``. That conversion is the one operation whose cost scales
    with the number of electrodes, so it is left to the caller to trigger.

    .. versionadded:: 0.10.0

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
        "not known", in which case :py:meth:`check_unique` will work it out.

    Examples
    --------
    >>> from pulse2percept.stimuli import ElectrodeNames
    >>> names = ElectrodeNames((3, 4))
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
            # The whole grid, in order, cannot contain duplicates:
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

        ``False`` means "not known to be unique", not "known to contain
        duplicates"; call :py:meth:`check_unique` to settle it.
        """
        return bool(self._unique)

    def __len__(self):
        shape = self.shape
        if not shape:
            raise TypeError("len() of unsized ElectrodeNames")
        return shape[0]

    def __getitem__(self, item):
        # A name is not a valid index. Raise KeyError so that callers which
        # accept either an index or a name can fall back to `index`, the same
        # way they do for a NumPy array (which raises IndexError):
        if isinstance(item, str):
            raise KeyError(item)
        idx = self.indices[item]
        if np.ndim(idx) == 0:
            return self._name_at(int(idx))
        # Uniqueness only ever carries over; it is never ruled out here. An
        # index expression that *may* repeat leaves it undetermined (None),
        # for `check_unique` to settle if anyone asks:
        unique = True if (self._unique and _is_pure_selection(item)) else None
        return ElectrodeNames(self._grid_shape, idx, unique=unique)

    def __iter__(self):
        # Generating names one at a time is slower per element than building
        # the whole array at once, but callers that break out early (or that
        # only ever look at a handful of electrodes) never pay for the rest:
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
        if isinstance(other, ElectrodeNames):
            # Two views of the same grid hold the same names iff they select
            # the same indices, which is far cheaper to check than the names:
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
        return (f"ElectrodeNames(grid_shape={self._grid_shape}, "
                f"size={self.size})")

    def reshape(self, *shape):
        """Return a view of the names with a new shape"""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, np.ndarray)):
            shape = tuple(shape[0])
        return ElectrodeNames(self._grid_shape, self.indices.reshape(shape),
                              unique=self._unique)

    def ravel(self):
        """Return a flattened view of the names"""
        if self._idx is None or self._idx.ndim == 1:
            return self
        return ElectrodeNames(self._grid_shape, self._idx.ravel(),
                              unique=self._unique)

    def copy(self):
        """Return an independent copy"""
        idx = None if self._idx is None else self._idx.copy()
        return ElectrodeNames(self._grid_shape, idx, unique=self._unique)

    def tolist(self):
        """Return the names as a list of strings"""
        return np.asarray(self).tolist()

    # -- Name <-> index mapping -----------------------------------------

    def index(self, name):
        """Return the position of ``name``

        Unlike ``list(names).index(name)``, this does not build (or even
        generate) the names: the position is recovered by parsing the name
        itself, which is why it costs the same for one electrode as for a
        million.

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
        # A subset (e.g. a cropped image) no longer has the grid's own
        # ordering, so the parsed grid index still has to be located. This is
        # a vectorized scan rather than a parse, but it touches integers
        # instead of strings and stays in C:
        hits = np.flatnonzero(self._idx.ravel() == flat)
        if hits.size == 0:
            raise ValueError(f"'{name}' is not in the list of electrodes.")
        return int(hits[0])

    def check_unique(self):
        """Determine (and remember) whether the names are free of duplicates

        The grid names are unique by construction, so duplicates can only come
        from a repeated index. Checking the indices is therefore equivalent to
        checking the names, and much cheaper.

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
        # Ask for exactly as many characters as the largest column number
        # needs. NumPy's own int-to-str conversion sizes for the widest
        # possible integer instead ('<U21'), which would make a materialized
        # name array several times larger than the names in it:
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
        """Build the actual array of name strings

        This is the only operation whose cost scales with the number of
        electrodes, so everything else is arranged to avoid it.
        """
        idx = self.indices
        if idx.size == 0:
            return np.empty(idx.shape, dtype=self.dtype)
        coords = np.unravel_index(idx.ravel(), self._grid_shape)
        names = np.char.add(self._row_labels()[coords[0]],
                            self._col_labels()[coords[1]])
        if len(self._grid_shape) > 2:
            names = np.char.add(names, self._channel_labels()[coords[2]])
        return names.reshape(idx.shape)
