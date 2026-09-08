""":py:class:`~pulse2percept.stimuli.ElectrodeNames`,
:py:class:`~pulse2percept.stimuli.ImageStimulus`,
:py:class:`~pulse2percept.stimuli.Stimulus`,
:py:class:`~pulse2percept.stimuli.VideoStimulus`

The fundamental stimulus containers: the
:py:class:`~pulse2percept.stimuli.Stimulus` data container, its
:py:class:`~pulse2percept.stimuli.ImageStimulus` and
:py:class:`~pulse2percept.stimuli.VideoStimulus` specializations for pixel
data, and the :py:class:`~pulse2percept.stimuli.ElectrodeNames` mapping that
names their electrodes. Other modules in :py:mod:`pulse2percept.stimuli`
construct particular content on top of these.
"""
import operator as ops
import os
import re
import warnings
from copy import copy, deepcopy
from math import isclose

import matplotlib.pyplot as plt
import numpy as np
from imageio import get_reader as video_reader
from scipy.integrate import trapezoid
from skimage import img_as_float32
from skimage.color import rgba2rgb, rgb2gray
from skimage.feature import canny
from skimage.filters import (threshold_mean, threshold_minimum, threshold_otsu,
                             threshold_local, threshold_isodata, scharr, sobel,
                             median)
from skimage.io import imread, imsave
from skimage.transform import resize as img_resize, rotate as img_rotate
# The video methods use their own aliases for the same two transforms:
from skimage.transform import resize as vid_resize, rotate as vid_rotate

from ._base import fast_compress_space, fast_compress_time
from ._merge import merge_time_axes
from ..units import (DimensionMismatchError, Quantity, Unit, as_value, deg,
                     dimensionless, ms, uA)
from ..units.base import has_units
from ..utils import (PrettyPrint, center_image, frame_interval, HTMLAnimation,
                     is_strictly_increasing, scale_image, shift_image,
                     trim_image)
from ..utils.array import _interp_rows, _slice_times
from ..utils.base import bijective26_name
from ..utils.constants import DT, MIN_AMP, MS_PER_S
from ..utils.images import _as_writable

__all__ = [
    'ElectrodeNames',
    'ImageStimulus',
    'Stimulus',
    'VideoStimulus',
]

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


def _as_scalar_column(source):
    """Convert a flat sequence of scalars into an (N, 1) data container"""
    if not isinstance(source, (list, tuple)) or not source:
        return None
    if not np.isscalar(source[0]) or isinstance(source[0], str):
        return None
    try:
        flat = np.asarray(source)
    except (TypeError, ValueError):
        return None
    # Strings, None and complex values all infer to a non-numeric dtype and so
    # fall through:
    if flat.ndim != 1 or flat.dtype.kind not in 'biuf':
        return None
    return flat.astype(np.float32).reshape((-1, 1))


def _names_equal(a, b):
    """Whether two containers hold the same electrode names"""
    if isinstance(a, ElectrodeNames) and isinstance(b, ElectrodeNames):
        if a.grid_shape == b.grid_shape:
            return np.array_equal(a.indices, b.indices)
    return np.array_equal(np.asarray(a), np.asarray(b))


def _index_of_name(electrodes, name):
    """Return the position of electrode ``name`` in ``electrodes``"""
    if isinstance(electrodes, ElectrodeNames):
        return electrodes.index(name)
    return list(electrodes).index(name)


class _AdoptableArray(np.ndarray):
    """Internal marker for an array that may be installed without copying

    Views keep the subclass, so the mark survives the reshaping on the way in.
    """
    __slots__ = ()


def _adoptable(arr):
    """Mark arr as safe for a stimulus to install without copying"""
    return arr.view(_AdoptableArray)


def _describe_unit(unit):
    """Name a unit the way an error message wants to read"""
    if unit.dimension.is_dimensionless:
        return 'dimensionless units'
    return f'{unit.dimension.name} ({unit})'


def _stimulus_sources(source):
    """The Stimulus objects a source is built from, if any"""
    if isinstance(source, Stimulus):
        return [source]
    if isinstance(source, dict):
        return [s for s in source.values() if isinstance(s, Stimulus)]
    if isinstance(source, (list, tuple)):
        return [s for s in source if isinstance(s, Stimulus)]
    return []


def _has_waveform(stim):
    """Whether a stimulus has already generated the samples it describes"""
    return stim._Stimulus__stim['data'] is not None


def _snapshot(source):
    """One entry of a collection, as it was when the collection was built"""
    if np.isscalar(source):
        return source
    return deepcopy(source)


def _component_shape(source):
    """What one entry of a collection contributes, without sampling it"""
    if isinstance(source, Stimulus):
        has_time = not _has_waveform(source) or source.time is not None
        return source.electrodes, len(source.electrodes), has_time
    return None, 1, not (np.isscalar(source) and not isinstance(source, str))


def _strip_units(source, unit):
    """Convert a source's quantities into plain numbers expressed in unit"""
    if isinstance(source, (Quantity, Unit)):
        return as_value(source, unit, 'source')
    if isinstance(source, dict):
        if any(has_units(v) for v in source.values()):
            return {k: _strip_units(v, unit) for k, v in source.items()}
    elif isinstance(source, (list, tuple)):
        if any(has_units(v) for v in source):
            return type(source)(_strip_units(v, unit) for v in source)
    return source


def _scale_factor(op, scalar, reverse=False):
    """The factor by which an arithmetic operator scales the stimulus data"""
    if op is ops.mul:
        factor = scalar
    elif op is ops.truediv:
        # `Stimulus` has no `__rtruediv__`, so this is always data/scalar.
        with np.errstate(divide='ignore', invalid='ignore'):
            factor = np.divide(1.0, scalar)
    elif scalar == 0:
        # `stim + 0` and `stim - 0` change nothing; `0 - stim` flips the sign:
        factor = -1.0 if reverse else 1.0
    else:
        return None
    return factor if np.isfinite(factor) else None


class Stimulus(PrettyPrint):
    """Stimulus

    A stimulus is comprised of a labeled 2D NumPy array that contains the data,
    where the rows denote electrodes and the columns denote points in time.
    A stimulus can be created from a variety of source types (e.g., scalars,
    lists, NumPy arrays, and dictionaries).

    The stimulus arrays (``data``, ``time``, ``electrodes``) and the pulse
    parameters of a stimulus that has any are read-only. Arbitrary stimuli are
    kept as sampled waveforms only.

    .. seealso ::

        *  `Basic Concepts > Electrical Stimuli <topics-stimuli>`

    .. versionadded:: 0.6

    .. versionchanged:: 0.10.0
        Stimulus arrays and pulse parameters are read-only, and waveforms
        are generated lazily.

    Parameters
    ----------
    source : source type
        A valid source type is one of the following:

        * Scalar value: interpreted as the current amplitude delivered to a
          single electrode (no time component).
        * NumPy array:
           * Nx1 array: interpreted as N current amplitudes delivered to N
             electrodes (no time component).
           * NxM array: interpreted as N electrodes each receiving M current
             amplitudes in time.

        In addition, you can also pass a collection of source types.
        Each element must be a valid source type for a single electrode (e.g.,
        scalar, 1-D array, :py:class:`~pulse2percept.stimuli.Stimulus`).

        * List or tuple: List elements will be assigned to electrodes in order.
        * Dictionary: Dictionary keys are used to address electrodes by name.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. If none are
        given, electrode names will be extracted from the source type (e.g.,
        the keys from a dictionary). If a scalar or NumPy array is passed,
        electrode names will be numbered 0..N.

        .. note::

           The number of electrode names provided must match the number of
           electrodes extracted from the source type (i.e., N).

    time : int, float or list thereof; optional
        Optionally, you can provide the time points of the source data.
        If none are given, time steps will be numbered 0..M.

        .. note::

           The number of time points provided must match the number of time
           points extracted from the source type (i.e., M).
           Stimuli created from scalars or 1-D NumPy arrays will have no time
           componenet, in which case you cannot provide your own time points.

    metadata : dict, optional
        Additional stimulus metadata can be stored in a dictionary.

    compress : bool, optional
        If True, will compress the source data in two ways:

        * Remove electrodes with all-zero activation.
        * Retain only the time points at which the stimulus changes.

        For example, in a pulse train, only the signal edges are saved. This
        drastically reduces the memory footprint of the stimulus.

    Notes
    -----
    *  Depending on the source type, a stimulus might have a time component or
       not (e.g., scalars: time=None).
    *  You can access the stimulus applied to electrode ``e`` at time ``t``
       by directly indexing into ``Stimulus[e, t]``. In this case, ``t`` is not
       a column index but a time point.
    *  If the time point is not explicitly stored in the ``data`` container,
       its value will be automatically interpolated from neighboring values.
    *  If a requested time point lies outside the range of stored data,
       the value of its closest end point will be returned.
    *  All transformations return a new stimulus, except for
       :py:meth:`compress` and :py:meth:`remove`.

    Examples
    --------
    Stimulate a single electrode with -13uA:

    >>> from pulse2percept.stimuli import Stimulus
    >>> stim = Stimulus(-13)

    Stimulate ten electrodes with 0uA:

    >>> from pulse2percept.stimuli import Stimulus
    >>> stim = Stimulus(np.zeros(10))

    Provide new electrode names for an existing Stimulus object:

    >>> from pulse2percept.stimuli import Stimulus
    >>> old_stim = Stimulus([3, 5])
    >>> new_stim = Stimulus(old_stim, electrodes=['new0', 'new1'])

    Interpolate the stimulus value at some point in time. Here, the stimulus
    is a single-electrode ramp stimulus (stimulus value == point in time):

    >>> from pulse2percept.stimuli import Stimulus
    >>> stim = Stimulus(np.arange(10).reshape((1, -1)))
    >>> stim[:, 3.45] # doctest: +ELLIPSIS
    3.45...

    """
    # Frozen class: Only the following class attributes are allowed
    __slots__ = ('metadata', '_is_compressed', '__stim', '_unit',
                 '_time_unit', '_components')

    # data is stored in microamps and millisecond
    _default_unit = uA
    _default_time_unit = ms

    #: Whether this stimulus provides a separate spatial-only view
    _has_spatial_view = False

    #: Whether dimensionless values represent encoded normalized drive.
    _is_normalized_drive = False

    #: whether the canonical state is a set of stim params
    _is_parametric = False

    def __init__(self, source, electrodes=None, time=None, metadata=None,
                 compress=False):
        self.metadata = self._wrap_metadata(metadata)
        # Flag will be flipped in the compress method:
        self._is_compressed = False
        # Set by `_factory` when this is a collection whose entries have not
        # been merged into a waveform yet (see `_render`):
        self._components = None
        self._unit, self._time_unit = self._resolve_units(source)
        source = _strip_units(source, self._unit)
        time = as_value(time, self._time_unit, 'time')
        # Extract the data and coordinates (electrodes, time) from the source:
        self._factory(source, electrodes, time, compress)

    @staticmethod
    def _wrap_metadata(metadata):
        """File the caller's metadata under ``user``"""
        return {'user': metadata}

    def _inherit_metadata(self, other):
        """Take on another stimulus' metadata dict, as it stands"""
        self.metadata = other.metadata
        return self

    def _defer(self, electrodes, unit=None, time_unit=None, metadata=None):
        """Set this stimulus up to generate its waveform later"""
        self.metadata = self._wrap_metadata(metadata)
        self._is_compressed = False
        self._components = None
        self._unit = self._default_unit if unit is None else unit
        self._time_unit = (self._default_time_unit if time_unit is None
                           else time_unit)
        if not isinstance(electrodes, ElectrodeNames):
            electrodes = np.array([electrodes]).ravel()
        # `data=None` is what says the waveform has not been generated yet
        self.__stim = {'data': None, 'time': None,
                       'electrodes': self._own_names(electrodes)}

    def _forget_waveform(self, electrodes):
        """Drop a cached waveform the components no longer describe"""
        self.__stim = {'data': None, 'time': None,
                       'electrodes': self._own_names(electrodes)}

    def _render(self):
        """Generate the waveform this stimulus describes

        A subclass that called :py:meth:`_defer` overrides this and returns
        the state to install::

            {'data': ..., 'electrodes': ..., 'time': ...}

        It runs at most once, and what it returns goes through the ``_stim``
        setter like any other state, so the waveform it built is owned,
        immutable and validated on the same terms as one that was passed in.
        """
        if self._components is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no stimulus data, and does not "
                f"know how to generate any. A subclass that defers its "
                f"waveform must override '_render'.")
        _data, _time = [], []
        for src, _ in self._components:
            d, t, _e = self._parse_source(src, nested=True)
            _data.append(d)
            _time.append(t)
        _data, _time = self._merge_sources(_data, _time)
        return {'data': _data, 'electrodes': self.electrodes, 'time': _time}

    def _resolve_units(self, source):
        """Determine the units this stimulus stores its data and time in"""
        unit, time_unit = self._default_unit, self._default_time_unit
        sources = _stimulus_sources(source)
        if not sources:
            return unit, time_unit
        for attr, expected in (('unit', unit), ('time_unit', time_unit)):
            found = {getattr(s, attr) for s in sources}
            if len(found) > 1:
                names = ', '.join(sorted(_describe_unit(u) for u in found))
                raise DimensionMismatchError(
                    f"Cannot build one {type(self).__name__} out of stimuli "
                    f"with different units ({names}). Convert them to a "
                    f"common unit first.")
            if attr == 'unit':
                unit = found.pop()
            else:
                time_unit = found.pop()
        return unit, time_unit

    def _inherit_units(self, other):
        """Adopt the units of another stimulus"""
        self._unit = other.unit
        self._time_unit = other.time_unit
        return self

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        return {'data': self.data, 'electrodes': self.electrodes,
                'time': self.time, 'shape': self.shape, 'dt': self.dt,
                'is_charge_balanced': self.is_charge_balanced,
                'metadata': self.metadata}

    @staticmethod
    def _defers_waveform(source, electrodes, time, compress):
        """Whether to keep this source's entries instead of merging them now

        Worth doing when an entry is a stimulus that is defined by its
        stimulation parameters, or that has not generated a waveform yet
        """
        if time is not None or compress:
            return False
        return any(s._is_parametric or not _has_waveform(s)
                   for s in _stimulus_sources(source))

    @staticmethod
    def _require_one_time_convention(no_time):
        """Every entry of a collection has a time axis, or none of them does"""
        if len(np.unique(no_time)) > 1:
            raise ValueError("If one stimulus has time=None, all others "
                             "must have time=None as well.")

    @classmethod
    def _merge_sources(cls, _data, _time):
        """Stack the entries of a collection onto one common time axis"""
        cls._require_one_time_convention([t is None for t in _time])
        # When none of the stimuli have time=None, we need to merge the
        # time axes (this is expensive because of interpolation):
        if len(_time) > 1 and _time[0] is not None:
            _data, _time = merge_time_axes(_data, _time)
        # Now make `_data` a 2-D NumPy array, with `_electrodes` as rows
        # and `_time` as columns (except sometimes `_time` is None).
        return (np.vstack(_data) if _data else np.array([]),
                _time[0] if _time else None)

    def _parse_source(self, source, nested=False):
        """Extract data, time and electrode names from a single source

        This private method converts input data from allowable source types
        into a 2-D NumPy array, where the first dimension denotes electrodes
        and the second dimension denotes points in time.

        The same source is read in one of two ways, depending on where it
        appears:

        * At the top level, a flat sequence of N values means N electrodes
          stimulated once each, with no time component.
        * As an element of a collection (a list entry or a dict value), that
          same sequence means a *single* electrode sampled at N points in
          time.

        ``nested`` selects between the two readings. Only a collection can
        contain a nested source, so only a collection passes ``nested=True``.
        """
        if isinstance(source, Stimulus):
            # e.g. a Stimulus being renamed, or a dict of Stimulus objects
            return source.data, source.time, source.electrodes
        if np.isscalar(source) and not isinstance(source, str):
            # Scalar: 1 electrode, no time component - either way round
            return np.array([source], dtype=np.float32).reshape((1, -1)), \
                None, None
        if isinstance(source, np.ndarray):
            if nested:
                if source.ndim > 1:
                    raise ValueError(f"Cannot create Stimulus object from a "
                                     f"{source.ndim}-D NumPy array. Must be "
                                     f"1-D.")
                # 1-D NumPy array with N elements: 1 electrode, N time points
                data = source.astype(np.float32).reshape((1, -1))
                return data, np.arange(data.shape[-1], dtype=np.float32), None
            if source.ndim == 1:
                # N electrodes, no time component
                return source.reshape((-1, 1)), None, None
            if source.ndim == 2:
                # N electrodes x M time points
                return source, np.arange(source.shape[-1],
                                         dtype=np.float32), None
            raise ValueError(f"Cannot create Stimulus object from a "
                             f"{source.ndim}-D NumPy array. Must be < 2-D.")
        if nested and isinstance(source, (list, tuple)):
            # List or tuple with N elements: 1 electrode, N time points.
            # At the top level these are collections, handled by `_factory`:
            data = np.array(source, dtype=np.float32).reshape((1, -1))
            return data, np.arange(data.shape[-1], dtype=np.float32), None
        raise TypeError(f"Cannot create Stimulus object from {type(source)}. Choose "
                        f"from: scalar, tuple, list, NumPy array, or "
                        f"Stimulus.")

    def _factory(self, source, electrodes, time, compress):
        """Build the Stimulus object from the specified source type"""
        # Whether we numbered the electrodes ourselves (0..N-1):
        _auto_electrodes = False
        if (_flat := _as_scalar_column(source)) is not None:
            # one electrode per element, no time component
            _data, _time, _electrodes = _flat, None, None
            _n_rows = _data.shape[0]
        elif isinstance(source, (dict, list, tuple)):
            # A collection: every entry is itself a source
            if isinstance(source, dict):
                iterator = source.items()
            else:
                iterator = enumerate(source)
            if self._defers_waveform(source, electrodes, time, compress):
                self._components = []
            _time = []
            _electrodes = []
            _data = []
            _no_time = []
            for ele, src in iterator:
                if self._components is None:
                    d, t, e = self._parse_source(src, nested=True)
                    _time.append(t)
                    _data.append(d)
                else:
                    # Nothing is sampled yet:
                    src = _snapshot(src)
                    e, n_rows, has_time = _component_shape(src)
                    self._components.append((src, n_rows))
                    _no_time.append(not has_time)
                if isinstance(source, dict):
                    # Special case, electrode names are specified in a dict:
                    _electrodes.append(ele)
                else:
                    # In all other cases, use the electrode names specified by
                    # the source (unless they're None):
                    _electrodes.append(e if e is not None else ele)
            if self._components is None:
                _data, _time = self._merge_sources(_data, _time)
                _n_rows = _data.shape[0]
            else:
                # Asked here as well as in `_merge_sources`:
                self._require_one_time_convention(_no_time)
                _n_rows = sum(n for _, n in self._components)
        else:
            # A single source: a scalar, a NumPy array, or a Stimulus
            if self._defers_waveform(source, electrodes, time, compress):
                # Renaming or re-wrapping a stimulus that has not generated
                # its waveform must not be what generates it:
                snapshot = _snapshot(source)
                _electrodes, _n_rows, _ = _component_shape(snapshot)
                self._components = [(snapshot, _n_rows)]
                _data, _time = None, None
            else:
                _data, _time, _electrodes = self._parse_source(source)
                _n_rows = _data.shape[0]
            if isinstance(source, Stimulus):
                # Re-wrapping or renaming a stimulus keeps the metadata it
                # came with:
                self._inherit_metadata(source)

        if _electrodes is None:
            # The source did not name its electrodes, so they are 0..N-1 --
            # unique by construction. Only build that array if something will
            # read it
            _auto_electrodes = True
            if electrodes is None:
                _electrodes = np.arange(_n_rows)

        # User can overwrite the names of the electrodes:
        if electrodes is not None:
            if isinstance(electrodes, ElectrodeNames):
                # Names generated from a grid pattern:
                _electrodes = electrodes.ravel()
                _auto_electrodes = _electrodes.check_unique()
            else:
                _electrodes = np.array([electrodes]).flatten()
                _auto_electrodes = False
        else:
            if isinstance(_electrodes, ElectrodeNames):
                # The source brought its own generated names along:
                _electrodes = _electrodes.ravel()
                _auto_electrodes = _electrodes.check_unique()
            elif not isinstance(_electrodes, np.ndarray):
                # Could be a list of NumPy arrays, need to flatten:
                try:
                    _electrodes = np.concatenate(_electrodes)
                except ValueError:
                    _electrodes = np.array(_electrodes)
        if len(_electrodes) != _n_rows:
            raise ValueError(f"Number of electrodes provided ({len(_electrodes)}) does "
                             f"not match the number of electrodes in the data "
                             f"({_n_rows}).")
        # Electrodes we numbered ourselves are 0..N-1 and therefore unique by
        # construction, so the sort that np.unique performs can be skipped:
        if not _auto_electrodes:
            if isinstance(_electrodes, ElectrodeNames):
                _electrodes = np.asarray(_electrodes)
            unq, nunq = np.unique(_electrodes, return_index=True)
            if len(unq) != _n_rows:
                # We found duplicate names: replace them by integer index
                idx = np.delete(np.arange(len(_electrodes)), nunq)
                msg = (f"Duplicate electrode names detected "
                       f"{_electrodes[idx]}, and replaced with integer values")
                warnings.warn(msg)
                if _electrodes.dtype.kind in 'US':
                    # A fixed-width string array may be too narrow to hold the
                    # integer replacements, which would truncate them silently
                    # (and could even reintroduce duplicates), so widen first:
                    n_digits = len(str(len(_electrodes) - 1))
                    _electrodes = _electrodes.astype(
                        np.result_type(_electrodes.dtype, f'U{n_digits}'))
                _electrodes[idx] = idx

        # User can overwrite time:
        if time is not None:
            if _time is None:
                raise ValueError(f"Cannot set times={time}, because stimulus does "
                                 f"not have a time component.")
            time = np.array(time).flatten()
            if len(time) != _data.shape[1]:
                raise ValueError(f"Number of time steps provided ({len(time)}) does not "
                                 f"match the number of time steps in the data "
                                 f"({_data.shape[1]}).")
            _time = time

        if self._components is not None:
            self.__stim = {'data': None, 'time': None,
                           'electrodes': self._own_names(_electrodes)}
            return
        self._stim = {
            'data': _data,
            'electrodes': _electrodes,
            'time': _time,
        }
        if compress:
            self.compress()

    def _shallow_copy(self):
        """Copy the object without duplicating the data container"""
        stim = copy(self)
        stim.metadata = deepcopy(self.metadata)
        return stim

    def _waveform_copy(self):
        """This stimulus' waveform, as an ordinary ``Stimulus``"""
        stim = Stimulus(self.data, electrodes=self.electrodes, time=self.time)
        stim.metadata = deepcopy(self.metadata)
        return stim._inherit_units(self)

    def _spatial_view(self):
        """This stimulus as a reader with no clock of its own can read it"""
        return self

    def _without_electrodes(self, electrodes):
        """A copy of this stimulus that no longer drives ``electrodes``"""
        stim = self._derived()
        stim.remove(electrodes)
        return stim

    def _derived(self):
        """The object a waveform-rewriting operation builds its result on"""
        if self._is_parametric:
            return self._waveform_copy()
        return self._shallow_copy()

    def __deepcopy__(self, memo):
        """A copy that shares the data container with the original"""
        stim = copy(self)
        memo[id(self)] = stim
        stim.metadata = deepcopy(self.metadata, memo)
        return stim

    def compress(self):
        """Compress the source data in place"""
        data = self.data
        electrodes = self.electrodes
        time = self.time
        keep_el = fast_compress_space(data)
        data = data[keep_el]
        electrodes = electrodes[keep_el]

        if time is not None:
            idx_time = fast_compress_time(data)
            data = data[:, idx_time]
            time = time[idx_time]

        self._stim = {
            'data': data,
            'electrodes': electrodes,
            'time': time,
        }
        self._is_compressed = True

    def append(self, other):
        """Append another stimulus

        This method appends another stimulus (with matching electrodes) in
        time. The combined stimulus duration will be the sum of the two
        individual stimuli.

        .. versionadded:: 0.7

        Parameters
        ----------
        other : :py:class:`~pulse2percept.stimuli.Stimulus`
            Another stimulus with matching electrodes.

        Returns
        -------
        comb : :py:class:`~pulse2percept.stimuli.Stimulus`
            A combined stimulus with the same number of electrodes and new
            stimulus duration equal to the sum of the two individual stimuli.

        """
        if not isinstance(other, Stimulus):
            raise TypeError(f"Other object must be a Stimulus, not "
                            f"{type(other)}.")
        # The result is a copy of `self` with `other`'s data concatenated onto
        # its own:
        if self.unit != other.unit:
            raise DimensionMismatchError(
                f"Cannot append a stimulus measured in "
                f"{_describe_unit(other.unit)} to one measured in "
                f"{_describe_unit(self.unit)}.")
        if self.time_unit != other.time_unit:
            raise DimensionMismatchError(
                f"Cannot append a stimulus whose time is measured in "
                f"{_describe_unit(other.time_unit)} to one whose time is "
                f"measured in {_describe_unit(self.time_unit)}.")
        if not _has_time_axis(self) or not _has_time_axis(other):
            raise ValueError("Cannot append another stimulus if time=None.")
        if not _names_equal(self.electrodes, other.electrodes):
            raise ValueError("Both stimuli must have the same electrodes.")
        if other.time[0] < 0:
            raise NotImplementedError("Appending a stimulus with a negative "
                                      "time axis is currently not supported.")
        # Last time point of `self` can be merged with first point of `other`
        # but only if they have the same amplitude(s):
        if isclose(other.time[0], 0, abs_tol=DT) and \
                not np.allclose(other.data[:, 0], self._end_column()):
            err_str = (f"Data mismatch: Cannot append other stimulus "
                       f"because other[t=0] != this[t={self.time[-1]}ms]. You may need "
                       f"to shift the other stimulus in time by at least "
                       f"{DT:.1e} ms.")
            raise ValueError(err_str)
        return self._append_waveform(other)

    def _end_column(self):
        """The last column of the waveform"""
        return self.data[:, -1]

    def _append_waveform(self, other):
        """Lay ``other``'s samples after this stimulus' own"""
        stim = self._derived()
        if isclose(other.time[0], 0, abs_tol=DT):
            # The shared endpoint is written once:
            time = np.hstack((self.time, other.time[1:] + self.time[-1]))
            data = np.hstack((self.data, other.data[:, 1:]))
        else:
            time = np.hstack((self.time, other.time + self.time[-1]))
            data = np.hstack((self.data, other.data))
        stim._stim = {'data': data,
                      'electrodes': self.electrodes,
                      'time': time}
        return stim

    def remove(self, electrodes):
        """Remove electrode(s)

        Removes the stimulus of a certain electrode or list of electrodes.

        .. versionadded:: 0.8

        Parameters
        ----------
        electrodes : int, string, or list of int/str
            The item(s) to remove from the stimulus. Can either be an electrode
            index, electrode name, or a list thereof.
        """
        if electrodes is None or np.size(electrodes) == 0:
            return  # nothing to remove
        if self._is_parametric:
            raise NotImplementedError(
                f"Cannot remove electrodes from a {type(self).__name__}, "
                f"which is defined by the pulse it delivers rather than by "
                f"its samples -- what was left would go on advertising that "
                f"pulse. Take the waveform first: "
                f"Stimulus(stim).remove(...).")
        if np.isscalar(electrodes) and electrodes == 'all':
            gone = np.zeros(len(self.electrodes), dtype=bool)
            if self._drop_components(gone):
                return
            self._stim = {
                'data': self.data[[]],
                # Keep `electrodes` an array (of the same dtype) so that it can
                # still be indexed with a boolean mask afterwards:
                'electrodes': self.electrodes[[]],
                'time': self.time
            }
            return
        keep_el = self._keep_mask(electrodes)
        if self._drop_components(keep_el):
            return
        self._stim = {
            'data': self.data[keep_el],
            'electrodes': self.electrodes[keep_el],
            'time': self.time,
        }

    def _keep_mask(self, electrodes):
        """Which rows survive removing ``electrodes``"""
        # Start with a list of True and set the removed electrodes to False:
        keep_el = np.ones(len(self.electrodes), dtype=bool)
        if np.isscalar(electrodes) and electrodes == 'all':
            keep_el[:] = False
            return keep_el
        for electrode in np.array([electrodes]).ravel():
            try:
                # Check if `electrode` is an index into the electrodes array:
                self.electrodes[electrode]
                keep_el[electrode] = False
            except (IndexError, KeyError):
                # Another possibility is that a string with the electrode name
                # was passed. In this case, find the corresponding list index:
                try:
                    keep_el[_index_of_name(self.electrodes, electrode)] = False
                except ValueError:
                    raise ValueError(f'Electrode "{electrode}" not found.')
        return keep_el

    def _drop_components(self, keep_el):
        """Forget whole entries of an unmerged collection"""
        if self._components is None or not keep_el.any():
            return False
        kept, start = [], 0
        for component in self._components:
            rows = keep_el[start:start + component[1]]
            start += component[1]
            if rows.all():
                kept.append(component)
            elif rows.any():
                return False
        self._components = kept
        self._forget_waveform(self.electrodes[keep_el])
        return True

    def _structured_sources(self):
        """Return ``(electrode, source)`` pairs for retained structured sources.

        Returns ``None`` if the stimulus is waveform-only or cannot be mapped
        one source per electrode.
        """
        if self._components is not None:
            if any(n_rows != 1 or not isinstance(src, Stimulus)
                   for src, n_rows in self._components):
                return None
            return [(name, src) for name, (src, _)
                    in zip(self.electrodes, self._components)]
        if self._is_parametric and len(self.electrodes) == 1:
            # The stimulus is the source:
            return [(self.electrodes[0], self)]
        return None

    def shift(self, dt):
        """Shift the stimulus in time.

        .. versionadded:: 0.10.0

        Parameters
        ----------
        dt : float or :py:class:`~pulse2percept.units.Quantity`
            Time shift. May be positive or negative. Bare values are interpreted
            in the stimulus' time unit.

        Returns
        -------
        shifted : :py:class:`~pulse2percept.stimuli.Stimulus`
            Shifted copy of the stimulus.

        Notes
        -----
        ``stim >> dt`` and ``stim << dt`` are shorthand for ``stim.shift(dt)``
        and ``stim.shift(-dt)``.
        """
        if self.time is None:
            raise ValueError("Cannot shift a stimulus in time if time=None.")
        return self._apply_operator(self.time, ops.add, self._as_time(dt),
                                    field='time')

    def pad(self, duration):
        """Pad the stimulus with zeros up to a given time.

        Adds zero-valued endpoints at ``t=0`` and ``t=duration`` as needed.
        ``duration`` specifies the final time, not the amount of padding to add.

        .. versionadded:: 0.10.0

        Parameters
        ----------
        duration : float or :py:class:`~pulse2percept.units.Quantity`
            Final time of the padded stimulus. Bare values are interpreted in the
            stimulus' time unit.

        Returns
        -------
        padded : :py:class:`~pulse2percept.stimuli.Stimulus`
            Padded copy of the stimulus.

        Notes
        -----
        Padding requires existing boundary values to be zero; otherwise
        interpolation would create a ramp. It never truncates the stimulus or
        removes negative time points.
        """
        if self.time is None:
            raise ValueError("Cannot pad a stimulus in time if time=None.")
        duration = self._as_time(duration)
        if duration < self.time[-1]:
            raise ValueError(
                f"Cannot pad stimulus ending at {self.time[-1]} to {duration}."
            )
        data = self.data
        time = self.time
        pad_left = time[0] > 0
        pad_right = duration > time[-1]
        # Padding a nonzero endpoint would create a ramp under interpolation:
        if pad_left and np.any(data[:, 0] != 0):
            raise ValueError(f"Cannot pad before a nonzero stimulus endpoint "
                             f"(t={time[0]}).")
        if pad_right and np.any(data[:, -1] != 0):
            raise ValueError(f"Cannot pad after a nonzero stimulus endpoint "
                             f"(t={time[-1]}).")
        zeros = np.zeros((data.shape[0], 1), dtype=data.dtype)
        if pad_left:
            data = np.hstack((zeros, data))
            time = np.hstack(([0], time))
        if pad_right:
            data = np.hstack((data, zeros))
            time = np.hstack((time, [duration]))
        if not pad_left and not pad_right:
            # hstack allocates; the no-op path must copy explicitly
            data = data.copy()
            time = time.copy()
        stim = self._derived()
        stim._stim = {'data': data,
                      'electrodes': self.electrodes.copy(),
                      'time': time}
        return stim

    def plot(self, electrodes=None, time=None, fmt='k-', ax=None, kind=None):
        """Plot the stimulus

        .. versionadded:: 0.7

        .. versionchanged:: 0.10.0
            Added ``kind``: a whole multi-electrode stimulus is now drawn as
            an electrode-by-time heatmap rather than as one subplot per
            electrode.

        Parameters
        ----------
        electrodes : int, string, or list thereof; optional, default: None
            The electrodes for which to plot the stimulus. If None, all
            electrodes are plotted.
        time : (t_min, t_max) tuple, slice, or list of exact time points
            The time points at which to plot the stimulus. Specify a range of
            time points with a tuple or a slice, or specify the exact time
            points to interpolate.
            If None, all time points are plotted.
        fmt : str, optional, default: 'k-'
            A Matplotlib format string; e.g., 'ro' for red circles to use
            when ``kind='traces'``.
        ax : matplotlib.axes.Axes or list thereof; optional, default: None
            A Matplotlib Axes object. ``kind='traces'`` also accepts a list
            thereof (one per electrode to plot); ``kind='heatmap'`` draws into
            a single Axes. If None, a new Axes object will be created.
        kind : {'traces', 'heatmap'}, optional, default: None
            What to draw:

            *  'traces': the waveform of each electrode, one Axes per
               electrode. Good for a handful of electrodes.
            *  'heatmap': an electrode-by-time image in a single Axes. Good
               for a whole implant's worth of electrodes.

            If None, a whole stimulus of more than one electrode is drawn as a
            heatmap and everything else as traces.

        Returns
        -------
        ax : matplotlib.axes.Axes or np.ndarray of them
            One Axes per electrode for ``kind='traces'``, a single Axes for
            ``kind='heatmap'``.

        """
        # Imported here so that a stimulus does not depend on Matplotlib:
        from ._plot import plot_stimulus
        return plot_stimulus(self, electrodes, time, fmt, ax, kind)

    def __getitem__(self, item):
        """Returns an item from the data array, interpolated if necessary

        There are many potential use cases:

        *  ``stim[i]`` or ``stim[i, :]``: access electrode ``i`` (int or str)
        *  ``stim[[i0,i1]]`` or ``stim[[i0, i1], :]``
        *  ``stim[stim.electrodes != 'A1', :]``
        *  ``stim[:, 1]``: always interpreted as t=1.0, not index=1
        *  ``stim[:, 1.234]``: interpolated time
        *  ``stim[:, stim.time < 0.4]``, ``stim[:, 0.3:1.9:0.001]``
        """
        # STEP 1: AVOID CONFUSING TIME POINTS WITH COLUMN INDICES
        # NumPy handles most indexing and slicing. However, we need to prevent
        # cases like stim[:, [0, 1]] which ask for time=[0.0, 1.0] and not for
        # column index 0 and 1:
        if isinstance(item, tuple):
            electrodes = item[0]
            time = item[1]
            if isinstance(time, slice):
                sliced = self._slice_times(time)
                if sliced is not None:
                    time = sliced
            elif time is not Ellipsis:
                time = self._as_time(time)
                # Convert to float so time is not mistaken for column index
                if np.array(time).dtype != bool:
                    time = np.float64(time)
        else:
            electrodes = item
            time = None

        # STEP 2: ELECTRODES COULD BE SPECIFIED AS INT OR STR
        if isinstance(electrodes, (list, np.ndarray)) or np.isscalar(electrodes):
            parsed_electrodes = []
            for e in np.array([electrodes]).ravel():
                if isinstance(e, str):
                    parsed_electrodes.append(_index_of_name(self.electrodes, e))
                else:
                    # Most likely an integer index:
                    parsed_electrodes.append(e)
            if not isinstance(electrodes, (list, np.ndarray)):
                # If a scalar was passed, return a scalar:
                electrodes = parsed_electrodes[0]
            else:
                # Otherwise return an array:
                electrodes = np.array(parsed_electrodes)
        try:
            self._stim['data'][electrodes]
        except IndexError:
            raise IndexError("Invalid electrode index", electrodes)

        # STEP 2: NUMPY HANDLES MOST INDEXING AND SLICING:
        # Rebuild original index from ``electrodes`` and ``time``:
        if time is None:
            item = electrodes
        else:
            item = (electrodes, time)
        try:
            return self._stim['data'][item]
        except IndexError as e:
            if not isinstance(item, tuple):
                raise IndexError(e)

        # STEP 3: INTERPOLATE TIME
        # From here on out, we know that ``item`` is a tuple, otherwise we
        # would have raised an IndexError above.
        if self.time is None:
            raise ValueError("Cannot interpolate time if time=None.")
        time = np.array([time]).flatten()
        if (not isinstance(electrodes, (list, np.ndarray)) and
                electrodes == Ellipsis):
            data = self.data
        else:
            data = self.data[electrodes, :].reshape(-1, len(self.time))
        data = _interp_rows(time, self.time, data).astype(np.float32)
        # Return a single element as scalar:
        if data.size == 1:
            data = data.ravel()[0].item()
        return data

    def __eq__(self, other):
        """Returns True if two Stimulus objects are identical

        Two Stimulus objects are considered identical if they have the same
        electrode names, time steps, and data points.

        Parameters
        ----------
        other : any
            Another object or variable to which the current object should be
            compared.

        Examples
        --------
        >>> from pulse2percept.stimuli import Stimulus
        >>> Stimulus([1, 2, 3]) == Stimulus([1, 2, 3])
        True

        >>> Stimulus(np.ones(3)) == Stimulus(np.zeros(5))
        False

        Compare a Stimulus with something else entirely:

        >>> Stimulus(np.ones(3)) == 1
        False

        """
        if not isinstance(other, Stimulus):
            return False
        # Two stimuli that hold the same numbers in different units are not
        # the same stimulus: 500 uA of current is not 500 gray levels.
        if self.unit != other.unit or self.time_unit != other.time_unit:
            return False
        if self.time is None:
            if other.time is not None:
                return False
        else:
            if other.time is None:
                return False
            if len(self.time) != len(other.time):
                return False
            if not np.allclose(self.time, other.time, atol=DT):
                return False
        if len(self.electrodes) != len(other.electrodes):
            return False
        if not _names_equal(self.electrodes, other.electrodes):
            return False
        if self.shape != other.shape:
            return False
        if not (np.array_equal(self.data, other.data) or
                np.allclose(self.data, other.data)):
            return False
        return True

    def __ne__(self, other):
        """Returns True if two Stimulus objects are different

        Two Stimulus objects are considered different if they store different
        electrode names, time steps, or data points.

        Parameters
        ----------
        other : any
            Another object or variable to which the current object should be
            compared.

        Examples
        --------
        Compare two Stimulus objects:

        >>> from pulse2percept.stimuli import Stimulus
        >>> stim1 = Stimulus(np.ones(3))
        >>> stim2 = Stimulus(np.zeros(5))
        >>> stim1 != stim2
        True

        """
        return not self.__eq__(other)

    def _apply_operator(self, a, op, b, field='data'):
        """Template for all arithmetic operators"""
        # One of the arguments must be a scalar (the other being self.data):
        a_supported = np.isscalar(a) and not isinstance(a, str)
        b_supported = np.isscalar(b) and not isinstance(b, str)
        if not a_supported and not b_supported:
            raise TypeError(f"Unsupported operand for types {(type(a))} and "
                            f"{type(b)}")
        # Return a copy of the current object with the new data. The operator
        # produces a new array for `field`; the other fields must be copied
        # explicitly:
        stim = self._derived()
        time = stim.time
        if field == 'time':
            time = op(a, b)
        elif time is not None:
            time = time.copy()
        stim._stim = {'data': op(a, b) if field == 'data' else stim.data.copy(),
                      'electrodes': stim.electrodes.copy(),
                      'time': time}
        return stim

    def _scaled(self, factor):
        """This stimulus with every amplitude scaled by ``factor``"""
        return self._scale_components(factor)

    def _scale_components(self, factor):
        """An unmerged collection scales its entries instead"""
        if self._components is None:
            return None
        if not all(isinstance(src, Stimulus) for src, _ in self._components):
            return None
        stim = self._shallow_copy()
        stim._components = [(src * factor, n) for src, n in self._components]
        stim._forget_waveform(self.electrodes)
        return stim

    def _operate(self, op, scalar, reverse=False):
        """Apply an arithmetic operator to the stimulus"""
        if np.isscalar(scalar) and not isinstance(scalar, str):
            factor = _scale_factor(op, scalar, reverse)
            if factor is not None:
                scaled = self._scaled(factor)
                if scaled is not None:
                    return scaled
        data = self.data
        a, b = (scalar, data) if reverse else (data, scalar)
        return self._apply_operator(a, op, b)

    def _as_amplitude(self, scalar):
        """Normalize an operand that is added to or subtracted from the data"""
        return as_value(scalar, self.unit)

    def _as_factor(self, scalar):
        """Normalize an operand that scales the data"""
        return as_value(scalar, dimensionless)

    def _as_time(self, scalar):
        """Normalize an operand that shifts the stimulus in time"""
        return as_value(scalar, self.time_unit)

    def _slice_times(self, time):
        """The time points a slice of the time axis asks for"""
        return _slice_times(time, self.time, self.time_unit)

    def __add__(self, scalar):
        """Add a scalar to every data point in the stimulus"""
        return self._operate(ops.add, self._as_amplitude(scalar))

    def __radd__(self, scalar):
        """Add a scalar to every data point in the stimulus"""
        return self.__add__(scalar)

    def __sub__(self, scalar):
        """Subtract a scalar from every data point in the stimulus"""
        return self._operate(ops.sub, self._as_amplitude(scalar))

    def __rsub__(self, scalar):
        """Subtract every data point in the stimulus from a scalar"""
        return self._operate(ops.sub, self._as_amplitude(scalar),
                             reverse=True)

    def __mul__(self, scalar):
        """Multiply every data point in the stimulus with a scalar"""
        return self._operate(ops.mul, self._as_factor(scalar))

    def __rmul__(self, scalar):
        """Multiply every data point in the stimulus with a scalar"""
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        """Divide every data point in the stimulus by a scalar"""
        return self._operate(ops.truediv, self._as_factor(scalar))

    def __neg__(self):
        """Flip the sign of every data point in the stimulus"""
        return self.__mul__(-1)

    def __rshift__(self, scalar):
        """Shift all times some ms into the future (shorthand for shift)"""
        return self.shift(scalar)

    def __lshift__(self, scalar):
        """Shift all times some ms into the past (shorthand for -shift)"""
        return self.shift(-self._as_time(scalar))

    def _check_stim(self, stim):
        """Check stimulus data for consistency"""
        for field in ['data', 'electrodes', 'time']:
            if field not in stim:
                raise AttributeError(f"Stimulus dict must contain a field "
                                     f"'{field}'.")
        data_shape = stim['data'].shape
        if data_shape[0] > 0 and stim['data'].ndim != 2:
            raise ValueError(f"Stimulus data must be a 2-D NumPy array, not "
                             f"{stim['data'].ndim}-D.")
        n_electrodes = len(stim['electrodes'])
        if n_electrodes != data_shape[0]:
            raise ValueError(f"Number of electrodes ({n_electrodes}) must match the number "
                             f"of rows in the data array "
                             f"({data_shape[0]}).")
        if stim['time'] is not None:
            n_time = len(stim['time'])
            if n_time != data_shape[1]:
                raise ValueError(f"Number of time points ({n_time}) must match the "
                                 f"number of columns in the data array "
                                 f"({data_shape[1]}).")
            if not is_strictly_increasing(stim['time'], tol=0.95*DT):
                # Report the offending points rather than the whole axis:
                t = np.asarray(stim['time'])
                bad = np.flatnonzero(np.diff(t) < 0.95 * DT)
                shown = ', '.join(f"t[{i}]={t[i]:g} -> t[{i + 1}]={t[i + 1]:g}"
                                  for i in bad[:5])
                more = f" (and {bad.size - 5} more)" if bad.size > 5 else ""
                warnings.warn(f"Time points must be strictly monotonically "
                              f"increasing, but {bad.size} of {n_time} are "
                              f"less than DT={DT} apart: {shown}{more}.")
        elif data_shape[0] > 0:
            if data_shape[1] > 1:
                raise ValueError("Number of columns in the data array must be "
                                 "1 if time=None.")

    @staticmethod
    def _own(arr, dtype):
        """An immutable, C-contiguous array of dtype"""
        if arr is None:
            return None
        if isinstance(arr, _AdoptableArray) and arr.dtype == dtype:
            owned = np.ascontiguousarray(arr, dtype=dtype)
        else:
            owned = np.array(arr, dtype=dtype, order='C', copy=True)
        owned.flags.writeable = False
        return owned

    @staticmethod
    def _own_names(electrodes):
        """The electrode names, in a container nobody can write into"""
        if isinstance(electrodes, ElectrodeNames):
            return electrodes
        owned = np.array(electrodes)
        owned.flags.writeable = False
        return owned

    @property
    def _stim(self):
        """A dictionary containing all the stimulus data

        Reading this is what materializes the waveform of a stimulus that
        deferred building one (see :py:meth:`_defer` and :py:meth:`_render`).
        """
        if self.__stim['data'] is None:
            promised = self.__stim['electrodes']
            # The setter installs the rendered state, so `_render` runs once.
            # It also clears the components:
            components = self._components
            self._stim = self._render()
            self._components = components
            if not _names_equal(promised, self.__stim['electrodes']):
                raise ValueError(
                    f"{type(self).__name__}._render() returned rows for "
                    f"different electrodes than the stimulus said it drives. "
                    f"Naming them is what lets 'electrodes' be read without "
                    f"generating a waveform, so the two cannot disagree.")
        return self.__stim

    @_stim.setter
    def _stim(self, stim):
        self._check_stim(stim)
        self._components = None
        self.__stim = {**stim,
                       'data': self._own(stim['data'], np.float32),
                       # Time is deliberately float64 while data is float32:
                       'time': self._own(stim['time'], np.float64),
                       'electrodes': self._own_names(stim['electrodes'])}

    @property
    def data(self):
        """Stimulus data container

        A read-only 2-D NumPy array that contains the sampled waveform, where
        the rows denote electrodes and the columns denote points in time.
        """
        return self._stim['data']

    @property
    def shape(self):
        """Data container shape"""
        return self.data.shape

    @property
    def unit(self):
        """The unit ``data`` is expressed in

        .. versionadded:: 0.10.0

        """
        return self._unit

    @property
    def time_unit(self):
        """The unit ``time`` is expressed in

        .. versionadded:: 0.10.0

        """
        return self._time_unit

    @property
    def quantity(self):
        """The stimulus data, with its unit attached

        .. versionadded:: 0.10.0

        Examples
        --------
        >>> from pulse2percept.stimuli import Stimulus
        >>> from pulse2percept.units import uA
        >>> Stimulus([500, 1000] * uA).quantity
        [[ 500.]
         [1000.]] uA

        """
        return Quantity(self.data, self.unit)

    @property
    def time_quantity(self):
        """The stimulus time axis with its unit attached, or None

        .. versionadded:: 0.10.0

        """
        if self.time is None:
            return None
        return Quantity(self.time, self.time_unit)

    def values(self, unit=None):
        """The stimulus data, expressed in ``unit``

        .. versionadded:: 0.10.0

        Parameters
        ----------
        unit : :py:class:`~pulse2percept.units.Unit`, optional
            The unit to express the data in. Must be compatible with
            :py:attr:`~pulse2percept.stimuli.Stimulus.unit`. If None, the
            stimulus' own unit is used and ``data`` is returned as it is
            stored.

        Returns
        -------
        values : np.ndarray
            An ordinary NumPy array, never a
            :py:class:`~pulse2percept.units.Quantity`.

        Examples
        --------
        >>> from pulse2percept.stimuli import Stimulus
        >>> from pulse2percept.units import uA, mA
        >>> Stimulus([500, 1000] * uA).values(mA)
        array([[0.5],
               [1. ]], dtype=float32)

        """
        if unit is None:
            return self.data
        return self.quantity.to_value(unit)

    def times(self, unit=None):
        """The stimulus time axis, expressed in ``unit``

        .. versionadded:: 0.10.0

        Parameters
        ----------
        unit : :py:class:`~pulse2percept.units.Unit`, optional
            The unit to express the time axis in. If None, ``time`` is
            returned as it is stored (milliseconds).

        Returns
        -------
        times : np.ndarray or None
            An ordinary NumPy array, or None if the stimulus has no time
            component.

        """
        if self.time is None:
            return None
        if unit is None:
            return self.time
        return self.time_quantity.to_value(unit)

    @property
    def electrodes(self):
        """Electrode names
        A list of electrode names, corresponding to the rows in the data
        container.
        """
        return self.__stim['electrodes']

    @property
    def time(self):
        """A list of time steps (i.e., the columns in the data container)"""
        return self._stim['time']

    @property
    def is_compressed(self):
        """Flag indicating whether the stimulus has been compressed"""
        return self._is_compressed

    @property
    def dt(self):
        """Sampling time step (duration of signal edge transitions)

        .. versionadded:: 0.7

        """
        return DT

    @property
    def is_charge_balanced(self):
        """Flag indicating whether the stimulus is charge-balanced

        A stimulus with a time component is considered charge-balanced if its
        net current is smaller than 10 pico Amps.
        For the whole stimulus to be charge-balanced, every electrode must be
        charge-balanced as well.

        .. versionchanged:: 0.10.0
            Returns None for a stimulus that is not measured in units of
            current.

        """
        if self.unit.dimension != uA.dimension:
            return None
        if self.time is None:
            return np.allclose(self.data, 0, atol=MIN_AMP)
        return np.allclose(trapezoid(self.data, x=self.time), 0, atol=MIN_AMP)

    @property
    def duration(self):
        """Stimulus duration (ms)"""
        return self.time[-1]


def _has_time_axis(stim):
    """Whether a stimulus has a time component, without sampling it"""
    return _component_shape(stim)[2]


def _as_filename(source):
    """Return ``source`` as a string path, or None if it names no file"""
    if isinstance(source, (str, os.PathLike)):
        return os.fsdecode(source)
    return None


class ImageStimulus(Stimulus):
    """ImageStimulus

    A stimulus made from an image, where each pixel gets assigned to an
    electrode, and grayscale values in the range [0, 255] get converted to
    activation values in the range [0, 1].

    .. seealso ::

        *  `Basic Concepts > Electrical Stimuli <topics-stimuli>`
        *  :py:class:`~pulse2percept.stimuli.VideoStimulus`

    .. versionadded:: 0.7

    Parameters
    ----------
    source : str, os.PathLike, ImageStimulus, or np.ndarray
        Path to an image file (``str`` or :py:class:`pathlib.Path`). File types
        are inferred from the file ending (support types include JPG, PNG, and
        TIF).

        .. note::

            For GIFs, use :py:class:`~pulse2percept.stimuli.VideoStimulus`.

        .. versionchanged:: 0.11.0
            A :py:class:`pathlib.Path` is accepted wherever a filename is.
            ``metadata['source']`` is always a string.

    resize : (height, width) or None, optional
        Shape of the resized image. If one of the dimensions is set to -1,
        its value will be inferred by keeping a constant aspect ratio.

    as_gray : bool, optional
        Flag whether to convert the image to grayscale.
        A four-channel image is interpreted as RGBA (e.g., a PNG), and the
        alpha channel will be blended with the color black.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. If none are
        given, each pixel is named after its place in the image: a letter for
        the row, a number for the column, and a suffix for the color channel
        (e.g. 'A1', 'C12', 'A1_R'). See
        :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional
        Additional stimulus metadata can be stored in a dictionary.

    compress : bool, optional
        If True, will remove pixels with 0 grayscale value.

    """
    __slots__ = ('img_shape',)

    #: Pixel intensities are gray levels in [0, 1], not currents
    _default_unit = dimensionless

    def __init__(self, source, resize=None, as_gray=False,
                 electrodes=None, metadata=None, compress=False):
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            metadata = {'user': metadata}
        # The buffer the caller still holds, if any:
        borrowed = None
        fname = _as_filename(source)
        if fname is not None:
            # Filename provided:
            img = imread(fname)
            metadata['source'] = fname
            metadata['source_shape'] = img.shape
        elif isinstance(source, ImageStimulus):
            img = source.data.reshape(source.img_shape)
            borrowed = source.data
            metadata.update(source.metadata)
            if electrodes is None:
                electrodes = source.electrodes
        elif isinstance(source, np.ndarray):
            img = source
            borrowed = source
        else:
            raise TypeError(f"Source must be a filename, an array, or "
                            f"another ImageStimulus, not {type(source)}.")
        if img.ndim < 2 or img.ndim > 3:
            raise ValueError(f"Images must have 2 or 3 dimensions, not "
                             f"{img.ndim}.")
        # Convert to grayscale if necessary:
        if as_gray:
            if img.ndim == 3 and img.shape[2] == 4:
                # Blend the background with black:
                img = rgba2rgb(img, background=(0, 0, 0))
            if img.ndim == 3:
                img = rgb2gray(img)
        # Resize if necessary:
        if resize is not None:
            height, width = resize
            if height < 0 and width < 0:
                raise ValueError('"height" and "width" cannot both be -1.')
            if height < 0:
                height = int(img.shape[0] * width / img.shape[1])
            if width < 0:
                width = int(img.shape[1] * height / img.shape[0])
            img = img_resize(img, (height, width))
        # Store the original image shape for resizing and color conversion:
        self.img_shape = img.shape
        if electrodes is None:
            # Name every pixel after its place in the image: 'A1' is the
            # top-left pixel, 'C12' sits in the third row and twelfth column,
            # and a color image suffixes the channel ('A1_R'). The names are
            # generated on demand rather than stored:
            electrodes = ElectrodeNames(self.img_shape)
        data = img_as_float32(img)
        if borrowed is not None and np.may_share_memory(data, borrowed):
            data = data.copy()
        super().__init__(_adoptable(data.ravel()),
                                            time=None, electrodes=electrodes,
                                            metadata=metadata,
                                            compress=compress)
        self.metadata = metadata

    def _pprint_params(self):
        params = super()._pprint_params()
        params.update({'img_shape': self.img_shape})
        return params

    def _names_for(self, img, electrodes):
        """Electrode names for an image derived from this one"""
        if electrodes is not None:
            return electrodes
        return self.electrodes if np.shape(img) == self.img_shape else None

    def apply(self, func, *args, electrodes=None, **kwargs):
        """Apply a function to the image

        .. versionchanged:: 0.10.0

            ``func`` may now change the shape of the image, and ``electrodes``
            can name the result.

        Parameters
        ----------
        func : function
            The function to apply to the image. Must accept a 2D or 3D image
            and return a 2D or 3D image. The returned image need not have the
            same shape as the original; see ``electrodes``.
        * args :
            Additional positional arguments passed to the function
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, the original names are carried over whenever ``func`` leaves
            the shape of the image alone, and the result is named after its
            place in the new image otherwise (e.g. for
            ``skimage.transform.resize``). See
            :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

            .. note::
               The number of electrode names provided must match the number of
               pixels in the returned image.
        **kwargs :
            Additional keyword arguments passed to the function

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object with the new image
        """
        # `func` gets a frame of its own: several of the scikit-image
        # transforms this exists to reach cannot take a read-only one.
        img = func(_as_writable(self.data.reshape(self.img_shape)),
                   *args, **kwargs)
        return ImageStimulus(img, electrodes=self._names_for(img, electrodes),
                             metadata=self.metadata)

    def invert(self):
        """Invert the gray levels of the image

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object with all grayscale values inverted
            in the range [0, 1].

        """
        img = self.data.reshape(self.img_shape)
        if len(self.img_shape) > 2:
            # Leave any alpha channel alone:
            img = img.copy()
            img[..., :3] = 1.0 - img[..., :3]
        else:
            img = 1.0 - img
        return ImageStimulus(img, electrodes=self.electrodes,
                             metadata=self.metadata)

    def rgb2gray(self, electrodes=None):
        """Convert the image to grayscale

        Parameters
        ----------
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, each pixel is named after its place in the image (e.g.
            'A1', 'C12', 'A1_R'). See
            :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

            .. note::
               The number of electrode names provided must match the number of
               pixels in the grayscale image.

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object with all RGB values converted to
            grayscale in the range [0, 1].

        Notes
        -----
        *  A four-channel image is interpreted as RGBA (e.g., a PNG), and the
           alpha channel will be blended with the color black.

        """
        img = self.data.reshape(self.img_shape)
        if img.ndim == 3 and img.shape[2] == 4:
            # Blend the background with black in one pass:
            img = np.clip(img[..., :3] * img[..., 3:4], 0.0, 1.0)
        if img.ndim == 3:
            img = rgb2gray(img)
        return ImageStimulus(img, electrodes=electrodes,
                             metadata=self.metadata)

    def resize(self, shape, electrodes=None, **kwargs):
        """Resize the image

        .. versionchanged:: 0.10.0

            Keyword arguments are passed on to scikit-image.

        .. _skimage.transform.resize: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize

        Parameters
        ----------
        shape : (rows, cols)
            Shape of the resized image
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, each pixel is named after its place in the image (e.g.
            'A1', 'C12', 'A1_R'). See
            :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

            .. note::
               The number of electrode names provided must match the number of
               pixels in the grayscale image.
        **kwargs :
            Additional keyword arguments passed to `skimage.transform.resize`_,
            such as ``order=0`` for nearest-neighbor interpolation (which keeps
            a binary image binary).

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object containing the resized image

        """
        height, width = shape
        if height < 0 and width < 0:
            raise ValueError('"height" and "width" cannot both be -1.')
        if height < 0:
            height = int(self.img_shape[0] * width / self.img_shape[1])
        if width < 0:
            width = int(self.img_shape[1] * height / self.img_shape[0])
        img = img_resize(self.data.reshape(self.img_shape), (height, width),
                         **kwargs)

        return ImageStimulus(img, electrodes=electrodes,
                             metadata=self.metadata)

    def crop(self, idx_rect=None, left=0, right=0, top=0, bottom=0,
             electrodes=None):
        """Crop the image

        This method maps a rectangle (defined by two corners) from the image
        to a rectangle of the given size. Alternatively, this method can be used
        to crop a number of columns either from the left or the right of the
        image, or a number of rows either from the top or the bottom.

        .. versionadded:: 0.8

        Parameters
        ----------
        idx_rect : 4-tuple (y0, x0, y1, x1)
            Image indices of the top-left corner ``[y0, x0]`` and bottom-right
            corner ``[y1, x1]`` (exclusive) of the rectangle to crop.
        left : int
            Number of columns to crop from the left
        right : int
            Number of columns to crop from the right
        top : int
            Number of rows to crop from the top
        bottom : int
            Number of rows to crop from the bottom
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, each pixel is named after its place in the image (e.g.
            'A1', 'C12', 'A1_R'). See
            :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

            .. note::

               The number of electrode names provided must match the number of
               pixels in the cropped image.

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object containing the cropped image

        """
        if idx_rect is not None:
            if left > 0 or right > 0 or top > 0 or bottom > 0:
                raise ValueError('Crop window "idx_rect" cannot be given at '
                                 'the same time as "left"/"right"/"top"/'
                                 '"bottom".')
            # Crop window is given by a rectangle (ignore left, right, etc.):
            try:
                y0, x0, y1, x1 = idx_rect
            except (ValueError, TypeError):
                raise TypeError('"idx_rect" must be a 4-tuple (y0, x0, y1, x1)')
        else:
            y0, x0 = top, left
            y1, x1 = self.img_shape[0] - bottom, self.img_shape[1] - right
        # Safety checks:
        if y1 <= y0 or x1 <= x0:
            raise ValueError(f"The corners do not define a valid rectangle:"
                             f"(y0,x0)=({y0},{x0}), (y1,x1)=({y1},{x1}).")
        if y0 < 0 or x0 < 0:
            raise ValueError(f"Top-left corner (y0,x0)=({y0},{x0}) lies "
                             f"outside the image.")
        if y1 > self.img_shape[0] or x1 > self.img_shape[1]:
            raise ValueError(f"Bottom-right corner (y1-1,x1-1)=({y1-1},{x1-1}) lies "
                             f"outside the image.")
        # Crop the image:
        img = self.data.reshape(self.img_shape)
        # Check if we have color channels & index appropriately
        if len(self.img_shape) == 3:
            cropped_img = img[y0:y1, x0:x1, :3]
        else:
            cropped_img = img[y0:y1, x0:x1]
        if electrodes is None:
            # Carry the cropped pixels' original names over, so that a pixel
            # keeps the same name before and after cropping:
            electrodes = self.electrodes.reshape(self.img_shape)
            if len(self.img_shape) == 3:
                electrodes = electrodes[y0:y1, x0:x1, :3].ravel()
            else:
                electrodes = electrodes[y0:y1, x0:x1].ravel()
        return ImageStimulus(cropped_img, electrodes=electrodes,
                             metadata=self.metadata)

    def trim(self, tol=0, electrodes=None):
        """Remove any black border around the image

        .. versionadded:: 0.7

        Parameters
        ----------
        tol : float
            Any pixels with gray levels > tol will be trimmed.
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, each pixel is named after its place in the image (e.g.
            'A1', 'C12', 'A1_R'). See
            :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

            .. note::
               The number of electrode names provided must match the number of
               pixels in the trimmed image.

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object with trimmed borders.

        """
        img = self.data.reshape(self.img_shape)
        return ImageStimulus(trim_image(img, tol=tol), electrodes=electrodes,
                             metadata=self.metadata)

    def threshold(self, thresh, **kwargs):
        """Threshold the image

        Parameters
        ----------
        thresh : str or float
            If a float in [0,1] is provided, pixels whose grayscale value is
            above said threshold will be white, others black.

            A number of additional methods are supported:

            *  'mean': Threshold image based on the mean of grayscale values.
            *  'minimum': Threshold image based on the minimum method, where
                          the histogram of the input image is computed and
                          smoothed until there are only two maxima.
            *  'local': Threshold image based on `local pixel neighborhood`_.
                        Requires ``block_size``: odd number of pixels in the
                        neighborhood.
            *  'otsu': `Otsu's method`_
            *  'isodata': `ISODATA method`_, also known as the Ridler-Calvard 
                          method or intermeans.

        .. _local pixel neighborhood: https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_local
        .. _Otsu's method: https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu
        .. _ISODATA method: https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_isodata

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object with two gray levels 0.0 and 1.0
        """
        if len(self.img_shape) > 2:
            raise ValueError("Thresholding is only supported for grayscale "
                             "(i.e., single-channel) images. Use `rgb2gray` "
                             "first.")
        img = self.data.reshape(self.img_shape)
        if isinstance(thresh, str):
            if thresh.lower() == 'mean':
                img = img > threshold_mean(img)
            elif thresh.lower() == 'minimum':
                img = img > threshold_minimum(img, **kwargs)
            elif thresh.lower() == 'local':
                img = img > threshold_local(img, **kwargs)
            elif thresh.lower() == 'otsu':
                img = img > threshold_otsu(img, **kwargs)
            elif thresh.lower() == 'isodata':
                img = img > threshold_isodata(img, **kwargs)
            else:
                raise ValueError(f"Unknown threshold method '{thresh}'.")
        elif np.isscalar(thresh):
            img = self.data.reshape(self.img_shape) > thresh
        else:
            raise TypeError(f"Threshold type must be str or float, not "
                            f"{type(thresh)}.")
        return ImageStimulus(img, electrodes=self.electrodes,
                             metadata=self.metadata)

    def rotate(self, angle, mode='constant', electrodes=None, **kwargs):
        """Rotate the image

        .. versionchanged:: 0.10.0

            Keyword arguments are passed on to scikit-image.

        .. _skimage.transform.rotate: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.rotate

        Parameters
        ----------
        angle : float or Quantity
            Angle by which to rotate the image (degrees).
            Positive: counter-clockwise, negative: clockwise
        mode : str, optional
            How to fill in the corners the rotation leaves empty; see
            `skimage.transform.rotate`_.
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, each pixel keeps the name it had before the rotation, unless
            ``resize=True`` grew the canvas, in which case the enlarged image is
            named after its own pixel grid. See
            :py:class:`~pulse2percept.stimuli.ElectrodeNames`.
        **kwargs :
            Additional keyword arguments passed to `skimage.transform.rotate`_,
            such as ``order``, ``cval``, or ``resize=True`` to grow the image so
            that it contains every rotated pixel.

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object containing the rotated image

        """
        # Rotating in place is the common case, and keeps the pixel names
        # meaningful; ``resize=True`` is available through kwargs:
        kwargs.setdefault('resize', False)
        angle = as_value(angle, deg, 'angle')
        img = img_rotate(_as_writable(self.data.reshape(self.img_shape)),
                         angle, mode=mode, **kwargs)
        return ImageStimulus(img, electrodes=self._names_for(img, electrodes),
                             metadata=self.metadata)

    def shift(self, shift_cols, shift_rows):
        """Shift the image foreground

        This function shifts the center of mass (CoM) of the image by the
        specified number of rows and columns.

        Parameters
        ----------
        shift_cols : float
            Number of columns by which to shift the CoM.
            Positive: to the right, negative: to the left
        shift_rows : float
            Number of rows by which to shift the CoM.
            Positive: downward, negative: upward

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object containing the shifted image

        """
        return self.apply(shift_image, shift_cols, shift_rows)

    def center(self, loc=None):
        """Center the image foreground

        This function shifts the center of mass (CoM) to the image center.

        Parameters
        ----------
        loc : (col, row), optional
            The pixel location at which to center the CoM. By default, shifts
            the CoM to the image center.

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object containing the centered image

        """
        # Calculate center of mass:
        img = self.data.reshape(self.img_shape)
        return ImageStimulus(center_image(img, loc=loc),
                             electrodes=self.electrodes,
                             metadata=self.metadata)

    def scale(self, scaling_factor):
        """Scale the image foreground

        This function scales the image foreground (excluding black pixels)
        by a factor.

        Parameters
        ----------
        scaling_factor : float
            Factory by which to scale the image

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object containing the scaled image

        """
        img = self.data.reshape(self.img_shape)
        return ImageStimulus(scale_image(img, scaling_factor),
                             electrodes=self.electrodes,
                             metadata=self.metadata)

    def filter(self, filt, **kwargs):
        """Filter the image

        Parameters
        ----------
        filt : str
            Image filter. Additional parameters can be passed as keyword
            arguments. The following filters are supported:

            *  'sobel': Edge filter the image using the `Sobel filter`_.
            *  'scharr': Edge filter the image using the `Scharr filter`_.
            *  'canny': Edge filter the image using the `Canny algorithm`_.
               You can also specify ``sigma``, ``low_threshold``,
               ``high_threshold``, ``mask``, and ``use_quantiles``.
            *  'median': Return local median of the image.
        **kwargs :
            Additional parameters passed to the filter

        .. _Sobel filter: https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel
        .. _Scharr filter: https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.scharr
        .. _Canny algorithm: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.canny

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object with the filtered image
        """
        if not isinstance(filt, str):
            raise TypeError(f"'filt' must be a string, not {type(filt)}.")
        filters = {'sobel': sobel, 'scharr': scharr, 'canny': canny,
                   'median': median}
        try:
            filt = filters[filt.lower()]
        except KeyError:
            raise ValueError(f"Unknown filter '{filt}'.")
        return self.apply(filt, **kwargs)

    def encode(self, amp_range=(0, 50), freq=20, implant=None, **kwargs):
        """Encode the image using amplitude modulation

        Encodes the image as a train of biphasic pulses, where the gray level
        of a pixel sets the amplitude of its pulses.

        This is a shorthand for
        :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`; use that directly
        for the full set of options.

        .. versionchanged:: 0.10.0

            Gray levels now map onto ``amp_range`` absolutely rather than being
            stretched to fill it (pass ``stretch=True`` for the old behavior),
            the image receives a pulse *train* rather than a single pulse, and
            ``implant`` encodes at electrode rather than pixel resolution.

        Parameters
        ----------
        amp_range : (min_amp, max_amp), optional
            Range of pulse amplitudes (uA). A gray level of 0 maps onto
            ``min_amp``, a gray level of 1 onto ``max_amp``.
        freq : float, optional
            Pulse train frequency (Hz). The image is treated as a single frame
            lasting 500 ms unless ``frame_dur`` says otherwise.
        implant : :py:class:`~pulse2percept.implants.Implant`, optional
            If given, the image is first sampled at the implant's electrode
            locations, so that the pulse trains are built at electrode rather
            than pixel resolution.
        **kwargs :
            Additional arguments passed to
            :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`.

        Returns
        -------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus`
            Encoded stimulus

        """
        # Imported here because `encoders` imports this module:
        from .encoders import AmplitudeEncoder
        return AmplitudeEncoder(amp_range=amp_range, freq=freq,
                                **kwargs).encode(self, implant=implant)

    def plot(self, ax=None, **kwargs):
        """Plot the stimulus

        Parameters
        ----------
        ax : matplotlib.axes.Axes or list thereof; optional, default: None
            A Matplotlib Axes object or a list thereof (one per electrode to
            plot). If None, a new Axes object will be created.

        Returns
        -------
        ax: matplotlib.axes.Axes
            Returns the axes with the plot on it

        """
        if ax is None:
            ax = plt.gca()
        if 'figsize' in kwargs:
            ax.figure.set_size_inches(kwargs.pop('figsize'))
        if 'vmin' in kwargs:
            vmin = kwargs.pop('vmin')
        else:
            vmin = 0

        cmap = None
        if len(self.img_shape) == 2:
            cmap = 'gray'
        if 'cmap' in kwargs:
            cmap = kwargs.pop('cmap')
        ax.imshow(self.data.reshape(self.img_shape), cmap=cmap, vmin=vmin,
                  **kwargs)
        return ax

    def save(self, fname, vmin=0, vmax=None):
        """Save the stimulus as an image

        Parameters
        ----------
        fname : str or os.PathLike
            The name of the image file to be created. Image type will be
            inferred from the file extension.

            .. versionchanged:: 0.11.0
                A :py:class:`pathlib.Path` is accepted.

        """
        fname = os.fsdecode(fname)
        # if vmax is not passed by user
        if vmax is None:
            vmax = self.data.max()
        # clip to vmin, vmax vals
        clipped_data = self.data.clip(vmin,vmax)
        # if not a TIFF file, scale to uint8
        if not fname.endswith(".tif") and not fname.endswith(".tiff"):
            # scale to [0,255] 
            scaled_data = ((clipped_data - vmin) * ( 1 / (vmax - vmin) * 255)).astype('uint8')
            imsave(fname, scaled_data.reshape(self.img_shape))
            warnings.warn(f"Stimulus {fname} has been scaled & compressed to the range [0, 255]. To retain the full precision and scaling of the original stimulus, please save using the TIFF format.", UserWarning)
        else:
            imsave(fname, clipped_data.reshape(self.img_shape))


#: Anything this close to a frame boundary is treated as being on it
_FRAME_TOL = 1e-6


def _read_video(source, format, start_time, stop_time):
    """Decode the frames of a video file that start in [start, stop) ms"""
    start_time = as_value(start_time, ms, 'start_time')
    stop_time = as_value(stop_time, ms, 'stop_time')
    clipped = start_time is not None or stop_time is not None
    for name, t in (('start_time', start_time), ('stop_time', stop_time)):
        if t is not None and not np.isfinite(t):
            raise ValueError(f'"{name}" must be a finite time in ms, not {t}.')
    if start_time is not None and start_time < 0:
        raise ValueError(f'"start_time" cannot be negative, but is '
                         f'{start_time} ms.')
    if (start_time is not None and stop_time is not None and
            stop_time <= start_time):
        raise ValueError(f'"stop_time" ({stop_time} ms) must be greater than '
                         f'"start_time" ({start_time} ms).')
    with video_reader(source, format=format) as reader:
        meta = reader.get_meta_data()
        fps = meta.get('fps') if meta is not None else None
        if clipped:
            if not fps:
                raise ValueError(f'"{source}" does not report a frame rate, '
                                 f'so "start_time"/"stop_time" cannot be '
                                 f'mapped onto frames.')
            first = 0 if start_time is None else _frame_index(start_time, fps)
            last = None if stop_time is None else _frame_index(stop_time, fps)
        else:
            first, last = 0, None
        if last is not None and last <= first:
            raise ValueError(f'No video frame starts in [{start_time}, '
                             f'{stop_time}) ms.')
        if first:
            reader.set_image_index(first)
        frames = []
        while last is None or first + len(frames) < last:
            try:
                frames.append(reader.get_next_data())
            except (IndexError, StopIteration, EOFError):
                break  # End of file
    if clipped and not frames:
        raise ValueError(f'No video frame starts in [{start_time}, '
                         f'{stop_time}) ms.')
    return np.array(frames), meta


def _frame_index(t, fps):
    """Index of the first frame that starts at or after ``t`` ms"""
    return int(np.ceil(t * fps / MS_PER_S - _FRAME_TOL))


class VideoStimulus(Stimulus):
    """VideoStimulus

    A stimulus made from a movie file, where each pixel gets assigned to an
    electrode, and grayscale values in the range [0, 255] get assigned to
    activation values in the range [0, 1].

    The frame rate of the movie is used to infer the time points at which to
    stimulate.

    .. seealso ::

        *  `Basic Concepts > Electrical Stimuli <topics-stimuli>`
        *  :py:class:`~pulse2percept.stimuli.ImageStimulus`

    .. versionadded:: 0.7

    Parameters
    ----------
    source : str, os.PathLike, VideoStimulus, or np.ndarray
        Path to a video file (``str`` or :py:class:`pathlib.Path`). File types
        are inferred from the file ending (support types include MP4, AVI, MOV,
        and GIF). Enforce a specific format via ``format``.

        .. versionchanged:: 0.11.0
            A :py:class:`pathlib.Path` is accepted wherever a filename is.
            ``metadata['source']`` is always a string.

        Alternatively, pass a <rows x columns x channels x frames> NumPy array
        or another :py:class:`~pulse2percept.stimuli.VideoStimulus` object.

    format : str
        A video format string supported by imageio, such as 'MP4', 'AVI', or
        'MOV'. Use if the file type cannot be inferred from ``source``.
        For a full list of supported formats, see
        https://imageio.readthedocs.io/en/stable/formats.html.

    resize : (height, width) or None, optional, default: None
        A tuple specifying the desired height and the width of each video frame

    as_gray : bool, optional
        Flag whether to convert the image to grayscale.
        A four-channel image is interpreted as RGBA (e.g., a PNG), and the
        alpha channel will be blended with the color black.

    electrodes : int, string or list thereof; optional, default: None
        Optionally, you can provide your own electrode names. If none are
        given, each pixel is named after its place in the image: a letter for
        the row, a number for the column, and a suffix for the color channel
        (e.g. 'A1', 'C12', 'A1_R'). See
        :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional, default: None
        Additional stimulus metadata can be stored in a dictionary.

    compress : bool, optional, default: False
        If True, will compress the source data in two ways:
        * Remove electrodes with all-zero activation.
        * Retain only the time points at which the stimulus changes.

    start_time, stop_time : float or Quantity, optional, default: None
        Load only the frames that start in the half-open interval
        ``[start_time, stop_time)`` of the source video, in milliseconds.
        Time-based clipping requires the video reader to report a frame rate.

        .. note::
           The clip starts at ``time[0] == 0`` no matter where it was cut
           from. To shorten a video that is already in memory, and keep its
           original time stamps, use
           :py:meth:`~pulse2percept.stimuli.VideoStimulus.crop` instead.

        .. versionadded:: 0.10.0

    """
    __slots__ = ('vid_shape',)

    #: Pixel intensities are gray levels in [0, 1], not currents; see
    #: :py:class:`~pulse2percept.stimuli.ImageStimulus`.
    _default_unit = dimensionless

    def __init__(self, source, format=None, resize=None, as_gray=False,
                 electrodes=None, time=None, metadata=None, compress=False,
                 start_time=None, stop_time=None):
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            metadata = {'user': metadata}
        # The buffer the caller still holds, if any (see below):
        borrowed = None
        fname = _as_filename(source)
        if fname is not None:
            vid, meta = _read_video(fname, format, start_time, stop_time)
            # Move frame index to the last dimension:
            if vid.ndim == 4:
                vid = np.ascontiguousarray(vid.transpose((1, 2, 3, 0)))
            elif vid.ndim == 3:
                vid = np.ascontiguousarray(vid.transpose((1, 2, 0)))
            # Combine video metadata with user-specified metadata:
            if meta is not None:
                metadata.update(meta)
            metadata['source'] = fname
            metadata['source_shape'] = vid.shape
            # Infer the time points from the video frame rate:
            time = np.arange(vid.shape[-1]) * MS_PER_S / meta['fps']
        elif isinstance(source, VideoStimulus):
            vid = source.data.reshape(source.vid_shape)
            borrowed = source.data
            metadata.update(source.metadata)
            if electrodes is None:
                electrodes = source.electrodes
            if time is None:
                time = source.time
        elif isinstance(source, np.ndarray):
            vid = source
            borrowed = source
            if time is None and 'fps' in metadata:
                # Infer the time points from the video frame rate:
                time = np.arange(vid.shape[-1]) * MS_PER_S / metadata['fps']
        else:
            raise TypeError(f"Source must be a filename, a 3D NumPy array or "
                            f"another VideoStimulus, not {type(source)}.")
        if fname is None and (start_time is not None or
                              stop_time is not None):
            raise ValueError('"start_time"/"stop_time" only apply to a video '
                             'read from a file. Use crop(idx_time=...) to '
                             'shorten an array or another VideoStimulus.')
        if vid.ndim < 3 or vid.ndim > 4:
            raise ValueError(f"Videos must have 3 or 4 dimensions, not "
                             f"{vid.ndim}.")
        # Convert to grayscale if necessary:
        if as_gray:
            if vid.ndim == 4:
                vid = rgb2gray(vid.transpose((0, 1, 3, 2)))
        # Convert to float array in [0, 1] and call the Stimulus constructor:
        vid = img_as_float32(vid)
        # Resize if necessary:
        if resize is not None:
            height, width = resize
            if height < 0 and width < 0:
                raise ValueError('"height" and "width" cannot both be -1.')
            if height < 0:
                height = int(vid.shape[0] * width / vid.shape[1])
            if width < 0:
                width = int(vid.shape[1] * height / vid.shape[0])
            vid = vid_resize(vid, (height, width, *vid.shape[2:]))
        # Store the original image shape for resizing and color conversion:
        self.vid_shape = vid.shape
        if electrodes is None:
            # One electrode per pixel, named after its place in the frame
            # ('A1', 'C12', 'A1_R' for a color video). The last axis holds the
            # frames, which are the time component and not electrodes:
            electrodes = ElectrodeNames(self.vid_shape[:-1])
        if borrowed is not None and np.may_share_memory(vid, borrowed):
            vid = vid.copy()
        super().__init__(_adoptable(vid.reshape((-1, vid.shape[-1]))),
                                            time=time, electrodes=electrodes,
                                            metadata=metadata,
                                            compress=compress)
        self.metadata = metadata

    def compress(self):
        """Compress the source data

        Also brings ``vid_shape`` back in line with the compressed data:
        compression drops the time points at which the video does not change,
        so the frame count of the source is no longer the frame count of the
        stimulus. Every ``data.reshape(vid_shape)`` in this module relies on
        that invariant. (Compression can also drop all-zero pixels, in which
        case no shape describes the data any more; see ``_frames``.)

        Returns
        -------
        compressed : :py:class:`~pulse2percept.stimuli.VideoStimulus`
        """
        super().compress()
        # ``Stimulus.__init__`` calls this method for ``compress=True``, which
        # is why ``vid_shape`` is set before the constructor runs: one
        # implementation then covers both that and an explicit ``compress()``.
        self.vid_shape = (*self.vid_shape[:-1], self.data.shape[-1])

    def _frames(self):
        """The stimulus as a dense <rows x columns [x channels] x frames> array

        Raises a ``ValueError`` if the video has been compressed in space,
        which removes all-zero pixels and therefore leaves nothing that can be
        reshaped back into a frame.
        """
        n_px = int(np.prod(self.vid_shape[:-1]))
        if self.data.shape[0] != n_px:
            raise ValueError(
                f"This video was compressed in space: {self.data.shape[0]} of "
                f"its {n_px} pixels are left, so its frames cannot be "
                f"reconstructed. Pass 'compress=False' to keep the video "
                f"dense.")
        return self.data.reshape(self.vid_shape)

    def _pprint_params(self):
        params = super()._pprint_params()
        params.update({'vid_shape': self.vid_shape})
        return params

    def _names_for(self, vid, electrodes):
        """Electrode names for a video derived from this one

        A pixel keeps its name across an operation that leaves the pixel grid
        alone, which is what makes 'A1' refer to the same thing before and
        after. An operation that resamples the grid (a resize, a rotation that
        grows the canvas) has no such correspondence to preserve, so the result
        is named afresh rather than inheriting names that no longer describe
        it. Only the frame layout is compared; the number of frames is the time
        axis, not an electrode count.
        """
        if electrodes is not None:
            return electrodes
        same = np.shape(vid)[:-1] == self.vid_shape[:-1]
        return self.electrodes if same else None

    def apply(self, func, *args, electrodes=None, **kwargs):
        """Apply a function to each frame of the video

        .. versionchanged:: 0.10.0

            ``func`` may now change the shape of a frame, and ``electrodes``
            can name the result.

        Parameters
        ----------
        func : function
            The function to apply to each frame in the video. Must accept a 2D
            or 3D image and return a 2D or 3D image. The returned frames need
            not have the same shape as the originals (but must all have the
            same shape as each other); see ``electrodes``.
        *args :
            Additional positional arguments passed to the function
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, the original names are carried over whenever ``func`` leaves
            the shape of a frame alone, and the result is named after its place
            in the new frame otherwise (e.g. for
            ``skimage.transform.resize``). See
            :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

            .. note::
               The number of electrode names provided must match the number of
               pixels in a returned frame.
        **kwargs :
            Additional keyword arguments passed to the function

        Returns
        -------
        stim : `VideoStimulus`
            A copy of the stimulus object with the new video
        """
        # `func` gets a frame of its own: several of the scikit-image
        # transforms this exists to reach cannot take a read-only one.
        frames = self._frames()
        vid = np.array([func(_as_writable(frames[..., idx]), *args, **kwargs)
                        for idx in range(frames.shape[-1])])
        # Move first axis (frames) to last:
        vid = np.moveaxis(vid, 0, -1)
        return VideoStimulus(vid, electrodes=self._names_for(vid, electrodes),
                             time=self.time, metadata=self.metadata)

    def invert(self):
        """Invert the gray levels of the video

        Returns
        -------
        stim : `VideoStimulus`
            A copy of the stimulus object with all grayscale values inverted
            in the range [0, 1].

        """
        return VideoStimulus(1.0 - self.data.reshape(self.vid_shape),
                             electrodes=self.electrodes, time=self.time,
                             metadata=self.metadata)

    def rgb2gray(self, electrodes=None):
        """Convert the video to grayscale

        Parameters
        ----------
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, each pixel is named after its place in the image (e.g.
            'A1', 'C12', 'A1_R'). See
            :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

            .. note::
               The number of electrode names provided must match the number of
               pixels in the grayscale video.

        Returns
        -------
        stim : `VideoStimulus`
            A copy of the stimulus object with all RGB values converted to
            grayscale in the range [0, 1].

        """
        vid = self.data.reshape(self.vid_shape)
        if len(self.vid_shape) == 4:
            vid = rgb2gray(vid.transpose((0, 1, 3, 2)))
        return VideoStimulus(vid, electrodes=electrodes, time=self.time,
                             metadata=self.metadata)

    def resize(self, shape, electrodes=None, **kwargs):
        """Resize the video

        .. versionchanged:: 0.10.0

            Keyword arguments are passed on to scikit-image.

        .. _skimage.transform.resize: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize

        Parameters
        ----------
        shape : (rows, cols)
            Shape of each frame in the resized video. If one of the dimensions
            is set to -1, its value will be inferred by keeping a constant
            aspect ratio.
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, each pixel is named after its place in the image (e.g.
            'A1', 'C12', 'A1_R'). See
            :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

            .. note::
               The number of electrode names provided must match the number of
               pixels in the resized video.
        **kwargs :
            Additional keyword arguments passed to `skimage.transform.resize`_,
            such as ``order=0`` for nearest-neighbor interpolation (which keeps
            a binary video binary).

        Returns
        -------
        stim : `VideoStimulus`
            A copy of the stimulus object containing the resized video

        """
        height, width = shape
        if height < 0 and width < 0:
            raise ValueError('"height" and "width" cannot both be -1.')
        if height < 0:
            height = int(self.vid_shape[0] * width / self.vid_shape[1])
        if width < 0:
            width = int(self.vid_shape[1] * height / self.vid_shape[0])
        vid = vid_resize(self.data.reshape(self.vid_shape),
                         (height, width, *self.vid_shape[2:]), **kwargs)
        return VideoStimulus(vid, electrodes=electrodes, time=self.time,
                             metadata=self.metadata)

    def crop(self, idx_space=None, idx_time=None, left=0, right=0, top=0,
             bottom=0, front=0, back=0, electrodes=None):
        """Crop the video

        This method maps a rectangle (defined by two corners) from each video
        frame to a rectangle of the given size. Similarly, the video can be
        shortened to a specified range of frames.

        Alternatively, this method can be used to crop a number of columns
        either from the left or the right of the video frame, or a number of
        rows either from the top or the bottom, or a number of frames from the
        front (beginning) or back (end) of the video.

        .. versionadded:: 0.8

        Parameters
        ----------
        idx_space : 4-tuple (y0, x0, y1, x1)
            Image indices of the top-left corner ``[y0, x0]`` and bottom-right
            corner ``[y1, x1]`` (exclusive) of the rectangle to crop.
        idx_time : tuple (t0, t1)
            Frame indices defining the start ``t0`` and end ``t1`` of the
            cropped video.
        left : int
            Number of columns to crop from the left of each video frame
        right: int
            Number of columns to crop from the right of each video frame
        top: int
            Number of rows to crop from the top of each video frame
        bottom : int
            Number of rows to crop from the bottom of each video frame
        front : int
            Number of frames to crop from the front (beginning) of the video
        back : int
            Number of frames to crop from the back (end) of the video
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, each pixel is named after its place in the image (e.g.
            'A1', 'C12', 'A1_R'). See
            :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

            .. note::

               The number of electrode names provided must match the number of
               pixels in the cropped image.

        Returns
        -------
        stim : `VideoStimulus`
            A copy of the stimulus object containing the video

        """
        if idx_space is not None:
            if left > 0 or right > 0 or top > 0 or bottom > 0:
                raise ValueError('Crop window "idx_space" cannot be given at '
                                 'the same time as "left"/"right"/"top"/'
                                 '"bottom".')
            # Crop window is given by a rectangle (ignore left, right, etc.):
            try:
                y0, x0, y1, x1 = idx_space
            except (ValueError, TypeError):
                raise TypeError('"idx_space" must be a 4-tuple (y0,x0,y1,x1)')
        else:
            # Crop window not given, use left/right/top/bottom:
            y0, x0 = top, left
            y1, x1 = self.vid_shape[0] - bottom, self.vid_shape[1] - right
        if idx_time is not None:
            if front > 0 or back > 0:
                raise ValueError('Crop window "idx_time" cannot be given at '
                                 'the same times as "front"/"back".')
            try:
                t0, t1 = idx_time
            except (ValueError, TypeError):
                raise TypeError('"idx_time" must be a tuple (t0, t1).')
        else:
            t0, t1 = front, self.vid_shape[-1] - back
        # Safety checks:
        if y1 <= y0 or x1 <= x0:
            raise ValueError(f"The corners do not define a valid rectangle:"
                             f"(y0,x0)=({y0},{x0}), (y1,x1)=({y1},{x1}).")
        if y0 < 0 or x0 < 0:
            raise ValueError(f"Top-left corner (y0,x0)=({y0},{x0}) lies "
                             f"outside the video frame.")
        if y1 >= self.vid_shape[0] or x1 >= self.vid_shape[1]:
            raise ValueError(f"Bottom-right corner (y1,x1)=({y1},{x1}) lies "
                             f"outside the video frame.")
        if t1 <= t0:
            raise ValueError(f"Start and stop frame do not form a valid range: "
                             f"t0={t0}, t1={t1}.")
        if t0 < 0 or t1 > self.vid_shape[-1]:
            raise ValueError(f"Start/stop frames lie outside the valid range: "
                             f"t0={t0}, t1={t1}")
        # Crop the video:
        vid = self.data.reshape(self.vid_shape)
        cropped_vid = vid[y0:y1, x0:x1, ..., t0:t1]  # could be RGB or gray
        time = self.time[t0:t1]
        if electrodes is None:
            # Carry the cropped pixels' original names over, so that a pixel
            # keeps the same name before and after cropping:
            electrodes = self.electrodes.reshape(self.vid_shape[:-1])
            electrodes = electrodes[y0:y1, x0:x1, ...].ravel()
        return VideoStimulus(cropped_vid, electrodes=electrodes, time=time,
                             metadata=self.metadata)

    def trim(self, tol=0, electrodes=None):
        """Remove any black border around the video

        .. versionadded:: 0.7

        Parameters
        ----------
        tol : float
            Any pixels with gray levels > tol will be trimmed.
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, each pixel is named after its place in the image (e.g.
            'A1', 'C12', 'A1_R'). See
            :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

            .. note::
               The number of electrode names provided must match the number of
               pixels in each frame of the trimmed video.

        Returns
        -------
        stim : `VideoStimulus`
            A copy of the stimulus object with trimmed borders.

        """
        vid = self.data.reshape(self.vid_shape)
        # First we trim each frame individually and record the start and stop
        # indices for rows and columns:
        rows, cols = [], []
        for i in range(vid.shape[-1]):
            _, r, c = trim_image(vid[..., i], return_coords=True)
            rows.append(r)
            cols.append(c)
        rows, cols = np.array(rows), np.array(cols)
        # Then we
        col_start, col_end = cols[:, 0].min(), cols[:, 1].max()
        row_start, row_end = rows[:, 0].min(), rows[:, 1].max()
        vid = vid[row_start:row_end, col_start:col_end, ...]
        return VideoStimulus(vid, electrodes=electrodes, metadata=self.metadata,
                             time=self.time)

    def rotate(self, angle, mode='constant', electrodes=None, **kwargs):
        """Rotate each frame of the video

        .. versionchanged:: 0.10.0

            Keyword arguments are passed on to scikit-image.

        .. _skimage.transform.rotate: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.rotate

        Parameters
        ----------
        angle : float or Quantity
            Angle by which to rotate each video frame (degrees).
            Positive: counter-clockwise, negative: clockwise
        mode : str, optional
            How to fill in the corners the rotation leaves empty; see
            `skimage.transform.rotate`_.
        electrodes : int, string or list thereof; optional
            Optionally, you can provide your own electrode names. If none are
            given, each pixel keeps the name it had before the rotation, unless
            ``resize=True`` grew the frame, in which case the enlarged video is
            named after its own pixel grid. See
            :py:class:`~pulse2percept.stimuli.ElectrodeNames`.
        **kwargs :
            Additional keyword arguments passed to `skimage.transform.rotate`_,
            such as ``order``, ``cval``, or ``resize=True`` to grow each frame
            so that it contains every rotated pixel.

        Returns
        -------
        stim : `VideoStimulus`
            A copy of the stimulus object containing the rotated video

        """
        # Rotating in place is the common case, and keeps the pixel names
        # meaningful; ``resize=True`` is available through kwargs:
        kwargs.setdefault('resize', False)
        angle = as_value(angle, deg, 'angle')
        data = self.data.reshape(self.vid_shape)
        if len(self.vid_shape) == 3:
            # A grayscale video can be fed to `rotate` in one go, with its
            # frames standing in for the color channels it expects:
            data = vid_rotate(_as_writable(data), angle, mode=mode,
                              **kwargs)
            return VideoStimulus(data,
                                 electrodes=self._names_for(data, electrodes),
                                 metadata=self.metadata, time=self.time)
        # Else need to feed in each frame individually:
        return self.apply(vid_rotate, angle, mode=mode, electrodes=electrodes,
                          **kwargs)

    def shift(self, shift_cols, shift_rows):
        """Shift the image foreground

        This function shifts the center of mass (CoM) of the image by the
        specified number of rows and columns.

        Parameters
        ----------
        shift_cols : float
            Number of columns by which to shift the CoM.
            Positive: to the right, negative: to the left
        shift_rows : float
            Number of rows by which to shift the CoM.
            Positive: downward, negative: upward

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object containing the shifted image

        """
        return self.apply(shift_image, shift_cols, shift_rows)

    def center(self, loc=None):
        """Center the image foreground

        This function shifts the center of mass (CoM) to the image center.

        Parameters
        ----------
        loc : (col, row), optional
            The pixel location at which to center the CoM. By default, shifts
            the CoM to the image center.

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object containing the centered image

        """
        return self.apply(center_image, loc=loc)

    def scale(self, scaling_factor):
        """Scale the image foreground

        This function scales the image foreground (excluding black pixels)
        by a factor.

        Parameters
        ----------
        scaling_factor : float
            Factory by which to scale the image

        Returns
        -------
        stim : `ImageStimulus`
            A copy of the stimulus object containing the scaled image

        """
        return self.apply(scale_image, scaling_factor)

    def filter(self, filt, **kwargs):
        """Filter each frame of the video

        Parameters
        ----------
        filt : str
            Image filter that will be applied to every frame of the video.
            Additional parameters can be passed as keyword arguments.
            The following filters are supported:

            *  'sobel': Edge filter the image using the `Sobel filter
               <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel>`_.
            *  'scharr': Edge filter the image using the `Scarr filter
               <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.scharr>`_.
            *  'canny': Edge filter the image using the `Canny algorithm
               <https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.canny>`_.
               You can also specify ``sigma``, ``low_threshold``,
               ``high_threshold``, ``mask``, and ``use_quantiles``.
            *  'median': Return local median of the image.
        **kwargs :
            Additional parameters passed to the filter

        Returns
        -------
        stim : `VideoStimulus`
            A copy of the stimulus object with the filtered image
        """
        if not isinstance(filt, str):
            raise TypeError(f"'filt' must be a string, not {type(filt)}.")
        if len(self.vid_shape) == 4:
            raise ValueError('Cannot apply filter to RGB video. Convert to '
                             'grayscale first.')
        filters = {'sobel': sobel, 'scharr': scharr, 'canny': canny,
                   'median': median}
        try:
            filt = filters[filt.lower()]
        except KeyError:
            raise ValueError(f"Unknown filter '{filt}'.")
        return self.apply(filt, **kwargs)

    def encode(self, amp_range=(0, 50), freq=20, implant=None, **kwargs):
        """Encode the video using amplitude modulation

        Encodes every frame of the video as a train of biphasic pulses, where
        the gray level of a pixel sets the amplitude of its pulses. Each train
        lasts one frame period.

        This is a shorthand for
        :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`; use that directly
        for the full set of options.

        .. versionchanged:: 0.10.0

            Gray levels now map onto ``amp_range`` absolutely rather than being
            stretched to fill it (pass ``stretch=True`` for the old behavior),
            each frame receives a pulse *train* rather than a single pulse, and
            ``implant`` encodes at electrode rather than pixel resolution.

        Parameters
        ----------
        amp_range : (min_amp, max_amp), optional
            Range of pulse amplitudes (uA). A gray level of 0 maps onto
            ``min_amp``, a gray level of 1 onto ``max_amp``.
        freq : float, optional
            Pulse train frequency (Hz).
        implant : :py:class:`~pulse2percept.implants.Implant`, optional
            If given, the video is first sampled at the implant's electrode
            locations, so that the pulse trains are built at electrode rather
            than pixel resolution. Strongly recommended: a video has orders of
            magnitude more pixels than an implant has electrodes.
        **kwargs :
            Additional arguments passed to
            :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`.

        Returns
        -------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus`
            Encoded stimulus

        """
        # Imported here because `encoders` imports this module:
        from .encoders import AmplitudeEncoder
        return AmplitudeEncoder(amp_range=amp_range, freq=freq,
                                **kwargs).encode(self, implant=implant)

    def __iter__(self):
        """Iterate over the video, one frame at a time

        .. versionchanged:: 0.11.0

            Each frame is handed out as a standalone
            :py:class:`~pulse2percept.stimuli.ImageStimulus` that carries the
            electrode names and metadata of the video, but no time axis

        Yields
        ------
        frame : :py:class:`~pulse2percept.stimuli.ImageStimulus`
            The frames of the video, in order.

        Raises
        ------
        ValueError
            If the video was compressed in space, in which case its frames
            cannot be reconstructed (see ``compress``).
        """
        frames = self._frames()
        for idx in range(frames.shape[-1]):
            yield ImageStimulus(frames[..., idx], electrodes=self.electrodes,
                                metadata=self.metadata)

    def play(self, fps=None, repeat=True, annotate_time=True, ax=None,
             fmt='jpg'):
        """Animate the video as HTML with JavaScript

        The video will be played in an interactive player in IPython or
        Jupyter Notebook.

        Parameters
        ----------
        fps : float or None
            If None, uses the video's time axis. Not supported for
            non-homogeneous time axis. May be given as a plain number of hertz
            or as a unitful frequency (e.g. ``30 * Hz``, ``0.03 * kHz``); see
            :py:mod:`pulse2percept.units`.
        repeat : bool, optional
            Whether the animation should repeat when the sequence of frames is
            completed.
        annotate_time : bool, optional
            If True, the time of the frame will be shown as t = X ms in the
            title of the panel.
        ax : matplotlib.axes.AxesSubplot, optional
            A Matplotlib axes object. If None, will create a new Axes object
        fmt : {'jpg', 'png'}, optional
            The image format used to embed the frames. 'jpg' keeps notebooks
            and doc pages an order of magnitude smaller; use 'png' if you need
            the frames to be pixel-exact.

            .. versionadded:: 0.10.0

        Returns
        -------
        ani : pulse2percept.utils.HTMLAnimation
            A Matplotlib animation object that will play the video
            frame-by-frame.

        Notes
        -----
        .. versionchanged:: 0.10.0

            The HTML player is now generated by
            :py:class:`~pulse2percept.utils.HTMLAnimation`, which renders the
            figure once and ships all frames as a single sprite sheet. This is
            roughly two orders of magnitude faster than Matplotlib's
            ``to_jshtml`` and produces much smaller notebooks and doc pages.
        """
        if self.time is None:
            raise ValueError("Cannot animate a percept with time=None.")
        frames = self._frames()

        # Only the inherited Matplotlib machinery (``save``,
        # ``to_html5_video``) runs these; the HTML player draws ``frames``
        # itself. Frames are handed out by index so that the title can be
        # looked up without tracking iterator state:
        def update(idx):
            if annotate_time:
                mat.axes.set_title(f't = {self.time[idx]:.2f} ms')
            mat.set_data(frames[..., idx])
            return mat

        def data_gen():
            return iter(range(frames.shape[-1]))

        # There are several options to animate a percept in Jupyter/IPython
        # (see https://stackoverflow.com/a/46878531). Displaying the animation
        # as HTML with JavaScript is compatible with most browsers and even
        # %matplotlib inline (although it can be kind of slow):
        plt.rcParams["animation.html"] = 'jshtml'
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure
        # Start from an empty frame:
        mat = ax.imshow(np.zeros(self.vid_shape[:-1]), cmap='gray',
                        vmin=0, vmax=self.data.max())
        plt.close(fig)
        # Create the animation. The frame data is handed to HTMLAnimation so
        # that it can render the HTML player without going through Matplotlib:
        labels = None
        if annotate_time:
            labels = [f't = {t:.2f} ms' for t in self.time]
        return HTMLAnimation(fig, update, data_gen, repeat=repeat,
                             interval=frame_interval(self.time, fps=fps),
                             save_count=len(self.time), image=mat,
                             labels=labels, fmt=fmt, frame_data=frames)
