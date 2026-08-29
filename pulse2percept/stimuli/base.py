""":py:class:`~pulse2percept.stimuli.Stimulus`, 
   :py:class:`~pulse2percept.stimuli.ImageStimulus`"""
import warnings
from ..units import (DimensionMismatchError, Quantity, Unit, as_value,
                     dimensionless, ms, uA)
from ..units.base import has_units
from ..utils import PrettyPrint, is_strictly_increasing
from ..utils.array import _interp_rows, _slice_times
from ..utils.constants import DT, MIN_AMP
from ._base import fast_compress_space, fast_compress_time
from ._merge import merge_time_axes
from .names import ElectrodeNames

from copy import copy, deepcopy
import operator as ops
from math import isclose
from scipy.integrate import trapezoid
import numpy as np


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
