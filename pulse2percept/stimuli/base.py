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
from .names import ElectrodeNames

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import operator as ops
from math import isclose
from scipy.integrate import trapezoid
import numpy as np


def _as_scalar_column(source):
    """Convert a flat sequence of scalars into an (N, 1) data container

    Returns None if ``source`` is not a non-empty list or tuple of scalars, in
    which case the caller has to fall back to the generic per-element path.
    An empty sequence is excluded on purpose: it must keep producing a 1-D
    (empty) data container.
    """
    if not isinstance(source, (list, tuple)) or not source:
        return None
    if not np.isscalar(source[0]) or isinstance(source[0], str):
        return None
    try:
        flat = np.asarray(source)
    except (TypeError, ValueError):
        # Ragged (e.g. [1, [2, 3]]): let the generic path run, so that it
        # raises the error it has always raised:
        return None
    # Let the dtype NumPy infers decide. Forcing float32 here would quietly
    # accept sequences the generic path rejects: `[1, None]` would become
    # `[1.0, nan]` rather than a TypeError. Strings, None and complex values
    # all infer to a non-numeric dtype and so fall through:
    if flat.ndim != 1 or flat.dtype.kind not in 'biuf':
        return None
    return flat.astype(np.float32).reshape((-1, 1))


def _names_equal(a, b):
    """Whether two containers hold the same electrode names

    Two ``ElectrodeNames`` over the same grid hold the same names iff they
    select the same indices, which compares a few million integers rather than
    generating (and then comparing) a few million strings.
    """
    if isinstance(a, ElectrodeNames) and isinstance(b, ElectrodeNames):
        if a.grid_shape == b.grid_shape:
            return np.array_equal(a.indices, b.indices)
    return np.array_equal(np.asarray(a), np.asarray(b))


def _index_of_name(electrodes, name):
    """Return the position of electrode ``name`` in ``electrodes``

    ``ElectrodeNames`` can do this arithmetically, in constant time.
    Everything else falls back to a linear scan, which is what looking up a
    name in a plain array of names has always cost.
    """
    if isinstance(electrodes, ElectrodeNames):
        return electrodes.index(name)
    return list(electrodes).index(name)


def _same_time_point(t, merge_tolerance):
    """How close two time points have to be to count as the same point

    Two stimuli that sample the very same instant hand us time points that
    differ by a few ulps: pulse trains build their time axis by accumulating a
    window duration, so the drift between two frequencies grows with t. Those
    are too far apart to merge on an exact comparison, yet far closer than the
    DT that the rest of the code expects to separate two distinct time points,
    so the tolerance scales with the magnitude of ``t``. The cap keeps it below
    DT no matter how large ``t`` gets, so that points which really are a time
    step apart are never merged.

    Parameters
    ----------
    t : np.ndarray
        The time points whose magnitude sets the tolerance.
    merge_tolerance : float
        Lower bound on the tolerance, used where the accumulated drift is
        smaller than it (i.e., for small ``t``).

    Returns
    -------
    tol : np.ndarray
        Element-wise tolerance, same shape as ``t``.
    """
    return np.minimum(0.5 * DT,
                      np.maximum(merge_tolerance,
                                 8 * np.spacing(np.abs(t))))


def unique_time_points(time, merge_tolerance=1e-6):
    """Sorted union of several time axes, merging points that coincide

    Two stimuli that sample the same instant rarely agree on it to the last
    bit, because each accumulated its own way there. An exact ``np.unique``
    would keep both copies, leaving the merged axis with a pair of points far
    closer together than the DT that separates two genuinely distinct ones.

    Parameters
    ----------
    time : list of 1-D arrays
        The time axes to merge.
    merge_tolerance : float, optional
        Two time points closer together than this (or than the accumulated
        drift at their magnitude, whichever is coarser) are the same point.

    Returns
    -------
    t_sorted : 1-D array
        The sorted, concatenated time points.
    starts_group : 1-D bool array
        Which entries of ``t_sorted`` start a new group, i.e. which of them
        survive the merge.
    order : 1-D int array
        The permutation that sorted the concatenated axes.

    """
    t_all = np.concatenate(time).astype(np.float64)
    order = np.argsort(t_all, kind='stable')
    t_sorted = t_all[order]
    tol = _same_time_point(t_sorted[:-1], merge_tolerance)
    starts_group = np.concatenate(([True], np.diff(t_sorted) > tol))
    return t_sorted, starts_group, order


def merge_time_axes(data, time, merge_tolerance=1e-6):
    """
    Merge time axes

    When a collection of source types is passed, it is possible that they
    have different time axes (e.g., different time steps, or a different
    stimulus duration). In this case, we need to merge all time axes into a
    single, coherent one. This is expensive, because of interpolation.

    Parameters
    ----------
    data: list
        List of numpy.ndarray's containing data points associated with time axes.
    time: list
        List of numpy.ndarray's containing time points to merge
    merge_tolerance: float
        Absolute tolerance used when collecting unique time points from the
        time axes. Two time points that are closer together than this (or
        closer than float32 can resolve at their own magnitude, whichever is
        coarser) are treated as the same point.
    Returns
        Tuple of: list of new data points (linearly interpolated from merged time axis), list of new merged time axis.
    -------

    """
    # We can skip the costly interpolation if all `time` vectors are
    # identical:
    t0 = time[0]
    t0_tol = None
    identical = True
    for t in time:
        # np.array_equal is a lot cheaper than the element-wise comparison
        # (which builds several full-size temporaries) and, whenever it
        # succeeds, implies it. Use it as a fast path for the common case
        # where all stimuli share the very same time axis:
        if len(t) != len(t0):
            identical = False
            break
        if np.array_equal(t, t0):
            continue
        if t0_tol is None:
            t0_tol = _same_time_point(t0, merge_tolerance)
        # The axes may still be the same axis up to float32 noise. This used
        # to be an `np.allclose`, whose relative tolerance is 0.01 ms at
        # t = 1000 ms - ten time steps, which silently threw away time points
        # that differ by much more than float32 noise:
        if not np.all(np.abs(np.subtract(t, t0, dtype=np.float64)) <= t0_tol):
            identical = False
            break
    if identical:
        return data, [t0]
    # Otherwise, we need to interpolate. Keep only the unique time points
    # across stimuli. We need a higher tolerance to ensure interpolation is
    # correct.
    lengths = [len(t) for t in time]
    t_sorted, starts_group, order = unique_time_points(time, merge_tolerance)
    new_time = t_sorted[starts_group]
    # Snap every time axis onto the merged one, so that interpolating below
    # reproduces each stimulus exactly at its own sample points rather than an
    # ulp before or after them:
    snapped = np.empty_like(t_sorted)
    snapped[order] = new_time[np.cumsum(starts_group) - 1]
    # Now we need to interpolate the data values at each of these
    # new time points.
    new_data = []
    for t, d in zip(np.split(snapped, np.cumsum(lengths)[:-1]), data):
        # t is a 1D vector, d is a 2D data matrix and might have more than
        # one row:
        new_rows = [np.interp(new_time, t, row) for row in d]
        new_rows = np.array(new_rows).reshape((-1, len(new_time)))
        new_data.append(new_rows)
    return new_data, [new_time]


def _describe_unit(unit):
    """Name a unit the way an error message wants to read

    A dimensionless unit has no symbol to show, so saying "dimensionless ()"
    is worse than saying nothing at all.
    """
    if unit.dimension.is_dimensionless:
        return 'dimensionless units'
    return f'{unit.dimension.name} ({unit})'


def _stimulus_sources(source):
    """The Stimulus objects a source is built from, if any

    A source is either one stimulus, a collection of them, or something with
    no unit of its own (a scalar, an array, a filename).
    """
    if isinstance(source, Stimulus):
        return [source]
    if isinstance(source, dict):
        return [s for s in source.values() if isinstance(s, Stimulus)]
    if isinstance(source, (list, tuple)):
        return [s for s in source if isinstance(s, Stimulus)]
    return []


def _strip_units(source, unit):
    """Convert a source's quantities into plain numbers expressed in ``unit``

    Runs before the source-dispatch machinery in ``Stimulus._factory``, which
    reads dicts, lists, tuples and arrays element by element. A
    :py:class:`~pulse2percept.units.Quantity` deliberately has no sequence
    protocol, so it cannot be read that way -- and normalizing here means the
    dispatch, the interpolation and the compression below all keep seeing
    ordinary numbers, exactly as they always have.

    Anything without a unit is returned untouched, including the container it
    came in.
    """
    if isinstance(source, (Quantity, Unit)):
        return as_value(source, unit, 'source')
    if isinstance(source, dict):
        if any(has_units(v) for v in source.values()):
            return {k: _strip_units(v, unit) for k, v in source.items()}
    elif isinstance(source, (list, tuple)):
        if any(has_units(v) for v in source):
            return type(source)(_strip_units(v, unit) for v in source)
    return source


def _scale_factor(a, op, b, field):
    """The factor by which an arithmetic operator scales the stimulus data

    Returns 1 for an operator that leaves every amplitude where it is (a shift
    in time, or adding zero), the factor for one that scales them all by the
    same number, and None for one that does neither: a DC offset moves the
    waveform rather than resizing it, and no factor describes that.

    Stimulus types that record the parameters of their waveform in their
    metadata read this to keep those parameters in sync with the data. See
    :py:meth:`~pulse2percept.stimuli.Stimulus._rescale_metadata`.
    """
    if field == 'time':
        # Shifting in time moves the whole stimulus, but every amplitude in it
        # stays what it was:
        return 1.0
    # `_apply_operator` has established that exactly one of the operands is a
    # scalar; the other one is the data:
    scalar = b if np.ndim(a) else a
    if op is ops.mul:
        # Multiplication is commutative, so the operand order is moot:
        factor = scalar
    elif op is ops.truediv:
        # `Stimulus` has no `__rtruediv__`, so this is always data/scalar.
        # Dividing the data by zero fills it with inf rather than raising, so
        # the factor must not raise either -- it comes out non-finite below:
        with np.errstate(divide='ignore', invalid='ignore'):
            factor = np.divide(1.0, scalar)
    elif scalar == 0:
        # `stim + 0` and `stim - 0` change nothing; `0 - stim` flips the sign:
        factor = -1.0 if np.ndim(b) else 1.0
    else:
        return None
    # `stim * np.inf`, `stim * np.nan` and `stim / 0` leave a waveform of
    # infinities and NaNs, which is not a scaled version of anything:
    return factor if np.isfinite(factor) else None


class Stimulus(PrettyPrint):
    """Stimulus

    A stimulus is comprised of a labeled 2D NumPy array that contains the data,
    where the rows denote electrodes and the columns denote points in time.
    A stimulus can be created from a variety of source types (e.g., scalars,
    lists, NumPy arrays, and dictionaries).

    .. seealso ::

        *  `Basic Concepts > Electrical Stimuli <topics-stimuli>`

    .. versionadded:: 0.6

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
    __slots__ = ('metadata', '_is_compressed', '__stim', '_unit', '_time_unit')

    #: The unit ``data`` is stored in. Electrical stimuli are microamps, which
    #: is what every model, pulse and safety check in the library assumes; a
    #: subclass whose data is not a current (an image's gray levels, say)
    #: overrides this. A stimulus built from another stimulus inherits its
    #: unit instead, so a copy of an image stimulus does not become a current.
    _default_unit = uA

    #: The unit ``time`` is stored in.
    _default_time_unit = ms

    def __init__(self, source, electrodes=None, time=None, metadata=None,
                 compress=False):
        if isinstance(metadata, dict) and 'electrodes' in metadata.keys():
            self.metadata = metadata
        else:
            self.metadata = {'electrodes': {}, 'user': metadata}
        # Flag will be flipped in the compress method:
        self._is_compressed = False
        # Settle what the numbers below mean before reading any of them, then
        # convert every quantity into that unit. From here on the source is
        # ordinary numbers, which is all `_factory` has ever had to handle:
        self._unit, self._time_unit = self._resolve_units(source)
        source = _strip_units(source, self._unit)
        time = as_value(time, self._time_unit, 'time')
        # Extract the data and coordinates (electrodes, time) from the source:
        self._factory(source, electrodes, time, compress)

    def _resolve_units(self, source):
        """Determine the units this stimulus stores its data and time in

        A stimulus built from other stimuli speaks their unit; anything else
        speaks this class's default. Sources that disagree are an error rather
        than a silent choice of one of them: an image stimulus and a pulse
        train in the same collection have no common interpretation.
        """
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
        """Adopt the units of another stimulus

        For the handful of places that rebuild a stimulus out of raw arrays
        (resampling it onto an implant's electrodes, evaluating it at
        particular time points) and would otherwise fall back to the class
        default. Returns ``self`` so it can be chained onto a constructor.
        """
        self._unit = other.unit
        self._time_unit = other.time_unit
        return self

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        return {'data': self.data, 'electrodes': self.electrodes,
                'time': self.time, 'shape': self.shape, 'dt': self.dt,
                'is_charge_balanced': self.is_charge_balanced,
                'metadata': self.metadata}

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

        Returns ``electrodes=None`` when the source does not name its own
        electrodes; it is then up to the caller to number them. Likewise,
        ``time=None`` means the source has no time component (e.g. a scalar).
        """
        if isinstance(source, Stimulus):
            # e.g. a Stimulus being renamed, or a dict of Stimulus objects.
            # Brings along its own electrode names and time axis:
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
        # Whether we numbered the electrodes ourselves (0..N-1), in which case
        # they cannot possibly contain duplicates:
        _auto_electrodes = False
        if (_flat := _as_scalar_column(source)) is not None:
            # A flat sequence of scalars is one electrode per element, with no
            # time component. The collection path below would build a separate
            # 1x1 array (and time axis) for every single electrode:
            _data, _time, _electrodes = _flat, None, None
        elif isinstance(source, (dict, list, tuple)):
            # A collection: every entry is itself a source, contributing one
            # electrode (or, for a Stimulus, however many it already has):
            if isinstance(source, dict):
                iterator = source.items()
            else:
                iterator = enumerate(source)
            _time = []
            _electrodes = []
            _data = []
            for ele, src in iterator:
                # Extract times and data from source:
                d, t, e = self._parse_source(src, nested=True)
                _time.append(t)
                _data.append(d)
                if isinstance(source, dict):
                    # Special case, electrode names are specified in a dict:
                    _electrodes.append(ele)
                else:
                    # In all other cases, use the electrode names specified by
                    # the source (unless they're None):
                    _electrodes.append(e if e is not None else ele)
                try:
                    self.metadata['electrodes'][str(ele)] = {
                        'metadata': src.metadata,
                        'type': type(src)
                    }
                except AttributeError:
                    pass
            # Make sure all stimuli have time=None or none of them do:
            if len(np.unique([t is None for t in _time])) > 1:
                raise ValueError("If one stimulus has time=None, all others "
                                 "must have time=None as well.")
            # When none of the stimuli have time=None, we need to merge the
            # time axes (this is expensive because of interpolation):
            if len(_time) > 1 and _time[0] is not None:
                _data, _time = merge_time_axes(_data, _time)
            # Now make `_data` a 2-D NumPy array, with `_electrodes` as rows
            # and `_time` as columns (except sometimes `_time` is None).
            _data = np.vstack(_data) if _data else np.array([])
            _time = _time[0] if _time else None
        else:
            # A single source: a scalar, a NumPy array, or a Stimulus. The
            # latter might be handed to us by ProsthesisSystem if the user
            # built the stimulus themselves, and is also how a stimulus gets
            # new electrode names or a new time axis:
            _data, _time, _electrodes = self._parse_source(source)
            if isinstance(source, Stimulus):
                if 'electrodes' not in source.metadata.keys():
                    self.metadata['electrodes'][str(_electrodes[0])] = {
                        'metadata': source.metadata, 'type': type(source)}
                else:
                    self.metadata = source.metadata

        if _electrodes is None:
            # The source did not name its electrodes, so they are 0..N-1 --
            # unique by construction. Only build that array if something will
            # read it: user-supplied `electrodes` replaces it immediately
            # below, and the sole other reader is the metadata rename further
            # down, which needs per-electrode metadata to do anything at all.
            # An image or video stimulus has neither, so skipping this keeps a
            # million-element arange off the path that builds one.
            _auto_electrodes = True
            if electrodes is None or self.metadata.get('electrodes'):
                _electrodes = np.arange(_data.shape[0])

        # User can overwrite the names of the electrodes:
        if electrodes is not None:
            # May still be None, when the block above declined to build it.
            # The rename below already guards against that.
            _renamed_from = _electrodes
            if isinstance(electrodes, ElectrodeNames):
                # Names generated from a grid pattern. Flattening one is a
                # view, not a copy, and it already knows whether it can
                # contain duplicates - so neither the copy below nor the
                # `np.unique` further down is needed. This is the path taken
                # by every image and video stimulus, where `electrodes` has
                # one entry per pixel:
                _electrodes = electrodes.ravel()
                _auto_electrodes = _electrodes.check_unique()
            else:
                _electrodes = np.array([electrodes]).flatten()
                _auto_electrodes = False
        else:
            _renamed_from = None
            if isinstance(_electrodes, ElectrodeNames):
                # The source brought its own generated names along (e.g.
                # `Stimulus(image_stim)`). Keep them lazy rather than
                # flattening them into actual strings below:
                _electrodes = _electrodes.ravel()
                _auto_electrodes = _electrodes.check_unique()
            elif not isinstance(_electrodes, np.ndarray):
                # Could be a list of NumPy arrays, need to flatten:
                try:
                    _electrodes = np.concatenate(_electrodes)
                except ValueError:
                    _electrodes = np.array(_electrodes)
        if len(_electrodes) != _data.shape[0]:
            raise ValueError(f"Number of electrodes provided ({len(_electrodes)}) does "
                             f"not match the number of electrodes in the data "
                             f"({_data.shape[0]}).")
        # Electrodes we numbered ourselves are 0..N-1 and therefore unique by
        # construction, so the sort that np.unique performs can be skipped
        # (it dominates the cost of building an image or video stimulus):
        if not _auto_electrodes:
            if isinstance(_electrodes, ElectrodeNames):
                # Only a repeated index can make grid names collide, and
                # `check_unique` has just established that one does. The
                # renaming below writes into the container, so it needs the
                # actual names:
                _electrodes = np.asarray(_electrodes)
            unq, nunq = np.unique(_electrodes, return_index=True)
            if len(unq) != _data.shape[0]:
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

        # Per-electrode metadata is addressed by electrode name (that is how
        # BiphasicAxonMapModel finds its stimulus parameters), so renaming the
        # electrodes has to rename those keys too. Only stimuli that carry
        # such metadata need any of this, and they are the small ones: an
        # image or video stimulus has one electrode per pixel and no
        # per-electrode metadata at all. Testing that first keeps the
        # pair-by-pair walk below off the path that renames a million
        # electrodes:
        elec_meta = self.metadata.get('electrodes')
        if (elec_meta and _renamed_from is not None and
                len(_renamed_from) == len(_electrodes)):
            # Keys that do not belong to any electrode are left alone, and
            # `metadata` may be shared with the source stimulus, so never
            # rename in place:
            rename = {str(old): str(new)
                      for old, new in zip(_renamed_from, _electrodes)
                      if str(old) != str(new)}
            if rename:
                self.metadata = dict(self.metadata)
                self.metadata['electrodes'] = {rename.get(k, k): v
                                               for k, v in elec_meta.items()}

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

        # Store the data in the private container. Setting all elements at once
        # enforces consistency; e.g., between shape of electrodes and time:
        self._stim = {
            'data': np.ascontiguousarray(_data, dtype=np.float32),
            'electrodes': _electrodes,
            # Time is float64 while data is float32. The asymmetry is
            # deliberate: a time axis has one entry per column where the data
            # has one per electrode per column, so widening it costs almost
            # nothing, and float32 cannot carry a time axis at all. Its
            # resolution reaches DT=1e-3 ms at t = 8.4 s, past which the
            # DT-wide edges of a pulse collapse to zero width -- a 30 s pulse
            # train lost 952 of its edges that way.
            'time': _time if _time is None else _time.astype(np.float64),
        }
        # Compress the data upon request:
        if compress:
            self.compress()

    def _shallow_copy(self):
        """Copy the object without duplicating the data container

        Methods that return a new stimulus (``append``, the arithmetic
        operators) replace ``_stim`` wholesale, so there is no point in
        deep-copying the (potentially large) data arrays first. Everything
        else is preserved as it would be by ``deepcopy``: the subclass, its
        additional attributes, and an independent copy of ``metadata``.

        Note that the returned object shares its ``_stim`` dict with ``self``
        until the caller assigns a new one, which the ``_stim`` setter always
        does (it never mutates the dict in place).
        """
        stim = copy(self)
        stim.metadata = deepcopy(self.metadata)
        return stim

    @classmethod
    def _rescale_params(cls, metadata, factor):
        """Rewrite waveform parameters for a waveform scaled by ``factor``

        Some stimulus types describe their waveform with parameters that a
        model reads back instead of measuring the data itself: a
        :py:class:`~pulse2percept.stimuli.BiphasicPulseTrain` records
        amplitude, frequency and phase duration, and
        :py:class:`~pulse2percept.models.BiphasicAxonMapModel` predicts from
        those rather than from ``data``. Such a type overrides this method, so
        that an operation on the data carries its parameters along.

        Returns the metadata a stimulus of this class would carry after its
        data was multiplied by ``factor``, or -- for ``factor=None`` -- after
        an operation that leaves it no longer describable by those parameters
        at all. A plain ``Stimulus`` has no such parameters, so there is
        nothing to keep in sync.

        Parameters
        ----------
        metadata : dict
            The metadata of a stimulus of this class. Never modified in place.
        factor : float or None
            The factor the data was scaled by, or None if the operation was
            not a scaling (see ``_scale_factor``).

        Returns
        -------
        metadata : dict
        """
        return metadata

    def _rescale_metadata(self, factor):
        """Keep the metadata in sync with data that was scaled by ``factor``

        Called on the *copy* returned by ``append`` and by the arithmetic
        operators, once its data container has been replaced. That copy owns
        its metadata outright (``_shallow_copy`` deep-copies it), so this is
        free to rewrite it in place.

        A stimulus assembled from a collection carries the metadata of each
        source under ``metadata['electrodes']``, filed by electrode name and
        tagged with the class it came from. Dispatch to that class, which is
        the one that knows what its own parameters mean -- otherwise
        ``implant.stim * 2`` would scale the data and go on advertising the
        amplitude the pulse trains were built with.
        """
        if factor == 1:
            # Nothing about the waveform changed
            return
        elec_meta = self.metadata.get('electrodes')
        if not elec_meta:
            return
        for entry in elec_meta.values():
            if not isinstance(entry, dict):
                continue
            src, meta = entry.get('type'), entry.get('metadata')
            if isinstance(meta, dict) and isinstance(src, type) and \
                    issubclass(src, Stimulus):
                entry['metadata'] = src._rescale_params(meta, factor)

    def compress(self):
        """Compress the source data

        Returns
        -------
        compressed : :py:class:`~pulse2percept.stimuli.Stimulus`
        """
        data = self.data
        electrodes = self.electrodes
        time = self.time
        # Remove rows (electrodes) with all zeros:
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
        # its own, so it would carry `self`'s unit over numbers that never
        # meant that. Two stimuli can only be laid end to end if they measure
        # the same thing:
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
        if self.time is None or other.time is None:
            raise ValueError("Cannot append another stimulus if time=None.")
        if not _names_equal(self.electrodes, other.electrodes):
            raise ValueError("Both stimuli must have the same electrodes.")
        if other.time[0] < 0:
            raise NotImplementedError("Appending a stimulus with a negative "
                                      "time axis is currently not supported.")
        stim = self._shallow_copy()
        # Last time point of `self` can be merged with first point of `other`
        # but only if they have the same amplitude(s):
        if isclose(other.time[0], 0, abs_tol=DT):
            if not np.allclose(other.data[:, 0], self.data[:, -1]):
                err_str = (f"Data mismatch: Cannot append other stimulus "
                           f"because other[t=0] != this[t={self.time[-1]}ms]. You may need "
                           f"to shift the other stimulus in time by at least "
                           f"{DT:.1e} ms.")
                raise ValueError(err_str)
            time = np.hstack((self.time, other.time[1:] + self.time[-1]))
            data = np.hstack((self.data, other.data[:, 1:]))
        else:
            time = np.hstack((self.time, other.time + self.time[-1]))
            data = np.hstack((self.data, other.data))
        # Append the data points. If there's something wrong with the
        # concatenated list of time points, the stim setter will catch it:
        stim._stim = {'data': data,
                      'electrodes': self.electrodes,
                      'time': time}
        # Concatenating two waveforms in time is not a rescaling of either, so
        # any parameters describing the first one no longer describe the
        # result -- a pulse train appended to another is not one pulse train
        # at one amplitude and frequency, whatever its type still says:
        stim._rescale_metadata(None)
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
        # Nothing to remove. Note that ``electrodes`` must not be tested for
        # falsiness here, because 0 is a perfectly valid electrode index:
        if electrodes is None or np.size(electrodes) == 0:
            return
        if np.isscalar(electrodes) and electrodes == 'all':
            self._stim = {
                'data': self.data[[]],
                # Keep `electrodes` an array (of the same dtype) so that it can
                # still be indexed with a boolean mask afterwards:
                'electrodes': self.electrodes[[]],
                'time': self.time
            }
            return
        # Start with a list of True and set the removed electrodes to False:
        keep_el = np.ones(len(self.electrodes), dtype=bool)
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
        self._stim = {
            'data': self.data[keep_el],
            'electrodes': self.electrodes[keep_el],
            'time': self.time,
        }

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
        Padding never truncates the stimulus. Existing negative time points are
        preserved.
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
        zeros = np.zeros((data.shape[0], 1), dtype=data.dtype)
        if time[0] > 0:
            data = np.hstack((zeros, data))
            time = np.hstack(([0], time))
        if duration > time[-1]:
            data = np.hstack((data, zeros))
            time = np.hstack((time, [duration]))
        stim = self._shallow_copy()
        stim._stim = {'data': data,
                      'electrodes': self.electrodes.copy(),
                      'time': time}
        return stim

    def plot(self, electrodes=None, time=None, fmt='k-', ax=None):
        """Plot the stimulus

        .. versionadded:: 0.7

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
            A Matplotlib format string; e.g., 'ro' for red circles.
        ax : matplotlib.axes.Axes or list thereof; optional, default: None
            A Matplotlib Axes object or a list thereof (one per electrode to
            plot). If None, a new Axes object will be created.

        Returns
        -------
        axes : matplotlib.axes.Axes or np.ndarray of them
            Returns one matplotlib.axes.Axes per electrode
        """
        if self.time is None:
            # Cannot plot stimulus with single time point:
            raise NotImplementedError
        if electrodes is None:
            # Plot all electrodes:
            electrodes = self.electrodes
        elif isinstance(electrodes, (int, str)):
            # Convert to list so we can iterate over it:
            electrodes = [electrodes]
        # The user can ask for a range, slice, or list of time points, which
        # are either interpolated or loaded directly.
        if time is None:
            # Ask for a slice instead of `self.time` to avoid interpolation,
            # which can be time-consuming for an uncompressed stimulus:
            time = slice(None)
        # A range, a list of time points, or the endpoints and step of a slice
        # may all be given as quantities:
        time = self._as_time(time)
        if isinstance(time, tuple):
            # Return a range of time points:
            t_idx = (self.time > time[0]) & (self.time < time[1])
            # Include the end points (might have to be interpolated):
            t_vals = [time[0]] + list(self.time[t_idx]) + [time[1]]
            t_idx = t_vals
        elif isinstance(time, (list, np.ndarray)):
            # Return list of exact time points:
            t_idx = time
            t_vals = time
        elif isinstance(time, slice):
            # A stepped slice is a time range, and `__getitem__` will
            # interpolate onto it. Resolve it to those time points here as
            # well, or the curve would be drawn against the time points that
            # happen to sit at those column *indices* instead:
            t_vals = self._slice_times(time)
            if t_vals is None:
                # Every stored sample, taken by position:
                t_idx = time
                t_vals = self.time[time]
            else:
                t_idx = t_vals
        elif time == Ellipsis:
            t_idx = time
            t_vals = self.time[t_idx]
        else:
            raise TypeError(f'"time" must be a tuple, slice, list, or NumPy '
                            f'array, not {type(time)}.')
        axes = ax
        if axes is None:
            if len(electrodes) == 1:
                axes = plt.gca()
            else:
                _, axes = plt.subplots(nrows=len(electrodes),
                                       figsize=(8, 1.2 * len(electrodes)))
        if not isinstance(axes, (list, np.ndarray)):
            # Convert to list so w can iterate over it:
            axes = [axes]
        for i, ax in enumerate(axes):
            if not isinstance(ax, Axes):
                raise TypeError(f"'axes' must be a list of subplots, but "
                                f"axes[{i}] is {type(ax)}.")
        if len(axes) != len(electrodes):
            raise ValueError(f"Number of subplots ({len(axes)}) must be equal to the "
                             f"number of electrodes ({len(electrodes)}).")
        # Plot each electrode in its own subplot:
        for ax, electrode in zip(axes, electrodes):
            # Slice or interpolate stimulus:
            slc = self.__getitem__((electrode, t_idx))
            ax.plot(t_vals, np.squeeze(slc), fmt, linewidth=2)
            # Turn off the ugly box spines:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            # Annotate the subplot:
            ax.set_xticks([])
            ax.set_yticks([slc.min(), 0, slc.max()])
            x_pad = 0.02 * (t_vals[-1] - t_vals[0])
            ax.set_xlim(t_vals[0] - x_pad, t_vals[-1] + x_pad)
            y_pad = np.maximum(1, 0.02 * (slc.max() - slc.min()))
            ax.set_ylim(slc.min() - y_pad, slc.max() + y_pad)
            ax.set_ylabel(electrode)
        # Show x-ticks only on last subplot:
        axes[-1].set_xticks(np.linspace(t_vals[0], t_vals[-1], num=5))
        # Labels are common to all subplots. What the y axis shows depends on
        # what the stimulus is made of: an image or video stimulus holds gray
        # levels, and calling those an amplitude in microamps is simply wrong.
        if self.unit.dimension.is_dimensionless:
            ylabel = 'Value'
        elif self.unit == uA:
            # Spelled the way Matplotlib renders it:
            ylabel = r'Amplitude ($\mu$A)'
        else:
            ylabel = f'Amplitude ({self.unit})'
        axes[-1].figure.subplots_adjust(bottom=0.2)
        axes[-1].figure.text(0.5, 0, f'Time ({self.time_unit})', va='top',
                             ha='center')
        axes[-1].figure.text(0, 0.5, ylabel, va='center', ha='center',
                             rotation='vertical')
        if len(axes) == 1:
            return axes[0]
        return axes

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
                # Otherwise the slice stays what it is, and NumPy takes the
                # columns it names below.
            elif time is not Ellipsis:
                # A requested time point (or a list of them) may be unitful;
                # after this it is an ordinary number, which is what the
                # indexing and interpolation below have always worked on:
                time = self._as_time(time)
                # Convert to float so time is not mistaken for column index
                if np.array(time).dtype != bool:
                    time = np.float64(time)
        else:
            electrodes = item
            time = None

        # STEP 2: ELECTRODES COULD BE SPECIFIED AS INT OR STR
        if isinstance(electrodes, (list, np.ndarray)) or np.isscalar(electrodes):
            # Electrodes cannot be interpolated, so convert from slice,
            # ellipsis or indices into a list:
            parsed_electrodes = []
            for e in np.array([electrodes]).ravel():
                if isinstance(e, str):
                    # Use string as index into the list of electrode names:
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
        # Make sure electrode index is valid:
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
            # IndexErrors must still be thrown except when `item` is a tuple,
            # in which case we might want to interpolate time:
            if not isinstance(item, tuple):
                raise IndexError(e)

        # STEP 3: INTERPOLATE TIME
        # From here on out, we know that ``item`` is a tuple, otherwise we
        # would have raised an IndexError above.
        # First of all, if time=None, we won't interp:
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
        # np.allclose builds several full-size temporaries. np.array_equal is
        # much cheaper and, whenever it succeeds, implies it - so use it as a
        # fast path for the common case of comparing identical stimuli:
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
        # explicitly, so that the returned stimulus shares no buffer with the
        # original (`_shallow_copy` does not duplicate the data container):
        stim = self._shallow_copy()
        time = stim.time
        if field == 'time':
            time = op(a, b)
        elif time is not None:
            time = time.copy()
        stim._stim = {'data': op(a, b) if field == 'data' else stim.data.copy(),
                      'electrodes': stim.electrodes.copy(),
                      'time': time}
        # Parameters that describe the waveform (a pulse train's amplitude,
        # say) have to follow the data, or a model reading them back predicts
        # from a stimulus that is no longer the one it was handed:
        stim._rescale_metadata(_scale_factor(a, op, b, field))
        return stim

    def _as_amplitude(self, scalar):
        """Normalize an operand that is added to or subtracted from the data

        Adding to the data means adding an amplitude, so a quantity has to be
        one: ``stim + 0.5 * mA`` is 500 uA more current everywhere, and
        ``stim + 5 * ms`` is nothing at all. A bare number is taken to be in
        the stimulus' own unit, as it always was.
        """
        return as_value(scalar, self.unit)

    def _as_factor(self, scalar):
        """Normalize an operand that scales the data

        Deliberately narrower than
        :py:meth:`~pulse2percept.stimuli.Stimulus._as_amplitude`: a scale
        factor is a plain number. Letting ``stim * ms`` through would turn a
        stimulus into a charge, and a stimulus is a stimulus rather than a
        general physical array.
        """
        return as_value(scalar, dimensionless)

    def _as_time(self, scalar):
        """Normalize an operand that shifts the stimulus in time"""
        return as_value(scalar, self.time_unit)

    def _slice_times(self, time):
        """The time points a slice of the time axis asks for

        See :py:func:`~pulse2percept.utils.array._slice_times`, which
        :py:class:`~pulse2percept.percepts.Percept` indexing shares.
        """
        return _slice_times(time, self.time, self.time_unit)

    def __add__(self, scalar):
        """Add a scalar to every data point in the stimulus"""
        return self._apply_operator(self.data, ops.add,
                                    self._as_amplitude(scalar))

    def __radd__(self, scalar):
        """Add a scalar to every data point in the stimulus"""
        return self.__add__(scalar)

    def __sub__(self, scalar):
        """Subtract a scalar from every data point in the stimulus"""
        return self._apply_operator(self.data, ops.sub,
                                    self._as_amplitude(scalar))

    def __rsub__(self, scalar):
        """Subtract every data point in the stimulus from a scalar"""
        return self._apply_operator(self._as_amplitude(scalar), ops.sub,
                                    self.data)

    def __mul__(self, scalar):
        """Multiply every data point in the stimulus with a scalar"""
        return self._apply_operator(self.data, ops.mul,
                                    self._as_factor(scalar))

    def __rmul__(self, scalar):
        """Multiply every data point in the stimulus with a scalar"""
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        """Divide every data point in the stimulus by a scalar"""
        return self._apply_operator(self.data, ops.truediv,
                                    self._as_factor(scalar))

    def __neg__(self):
        """Flip the sign of every data point in the stimulus"""
        return self.__mul__(-1)

    def __rshift__(self, scalar):
        """Shift every time point in the stimulus some ms into the future

        Shorthand for :py:meth:`~pulse2percept.stimuli.Stimulus.shift`.
        """
        return self.shift(scalar)

    def __lshift__(self, scalar):
        """Shift every time point in the stimulus some ms into the past

        Shorthand for ``shift(-scalar)``; see
        :py:meth:`~pulse2percept.stimuli.Stimulus.shift`.
        """
        return self.shift(-self._as_time(scalar))

    def _check_stim(self, stim):
        # Check stimulus data for consistency:
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
                # Report the offending points rather than the whole axis: a
                # long pulse train has hundreds of thousands of time points,
                # and printing all of them buries the handful that are wrong
                # under megabytes of output.
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

    @property
    def _stim(self):
        """A dictionary containing all the stimulus data"""
        return self.__stim

    @_stim.setter
    def _stim(self, stim):
        self._check_stim(stim)
        # Every Cython kernel in the library takes the data as
        # ``float32[:, ::1]``, so the container has to hold it C-contiguous.
        # Not everything that builds a stimulus produces that: selecting
        # columns, as ``compress`` does, hands back an F-ordered array for a
        # multi-electrode stimulus. Left alone it would surface much later, as
        # a "ndarray is not C-contiguous" from whichever kernel happened to
        # receive it -- including ``fast_compress_space`` on a second
        # ``compress``. Fixing it here rather than at each call site keeps the
        # invariant in one place.
        data = np.ascontiguousarray(stim['data'])
        if data is not stim['data']:
            # `copy` hands out objects that share this dict, so replace it
            # rather than writing through:
            stim = {**stim, 'data': data}
        # All checks passed, store the data:
        self.__stim = stim

    @property
    def data(self):
        """Stimulus data container
        A 2-D NumPy array that contains the stimulus data, where the rows
        denote electrodes and the columns denote points in time.
        """
        return self._stim['data']

    @property
    def shape(self):
        """Data container shape"""
        return self.data.shape

    @property
    def unit(self):
        """The unit ``data`` is expressed in

        Microamps for an electrical stimulus, dimensionless for the gray
        levels of an :py:class:`~pulse2percept.stimuli.ImageStimulus` or
        :py:class:`~pulse2percept.stimuli.VideoStimulus`.

        Read-only. The canonical storage unit is fixed so that models, safety
        checks and Cython kernels can rely on it; ask for another unit with
        :py:meth:`~pulse2percept.stimuli.Stimulus.values`.

        .. versionadded:: 0.10.0

        """
        return self._unit

    @property
    def time_unit(self):
        """The unit ``time`` is expressed in (milliseconds)

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
            :py:class:`~pulse2percept.units.Quantity`. This is the boundary a
            numerical implementation should take its data across.

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
        return self._stim['electrodes']

    @property
    def time(self):
        """Time steps
        A list of time steps, corresponding to the columns in the data
        container.
        """
        return self._stim['time']

    @property
    def is_compressed(self):
        """Flag indicating whether the stimulus has been compressed

        Read-only: the flag is maintained by ``compress``. Assigning to it
        raises an ``AttributeError``.
        """
        return self._is_compressed

    @property
    def dt(self):
        """Sampling time step (ms)

        Defines the duration of the signal edge transitions.

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

        Returns None if the stimulus is not a current at all: the gray levels
        of an :py:class:`~pulse2percept.stimuli.ImageStimulus` integrate to a
        number like any others, but that number is not a charge and asking
        whether it is zero answers nothing. Note that this is "not applicable",
        not "unbalanced" -- it is
        :py:attr:`~pulse2percept.implants.ProsthesisSystem.safe_mode` that
        turns the question into an error, since a safety system genuinely
        cannot do its job on a stimulus that is not electrical.

        .. versionchanged:: 0.10.0
            Returns None for a stimulus that is not measured in units of
            current (was: integrated the values anyway).

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
