""":py:class:`~pulse2percept.stimuli.Stimulus`, 
   :py:class:`~pulse2percept.stimuli.ImageStimulus`"""
import warnings
from ..utils import PrettyPrint, is_strictly_increasing
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


def _interp_rows(x, xp, fp):
    """Linearly interpolate every row of ``fp`` at the time points ``x``

    Vectorized equivalent of ``[np.interp(x, xp, row) for row in fp]``, which
    is otherwise a Python-level loop over (potentially many thousands of)
    electrodes.

    The arithmetic is deliberately carried out in double precision and in the
    same order as ``np.interp``'s C loop, because temporal models resolve
    stimulus edges on a fixed simulation grid.

    Agreement with ``np.interp`` is exact wherever no arithmetic is needed:
    on a knot, or beyond the end points, where the stored value is assigned
    verbatim. For interior points it is exact to within one rounding - a C
    compiler may contract ``slope * dx + y0`` into a single fused
    multiply-add (it does on arm64), whereas the NumPy expression below
    always rounds twice. That is at most a few ULP of float64, well below the
    float32 that the caller ends up storing.

    Parameters
    ----------
    x : 1-D array
        Time points at which to interpolate.
    xp : 1-D array
        The stimulus time axis.
    fp : 2-D array
        The stimulus data, one row per electrode.

    Returns
    -------
    data : 2-D array
        Interpolated data, of shape ``(len(fp), len(x))``.
    """
    x = np.asarray(x, dtype=np.float64)
    xp = np.asarray(xp, dtype=np.float64)
    fp = np.asarray(fp)
    # np.interp's C loop is hard to beat per element; what the vectorized path
    # saves is one Python-level call per electrode. That only pays off with
    # enough electrodes to amortize its setup, and only while the result stays
    # small enough that the extra passes over it are cheaper than those calls.
    # Outside that regime (and for a non-monotonic time axis, where
    # np.interp's guess-based bracket search cannot be reproduced in a
    # vectorized way because its result depends on the preceding query point),
    # defer to np.interp itself:
    if (fp.shape[0] < 32 or x.size > 256 or xp.size < 2 or
            not np.all(np.diff(xp) > 0)):
        return np.array([np.interp(x, xp, row)
                         for row in fp]).reshape((-1, x.size))
    # Bracket index j such that xp[j] <= x < xp[j+1], as np.interp does. Note
    # that `j`, `x0` and `x1` are all 1-D (one entry per requested time point):
    j = np.clip(np.searchsorted(xp, x, side='right') - 1, 0, xp.size - 2)
    x0, x1 = xp[j], xp[j + 1]
    # Gather first and widen afterwards: upcasting all of `fp` would touch the
    # whole (potentially large) data container instead of just two columns
    # per requested time point:
    y0 = fp[:, j].astype(np.float64)
    y1 = fp[:, j + 1].astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = (y1 - y0) / (x1 - x0)     # slope; reused in place below
        out *= x - x0
        out += y0
        # np.interp retries from the right end of the interval if that gave a
        # NaN (which happens for infinite slopes), then gives up:
        nan = np.isnan(out)
        if nan.any():
            slope = (y1 - y0) / (x1 - x0)
            out = np.where(nan, slope * (x - x1) + y1, out)
            out = np.where(np.isnan(out) & (y0 == y1), y0, out)
    # The remaining corrections all select whole columns, so build the masks
    # on the 1-D time axis and write in place rather than allocating another
    # full-size array per correction:
    exact = x == x0
    if exact.any():
        # Exact hits on a knot return the stored value verbatim:
        out[:, exact] = y0[:, exact]
    # Beyond the end points, the value of the closest end point is returned:
    below = x <= xp[0]
    if below.any():
        out[:, below] = fp[:, :1]
    above = x >= xp[-1]
    if above.any():
        out[:, above] = fp[:, -1:]
    undefined = np.isnan(x)
    if undefined.any():
        out[:, undefined] = x[undefined]
    return out


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

    Time is stored as float32, whose resolution (7.6e-6 ms at t = 100 ms,
    6.1e-5 ms at t = 1000 ms) is much coarser than ``merge_tolerance``. Two
    stimuli that sample the very same instant therefore hand us time points
    that differ by a few ulps - pulse trains build their time axis by
    accumulating a window duration, so the drift between two frequencies grows
    with t. Those are far too far apart to merge on an absolute tolerance, yet
    far closer than the DT that the rest of the code expects to separate two
    distinct time points, so scale the tolerance with the magnitude of ``t``.
    The cap keeps the tolerance below DT no matter how large ``t`` gets, so
    that points which really are a time step apart are never merged.

    Parameters
    ----------
    t : np.ndarray
        The time points whose magnitude sets the tolerance.
    merge_tolerance : float
        Lower bound on the tolerance, used where float32 is more precise than
        it (i.e., for small ``t``).

    Returns
    -------
    tol : np.ndarray
        Element-wise tolerance, same shape as ``t``.
    """
    return np.minimum(0.5 * DT,
                      np.maximum(merge_tolerance,
                                 8 * np.spacing(np.abs(t).astype(np.float32))))


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
    t_all = np.concatenate(time).astype(np.float64)
    order = np.argsort(t_all, kind='stable')
    t_sorted = t_all[order]
    tol = _same_time_point(t_sorted[:-1], merge_tolerance)
    starts_group = np.concatenate(([True], np.diff(t_sorted) > tol))
    new_time = t_sorted[starts_group]
    # Snap every time axis onto the merged one, so that interpolating below
    # reproduces each stimulus exactly at its own sample points rather than an
    # ulp before or after them:
    snapped = np.empty_like(t_all)
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
    __slots__ = ('metadata', '_is_compressed', '__stim')

    def __init__(self, source, electrodes=None, time=None, metadata=None,
                 compress=False):
        if isinstance(metadata, dict) and 'electrodes' in metadata.keys():
            self.metadata = metadata
        else:
            self.metadata = {'electrodes': {}, 'user': metadata}
        # Flag will be flipped in the compress method:
        self._is_compressed = False
        # Extract the data and coordinates (electrodes, time) from the source:
        self._factory(source, electrodes, time, compress)

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
            'time': _time if _time is None else _time.astype(np.float32),
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
        elif isinstance(time, slice) or time == Ellipsis:
            # Return a slice of time points:
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
        # Labels are common to all subplots:
        axes[-1].figure.subplots_adjust(bottom=0.2)
        axes[-1].figure.text(0.5, 0, 'Time (ms)', va='top', ha='center')
        axes[-1].figure.text(0, 0.5, r'Amplitude ($\mu$A)', va='center',
                             ha='center', rotation='vertical')
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
                if not time.step:
                    # We can't interpolate if we don't know the step size, so
                    # the only allowed option is slice(None, None, None), which
                    # is the same as ':'
                    if time.start or time.stop:
                        raise ValueError("You must provide a step size when "
                                         "slicing the time axis.")
                else:
                    start = self.time[0] if time.start is None else time.start
                    stop = self.time[-1] if time.stop is None else time.stop
                    time = np.arange(start, stop, time.step, dtype=np.float32)
            elif time is not Ellipsis:
                # Convert to float so time is not mistaken for column index
                if np.array(time).dtype != bool:
                    time = np.float32(time)
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
        return stim

    def __add__(self, scalar):
        """Add a scalar to every data point in the stimulus"""
        return self._apply_operator(self.data, ops.add, scalar)

    def __radd__(self, scalar):
        """Add a scalar to every data point in the stimulus"""
        return self.__add__(scalar)

    def __sub__(self, scalar):
        """Subtract a scalar from every data point in the stimulus"""
        return self._apply_operator(self.data, ops.sub, scalar)

    def __rsub__(self, scalar):
        """Subtract every data point in the stimulus from a scalar"""
        return self._apply_operator(scalar, ops.sub, self.data)

    def __mul__(self, scalar):
        """Multiply every data point in the stimulus with a scalar"""
        return self._apply_operator(self.data, ops.mul, scalar)

    def __rmul__(self, scalar):
        """Multiply every data point in the stimulus with a scalar"""
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        """Divide every data point in the stimulus by a scalar"""
        return self._apply_operator(self.data, ops.truediv, scalar)

    def __neg__(self):
        """Flip the sign of every data point in the stimulus"""
        return self.__mul__(-1)

    def __rshift__(self, scalar):
        """Shift every time point in the stimulus some ms into the future"""
        if self.time is None:
            raise ValueError("Cannot shift a stimulus in time if time=None.")
        return self._apply_operator(self.time, ops.add, scalar, field='time')

    def __lshift__(self, scalar):
        """Shift every time point in the stimulus some ms into the past"""
        return self.__rshift__(-scalar)

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
                msg = (f"Time points must be strictly monotonically "
                       f"increasing: {list(stim['time'])}")
                warnings.warn(msg)
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
        """
        if self.time is None:
            return np.allclose(self.data, 0, atol=MIN_AMP)
        return np.allclose(trapezoid(self.data, x=self.time), 0, atol=MIN_AMP)

    @property
    def duration(self):
        """Stimulus duration (ms)"""
        return self.time[-1]
