""":py:class:`~pulse2percept.utils.is_strictly_increasing`, 
   :py:class:`~pulse2percept.utils.sample`, 
   :py:class:`~pulse2percept.utils.unique`, 
   :py:class:`~pulse2percept.utils.radial_mask`"""

from ._fast_array import fast_is_strictly_increasing
from ..units import as_value
import numpy as np


def is_strictly_increasing(arr, tol=1e-6):
    a = np.ascontiguousarray(arr[:-1], dtype=np.float64)
    b = np.ascontiguousarray(arr[1:], dtype=np.float64)
    return fast_is_strictly_increasing(a, b, np.float64(tol))


def sample(sequence, k=1):
    """Randomly selects ``k`` elements from a ``sequence``

    .. versionadded:: 0.8

    Parameters
    ----------
    sequence : list, tuple, np.ndarray
        A sequence like a list, a tuple, an array, etc.
    k : int or float, optional
        If an integer, the number of elements to pick
        If a float between 0 and 1, the fraction of elements to pick

    Returns
    -------
    sample : list
        List of randomly chosen elements from the sequence
    """
    sequence = np.asarray(sequence)
    if isinstance(k, float):
        k = int(k * len(sequence))
    elif not isinstance(k, int):
        raise TypeError(f'"k" must be an int or float, not {type(k)}.')
    if k < 0 or k > len(sequence):
        raise ValueError(f'"k must be smaller than {len(sequence)}.')
    idx_sample = np.arange(len(sequence))
    np.random.shuffle(idx_sample)
    return sequence[idx_sample[:k]]


def unique(a, tol=1e-6, return_index=False):
    """Find the unique elements of a sorted 1D array

    Special case of ``numpy.unique`` (array is flat, sortened) with a tolerance
    level ``tol``.

    .. versionadded:: 0.7

    Parameters
    ----------
    a : array_like
        Input array: must be sorted, and will be flattened if it is not
        already 1-D.
    tol : float, optional
        If the difference between two elements in the array is smaller than
        ``tol``, the two elements are considered equal.
    return_index : bool, optional
        If True, also return the indices of ``a`` that result in the unique
        array.

    Returns
    -------
    unique : ndarray
        The sorted unique values
    unique_indices : ndarray, optional
        The indices of the first occurrences of the unique values in the
        original array. Only provided if `return_index` is True.

    """
    result = np.unique(np.round(np.asarray(a) / tol),
                       return_index=return_index)
    if return_index:
        unique, unique_indices = result
        return tol * unique, unique_indices
    return tol * result


def radial_mask(shape, mask='gauss', sd=3):
    ny, nx = shape
    x, y = np.meshgrid(np.linspace(-1, 1, num=nx),
                       np.linspace(-1, 1, num=ny))
    rad = np.sqrt(x ** 2 + y ** 2)
    if mask == "circle":
        intensity = rad <= 1
    elif mask == "gauss":
        # 3 standard deviations by the edge of the stimulus:
        inv_var = (1.0 / sd) ** 2.0
        intensity = np.exp(-rad ** 2.0 / (2.0 * inv_var))
    else:
        raise ValueError(f'Unknown mask "{mask}". Choose either "circle" or '
                         f'"gauss".')
    return intensity


def _interp_rows(x, xp, fp):
    """Linearly interpolate every row of ``fp`` at the time points ``x``

    Vectorized equivalent of ``[np.interp(x, xp, row) for row in fp]``.
    Uses float64 and the same operation order as ``np.interp``'s C loop, so
    temporal models resolve stimulus edges identically on the simulation grid.

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
    # Vectorizing only saves one Python call per electrode, so use np.interp
    # for few rows or many time points:
    if (fp.shape[0] < 32 or x.size > 256 or xp.size < 2 or
            not np.all(np.diff(xp) > 0)):
        return np.array([np.interp(x, xp, row)
                         for row in fp]).reshape((-1, x.size))
    # Bracket index j such that xp[j] <= x < xp[j+1], as in np.interp. `j`,
    # `x0`, and `x1` are 1-D (one entry per requested time point):
    j = np.clip(np.searchsorted(xp, x, side='right') - 1, 0, xp.size - 2)
    x0, x1 = xp[j], xp[j + 1]
    # Upcast only the gathered columns, not all of `fp`:
    y0 = fp[:, j].astype(np.float64)
    y1 = fp[:, j + 1].astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = (y1 - y0) / (x1 - x0)     # slope; reused in place below
        out *= x - x0
        out += y0
        # As in np.interp, retry NaNs (infinite slopes) from the right end of
        # the interval:
        nan = np.isnan(out)
        if nan.any():
            slope = (y1 - y0) / (x1 - x0)
            out = np.where(nan, slope * (x - x1) + y1, out)
            out = np.where(np.isnan(out) & (y0 == y1), y0, out)
    # The remaining corrections select whole columns, so use 1-D masks and
    # write in place:
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


def _slice_times(sl, time, time_unit):
    """Return the time points selected by a slice of the time axis

    ``start``, ``stop``, and ``step`` are times (not column indices) and may
    be quantities: ``stim[:, 0:10:0.5]`` is the stimulus every 0.5 ms from 0
    to 10 ms.

    Parameters
    ----------
    sl : slice
        The slice to resolve.
    time : 1-D array
        The stored time axis, in ``time_unit``.
    time_unit : :py:class:`~pulse2percept.units.Unit`
        The unit the slice endpoints are read in.

    Returns
    -------
    times : 1-D array or None
        The requested time points, or None for a stepless slice.
    """
    if not sl.step:
        # We can't interpolate if we don't know the step size, so the only
        # allowed option is slice(None, None, None), i.e. ':'
        if sl.start or sl.stop:
            raise ValueError("You must provide a step size when slicing the "
                             "time axis.")
        return None
    start = as_value(sl.start, time_unit, 'time')
    stop = as_value(sl.stop, time_unit, 'time')
    step = as_value(sl.step, time_unit, 'time')
    start = time[0] if start is None else start
    stop = time[-1] if stop is None else stop
    return np.arange(start, stop, step, dtype=np.float64)
