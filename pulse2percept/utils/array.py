"""`is_strictly_increasing`, `sample`, `unique`, `radial_mask`"""

from ._array import fast_is_strictly_increasing
import numpy as np


def is_strictly_increasing(arr, tol=1e-6):
    a = np.ascontiguousarray(arr[:-1], dtype=np.float32)
    b = np.ascontiguousarray(arr[1:], dtype=np.float32)
    return fast_is_strictly_increasing(a, b, np.float32(tol))


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
        raise ValueError('Unknown mask "%s". Choose either "circle" or '
                         '"gauss".' % mask)
    return intensity
