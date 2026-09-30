"""Merging the time axes of several stimuli

Also used by :py:class:`~pulse2percept.implants.EnsembleImplant`.
"""
import numpy as np

from ..utils.constants import DT


def _same_time_point(t, merge_tolerance):
    """Return the tolerance within which two time points are the same point

    Pulse trains build their time axis by accumulating a window duration, so
    the same instant can differ by a few ulps, growing with ``t``. The
    tolerance therefore scales with ``|t|``, capped at 0.5 * DT so that points
    one time step apart are never merged.

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

    Unlike ``np.unique``, points that differ only by accumulated rounding
    error are merged.

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
    """Merge the time axes of a collection of sources into a single one

    Sources may sample different instants or have different durations.
    Identical axes are returned unchanged, skipping interpolation.

    Parameters
    ----------
    data : list of np.ndarray
        The data associated with each time axis.
    time : list of np.ndarray
        The time axes to merge.
    merge_tolerance : float, optional
        Two time points closer together than this (or than the accumulated
        drift at their magnitude, whichever is coarser) are the same point.

    Returns
    -------
    data : list of np.ndarray
        The data, linearly interpolated onto the merged axis.
    time : list of one np.ndarray
        The merged axis.

    """
    t0 = time[0]
    t0_tol = None
    identical = True
    for t in time:
        # Fast path for the common case where all axes are exactly equal:
        if len(t) != len(t0):
            identical = False
            break
        if np.array_equal(t, t0):
            continue
        if t0_tol is None:
            t0_tol = _same_time_point(t0, merge_tolerance)
        # Same axis up to rounding drift? (`np.allclose` is too loose: its
        # rtol is 0.01 ms at t = 1000 ms, i.e. ten time steps.)
        if not np.all(np.abs(np.subtract(t, t0, dtype=np.float64)) <= t0_tol):
            identical = False
            break
    if identical:
        return data, [t0]
    lengths = [len(t) for t in time]
    t_sorted, starts_group, order = unique_time_points(time, merge_tolerance)
    new_time = t_sorted[starts_group]
    # Snap every time axis onto the merged one, so interpolation reproduces
    # each stimulus exactly at its own sample points:
    snapped = np.empty_like(t_sorted)
    snapped[order] = new_time[np.cumsum(starts_group) - 1]
    new_data = []
    for t, d in zip(np.split(snapped, np.cumsum(lengths)[:-1]), data):
        # `d` is a 2-D data matrix and might have more than one row:
        new_rows = [np.interp(new_time, t, row) for row in d]
        new_rows = np.array(new_rows).reshape((-1, len(new_time)))
        new_data.append(new_rows)
    return new_data, [new_time]
