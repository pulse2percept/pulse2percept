"""Merging the time axes of several stimuli

Kept apart from :py:mod:`~pulse2percept.stimuli.base` because
:py:class:`~pulse2percept.implants.EnsembleImplant` merges its children on the
same notion of "the same instant".
"""
import numpy as np

from ..utils.constants import DT


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
    """Merge the time axes of a collection of sources into a single one

    Sources passed together may sample different instants, or run for
    different durations. Interpolating them onto one axis is expensive, so
    identical axes are detected and returned untouched.

    Parameters
    ----------
    data : list of np.ndarray
        The data associated with each time axis.
    time : list of np.ndarray
        The time axes to merge.
    merge_tolerance : float, optional
        Two time points closer together than this (or than float32 can
        resolve at their own magnitude, whichever is coarser) are the same
        point.

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
    lengths = [len(t) for t in time]
    t_sorted, starts_group, order = unique_time_points(time, merge_tolerance)
    new_time = t_sorted[starts_group]
    # Snap every time axis onto the merged one, so that interpolating below
    # reproduces each stimulus exactly at its own sample points rather than an
    # ulp before or after them:
    snapped = np.empty_like(t_sorted)
    snapped[order] = new_time[np.cumsum(starts_group) - 1]
    new_data = []
    for t, d in zip(np.split(snapped, np.cumsum(lengths)[:-1]), data):
        # `d` is a 2-D data matrix and might have more than one row:
        new_rows = [np.interp(new_time, t, row) for row in d]
        new_rows = np.array(new_rows).reshape((-1, len(new_time)))
        new_data.append(new_rows)
    return new_data, [new_time]
