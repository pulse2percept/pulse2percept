""":py:class:`~pulse2percept.vision.Gaze`"""
import numpy as np

from ..units import DimensionMismatchError, Quantity, as_value, dva, ms
from ..utils import PrettyPrint


class Gaze(PrettyPrint):
    """A sparse sequence of fixations, each starting at a given time

    Gaze is piecewise constant: position ``i`` becomes active at ``time[i]``
    and is held until ``time[i + 1]``; the last fixation is held indefinitely.
    Nothing is interpolated, so a saccade takes no time. Resolution is
    right-continuous, so an event landing exactly on a frame time applies to
    that frame; times before the first event raise.

    Accepted wherever :py:class:`~pulse2percept.vision.Scene` and
    :py:meth:`~pulse2percept.models.Model.predict_percept` take a ``gaze``,
    given a clock to resolve against: a video scene's frame times, or a timed
    percept's for a still scene.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    positions : (n, 2) or (n, 3) array_like
        Scene locations that fall on the fovea, in degrees of visual angle.
        Unitful values are accepted. Without ``time``, each row is
        ``(x, y, time)``: x and y in dva, time in ms, unless unitful. Each
        entry converts on its own, so a bare 0 needs no unit.
    time : (n,) array_like, optional
        When each fixation begins, in milliseconds unless given as a unitful
        time. Must be finite and strictly increasing. Required for ``(n, 2)``
        positions; not allowed for ``(x, y, time)`` rows.

    Examples
    --------
    >>> from pulse2percept.units import dva, ms, s
    >>> from pulse2percept.vision import Gaze
    >>> gaze = Gaze([(0, 0), (6, 2)] * dva, time=[0, 400] * ms)
    >>> gaze.positions
    array([[0., 0.],
           [6., 2.]])
    >>> gaze = Gaze([(0, 0, 0), (6 * dva, 2 * dva, 0.4 * s)])
    >>> gaze.time
    array([  0., 400.])

    """
    __slots__ = ('_positions', '_time', '_time_unit')

    def __init__(self, positions, time=None):
        if time is None:
            positions, time = _split_rows(positions)
        elif _row_width(positions) == 3:
            raise ValueError("'positions' has (x, y, time) rows and 'time' "
                             "is given too. Pass one or the other.")
        # Copied, then frozen below: an array the caller can still mutate
        # would silently change gaze that has already been resolved.
        positions = np.array(as_value(positions, dva, 'positions'),
                             dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(f"'positions' must be an (n, 2) array of (x, y) "
                             f"fixations in dva, not an array of shape "
                             f"{positions.shape}.")
        if positions.shape[0] == 0:
            raise ValueError("'positions' must hold at least one fixation; "
                             "an empty trajectory says nothing about where "
                             "the eye was pointing.")
        if not np.all(np.isfinite(positions)):
            raise ValueError(f"'positions' must be finite, not "
                             f"{positions.tolist()}.")
        # A quantity keeps the unit it was written in; a bare number is ms:
        unit = time.unit if isinstance(time, Quantity) else ms
        if unit.dimension != ms.dimension:
            raise DimensionMismatchError(
                f"'time' must be counted in a unit of time (e.g. ms, s), not "
                f"{unit.dimension.name} ({unit}).")
        time = np.array(as_value(time, unit, 'time'), dtype=float).ravel()
        if time.size != positions.shape[0]:
            raise ValueError(f"'time' needs one timestamp per fixation, so "
                             f"{positions.shape[0]} of them, not {time.size}.")
        if not np.all(np.isfinite(time)):
            raise ValueError(f"'time' must be finite, not {time.tolist()}.")
        if time.size > 1 and np.any(np.diff(time) <= 0):
            # Equal timestamps leave `searchsorted` to pick between two
            # fixations:
            raise ValueError(f"'time' must be strictly increasing, not "
                             f"{time.tolist()}.")
        positions.flags.writeable = False
        time.flags.writeable = False
        self._positions = positions
        self._time = time
        self._time_unit = unit

    def _pprint_params(self):
        """Return a dict of class attributes to pretty-print"""
        return {'positions': self.positions, 'time': self.time,
                'time_unit': self.time_unit}

    @property
    def positions(self):
        """The ``(n, 2)`` fixations, in dva; read-only"""
        return self._positions

    @property
    def time(self):
        """When each fixation begins, counted in ``time_unit``; read-only"""
        return self._time

    @property
    def time_unit(self):
        """The unit ``time`` is counted in"""
        return self._time_unit

    def _at(self, time, time_unit=None):
        """Fixations active at each instant of ``time``, as ``(n, 2)``

        ``time_unit`` is the unit ``time`` is counted in; None means this
        object's own.
        """
        unit = self.time_unit if time_unit is None else time_unit
        events = np.asarray(as_value(Quantity(self.time, self.time_unit),
                                     unit, 'time'), dtype=float)
        time = np.asarray(time, dtype=float).ravel()
        # 'right' so an event landing exactly on a frame already applies to it:
        idx = np.searchsorted(events, time, side='right') - 1
        if idx.size and idx.min() < 0:
            first = time.min()
            raise ValueError(
                f"Gaze is undefined before its first event at "
                f"{events[0]:g} {unit}, and this asks for {first:g} {unit}. "
                f"Start the trajectory at or before the first frame.")
        return self.positions[idx]


def _row_width(rows):
    """Common length of the rows of ``rows``, or None if there is none"""
    if isinstance(rows, Quantity):
        rows = as_value(rows, rows.unit)
    try:
        widths = {len(row) for row in rows}
    except TypeError:
        return None
    return widths.pop() if len(widths) == 1 else None


def _split_rows(rows):
    """Split ``(x, y, time)`` rows into dva positions and ms timestamps"""
    width = _row_width(rows)
    if width == 2:
        raise ValueError("(x, y) positions require 'time'. Pass time=..., "
                         "or give (x, y, time) rows.")
    if width != 3:
        raise ValueError("Without 'time', 'positions' must be (x, y, time) "
                         "rows, at least one of them.")
    if isinstance(rows, Quantity):
        # One unit cannot be both visual angle and time:
        raise DimensionMismatchError(
            f"(x, y, time) rows carry one unit ({rows.unit}) for all three "
            f"columns. Give each entry its own unit, e.g. "
            f"(x * dva, y * dva, t * ms).")
    rows = [tuple(row) for row in rows]
    positions = [(as_value(x, dva, 'positions'), as_value(y, dva, 'positions'))
                 for x, y, _ in rows]
    time = [as_value(t, ms, 'time') for _, _, t in rows]
    return positions, time


def _gaze_points(gaze, n_frames, time=None, time_unit=None):
    """Gaze as one (x, y) in dva, or one per frame

    A :py:class:`~pulse2percept.vision.Gaze` is resolved onto ``time``,
    counted in ``time_unit``.
    """
    if gaze is None:
        return np.zeros((1, 2))
    if isinstance(gaze, Gaze):
        if time is None:
            raise ValueError(
                "A timestamped Gaze needs a clock to resolve against, and "
                "there is no time axis here. Pass a single (x, y) gaze "
                "instead.")
        return gaze._at(time, time_unit)
    gaze = np.atleast_2d(np.asarray(as_value(gaze, dva, 'gaze'), dtype=float))
    if gaze.shape not in {(1, 2), (n_frames, 2)}:
        raise ValueError(f"'gaze' must be an (x, y) pair in dva, or one per "
                         f"frame ({n_frames} of them), not an array of shape "
                         f"{gaze.shape}.")
    if not np.all(np.isfinite(gaze)):
        # Left to reach the interpolator, this would come back as a blank
        # percept rather than as a question about where the eye was pointing:
        raise ValueError(f"'gaze' must be finite, not {gaze.tolist()}.")
    return gaze
