""":py:class:`~pulse2percept.vision.Gaze`"""
import numpy as np

from ..units import Quantity, as_value, dva, ms
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
    positions : (n, 2) array_like
        Scene locations that fall on the fovea, in degrees of visual angle.
        Unitful values are accepted.
    time : (n,) array_like
        When each fixation begins, in milliseconds unless given as a unitful
        quantity. Must be finite and strictly increasing.

    Examples
    --------
    >>> from pulse2percept.units import dva, ms
    >>> from pulse2percept.vision import Gaze
    >>> gaze = Gaze([(0, 0), (6, 2)] * dva, time=[0, 400] * ms)
    >>> gaze.positions
    array([[0., 0.],
           [6., 2.]])

    """
    __slots__ = ('_positions', '_time', '_time_unit')

    def __init__(self, positions, time):
        positions = np.asarray(as_value(positions, dva, 'positions'),
                               dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(f"'positions' must be an (n, 2) array of (x, y) "
                             f"fixations in dva, not an array of shape "
                             f"{positions.shape}.")
        if not np.all(np.isfinite(positions)):
            raise ValueError(f"'positions' must be finite, not "
                             f"{positions.tolist()}.")
        # A quantity keeps the unit it was written in; a bare number is ms:
        unit = time.unit if isinstance(time, Quantity) else ms
        time = np.asarray(as_value(time, unit, 'time'), dtype=float).ravel()
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
        self._positions = positions
        self._time = time
        self._time_unit = unit

    def _pprint_params(self):
        """Return a dict of class attributes to pretty-print"""
        return {'positions': self.positions, 'time': self.time,
                'time_unit': self.time_unit}

    @property
    def positions(self):
        """The ``(n, 2)`` fixations, in degrees of visual angle"""
        return self._positions

    @property
    def time(self):
        """When each fixation begins, counted in ``time_unit``"""
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
