""":py:class:`~pulse2percept.vision.Gaze`,
:py:class:`~pulse2percept.vision.FixationalMotion`"""
import numpy as np
from scipy.signal import lfilter

from ..units import DimensionMismatchError, Quantity, as_value, dva, ms
from ..utils import PrettyPrint


class FixationalMotion(PrettyPrint):
    """Involuntary drift and microsaccades around each fixation of a Gaze

    The offset is zero at fixation onset, generated on a 10 ms grid, and
    linearly interpolated. It depends only on ``seed``, fixation index, and
    time since onset, not on frame rate or evaluation order.

    Drift: Ornstein-Uhlenbeck process per axis (correlation time 500 ms,
    stationary SD 0.02 dva). Microsaccades: mean rate 1/s, at least 150 ms
    apart and after onset; each a 0.1-0.3 dva jump within one 10 ms step,
    landing within 0.3 dva of the target if possible. Tremor is not modeled.

    The defaults are illustrative, informed by published ranges of human
    fixational eye movements [OteroMillan2014]_, [Rucci2015]_. They are not
    fitted to measured trajectories and should not be treated as a validated
    physiological model.

    .. versionadded:: 0.12.0

    Parameters
    ----------
    drift : bool, optional
        Whether to generate drift.
    microsaccades : bool, optional
        Whether to generate microsaccades.
    seed : int, optional
        Non-negative random seed. None draws one at construction and stores
        it in ``seed``.

    Examples
    --------
    >>> from pulse2percept.units import dva, ms
    >>> from pulse2percept.vision import FixationalMotion, Gaze
    >>> gaze = Gaze([(0, 0), (6, 2)] * dva, time=[0, 400] * ms,
    ...             fixational_motion=FixationalMotion(seed=42))
    >>> gaze.positions  # unchanged
    array([[0., 0.],
           [6., 2.]])

    """
    __slots__ = ('_drift', '_microsaccades', '_seed')

    # Times in ms, distances in dva:
    _step = 10.0
    _drift_tau = 500.0
    _drift_sd = 0.02
    _msacc_interval = 1000.0  # mean onset-to-onset interval
    _msacc_refractory = 150.0
    _msacc_amplitude = (0.1, 0.3)
    _msacc_reach = 0.3

    def __init__(self, drift=True, microsaccades=True, seed=None):
        for name, value in (('drift', drift),
                            ('microsaccades', microsaccades)):
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"'{name}' must be True or False, not "
                                f"{value!r}.")
        if seed is None:
            seed = np.random.SeedSequence().entropy
        elif (isinstance(seed, (bool, np.bool_)) or
              not isinstance(seed, (int, np.integer)) or seed < 0):
            raise ValueError(f"'seed' must be a non-negative integer or None, "
                             f"not {seed!r}.")
        self._drift = bool(drift)
        self._microsaccades = bool(microsaccades)
        self._seed = int(seed)

    def _pprint_params(self):
        """Return a dict of class attributes to pretty-print"""
        return {'drift': self.drift, 'microsaccades': self.microsaccades,
                'seed': self.seed}

    @property
    def drift(self):
        """Whether drift is generated"""
        return self._drift

    @property
    def microsaccades(self):
        """Whether microsaccades are generated"""
        return self._microsaccades

    @property
    def seed(self):
        """Random seed; fixed at construction"""
        return self._seed

    def _offsets(self, fixation, elapsed):
        """Returns ``(n, 2)`` offsets in dva, ``elapsed`` ms after the onset
        of each ``fixation`` (an index into the Gaze events)"""
        offsets = np.zeros((elapsed.size, 2))
        if not (self.drift or self.microsaccades):
            return offsets
        for index in np.unique(fixation):
            here = fixation == index
            segment = self._segment(int(index), elapsed[here].max())
            steps = elapsed[here] / self._step
            grid = np.arange(len(segment))
            for axis in range(2):
                offsets[here, axis] = np.interp(steps, grid, segment[:, axis])
        return offsets

    def _segment(self, fixation, duration):
        """Returns ``(n, 2)`` offsets in dva on the internal grid, from onset
        of ``fixation`` through ``duration`` ms"""
        n = int(np.ceil(duration / self._step)) + 1
        offsets = np.zeros((n, 2))
        # One stream per component, so toggling one keeps the other:
        if self.drift:
            rng = np.random.default_rng(
                np.random.SeedSequence(self.seed, spawn_key=(fixation, 0)))
            # Drawn in order, so a longer segment extends a shorter one:
            noise = rng.standard_normal((n, 2))
            noise[0] = 0  # zero offset at onset
            decay = np.exp(-self._step / self._drift_tau)
            gain = self._drift_sd * np.sqrt(1 - decay ** 2)
            offsets += lfilter([gain], [1, -decay], noise, axis=0)
        if self.microsaccades:
            rng = np.random.default_rng(
                np.random.SeedSequence(self.seed, spawn_key=(fixation, 1)))
            low, high = self._msacc_amplitude
            onset = 0.0
            while True:
                # Fixed draws per event, so events ignore `duration`:
                gap, amplitude, direction = rng.random(3)
                # Refractory period plus an exponential wait:
                onset += (self._msacc_refractory -
                          (self._msacc_interval - self._msacc_refractory) *
                          np.log1p(-gap))
                k = int(onset // self._step) + 1
                if k >= n:
                    break
                offsets[k:] += _microsaccade(offsets[k - 1],
                                             low + (high - low) * amplitude,
                                             direction, self._msacc_reach)
        return offsets


def _microsaccade(start, amplitude, u, reach):
    """Returns the ``(dx, dy)`` jump of a microsaccade from ``start``

    Direction is uniform (``u`` in [0, 1)) over landings within ``reach`` of
    the target; if there are none, the jump aims at the target.
    """
    r = np.hypot(*start)
    if r == 0:
        theta = 2 * np.pi * u
    else:
        # Law of cosines: the landing is within reach iff the angle from
        # `start`'s direction is at least arccos(c):
        c = (reach ** 2 - r ** 2 - amplitude ** 2) / (2 * amplitude * r)
        lo = np.arccos(np.clip(c, -1, 1))
        theta = np.arctan2(start[1], start[0]) + lo + u * 2 * (np.pi - lo)
    return amplitude * np.array([np.cos(theta), np.sin(theta)])


class Gaze(PrettyPrint):
    """A sequence of fixations, each starting at a given time

    Gaze is piecewise constant: position ``i`` is active from ``time[i]``
    until ``time[i + 1]``; the last fixation is held indefinitely. Saccades
    are instantaneous (no interpolation). An event exactly on a frame time
    applies to that frame; times before the first event are a ValueError.
    ``fixational_motion`` adds involuntary motion around each fixation.

    Accepted wherever :py:class:`~pulse2percept.vision.Scene` and
    :py:meth:`~pulse2percept.models.Model.predict_percept` take a ``gaze``,
    if there are frame times to resolve against: a video scene's, or a timed
    percept's for a still scene.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    positions : (n, 2) or (n, 3) array_like
        Scene locations on the fovea, in dva. Unitful values are accepted.
        Without ``time``, each row is ``(x, y, time)``: x and y in dva, time
        in ms, unless unitful. Each entry is converted separately, so a bare
        0 needs no unit.
    time : (n,) array_like, optional
        Start time of each fixation, in ms unless unitful. Must be finite and
        strictly increasing. Required for ``(n, 2)`` positions; not allowed
        for ``(x, y, time)`` rows.
    fixational_motion : :py:class:`~pulse2percept.vision.FixationalMotion`, optional
        Drift and microsaccades around each fixation. ``positions`` and
        ``time`` keep the intended fixations.

        .. versionadded:: 0.12.0

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
    __slots__ = ('_positions', '_time', '_time_unit', '_fixational_motion')

    def __init__(self, positions, time=None, fixational_motion=None):
        if not isinstance(fixational_motion, (FixationalMotion, type(None))):
            raise TypeError(f"'fixational_motion' must be a FixationalMotion "
                            f"or None, not {type(fixational_motion)}.")
        if time is None:
            positions, time = _split_rows(positions)
        elif _row_width(positions) == 3:
            raise ValueError("'positions' has (x, y, time) rows and 'time' "
                             "is given too. Pass one or the other.")
        # Copy, then make read-only below, so caller mutations have no effect:
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
        # Keep a Quantity's unit; bare numbers are ms:
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
            # Equal timestamps would make `searchsorted` ambiguous:
            raise ValueError(f"'time' must be strictly increasing, not "
                             f"{time.tolist()}.")
        positions.flags.writeable = False
        time.flags.writeable = False
        self._positions = positions
        self._time = time
        self._time_unit = unit
        self._fixational_motion = fixational_motion

    def _pprint_params(self):
        """Return a dict of class attributes to pretty-print"""
        return {'positions': self.positions, 'time': self.time,
                'time_unit': self.time_unit,
                'fixational_motion': self.fixational_motion}

    @property
    def positions(self):
        """``(n, 2)`` fixations in dva; read-only"""
        return self._positions

    @property
    def time(self):
        """Fixation start times in ``time_unit``; read-only"""
        return self._time

    @property
    def time_unit(self):
        """Unit of ``time``"""
        return self._time_unit

    @property
    def fixational_motion(self):
        """:py:class:`~pulse2percept.vision.FixationalMotion`, or None"""
        return self._fixational_motion

    def _at(self, time, time_unit=None):
        """Returns the active fixation plus any fixational motion at each
        ``time``, as ``(n, 2)`` in dva

        ``time_unit`` is the unit of ``time``; None uses ``self.time_unit``.
        """
        unit = self.time_unit if time_unit is None else time_unit
        events = np.asarray(as_value(Quantity(self.time, self.time_unit),
                                     unit, 'time'), dtype=float)
        time = np.asarray(time, dtype=float).ravel()
        # 'right', so an event exactly on a frame time applies to that frame:
        idx = np.searchsorted(events, time, side='right') - 1
        if idx.size and idx.min() < 0:
            first = time.min()
            raise ValueError(
                f"Gaze is undefined before its first event at "
                f"{events[0]:g} {unit}, and this asks for {first:g} {unit}. "
                f"Start the trajectory at or before the first frame.")
        if self.fixational_motion is None:
            return self.positions[idx]
        elapsed = np.asarray(as_value(Quantity(time - events[idx], unit), ms,
                                      'time'), dtype=float)
        return (self.positions[idx] +
                self.fixational_motion._offsets(idx, elapsed))


def _row_width(rows):
    """Returns the common row length of ``rows``, or None"""
    if isinstance(rows, Quantity):
        rows = as_value(rows, rows.unit)
    try:
        widths = {len(row) for row in rows}
    except TypeError:
        return None
    return widths.pop() if len(widths) == 1 else None


def _split_rows(rows):
    """Splits ``(x, y, time)`` rows into dva positions and ms timestamps"""
    width = _row_width(rows)
    if width == 2:
        raise ValueError("(x, y) positions require 'time'. Pass time=..., "
                         "or give (x, y, time) rows.")
    if width != 3:
        raise ValueError("Without 'time', 'positions' must be (x, y, time) "
                         "rows, at least one of them.")
    if isinstance(rows, Quantity):
        # A single unit cannot be both dva and time:
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
    """Returns gaze as one (x, y) in dva, or one per frame

    A :py:class:`~pulse2percept.vision.Gaze` is resolved at ``time`` (in
    ``time_unit``).
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
        # Otherwise the interpolator would return a blank percept:
        raise ValueError(f"'gaze' must be finite, not {gaze.tolist()}.")
    return gaze
