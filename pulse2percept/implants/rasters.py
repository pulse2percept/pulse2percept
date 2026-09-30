""":py:class:`~pulse2percept.implants.Raster`,
   :py:class:`~pulse2percept.implants.SequentialRaster`,
   :py:class:`~pulse2percept.implants.CheckerboardRaster`,
   :py:class:`~pulse2percept.implants.CustomRaster`"""
from abc import ABCMeta, abstractmethod
from itertools import permutations
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import numpy as np
from scipy.spatial import cKDTree

from ..units import as_value, ms, um
from ..utils import PrettyPrint
from ..utils.constants import ZORDER


def _finite(name, value):
    """Raise ValueError for NaN or inf (both pass ``<`` checks silently)"""
    if not np.all(np.isfinite(np.asarray(value, dtype=np.float64))):
        raise ValueError(f"'{name}' must be finite, not {value}.")


def _whole(name, value):
    """Return ``value`` as int; raise ValueError if it is not whole"""
    _finite(name, value)
    if int(value) != value:
        raise ValueError(f"'{name}' must be a whole number, not {value}.")
    return int(value)


def _electrode_array(implant):
    """Return the electrode array of an implant, or the array itself."""
    electrode_array = getattr(implant, 'electrode_array', implant)
    if (getattr(electrode_array, 'electrode_names', None) is None or
            getattr(electrode_array, 'coordinates', None) is None):
        raise TypeError(f"'implant' must be an Implant or an "
                        f"ElectrodeArray, not {type(implant)}.")
    return electrode_array


class Raster(PrettyPrint, metaclass=ABCMeta):
    """Abstract base class for raster patterns.

    A raster partitions electrodes into groups that take turns
    stimulating. Different groups must not be active at the same time.
    Raster timing is applied by :class:`~pulse2percept.stimuli.PulseEncoder`.

    Assigning a raster to ``implant.raster`` binds it to that implant.
    Subclasses implement :meth:`groups` and may override :meth:`bind`
    when the grouping depends on electrode geometry.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    group_dur : float, optional
        Duration of one group's slot (ms). If None, groups divide the
        shortest pulse period evenly. Otherwise, one raster sweep lasts
        ``n_groups * group_dur``.
    """
    __slots__ = ('group_dur', '_implant')

    def __init__(self, group_dur=None):
        # ms, to match pulse periods, encoder clock, and DT:
        group_dur = as_value(group_dur, ms, 'group_dur')
        if group_dur is not None:
            _finite('group_dur', group_dur)
            if group_dur <= 0:
                raise ValueError("'group_dur' must be positive.")
        self.group_dur = group_dur
        self._implant = None

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        # Omit the implant to avoid infinite recursion (Implant prints raster):
        return {'group_dur': self.group_dur, 'n_groups': self.n_groups}

    @property
    def implant(self):
        """The implant this raster is bound to, or None

        Set by assigning the raster to
        :py:attr:`~pulse2percept.implants.Implant.raster` (see
        :py:meth:`bind`).
        """
        return getattr(self, '_implant', None)

    def bind(self, implant):
        r"""Bind the raster to an implant or electrode array.

        Geometry-dependent rasters may override this method to recompute
        their grouping.

        Parameters
        ----------
        implant : :class:`~pulse2percept.implants.Implant` or \
                  :class:`~pulse2percept.implants.ElectrodeArray`
            Implant or electrode array to bind.

        Returns
        -------
        self : :class:`~pulse2percept.implants.Raster`
        """
        _electrode_array(implant)
        self._implant = implant
        return self

    def _bound(self, implant=None):
        """Return the electrode array of ``implant`` or the bound implant"""
        implant = self.implant if implant is None else implant
        if implant is None:
            raise ValueError(
                f"This {type(self).__name__} is not bound to an implant. "
                f"Assign it to 'implant.raster' first, or pass the implant "
                f"explicitly.")
        return _electrode_array(implant)

    @property
    @abstractmethod
    def n_groups(self):
        """Number of raster groups"""
        raise NotImplementedError

    @abstractmethod
    def groups(self, electrodes):
        """Assign each electrode to a raster group

        Parameters
        ----------
        electrodes : array_like
            Electrode names, in the order they appear in the stimulus.

        Returns
        -------
        group : ``(n_electrodes,)`` int array
            The group each electrode belongs to, in ``0..n_groups-1``.

        """
        raise NotImplementedError

    def members(self, electrodes, group):
        """Return the electrodes in one raster group.

        Parameters
        ----------
        electrodes : array_like
            Electrode names.
        group : int
            Group index in ``0..n_groups-1``.

        Returns
        -------
        members : array
            Electrode names belonging to ``group``.
        """
        group = _whole('group', group)
        if group < 0 or group >= self.n_groups:
            raise ValueError(f"'group' must be in 0..{self.n_groups - 1}, not "
                             f"{group}.")
        return np.asarray(electrodes)[self.groups(electrodes) == group]

    def plot(self, implant=None, annotate=None, ax=None, cmap='viridis',
             autoscale=True):
        """Plot electrodes colored by raster group.

        Parameters
        ----------
        implant : :class:`~pulse2percept.implants.Implant`, optional
            Implant to draw. If None, use the bound implant.
        annotate : bool, optional
            Write group indices on electrodes. If None, annotate arrays
            with at most 120 electrodes.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.
        cmap : str, optional
            Matplotlib colormap.
        autoscale : bool, optional
            Fit the axes to the implant.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes drawn on.
        """
        electrode_array = self._bound(implant)
        names = list(electrode_array.electrode_names)
        group = np.asarray(self.groups(names), dtype=np.int64)
        if annotate is None:
            annotate = len(names) <= 120
        if ax is None:
            ax = plt.gca()
        ax.set_aspect('equal')
        # One color per group; a single group uses the colormap center:
        spread = (np.linspace(0, 1, self.n_groups) if self.n_groups > 1
                  else np.array([0.5]))
        colors = plt.get_cmap(cmap)(spread)
        xy = electrode_array.coordinates(um)[:, :2]
        # Size circles by the closest electrode gap rather than electrode
        # shape (PointSource dots are 5 um, HexElectrodes nearly transparent).
        # 0.38 * gap keeps neighbors from overlapping:
        gap = cKDTree(xy).query(xy, k=2)[0][:, 1].min() if len(xy) > 1 else 1.0
        patches = [Circle(pos, radius=0.38 * gap, fc=colors[g],
                          ec=(0.3, 0.3, 0.3, 1), lw=0.5)
                   for pos, g in zip(xy, group)]
        if annotate:
            for pos, g in zip(xy, group):
                # White box keeps labels readable on any colormap value:
                ax.text(pos[0], pos[1], str(g), ha='center', va='center',
                        color='black', size='large',
                        bbox={'boxstyle': 'square,pad=0.1', 'ec': 'none',
                              'fc': (1, 1, 1, 0.7)},
                        zorder=ZORDER['annotate'])
        ax.add_collection(PatchCollection(patches, match_original=True,
                                          zorder=ZORDER['foreground']))
        if autoscale:
            ax.autoscale(True)
        if ax.get_xlabel() == "":
            ax.set_xlabel('x (microns)')
        if ax.get_ylabel() == "":
            ax.set_ylabel('y (microns)')
        return ax

    def slot_dur(self, period):
        """Duration (ms) of one group's slot

        Parameters
        ----------
        period : float
            Pulse period (ms). One sweep over all groups must fit into it.

        Returns
        -------
        slot_dur : float
            ``group_dur`` if one was given, else the period split evenly
            between the groups.

        """
        period = as_value(period, ms, 'period')
        if self.group_dur is not None:
            return float(self.group_dur)
        return float(period) / self.n_groups

    def offsets(self, electrodes, period):
        """Start time of each electrode's slot relative to group 0

        Parameters
        ----------
        electrodes : array_like
            Electrode names, in the order they appear in the stimulus.
        period : float
            Pulse period (ms). One sweep over all groups must fit into it.

        Returns
        -------
        offset : ``(n_electrodes,)`` float array
            Time (ms) between the start of a sweep and the start of this
            electrode's slot.

        """
        period = as_value(period, ms, 'period')
        group = np.asarray(self.groups(electrodes), dtype=np.int64)
        if group.min(initial=0) < 0 or group.max(initial=0) >= self.n_groups:
            raise ValueError(f"'groups' must be in 0..{self.n_groups - 1}.")
        dur = self.slot_dur(period)
        # Relative tolerance, since periods are rarely round (300 Hz is
        # 3.333 ms) and an exact `>` would reject the default even split:
        if self.n_groups * dur > period * (1 + 1e-9):
            raise ValueError(f"A raster of {self.n_groups} groups "
                             f"{dur:.3f} ms apart sweeps in "
                             f"{self.n_groups * dur:.3f} ms, which does not "
                             f"fit into a pulse period of {period:.3f} ms. "
                             f"Shorten 'group_dur', use fewer groups, or lower "
                             f"the pulse frequency.")
        return group * dur


class SequentialRaster(Raster):
    """Split electrodes into groups that fire one after another

    Electrodes are grouped by their order in the stimulus, which is row by row
    for an :py:class:`~pulse2percept.implants.ElectrodeGrid`. On a 6x10 array
    such as :py:class:`~pulse2percept.implants.retina.ArgusII`,
    ``SequentialRaster(6)`` puts each row in its own group (a line raster).

    .. versionadded:: 0.10.0

    Parameters
    ----------
    n_groups : int
        Number of groups to split the electrodes into.
    interleave : bool, optional
        If False (default), each group is a contiguous block of electrodes.
        If True, consecutive electrodes go to different groups, which spreads
        each group's current across the array.
    group_dur : float, optional
        See :py:class:`~pulse2percept.implants.Raster`.

    Examples
    --------
    A line raster for Argus II, one row of ten electrodes at a time:

    >>> from pulse2percept.implants import SequentialRaster
    >>> from pulse2percept.implants.retina import ArgusII
    >>> implant = ArgusII()
    >>> implant.raster = SequentialRaster(6)

    """
    __slots__ = ('_n_groups', 'interleave')

    def __init__(self, n_groups, interleave=False, group_dur=None):
        super().__init__(group_dur=group_dur)
        _finite('n_groups', n_groups)
        if int(n_groups) != n_groups or n_groups < 1:
            raise ValueError(f"'n_groups' must be a positive integer, not "
                             f"{n_groups}.")
        self._n_groups = int(n_groups)
        self.interleave = interleave

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super()._pprint_params()
        params.update({'interleave': self.interleave})
        return params

    @property
    def n_groups(self):
        """Number of raster groups"""
        return self._n_groups

    def groups(self, electrodes):
        """Assign each electrode to a raster group"""
        idx = np.arange(len(electrodes))
        if self.interleave:
            return idx % self._n_groups
        # Contiguous blocks of near-equal size:
        return idx * self._n_groups // max(1, len(electrodes))


def _reduce(w1, w2):
    """Shortest basis of the lattice spanned by ``w1`` and ``w2``

    Lagrange-Gauss reduction. Returns a basis of the same lattice where ``w1``
    is the shortest nonzero vector and ``w2`` the shortest independent one, so
    short lattice vectors have small coefficients.
    """
    w1, w2 = np.asarray(w1, dtype=float), np.asarray(w2, dtype=float)
    for _ in range(100):
        # Tolerances make exact ties (square grid: equal lengths; hex grid:
        # mu = 0.5) resolve the same way on every platform:
        if w1 @ w1 > w2 @ w2 * (1 + 1e-9):
            w1, w2 = w2, w1
        mu = np.round(np.round((w2 @ w1) / (w1 @ w1), 9))
        if mu == 0:
            break
        w2 = w2 - mu * w1
    return w1, w2


def _canonical(vectors, scale):
    """Return two lattice steps from ``vectors`` in a deterministic order

    Drops one of each +/- pair, then sorts by length and direction, so tied
    shortest gaps (4 on a square grid, 6 on a hex grid) do not depend on input
    order. The second step is None if all vectors are collinear (the
    electrodes considered so far lie on a line).
    """
    d = np.linalg.norm(vectors, axis=1)
    tol = 1e-9 * scale
    flat = np.abs(vectors[:, 1]) <= tol
    keep = (d > tol) & ((vectors[:, 1] > tol) | (flat & (vectors[:, 0] > 0)))
    vectors, d = vectors[keep], d[keep]
    if not len(vectors):
        raise NotImplementedError(
            "A checkerboard needs electrodes on a regular grid, and these are "
            "all in the same place.")
    # Sort by length relative to the shortest gap, so the order does not
    # depend on how many neighbors were queried:
    vectors = vectors[np.lexsort((-vectors[:, 1], -vectors[:, 0],
                                  np.round(d / d.min(), 9)))]
    u = vectors[0]
    # Second step must not be collinear with the first:
    cross = np.abs(u[0] * vectors[:, 1] - u[1] * vectors[:, 0])
    off_axis = np.flatnonzero(cross > 1e-6 * (u @ u))
    return u, (vectors[off_axis[0]] if len(off_axis) else None)


def _closest(xy, labels, n_groups):
    """Return the smallest distance between two electrodes of the same group

    Measured on the actual electrodes, which on a small or trimmed array can
    exceed the shortest sublattice vector. Returns inf if no group has more
    than one electrode.
    """
    closest = np.inf
    for g in range(n_groups):
        pos = xy[labels == g]
        if len(pos) > 1:
            closest = min(closest,
                          float(cKDTree(pos).query(pos, k=2)[0][:, 1].min()))
    return closest


def _combos(w1, w2, reach):
    """Every combination ``m * w1 + n * w2`` with ``|m|, |n| <= reach``"""
    steps = np.arange(-reach, reach + 1)
    m, n = np.meshgrid(steps, steps)
    return m.ravel()[:, None] * w1 + n.ravel()[:, None] * w2


def _spectrum(w1, w2, n_terms=8):
    """Lengths of the shortest nonzero vectors of the lattice, ascending

    Used to rank how widely spaced a group is. Comparing the full spectrum
    breaks ties: on a square grid with four groups, every-other-row-and-column
    puts four neighbors at the minimum distance, while the offset pattern puts
    only two there.
    """
    d = np.linalg.norm(_combos(*_reduce(w1, w2), reach=4), axis=1)
    return np.sort(d[d > 1e-9])[:n_terms]


def _min_rep(delta, w1, w2):
    """Shortest vector that differs from ``delta`` by a lattice vector

    Approximates the apparent jump of the percept from one group to the next,
    given two groups offset by ``delta`` on a periodic sublattice.
    """
    w1, w2 = _reduce(w1, w2)
    basis = np.column_stack([w1, w2])
    # Shift near the origin first so a small search suffices:
    delta = delta - basis @ np.round(np.linalg.solve(basis, delta))
    cand = delta + _combos(w1, w2, reach=2)
    d = np.linalg.norm(cand, axis=1) / np.linalg.norm(w1)
    # Break ties deterministically for a reproducible schedule:
    return cand[np.lexsort((cand[:, 1], cand[:, 0], np.round(d, 9)))[0]]


def _drift(steps, scale):
    """Return (max, sum) of net displacement over all runs of consecutive jumps

    ``steps`` holds the jump from each group to the next. Growing cumulative
    displacement means apparent motion; orders that double back keep it
    bounded. Includes the full sweep, so drift across sweeps counts too. Both
    values are in units of the electrode spacing; smaller is better.
    """
    n = steps.shape[-2]
    zero = np.zeros(steps.shape[:-2] + (1, 2))
    walk = np.concatenate([zero, np.cumsum(steps, axis=-2)], axis=-2)
    start, stop = np.triu_indices(n + 1, k=1)
    d = np.linalg.norm(walk[..., stop, :] - walk[..., start, :], axis=-1)
    d = np.round(d / scale, 6)
    return d.max(axis=-1), d.sum(axis=-1)


def _firing_order(jump, scale):
    """Return the group firing order with the least drift (see `_drift`)"""
    n = len(jump)
    if n < 3:
        return list(range(n))
    if n <= 8:
        # Exhaustive search. Group 0 is fixed first, since rotating a sweep
        # only shifts time zero:
        order = np.array([(0,) + p for p in permutations(range(1, n))])
        steps = jump[order, np.roll(order, -1, axis=1)]
        worst, total = _drift(steps, scale)
        # Stable lexsort + ordered permutations make ties deterministic:
        return order[np.lexsort((total, worst))[0]].tolist()

    def score(order):
        steps = jump[order, np.roll(order, -1)]
        return _drift(steps, scale)

    # Heuristic for n > 8: greedy order, then 2-opt segment reversals until no
    # improvement (matches the exhaustive search where both were compared):
    order = [0]
    while len(order) < n:
        rest = [g for g in range(n) if g not in order]
        order.append(min(rest, key=lambda g: score(order + [g])))
    order = np.array(order)
    for _ in range(100):
        best, before = score(order), order
        for i in range(1, n):
            for j in range(i + 1, n):
                cand = np.concatenate([order[:i], order[i:j + 1][::-1],
                                       order[j + 1:]])
                cand_score = score(cand)
                if cand_score < best:
                    order, best = cand, cand_score
        if np.array_equal(order, before):
            break
    return order.tolist()


def _lattice(xy):
    """Return integer lattice coordinates of each electrode and the basis

    Works for any regular grid (rectangular or hexagonal, any rotation).
    """
    n = len(xy)
    if n < 2:
        return np.zeros((n, 2), dtype=np.int64), np.eye(2)
    scale = float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)))
    tree = cKDTree(xy)
    # Query more neighbors until a second, non-collinear step appears (e.g.,
    # rows 1050 um apart with 100 um electrode spacing need k > 20):
    k = min(n, 9)
    while True:
        _, idx = tree.query(xy, k=k)
        diff = (xy[idx[:, 1:]] - xy[:, None, :]).reshape(-1, 2)
        u, v = _canonical(diff, scale)
        if v is not None or k == n:
            break
        k = min(n, 2 * k)
    if v is None:
        # Electrodes on a line: any perpendicular second step works:
        v = np.array([-u[1], u[0]])
    # Reduce to the shortest basis, then re-canonicalize (reduction may flip
    # signs or order):
    u, v = _canonical(_combos(*_reduce(u, v), reach=2), np.linalg.norm(u))
    basis = np.column_stack([u, v])
    ij = np.linalg.solve(basis, (xy - xy[0]).T).T
    # Negated so NaN also raises NotImplementedError:
    if not np.abs(ij - np.rint(ij)).max() <= 1e-6:
        raise NotImplementedError(
            "A checkerboard needs electrodes on a regular grid, and these do "
            "not lie on one. Use a grid implant (an ElectrodeGrid, such as "
            "ArgusII or PRIMA), or assign the groups by hand with a "
            "CustomRaster.")
    return np.rint(ij).astype(np.int64), basis


def _sublattices(ij, n_groups, balance):
    """Every way of splitting the grid into ``n_groups`` even groups

    A group is one coset of an index-``n_groups`` sublattice: every ``a``-th
    electrode along the first step and every ``d``-th along the second
    (``a * d = n_groups``), skewed by ``k``. Every such sublattice has exactly
    one (Hermite normal) form, so the enumeration is exhaustive.

    Yields ``(labels, a, d, k, biggest)`` for splits within ``balance``, where
    ``biggest`` is the largest group size (sets the peak current).
    """
    even = int(np.ceil(len(ij) / n_groups))
    for a in range(1, n_groups + 1):
        if n_groups % a:
            continue
        d = n_groups // a
        for k in range(a):
            # Coset index: position along the second step, then along the
            # first after removing the skew:
            q = np.mod(ij[:, 1], d)
            p = np.mod(ij[:, 0] - k * ((ij[:, 1] - q) // d), a)
            labels = q * a + p
            count = np.bincount(labels, minlength=n_groups)
            # Reject empty groups and groups larger than the balance allows:
            if count.min() and count.max() <= even * (1 + balance) + 1e-9:
                yield labels, a, d, k, int(count.max())


def _suggest(ij, n_groups, balance, n_show=4):
    """Return the nearest group counts that fit this grid, as a string"""
    reach = range(2, min(len(ij), 2 * n_groups + 8) + 1)
    fits = [n for n in reach
            if next(_sublattices(ij, n, balance), None) is not None]
    near = sorted(sorted(fits, key=lambda n: (abs(n - n_groups), n))[:n_show])
    return ', '.join(str(n) for n in near) if near else 'no other count'


class CheckerboardRaster(Raster):
    """Split electrodes into groups that are spread as far apart as possible

    Generalizes the checkerboard raster tested in [Kasowski2025]_, where
    spreading raster groups over the whole array outperformed horizontal,
    vertical, and random rasters in letter recognition and motion
    discrimination, and matched no rastering.

    Each raster group is a coset of a sublattice of the electrode grid, with
    electrodes as far apart as the electrode count allows. The firing order
    doubles back to reduce apparent motion. For example, five groups on a
    square grid step right, down, left, down, and back instead of sliding
    across the array.

    Supports hexagonal, rotated, anisotropic, and trimmed grids, since the
    pattern is computed from electrode positions. Arrays not on a grid raise
    ``NotImplementedError``.

    The pattern is computed when the raster is bound to an implant (e.g., by
    assigning it to :py:attr:`~pulse2percept.implants.Implant.raster`).
    Before that, :py:meth:`groups`, :py:attr:`min_spacing`, and
    :py:meth:`~pulse2percept.implants.Raster.plot` raise ValueError. Binding
    to another implant recomputes the pattern.

    .. note::

        If ``n_groups`` does not fit the grid, binding raises ``ValueError``
        and lists counts that do.

        The firing order is found by exhaustive search for up to eight
        groups and by a heuristic beyond that. Search time grows with the
        group count.

        Check :py:attr:`min_spacing`: an accepted count can still put
        neighbors in the same group. For example, two groups on a hex grid
        degenerate to a line raster, so hex implants like PRIMA need 3, 4,
        or 7 groups.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    n_groups : int
        Number of groups to split the electrodes into.
    balance : float, optional
        Allowed excess size of the largest group over an even split, as a
        fraction. The largest group sets the peak current; allowing imbalance
        can buy wider spacing on trimmed grids. With 0, only rounding
        imbalance is allowed (e.g., 378 electrodes in 5 groups: 76, 76, 75,
        75, 76).
    group_dur : float, optional
        See :py:class:`~pulse2percept.implants.Raster`.

    Examples
    --------
    Five groups of twelve on Argus II, as in [Kasowski2025]_:

    >>> from pulse2percept.implants import CheckerboardRaster
    >>> from pulse2percept.implants.retina import ArgusII
    >>> implant = ArgusII()
    >>> implant.raster = CheckerboardRaster(5)
    >>> implant.raster.n_groups
    5

    No two electrodes of a group are closer than sqrt(5) pitches, where a line
    raster would have them adjacent:

    >>> round(implant.raster.min_spacing / 575, 3)  # 575 um pitch
    2.236

    Plot the pattern with :py:meth:`~pulse2percept.implants.Raster.plot`, or
    list a group's electrodes with
    :py:meth:`~pulse2percept.implants.Raster.members`:

    >>> implant.raster.members(implant.electrode_names, 0)[:4].tolist()
    ['A1', 'A6', 'B3', 'B8']

    """
    __slots__ = ('_n_groups', '_balance', '_group_of', '_min_spacing')

    def __init__(self, n_groups, balance=0.05, group_dur=None):
        super().__init__(group_dur=group_dur)
        _finite('n_groups', n_groups)
        if int(n_groups) != n_groups or n_groups < 1:
            raise ValueError(f"'n_groups' must be a positive integer, not "
                             f"{n_groups}.")
        _finite('balance', balance)
        if balance < 0:
            raise ValueError(f"'balance' cannot be negative, not {balance}.")
        self._n_groups = int(n_groups)
        self._balance = balance
        # Set in `bind`, since grouping depends on electrode positions:
        self._group_of = None
        self._min_spacing = None

    def bind(self, implant):
        """Compute the checkerboard for this implant's electrode grid

        See :py:meth:`~pulse2percept.implants.Raster.bind`. Called
        automatically when the raster is assigned to
        :py:attr:`~pulse2percept.implants.Implant.raster`.
        """
        electrode_array = _electrode_array(implant)
        names = list(electrode_array.electrode_names)
        n_groups, balance = self._n_groups, self._balance
        # um, the unit of `min_spacing`:
        xy = electrode_array.coordinates(um)[:, :2]
        if len(xy) < n_groups:
            raise ValueError(f"{len(xy)} electrode(s) cannot be split into "
                             f"{n_groups} groups.")
        ij, basis = _lattice(xy)
        u, v = basis.T
        scale = min(np.linalg.norm(u), np.linalg.norm(v))

        # Rank splits by actual closest same-group pair, then by lattice
        # spectrum (ties, regularity), then by evenness:
        best = None
        for labels, a, d, k, biggest in _sublattices(ij, n_groups, balance):
            w1, w2 = a * u, k * u + d * v
            spacing = _closest(xy, labels, n_groups)
            key = (round(spacing / scale, 6),
                   tuple(np.round(_spectrum(w1, w2) / scale, 6)), -biggest)
            if best is None or key > best[0]:
                best = (key, labels, w1, w2, spacing)
        if best is None:
            raise ValueError(
                f"No checkerboard of {n_groups} groups fits these "
                f"{len(xy)} electrodes. A group is every a-th electrode "
                f"across by every d-th down (a * d = {n_groups}), and on this "
                f"grid every such pattern leaves one group more than "
                f"{balance:.0%} bigger than an even split. Try "
                f"{_suggest(ij, n_groups, balance)} groups instead, or raise "
                f"'balance' to allow groups of unequal size.")
        labels, w1, w2, min_spacing = best[1:]

        # Map coset labels to firing slots in least-drift order:
        first = np.array([np.flatnonzero(labels == g)[0]
                          for g in range(n_groups)])
        jump = np.zeros((n_groups, n_groups, 2))
        for g1 in range(n_groups):
            for g2 in range(n_groups):
                if g1 != g2:
                    jump[g1, g2] = _min_rep(xy[first[g2]] - xy[first[g1]],
                                            w1, w2)
        slot = np.empty(n_groups, dtype=np.int64)
        slot[_firing_order(jump, scale)] = np.arange(n_groups)

        # Bind only after success, so a failed bind keeps the previous state:
        super().bind(implant)
        self._min_spacing = min_spacing
        self._group_of = {str(name): int(slot[label])
                          for name, label in zip(names, labels)}
        return self

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super()._pprint_params()
        params.update({'balance': self._balance,
                       'min_spacing': self.min_spacing})
        return params

    @property
    def n_groups(self):
        """Number of raster groups"""
        return self._n_groups

    @property
    def min_spacing(self):
        """Distance (um) between the closest two electrodes of a group

        A line raster would give the electrode pitch. Measured on the actual
        electrodes, so small or trimmed arrays can exceed the sublattice
        spacing. Infinite if no group has more than one electrode. None until
        the raster is bound to an implant.
        """
        return self._min_spacing

    def groups(self, electrodes):
        """Assign each electrode to a raster group"""
        if self._group_of is None:
            raise ValueError(
                "This CheckerboardRaster is not bound to an implant, so it "
                "does not know where the electrodes are. Assign it to "
                "'implant.raster' first.")
        try:
            return np.array([self._group_of[str(e)] for e in electrodes])
        except KeyError:
            missing = sorted({str(e) for e in electrodes} -
                             set(self._group_of))
            raise ValueError(f"Electrode(s) {missing[:10]} are not on the "
                             f"grid this raster was bound to. Assign the "
                             f"raster to the implant the stimulus is applied "
                             f"to.")


class CustomRaster(Raster):
    """Assign electrodes to raster groups by name

    .. versionadded:: 0.10.0

    Parameters
    ----------
    groups : list of lists, or dict
        A list whose i-th element holds the electrode names in group i, or a
        dict mapping electrode names to group indices. Every electrode in the
        stimulus must be assigned to exactly one group.
    group_dur : float, optional
        See :py:class:`~pulse2percept.implants.Raster`.

    Examples
    --------
    Fire the four corners of Argus II first, then all other electrodes (every
    electrode requires a group):

    >>> from pulse2percept.implants import CustomRaster
    >>> from pulse2percept.implants.retina import ArgusII
    >>> corners = ['A1', 'A10', 'F1', 'F10']
    >>> rest = [e for e in ArgusII().electrode_names if e not in corners]
    >>> raster = CustomRaster([corners, rest])
    >>> raster.n_groups
    2

    """
    __slots__ = ('_group_of', '_n_groups')

    def __init__(self, groups, group_dur=None):
        super().__init__(group_dur=group_dur)
        if isinstance(groups, dict):
            group_of = {str(k): _whole(f'group of {k}', v)
                        for k, v in groups.items()}
        else:
            group_of = {}
            for idx, names in enumerate(groups):
                if isinstance(names, str):
                    raise TypeError(f"Group {idx} must be a list of electrode "
                                    f"names, not the string '{names}'.")
                for name in names:
                    name = str(name)
                    # An electrode in two groups would fire in both slots:
                    if name in group_of:
                        raise ValueError(
                            f"Electrode '{name}' is in group "
                            f"{group_of[name]} and group {idx}. Every "
                            f"electrode belongs to exactly one group.")
                    group_of[name] = idx
        if not group_of:
            raise ValueError("'groups' cannot be empty.")
        if min(group_of.values()) < 0:
            raise ValueError("Group indices cannot be negative.")
        self._group_of = group_of
        self._n_groups = max(group_of.values()) + 1

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super()._pprint_params()
        params.update({'n_electrodes': len(self._group_of)})
        return params

    @property
    def n_groups(self):
        """Number of raster groups"""
        return self._n_groups

    def groups(self, electrodes):
        """Assign each electrode to a raster group"""
        try:
            return np.array([self._group_of[str(e)] for e in electrodes])
        except KeyError:
            missing = sorted({str(e) for e in electrodes} -
                             set(self._group_of))
            raise ValueError(f"No raster group given for electrode(s) "
                             f"{missing[:10]}. Every electrode in the "
                             f"stimulus must be assigned to a group.")
