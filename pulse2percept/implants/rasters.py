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
    """Reject NaN and infinity, which slip through every ``<`` comparison"""
    if not np.all(np.isfinite(np.asarray(value, dtype=np.float64))):
        raise ValueError(f"'{name}' must be finite, not {value}.")


def _whole(name, value):
    """Reject a non-integer count or index, which would silently truncate"""
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
    Raster timing is applied by :class:`~pulse2percept.stimuli.StimulusEncoder`.

    Assigning a raster to ``implant.raster`` binds it to that implant.
    Subclasses implement :meth:`groups` and may override :meth:`bind`
    when the grouping depends on electrode geometry.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    group_dur : float, optional
        Duration of one group's slot (ms). If None, groups divide the
        shortest pulse period evenly. An explicit value fixes the
        raster sweep to ``n_groups * group_dur``.
    """
    __slots__ = ('group_dur', '_implant')

    def __init__(self, group_dur=None):
        # A slot is a duration, and it is combined with pulse periods, the
        # encoder's clock and DT -- all in ms -- further down:
        group_dur = as_value(group_dur, ms, 'group_dur')
        if group_dur is not None:
            _finite('group_dur', group_dur)
            if group_dur <= 0:
                raise ValueError("'group_dur' must be positive.")
        self.group_dur = group_dur
        self._implant = None

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        # Deliberately without the implant: an Implant pretty-prints
        # its raster, so naming it back here would recurse.
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
        """The implant to answer a question about, bound or given"""
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
        group : (n_electrodes,) int array
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
        # One color per group, spread over the colormap. A single group would
        # otherwise sit at the very end of it:
        spread = (np.linspace(0, 1, self.n_groups) if self.n_groups > 1
                  else np.array([0.5]))
        colors = plt.get_cmap(cmap)(spread)
        # Microns, which is what the axis labels below say and what the patch
        # radii are sized in:
        xy = electrode_array.coordinates(um)[:, :2]
        # Sized by the array rather than by what each electrode reports, since
        # neither of the two shapes an implant is usually built from would show
        # its color: a PointSource is a 5 um dot however far apart they are,
        # and a HexElectrode is drawn nearly transparent. Just short of half
        # the closest gap, so that neighbors nearly touch and never overlap:
        gap = cKDTree(xy).query(xy, k=2)[0][:, 1].min() if len(xy) > 1 else 1.0
        patches = [Circle(pos, radius=0.38 * gap, fc=colors[g],
                          ec=(0.3, 0.3, 0.3, 1), lw=0.5)
                   for pos, g in zip(xy, group)]
        if annotate:
            for pos, g in zip(xy, group):
                # A white backing, since a group index has to stay readable
                # against both ends of the colormap:
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
            The pulse period (ms) a sweep has to fit into, so that every group
            gets its turn before the first one comes round again.

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
        """How far behind group 0 each electrode's slot begins

        Parameters
        ----------
        electrodes : array_like
            Electrode names, in the order they appear in the stimulus.
        period : float
            The pulse period (ms) a sweep has to fit into.

        Returns
        -------
        offset : (n_electrodes,) float array
            Time (ms) between the start of a sweep and the start of this
            electrode's slot.

        """
        period = as_value(period, ms, 'period')
        group = np.asarray(self.groups(electrodes), dtype=np.int64)
        if group.min(initial=0) < 0 or group.max(initial=0) >= self.n_groups:
            raise ValueError(f"'groups' must be in 0..{self.n_groups - 1}.")
        dur = self.slot_dur(period)
        # A tick of slack: the period is generally not a round number of ms
        # (a 300 Hz period is 3.333... ms), so an exact `>` would reject the
        # even split this class computes itself:
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

    Electrodes are assigned to groups by their position in the stimulus, which
    for an :py:class:`~pulse2percept.implants.ElectrodeGrid` runs row by row.
    So on a 6x10 array such as :py:class:`~pulse2percept.implants.ArgusII`,
    ``SequentialRaster(6)`` puts each row in its own group -- a line raster.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    n_groups : int
        Number of groups to split the electrodes into.
    interleave : bool, optional
        If False (the default), each group is a contiguous block of
        electrodes. If True, groups are interleaved, so that consecutive
        electrodes end up in different groups. Interleaving spreads each
        group's current further across the array.
    group_dur : float, optional
        See :py:class:`~pulse2percept.implants.Raster`.

    Examples
    --------
    A line raster for Argus II, one row of ten electrodes at a time:

    >>> from pulse2percept.implants import ArgusII, SequentialRaster
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
        # Contiguous blocks, as evenly sized as the electrode count allows:
        return idx * self._n_groups // max(1, len(electrodes))


def _reduce(w1, w2):
    """Shortest basis of the lattice spanned by ``w1`` and ``w2``

    Lagrange-Gauss reduction: repeatedly subtract the one vector from the other
    until neither can be shortened. What comes back spans the same lattice, but
    ``w1`` is now its shortest nonzero vector and ``w2`` the shortest one
    independent of it -- so every short vector of the lattice is a combination
    with small coefficients, and the searches below only have to look a few
    steps in each direction.
    """
    w1, w2 = np.asarray(w1, dtype=float), np.asarray(w2, dtype=float)
    for _ in range(100):
        # Compared with room to spare, and rounded off a step short of the
        # halfway mark, because both decisions are otherwise made on the last
        # bit of a float: on a square grid the two steps are exactly as long as
        # each other, and on a hexagonal one they lean on each other by exactly
        # half a step. Which way an exact comparison then falls is a matter of
        # how the positions were arrived at, and differs between platforms.
        if w1 @ w1 > w2 @ w2 * (1 + 1e-9):
            w1, w2 = w2, w1
        mu = np.round(np.round((w2 @ w1) / (w1 @ w1), 9))
        if mu == 0:
            break
        w2 = w2 - mu * w1
    return w1, w2


def _canonical(vectors, scale):
    """Two lattice steps picked out of ``vectors`` by a rule, not by position

    A grid has four shortest gaps, or six on a hexagonal one, all exactly as
    long as each other, and every gap turns up with both signs. Which of them
    is met first is an accident of how they were collected, so taking the
    first would let the same implant come out mirrored on one machine and not
    on another. Instead one of each +/- pair is dropped, and the rest are
    ordered by length and then by direction, which leaves nothing to chance.

    The second step comes back as None when every vector given points the same
    way, which is for the caller to make sense of: it means the electrodes
    considered so far are in a line, not that the array is.
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
    # Counted in units of the shortest gap, so that the order they come out in
    # cannot depend on how wide a net was cast to find them:
    vectors = vectors[np.lexsort((-vectors[:, 1], -vectors[:, 0],
                                  np.round(d / d.min(), 9)))]
    u = vectors[0]
    # The second step has to leave the line the first one traces out:
    cross = np.abs(u[0] * vectors[:, 1] - u[1] * vectors[:, 0])
    off_axis = np.flatnonzero(cross > 1e-6 * (u @ u))
    return u, (vectors[off_axis[0]] if len(off_axis) else None)


def _closest(xy, labels, n_groups):
    """Distance between the closest two electrodes that ever fire together

    A pattern is judged by the closest pair it actually leaves active at the
    same time, which is not the shortest step of the lattice it was cut from:
    the implant is a finite piece of that lattice, and on a small or trimmed
    one the electrodes that would have been closest need not be there at all.
    Infinite when no group holds more than one electrode, since then there is
    no pair to keep apart.
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

    This is what "maximally spaced" is measured by. Two electrodes of the same
    group are always some lattice vector apart, so the shortest vector is the
    closest two simultaneously active electrodes ever come, the next one is the
    second-closest, and so on. Comparing whole spectra rather than just the
    first entry settles the frequent ties: on a square grid, four groups can be
    laid out as every other row and column, which puts *four* neighbours at the
    minimum distance, or in the offset pattern this picks, which puts only two
    there and the rest further out.
    """
    d = np.linalg.norm(_combos(*_reduce(w1, w2), reach=4), axis=1)
    return np.sort(d[d > 1e-9])[:n_terms]


def _min_rep(delta, w1, w2):
    """Shortest vector that differs from ``delta`` by a lattice vector

    Two groups are the same pattern displaced by ``delta``, but the pattern
    repeats with the lattice, so the eye is free to match any electrode of the
    first group with any electrode of the second. What it sees is the shortest
    of those matches, which is what this returns -- the jump the percept
    appears to make when one group hands over to the next.
    """
    w1, w2 = _reduce(w1, w2)
    basis = np.column_stack([w1, w2])
    # Land in the neighborhood of the origin first, so a short search finishes
    # the job whatever `delta` came in as:
    delta = delta - basis @ np.round(np.linalg.solve(basis, delta))
    cand = delta + _combos(w1, w2, reach=2)
    d = np.linalg.norm(cand, axis=1) / np.linalg.norm(w1)
    # Ties are common and are genuinely ambiguous percepts; break them the same
    # way every time so that the schedule is reproducible:
    return cand[np.lexsort((cand[:, 1], cand[:, 0], np.round(d, 9)))[0]]


def _drift(steps, scale):
    """How far the percept wanders over every run of consecutive jumps

    ``steps`` holds the jump from each group to the next, so a run of them adds
    up to the displacement the pattern accumulates over that stretch of the
    sweep. A raster that shifts one electrode over and over is exactly the case
    where those sums keep growing, which is the apparent motion the pattern is
    there to avoid; one that doubles back keeps them bounded. The last run is
    the whole sweep, so drift that survives from one sweep to the next is
    counted too.

    Returns the worst run and the total over all runs, both in units of the
    electrode spacing. Smaller is better on both counts.
    """
    n = steps.shape[-2]
    zero = np.zeros(steps.shape[:-2] + (1, 2))
    walk = np.concatenate([zero, np.cumsum(steps, axis=-2)], axis=-2)
    start, stop = np.triu_indices(n + 1, k=1)
    d = np.linalg.norm(walk[..., stop, :] - walk[..., start, :], axis=-1)
    d = np.round(d / scale, 6)
    return d.max(axis=-1), d.sum(axis=-1)


def _firing_order(jump, scale):
    """The order to fire the groups in, as a list of group labels

    Any order fires every group exactly once, so they all obey the current
    limit equally; what differs is what the sequence looks like. This picks the
    one whose percept wanders least (see :py:func:`_drift`).
    """
    n = len(jump)
    if n < 3:
        return list(range(n))
    if n <= 8:
        # Small enough to settle exactly. The first group is fixed, since
        # rotating a sweep only moves the origin of time:
        order = np.array([(0,) + p for p in permutations(range(1, n))])
        steps = jump[order, np.roll(order, -1, axis=1)]
        worst, total = _drift(steps, scale)
        # `lexsort` is stable, so among equally good orders this takes the
        # first, and `permutations` yields them in a fixed order:
        return order[np.lexsort((total, worst))[0]].tolist()

    def score(order):
        steps = jump[order, np.roll(order, -1)]
        return _drift(steps, scale)

    # Too many orders to enumerate, so take the groups one at a time and then
    # improve on that by reversing stretches until nothing helps. This lands on
    # the same answer as the exhaustive search wherever both can be run:
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
    """Integer coordinates of each electrode, and the two steps they count off

    Every regular grid (rectangular or hexagonal, at any rotation) is a
    lattice: two steps that reach every electrode by whole numbers of each.
    """
    n = len(xy)
    if n < 2:
        return np.zeros((n, 2), dtype=np.int64), np.eye(2)
    scale = float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)))
    tree = cKDTree(xy)
    # The lattice steps are among the shortest gaps between electrodes, and a
    # handful of nearest neighbors usually turns both of them up. Usually, but
    # not always: rows 1050 um apart with electrodes 100 um along them put
    # twenty gaps in the near neighborhood of every electrode before the step
    # to the next row appears. So the net is cast wider until the second step
    # shows up, and only an array that is a line all the way out has none.
    k = min(n, 9)
    while True:
        _, idx = tree.query(xy, k=k)
        diff = (xy[idx[:, 1:]] - xy[:, None, :]).reshape(-1, 2)
        u, v = _canonical(diff, scale)
        if v is not None or k == n:
            break
        k = min(n, 2 * k)
    if v is None:
        # Electrodes on a single line: any second step will do, since nothing
        # is ever placed along it.
        v = np.array([-u[1], u[0]])
    # The gaps that were seen need not be the shortest two the lattice has, so
    # they are reduced onto those -- and then settled by the rule a second
    # time, since the reduction is free to hand them back either way round:
    u, v = _canonical(_combos(*_reduce(u, v), reach=2), np.linalg.norm(u))
    basis = np.column_stack([u, v])
    ij = np.linalg.solve(basis, (xy - xy[0]).T).T
    # Negated so that a NaN, which fails every comparison, raises rather than
    # slipping through as a grid:
    if not np.abs(ij - np.rint(ij)).max() <= 1e-6:
        raise NotImplementedError(
            "A checkerboard needs electrodes on a regular grid, and these do "
            "not lie on one. Use a grid implant (an ElectrodeGrid, such as "
            "ArgusII or PRIMA), or assign the groups by hand with a "
            "CustomRaster.")
    return np.rint(ij).astype(np.int64), basis


def _sublattices(ij, n_groups, balance):
    """Every way of splitting the grid into ``n_groups`` even groups

    A group is one coset of a sublattice of index ``n_groups``: take every
    ``a``-th electrode along the first step and every ``d``-th along the
    second, with ``a * d = n_groups``, and skew the two against each other by
    ``k``. That enumeration is exhaustive -- every index-``n_groups``
    sublattice has exactly one such (Hermite) form -- so the best pattern is
    the best of these and there is nothing else to look for.

    Yields ``(labels, a, d, k, biggest)`` for the splits that come out even
    enough, biggest group first, since that is the one the current limit is
    read off.
    """
    even = int(np.ceil(len(ij) / n_groups))
    for a in range(1, n_groups + 1):
        if n_groups % a:
            continue
        d = n_groups // a
        for k in range(a):
            # Which coset an electrode falls in: how far along the second step
            # it sits, and how far along the first once the skew is undone.
            q = np.mod(ij[:, 1], d)
            p = np.mod(ij[:, 0] - k * ((ij[:, 1] - q) // d), a)
            labels = q * a + p
            count = np.bincount(labels, minlength=n_groups)
            # An idle group is a group's worth of current left unused, and an
            # oversized one is what the limit ends up being set by:
            if count.min() and count.max() <= even * (1 + balance) + 1e-9:
                yield labels, a, d, k, int(count.max())


def _suggest(ij, n_groups, balance, n_show=4):
    """The nearest group counts that do fit this grid, for an error message"""
    reach = range(2, min(len(ij), 2 * n_groups + 8) + 1)
    fits = [n for n in reach
            if next(_sublattices(ij, n, balance), None) is not None]
    near = sorted(sorted(fits, key=lambda n: (abs(n - n_groups), n))[:n_show])
    return ', '.join(str(n) for n in near) if near else 'no other count'


class CheckerboardRaster(Raster):
    """Split electrodes into groups that are spread as far apart as possible

    Implements a generalized form of the checkerboard raster pattern tested
    in [Kasowski2025]_, which found that scattering raster groups over the
    whole array beat horizontal, vertical, and random rasters at letter
    recognition and motion discrimination, and matched not rastering at all.

    Mathematically speaking, a raster group is a coset of a sublattice of the
    electrode grid. Within a group, electrodes sit as far from one another as
    the electrode count allows. Each group is a coarser copy of the grid.
    Between groups, the order is chosen so the pattern doubles back rather
    than marching on (to reduce apparent motion). For example: Five groups on
    a square grid come out one over, two over, one back, two over, so that
    the percept steps right, down, left, down, and back rather than sliding
    across the array.

    The grid does not have to be rectangular: hexagonal grids, rotated grids,
    grids with unequal row and column spacing, and grids with electrodes
    trimmed off are all handled, since the pattern is derived from where the
    electrodes actually are. Arrays whose electrodes do not lie on a grid at
    all raise ``NotImplementedError``.

    The pattern depends on where the electrodes are, so it is worked out when
    the raster is **bound** to an implant -- which assigning it to
    :py:attr:`~pulse2percept.implants.Implant.raster` does. Until
    then the raster knows how many groups it will have but not which electrode
    goes in which, and :py:meth:`groups`, :py:attr:`min_spacing` and
    :py:meth:`~pulse2percept.implants.Raster.plot` all raise. Binding it to a
    second implant recomputes the pattern for that one.

    .. note::

        Not every ``n_groups`` fits a given grid, and one that does not
        raises a ``ValueError`` (naming counts that do) when the raster is
        bound.

        Both halves of the pattern are searched for at binding time, and the
        order the groups fire in is settled exactly only up to eight groups;
        beyond that a heuristic stands in for it, and the search grows with the
        group count.

        It is worth checking :py:attr:`min_spacing` on the ones that do fit,
        because a count can be accepted and still leave neighbors in the same
        group. The standard example is two groups on a hex grid, which
        degenerates to a line raster. In other words, implants like PRIMA
        cannot be two-colored; they want 3, 4, or 7 raster groups instead.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    n_groups : int
        Number of groups to split the electrodes into.
    balance : float, optional
        How much bigger the largest group may be than an even split would make
        it, as a fraction of it. The largest group is what sets the current the
        stimulator has to source, so this is the price being paid; what it buys
        is spacing, because the patterns that spread furthest do not always
        land evenly on a grid whose edges have been trimmed. Pass 0 to add no
        imbalance beyond the rounding an uneven electrode count forces anyway
        -- 378 electrodes in 5 groups are 76, 76, 75, 75, 76 at ``balance=0``,
        never 76 apiece -- and take whatever spacing comes with it.
    group_dur : float, optional
        See :py:class:`~pulse2percept.implants.Raster`.

    Examples
    --------
    Five groups of twelve on Argus II, as in [Kasowski2025]_:

    >>> from pulse2percept.implants import ArgusII, CheckerboardRaster
    >>> implant = ArgusII()
    >>> implant.raster = CheckerboardRaster(5)
    >>> implant.raster.n_groups
    5

    No two electrodes of a group are closer than sqrt(5) pitches, where a line
    raster would have them adjacent:

    >>> round(implant.raster.min_spacing / 575, 3)  # 575 um pitch
    2.236

    The pattern is easiest to check by eye
    (:py:meth:`~pulse2percept.implants.Raster.plot`), and the electrodes that
    fire together are :py:meth:`~pulse2percept.implants.Raster.members`:

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
        # Which electrode goes in which group is a fact about a particular
        # array, and is worked out in `bind`:
        self._group_of = None
        self._min_spacing = None

    def bind(self, implant):
        """Work the checkerboard out for this implant's electrode grid

        See :py:meth:`~pulse2percept.implants.Raster.bind`. Called for you
        when the raster is assigned to
        :py:attr:`~pulse2percept.implants.Implant.raster`.
        """
        electrode_array = _electrode_array(implant)
        names = list(electrode_array.electrode_names)
        n_groups, balance = self._n_groups, self._balance
        # Microns, which is what `min_spacing` reports the answer in:
        xy = electrode_array.coordinates(um)[:, :2]
        if len(xy) < n_groups:
            raise ValueError(f"{len(xy)} electrode(s) cannot be split into "
                             f"{n_groups} groups.")
        ij, basis = _lattice(xy)
        u, v = basis.T
        scale = min(np.linalg.norm(u), np.linalg.norm(v))

        # Of the splits that are even enough, keep the one whose groups are
        # spread furthest apart, and of those the most even. What "furthest
        # apart" means is the closest pair of electrodes the split actually
        # leaves firing together: the lattice a split is cut from says how far
        # apart its sites are, but the implant only holds a finite piece of
        # that lattice, and on a small or trimmed one the sites that would have
        # been closest need not be there at all. The lattice spectrum comes in
        # behind it, to settle ties and keep the pattern regular:
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

        # Fire them in the order that wanders least. `labels` says which
        # pattern an electrode belongs to; the group index it gets is when that
        # pattern takes its turn:
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

        # Only once the pattern is known, so that a raster that could not be
        # laid out on this array keeps whatever it was bound to before:
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

        How much the checkerboard bought over a line raster, which leaves
        neighboring electrodes in the same group and so would report the
        electrode pitch. Measured between electrodes the implant actually has,
        so a small or trimmed array can come out better spaced than the pattern
        it was cut from. Infinite when no group holds more than one electrode,
        since then no two electrodes ever fire together. None until the raster
        is bound to an implant, since there is no grid to measure yet.
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
        Either a list whose i-th element holds the names of the electrodes in
        group i, or a dict mapping each electrode name onto its group index.
        Every electrode in the stimulus must be accounted for, and no electrode
        may appear in two groups.
    group_dur : float, optional
        See :py:class:`~pulse2percept.implants.Raster`.

    Examples
    --------
    Fire the four corners of Argus II before everything else. Every other
    electrode has to be given a group too, or the current limit that the raster
    exists to respect could be violated without anyone noticing:

    >>> from pulse2percept.implants import ArgusII, CustomRaster
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
                    # Silently letting the last group win would break the very
                    # guarantee a raster exists to make, since the electrode
                    # would go on firing in the group it was taken out of:
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
