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


class Raster(PrettyPrint, metaclass=ABCMeta):
    """Abstract base class for all raster patterns

    A stimulator usually cannot drive every electrode at once, because the
    total current it can source at any instant is limited. Electrodes are
    therefore split into *raster groups* that take turns.

    A raster is a **scheduling constraint, not a hardware state machine**. What
    it has to deliver is one property: no two groups are ever active at the same
    instant, so that the stimulator sources at most one group's worth of current
    however the video is modulated. It is not a switch that cyclically enables
    group 0, then group 1, then group 0 again forever, and a group's pulses do
    not have to land at the same phase of a repeating cycle.

    Taking turns is described by a **raster sweep**: group *g* starts its pulse
    ``g * group_dur`` after group 0 does, so a sweep spans ``n_groups *
    group_dur``. Two things then keep groups apart for good (see
    :py:class:`~pulse2percept.stimuli.Encoder`):

    1.  A pulse has to be short enough to finish before the next group's turn
        begins.
    2.  Electrodes on *different* pulse periods drift relative to one another,
        and would eventually collide however they started out. Their periods are
        therefore pinned to whole numbers of the sweep, which fixes their
        relative phase. Pinning rounds the period *up*, so multiplexing never
        drives an electrode faster -- and so never delivers more charge -- than
        asked.

        Electrodes that share one period cannot drift in the first place: their
        onsets stay ``group_dur`` apart forever, whatever that period is.
        Nothing is quantized in that case and the requested rate is delivered
        exactly, even when the period is not a whole number of sweeps. This is
        the usual case under amplitude modulation, and it is why rastering costs
        no frequency there.

        So with two groups 1.5 ms apart on a common 10 ms period, group 0 pulses
        at 0, 10, 20, ... and group 1 at 1.5, 11.5, 21.5, ... -- collision-free,
        but not a repeating 3 ms schedule, and the 10 ms period is left alone.

    The sweep belongs to the *stimulation* schedule, not to the video: it is
    tied to the pulse period, not to the frame rate. Two rules settle how long
    it is and what it costs:

    *  With ``group_dur=None`` the groups divide the *shortest* pulse period
       between them, so the sweep is exactly that period. Under frequency
       modulation that means the fastest electrode pulses once per sweep and
       slower ones every *m*-th sweep.
    *  With an explicit ``group_dur`` the sweep is ``n_groups * group_dur``
       whatever rate the electrodes run at -- six groups of 1 ms sweep in 6 ms.
       It is then generally much shorter than a pulse period, so even the
       fastest electrode may pulse only every *m*-th sweep.

    Either way, only periods that *differ* from one another are rounded up onto
    the sweep; a period they all share is delivered exactly, since fixed group
    offsets cannot drift into one another.

    Subclasses only implement ``groups``.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    group_dur : float, optional
        Duration (ms) of a single group's slot, and hence the spacing between
        one group's turn and the next. If None, the groups are spread evenly
        over the pulse period, so that a sweep takes exactly one period to
        complete -- which is what an encoder wants whenever every electrode
        pulses at the same rate.

        Setting it explicitly makes the sweep ``n_groups * group_dur``
        regardless of the pulse period, which is how you buy back frequency
        resolution under frequency modulation: a shorter slot means a shorter
        sweep, and the periods that have to be pinned are pinned onto a finer
        grid. It cannot be shorter than a single pulse.

        An encoder with a ``clock`` rounds the slot onto it and rebuilds the
        sweep from the result, so every group keeps a turn of the same length.

    """
    __slots__ = ('group_dur',)

    def __init__(self, group_dur=None):
        if group_dur is not None:
            _finite('group_dur', group_dur)
            if group_dur <= 0:
                raise ValueError("'group_dur' must be positive.")
        self.group_dur = group_dur

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        return {'group_dur': self.group_dur, 'n_groups': self.n_groups}

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
        """The electrodes that take their turn together in one group

        The inverse of :py:meth:`groups`, which says what group each electrode
        is in. This says which electrodes are in a group.

        Parameters
        ----------
        electrodes : array_like
            Electrode names, in the order they appear in the stimulus.
        group : int
            Which group to look up, in ``0..n_groups-1``. Groups take their
            turns in index order, so group 0 is the one that goes first.

        Returns
        -------
        members : array
            The entries of ``electrodes`` belonging to ``group``, in the order
            they were given: names in, names out.

        Examples
        --------
        >>> from pulse2percept.implants import ArgusII, SequentialRaster
        >>> names = ArgusII().electrode_names
        >>> SequentialRaster(6).members(names, 0)[:4]
        array(['A1', 'A2', 'A3', 'A4'], dtype='<U3')

        """
        group = _whole('group', group)
        if group < 0 or group >= self.n_groups:
            raise ValueError(f"'group' must be in 0..{self.n_groups - 1}, not "
                             f"{group}.")
        return np.asarray(electrodes)[self.groups(electrodes) == group]

    def plot(self, implant, annotate=None, ax=None, cmap='viridis',
             autoscale=True):
        """Plot the electrode array, colored by raster group

        What a raster does is spatial, so the quickest way to tell whether it
        does what was wanted is to look at it. Colors run in the order the
        groups take their turns, so the picture shows the schedule as well as
        the pattern: with
        :py:class:`~pulse2percept.implants.CheckerboardRaster` a group's
        electrodes should be scattered over the whole array rather than
        gathered into a line, and neighboring colors should not lie next to
        one another in a consistent direction.

        Parameters
        ----------
        implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`
            The implant to draw, or its
            :py:class:`~pulse2percept.implants.ElectrodeArray`. Its electrodes
            are the ones the raster is asked about, so this has to be an
            implant the raster covers.
        annotate : bool, optional
            Whether to write the group index into each electrode. If None,
            they are written whenever there are few enough electrodes
            (at most 120) for the numbers to be readable.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If None, uses the current axes.
        cmap : str, optional
            Matplotlib colormap the group colors are taken from, evenly
            spaced. A sequential map is the useful default, since the order
            the colors run in is the order the groups fire in.
        autoscale : bool, optional
            Whether to fit the x/y limits to the implant.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes drawn on.

        """
        earray = getattr(implant, 'earray', implant)
        names = getattr(earray, 'electrode_names', None)
        elecs = getattr(earray, 'electrode_objects', None)
        if names is None or elecs is None:
            raise TypeError(f"'implant' must be a ProsthesisSystem or an "
                            f"ElectrodeArray, not {type(implant)}.")
        names, elecs = list(names), list(elecs)
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
        xy = np.array([[e.x, e.y] for e in elecs], dtype=np.float64)
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
        if w1 @ w1 > w2 @ w2:
            w1, w2 = w2, w1
        mu = np.round((w2 @ w1) / (w1 @ w1))
        if mu == 0:
            break
        w2 = w2 - mu * w1
    return w1, w2


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
    # The lattice steps are among the shortest gaps between electrodes, and a
    # handful of nearest neighbors is enough to turn them up:
    _, idx = cKDTree(xy).query(xy, k=min(n, 9))
    diff = (xy[idx[:, 1:]] - xy[:, None, :]).reshape(-1, 2)
    d = np.linalg.norm(diff, axis=1)
    # Two electrodes in the same spot are no step at all, and would otherwise
    # be taken for one of zero length:
    apart = d > 1e-9 * d.max(initial=0.0)
    if not np.any(apart):
        raise NotImplementedError(
            "A checkerboard needs electrodes on a regular grid, and these are "
            "all in the same place.")
    diff, d = diff[apart], d[apart]
    diff = diff[np.argsort(np.round(d / d.max(), 9), kind='stable')]
    u = diff[0]
    # The second step has to leave the line the first one traces out:
    cross = np.abs(u[0] * diff[:, 1] - u[1] * diff[:, 0])
    off_axis = np.flatnonzero(cross > 1e-6 * (u @ u))
    if len(off_axis) == 0:
        # Electrodes on a single line: any second step will do, since nothing
        # is ever placed along it.
        v = np.array([-u[1], u[0]])
    else:
        v = diff[off_axis[0]]
    u, v = _reduce(u, v)
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

    .. note::

        Not every ``n_groups`` fits a given grid, and one that does not
        raises a ``ValueError`` listing counts that do.
        
        It is worth checking :py:attr:`min_spacing` on the ones that do fit,
        because a count can be accepted and still leave neighbors in the same
        group. The standard example is two groups on a hex grid, which
        degenerates to a line raster. In other words, implants like PRIMA
        cannot be two-colored; they want 3, 4, or 7 raster groups instead.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`
        The implant to build the pattern for, or its
        :py:class:`~pulse2percept.implants.ElectrodeArray`. The electrodes have
        to lie on a grid, and their names are how the raster recognizes them
        later, so this has to be the implant the stimulus will be applied to.
    n_groups : int
        Number of groups to split the electrodes into.
    balance : float, optional
        How much bigger the largest group may be than an even split would make
        it, as a fraction of it. The largest group is what sets the current the
        stimulator has to source, so this is the price being paid; what it buys
        is spacing, because the patterns that spread furthest do not always
        land evenly on a grid whose edges have been trimmed. Pass 0 to insist
        on an even split and take whatever spacing comes with it.
    group_dur : float, optional
        See :py:class:`~pulse2percept.implants.Raster`.

    Examples
    --------
    Five groups of twelve on Argus II, as in [Kasowski2025]_:

    >>> from pulse2percept.implants import ArgusII, CheckerboardRaster
    >>> implant = ArgusII()
    >>> implant.raster = CheckerboardRaster(implant, 5)
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
    __slots__ = ('_n_groups', '_group_of', '_min_spacing')

    def __init__(self, implant, n_groups, balance=0.05, group_dur=None):
        super().__init__(group_dur=group_dur)
        _finite('n_groups', n_groups)
        if int(n_groups) != n_groups or n_groups < 1:
            raise ValueError(f"'n_groups' must be a positive integer, not "
                             f"{n_groups}.")
        n_groups = int(n_groups)
        _finite('balance', balance)
        if balance < 0:
            raise ValueError(f"'balance' cannot be negative, not {balance}.")
        # A ProsthesisSystem carries the array; anything else has to be one.
        # Duck-typed rather than imported, since `base` imports this module:
        earray = getattr(implant, 'earray', implant)
        names = getattr(earray, 'electrode_names', None)
        elecs = getattr(earray, 'electrode_objects', None)
        if names is None or elecs is None:
            raise TypeError(f"'implant' must be a ProsthesisSystem or an "
                            f"ElectrodeArray, not {type(implant)}.")
        names = list(names)
        xy = np.array([[e.x, e.y] for e in elecs], dtype=np.float64)
        if len(xy) < n_groups:
            raise ValueError(f"{len(xy)} electrode(s) cannot be split into "
                             f"{n_groups} groups.")
        ij, basis = _lattice(xy)
        u, v = basis.T
        scale = min(np.linalg.norm(u), np.linalg.norm(v))

        # Of the splits that are even enough, keep the one whose groups are
        # spread furthest apart, and of those the most even:
        best = None
        for labels, a, d, k, biggest in _sublattices(ij, n_groups, balance):
            w1, w2 = a * u, k * u + d * v
            key = (tuple(np.round(_spectrum(w1, w2) / scale, 6)), -biggest)
            if best is None or key > best[0]:
                best = (key, labels, w1, w2)
        if best is None:
            raise ValueError(
                f"No checkerboard of {n_groups} groups fits these "
                f"{len(xy)} electrodes. A group is every a-th electrode "
                f"across by every d-th down (a * d = {n_groups}), and on this "
                f"grid every such pattern leaves one group more than "
                f"{balance:.0%} bigger than an even split. Try "
                f"{_suggest(ij, n_groups, balance)} groups instead, or raise "
                f"'balance' to allow groups of unequal size.")
        labels, w1, w2 = best[1:]
        self._min_spacing = float(_spectrum(w1, w2, n_terms=1)[0])

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

        self._n_groups = n_groups
        self._group_of = {str(name): int(slot[label])
                          for name, label in zip(names, labels)}

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super()._pprint_params()
        params.update({'min_spacing': self.min_spacing})
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
        electrode pitch.
        """
        return self._min_spacing

    def groups(self, electrodes):
        """Assign each electrode to a raster group"""
        try:
            return np.array([self._group_of[str(e)] for e in electrodes])
        except KeyError:
            missing = sorted({str(e) for e in electrodes} -
                             set(self._group_of))
            raise ValueError(f"Electrode(s) {missing[:10]} are not on the "
                             f"grid this raster was built for. Build it from "
                             f"the implant the stimulus is applied to.")


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
