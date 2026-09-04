from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest
from scipy.spatial import cKDTree

from pulse2percept.implants import (AlphaIMS, ArgusII, BVT24,
                                    CheckerboardRaster, CustomRaster,
                                    ElectrodeGrid, PRIMAPivotal,
                                    Implant, Raster,
                                    SequentialRaster)
from pulse2percept.implants import rasters
from pulse2percept.units import (DimensionMismatchError, Quantity, mA,
                                 mm, uA, us)
from pulse2percept.units import s as sec


def test_Raster_is_abstract():
    with pytest.raises(TypeError):
        Raster()


def test_SequentialRaster():
    with pytest.raises(ValueError):
        SequentialRaster(0)
    with pytest.raises(ValueError):
        SequentialRaster(2.5)
    with pytest.raises(ValueError):
        SequentialRaster(2, group_dur=-1)
    # NaN slips through every `<` comparison, so it has to be rejected on its
    # own or it turns into a silently empty schedule much later:
    with pytest.raises(ValueError):
        SequentialRaster(np.nan)
    with pytest.raises(ValueError):
        SequentialRaster(2, group_dur=np.nan)
    with pytest.raises(ValueError):
        SequentialRaster(2, group_dur=np.inf)

    names = ArgusII().electrode_names
    # On a 6x10 grid, whose electrodes run row by row, six contiguous groups
    # are the six rows -- a line raster:
    raster = SequentialRaster(6)
    npt.assert_equal(raster.n_groups, 6)
    groups = raster.groups(names)
    npt.assert_equal(groups, np.repeat(np.arange(6), 10))
    npt.assert_equal([names[i] for i in np.flatnonzero(groups == 0)][:3],
                     ['A1', 'A2', 'A3'])
    # Interleaving puts consecutive electrodes in different groups:
    inter = SequentialRaster(6, interleave=True).groups(names)
    npt.assert_equal(inter, np.tile(np.arange(6), 10))
    # Either way, every group is the same size:
    npt.assert_equal(np.bincount(groups), np.full(6, 10))
    npt.assert_equal(np.bincount(inter), np.full(6, 10))
    npt.assert_equal('n_groups' in str(raster), True)


def test_Raster_offsets():
    names = ArgusII().electrode_names
    # By default the groups are spread evenly over the raster cycle, so the
    # sequence takes exactly one cycle to get through:
    offsets = SequentialRaster(6).offsets(names, 30.0)
    npt.assert_equal(np.unique(offsets), np.arange(6) * 5.0)
    npt.assert_almost_equal(offsets.max() + 5.0, 30.0)
    npt.assert_almost_equal(SequentialRaster(6).slot_dur(30.0), 5.0)
    # An explicit slot is used instead, as long as the groups still fit:
    offsets = SequentialRaster(6, group_dur=2).offsets(names, 30.0)
    npt.assert_equal(np.unique(offsets), np.arange(6) * 2.0)
    npt.assert_almost_equal(SequentialRaster(6, group_dur=2).slot_dur(30.0), 2)
    with pytest.raises(ValueError):
        SequentialRaster(6, group_dur=10).offsets(names, 30.0)
    # A raster of one group is no raster at all:
    npt.assert_equal(SequentialRaster(1).offsets(names, 30.0), 0)
    # The cycle is generally not a round number of ms -- a 300 Hz period is
    # 3.333... ms -- so an even split of it must not trip the fit check:
    cycle = 1000.0 / 300
    offsets = SequentialRaster(3).offsets(names, cycle)
    npt.assert_almost_equal(np.unique(offsets), np.arange(3) * cycle / 3)


def _min_spacing(implant, raster):
    """Closest two electrodes that the raster ever activates together"""
    electrode_array = getattr(implant, 'electrode_array', implant)
    xy = np.array([[e.x, e.y] for e in electrode_array.electrode_objects])
    groups = raster.groups(electrode_array.electrode_names)
    closest = np.inf
    for group in np.unique(groups):
        pos = xy[groups == group]
        if len(pos) < 2:
            continue
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        closest = min(closest, d[~np.eye(len(pos), dtype=bool)].min())
    return closest


def test_CheckerboardRaster():
    with pytest.raises(ValueError):
        CheckerboardRaster(0)
    with pytest.raises(ValueError):
        CheckerboardRaster(2.5)
    with pytest.raises(ValueError):
        CheckerboardRaster(np.nan)
    with pytest.raises(ValueError):
        CheckerboardRaster(2, balance=-0.1)
    with pytest.raises(ValueError):
        CheckerboardRaster(2, group_dur=-1)
    # More groups than electrodes is not a raster:
    with pytest.raises(ValueError):
        CheckerboardRaster(61).bind(ArgusII())
    with pytest.raises(TypeError):
        CheckerboardRaster(2).bind('ArgusII')

    implant = ArgusII()
    names = implant.electrode_names
    raster = CheckerboardRaster(5).bind(implant)
    npt.assert_equal(raster.n_groups, 5)
    groups = raster.groups(names)
    # Every electrode is in exactly one group, and the groups are the same
    # size -- an oversized group is what the current limit gets set by:
    npt.assert_equal(np.bincount(groups), np.full(5, 12))
    # Two groups is the checkerboard the pattern is named after: neighbors
    # always land in different groups, so nothing closer than the diagonal is
    # ever active at once:
    two = CheckerboardRaster(2).bind(implant)
    npt.assert_almost_equal(two.min_spacing, 575 * np.sqrt(2), decimal=3)
    # Five groups do better still, at sqrt(5) pitches -- the knight's move
    # pattern of Kasowski et al. (2025):
    npt.assert_almost_equal(raster.min_spacing, 575 * np.sqrt(5), decimal=3)
    # ... and `min_spacing` is what it says it is:
    for r in [two, raster, CheckerboardRaster(4).bind(implant)]:
        npt.assert_almost_equal(_min_spacing(implant, r), r.min_spacing,
                                decimal=3)
    # A line raster leaves neighbors in the same group, which is the whole
    # point of not using one:
    npt.assert_equal(_min_spacing(implant, SequentialRaster(6)), 575)
    npt.assert_equal('min_spacing' in str(raster), True)

    # Groups take turns in an order that doubles back instead of marching
    # across the array. On a 6x10 grid five groups lie one per column, and
    # firing them in that order would sweep steadily to the right:
    order = [np.flatnonzero(groups == g)[0] for g in range(5)]
    npt.assert_equal(order, [0, 1, 3, 2, 4])


def test_CheckerboardRaster_grids():
    # A hex grid is handled the same way, and its 7-group pattern is the one
    # that puts every group on a hex lattice of its own, sqrt(7) pitches wide:
    hexgrid = Implant(ElectrodeGrid((14, 14), 200, grid_type='hex'))
    raster = CheckerboardRaster(7).bind(hexgrid)
    npt.assert_almost_equal(raster.min_spacing, 200 * np.sqrt(7), decimal=3)
    npt.assert_almost_equal(_min_spacing(hexgrid, raster), raster.min_spacing,
                            decimal=3)
    # A hex grid cannot be two-colored the way a square one can, so two groups
    # buy nothing there and the caller has to notice through `min_spacing`:
    npt.assert_almost_equal(CheckerboardRaster(2).bind(hexgrid).min_spacing,
                            200)

    # Rotating the implant rotates the pattern with it, since the pattern is
    # read off the electrode positions:
    upright = Implant(ElectrodeGrid((10, 10), 400))
    expected = CheckerboardRaster(5).bind(upright).groups(
        upright.electrode_names)
    for angle in [11, 37, 84]:
        turned = Implant(ElectrodeGrid((10, 10), 400, rot=angle))
        npt.assert_equal(
            CheckerboardRaster(5).bind(turned).groups(turned.electrode_names),
            expected)
    # Past a quarter turn the two grid directions trade places, so the pattern
    # comes out transposed. It is the same pattern in every way that matters --
    # which is what is checked here -- just not the same labelling:
    for angle in [117, 300]:
        turned = Implant(ElectrodeGrid((10, 10), 400, rot=angle))
        raster = CheckerboardRaster(5).bind(turned)
        npt.assert_almost_equal(raster.min_spacing, 400 * np.sqrt(5),
                                decimal=3)
        npt.assert_equal(np.bincount(raster.groups(turned.electrode_names)),
                         np.full(5, 20))

    # Grids with electrodes trimmed off still work. PRIMA's 378 electrodes do
    # not divide by four, so the groups come out as even as 378 allows:
    prima = PRIMAPivotal()
    raster = CheckerboardRaster(4).bind(prima)
    count = np.bincount(raster.groups(prima.electrode_names))
    npt.assert_equal(count.sum(), 378)
    npt.assert_equal(count.max() <= np.ceil(378 / 4) * 1.05, True)
    npt.assert_almost_equal(raster.min_spacing, 200)
    # Demanding an exactly even split is allowed, and costs spacing:
    npt.assert_equal(
        CheckerboardRaster(5, balance=0).bind(prima).min_spacing <=
        CheckerboardRaster(5, balance=0.5).bind(prima).min_spacing, True)

    # Rows and columns need not be spaced the same. A grid can be stretched
    # far enough that an electrode's twenty nearest neighbors are all in its
    # own row, and the step to the next row still has to be found -- looking
    # only at the near neighborhood used to miss it and reject the grid:
    for spacing, n in [((100, 1050), 5), ((100, 1050), 4), ((25, 3000), 5)]:
        stretched = ElectrodeGrid((3, 20), spacing=spacing)
        raster = CheckerboardRaster(n).bind(stretched)
        count = np.bincount(raster.groups(stretched.electrode_names))
        npt.assert_equal(count, np.full(n, 60 // n))
        npt.assert_almost_equal(_min_spacing(stretched, raster),
                                raster.min_spacing, decimal=3)
    # Electrodes really in a line have no second direction to find, and are
    # still split into groups spread along it:
    row = ElectrodeGrid((1, 12), 200)
    npt.assert_equal(np.bincount(CheckerboardRaster(4).bind(row).groups(
        row.electrode_names)), np.full(4, 3))

    # An implant whose electrodes are not on a grid cannot be checkered:
    with pytest.raises(NotImplementedError):
        CheckerboardRaster(2).bind(BVT24())
    # Neither can a count that leaves no pattern even enough to be worth
    # having. PRIMA's trimmed edges are what put 20 groups out of reach, and
    # allowing bigger groups is what buys it back:
    with pytest.raises(ValueError):
        CheckerboardRaster(20).bind(prima)
    npt.assert_equal(CheckerboardRaster(20, balance=0.2).bind(prima).n_groups,
                     20)


def test_CheckerboardRaster_min_spacing():
    # `min_spacing` is measured between electrodes the implant actually has,
    # not between the sites of the endless lattice the pattern was cut from.
    # The two agree on an array big enough that the closest sites are all
    # present, and part company on a small one -- where the finite array is
    # the better spaced of the two, so reporting the lattice would undersell
    # it and picking a pattern by it would settle for less:
    for shape, n_groups in [((2, 6), 6), ((2, 3), 4), ((3, 4), 4), ((4, 4), 8),
                            ((6, 10), 5), ((5, 5), 5)]:
        grid = ElectrodeGrid(shape, 100)
        raster = CheckerboardRaster(n_groups).bind(grid)
        npt.assert_almost_equal(raster.min_spacing, _min_spacing(grid, raster),
                                decimal=6)
    # Two rows of six in six groups is a pair per group, and the pairs can be
    # put a whole diagonal apart -- ranking by the lattice alone settled for
    # sqrt(5) here, since the lattice cannot tell that the sites in between
    # are not on the implant:
    pairs = ElectrodeGrid((2, 6), 100)
    npt.assert_almost_equal(CheckerboardRaster(6).bind(pairs).min_spacing,
                            100 * np.sqrt(10), decimal=6)
    # A group of one electrode has no pair to keep apart:
    singles = ElectrodeGrid((2, 2), 100)
    npt.assert_equal(np.isinf(CheckerboardRaster(4).bind(singles).min_spacing),
                     True)


def test_CheckerboardRaster_is_reproducible(monkeypatch):
    # The pattern is built from the gaps between electrodes, and a grid has
    # four gaps of exactly the same length (six on a hex grid), each of which
    # turns up with both signs. Nothing about the order they are found in may
    # reach the answer, or the same implant comes out mirrored on someone
    # else's machine -- which is what used to happen, since neighbors at equal
    # distance come back from the tree in a platform-dependent order.
    class Scrambled(cKDTree):
        seed = 0

        def query(self, x, k):
            dist, idx = super().query(x, k)
            rng = np.random.RandomState(Scrambled.seed)
            for row_d, row_i in zip(dist, idx):
                tie = np.round(row_d, 6)
                for t in np.unique(tie):
                    at = np.flatnonzero(tie == t)
                    to = rng.permutation(at)
                    row_d[at], row_i[at] = row_d[to], row_i[to]
            return dist, idx

    hexgrid = Implant(ElectrodeGrid((10, 10), 400, grid_type='hex'))
    for implant in [ArgusII(), hexgrid, PRIMAPivotal()]:
        names = implant.electrode_names
        expected = CheckerboardRaster(5).bind(implant).groups(names)
        for seed in range(4):
            Scrambled.seed = seed
            monkeypatch.setattr(rasters, 'cKDTree', Scrambled)
            npt.assert_equal(CheckerboardRaster(5).bind(implant).groups(names),
                             expected)
            monkeypatch.undo()

    # Nor may the last bit of a float, which is all that separates one
    # platform's trigonometry from another's:
    implant = ArgusII()
    names = implant.electrode_names
    expected = CheckerboardRaster(5).bind(implant).groups(names)
    rng = np.random.RandomState(0)
    for _ in range(4):
        nudged = Implant(deepcopy(implant.electrode_array))
        for elec in nudged.electrode_array.electrode_objects:
            elec.x *= 1 + rng.uniform(-1, 1) * 1e-13
            elec.y *= 1 + rng.uniform(-1, 1) * 1e-13
        npt.assert_equal(CheckerboardRaster(5).bind(nudged).groups(names),
                         expected)


def test_CheckerboardRaster_groups():
    implant = ArgusII()
    raster = CheckerboardRaster(5).bind(implant)
    # The raster only knows the electrodes it was built for. Silently dropping
    # the others would break the current limit it exists to respect:
    with pytest.raises(ValueError):
        raster.groups(['A1', 'not-an-electrode'])
    # A subset of the stimulus is fine, and keeps the assignment it had:
    subset = ['F10', 'A1', 'C5']
    npt.assert_equal(raster.groups(subset),
                     [raster.groups(implant.electrode_names)[i]
                      for i in [59, 0, 24]])
    # It plugs into the schedule like any other raster:
    npt.assert_equal(np.unique(raster.offsets(implant.electrode_names, 25.0)),
                     np.arange(5) * 5.0)
    implant.raster = raster
    npt.assert_equal(implant.raster.n_groups, 5)


def test_Raster_members():
    implant = ArgusII()
    names = implant.electrode_names
    # `members` is the inverse of `groups`: the electrodes of one group, in
    # the order they were passed in:
    raster = SequentialRaster(6)
    npt.assert_equal(raster.members(names, 0), names[:10])
    npt.assert_equal(raster.members(names, 5), names[50:])
    # Names in, names out -- whatever was passed is what comes back, so a
    # list of indices gives the indices of the group:
    npt.assert_equal(SequentialRaster(6).members(range(60), 1),
                     np.arange(10, 20))
    # Every electrode is in exactly one group, and no group is lost:
    for r in [SequentialRaster(4), CheckerboardRaster(4).bind(implant),
              CustomRaster([names[:20], names[20:]])]:
        found = np.concatenate([r.members(names, g)
                                for g in range(r.n_groups)])
        npt.assert_equal(sorted(found), sorted(names))
    # A group that does not exist is a mistake, not an empty answer:
    with pytest.raises(ValueError):
        raster.members(names, 6)
    with pytest.raises(ValueError):
        raster.members(names, -1)
    with pytest.raises(ValueError):
        raster.members(names, 1.5)
    with pytest.raises(ValueError):
        raster.members(names, np.nan)


def test_Raster_plot():
    implant = ArgusII()
    raster = CheckerboardRaster(5).bind(implant)
    ax = raster.plot(implant)
    # One patch per electrode, colored by group, and a group index written
    # into each of them:
    npt.assert_equal(len(ax.collections[0].get_paths()), 60)
    npt.assert_equal(len(ax.texts), 60)
    npt.assert_equal(sorted(t.get_text() for t in ax.texts),
                     sorted(str(g) for g in raster.groups(
                         implant.electrode_names)))
    # Electrodes of one group share a color, and different groups do not:
    colors = ax.collections[0].get_facecolor()
    groups = raster.groups(implant.electrode_names)
    npt.assert_equal(len(np.unique(colors[groups == 0], axis=0)), 1)
    npt.assert_equal(len(np.unique(colors, axis=0)), 5)
    plt.close('all')

    # Annotating 1500 electrodes would be unreadable, so it is left off unless
    # asked for:
    npt.assert_equal(len(SequentialRaster(2).plot(AlphaIMS()).texts), 0)
    plt.close('all')
    npt.assert_equal(
        len(SequentialRaster(2).plot(implant, annotate=False).texts), 0)
    plt.close('all')

    # Any raster can be plotted on any implant it covers, grid or not:
    for r, imp in [(SequentialRaster(3), BVT24()),
                   (CheckerboardRaster(7).bind(PRIMAPivotal()),
                    PRIMAPivotal()),
                   (CustomRaster({n: 0 for n in ArgusII().electrode_names}),
                    ArgusII())]:
        npt.assert_equal(len(r.plot(imp).collections[0].get_paths()),
                         imp.n_electrodes)
        plt.close('all')
    # The array is what the electrodes are read from, so it has to be one:
    with pytest.raises(TypeError):
        raster.plot('ArgusII')


def test_CustomRaster():
    with pytest.raises(ValueError):
        CustomRaster([])
    with pytest.raises(TypeError):
        # A bare string is a common slip, and would otherwise be read as a
        # group of single-character electrode names:
        CustomRaster(['A1', 'A2'])

    raster = CustomRaster([['A1', 'A2'], ['A3']])
    npt.assert_equal(raster.n_groups, 2)
    npt.assert_equal(raster.groups(['A1', 'A3', 'A2']), [0, 1, 0])
    npt.assert_equal(raster.offsets(['A1', 'A3'], 10.0), [0, 5])
    # A dict says the same thing:
    same = CustomRaster({'A1': 0, 'A2': 0, 'A3': 1})
    npt.assert_equal(same.groups(['A1', 'A3', 'A2']), [0, 1, 0])
    # Every electrode in the stimulus has to be accounted for, or the current
    # limit the raster exists to respect would be violated silently:
    with pytest.raises(ValueError):
        raster.groups(['A1', 'B7'])
    # An electrode in two groups would go on firing in the group it was taken
    # out of, which is the one thing a raster is there to prevent:
    with pytest.raises(ValueError):
        CustomRaster([['A1', 'A2'], ['A2', 'A3']])
    # A fractional group index would silently truncate onto a real group:
    with pytest.raises(ValueError):
        CustomRaster({'A1': 1.9, 'A2': 0})
    with pytest.raises(ValueError):
        CustomRaster({'A1': np.nan, 'A2': 0})
    with pytest.raises(ValueError):
        CustomRaster({'A1': -1, 'A2': 0})
    # The docstring example has to cover every electrode of the implant it
    # names, or `groups` raises for the ones left out:
    corners = ['A1', 'A10', 'F1', 'F10']
    names = ArgusII().electrode_names
    full = CustomRaster([corners, [e for e in names if e not in corners]])
    npt.assert_equal(full.n_groups, 2)
    npt.assert_equal(np.bincount(full.groups(names)), [4, 56])


def test_Implant_raster():
    implant = ArgusII()
    # Implants that do not set a raster in their constructor still report one:
    npt.assert_equal(Implant(implant.electrode_array).raster, None)
    implant.raster = SequentialRaster(6)
    npt.assert_equal(implant.raster.n_groups, 6)
    npt.assert_equal('raster' in str(implant), True)
    with pytest.raises(TypeError):
        implant.raster = 'line'
    # It can be set through the constructor too:
    npt.assert_equal(
        Implant(implant.electrode_array,
                raster=SequentialRaster(3)).raster.n_groups, 3)


def test_Implant_raster_binds():
    # Assigning a raster binds it, which is what lets a geometry-dependent
    # pattern work itself out and what lets `plot` be called with no argument:
    implant = ArgusII()
    raster = CheckerboardRaster(5)
    # Before binding it knows how many groups it will have and nothing else:
    npt.assert_equal(raster.n_groups, 5)
    npt.assert_equal(raster.implant, None)
    npt.assert_equal(raster.min_spacing, None)
    with pytest.raises(ValueError):
        raster.groups(implant.electrode_names)
    with pytest.raises(ValueError):
        raster.plot()

    implant.raster = raster
    npt.assert_equal(raster.implant is implant, True)
    npt.assert_almost_equal(raster.min_spacing, 575 * np.sqrt(5), decimal=3)
    groups = raster.groups(implant.electrode_names)
    npt.assert_equal(np.bincount(groups), np.full(5, 12))
    # And `plot` no longer needs to be told what to draw:
    npt.assert_equal(len(raster.plot().collections[0].get_paths()), 60)
    plt.close('all')

    # Rebinding recomputes: the same object on a different array describes
    # *that* array, rather than answering about the one it came from:
    other = Implant(ElectrodeGrid((4, 5), 400))
    other.raster = raster
    npt.assert_equal(raster.implant is other, True)
    npt.assert_almost_equal(raster.min_spacing, 400 * np.sqrt(5), decimal=3)
    npt.assert_equal(np.bincount(raster.groups(other.electrode_names)),
                     np.full(5, 4))
    with pytest.raises(ValueError):
        # Argus II's electrodes are not on the grid it is now bound to:
        raster.groups(implant.electrode_names)
    # A raster that cannot be laid out on an array leaves the implant alone:
    with pytest.raises(NotImplementedError):
        BVT24().raster = CheckerboardRaster(2)
    # The other two bind too, even though there is no geometry to work out:
    for r in [SequentialRaster(6), CustomRaster([implant.electrode_names])]:
        implant.raster = r
        npt.assert_equal(r.implant is implant, True)
        npt.assert_equal(len(r.plot().collections[0].get_paths()), 60)
        plt.close('all')


def test_Implant_max_current():
    implant = ArgusII()
    npt.assert_equal(implant.max_current, None)
    with pytest.raises(ValueError):
        implant.max_current = 0
    # 60 electrodes at 20 uA each is 1200 uA at once:
    implant.max_current = 1500
    npt.assert_equal(implant.prepare_stim(np.full(60, 20)).shape, (60, 1))
    implant.max_current = 1000
    with pytest.raises(ValueError):
        implant.prepare_stim(np.full(60, 20))
    # The sign does not matter: what the stimulator sources is the sum of the
    # magnitudes:
    with pytest.raises(ValueError):
        implant.prepare_stim(np.full(60, -20))
    # A single electrode is well within the limit:
    npt.assert_almost_equal(implant.prepare_stim({'A1': 900}).data.max(), 900)
    # An empty stimulus has nothing to check:
    npt.assert_equal(implant.prepare_stim(None), None)


def test_Raster_units():
    names = ArgusII().electrode_names
    bare = SequentialRaster(6, group_dur=1)
    unitful = SequentialRaster(6, group_dur=1000 * us)
    npt.assert_almost_equal(unitful.group_dur, 1)
    npt.assert_equal(isinstance(unitful.group_dur, Quantity), False)
    # `period` is a duration too, in both methods that take one:
    npt.assert_almost_equal(bare.slot_dur(10), unitful.slot_dur(0.01 * sec))
    npt.assert_array_equal(bare.offsets(names, 10),
                           unitful.offsets(names, 0.01 * sec))
    # An even split has no group_dur of its own, and still takes a unitful
    # period:
    even = SequentialRaster(6)
    npt.assert_almost_equal(even.slot_dur(12), even.slot_dur(0.012 * sec))
    npt.assert_array_equal(even.offsets(names, 12),
                           even.offsets(names, 0.012 * sec))
    with pytest.raises(DimensionMismatchError):
        SequentialRaster(6, group_dur=1 * uA)
    with pytest.raises(DimensionMismatchError):
        bare.slot_dur(10 * uA)
    with pytest.raises(DimensionMismatchError):
        bare.offsets(names, 10 * uA)


def test_Raster_units_end_to_end():
    """A rastered encoding is the same whichever way its timings are spelled"""
    from pulse2percept.stimuli import AmplitudeEncoder, ImageStimulus
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    plain = ArgusII(raster=SequentialRaster(6, group_dur=1))
    unitful_raster = ArgusII(raster=SequentialRaster(6, group_dur=1000 * us))
    bare = AmplitudeEncoder(amp_range=(0, 50)).encode(img, implant=plain)
    unitful = AmplitudeEncoder(amp_range=(0, 0.05 * mA)).encode(
        img, implant=unitful_raster)
    npt.assert_array_equal(bare.data, unitful.data)
    npt.assert_array_equal(bare.time, unitful.time)


def test_Raster_reads_coordinates_in_microns():
    """Raster geometry goes through the array's coordinate API"""
    implant = ArgusII()
    raster = CheckerboardRaster(5).bind(implant)
    # `min_spacing` is documented in microns, which is what `coordinates()`
    # returns, and Argus II has a 575 um pitch:
    npt.assert_allclose(raster.min_spacing, np.sqrt(5) * 575, rtol=1e-12)
    # Rotating the device does not change the pattern or the spacing:
    turned = CheckerboardRaster(5).bind(ArgusII(rot=30))
    npt.assert_allclose(turned.min_spacing, raster.min_spacing, rtol=1e-12)
    npt.assert_array_equal(turned.groups(implant.electrode_names),
                           raster.groups(implant.electrode_names))
    # Both entry points accept an implant or its array, and refuse anything
    # that cannot say where its electrodes are:
    npt.assert_equal(
        CheckerboardRaster(5).bind(implant.electrode_array).n_groups, 5)
    for call in (lambda: CheckerboardRaster(2).bind('not an implant'),
                 lambda: SequentialRaster(2).plot('not an implant')):
        with pytest.raises(TypeError):
            call()
    ax = raster.plot(implant)
    npt.assert_equal(ax.get_xlabel(), 'x (microns)')
    plt.close('all')
