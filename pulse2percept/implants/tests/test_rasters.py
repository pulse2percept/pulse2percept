import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.implants import (AlphaIMS, ArgusII, BVT24,
                                    CheckerboardRaster, CustomRaster,
                                    ElectrodeGrid, PRIMA, ProsthesisSystem,
                                    Raster, SequentialRaster)


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
    names = implant.electrode_names
    xy = np.array([[e.x, e.y] for e in implant.earray.electrode_objects])
    groups = raster.groups(names)
    closest = np.inf
    for group in np.unique(groups):
        pos = xy[groups == group]
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        closest = min(closest, d[~np.eye(len(pos), dtype=bool)].min())
    return closest


def test_CheckerboardRaster():
    with pytest.raises(ValueError):
        CheckerboardRaster(ArgusII(), 0)
    with pytest.raises(ValueError):
        CheckerboardRaster(ArgusII(), 2.5)
    with pytest.raises(ValueError):
        CheckerboardRaster(ArgusII(), np.nan)
    with pytest.raises(ValueError):
        CheckerboardRaster(ArgusII(), 2, balance=-0.1)
    with pytest.raises(ValueError):
        CheckerboardRaster(ArgusII(), 2, group_dur=-1)
    # More groups than electrodes is not a raster:
    with pytest.raises(ValueError):
        CheckerboardRaster(ArgusII(), 61)
    with pytest.raises(TypeError):
        CheckerboardRaster('ArgusII', 2)

    implant = ArgusII()
    names = implant.electrode_names
    raster = CheckerboardRaster(implant, 5)
    npt.assert_equal(raster.n_groups, 5)
    groups = raster.groups(names)
    # Every electrode is in exactly one group, and the groups are the same
    # size -- an oversized group is what the current limit gets set by:
    npt.assert_equal(np.bincount(groups), np.full(5, 12))
    # Two groups is the checkerboard the pattern is named after: neighbors
    # always land in different groups, so nothing closer than the diagonal is
    # ever active at once:
    two = CheckerboardRaster(implant, 2)
    npt.assert_almost_equal(two.min_spacing, 575 * np.sqrt(2), decimal=3)
    # Five groups do better still, at sqrt(5) pitches -- the knight's move
    # pattern of Kasowski et al. (2025):
    npt.assert_almost_equal(raster.min_spacing, 575 * np.sqrt(5), decimal=3)
    # ... and `min_spacing` is what it says it is:
    for r in [two, raster, CheckerboardRaster(implant, 4)]:
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
    hexgrid = ProsthesisSystem(ElectrodeGrid((14, 14), 200, type='hex'))
    raster = CheckerboardRaster(hexgrid, 7)
    npt.assert_almost_equal(raster.min_spacing, 200 * np.sqrt(7), decimal=3)
    npt.assert_almost_equal(_min_spacing(hexgrid, raster), raster.min_spacing,
                            decimal=3)
    # A hex grid cannot be two-colored the way a square one can, so two groups
    # buy nothing there and the caller has to notice through `min_spacing`:
    npt.assert_almost_equal(CheckerboardRaster(hexgrid, 2).min_spacing, 200)

    # Rotating the implant rotates the pattern with it, since the pattern is
    # read off the electrode positions:
    upright = ProsthesisSystem(ElectrodeGrid((10, 10), 400))
    turned = ProsthesisSystem(ElectrodeGrid((10, 10), 400, rot=37))
    npt.assert_equal(CheckerboardRaster(turned, 5).groups(
        turned.electrode_names),
        CheckerboardRaster(upright, 5).groups(upright.electrode_names))

    # Grids with electrodes trimmed off still work. PRIMA's 378 electrodes do
    # not divide by four, so the groups come out as even as 378 allows:
    prima = PRIMA()
    raster = CheckerboardRaster(prima, 4)
    count = np.bincount(raster.groups(prima.electrode_names))
    npt.assert_equal(count.sum(), 378)
    npt.assert_equal(count.max() <= np.ceil(378 / 4) * 1.05, True)
    npt.assert_almost_equal(raster.min_spacing, 200)
    # Demanding an exactly even split is allowed, and costs spacing:
    npt.assert_equal(
        CheckerboardRaster(prima, 5, balance=0).min_spacing <=
        CheckerboardRaster(prima, 5, balance=0.5).min_spacing, True)

    # An implant whose electrodes are not on a grid cannot be checkered:
    with pytest.raises(NotImplementedError):
        CheckerboardRaster(BVT24(), 2)
    # Neither can a count that leaves no pattern even enough to be worth
    # having. PRIMA's trimmed edges are what put 20 groups out of reach, and
    # allowing bigger groups is what buys it back:
    with pytest.raises(ValueError):
        CheckerboardRaster(prima, 20)
    npt.assert_equal(CheckerboardRaster(prima, 20, balance=0.2).n_groups, 20)


def test_CheckerboardRaster_groups():
    implant = ArgusII()
    raster = CheckerboardRaster(implant, 5)
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
    for r in [SequentialRaster(4), CheckerboardRaster(implant, 4),
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
    raster = CheckerboardRaster(implant, 5)
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
                   (CheckerboardRaster(PRIMA(), 7), PRIMA()),
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


def test_ProsthesisSystem_raster():
    implant = ArgusII()
    # Implants that do not set a raster in their constructor still report one:
    npt.assert_equal(implant.raster, None)
    npt.assert_equal(ProsthesisSystem(implant.earray).raster, None)
    implant.raster = SequentialRaster(6)
    npt.assert_equal(implant.raster.n_groups, 6)
    npt.assert_equal('raster' in str(implant), True)
    with pytest.raises(TypeError):
        implant.raster = 'line'
    # It can be set through the constructor too:
    npt.assert_equal(
        ProsthesisSystem(implant.earray,
                         raster=SequentialRaster(3)).raster.n_groups, 3)


def test_ProsthesisSystem_max_current():
    implant = ArgusII()
    npt.assert_equal(implant.max_current, None)
    with pytest.raises(ValueError):
        implant.max_current = 0
    # 60 electrodes at 20 uA each is 1200 uA at once:
    implant.max_current = 1500
    implant.stim = np.full(60, 20)
    npt.assert_equal(implant.stim.shape, (60, 1))
    implant.max_current = 1000
    with pytest.raises(ValueError):
        implant.stim = np.full(60, 20)
    # The sign does not matter: what the stimulator sources is the sum of the
    # magnitudes:
    with pytest.raises(ValueError):
        implant.stim = np.full(60, -20)
    # A single electrode is well within the limit:
    implant.stim = {'A1': 900}
    npt.assert_almost_equal(implant.stim.data.max(), 900)
    # An empty stimulus has nothing to check:
    implant.stim = None
    npt.assert_equal(implant.stim, None)
