from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest
from scipy.spatial import cKDTree

from pulse2percept.implants import (CheckerboardRaster, CustomRaster,
                                    ElectrodeGrid, Implant, Raster,
                                    SequentialRaster)
from pulse2percept.implants.retina import (AlphaIMS, ArgusII, Suprachoroidal24,
                                           PRIMAPivotal)
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
    # NaN passes every `<` comparison, so it is checked separately (otherwise
    # it gives an empty schedule later):
    with pytest.raises(ValueError):
        SequentialRaster(np.nan)
    with pytest.raises(ValueError):
        SequentialRaster(2, group_dur=np.nan)
    with pytest.raises(ValueError):
        SequentialRaster(2, group_dur=np.inf)

    names = ArgusII().electrode_names
    # Argus II electrodes are ordered row by row, so six contiguous groups are
    # the six rows (a line raster):
    raster = SequentialRaster(6)
    npt.assert_equal(raster.n_groups, 6)
    groups = raster.groups(names)
    npt.assert_equal(groups, np.repeat(np.arange(6), 10))
    npt.assert_equal([names[i] for i in np.flatnonzero(groups == 0)][:3],
                     ['A1', 'A2', 'A3'])
    # Interleaving puts consecutive electrodes in different groups:
    inter = SequentialRaster(6, interleave=True).groups(names)
    npt.assert_equal(inter, np.tile(np.arange(6), 10))
    # Every group has the same size:
    npt.assert_equal(np.bincount(groups), np.full(6, 10))
    npt.assert_equal(np.bincount(inter), np.full(6, 10))
    npt.assert_equal('n_groups' in str(raster), True)


def test_Raster_offsets():
    names = ArgusII().electrode_names
    # By default, groups are spread evenly over one raster cycle:
    offsets = SequentialRaster(6).offsets(names, 30.0)
    npt.assert_equal(np.unique(offsets), np.arange(6) * 5.0)
    npt.assert_almost_equal(offsets.max() + 5.0, 30.0)
    npt.assert_almost_equal(SequentialRaster(6).slot_dur(30.0), 5.0)
    # An explicit group_dur is used if all groups fit in the cycle:
    offsets = SequentialRaster(6, group_dur=2).offsets(names, 30.0)
    npt.assert_equal(np.unique(offsets), np.arange(6) * 2.0)
    npt.assert_almost_equal(SequentialRaster(6, group_dur=2).slot_dur(30.0), 2)
    with pytest.raises(ValueError):
        SequentialRaster(6, group_dur=10).offsets(names, 30.0)
    # A single group has zero offset:
    npt.assert_equal(SequentialRaster(1).offsets(names, 30.0), 0)
    # The cycle is often not a round number of ms (300 Hz is 3.333... ms), so
    # an even split must pass the fit check:
    cycle = 1000.0 / 300
    offsets = SequentialRaster(3).offsets(names, cycle)
    npt.assert_almost_equal(np.unique(offsets), np.arange(3) * cycle / 3)


def _min_spacing(implant, raster):
    """Returns the smallest distance between two same-group electrodes"""
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
    # More groups than electrodes:
    with pytest.raises(ValueError):
        CheckerboardRaster(61).bind(ArgusII())
    with pytest.raises(TypeError):
        CheckerboardRaster(2).bind('ArgusII')

    implant = ArgusII()
    names = implant.electrode_names
    raster = CheckerboardRaster(5).bind(implant)
    npt.assert_equal(raster.n_groups, 5)
    groups = raster.groups(names)
    # Every electrode is in exactly one group, and groups have equal size (the
    # largest group sets the current limit):
    npt.assert_equal(np.bincount(groups), np.full(5, 12))
    # With two groups, neighbors are always in different groups, so the closest
    # co-active pair is diagonal:
    two = CheckerboardRaster(2).bind(implant)
    npt.assert_almost_equal(two.min_spacing, 575 * np.sqrt(2), decimal=3)
    # Five groups give sqrt(5) pitches, the knight's move pattern of
    # Kasowski et al. (2025):
    npt.assert_almost_equal(raster.min_spacing, 575 * np.sqrt(5), decimal=3)
    # `min_spacing` matches the brute-force distance:
    for r in [two, raster, CheckerboardRaster(4).bind(implant)]:
        npt.assert_almost_equal(_min_spacing(implant, r), r.min_spacing,
                                decimal=3)
    # A line raster puts neighbors in the same group:
    npt.assert_equal(_min_spacing(implant, SequentialRaster(6)), 575)
    npt.assert_equal('min_spacing' in str(raster), True)

    # Groups fire in an order that doubles back. On a 6x10 grid, five groups
    # lie one per column, so firing in index order would sweep to the right:
    order = [np.flatnonzero(groups == g)[0] for g in range(5)]
    npt.assert_equal(order, [0, 1, 3, 2, 4])


def test_CheckerboardRaster_grids():
    # On a hex grid, 7 groups form interleaved hexagonal lattices with
    # sqrt(7) times the electrode spacing.
    hexgrid = Implant(ElectrodeGrid((14, 14), 200, grid_type='hex'))
    raster = CheckerboardRaster(7).bind(hexgrid)
    npt.assert_almost_equal(raster.min_spacing, 200 * np.sqrt(7), decimal=3)
    npt.assert_almost_equal(_min_spacing(hexgrid, raster), raster.min_spacing,
                            decimal=3)

    # A hex grid is not bipartite, so splitting it into two groups does not
    # increase the minimum within-group spacing.
    npt.assert_almost_equal(
        CheckerboardRaster(2).bind(hexgrid).min_spacing, 200)

    # Rotation preserves the grouping while the inferred grid axes stay in the
    # same order:
    upright = Implant(ElectrodeGrid((10, 10), 400))
    expected = CheckerboardRaster(5).bind(upright).groups(
        upright.electrode_names)
    for angle in [11, 37, 84]:
        turned = Implant(ElectrodeGrid((10, 10), 400, rot=angle))
        npt.assert_equal(
            CheckerboardRaster(5).bind(turned).groups(turned.electrode_names),
            expected)

    # Larger rotations may transpose the inferred grid axes, but spacing and
    # group sizes stay the same:
    for angle in [117, 300]:
        turned = Implant(ElectrodeGrid((10, 10), 400, rot=angle))
        raster = CheckerboardRaster(5).bind(turned)
        npt.assert_almost_equal(raster.min_spacing, 400 * np.sqrt(5),
                                decimal=3)
        npt.assert_equal(
            np.bincount(raster.groups(turned.electrode_names)),
            np.full(5, 20))

    # Trimmed grids still give approximately balanced groups:
    prima = PRIMAPivotal()
    raster = CheckerboardRaster(4).bind(prima)
    count = np.bincount(raster.groups(prima.electrode_names))
    npt.assert_equal(count.sum(), 378)
    npt.assert_equal(count.max() <= np.ceil(378 / 4) * 1.05, True)
    npt.assert_almost_equal(raster.min_spacing, 200)

    # Requiring an exactly balanced split may reduce the achievable spacing:
    npt.assert_equal(
        CheckerboardRaster(5, balance=0).bind(prima).min_spacing <=
        CheckerboardRaster(5, balance=0.5).bind(prima).min_spacing, True)

    # Grid detection handles strongly anisotropic spacing, where the second
    # grid direction may lie outside the nearest neighborhood:
    for spacing, n in [((100, 1050), 5), ((100, 1050), 4), ((25, 3000), 5)]:
        stretched = ElectrodeGrid((3, 20), spacing=spacing)
        raster = CheckerboardRaster(n).bind(stretched)
        count = np.bincount(raster.groups(stretched.electrode_names))
        npt.assert_equal(count, np.full(n, 60 // n))
        npt.assert_almost_equal(_min_spacing(stretched, raster),
                                raster.min_spacing, decimal=3)

    # A one-dimensional grid is split along the row:
    row = ElectrodeGrid((1, 12), 200)
    npt.assert_equal(
        np.bincount(
            CheckerboardRaster(4).bind(row).groups(row.electrode_names)),
        np.full(4, 3))

    # Non-grid electrode layouts are not supported:
    with pytest.raises(NotImplementedError):
        CheckerboardRaster(2).bind(Suprachoroidal24())

    # Group counts that violate the default balance constraint are rejected;
    # relaxing the constraint allows them:
    with pytest.raises(ValueError):
        CheckerboardRaster(20).bind(prima)
    npt.assert_equal(
        CheckerboardRaster(20, balance=0.2).bind(prima).n_groups, 20)


def test_CheckerboardRaster_min_spacing():
    # `min_spacing` is measured between the implant's electrodes, not lattice
    # sites. The two agree on a large array; on a small one the finite array is
    # better spaced, and patterns are ranked by that:
    for shape, n_groups in [((2, 6), 6), ((2, 3), 4), ((3, 4), 4), ((4, 4), 8),
                            ((6, 10), 5), ((5, 5), 5)]:
        grid = ElectrodeGrid(shape, 100)
        raster = CheckerboardRaster(n_groups).bind(grid)
        npt.assert_almost_equal(raster.min_spacing, _min_spacing(grid, raster),
                                decimal=6)
    # Two rows of six in six groups: one pair per group, a full diagonal apart
    # (the lattice alone would give sqrt(5)):
    pairs = ElectrodeGrid((2, 6), 100)
    npt.assert_almost_equal(CheckerboardRaster(6).bind(pairs).min_spacing,
                            100 * np.sqrt(10), decimal=6)
    # Groups of one electrode give infinite spacing:
    singles = ElectrodeGrid((2, 2), 100)
    npt.assert_equal(np.isinf(CheckerboardRaster(4).bind(singles).min_spacing),
                     True)


def test_CheckerboardRaster_is_reproducible(monkeypatch):
    # cKDTree returns equidistant neighbors in platform-dependent order; the
    # pattern must not depend on it:
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

    # Nor on last-bit float differences between platforms' trigonometry:
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
    # Electrodes the raster was not built for are rejected, since dropping them
    # would break the current limit:
    with pytest.raises(ValueError):
        raster.groups(['A1', 'not-an-electrode'])
    # A subset of the bound electrodes keeps its group assignment:
    subset = ['F10', 'A1', 'C5']
    npt.assert_equal(raster.groups(subset),
                     [raster.groups(implant.electrode_names)[i]
                      for i in [59, 0, 24]])
    # Works in the pulse schedule like any other raster:
    npt.assert_equal(np.unique(raster.offsets(implant.electrode_names, 25.0)),
                     np.arange(5) * 5.0)
    implant.raster = raster
    npt.assert_equal(implant.raster.n_groups, 5)


def test_Raster_members():
    implant = ArgusII()
    names = implant.electrode_names
    # `members` is the inverse of `groups`: electrodes of one group, in input
    # order:
    raster = SequentialRaster(6)
    npt.assert_equal(raster.members(names, 0), names[:10])
    npt.assert_equal(raster.members(names, 5), names[50:])
    # Returns the same kind of labels it was given (indices in, indices out):
    npt.assert_equal(SequentialRaster(6).members(range(60), 1),
                     np.arange(10, 20))
    # Every electrode is in exactly one group:
    for r in [SequentialRaster(4), CheckerboardRaster(4).bind(implant),
              CustomRaster([names[:20], names[20:]])]:
        found = np.concatenate([r.members(names, g)
                                for g in range(r.n_groups)])
        npt.assert_equal(sorted(found), sorted(names))
    # A nonexistent group index is a ValueError:
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
    # One patch per electrode, colored and labeled by group index:
    npt.assert_equal(len(ax.collections[0].get_paths()), 60)
    npt.assert_equal(len(ax.texts), 60)
    npt.assert_equal(sorted(t.get_text() for t in ax.texts),
                     sorted(str(g) for g in raster.groups(
                         implant.electrode_names)))
    # Same group, same color; different groups, different colors:
    colors = ax.collections[0].get_facecolor()
    groups = raster.groups(implant.electrode_names)
    npt.assert_equal(len(np.unique(colors[groups == 0], axis=0)), 1)
    npt.assert_equal(len(np.unique(colors, axis=0)), 5)
    plt.close('all')

    # Annotation is off by default for large arrays (1500 electrodes):
    npt.assert_equal(len(SequentialRaster(2).plot(AlphaIMS()).texts), 0)
    plt.close('all')
    npt.assert_equal(
        len(SequentialRaster(2).plot(implant, annotate=False).texts), 0)
    plt.close('all')

    # Any raster can be plotted on any implant it covers, grid or not:
    for r, imp in [(SequentialRaster(3), Suprachoroidal24()),
                   (CheckerboardRaster(7).bind(PRIMAPivotal()),
                    PRIMAPivotal()),
                   (CustomRaster({n: 0 for n in ArgusII().electrode_names}),
                    ArgusII())]:
        npt.assert_equal(len(r.plot(imp).collections[0].get_paths()),
                         imp.n_electrodes)
        plt.close('all')
    # Requires an implant or electrode array:
    with pytest.raises(TypeError):
        raster.plot('ArgusII')


def test_CustomRaster():
    with pytest.raises(ValueError):
        CustomRaster([])
    with pytest.raises(TypeError):
        # A list of strings would be read as groups of single-character names:
        CustomRaster(['A1', 'A2'])

    raster = CustomRaster([['A1', 'A2'], ['A3']])
    npt.assert_equal(raster.n_groups, 2)
    npt.assert_equal(raster.groups(['A1', 'A3', 'A2']), [0, 1, 0])
    npt.assert_equal(raster.offsets(['A1', 'A3'], 10.0), [0, 5])
    # Equivalent dict form:
    same = CustomRaster({'A1': 0, 'A2': 0, 'A3': 1})
    npt.assert_equal(same.groups(['A1', 'A3', 'A2']), [0, 1, 0])
    # Every stimulated electrode must be in a group, or the current limit could
    # be violated:
    with pytest.raises(ValueError):
        raster.groups(['A1', 'B7'])
    # An electrode may not be in two groups:
    with pytest.raises(ValueError):
        CustomRaster([['A1', 'A2'], ['A2', 'A3']])
    # A fractional group index would be truncated onto a real group:
    with pytest.raises(ValueError):
        CustomRaster({'A1': 1.9, 'A2': 0})
    with pytest.raises(ValueError):
        CustomRaster({'A1': np.nan, 'A2': 0})
    with pytest.raises(ValueError):
        CustomRaster({'A1': -1, 'A2': 0})
    # The docstring example covers every Argus II electrode (otherwise `groups`
    # fails on the missing ones):
    corners = ['A1', 'A10', 'F1', 'F10']
    names = ArgusII().electrode_names
    full = CustomRaster([corners, [e for e in names if e not in corners]])
    npt.assert_equal(full.n_groups, 2)
    npt.assert_equal(np.bincount(full.groups(names)), [4, 56])


def test_Implant_raster():
    implant = ArgusII()
    # An Implant without a constructor raster has `raster` None:
    npt.assert_equal(Implant(implant.electrode_array).raster, None)
    implant.raster = SequentialRaster(6)
    npt.assert_equal(implant.raster.n_groups, 6)
    npt.assert_equal('raster' in str(implant), True)
    with pytest.raises(TypeError):
        implant.raster = 'line'
    # Also settable in the constructor:
    npt.assert_equal(
        Implant(implant.electrode_array,
                raster=SequentialRaster(3)).raster.n_groups, 3)


def test_Implant_raster_binds():
    # Assigning a raster binds it, so a geometry-dependent pattern is computed
    # and `plot` needs no argument:
    implant = ArgusII()
    raster = CheckerboardRaster(5)
    # Unbound, only `n_groups` is known:
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
    # `plot` uses the bound implant:
    npt.assert_equal(len(raster.plot().collections[0].get_paths()), 60)
    plt.close('all')

    # Rebinding to a different array recomputes the pattern for that array:
    other = Implant(ElectrodeGrid((4, 5), 400))
    other.raster = raster
    npt.assert_equal(raster.implant is other, True)
    npt.assert_almost_equal(raster.min_spacing, 400 * np.sqrt(5), decimal=3)
    npt.assert_equal(np.bincount(raster.groups(other.electrode_names)),
                     np.full(5, 4))
    with pytest.raises(ValueError):
        # Argus II electrodes are not on the newly bound grid:
        raster.groups(implant.electrode_names)
    # A raster that cannot be laid out on the array is not supported:
    with pytest.raises(NotImplementedError):
        Suprachoroidal24().raster = CheckerboardRaster(2)
    # Sequential and custom rasters bind too (no geometry needed):
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
    # Sign does not matter; the limit applies to the sum of magnitudes:
    with pytest.raises(ValueError):
        implant.prepare_stim(np.full(60, -20))
    # A single electrode is within the limit:
    npt.assert_almost_equal(implant.prepare_stim({'A1': 900}).data.max(), 900)
    # An empty stimulus passes:
    npt.assert_equal(implant.prepare_stim(None), None)


def test_Raster_units():
    names = ArgusII().electrode_names
    bare = SequentialRaster(6, group_dur=1)
    unitful = SequentialRaster(6, group_dur=1000 * us)
    npt.assert_almost_equal(unitful.group_dur, 1)
    npt.assert_equal(isinstance(unitful.group_dur, Quantity), False)
    # `period` accepts time units in both methods:
    npt.assert_almost_equal(bare.slot_dur(10), unitful.slot_dur(0.01 * sec))
    npt.assert_array_equal(bare.offsets(names, 10),
                           unitful.offsets(names, 0.01 * sec))
    # An even split (no group_dur) also accepts a unitful period:
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
    """A rastered encoding is the same with unitful or plain ms timings"""
    from pulse2percept.stimuli import AmplitudeEncoder, ImageStimulus
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    plain = ArgusII(raster=SequentialRaster(6, group_dur=1))
    unitful_raster = ArgusII(raster=SequentialRaster(6, group_dur=1000 * us))
    bare = AmplitudeEncoder(plain, amp_range=(0, 50)).encode(img)
    unitful = AmplitudeEncoder(unitful_raster,
                               amp_range=(0, 0.05 * mA)).encode(img)
    npt.assert_array_equal(bare.data, unitful.data)
    npt.assert_array_equal(bare.time, unitful.time)


def test_Raster_reads_coordinates_in_microns():
    """Raster geometry uses the array's coordinates() in um"""
    implant = ArgusII()
    raster = CheckerboardRaster(5).bind(implant)
    # `min_spacing` is in um, as returned by `coordinates()`; Argus II has a
    # 575 um pitch:
    npt.assert_allclose(raster.min_spacing, np.sqrt(5) * 575, rtol=1e-12)
    # Both entry points accept an implant or its array, and reject anything
    # without electrode coordinates:
    npt.assert_equal(
        CheckerboardRaster(5).bind(implant.electrode_array).n_groups, 5)
    for call in (lambda: CheckerboardRaster(2).bind('not an implant'),
                 lambda: SequentialRaster(2).plot('not an implant')):
        with pytest.raises(TypeError):
            call()
    ax = raster.plot(implant)
    npt.assert_equal(ax.get_xlabel(), 'x (microns)')
    plt.close('all')
