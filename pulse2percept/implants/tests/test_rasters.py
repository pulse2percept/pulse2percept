import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.implants import (ArgusII, CustomRaster, ProsthesisSystem,
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
