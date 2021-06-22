import numpy as np
import pytest
import numpy.testing as npt
from collections import OrderedDict

from pulse2percept.implants import (DiskElectrode, PointSource,
                                    ElectrodeArray, ElectrodeGrid)


def test_ElectrodeArray():
    with pytest.raises(TypeError):
        ElectrodeArray("foo")
    with pytest.raises(TypeError):
        ElectrodeArray(OrderedDict({'A1': 0}))
    with pytest.raises(TypeError):
        ElectrodeArray([0])

    # Empty array:
    earray = ElectrodeArray([])
    npt.assert_equal(earray.n_electrodes, 0)
    # npt.assert_equal(earray[0], None)
    npt.assert_equal(earray['A01'], None)
    with pytest.raises(TypeError):
        earray[PointSource(0, 0, 0)]
    ElectrodeArray([])

    # A single electrode:
    earray = ElectrodeArray(PointSource(0, 1, 2))
    npt.assert_equal(earray.n_electrodes, 1)
    npt.assert_equal(isinstance(earray[0], PointSource), True)
    npt.assert_equal(isinstance(earray[[0]], list), True)
    npt.assert_equal(isinstance(earray[[0]][0], PointSource), True)
    npt.assert_almost_equal(earray[0].x, 0)
    npt.assert_almost_equal(earray[0].y, 1)
    npt.assert_almost_equal(earray[0].z, 2)

    # Indexing:
    ps1, ps2 = PointSource(0, 0, 0), PointSource(1, 1, 1)
    earray = ElectrodeArray({'A01': ps1, 'D07': ps2})
    npt.assert_equal(earray['A01'], ps1)
    npt.assert_equal(earray['D07'], ps2)
    # Slots:
    npt.assert_equal(hasattr(earray, '__slots__'), True)
    npt.assert_equal(hasattr(earray, '__dict__'), False)


def test_ElectrodeArray_add_electrode():
    earray = ElectrodeArray([])
    npt.assert_equal(earray.n_electrodes, 0)

    with pytest.raises(TypeError):
        earray.add_electrode('A01', ElectrodeArray([]))

    # Add an electrode:
    key0 = 'A04'
    earray.add_electrode(key0, PointSource(0, 1, 2))
    npt.assert_equal(earray.n_electrodes, 1)
    # Both numeric and string index should work:
    for key in [key0, 0]:
        npt.assert_equal(isinstance(earray[key], PointSource), True)
        npt.assert_almost_equal(earray[key].x, 0)
        npt.assert_almost_equal(earray[key].y, 1)
        npt.assert_almost_equal(earray[key].z, 2)
    with pytest.raises(ValueError):
        # Can't add the same electrode twice:
        earray.add_electrode(key0, PointSource(0, 1, 2))

    # Add another electrode:
    key1 = 'A01'
    earray.add_electrode(key1, DiskElectrode(4, 5, 6, 7))
    npt.assert_equal(earray.n_electrodes, 2)
    # Both numeric and string index should work:
    for key in [key1, 1]:
        npt.assert_equal(isinstance(earray[key], DiskElectrode), True)
        npt.assert_almost_equal(earray[key].x, 4)
        npt.assert_almost_equal(earray[key].y, 5)
        npt.assert_almost_equal(earray[key].z, 6)
        npt.assert_almost_equal(earray[key].r, 7)

    # We can also get a list of electrodes:
    for keys in [[key0, key1], [0, key1], [key0, 1], [0, 1]]:
        selected = earray[keys]
        npt.assert_equal(isinstance(selected, list), True)
        npt.assert_equal(isinstance(selected[0], PointSource), True)
        npt.assert_equal(isinstance(selected[1], DiskElectrode), True)


def test_ElectrodeArray_remove_electrode():
    earray1 = ElectrodeArray([])
    earray2 = ElectrodeArray([])
    npt.assert_equal(earray1.n_electrodes, 0)

    # Can't remove electrodes from empty electrodeArray
    with pytest.raises(ValueError):
        earray1.remove_electrode(None)
    with pytest.raises(ValueError):
        earray1.remove_electrode("foo")

    key = [0] * 4
    key[0] = 'D03'
    key[1] = 'A02'
    key[2] = 'F10'
    key[3] = 'E12'

    earray1.add_electrode(key[0], PointSource(0, 1, 2))
    earray1.add_electrode(key[1], PointSource(3, 4, 5))
    earray1.add_electrode(key[2], PointSource(6, 7, 8))
    earray1.add_electrode(key[3], PointSource(9, 10, 11))
    npt.assert_equal(earray1.n_electrodes, 4)

    earray2.add_electrode(key[0], PointSource(0, 1, 2))
    earray2.add_electrode(key[1], PointSource(3, 4, 5))
    earray2.add_electrode(key[2], PointSource(6, 7, 8))
    earray2.add_electrode(key[3], PointSource(9, 10, 11))
    npt.assert_equal(earray2.n_electrodes, 4)

    # Remove one electrode key[1] from the electrodeArray
    earray1.remove_electrode(key[0])
    npt.assert_equal(earray1.n_electrodes, 3)
    # Can't remove an electrode that has been removed
    with pytest.raises(ValueError):
        earray1.remove_electrode(key[0])

    # List keeps order:
    npt.assert_equal(earray1[0], earray1[key[1]])
    npt.assert_equal(earray1[1], earray1[key[2]])
    npt.assert_equal(earray1[2], earray1[key[3]])

    # Other electrodes stay the same
    for k in [key[1], key[2], key[3]]:
        npt.assert_equal(earray1[k].x, earray2[k].x)
        npt.assert_equal(earray1[k].y, earray2[k].y)
        npt.assert_equal(earray1[k].z, earray2[k].z)

    # Remove two more electrodes from the electrodeArray
    # List keeps order
    earray1.remove_electrode(key[1])
    earray1.remove_electrode(key[2])
    npt.assert_equal(earray1.n_electrodes, 1)
    npt.assert_equal(earray1[0], earray1[key[3]])

    # The last electrode stays the same
    for key in [key[3]]:
        npt.assert_equal(earray1[key].x, earray2[key].x)
        npt.assert_equal(earray1[key].y, earray2[key].y)
        npt.assert_equal(earray1[key].z, earray2[key].z)


@pytest.mark.parametrize('gtype', ('rect', 'hex'))
def test_ElectrodeGrid(gtype):
    # Must pass in tuple/list of (rows, cols) for grid shape:
    with pytest.raises(TypeError):
        ElectrodeGrid("badinstantiation")
    with pytest.raises(TypeError):
        ElectrodeGrid(OrderedDict({'badinstantiation': 0}))
    with pytest.raises(ValueError):
        ElectrodeGrid([0], 10)
    with pytest.raises(ValueError):
        ElectrodeGrid([1, 2, 3], 10)
    with pytest.raises(TypeError):
        ElectrodeGrid({'1': 2}, 10)

    # Must pass in valid Electrode type:
    with pytest.raises(TypeError):
        ElectrodeGrid((2, 3), 10, type=gtype, etype=ElectrodeArray)
    with pytest.raises(TypeError):
        ElectrodeGrid((2, 3), 10, type=gtype, etype="foo")

    # Must pass in valid Orientation value:
    with pytest.raises(ValueError):
        ElectrodeGrid((2, 3), 10, type=gtype, orientation="foo")
    with pytest.raises(TypeError):
        ElectrodeGrid((2, 3), 10, type=gtype, orientation=False)

    # Must pass in radius `r` for grid of DiskElectrode objects:
    gshape = (4, 5)
    spacing = 100
    grid = ElectrodeGrid(gshape, spacing, type=gtype, etype=DiskElectrode,
                         r=13)
    for (_, e) in grid.electrodes.items():
        npt.assert_almost_equal(e.r, 13)
    grid = ElectrodeGrid(gshape, spacing, type=gtype, etype=DiskElectrode,
                         r=np.arange(1, np.prod(gshape) + 1))
    for i, (_, e) in enumerate(grid.electrodes.items()):
        npt.assert_almost_equal(e.r, i + 1)
    with pytest.raises(ValueError):
        ElectrodeGrid(gshape, spacing, type=gtype, etype=DiskElectrode)
    with pytest.raises(ValueError):
        ElectrodeGrid(gshape, spacing, type=gtype, etype=DiskElectrode,
                      radius=10)
    # Number of radii must match number of electrodes
    with pytest.raises(ValueError):
        ElectrodeGrid(gshape, spacing, type=gtype, etype=DiskElectrode,
                      radius=[2, 13, 14])
    # Only DiskElectrode needs r, not PointSource:
    with pytest.raises(TypeError):
        ElectrodeGrid(gshape, spacing, type=gtype, r=10)

    # Must pass in radius `r` for grid of DiskElectrode objects:
    gshape = (4, 5)
    spacing = 100
    with pytest.raises(ValueError):
        ElectrodeGrid(gshape, spacing, type=gtype, etype=DiskElectrode)
    with pytest.raises(ValueError):
        ElectrodeGrid(gshape, spacing, type=gtype, etype=DiskElectrode,
                      radius=10)
    # Number of radii must match number of electrodes
    with pytest.raises(ValueError):
        ElectrodeGrid(gshape, spacing, type=gtype, etype=DiskElectrode,
                      radius=[2, 13])

    # Must pass in valid grid type:
    with pytest.raises(TypeError):
        ElectrodeGrid(gshape, spacing, type=DiskElectrode)
    with pytest.raises(ValueError):
        ElectrodeGrid(gshape, spacing, type='unknown')

    # Verify spacing is correct:
    grid = ElectrodeGrid(gshape, spacing, type=gtype, etype=DiskElectrode,
                         r=30)
    npt.assert_almost_equal(np.sqrt((grid['A1'].x - grid['A2'].x) ** 2 +
                                    (grid['A1'].y - grid['A2'].y) ** 2),
                            spacing)
    npt.assert_almost_equal(np.sqrt((grid['A1'].x - grid['B1'].x) ** 2 +
                                    (grid['A1'].y - grid['B1'].y) ** 2),
                            spacing)
    grid = ElectrodeGrid(gshape, spacing, type=gtype, orientation='vertical',
                         etype=DiskElectrode, r=30)
    npt.assert_almost_equal(np.sqrt((grid['B1'].x - grid['B2'].x) ** 2 +
                                    (grid['B1'].y - grid['B2'].y) ** 2),
                            spacing)
    npt.assert_almost_equal(np.sqrt((grid['A1'].x - grid['B1'].x) ** 2 +
                                    (grid['A1'].y - grid['B1'].y) ** 2),
                            spacing)

    # A valid 2x5 grid centered at (0, 500):
    x, y = 0, 500
    radius = 30
    egrid = ElectrodeGrid(gshape, spacing, x=x, y=y, type='rect',
                          etype=DiskElectrode, r=radius)
    npt.assert_equal(egrid.shape, gshape)
    npt.assert_equal(egrid.n_electrodes, np.prod(gshape))
    # Make sure different electrodes have different coordinates:
    npt.assert_equal(len(np.unique([e.x for e in egrid.electrode_objects])),
                     gshape[1])
    npt.assert_equal(len(np.unique([e.y for e in egrid.electrode_objects])),
                     gshape[0])
    # Make sure the average of all x-coordinates == x:
    # (Note: egrid has all electrodes in a dictionary, with (name, object)
    # as (key, value) pairs. You can get the electrode names by iterating over
    # egrid.keys(). You can get the electrode objects by iterating over
    # egrid.values().)
    npt.assert_almost_equal(np.mean([e.x for e in egrid.electrode_objects]), x)
    # Same for y:
    npt.assert_almost_equal(np.mean([e.y for e in egrid.electrode_objects]), y)

    # Test whether egrid.z is set correctly, when z is a constant:
    z = 12
    egrid = ElectrodeGrid(gshape, spacing, z=z, type=gtype,
                          etype=DiskElectrode, r=radius)
    for i in egrid.electrode_objects:
        npt.assert_equal(i.z, z)

    # and when every electrode has a different z:
    z = np.arange(np.prod(gshape))
    egrid = ElectrodeGrid(gshape, spacing, z=z, type=gtype,
                          etype=DiskElectrode, r=radius)
    x = -1
    for i in egrid.electrode_objects:
        npt.assert_equal(i.z, x + 1)
        x = i.z

    # TODO test rotation, making sure positive angles rotate CCW
    egrid1 = ElectrodeGrid((2, 2), spacing, type=gtype, etype=DiskElectrode,
                           r=radius)
    egrid2 = ElectrodeGrid((2, 2), spacing, rot=10, type=gtype,
                           etype=DiskElectrode, r=radius)
    npt.assert_equal(egrid1["A1"].x < egrid2["A1"].x, True)
    npt.assert_equal(egrid1["A1"].y > egrid2["A1"].y, True)
    npt.assert_equal(egrid1["B2"].x > egrid2["B2"].x, True)
    npt.assert_equal(egrid1["B2"].y < egrid2["B2"].y, True)

    # Smallest possible grid:
    egrid = ElectrodeGrid((1, 1), spacing, type=gtype, etype=DiskElectrode,
                          r=radius)
    npt.assert_equal(egrid.shape, (1, 1))
    npt.assert_equal(egrid.n_electrodes, 1)

    # Grid has same size as 'names':
    egrid = ElectrodeGrid((1, 2), spacing, type=gtype, names=('C1', '4'))
    npt.assert_equal(egrid[0, 0], egrid['C1'])
    npt.assert_equal(egrid[0, 1], egrid['4'])

    # Can't have a zero-sized grid:
    with pytest.raises(ValueError):
        egrid = ElectrodeGrid((0, 0), spacing, type=gtype)
    with pytest.raises(ValueError):
        egrid = ElectrodeGrid((5, 0), spacing, type=gtype)

    # Invalid naming conventions:
    with pytest.raises(ValueError):
        egrid = ElectrodeGrid(gshape, spacing, type=gtype, names=[1])
    with pytest.raises(ValueError):
        egrid = ElectrodeGrid(gshape, spacing, type=gtype, names=[])
    with pytest.raises(TypeError):
        egrid = ElectrodeGrid(gshape, spacing, type=gtype, names={1})
    with pytest.raises(TypeError):
        egrid = ElectrodeGrid(gshape, spacing, type=gtype, names={})
    with pytest.raises(TypeError):
        ElectrodeGrid(gshape, spacing, names={'1': 2})
    with pytest.raises(ValueError):
        ElectrodeGrid(gshape, spacing, names=('A', '1', 'A'))
    with pytest.raises(TypeError):
        ElectrodeGrid(gshape, spacing, names=(1, 'A'))
    with pytest.raises(TypeError):
        ElectrodeGrid(gshape, spacing, names=('A', 1))
    with pytest.raises(ValueError):
        ElectrodeGrid(gshape, spacing, names=('A', '~'))
    with pytest.raises(ValueError):
        ElectrodeGrid(gshape, spacing, names=('~', 'A'))

    # Test all naming conventions:
    gshape = (2, 3)
    egrid = ElectrodeGrid(gshape, spacing, type=gtype, names=('A', '1'))
    # print([e for e in egrid.keys()])
    npt.assert_equal([e for e in egrid.electrode_names],
                     ['A1', 'A2', 'A3', 'B1', 'B2', 'B3'])
    egrid = ElectrodeGrid(gshape, spacing, type=gtype, names=('1', 'A'))
    # print([e for e in egrid.keys()])
    # egrid = ElectrodeGrid(shape, names=('A', '1'))
    npt.assert_equal([e for e in egrid.electrode_names],
                     ['A1', 'B1', 'C1', 'A2', 'B2', 'C2'])

    egrid = ElectrodeGrid(gshape, spacing, type=gtype, names=('1', '1'))
    # print([e for e in egrid.keys()])
    npt.assert_equal([e for e in egrid.electrode_names],
                     ['11', '12', '13', '21', '22', '23'])
    egrid = ElectrodeGrid(gshape, spacing, type=gtype, names=('A', 'A'))
    # print([e for e in egrid.keys()])
    npt.assert_equal([e for e in egrid.electrode_names],
                     ['AA', 'AB', 'AC', 'BA', 'BB', 'BC'])

    # Still starts at A:
    egrid = ElectrodeGrid(gshape, spacing, type=gtype, names=('B', '1'))
    npt.assert_equal([e for e in egrid.electrode_names],
                     ['A1', 'A2', 'A3', 'B1', 'B2', 'B3'])
    egrid = ElectrodeGrid(gshape, spacing, type=gtype, names=('A', '2'))
    npt.assert_equal([e for e in egrid.electrode_names],
                     ['A1', 'A2', 'A3', 'B1', 'B2', 'B3'])

    # test unique names
    egrid = ElectrodeGrid(gshape, spacing, type=gtype,
                          names=['53', '18', '00', '81', '11', '12'])
    npt.assert_equal([e for e in egrid.electrode_names],
                     ['53', '18', '00', '81', '11', '12'])

    # Slots:
    npt.assert_equal(hasattr(egrid, '__slots__'), True)
    npt.assert_equal(hasattr(egrid, '__dict__'), False)


@pytest.mark.parametrize('gtype', ('rect', 'hex'))
def test_ElectrodeGrid_get_params(gtype):
    # When the electrode_type is 'DiskElectrode'
    # test the default value
    egrid = ElectrodeGrid((2, 3), 40, type=gtype, etype=DiskElectrode, r=20)
    npt.assert_equal(egrid.shape, (2, 3))
    npt.assert_equal(egrid.type, gtype)


@pytest.mark.parametrize('gtype', ('rect', 'hex'))
def test_ElectrodeGrid___get_item__(gtype):
    grid = ElectrodeGrid((2, 4), 20, names=('A', '1'), type=gtype,
                         etype=DiskElectrode, r=20)
    npt.assert_equal(grid[0], grid['A1'])
    npt.assert_equal(grid[0, 0], grid['A1'])
    npt.assert_equal(grid[1], grid['A2'])
    npt.assert_equal(grid[0, 1], grid['A2'])
    npt.assert_equal(grid[['A1', 1, (0, 2)]],
                     [grid['A1'], grid['A2'], grid['A3']])
