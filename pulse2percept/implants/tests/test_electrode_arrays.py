import numpy as np
import pytest
import numpy.testing as npt
from collections import OrderedDict

from pulse2percept.implants import (DiskElectrode, HexElectrode,
                                    PointSource, ElectrodeArray,
                                    ElectrodeGrid)
from pulse2percept.implants import ArgusII
from pulse2percept.stimuli import ElectrodeNames, Stimulus
from pulse2percept.units import (DimensionMismatchError, Quantity, cm, deg,
                                 dva, mm, ms, rad, uA, um)


def test_ElectrodeArray():
    with pytest.raises(TypeError):
        ElectrodeArray("foo")
    with pytest.raises(TypeError):
        ElectrodeArray(OrderedDict({'A1': 0}))
    with pytest.raises(TypeError):
        ElectrodeArray([0])

    # Empty array:
    electrode_array = ElectrodeArray([])
    npt.assert_equal(electrode_array.n_electrodes, 0)
    npt.assert_equal(len(electrode_array), 0)
    with pytest.raises(IndexError):
        electrode_array[0]
    with pytest.raises(KeyError):
        electrode_array['A01']
    with pytest.raises(TypeError):
        electrode_array[PointSource(0, 0, 0)]
    ElectrodeArray([])

    # A single electrode:
    electrode_array = ElectrodeArray(PointSource(0, 1, 2))
    npt.assert_equal(electrode_array.n_electrodes, 1)
    npt.assert_equal(isinstance(electrode_array[0], PointSource), True)
    npt.assert_equal(isinstance(electrode_array[[0]], list), True)
    npt.assert_equal(isinstance(electrode_array[[0]][0], PointSource), True)
    npt.assert_almost_equal(electrode_array[0].x, 0)
    npt.assert_almost_equal(electrode_array[0].y, 1)
    npt.assert_almost_equal(electrode_array[0].z, 2)

    # Indexing:
    ps1, ps2 = PointSource(0, 0, 0), PointSource(1, 1, 1)
    electrode_array = ElectrodeArray({'A01': ps1, 'D07': ps2})
    npt.assert_equal(electrode_array['A01'], ps1)
    npt.assert_equal(electrode_array['D07'], ps2)
    # Slots:
    npt.assert_equal(hasattr(electrode_array, '__slots__'), True)
    npt.assert_equal(hasattr(electrode_array, '__dict__'), False)


def test_ElectrodeArray_add_electrode():
    electrode_array = ElectrodeArray([])
    npt.assert_equal(electrode_array.n_electrodes, 0)

    with pytest.raises(TypeError):
        electrode_array.add_electrode('A01', ElectrodeArray([]))

    # Add an electrode:
    key0 = 'A04'
    electrode_array.add_electrode(key0, PointSource(0, 1, 2))
    npt.assert_equal(electrode_array.n_electrodes, 1)
    # Both numeric and string index should work:
    for key in [key0, 0]:
        npt.assert_equal(isinstance(electrode_array[key], PointSource), True)
        npt.assert_almost_equal(electrode_array[key].x, 0)
        npt.assert_almost_equal(electrode_array[key].y, 1)
        npt.assert_almost_equal(electrode_array[key].z, 2)
    with pytest.raises(ValueError):
        # Can't add the same electrode twice:
        electrode_array.add_electrode(key0, PointSource(0, 1, 2))

    # Add another electrode:
    key1 = 'A01'
    electrode_array.add_electrode(key1, DiskElectrode(4, 5, 6, 7))
    npt.assert_equal(electrode_array.n_electrodes, 2)
    # Both numeric and string index should work:
    for key in [key1, 1]:
        npt.assert_equal(isinstance(electrode_array[key], DiskElectrode), True)
        npt.assert_almost_equal(electrode_array[key].x, 4)
        npt.assert_almost_equal(electrode_array[key].y, 5)
        npt.assert_almost_equal(electrode_array[key].z, 6)
        npt.assert_almost_equal(electrode_array[key].radius, 7)

    # We can also get a list of electrodes:
    for keys in [[key0, key1], [0, key1], [key0, 1], [0, 1]]:
        selected = electrode_array[keys]
        npt.assert_equal(isinstance(selected, list), True)
        npt.assert_equal(isinstance(selected[0], PointSource), True)
        npt.assert_equal(isinstance(selected[1], DiskElectrode), True)


def test_ElectrodeArray_remove_electrode():
    electrode_array1 = ElectrodeArray([])
    electrode_array2 = ElectrodeArray([])
    npt.assert_equal(electrode_array1.n_electrodes, 0)

    # Can't remove electrodes from empty electrodeArray
    with pytest.raises(ValueError):
        electrode_array1.remove_electrode(None)
    with pytest.raises(ValueError):
        electrode_array1.remove_electrode("foo")

    key = [0] * 4
    key[0] = 'D03'
    key[1] = 'A02'
    key[2] = 'F10'
    key[3] = 'E12'

    electrode_array1.add_electrode(key[0], PointSource(0, 1, 2))
    electrode_array1.add_electrode(key[1], PointSource(3, 4, 5))
    electrode_array1.add_electrode(key[2], PointSource(6, 7, 8))
    electrode_array1.add_electrode(key[3], PointSource(9, 10, 11))
    npt.assert_equal(electrode_array1.n_electrodes, 4)

    electrode_array2.add_electrode(key[0], PointSource(0, 1, 2))
    electrode_array2.add_electrode(key[1], PointSource(3, 4, 5))
    electrode_array2.add_electrode(key[2], PointSource(6, 7, 8))
    electrode_array2.add_electrode(key[3], PointSource(9, 10, 11))
    npt.assert_equal(electrode_array2.n_electrodes, 4)

    # Remove one electrode key[1] from the electrodeArray
    electrode_array1.remove_electrode(key[0])
    npt.assert_equal(electrode_array1.n_electrodes, 3)
    # Can't remove an electrode that has been removed
    with pytest.raises(ValueError):
        electrode_array1.remove_electrode(key[0])

    # List keeps order:
    npt.assert_equal(electrode_array1[0], electrode_array1[key[1]])
    npt.assert_equal(electrode_array1[1], electrode_array1[key[2]])
    npt.assert_equal(electrode_array1[2], electrode_array1[key[3]])

    # Other electrodes stay the same
    for k in [key[1], key[2], key[3]]:
        npt.assert_equal(electrode_array1[k].x, electrode_array2[k].x)
        npt.assert_equal(electrode_array1[k].y, electrode_array2[k].y)
        npt.assert_equal(electrode_array1[k].z, electrode_array2[k].z)

    # Remove two more electrodes from the electrodeArray
    # List keeps order
    electrode_array1.remove_electrode(key[1])
    electrode_array1.remove_electrode(key[2])
    npt.assert_equal(electrode_array1.n_electrodes, 1)
    npt.assert_equal(electrode_array1[0], electrode_array1[key[3]])

    # The last electrode stays the same
    for key in [key[3]]:
        npt.assert_equal(electrode_array1[key].x, electrode_array2[key].x)
        npt.assert_equal(electrode_array1[key].y, electrode_array2[key].y)
        npt.assert_equal(electrode_array1[key].z, electrode_array2[key].z)


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
        ElectrodeGrid((2, 3), 10, grid_type=gtype,
                      electrode_type=ElectrodeArray)
    with pytest.raises(TypeError):
        ElectrodeGrid((2, 3), 10, grid_type=gtype, electrode_type="foo")

    # Must pass in valid Orientation value:
    with pytest.raises(ValueError):
        ElectrodeGrid((2, 3), 10, grid_type=gtype, orientation="foo")
    with pytest.raises(TypeError):
        ElectrodeGrid((2, 3), 10, grid_type=gtype, orientation=False)

    # Must pass in a `radius` for a grid of DiskElectrode objects:
    gshape = (4, 5)
    spacing = 100
    grid = ElectrodeGrid(gshape, spacing, grid_type=gtype,
                         electrode_type=DiskElectrode, radius=13)
    for (_, e) in grid.electrodes.items():
        npt.assert_almost_equal(e.radius, 13)
    grid = ElectrodeGrid(gshape, spacing, grid_type=gtype,
                         electrode_type=DiskElectrode,
                         radius=np.arange(1, np.prod(gshape) + 1))
    for i, (_, e) in enumerate(grid.electrodes.items()):
        npt.assert_almost_equal(e.radius, i + 1)
    # A required electrode parameter that is missing is the electrode class's
    # error, not the grid's:
    with pytest.raises(TypeError):
        ElectrodeGrid(gshape, spacing, grid_type=gtype,
                      electrode_type=DiskElectrode)
    # Number of radii must match number of electrodes
    with pytest.raises(ValueError):
        ElectrodeGrid(gshape, spacing, grid_type=gtype,
                      electrode_type=DiskElectrode, radius=[2, 13, 14])
    # Only DiskElectrode takes a radius, not PointSource:
    with pytest.raises(TypeError):
        ElectrodeGrid(gshape, spacing, grid_type=gtype, radius=10)

    # Must pass in valid grid type:
    with pytest.raises(TypeError):
        ElectrodeGrid(gshape, spacing, grid_type=DiskElectrode)
    with pytest.raises(ValueError):
        ElectrodeGrid(gshape, spacing, grid_type='unknown')

    # Slots:
    npt.assert_equal(hasattr(grid, '__slots__'), True)
    npt.assert_equal(hasattr(grid, '__dict__'), False)


@pytest.mark.parametrize('gtype', ('rect', 'hex'))
@pytest.mark.parametrize('orientation', ('vertical', 'horizontal'))
def test_ElectrodeGrid__make_grid(gtype, orientation):
    # A valid 2x5 grid centered at (0, 500):
    x, y = 0, 500
    radius = 30
    gshape = (4, 6)
    spacing = 100
    egrid = ElectrodeGrid(gshape, spacing, x=x, y=y, grid_type='rect',
                          radius=radius, electrode_type=DiskElectrode,
                          orientation=orientation)
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
    egrid = ElectrodeGrid(gshape, spacing, z=z, grid_type=gtype,
                          radius=radius, electrode_type=DiskElectrode,
                          orientation=orientation)
    for i in egrid.electrode_objects:
        npt.assert_equal(i.z, z)

    # and when every electrode has a different z:
    z = np.arange(np.prod(gshape))
    egrid = ElectrodeGrid(gshape, spacing, z=z, grid_type=gtype,
                          radius=radius, electrode_type=DiskElectrode,
                          orientation=orientation)
    x = -1
    for i in egrid.electrode_objects:
        npt.assert_equal(i.z, x + 1)
        x = i.z

    # TODO test rotation, making sure positive angles rotate CCW
    egrid1 = ElectrodeGrid((2, 2), spacing, grid_type=gtype,
                           electrode_type=DiskElectrode, radius=radius,
                           orientation=orientation)
    egrid2 = ElectrodeGrid((2, 2), spacing, rot=10, grid_type=gtype,
                           radius=radius, electrode_type=DiskElectrode,
                           orientation=orientation)
    npt.assert_equal(egrid1["A1"].x < egrid2["A1"].x, True)
    npt.assert_equal(egrid1["A1"].y > egrid2["A1"].y, True)
    npt.assert_equal(egrid1["B2"].x > egrid2["B2"].x, True)
    npt.assert_equal(egrid1["B2"].y < egrid2["B2"].y, True)

    # Smallest possible grid:
    egrid = ElectrodeGrid((1, 1), spacing, grid_type=gtype,
                          electrode_type=DiskElectrode, radius=radius,
                          orientation=orientation)
    npt.assert_equal(egrid.shape, (1, 1))
    npt.assert_equal(egrid.n_electrodes, 1)

    # Can't have a zero-sized grid:
    with pytest.raises(ValueError):
        egrid = ElectrodeGrid((0, 0), spacing, grid_type=gtype)
    with pytest.raises(ValueError):
        egrid = ElectrodeGrid((5, 0), spacing, grid_type=gtype)

    # Verify spacing is correct:
    grid = ElectrodeGrid(gshape, spacing, grid_type=gtype,
                         electrode_type=DiskElectrode, radius=30,
                         orientation=orientation)
    npt.assert_almost_equal(np.sqrt((grid['A1'].x - grid['B1'].x) ** 2 +
                                    (grid['A1'].y - grid['B1'].y) ** 2),
                            spacing)
    npt.assert_almost_equal(np.sqrt((grid['A1'].x - grid['A2'].x) ** 2 +
                                    (grid['A1'].y - grid['A2'].y) ** 2),
                            spacing)
    if gtype == 'hex':
        npt.assert_almost_equal(np.sqrt((grid['A1'].x - grid['B2'].x) ** 2 +
                                        (grid['A1'].y - grid['B2'].y) ** 2),
                                spacing)

    # Different spacing in x and y:
    x_spc, y_spc = 50, 100
    grid = ElectrodeGrid(gshape, (x_spc, y_spc), grid_type=gtype, radius=30,
                         electrode_type=DiskElectrode,
                         orientation=orientation)
    print(gtype, orientation)
    npt.assert_almost_equal(grid['A2'].x - grid['A1'].x, x_spc)
    npt.assert_almost_equal(grid['B2'].y - grid['A2'].y, y_spc)

    # Grid has same size as 'names':
    egrid = ElectrodeGrid((1, 2), spacing, grid_type=gtype, names=('C1', '4'))
    npt.assert_equal(egrid[0, 0], egrid['C1'])
    npt.assert_equal(egrid[0, 1], egrid['4'])

    # Invalid naming conventions:
    with pytest.raises(ValueError):
        egrid = ElectrodeGrid(gshape, spacing, grid_type=gtype, names=[1])
    with pytest.raises(ValueError):
        egrid = ElectrodeGrid(gshape, spacing, grid_type=gtype, names=[])
    with pytest.raises(TypeError):
        egrid = ElectrodeGrid(gshape, spacing, grid_type=gtype, names={1})
    with pytest.raises(TypeError):
        egrid = ElectrodeGrid(gshape, spacing, grid_type=gtype, names={})
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
    egrid = ElectrodeGrid(gshape, spacing, grid_type=gtype, names=('A', '1'))
    # print([e for e in egrid.keys()])
    npt.assert_equal([e for e in egrid.electrode_names],
                     ['A1', 'A2', 'A3', 'B1', 'B2', 'B3'])
    egrid = ElectrodeGrid(gshape, spacing, grid_type=gtype, names=('1', 'A'))
    # print([e for e in egrid.keys()])
    # egrid = ElectrodeGrid(shape, names=('A', '1'))
    npt.assert_equal([e for e in egrid.electrode_names],
                     ['A1', 'B1', 'C1', 'A2', 'B2', 'C2'])

    egrid = ElectrodeGrid(gshape, spacing, grid_type=gtype, names=('1', '1'))
    # print([e for e in egrid.keys()])
    npt.assert_equal([e for e in egrid.electrode_names],
                     ['11', '12', '13', '21', '22', '23'])
    egrid = ElectrodeGrid(gshape, spacing, grid_type=gtype, names=('A', 'A'))
    # print([e for e in egrid.keys()])
    npt.assert_equal([e for e in egrid.electrode_names],
                     ['AA', 'AB', 'AC', 'BA', 'BB', 'BC'])

    # Still starts at A:
    egrid = ElectrodeGrid(gshape, spacing, grid_type=gtype, names=('B', '1'))
    npt.assert_equal([e for e in egrid.electrode_names],
                     ['A1', 'A2', 'A3', 'B1', 'B2', 'B3'])
    egrid = ElectrodeGrid(gshape, spacing, grid_type=gtype, names=('A', '2'))
    npt.assert_equal([e for e in egrid.electrode_names],
                     ['A1', 'A2', 'A3', 'B1', 'B2', 'B3'])
    # Reversal:
    egrid = ElectrodeGrid(gshape, spacing, grid_type=gtype, names=('-A', '1'))
    npt.assert_equal([e for e in egrid.electrode_names],
                     ['B1', 'B2', 'B3', 'A1', 'A2', 'A3'])
    egrid = ElectrodeGrid(gshape, spacing, grid_type=gtype, names=('A', '-1'))
    npt.assert_equal([e for e in egrid.electrode_names],
                     ['A3', 'A2', 'A1', 'B3', 'B2', 'B1'])

    # test unique names
    egrid = ElectrodeGrid(gshape, spacing, grid_type=gtype,
                          names=['53', '18', '00', '81', '11', '12'])
    npt.assert_equal([e for e in egrid.electrode_names],
                     ['53', '18', '00', '81', '11', '12'])


@pytest.mark.parametrize('gtype', ('rect', 'hex'))
@pytest.mark.parametrize('orientation', ('horizontal', 'vertical'))
@pytest.mark.parametrize('shape', [(1, 1), (1, 5), (5, 1), (1, 2), (2, 1),
                                   (2, 2), (3, 3), (3, 4), (4, 3), (4, 4),
                                   (5, 7), (7, 5)])
def test_ElectrodeGrid_is_centered(gtype, orientation, shape):
    """(x, y) is the middle of the electrode extent, whatever the shape.

    A hex grid staggers every other row, which widens its extent by half a
    pitch -- but only once both staggers are present. A single-row (or, for a
    vertical grid, single-column) hex grid used to come out a quarter pitch
    off, because the correction was applied unconditionally.
    """
    x, y = 200, -300
    coords = ElectrodeGrid(shape, 100, x=x, y=y, grid_type=gtype,
                           orientation=orientation).coordinates()
    npt.assert_almost_equal((coords[:, 0].min() + coords[:, 0].max()) / 2, x)
    npt.assert_almost_equal((coords[:, 1].min() + coords[:, 1].max()) / 2, y)


def test_ElectrodeGrid_centers_the_extent_not_the_centroid():
    """The centroid is deliberately not the thing being centered.

    An odd number of rows carries one stagger more often than the other, so
    the mean of the electrode centers sits a fraction of a pitch off. `x`/`y`
    describe where the physical array sits, so the extent is what must land
    on them.
    """
    coords = ElectrodeGrid((3, 4), 100, grid_type='hex').coordinates()
    npt.assert_almost_equal((coords[:, 0].min() + coords[:, 0].max()) / 2, 0)
    npt.assert_almost_equal(coords[:, 0].mean(), 100 / 12)


@pytest.mark.parametrize('orientation', ('horizontal', 'vertical'))
@pytest.mark.parametrize('shape', [(2, 2), (3, 4), (4, 3), (5, 5)])
def test_ElectrodeGrid_hex_is_a_triangular_lattice(orientation, shape):
    """Scalar `spacing` is the nearest-neighbor distance, in every direction

    This is what makes a hex grid hexagonal: on a rectangular grid the
    diagonal neighbor is further away than the orthogonal one.
    """
    spacing = 100
    coords = ElectrodeGrid(shape, spacing, grid_type='hex',
                           orientation=orientation).coordinates()[:, :2]
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    # Every electrode has at least one neighbor exactly `spacing` away, and
    # none any closer:
    npt.assert_almost_equal(dist.min(axis=1), spacing)
    # An interior electrode has all six:
    if min(shape) >= 3:
        n_neighbors = np.isclose(dist, spacing).sum(axis=1)
        npt.assert_equal(n_neighbors.max(), 6)


@pytest.mark.parametrize('orientation', ('horizontal', 'vertical'))
@pytest.mark.parametrize('rot', (0, 17))
def test_ElectrodeGrid_hex_bodies_tile_the_lattice(orientation, rot):
    """Hexagonal bodies face their nearest neighbors and turn with the grid

    A hexagon of apothem ``spacing / 2`` tiles the lattice only if its flats
    are perpendicular to the nearest-neighbor axes; otherwise the drawn
    pixels overlap their neighbors while leaving gaps elsewhere.
    """
    spacing = 100
    grid = ElectrodeGrid((3, 4), spacing, grid_type='hex', rot=rot,
                         orientation=orientation, electrode_type=HexElectrode,
                         apothem=spacing / 2)
    elec = grid['B2']
    npt.assert_equal(elec.orientation, orientation)
    npt.assert_almost_equal(elec.rot, rot)
    npt.assert_almost_equal(elec.width, spacing)
    # The direction a flat faces, as an angle from +x:
    flat = np.radians(rot) + (0 if orientation == 'horizontal'
                              else np.radians(90))
    coords = grid.coordinates()[:, :2]
    offsets = coords - coords[list(grid.electrodes).index('B2')]
    offsets = offsets[np.isclose(np.linalg.norm(offsets, axis=1), spacing)]
    # Every nearest neighbor sits on a flat, i.e. its bearing differs from
    # the flat direction by a multiple of 60 deg:
    bearing = np.arctan2(offsets[:, 1], offsets[:, 0]) - flat
    npt.assert_almost_equal(np.sin(3 * bearing), 0)


@pytest.mark.parametrize('gtype', ('rect', 'hex'))
def test_ElectrodeGrid_get_params(gtype):
    # When the electrode_type is 'DiskElectrode'
    # test the default value
    egrid = ElectrodeGrid((2, 3), 40, grid_type=gtype,
                          electrode_type=DiskElectrode, radius=20)
    npt.assert_equal(egrid.shape, (2, 3))
    npt.assert_equal(egrid.grid_type, gtype)


@pytest.mark.parametrize('gtype', ('rect', 'hex'))
def test_ElectrodeGrid___get_item__(gtype):
    grid = ElectrodeGrid((2, 4), 20, names=('A', '1'), grid_type=gtype,
                         electrode_type=DiskElectrode, radius=20)
    npt.assert_equal(grid[0], grid['A1'])
    npt.assert_equal(grid[0, 0], grid['A1'])
    npt.assert_equal(grid[1], grid['A2'])
    npt.assert_equal(grid[0, 1], grid['A2'])
    npt.assert_equal(grid[['A1', 1, (0, 2)]],
                     [grid['A1'], grid['A2'], grid['A3']])


@pytest.mark.parametrize('shape', [(3, 4), (5, 5), (1, 3), (30, 40), (2, 3),
                                   (1, 2), (2, 1), (1, 1)])
def test_ElectrodeGrid_canonical_names(shape):
    # A generic grid names its electrodes the same way an ImageStimulus names
    # its pixels: a letter for the row, a number for the column. Both come
    # from ElectrodeNames, so this pins them together.
    grid = ElectrodeGrid(shape, 20, names=('A', '1'))
    npt.assert_equal(grid.electrode_names,
                     np.asarray(ElectrodeNames(shape)).tolist())
    # ... and the name still addresses the electrode it describes:
    npt.assert_equal(grid['A1'], grid[0])
    npt.assert_equal(grid[ElectrodeNames(shape)[-1]], grid[np.prod(shape) - 1])


def test_ElectrodeGrid_naming_schemes():
    # The non-default schemes exist to reproduce published implants (ArgusI
    # uses ('1', 'A'), Orion ('A', '-1')). They are pinned here so that
    # routing the default through ElectrodeNames cannot disturb them.
    expected = {
        ('A', '1'): ['A1', 'A2', 'A3', 'B1', 'B2', 'B3'],
        ('1', 'A'): ['A1', 'B1', 'C1', 'A2', 'B2', 'C2'],
        ('A', '-1'): ['A3', 'A2', 'A1', 'B3', 'B2', 'B1'],
        ('-A', '1'): ['B1', 'B2', 'B3', 'A1', 'A2', 'A3'],
        ('1', '1'): ['11', '12', '13', '21', '22', '23'],
        ('A', 'A'): ['AA', 'AB', 'AC', 'BA', 'BB', 'BC'],
    }
    for names, want in expected.items():
        npt.assert_equal(ElectrodeGrid((2, 3), 20, names=names).electrode_names,
                         want)
    # An explicit list of names is passed through verbatim:
    npt.assert_equal(ElectrodeGrid((2, 2), 20,
                                   names=['w', 'x', 'y', 'z']).electrode_names,
                     ['w', 'x', 'y', 'z'])
    # A two-entry tuple is the naming scheme at every grid size, including the
    # two-electrode grids where it used to be read as the names themselves:
    npt.assert_equal(ElectrodeGrid((1, 2), 20,
                                   names=('A', '1')).electrode_names,
                     ['A1', 'A2'])
    npt.assert_equal(ElectrodeGrid((2, 1), 20,
                                   names=('A', '1')).electrode_names,
                     ['A1', 'B1'])
    npt.assert_equal(ElectrodeGrid((1, 2), 20,
                                   names=('1', 'A')).electrode_names,
                     ['A1', 'B1'])
    # On a two-electrode grid the two readings collide, so only something that
    # could actually be a scheme is read as one. Entries that cannot be name
    # the two electrodes instead:
    npt.assert_equal(ElectrodeGrid((1, 2), 20,
                                   names=('C1', '4')).electrode_names,
                     ['C1', '4'])
    # ... as does a list or array, whatever it contains:
    npt.assert_equal(ElectrodeGrid((1, 2), 20,
                                   names=['x', 'y']).electrode_names,
                     ['x', 'y'])
    npt.assert_equal(ElectrodeGrid((2, 1), 20,
                                   names=['A', '1']).electrode_names,
                     ['A', '1'])
    npt.assert_equal(ElectrodeGrid((2, 1), 20,
                                   names=np.array(['x', 'y'])).electrode_names,
                     ['x', 'y'])


def test_ElectrodeArray_coordinates():
    electrode_array = ElectrodeArray([DiskElectrode(1000, -500, 100, 50),
                                      DiskElectrode(0, 0, 0, 50)])
    npt.assert_equal(electrode_array.coordinate_unit, um)
    npt.assert_almost_equal(electrode_array.coordinates(),
                            [[1000, -500, 100], [0, 0, 0]])
    npt.assert_allclose(electrode_array.coordinates(mm), [[1, -0.5, 0.1], [0,
                                                                           0,
                                                                           0]],
                        rtol=1e-12)
    # Ordinary arrays, in electrode order, never quantities:
    npt.assert_equal(isinstance(electrode_array.coordinates(mm), np.ndarray),
                     True)
    npt.assert_equal(electrode_array.coordinates().shape, (2, 3))
    # An empty array still has the right shape, so callers can index into it:
    npt.assert_equal(ElectrodeArray([]).coordinates().shape, (0, 3))
    with pytest.raises(DimensionMismatchError):
        electrode_array.coordinates(ms)


def test_ElectrodeGrid_units():
    """Every spatial argument may be unitful, and they may be mixed"""
    bare = ElectrodeGrid((2, 3), 575.0, x=1200.0, y=-100.0,
                         z=[100., 200., 300., 400., 500., 600.], radius=112.5,
                         electrode_type=DiskElectrode)
    unitful = ElectrodeGrid((2, 3), 0.575 * mm, x=1.2 * mm, y=-100 * um,
                            z=[100 * um, 0.2 * mm, 300 * um, 0.4 * mm,
                               500 * um, 0.06 * cm],
                            radius=112.5 * um, electrode_type=DiskElectrode)
    npt.assert_allclose(unitful.coordinates(), bare.coordinates(), rtol=1e-12)
    npt.assert_allclose([e.radius for e in unitful.electrode_objects],
                        [e.radius for e in bare.electrode_objects], rtol=1e-12)
    # The grid stores plain numbers, so its repr is unchanged:
    npt.assert_almost_equal(unitful.spacing, 575.0)
    npt.assert_equal(isinstance(unitful.spacing, Quantity), False)
    # x and y spacing may be spelled differently from each other:
    split = ElectrodeGrid((2, 2), (0.5 * mm, 600 * um))
    npt.assert_allclose(split.coordinates(),
                        ElectrodeGrid((2, 2), (500., 600.)).coordinates(),
                        rtol=1e-12)
    # A per-electrode radius, too:
    radii = ElectrodeGrid((1, 2), 100, radius=[10 * um, 0.02 * mm],
                          electrode_type=DiskElectrode)
    npt.assert_allclose([e.radius for e in radii.electrode_objects], [10, 20],
                        rtol=1e-12)
    # An awkward conversion still lands where the bare spelling does:
    npt.assert_allclose(ElectrodeGrid((1, 2), 0.0417 * mm).coordinates(),
                        ElectrodeGrid((1, 2), 41.7).coordinates(), rtol=1e-12)


def test_ElectrodeGrid_dimension_errors():
    for kwargs in ({'spacing': 2 * dva}, {'spacing': 400 * ms},
                   {'x': 5 * ms}, {'y': 5 * ms}, {'z': 5 * uA},
                   {'z': [1 * um, 2 * ms, 3 * um, 4 * um]}):
        with pytest.raises(DimensionMismatchError):
            ElectrodeGrid(**{'shape': (2, 2), 'spacing': 400, **kwargs})
    with pytest.raises(DimensionMismatchError):
        ElectrodeGrid((2, 2), 400, radius=10 * uA,
                      electrode_type=DiskElectrode)
    # A rotation is an ordinary angle. `dva` is visual angle, which is not the
    # same thing, so it is refused rather than quietly reinterpreted:
    with pytest.raises(DimensionMismatchError) as excinfo:
        ElectrodeGrid((2, 2), 400, rot=5 * dva)
    npt.assert_equal("expects angle (deg)" in str(excinfo.value), True)
    # A bare `rot` still means degrees, exactly as it always has:
    phi = np.deg2rad(5)
    npt.assert_allclose(ElectrodeGrid((2, 2), 400, rot=5)['A1'].x,
                        -200 * np.cos(phi) + 200 * np.sin(phi), rtol=1e-12)


def test_ElectrodeGrid_rot_units():
    """`rot` accepts bare degrees, `deg`, and the equivalent `rad`"""
    bare = ElectrodeGrid((2, 3), 575.0, x=1200.0, rot=45)
    for rot in (45 * deg, np.pi / 4 * rad):
        npt.assert_allclose(ElectrodeGrid((2, 3), 575.0, x=1200.0,
                                          rot=rot).coordinates(),
                            bare.coordinates(), rtol=1e-12)
    # The grid stores a plain number of degrees, whatever it was given:
    npt.assert_almost_equal(
        ElectrodeGrid((2, 2), 400, rot=np.pi / 4 * rad).rot, 45)


def test_ElectrodeArray_coordinates_subset():
    """`electrodes=` selects and reorders, which is what a stimulus needs"""
    implant = ArgusII()
    electrode_array = implant.electrode_array
    names = ['F10', 'A1', 'C5']
    coords = electrode_array.coordinates(electrodes=names)
    npt.assert_equal(coords.shape, (3, 3))
    npt.assert_almost_equal(coords[:, 0], [implant[e].x for e in names])
    npt.assert_almost_equal(coords[:, 1], [implant[e].y for e in names])
    # Order follows the request, not the array:
    npt.assert_almost_equal(electrode_array.coordinates(electrodes=['A1',
                                                                    'F10']),
                            electrode_array.coordinates(
                                electrodes=['F10', 'A1'])[::-1])
    # Converted the same way as the full array:
    npt.assert_allclose(electrode_array.coordinates(mm, electrodes=names),
                        electrode_array.coordinates(electrodes=names) / 1000,
                        rtol=1e-12)
    # An electrode the array does not have says so, rather than surfacing as
    # an AttributeError somewhere downstream:
    with pytest.raises(KeyError) as excinfo:
        electrode_array.coordinates(electrodes=['A1', 'Z99'])
    npt.assert_equal('Z99' in str(excinfo.value), True)
    # Repeats are allowed: nothing here says an electrode may appear once.
    npt.assert_almost_equal(electrode_array.coordinates(electrodes=['A1',
                                                                    'A1']),
                            electrode_array.coordinates(electrodes=['A1',
                                                                    'A1']))


def test_ElectrodeArray_coordinates_selector():
    """A selector means the same thing here as it does in
    `electrode_array[...]`"""
    implant = ArgusII()
    electrode_array = implant.electrode_array
    grid = ElectrodeGrid((3, 3), 20)
    # Only a list or an array stands for several electrodes. A name, an index,
    # or a grid's (row, col) pair stands for one, and comes back as one row:
    for selector, expected in [('A1', implant['A1']), (0, implant['A1'])]:
        coords = electrode_array.coordinates(electrodes=selector)
        npt.assert_equal(coords.shape, (1, 3))
        npt.assert_almost_equal(coords[0], expected.coordinates())
    npt.assert_equal(grid.coordinates(electrodes=(0, 0)).shape, (1, 3))
    npt.assert_almost_equal(grid.coordinates(electrodes=(0, 0))[0],
                            grid[0, 0].coordinates())
    # A one-character name is a name, not two electrodes:
    single = ElectrodeArray({'7': DiskElectrode(1, 2, 3, 4)})
    npt.assert_almost_equal(single.coordinates(electrodes='7'), [[1, 2, 3]])
    # Whatever else can be iterated is a collection -- including the
    # `ElectrodeNames` a stimulus reports, which is what models pass:
    npt.assert_equal(
        grid.coordinates(electrodes=ElectrodeNames((3, 3))).shape, (9, 3))
    npt.assert_almost_equal(
        grid.coordinates(electrodes=Stimulus(np.ones(9)).electrodes),
        grid.coordinates())
    # Lists and arrays are collections, and an empty one keeps the shape:
    npt.assert_equal(grid.coordinates(electrodes=['A1', 'C3']).shape, (2, 3))
    npt.assert_equal(
        grid.coordinates(electrodes=np.array(['A1', 'C3'])).shape, (2, 3))
    npt.assert_equal(grid.coordinates(electrodes=[]).shape, (0, 3))
    # Anything the array does not have says so, however it was spelled:
    with pytest.raises(KeyError):
        grid.coordinates(electrodes='Z99')
    with pytest.raises(IndexError):
        grid.coordinates(electrodes=(0, 99))
    # A tuple is one (row, col), so a pair of names is not a collection:
    with pytest.raises(TypeError):
        grid.coordinates(electrodes=('A1', 'B2'))


def test_ElectrodeGrid_parameter_names():
    """The grid takes `grid_type`/`electrode_type`, not `type`/`etype`"""
    grid = ElectrodeGrid((10, 10), spacing=400, grid_type='hex',
                         electrode_type=DiskElectrode, radius=75)
    npt.assert_equal(grid.grid_type, 'hex')
    npt.assert_equal(grid.n_electrodes, 100)
    npt.assert_almost_equal(grid['A1'].radius, 75)
    npt.assert_equal('grid_type=' in str(grid), True)
    npt.assert_equal(hasattr(grid, 'type'), False)
    for kwargs in ({'type': 'hex'}, {'etype': DiskElectrode, 'r': 75}):
        with pytest.raises(TypeError):
            ElectrodeGrid((2, 2), 400, **kwargs)


def test_ElectrodeGrid_radius_per_electrode():
    """A scalar radius is shared; a list gives one radius per electrode"""
    npt.assert_almost_equal(
        [e.radius for e in ElectrodeGrid((2, 2), 400, radius=100,
                                         electrode_type=DiskElectrode
                                         ).electrode_objects],
        [100] * 4)
    npt.assert_almost_equal(
        [e.radius for e in ElectrodeGrid((2, 2), 400,
                                         electrode_type=DiskElectrode,
                                         radius=[100, 100, 200, 200]
                                         ).electrode_objects],
        [100, 100, 200, 200])


def test_ElectrodeGrid_electrode_subclass():
    """A subclass is built as itself, and owns its own radius contract"""
    class DefaultDisk(DiskElectrode):
        __slots__ = ()

        def __init__(self, x, y, z, radius=50, **kwargs):
            super().__init__(x, y, z, radius, **kwargs)

    grid = ElectrodeGrid((2, 2), 400, electrode_type=DefaultDisk)
    for elec in grid.electrode_objects:
        npt.assert_equal(isinstance(elec, DefaultDisk), True)
        npt.assert_almost_equal(elec.radius, 50)
    # An explicit radius still wins, and may still be given per electrode:
    npt.assert_almost_equal(
        [e.radius for e in ElectrodeGrid((1, 2), 400,
                                         electrode_type=DefaultDisk,
                                         radius=[10, 20]).electrode_objects],
        [10, 20])


def test_ElectrodeGrid_forwards_electrode_params():
    """Anything but `radius` reaches the electrode class untouched"""
    grid = ElectrodeGrid((2, 2), 400, electrode_type=HexElectrode,
                         apothem=150, activated=False)
    for elec in grid.electrode_objects:
        npt.assert_almost_equal(elec.apothem, 150)
        npt.assert_equal(elec.activated, False)
    # An unknown parameter is the electrode class's error, not a silent drop:
    with pytest.raises(TypeError):
        ElectrodeGrid((2, 2), 400, electrode_type=HexElectrode, apothem=150,
                      not_a_parameter=1)


def test_ElectrodeArray_is_a_container():
    """Lookup follows normal Python container semantics"""
    array = ElectrodeArray({'A1': DiskElectrode(0, 0, 0, 10),
                            'A2': DiskElectrode(10, 0, 0, 10),
                            'A3': DiskElectrode(20, 0, 0, 10)})
    npt.assert_equal(len(array), 3)
    npt.assert_equal(array['A1'] is array[0], True)
    npt.assert_equal(array['A3'] is array[-1], True)
    npt.assert_equal(array['A2'] is array[np.int64(1)], True)
    # Several selectors give a list, and may be mixed:
    npt.assert_equal(array[['A1', 1, -1]],
                     [array['A1'], array['A2'], array['A3']])
    npt.assert_equal(array[np.array([0, 2])], [array['A1'], array['A3']])
    # So does a slice, which is always positional:
    npt.assert_equal(array[:], array.electrode_objects)
    npt.assert_equal(array[1:], [array['A2'], array['A3']])
    npt.assert_equal(array[::-1], array.electrode_objects[::-1])
    npt.assert_equal(array[9:], [])
    with pytest.raises(KeyError):
        array['missing']
    with pytest.raises(IndexError):
        array[999]
    with pytest.raises(IndexError):
        array[-4]
    with pytest.raises(TypeError):
        array[1.2]


def test_ElectrodeArray_prefers_a_name_over_a_position():
    """Integer electrode names still resolve to the electrode of that name"""
    array = ElectrodeArray({2: DiskElectrode(0, 0, 0, 10),
                            0: DiskElectrode(10, 0, 0, 10),
                            1: DiskElectrode(20, 0, 0, 10)})
    # Names, not positions: electrode 0 is the second one in the array.
    npt.assert_almost_equal(array[2].x, 0)
    npt.assert_almost_equal(array[0].x, 10)
    npt.assert_almost_equal(array[1].x, 20)
    # Only an integer that names nothing falls through to a position:
    npt.assert_almost_equal(array[-1].x, 20)


@pytest.mark.parametrize('gtype', ('rect', 'hex'))
def test_ElectrodeGrid_is_a_container(gtype):
    """A grid adds (row, col) lookup without giving up the rest"""
    grid = ElectrodeGrid((2, 3), 20, grid_type=gtype, names=('A', '1'))
    npt.assert_equal(len(grid), 6)
    npt.assert_equal(grid[0, 0] is grid['A1'], True)
    npt.assert_equal(grid[1, 2] is grid['B3'], True)
    npt.assert_equal(grid[-1, -1] is grid['B3'], True)
    npt.assert_equal(grid[-1] is grid['B3'], True)
    npt.assert_equal(grid[[(0, 0), 'A2', 2]],
                     [grid['A1'], grid['A2'], grid['A3']])
    with pytest.raises(KeyError):
        grid['Z9']
    with pytest.raises(IndexError):
        grid[6]
    for item in ((2, 0), (0, 3), (-3, 0)):
        with pytest.raises(IndexError):
            grid[item]
    for item in (1.2, ('A1', 'B2'), (0, 1, 2)):
        with pytest.raises(TypeError):
            grid[item]


