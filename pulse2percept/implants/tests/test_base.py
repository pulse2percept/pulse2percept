import numpy as np
import collections as coll
import pytest
import numpy.testing as npt

from pulse2percept.implants import (Electrode, DiskElectrode, PointSource,
                                    ElectrodeArray, ElectrodeGrid,
                                    ProsthesisSystem)


class ValidElectrode(Electrode):

    def electric_potential(self, x, y, z):
        r = np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2 + (z - self.z) ** 2)
        return r


def test_Electrode():
    electrode = ValidElectrode(0, 1, 2)
    npt.assert_almost_equal(electrode.x, 0)
    npt.assert_almost_equal(electrode.y, 1)
    npt.assert_almost_equal(electrode.z, 2)
    npt.assert_almost_equal(electrode.electric_potential(0, 1, 2), 0)
    with pytest.raises(TypeError):
        ValidElectrode([0], 1, 2)
    with pytest.raises(TypeError):
        ValidElectrode(0, np.array([1, 2]), 2)
    with pytest.raises(TypeError):
        ValidElectrode(0, 1, [2, 3])


def test_PointSource():
    electrode = PointSource(0, 1, 2)
    npt.assert_almost_equal(electrode.x, 0)
    npt.assert_almost_equal(electrode.y, 1)
    npt.assert_almost_equal(electrode.z, 2)
    npt.assert_almost_equal(electrode.electric_potential(0, 1, 2, 1, 1), 1)
    npt.assert_almost_equal(electrode.electric_potential(0, 0, 0, 1, 1), 0.035,
                            decimal=3)


def test_DiskElectrode():
    with pytest.raises(TypeError):
        DiskElectrode(0, 0, 0, [1, 2])
    with pytest.raises(TypeError):
        DiskElectrode(0, np.array([0, 1]), 0, 1)
    # Invalid radius:
    with pytest.raises(ValueError):
        DiskElectrode(0, 0, 0, -5)
    # Check params:
    electrode = DiskElectrode(0, 1, 2, 100)
    npt.assert_almost_equal(electrode.x, 0)
    npt.assert_almost_equal(electrode.y, 1)
    npt.assert_almost_equal(electrode.z, 2)
    # On the electrode surface (z=2, x^2+y^2<=100^2)
    npt.assert_almost_equal(electrode.electric_potential(0, 1, 2, 1), 1)
    npt.assert_almost_equal(electrode.electric_potential(30, -30, 2, 1), 1)
    npt.assert_almost_equal(electrode.electric_potential(0, 101, 2, 1), 1)
    npt.assert_almost_equal(electrode.electric_potential(0, -99, 2, 1), 1)
    npt.assert_almost_equal(electrode.electric_potential(100, 1, 2, 1), 1)
    npt.assert_almost_equal(electrode.electric_potential(-100, 1, 2, 1), 1)
    # Right off the surface (z=2, x^2+y^2>100^2)
    npt.assert_almost_equal(electrode.electric_potential(0, 102, 2, 1), 0.910,
                            decimal=3)
    npt.assert_almost_equal(electrode.electric_potential(0, -100, 2, 1), 0.910,
                            decimal=3)
    # Some distance away from the electrode (z>2):
    npt.assert_almost_equal(electrode.electric_potential(0, 1, 38, 1), 0.780,
                            decimal=3)


def test_ElectrodeArray():
    with pytest.raises(TypeError):
        ElectrodeArray("foo")
    with pytest.raises(TypeError):
        ElectrodeArray(coll.OrderedDict({'A1': 0}))
    with pytest.raises(TypeError):
        ElectrodeArray([0])

    # Empty array:
    earray = ElectrodeArray([])
    npt.assert_equal(earray.n_electrodes, 0)
    # npt.assert_equal(earray[0], None)
    npt.assert_equal(earray['A01'], None)
    with pytest.raises(TypeError):
        earray[PointSource(0, 0, 0)]

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


def test_ElectrodeArray_add_electrodes():
    earray = ElectrodeArray([])
    npt.assert_equal(earray.n_electrodes, 0)

    with pytest.raises(TypeError):
        earray.add_electrodes(None)

    with pytest.raises(TypeError):
        earray.add_electrodes("foo")

    # Add 2 electrodes, keep order:
    key = [0] * 6
    key[0] = 'D03'
    key[1] = 'A02'
    earray.add_electrodes({key[0]: PointSource(0, 1, 2)})
    earray.add_electrodes({key[1]: PointSource(3, 4, 5)})
    npt.assert_equal(earray[0], earray[key[0]])
    npt.assert_equal(earray[1], earray[key[1]])
    # Can't add the same key twice:
    with pytest.raises(ValueError):
        earray.add_electrodes({key[0]: PointSource(3, 5, 7)})

    # Add 2 more, now keep order:
    key[2] = 'F10'
    key[3] = 'E12'
    earray.add_electrodes({key[2]: PointSource(6, 7, 8)})
    earray.add_electrodes({key[3]: PointSource(9, 10, 11)})
    npt.assert_equal(earray[0], earray[key[0]])
    npt.assert_equal(earray[1], earray[key[1]])
    npt.assert_equal(earray[2], earray[key[2]])
    npt.assert_equal(earray[3], earray[key[3]])

    # List keeps order:
    earray.add_electrodes([PointSource(12, 13, 14),
                           PointSource(15, 16, 17)])
    npt.assert_equal(earray[0], earray[key[0]])
    npt.assert_equal(earray[1], earray[key[1]])
    npt.assert_equal(earray[2], earray[key[2]])
    npt.assert_equal(earray[3], earray[key[3]])
    npt.assert_equal(earray[4].x, 12)
    npt.assert_equal(earray[5].x, 15)

    # Order is preserved in for loop:
    for i, (key, val) in enumerate(earray.items()):
        npt.assert_equal(earray[i], earray[key])
        npt.assert_equal(earray[i], val)


def test_ProsthesisSystem():
    # Invalid instantiations:
    with pytest.raises(TypeError):
        ProsthesisSystem(PointSource(0, 0, 0))
    with pytest.raises(ValueError):
        ProsthesisSystem(ElectrodeArray(PointSource(0, 0, 0)),
                         eye='both')

    # Iterating over the electrode array:
    earray = ElectrodeArray(PointSource(0, 0, 0))
    implant = ProsthesisSystem(earray)
    npt.assert_equal(implant.n_electrodes, 1)
    npt.assert_equal(implant[0], earray[0])
    npt.assert_equal(implant.keys(), earray.keys())
    # for i, el in enumerate(implant):
    #     npt.assert_equal(el, earray[i])

    # Set a stimulus after the constructor:
    npt.assert_equal(implant.stim, None)
    with pytest.raises(NotImplementedError):
        implant.stim = {'0': 1}
    with pytest.raises(NotImplementedError):
        implant.stim = [1]
    implant.stim = np.array([1])
    npt.assert_equal(implant.stim.ndim, 1)
    npt.assert_equal(implant.stim.dims[0], 'electrodes')
    npt.assert_equal(implant.stim.data, np.array([1]))


def test_ElectrodeGrid():
    # Must pass in tuple/list of (rows, cols) for grid shape:
    with pytest.raises(TypeError):
        ElectrodeGrid("badinstantiation")
    with pytest.raises(TypeError):
        ElectrodeGrid(coll.OrderedDict({'badinstantiation': 0}))
    with pytest.raises(ValueError):
        ElectrodeGrid([0])
    with pytest.raises(ValueError):
        ElectrodeGrid([1, 2, 3])

    # A valid 2x5 grid centered at (0, 500):
    shape = (2, 3)
    x, y = 0, 500
    spacing = 100
    egrid = ElectrodeGrid(shape, x=x, y=y, spacing=spacing)
    npt.assert_equal(egrid.shape, shape)
    npt.assert_equal(egrid.n_electrodes, np.prod(shape))
    npt.assert_equal(egrid.x, x)
    npt.assert_equal(egrid.y, y)
    npt.assert_almost_equal(egrid.spacing, spacing)
    # Make sure different electrodes have different coordinates:
    npt.assert_equal(len(np.unique([e.x for e in egrid.values()])), shape[1])
    npt.assert_equal(len(np.unique([e.y for e in egrid.values()])), shape[0])
    # Make sure the average of all x-coordinates == x:
    # (Note: egrid has all electrodes in a dictionary, with (name, object)
    # as (key, value) pairs. You can get the electrode names by iterating over
    # egrid.keys(). You can get the electrode objects by iterating over
    # egrid.values().)
    npt.assert_almost_equal(np.mean([e.x for e in egrid.values()]), x)
    # Same for y:
    npt.assert_almost_equal(np.mean([e.y for e in egrid.values()]), y)

    # Test whether egrid.z is set correctly, when z is a constant:
    z = 12
    egrid = ElectrodeGrid(shape, z=z)
    npt.assert_equal(egrid.z, z)
    for i in egrid.values():
        npt.assert_equal(i.z, z)

    # and when every electrode has a different z:
    z = np.arange(np.prod(shape))
    egrid = ElectrodeGrid(shape, z=z)
    npt.assert_equal(egrid.z, z)
    x = -1
    for i in egrid.values():
        npt.assert_equal(i.z, x + 1)
        x = i.z

    # TODO display a warning when rot > 2pi (meaning it might be in degrees).
    # I think we did this somewhere in the old Argus code

    # TODO test rotation, making sure positive angles rotate CCW
    egrid1 = ElectrodeGrid(shape=(2, 2))
    egrid2 = ElectrodeGrid(shape=(2, 2), rot=np.deg2rad(10))
    npt.assert_equal(egrid1["A1"].x < egrid2["A1"].x, True)
    npt.assert_equal(egrid1["A1"].y > egrid2["A1"].y, True)
    npt.assert_equal(egrid1["B2"].x > egrid2["B2"].x, True)
    npt.assert_equal(egrid1["B2"].y < egrid2["B2"].y, True)

    # Smallest possible grid:
    egrid = ElectrodeGrid((1, 1))
    npt.assert_equal(egrid.shape, (1, 1))
    npt.assert_equal(egrid.n_electrodes, 1)

    # Can't have a zero-sized grid:
    with pytest.raises(ValueError):
        egrid = ElectrodeGrid((0, 0))
    with pytest.raises(ValueError):
        egrid = ElectrodeGrid((5, 0))

    # Invalid naming conventions:
    with pytest.raises(ValueError):
        egrid = ElectrodeGrid(shape, names=[1])
    with pytest.raises(ValueError):
        egrid = ElectrodeGrid(shape, names=[])
    with pytest.raises(TypeError):
        egrid = ElectrodeGrid(shape, names={1})
    with pytest.raises(TypeError):
        egrid = ElectrodeGrid(shape, names={})

    # Test all naming conventions:
    egrid = ElectrodeGrid(shape, names=('A', '1'))
    print([e for e in egrid.keys()])
    npt.assert_equal([e for e in egrid.keys()],
                     ['A1', 'A2', 'A3', 'B1', 'B2', 'B3'])
    egrid = ElectrodeGrid(shape, names=('1', 'A'))
    print([e for e in egrid.keys()])
    npt.assert_equal([e for e in egrid.keys()],
                     ['A1', 'B1', 'C1', 'A2', 'B2', 'C2'])
    npt.assert_equal([e for e in egrid.keys()],
                     ['A1', 'A1', 'C1', 'A2', 'B2', 'C2'])
    egrid = ElectrodeGrid(shape, names=('1', '1'))
    #print([e for e in egrid.keys()])
    npt.assert_equal([e for e in egrid.keys()],
                     ['11', '12', '13', '21', '22', '23'])
    egrid = ElectrodeGrid(shape, names=('A', 'A'))
    #print([e for e in egrid.keys()])
    npt.assert_equal([e for e in egrid.keys()],
                     ['AA', 'AB', 'AC', 'BA', 'BB', 'BC'])

    # rows and columns start at values other than A or 1
    egrid = ElectrodeGrid(shape, names=('B', '1'))
    npt.assert_equal([e for e in egrid.keys()],
                     ['B1', 'B2', 'B3', 'C1', 'C2', 'C3'])
    egrid = ElectrodeGrid(shape, names=('A', '2'))
    npt.assert_equal([e for e in egrid.keys()],
                     ['A2', 'A3', 'A4', 'B2', 'B3', 'B4'])

    # test unique names
    egrid = ElectrodeGrid(shape, names=['53', '18', '00', '81', '11', '12'])
    npt.assert_equal([e for e in egrid.keys()],
                     ['53', '18', '00', '81', '11', '12'])
