import numpy as np
import collections as coll
import pytest
import numpy.testing as npt

from pulse2percept.implants import base


class ValidElectrode(base.Electrode):

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
    electrode = base.PointSource(0, 1, 2)
    npt.assert_almost_equal(electrode.x, 0)
    npt.assert_almost_equal(electrode.y, 1)
    npt.assert_almost_equal(electrode.z, 2)
    npt.assert_almost_equal(electrode.electric_potential(0, 1, 2, 1, 1), 1)
    npt.assert_almost_equal(electrode.electric_potential(0, 0, 0, 1, 1), 0.035,
                            decimal=3)


def test_DiskElectrode():
    with pytest.raises(TypeError):
        base.DiskElectrode(0, 0, 0, [1, 2])
    with pytest.raises(TypeError):
        base.DiskElectrode(0, np.array([0, 1]), 0, 1)
    # Invalid radius:
    with pytest.raises(ValueError):
        base.DiskElectrode(0, 0, 0, -5)
    # Check params:
    electrode = base.DiskElectrode(0, 1, 2, 100)
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
        base.ElectrodeArray("foo")
    with pytest.raises(TypeError):
        base.ElectrodeArray(coll.OrderedDict({'A1': 0}))
    with pytest.raises(TypeError):
        base.ElectrodeArray([0])

    # Empty array:
    earray = base.ElectrodeArray([])
    npt.assert_equal(earray.n_electrodes, 0)
    # npt.assert_equal(earray[0], None)
    npt.assert_equal(earray['A01'], None)
    with pytest.raises(TypeError):
        earray[base.PointSource(0, 0, 0)]

    # A single electrode:
    earray = base.ElectrodeArray(base.PointSource(0, 1, 2))
    npt.assert_equal(earray.n_electrodes, 1)
    npt.assert_equal(isinstance(earray[0], base.PointSource), True)
    npt.assert_equal(isinstance(earray[[0]], list), True)
    npt.assert_equal(isinstance(earray[[0]][0], base.PointSource), True)
    npt.assert_almost_equal(earray[0].x, 0)
    npt.assert_almost_equal(earray[0].y, 1)
    npt.assert_almost_equal(earray[0].z, 2)

    # Indexing:
    ps1, ps2 = base.PointSource(0, 0, 0), base.PointSource(1, 1, 1)
    earray = base.ElectrodeArray({'A01': ps1, 'D07': ps2})
    npt.assert_equal(earray['A01'], ps1)
    npt.assert_equal(earray['D07'], ps2)


def test_ElectrodeArray_add_electrode():
    earray = base.ElectrodeArray([])
    npt.assert_equal(earray.n_electrodes, 0)

    with pytest.raises(TypeError):
        earray.add_electrode('A01', base.ElectrodeArray([]))

    # Add an electrode:
    key0 = 'A04'
    earray.add_electrode(key0, base.PointSource(0, 1, 2))
    npt.assert_equal(earray.n_electrodes, 1)
    # Both numeric and string index should work:
    for key in [key0, 0]:
        npt.assert_equal(isinstance(earray[key], base.PointSource), True)
        npt.assert_almost_equal(earray[key].x, 0)
        npt.assert_almost_equal(earray[key].y, 1)
        npt.assert_almost_equal(earray[key].z, 2)
    with pytest.raises(ValueError):
        # Can't add the same electrode twice:
        earray.add_electrode(key0, base.PointSource(0, 1, 2))

    # Add another electrode:
    key1 = 'A01'
    earray.add_electrode(key1, base.DiskElectrode(4, 5, 6, 7))
    npt.assert_equal(earray.n_electrodes, 2)
    # Both numeric and string index should work:
    for key in [key1, 1]:
        npt.assert_equal(isinstance(earray[key], base.DiskElectrode), True)
        npt.assert_almost_equal(earray[key].x, 4)
        npt.assert_almost_equal(earray[key].y, 5)
        npt.assert_almost_equal(earray[key].z, 6)
        npt.assert_almost_equal(earray[key].r, 7)

    # We can also get a list of electrodes:
    for keys in [[key0, key1], [0, key1], [key0, 1], [0, 1]]:
        selected = earray[keys]
        npt.assert_equal(isinstance(selected, list), True)
        npt.assert_equal(isinstance(selected[0], base.PointSource), True)
        npt.assert_equal(isinstance(selected[1], base.DiskElectrode), True)


def test_ElectrodeArray_add_electrodes():
    earray = base.ElectrodeArray([])
    npt.assert_equal(earray.n_electrodes, 0)

    with pytest.raises(TypeError):
        earray.add_electrodes(None)

    with pytest.raises(TypeError):
        earray.add_electrodes("foo")

    # Add 2 electrodes, keep order:
    key = [0] * 6
    key[0] = 'D03'
    key[1] = 'A02'
    earray.add_electrodes({key[0]: base.PointSource(0, 1, 2)})
    earray.add_electrodes({key[1]: base.PointSource(3, 4, 5)})
    npt.assert_equal(earray[0], earray[key[0]])
    npt.assert_equal(earray[1], earray[key[1]])
    # Can't add the same key twice:
    with pytest.raises(ValueError):
        earray.add_electrodes({key[0]: base.PointSource(3, 5, 7)})

    # Add 2 more, now keep order:
    key[2] = 'F10'
    key[3] = 'E12'
    earray.add_electrodes({key[2]: base.PointSource(6, 7, 8)})
    earray.add_electrodes({key[3]: base.PointSource(9, 10, 11)})
    npt.assert_equal(earray[0], earray[key[0]])
    npt.assert_equal(earray[1], earray[key[1]])
    npt.assert_equal(earray[2], earray[key[2]])
    npt.assert_equal(earray[3], earray[key[3]])

    # List keeps order:
    earray.add_electrodes([base.PointSource(12, 13, 14),
                           base.PointSource(15, 16, 17)])
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
        base.ProsthesisSystem(base.PointSource(0, 0, 0))
    with pytest.raises(ValueError):
        base.ProsthesisSystem(base.ElectrodeArray(base.PointSource(0, 0, 0)),
                              eye='both')

    # Iterating over the electrode array:
    earray = base.ElectrodeArray(base.PointSource(0, 0, 0))
    implant = base.ProsthesisSystem(earray)
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
    with pytest.raises(AttributeError):
        base.ElectrodeGrid.set_grid("badinstantiation")
    with pytest.raises(AttributeError):
        base.ElectrodeGrid.set_grid(coll.OrderedDict({'badinstantiation': 0}))
    with pytest.raises(AttributeError):
        base.ElectrodeGrid.set_grid([0])

    # naming restrictions
    egrid = base.ElectrodeGrid(name_rows=[1])
    with pytest.raises(ValueError):
        egrid.set_grid()

    egrid = base.ElectrodeGrid(name_rows={1})
    with pytest.raises(ValueError):
        egrid.set_grid()

    egrid = base.ElectrodeGrid(name_rows={})
    with pytest.raises(NameError):
        egrid.set_grid()

    egrid = base.ElectrodeGrid(name_cols=[])
    with pytest.raises(NameError):
        egrid.set_grid()

    # Empty array:
    #earray = base.ElectrodeArray([])
    egrid = base.ElectrodeGrid(cols=0, rows=0)

    # not sure if it returns 0
    npt.assert_equal(egrid.get_x_arr(), 0)

    # inherets electrode array
    npt.assert_equal(egrid.n_electrodes, 0)

    # A single electrode:
    #earray = base.ElectrodeArray(base.PointSource(0, 1, 2))
    # redundant to pass in, just for clarity
    egrid = base.ElectrodeGrid(cols=1, rows=1)
    params = egrid.get_params()
    npt.assert_equal(params["rows"], 1)
    npt.assert_equal(params["cols"], 1)
