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
    npt.assert_equal(earray[0], None)
    npt.assert_equal(earray['A01'], None)
    npt.assert_equal(earray[base.PointSource(0, 0, 0)], None)

    # A single electrode:
    earray = base.ElectrodeArray(base.PointSource(0, 1, 2))
    npt.assert_equal(earray.n_electrodes, 1)
    npt.assert_equal(isinstance(earray[0], base.PointSource), True)
    npt.assert_equal(isinstance(earray[[0]], list), True)
    npt.assert_equal(isinstance(earray[[0]][0], base.PointSource), True)
    npt.assert_almost_equal(earray[0].x, 0)
    npt.assert_almost_equal(earray[0].y, 1)
    npt.assert_almost_equal(earray[0].z, 2)


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


# def test_parse_pulse_trains():
#     # Specify pulse trains in a number of different ways and make sure they
#     # are all identical after parsing

#     # Create some p2p.implants
#     argus = implants.ArgusI()
#     simple = implants.ElectrodeArray(implants.DiskElectrode(0, 0, 0, 10))

#     pt_zero = utils.TimeSeries(1, np.zeros(1000))
#     pt_nonzero = utils.TimeSeries(1, np.random.rand(1000))

#     # Test 1
#     # ------
#     # Specify wrong number of pulse trains
#     with pytest.raises(ValueError):
#         stimuli.parse_pulse_trains(pt_nonzero, argus)
#     with pytest.raises(ValueError):
#         stimuli.parse_pulse_trains([pt_nonzero], argus)
#     with pytest.raises(ValueError):
#         stimuli.parse_pulse_trains([pt_nonzero] * (argus.n_electrodes - 1),
#                                    argus)
#     with pytest.raises(ValueError):
#         stimuli.parse_pulse_trains([pt_nonzero] * 2, simple)

#     # Test 2
#     # ------
#     # Send non-zero pulse train to specific electrode
#     el_name = 'B3'
#     el_idx = argus.get_index(el_name)

#     # Specify a list of 16 pulse trains (one for each electrode)
#     pt0_in = [pt_zero] * argus.n_electrodes
#     pt0_in[el_idx] = pt_nonzero
#     pt0_out = stimuli.parse_pulse_trains(pt0_in, argus)

#     # Specify a dict with non-zero pulse trains
#     pt1_in = {el_name: pt_nonzero}
#     pt1_out = stimuli.parse_pulse_trains(pt1_in, argus)

#     # Make sure the two give the same result
#     for p0, p1 in zip(pt0_out, pt1_out):
#         npt.assert_equal(p0.data, p1.data)

#     # Test 3
#     # ------
#     # Smoke testing
#     stimuli.parse_pulse_trains([pt_zero] * argus.n_electrodes, argus)
#     stimuli.parse_pulse_trains(pt_zero, simple)
#     stimuli.parse_pulse_trains([pt_zero], simple)
