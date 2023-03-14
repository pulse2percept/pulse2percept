import copy
import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept.topography import (Curcio1990Map, Watson2014Map,
                                 Watson2014DisplaceMap)


def test_Curcio1990Map():
    # Curcio1990 uses a linear dva_to_ret conversion factor:
    for factor in [0.0, 1.0, 2.0]:
        npt.assert_almost_equal(Curcio1990Map().dva_to_ret(factor, factor),
                                (280.0 * factor, 280.0 * factor))
    for factor in [0.0, 1.0, 2.0]:
        npt.assert_almost_equal(Curcio1990Map().ret_to_dva(280.0 * factor,
                                                      280.0 * factor),
                                (factor, factor))


def test_eq_Curcio19990Map():
    curcio_map = Curcio1990Map()

    # Assert not equal for differing classes
    npt.assert_equal(curcio_map == int, False)

    # Assert equal to itself
    npt.assert_equal(curcio_map == curcio_map, True)

    # Assert equal for shallow references
    copied = curcio_map
    npt.assert_equal(curcio_map == copied, True)

    # Assert deep copies are equal
    copied = copy.deepcopy(curcio_map)
    npt.assert_equal(curcio_map == copied, True)

    # Assert differing objects aren't equal
    differing_map = Watson2014Map()
    npt.assert_equal(curcio_map == differing_map, False)


def test_Watson2014Map():
    trafo = Watson2014Map()
    with pytest.raises(ValueError):
        trafo.ret_to_dva(0, 0, coords='invalid')
    with pytest.raises(ValueError):
        trafo.dva_to_ret(0, 0, coords='invalid')

    # Below 15mm eccentricity, relationship is linear with slope 3.731
    npt.assert_equal(trafo.ret_to_dva(0.0, 0.0), (0.0, 0.0))
    for sign in [-1, 1]:
        for exp in [2, 3, 4]:
            ret = sign * 10 ** exp  # mm
            dva = 3.731 * sign * 10 ** (exp - 3)  # dva
            npt.assert_almost_equal(trafo.ret_to_dva(0, ret)[1], dva,
                                    decimal=3 - exp)  # adjust precision
    # Below 50deg eccentricity, relationship is linear with slope 0.268
    npt.assert_equal(trafo.dva_to_ret(0.0, 0.0), (0.0, 0.0))
    for sign in [-1, 1]:
        for exp in [-2, -1, 0]:
            dva = sign * 10 ** exp  # deg
            ret = 0.268 * sign * 10 ** (exp + 3)  # mm
            npt.assert_almost_equal(trafo.dva_to_ret(0, dva)[1], ret,
                                    decimal=-exp)  # adjust precision


def test_eq_Watson2014Map():
    map = Watson2014Map()

    # Assert not equal for differing classes
    npt.assert_equal(map == int, False)

    # Assert equal to itself
    npt.assert_equal(map == map, True)

    # Assert equal for shallow references
    copied = map
    npt.assert_equal(map == copied, True)

    # Assert deep copies are equal
    copied = copy.deepcopy(map)
    npt.assert_equal(map == copied, True)

    # Assert differing objects aren't equal
    differing_map = Curcio1990Map()
    npt.assert_equal(map == differing_map, False)


def test_Watson2014DisplaceMap():
    trafo = Watson2014DisplaceMap()
    with pytest.raises(ValueError):
        trafo.watson_displacement(0, meridian='invalid')
    npt.assert_almost_equal(trafo.watson_displacement(0), 0.4957506)
    npt.assert_almost_equal(trafo.watson_displacement(100), 0)

    # Check the max of the displacement function for the temporal meridian:
    radii = np.linspace(0, 30, 100)
    all_displace = trafo.watson_displacement(radii, meridian='temporal')
    npt.assert_almost_equal(np.max(all_displace), 2.153532)
    npt.assert_almost_equal(radii[np.argmax(all_displace)], 1.8181818)

    # Check the max of the displacement function for the nasal meridian:
    all_displace = trafo.watson_displacement(radii, meridian='nasal')
    npt.assert_almost_equal(np.max(all_displace), 1.9228664)
    npt.assert_almost_equal(radii[np.argmax(all_displace)], 2.1212121)
    # Smoke test
    trafo.dva_to_ret(0, 0)