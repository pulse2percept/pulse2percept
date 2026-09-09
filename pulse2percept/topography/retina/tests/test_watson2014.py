import copy
import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.topography.retina import (Curcio1990Map,
                                             Watson2014Map,
                                             Watson2014DisplaceMap)


def test_Watson2014Map():
    trafo = Watson2014Map()
    with pytest.raises(ValueError):
        trafo.ret_to_dva(0, 0, coords='invalid')
    with pytest.raises(ValueError):
        trafo.dva_to_ret(0, 0, coords='invalid')

    # Below 15mm eccentricity, relationship is linear with slope 3.731
    npt.assert_almost_equal(trafo.ret_to_dva(0.0, 0.0), (0.0, 0.0))
    for sign in [-1, 1]:
        for exp in [2, 3, 4]:
            ret = sign * 10 ** exp  # mm
            dva = -3.731 * sign * 10 ** (exp - 3)  # dva
            npt.assert_almost_equal(trafo.ret_to_dva(0, ret)[1], dva,
                                    decimal=3 - exp)  # adjust precision
    # Below 50deg eccentricity, relationship is linear with slope 0.268
    npt.assert_almost_equal(trafo.dva_to_ret(0.0, 0.0), (0.0, 0.0))
    for sign in [-1, 1]:
        for exp in [-2, -1, 0]:
            dva = sign * 10 ** exp  # deg
            ret = -0.268 * sign * 10 ** (exp + 3)  # mm
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


def test_Watson2014DisplaceMap_meridian_by_eye():
    """`eye` decides which displacement curve a signed x coordinate gets"""
    ecc = 2.0  # near the peak, where the two curves differ most
    re, le = Watson2014DisplaceMap(), Watson2014DisplaceMap(eye='LE')
    plain = Watson2014Map()
    # What the two published curves predict on the horizontal meridian:
    nasal, temporal = [
        plain.dva_to_ret(ecc + re.watson_displacement(ecc, meridian=m), 0)[0]
        for m in ('nasal', 'temporal')]
    # Guard the test itself: the curves must be far enough apart (um) to tell
    # which one was used.
    npt.assert_equal(np.abs(nasal - temporal) > 50, True)

    npt.assert_almost_equal(re.dva_to_ret(ecc, 0)[0], nasal, decimal=6)
    npt.assert_almost_equal(re.dva_to_ret(-ecc, 0)[0], -temporal, decimal=6)
    npt.assert_almost_equal(le.dva_to_ret(ecc, 0)[0], temporal, decimal=6)
    npt.assert_almost_equal(le.dva_to_ret(-ecc, 0)[0], -nasal, decimal=6)


def test_Watson2014DisplaceMap_mirror_invariant():
    """Corresponding locations in opposite eyes mirror horizontally"""
    # (0, 0) is excluded: displacement pushes it out along theta=0, so the
    # mirrored point is not its own reflection there.
    x = np.array([-12.0, -8.0, -3.0, 0.0, 3.0, 8.0, 12.0])
    y = np.array([-6.0, -2.0, 1.0, 4.0, 2.0, -5.0, 7.0])
    x_re, y_re = Watson2014DisplaceMap(eye='RE').dva_to_ret(x, y)
    x_le, y_le = Watson2014DisplaceMap(eye='LE').dva_to_ret(-x, y)
    npt.assert_allclose(x_le, -x_re, rtol=1e-12, atol=1e-9)
    npt.assert_allclose(y_le, y_re, rtol=1e-12, atol=1e-9)


def test_Watson2014DisplaceMap_vertical_meridian():
    """x == 0 counts as nasal in both eyes"""
    y = np.array([-6.0, -1.0, 0.0, 1.0, 6.0])
    x = np.zeros_like(y)
    npt.assert_allclose(Watson2014DisplaceMap(eye='LE').dva_to_ret(x, y),
                        Watson2014DisplaceMap(eye='RE').dva_to_ret(x, y),
                        rtol=1e-12)
    npt.assert_almost_equal(Watson2014DisplaceMap().dva_to_ret(0, 0),
                            Watson2014DisplaceMap(eye='LE').dva_to_ret(0, 0))
