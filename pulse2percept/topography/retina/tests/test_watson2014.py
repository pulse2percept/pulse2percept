import copy
import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.topography.retina import (Curcio1990Map,
                                             Watson2014Map,
                                             Watson2014DisplaceMap)
from pulse2percept.utils.testing import assert_warns_msg

# Every test below is about the deprecated class itself, so its construction
# warning is expected. `pytest.warns` is unaffected by the filter.
pytestmark = pytest.mark.filterwarnings(
    'ignore:Class Watson2014DisplaceMap is deprecated:DeprecationWarning')


def test_Watson2014DisplaceMap_is_deprecated():
    assert_warns_msg(DeprecationWarning, Watson2014DisplaceMap,
                     'Class Watson2014DisplaceMap is deprecated since version '
                     '0.11.0. Use ``Montesano2020Map`` instead. Eq. 5 fits '
                     'the horizontal meridian only')


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


def test_Watson2014DisplaceMap_has_no_inverse():
    # Eq. 5 is not invertible in closed form and was never inverted
    # numerically; kept as documented behavior.
    with pytest.raises(NotImplementedError):
        Watson2014DisplaceMap().ret_to_dva(100, 100)


def test_Watson2014DisplaceMap_eye():
    npt.assert_equal(Watson2014DisplaceMap().eye, 'right')
    npt.assert_equal(Watson2014DisplaceMap(eye='LEFT').eye, 'left')
    npt.assert_equal(Watson2014DisplaceMap(eye='Right').eye, 'right')
    with pytest.raises(TypeError):
        Watson2014DisplaceMap(eye=0)
    with pytest.raises(ValueError):
        Watson2014DisplaceMap(eye='both')
    # `eye` is a regular parameter:
    npt.assert_equal('right' in repr(Watson2014DisplaceMap()), True)
    npt.assert_equal(Watson2014DisplaceMap() == Watson2014DisplaceMap(), True)
    npt.assert_equal(Watson2014DisplaceMap() ==
                     Watson2014DisplaceMap(eye='left'), False)


@pytest.mark.parametrize('eye', ('right', 'left'))
def test_Watson2014DisplaceMap_meridian(eye):
    trafo = Watson2014DisplaceMap(eye=eye)
    # An eccentricity where the two fits differ clearly:
    rho = 2.0
    expect = {m: rho + trafo.watson_displacement(rho, meridian=m)
              for m in ('nasal', 'temporal')}
    npt.assert_equal(np.isclose(expect['nasal'], expect['temporal']), False)
    # Nasal is on the right of a right eye and on the left of a left eye:
    nasal_sign = -1 if eye == 'left' else 1
    for sign, meridian in [(nasal_sign, 'nasal'), (-nasal_sign, 'temporal')]:
        ref = Watson2014Map().dva_to_ret(sign * expect[meridian], 0)
        npt.assert_almost_equal(trafo.dva_to_ret(sign * rho, 0), ref)


def test_Watson2014DisplaceMap_mirror():
    right = Watson2014DisplaceMap(eye='right')
    left = Watson2014DisplaceMap(eye='left')
    x = np.array([-8.0, -2.5, -0.5, 0.5, 2.5, 8.0])
    y = np.array([-6.0, 3.0, -1.5, 0.0, 4.5, -2.0])
    x_right, y_right = right.dva_to_ret(x, y)
    x_left, y_left = left.dva_to_ret(-x, y)
    npt.assert_almost_equal(x_left, -x_right)
    npt.assert_almost_equal(y_left, y_right)


@pytest.mark.parametrize('eye', ('right', 'left'))
def test_Watson2014DisplaceMap_vertical_meridian(eye):
    # Legacy tie-break: x == 0 takes the nasal fit for either eye.
    trafo = Watson2014DisplaceMap(eye=eye)
    for ydva in [-5.0, -1.0, 1.0, 5.0]:
        rho = np.abs(ydva) + trafo.watson_displacement(np.abs(ydva),
                                                       meridian='nasal')
        ref = Watson2014Map().dva_to_ret(0, np.sign(ydva) * rho)
        npt.assert_almost_equal(trafo.dva_to_ret(0, ydva), ref)
