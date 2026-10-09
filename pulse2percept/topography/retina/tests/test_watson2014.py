import copy
import numpy.testing as npt
import pytest

import pulse2percept.topography.retina as retina
from pulse2percept.topography.retina import Curcio1990Map, Watson2014Map


def test_Watson2014DisplaceMap_removed():
    # Removed in 0.12; use Montesano2020Map:
    npt.assert_equal(hasattr(retina, 'Watson2014DisplaceMap'), False)


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
