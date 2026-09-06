import copy
import numpy.testing as npt

from pulse2percept.topography.retina import (Curcio1990Map,
                                             Watson2014Map)


def test_Curcio1990Map():
    # Curcio1990 uses a linear dva_to_ret conversion factor:
    for factor in [0.0, 1.0, 2.0]:
        npt.assert_almost_equal(Curcio1990Map().dva_to_ret(factor, factor),
                                (280.0 * factor, -280.0 * factor))
    for factor in [0.0, 1.0, 2.0]:
        npt.assert_almost_equal(Curcio1990Map().ret_to_dva(280.0 * factor,
                                                      -280.0 * factor),
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
