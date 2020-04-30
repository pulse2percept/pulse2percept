import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept.utils import (RetinalCoordTransform, Curcio1990Transform,
                                 GridXY, Watson2014Transform,
                                 Watson2014DisplaceTransform,
                                 cart2pol, pol2cart)


@pytest.mark.parametrize('x_range', [(0, 0), (-3, 3), (4, -2), (1, -1)])
@pytest.mark.parametrize('y_range', [(0, 0), (0, 7), (-3, 3), (2, -2)])
def test_GridXY(x_range, y_range):
    grid = GridXY(x_range, y_range, step=1, grid_type='rectangular')
    npt.assert_equal(grid.x_range, x_range)
    npt.assert_equal(grid.y_range, y_range)
    npt.assert_equal(grid.step, 1)
    npt.assert_equal(grid.type, 'rectangular')

    # Grid is created with indexing='xy', so check coordinates:
    npt.assert_equal(grid.x.shape,
                     (np.abs(np.diff(y_range)) + 1,
                      np.abs(np.diff(x_range)) + 1))
    npt.assert_equal(grid.x.shape, grid.y.shape)
    npt.assert_equal(grid.x.shape, grid.shape)
    npt.assert_almost_equal(grid.x[0, 0], x_range[0])
    npt.assert_almost_equal(grid.x[0, -1], x_range[1])
    npt.assert_almost_equal(grid.x[-1, 0], x_range[0])
    npt.assert_almost_equal(grid.x[-1, -1], x_range[1])
    npt.assert_almost_equal(grid.y[0, 0], y_range[0])
    npt.assert_almost_equal(grid.y[0, -1], y_range[0])
    npt.assert_almost_equal(grid.y[-1, 0], y_range[1])
    npt.assert_almost_equal(grid.y[-1, -1], y_range[1])


class ValidCoordTransform(RetinalCoordTransform):

    @staticmethod
    def dva2ret(dva):
        return dva

    @staticmethod
    def ret2dva(ret):
        return ret


def test_RetinalCoordTransform():
    npt.assert_almost_equal(ValidCoordTransform.dva2ret(1.1), 1.1)
    npt.assert_almost_equal(ValidCoordTransform.ret2dva(1.1), 1.1)


def test_Curcio1990Transform():
    # Curcio1990 uses a linear dva2ret conversion factor:
    for factor in [0.0, 1.0, 2.0]:
        npt.assert_almost_equal(Curcio1990Transform.dva2ret(factor),
                                280.0 * factor)
    for factor in [0.0, 1.0, 2.0]:
        npt.assert_almost_equal(Curcio1990Transform.ret2dva(280.0 * factor),
                                factor)


def test_Watson2014Transform():
    trafo = Watson2014Transform()
    # Below 15mm eccentricity, relationship is linear with slope 3.731
    npt.assert_equal(trafo.ret2dva(0.0), 0.0)
    for sign in [-1, 1]:
        for exp in [2, 3, 4]:
            ret = sign * 10 ** exp  # mm
            dva = 3.731 * sign * 10 ** (exp - 3)  # dva
            npt.assert_almost_equal(trafo.ret2dva(ret), dva,
                                    decimal=3 - exp)  # adjust precision
    # Below 50deg eccentricity, relationship is linear with slope 0.268
    npt.assert_equal(trafo.dva2ret(0.0), 0.0)
    for sign in [-1, 1]:
        for exp in [-2, -1, 0]:
            dva = sign * 10 ** exp  # deg
            ret = 0.268 * sign * 10 ** (exp + 3)  # mm
            npt.assert_almost_equal(trafo.dva2ret(dva), ret,
                                    decimal=-exp)  # adjust precision


def test_Watson2014DisplaceTransform():
    trafo = Watson2014DisplaceTransform()
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


def test_cart2pol():
    npt.assert_almost_equal(cart2pol(0, 0), (0, 0))
    npt.assert_almost_equal(cart2pol(10, 0), (0, 10))
    npt.assert_almost_equal(cart2pol(3, 4), (np.arctan(4 / 3.0), 5))
    npt.assert_almost_equal(cart2pol(4, 3), (np.arctan(3 / 4.0), 5))


def test_pol2cart():
    npt.assert_almost_equal(pol2cart(0, 0), (0, 0))
    npt.assert_almost_equal(pol2cart(0, 10), (10, 0))
    npt.assert_almost_equal(pol2cart(np.arctan(4 / 3.0), 5), (3, 4))
    npt.assert_almost_equal(pol2cart(np.arctan(3 / 4.0), 5), (4, 3))
