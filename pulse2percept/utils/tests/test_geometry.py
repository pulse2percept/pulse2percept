import numpy as np
import numpy.testing as npt

from pulse2percept.utils import (cart2pol, pol2cart, delta_angle)

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


def test_delta_angle():
    npt.assert_almost_equal(delta_angle(0.1, 0.2), 0.1)
    npt.assert_almost_equal(delta_angle(0.1, 0.2 + 2 * np.pi), 0.1)
    npt.assert_almost_equal(delta_angle(0.1, 0.2 - 2 * np.pi), 0.1)
    npt.assert_almost_equal(delta_angle(0.1 + 2 * np.pi, 0.2), 0.1)
    npt.assert_almost_equal(delta_angle(0.1 - 2 * np.pi, 0.2), 0.1)

    npt.assert_almost_equal(delta_angle(0.2, 0.1), -0.1)
    npt.assert_almost_equal(delta_angle(0.2, 0.1 + 2 * np.pi), -0.1)
    npt.assert_almost_equal(delta_angle(0.2, 0.1 - 2 * np.pi), -0.1)
    npt.assert_almost_equal(delta_angle(0.2 + 2 * np.pi, 0.1), -0.1)
    npt.assert_almost_equal(delta_angle(0.2 - 2 * np.pi, 0.1), -0.1)

    npt.assert_almost_equal(delta_angle(0, 2 * np.pi), 0)
    npt.assert_almost_equal(delta_angle(-np.pi, np.pi), 0)

    npt.assert_almost_equal(delta_angle(0, np.pi / 2), np.pi / 2)
    npt.assert_almost_equal(delta_angle(-np.pi / 2, 0), np.pi / 2)
    npt.assert_almost_equal(delta_angle(4*np.pi, np.pi, np.pi/2), 0)
