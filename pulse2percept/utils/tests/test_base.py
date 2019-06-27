import numpy as np
import copy
import pytest
import numpy.testing as npt

from pulse2percept.utils import (Frozen, gamma, find_files_like, cart2pol,
                                 pol2cart)


class FrozenChild(Frozen):

    def __init__(self, a, b=0):
        self.a = a
        self.b = b


def test_Frozen():
    # Cannot set attributes outside constructor:
    with pytest.raises(AttributeError):
        frozen = Frozen()
        frozen.newvar = 0

    # Setting attributes in constructor is fine:
    frozen_child = FrozenChild(1)
    npt.assert_almost_equal(frozen_child.a, 1)
    npt.assert_almost_equal(frozen_child.b, 0)
    # But not outside constructor:
    with pytest.raises(AttributeError):
        frozen_child.c = 3


def test_gamma():
    tsample = 0.005 / 1000

    with pytest.raises(ValueError):
        t, g = gamma(0, 0.1, tsample)
    with pytest.raises(ValueError):
        t, g = gamma(2, -0.1, tsample)
    with pytest.raises(ValueError):
        t, g = gamma(2, 0.1, -tsample)

    for tau in [0.001, 0.01, 0.1]:
        for n in [1, 2, 5]:
            t, g = gamma(n, tau, tsample)
            npt.assert_equal(np.arange(0, t[-1] + tsample / 2.0, tsample), t)
            if n > 1:
                npt.assert_equal(g[0], 0.0)

            # Make sure area under the curve is normalized
            npt.assert_almost_equal(np.trapz(np.abs(g), dx=tsample), 1.0,
                                    decimal=2)

            # Make sure peak sits correctly
            npt.assert_almost_equal(g.argmax() * tsample, tau * (n - 1))


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


def test_find_files_like():
    # Not sure how to check a valid pattern match, because we
    # don't know in which directory the test suite is
    # executed... so smoke test
    find_files_like(".", ".*")

    # Or fail to find an unlikely file name
    filenames = find_files_like(".", r"theresnowaythisexists\.xyz")
    npt.assert_equal(filenames, [])
