import numpy as np
import copy
import pytest
import numpy.testing as npt

from pulse2percept.utils import Frozen, FreezeError, PrettyPrint, gamma


class PrettyPrinter(PrettyPrint):

    def _pprint_params(self):
        return {}


class PrettyPrinter2(PrettyPrint):

    def _pprint_params(self):
        return {'b': None, 'a': 3}


def test_PrettyPrint():
    npt.assert_equal(str(PrettyPrinter()), "PrettyPrinter()")
    npt.assert_equal(str(PrettyPrinter2()), "PrettyPrinter2(a=3, b=None)")


class FrozenChild(Frozen):

    def __init__(self, a, b=0):
        self.a = a
        self.b = b


def test_Frozen():
    # Cannot set attributes outside constructor:
    with pytest.raises(FreezeError):
        frozen = Frozen()
        frozen.newvar = 0

    # Setting attributes in constructor is fine:
    frozen_child = FrozenChild(1)
    npt.assert_almost_equal(frozen_child.a, 1)
    npt.assert_almost_equal(frozen_child.b, 0)
    # But not outside constructor:
    with pytest.raises(FreezeError):
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
