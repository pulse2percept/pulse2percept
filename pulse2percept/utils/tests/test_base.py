import numpy as np
import copy
import pytest
import numpy.testing as npt

from pulse2percept.utils import (Frozen, FreezeError, PrettyPrint, Data, gamma,
                                 unique)


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


def test_Data():
    # Test basic usage:
    ndarray = np.arange(6).reshape((2, 3))
    data = Data(ndarray, axes=[('b', [0.3, 1]), ('a', [2, 3.1, 4.5])],
                metadata='meta')
    npt.assert_equal(data.shape, ndarray.shape)
    npt.assert_equal(hasattr(data, 'b'), True)
    npt.assert_almost_equal(data.b, [0.3, 1])
    npt.assert_equal(hasattr(data, 'a'), True)
    npt.assert_almost_equal(data.a, [2, 3.1, 4.5])
    npt.assert_equal(hasattr(data, 'metadata'), True)
    npt.assert_equal(data.metadata, 'meta')
    # Cannot overwrite any of the properties:
    for param in data._pprint_params().keys():
        with pytest.raises(AttributeError):
            setattr(data, param, 0)

    # Automatic axes:
    data = Data(ndarray)
    npt.assert_equal(hasattr(data, 'axis0'), True)
    npt.assert_equal(data.axis0, np.arange(ndarray.shape[0]))
    npt.assert_equal(hasattr(data, 'axis1'), True)
    npt.assert_equal(data.axis1, np.arange(ndarray.shape[1]))
    npt.assert_equal(hasattr(data, 'axis2'), False)

    # Order is preserved even if None:
    data = Data(np.zeros(12).reshape((3, 4, 1, 1)),
                axes=[('x', None), ('y', None), ('t', None), ('t2', 0)])
    npt.assert_equal(hasattr(data, 'x'), True)
    npt.assert_equal(data.x, [0, 1, 2])
    npt.assert_equal(hasattr(data, 'y'), True)
    npt.assert_equal(data.y, [0, 1, 2, 3])
    npt.assert_equal(hasattr(data, 't'), True)
    npt.assert_equal(data.t, None)
    npt.assert_equal(hasattr(data, 't2'), True)
    npt.assert_equal(data.t2, [0])
    npt.assert_equal(hasattr(data, 'axis0'), False)

    # Some axes given, others inferred automatically:
    data = Data(ndarray, axes=[('c', None), ('a', [0.1, 0.2, 0.5])])
    npt.assert_almost_equal(data.c, [0, 1])
    npt.assert_almost_equal(data.a, [0.1, 0.2, 0.5])

    # Invalid axes:
    with pytest.raises(TypeError):
        # Not iterable:
        Data(ndarray, axes=3)
    with pytest.raises(TypeError):
        # Not iterable:
        Data(ndarray, axes={'x': 0, 'y': [0, 1]})
    with pytest.raises(ValueError):
        # Wrong number of labels:
        Data(ndarray, axes=[('x', [0, 1])])
    with pytest.raises(ValueError):
        # Wrong number of data points for 'y':
        Data(ndarray, axes=[('x', [0, 1]), ('y', [0, 1])])
    with pytest.raises(ValueError):
        # Duplicate labels:
        Data(ndarray, axes=[('x', [0, 1]), ('x', [0, 1, 2])])

    # Special cases:
    data = Data([])
    npt.assert_equal(data.shape, (0,))
    npt.assert_equal(hasattr(data, 'axis0'), True)
    npt.assert_equal(data.axis0, [])
    data = Data([[0]])
    npt.assert_equal(data.shape, (1, 1))
    npt.assert_equal(hasattr(data, 'axis0'), True)
    npt.assert_equal(data.axis0, [0])
    npt.assert_equal(hasattr(data, 'axis1'), True)
    npt.assert_equal(data.axis1, [0])
    data = Data(0)
    npt.assert_equal(data.shape, (1,))
    npt.assert_equal(hasattr(data, 'axis0'), True)
    npt.assert_equal(data.axis0, [0])

    # Column vector:
    data = Data([0, 1])
    npt.assert_equal(data.shape, (2,))
    npt.assert_equal(hasattr(data, 'axis0'), True)
    npt.assert_equal(data.axis0, [0, 1])
    npt.assert_equal(hasattr(data, 'axis1'), False)

    # Row vector:
    data = Data([[0, 1]])
    npt.assert_equal(data.shape, (1, 2))
    npt.assert_equal(hasattr(data, 'axis0'), True)
    npt.assert_equal(data.axis0, [0])
    npt.assert_equal(hasattr(data, 'axis1'), True)
    npt.assert_equal(data.axis1, [0, 1])
    npt.assert_equal(hasattr(data, 'axis2'), False)


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


def test_unique():
    a = [0, 0.001, 0.1, 0.2, 1]
    npt.assert_almost_equal(unique(a, tol=1e-6), a)
    npt.assert_almost_equal(unique(a, tol=0.001), a)
    npt.assert_almost_equal(unique(a, tol=0.01), [0, 0.1, 0.2, 1])
    npt.assert_almost_equal(unique(a, tol=1), [0, 1])

    val, idx = unique(a, tol=1e-6, return_index=True)
    npt.assert_almost_equal(val, a)
    npt.assert_almost_equal(idx, np.arange(len(a)))
    val, idx = unique(a, tol=1, return_index=True)
    npt.assert_almost_equal(val, [0, 1])
    npt.assert_almost_equal(idx, [0, 4])
