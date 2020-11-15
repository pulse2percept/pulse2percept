import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.utils import radial_mask, unique


def test_unique():
    a = [0, 0.001, 0.1, 0.2, 1]
    npt.assert_almost_equal(unique(a, tol=1e-6), a)
    npt.assert_almost_equal(unique(a, tol=0.001), a)
    npt.assert_almost_equal(unique(a, tol=0.1), [0, 0.1, 0.2, 1])
    npt.assert_almost_equal(unique(a, tol=1), [0, 1])

    val, idx = unique(a, tol=1e-6, return_index=True)
    npt.assert_almost_equal(val, a)
    npt.assert_almost_equal(idx, np.arange(len(a)))
    val, idx = unique(a, tol=1, return_index=True)
    npt.assert_almost_equal(val, [0, 1])
    npt.assert_almost_equal(idx, [0, 4])


def test_radial_mask():
    # Circle:
    mask = radial_mask((3, 5), mask='circle')
    npt.assert_equal(mask, np.array([[False, False,  True, False, False],
                                     [True,  True,  True,  True,  True],
                                     [False, False,  True, False, False]]))
    # Gauss:
    mask = radial_mask((3, 5), mask='gauss', sd=3)
    npt.assert_almost_equal(mask[1, 2], 1)
    npt.assert_almost_equal(mask[0, 0], 0.0001234)
    npt.assert_almost_equal(mask[0, 4], 0.0001234)
    npt.assert_almost_equal(mask[2, 0], 0.0001234)
    npt.assert_almost_equal(mask[2, 4], 0.0001234)
    npt.assert_almost_equal(mask[0, 2], 0.01111, decimal=5)
    npt.assert_almost_equal(mask[1, 0], 0.01111, decimal=5)
    npt.assert_almost_equal(mask[1, 4], 0.01111, decimal=5)
    npt.assert_almost_equal(mask[2, 2], 0.01111, decimal=5)

    with pytest.raises(ValueError):
        radial_mask((10, 10), mask='invalid')
