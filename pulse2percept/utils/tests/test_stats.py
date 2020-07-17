import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.utils import r2_score, circ_r2_score


def test_r2_score():
    # At the limit:
    npt.assert_almost_equal(r2_score([0., 1], [0., 1]), 1)
    npt.assert_almost_equal(r2_score([1., 1], [0., 1]), 0)

    y_true = np.arange(100)
    y_pred = y_true + 1
    npt.assert_almost_equal(r2_score(y_true, y_pred), 0.995, decimal=2)

    # Need same number of samples:
    with pytest.raises(ValueError):
        r2_score([0, 1], [0, 1, 2])
    # Need at least 2 samples:
    with pytest.raises(ValueError):
        r2_score([0], [1])


def test_circ_r2_score():
    # At the limit:
    npt.assert_almost_equal(circ_r2_score([0., 1], [0., 1]), 1)
    npt.assert_almost_equal(circ_r2_score([1., 1], [0., 1]), 0)

    y_true = np.arange(100) / 200.0
    y_pred = y_true + 0.0001
    npt.assert_almost_equal(circ_r2_score(y_true, y_pred), 0.995, decimal=2)

    # Need same number of samples:
    with pytest.raises(ValueError):
        circ_r2_score([0, 1], [0, 1, 2])
    # Need at least 2 samples:
    with pytest.raises(ValueError):
        circ_r2_score([0], [1])
