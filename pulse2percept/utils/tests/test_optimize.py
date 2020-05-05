import numpy as np
import numpy.testing as npt
from pulse2percept.utils import bisect


def test_bisect():
    func = lambda x: 1.0 / (1.0 + np.exp(-x))
    for y_target in [0.0, 0.111, 0.586, 0.857, 1.0]:
        for y_tol in [0.1, 0.01, 0.001, 0.0001]:
            x_mid = bisect(y_target, func, x_lo=-10, x_hi=10,
                           x_tol=1e-6, y_tol=y_tol,
                           max_iter=100)
            decimal = int(np.abs(np.log10(y_tol))) - 1
            npt.assert_almost_equal(func(x_mid), y_target, decimal=decimal)

    with pytest.raises(ValueError):
        bisect(0, func, x_lo=1, x_hi=0)
    with pytest.raises(ValueError):
        bisect(0, func, x_tol=-1)
    with pytest.raises(ValueError):
        bisect(0, func, y_tol=-1)
    with pytest.raises(ValueError):
        bisect(0, func, max_iter=0)
