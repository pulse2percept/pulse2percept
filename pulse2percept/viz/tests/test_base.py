import numpy as np
import pandas as pd
import pytest
import numpy.testing as npt
import matplotlib.pyplot as plt

from pulse2percept.viz import scatter_correlation, correlation_matrix

try:
    import scipy
    has_scipy = True
except ImportError:
    has_scipy = False


def test_scatter_correlation():
    x = np.arange(100)
    _, ax = plt.subplots()
    if has_scipy:
        ax = scatter_correlation(x, x, ax=ax, show_regression=True)
        npt.assert_equal(len(ax.texts), 1)
        npt.assert_equal('$r$=1.000' in ax.texts[0].get_text(), True)
        # Ignore NaN:
        ax = scatter_correlation([0, 1, np.nan, 3], [0, 1, 2, 3],
                                 show_regression=True)
        npt.assert_equal('$r$=1.000' in ax.texts[0].get_text(), True)
    else:
        # smoke test
        ax = scatter_correlation(x, x, ax=ax, show_regression=False)
    with pytest.raises(ValueError):
        scatter_correlation(np.arange(10), np.arange(11), show_regression=False)
    with pytest.raises(ValueError):
        scatter_correlation([1], [2], show_regression=False)


def test_correlation_matrix():
    df = pd.DataFrame()
    df['a'] = pd.Series(np.arange(100))
    df['b'] = pd.Series(list(df['a'][::-1]))
    _, ax = plt.subplots()
    ax = correlation_matrix(df, ax=ax)
    with pytest.raises(TypeError):
        correlation_matrix(np.zeros((10, 20)))
