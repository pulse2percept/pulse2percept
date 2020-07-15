import numpy as np
import pandas as pd
import numpy.testing as npt
from pulse2percept.viz import scatter_correlation, correlation_matrix


def test_scatter_correlation():
    x = np.arange(100)
    ax = scatter_correlation(x, x)
    npt.assert_equal(len(ax.texts), 1)
    npt.assert_equal('$r$=1.000' in ax.texts[0].get_text(), True)


def test_correlation_matrix():
    df = pd.DataFrame()
    df['a'] = pd.Series(np.arange(100))
    df['b'] = pd.Series(list(df['a'][::-1]))
    ax = correlation_matrix(df)
