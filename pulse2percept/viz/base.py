"""`correlation_matrix`, `scatter_correlation`"""
import numpy as np
import pandas as pd
import scipy.stats as spst
import matplotlib.pyplot as plt


def scatter_correlation(x, y, marker='o', color='k', ax=None, autoscale=True):
    """Scatter plots some data points and fits a regression curve to them

    .. versionadded:: 0.7

    Parameters
    ----------
    x, y : array-like
        x, y coordinates of data points to scatter
    marker : str
        Matplotlib marker style
    color : str
        Matplotlib marker color
    ax : axis
        Matplotlib axis
    autoscale : {True, False}
        Flag whether to automatically adjust the axis limits

    """
    # If data are Pandas series, use their names to label the axes:
    x_label = x.name if hasattr(x, 'name') else ''
    y_label = y.name if hasattr(y, 'name') else ''
    x = np.asarray(x)
    y = np.asarray(y)
    # Ignore NaN:
    isnan = np.isnan(x) | np.isnan(y)
    x = x[~isnan]
    y = y[~isnan]
    if not np.all(x.shape == y.shape):
        raise ValueError("x and y must have the same shape")
    if len(x) < 2:
        raise ValueError("x and y must at least have 2 data points.")
    # Scatter plot the data:
    if ax is None:
        ax = plt.gca()
    ax.scatter(x, y, marker=marker, s=50, c=color, edgecolors='w', alpha=0.5)
    # Fit the regression curve:
    slope, intercept, rval, pval, _ = spst.linregress(x, y)
    def fit(x): return slope * x + intercept
    ax.plot([np.min(x), np.max(x)], [fit(np.min(x)), fit(np.max(x))], 'k--')
    # Annotate with fitting results:
    pval = ("%.2e" % pval) if pval < 0.001 else ("%.03f" % pval)
    a = ax.axis()
    ax.text(a[1], a[2], "$N$=%d\n$r$=%.3f, $p$=%s" % (len(y), rval, pval),
            va='bottom', ha='right')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if autoscale:
        ax.autoscale(True)
    return ax


def correlation_matrix(X, cols=None, dropna=True, ax=None):
    """Plot feature correlation matrix (requires seaborn)

    .. versionadded:: 0.7

    Parameters
    ----------
    X : pd.DataFrame
        Data matrix as a Pandas DataFrame
    cols : list or None
        List of columns to include in the correlation matrix
    dropna : {True, False}
        Flag whether to drop columns or rows with NaN values
    ax : matplotlib.axes.Axes or list thereof; optional, default: None
        A Matplotlib Axes object or a list thereof (one per electrode to
        plot). If None, a new Axes object will be created.

    """
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError("This function requires seaborn. "
                          "You can install it via $ pip install seaborn.")
    if not isinstance(X, pd.DataFrame):
        raise TypeError('"X" must be a Pandas DataFrame, not %s.' % type(X))
    if cols is not None:
        X = X[cols]
    # Calculate column-wise correlation:
    corr = X.corr()
    if dropna:
        corr.dropna(how='all', axis=0, inplace=True)
        corr.dropna(how='all', axis=1, inplace=True)
    if ax is None:
        ax = plt.gca()
    sns.heatmap(corr, mask=np.triu(np.ones(corr.shape), k=1).astype(bool),
                vmin=-1, vmax=1, cmap='coolwarm', linewidths=1, ax=ax,
                cbar_kws={"shrink": .75})
    return ax
