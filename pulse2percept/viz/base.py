import numpy as np
import scipy.stats as spst
import matplotlib.pyplot as plt


def scatter_correlation(x, y, marker='o', color='k', ax=None, autoscale=True):
    """Scatter plots some data points and fits a regression curve to them

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
    if ax is None:
        ax = plt.gca()
    x = np.asarray(x)
    y = np.asarray(y)
    # Ignore NaN:
    isnan = np.isnan(x) | np.isnan(y)
    x = x[~isnan]
    y = y[~isnan]
    if not np.all(x.shape == y.shape):
        raise ValueError("x and y must have the same shape")
    # Need at least two data points to fit the regression curve:
    if len(x) < 2:
        return
    # Scatter plot the data:
    ax.scatter(x, y, marker=marker, s=50, c=color, edgecolors='w', alpha=0.5)
    # Fit the regression curve:
    slope, intercept, rval, pval, _ = spst.linregress(x, y)
    fit = lambda x: slope * x + intercept
    ax.plot([np.min(x), np.max(x)], [fit(np.min(x)), fit(np.max(x))], 'k--')
    # Annotate with fitting results:
    pval = ("%.2e" % pval) if pval < 0.001 else ("%.03f" % pval)
    a = ax.axis()
    ax.text(a[1], a[2], "$N$=%d\n$r$=%.3f, $p$=%s" % (len(y), rval, pval),
            va='bottom', ha='right')
    if autoscale:
        ax.autoscale(True)
    return ax


def correlation_matrix(X, cols=None, dropna=True, ax=None):
    """Plot feature correlation matrix

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
    if cols is not None:
        X = X[cols]
    # Calculate column-wise correlation:
    corr = X.corr()
    if dropna:
        corr.dropna(how='all', axis=0, inplace=True)
        corr.dropna(how='all', axis=1, inplace=True)
    if ax is None:
        ax = plt.gca()
    sns.heatmap(corr, mask=np.triu(np.ones(corr.shape), k=1).astype(np.bool),
                vmin=-1, vmax=1, cmap='coolwarm', linewidths=1, ax=ax,
                cbar_kws={"shrink": .75})
    return ax
