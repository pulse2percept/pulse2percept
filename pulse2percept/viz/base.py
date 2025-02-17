""":py:class:`~pulse2percept.viz.correlation_matrix`, 
   :py:class:`~pulse2percept.viz.scatter_correlation`"""
import numpy as np
import pandas as pd
import scipy.stats as spst
import matplotlib.pyplot as plt


def scatter_correlation(x, y, marker='o', marker_size=50, marker_alpha=0.5,
                        color='k', text_size=10, show_slope_intercept=False,
                        ax=None, autoscale=True):
    """Scatter plots some data points and fits a regression curve to them

    .. versionadded:: 0.7

    Parameters
    ----------
    x, y : array-like
        x, y coordinates of data points to scatter
    marker : str, optional
        Marker style passed to Matplotlib's ``scatter``
    marker_size : float or array-like, shape (n, ), optional
        Marker size in points**2 passed to Matplotlib's ``scatter``
    marker_alpha : float, optional
        Marker alpha value between 0 and 1
    color : array-like or list of colors or color, optional
        Marker color passed to Matplotlib's ``scatter``
    text_size : int, optional
        Font size for inset text and axis labels
    ax : axis, optional
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
    ax.scatter(x, y, marker=marker, s=marker_size, c=color, edgecolors='w',
               alpha=marker_alpha)
    # Fit the regression curve:
    slope, intercept, rval, pval, _ = spst.linregress(x, y)
    def fit(x): return slope * x + intercept
    ax.plot([np.min(x), np.max(x)], [fit(np.min(x)), fit(np.max(x))], 'k--')
    # Annotate with fitting results:
    pval = (f"{pval:.2e}") if pval < 0.001 else (f"{pval:.03f}")
    annot_str = f"$N$={len(y)}"
    if show_slope_intercept:
        annot_str += f"\n$y$={slope:.3f}$x$+{intercept:.3f}"
    annot_str += f"\n$r$={rval:.3f}, $p$={pval}"
    a = ax.axis()
    t = ax.text(0.98 * (a[1] - a[0]) + a[0], 0.05 * (a[3] - a[2]) + a[2],
                annot_str, va='bottom', ha='right', fontsize=text_size)
    t.set_bbox(dict(facecolor='w', edgecolor='w', alpha=0.5))
    ax.set_xlabel(x_label, fontsize=text_size)
    ax.set_ylabel(y_label, fontsize=text_size)
    ax.tick_params(labelsize=text_size)
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
        raise TypeError(f'"X" must be a Pandas DataFrame, not {type(X)}.')
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
