"""`r2_score`, `circ_r2_score`"""
import numpy as np
from math import isclose
from scipy.stats import circvar
from .geometry import delta_angle


def r2_score(y_true, y_pred):
    """Calculate R² (the coefficient of determination)

    The :func:`r2_score` function computes the `coefficient of
    determination <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_,
    usually denoted as R².

    The best possible score is 1.0, lower values are worse.

    It represents the proportion of variance (of y) that has been explained by
    the independent variables in the model. It provides an indication of
    goodness of fit and therefore a measure of how well unseen samples are
    likely to be predicted by the model.

    If :math:`\\hat{y}_i` is the predicted value of the :math:`i`-th sample
    and :math:`y_i` is the corresponding true value for total :math:`n` samples,
    the estimated R² is defined as:

    .. math::

      R^2(y, \\hat{y}) = 1 - \\frac{\\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2}{\\sum_{i=1}^{n} (y_i - \\bar{y})^2}

    where :math:`\\bar{y} = \\frac{1}{n} \\sum_{i=1}^{n} y_i` and :math:`\\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2 = \\sum_{i=1}^{n} \\epsilon_i^2`.

    Note that :func:`r2_score` calculates unadjusted R² without correcting for
    bias in sample variance of y.

    .. versionadded:: 0.7

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.

    Returns
    -------
    z : float
        The R² score

    Notes
    -----
    *  If the ground-truth data has zero variance, R² will be zero.
    *  This is not a symmetric function

    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size != y_pred.size:
        raise ValueError('"y_true" (%d) and "y_pred" (%d) must have the same '
                         'size.' % (y_true.size, y_pred.size))
    if y_true.size < 2:
        raise ValueError('Need at least two data points.')
    # Sum of squares of residuals:
    ss_res = np.sum((y_true - y_pred) ** 2, dtype=np.float32)
    # Total sum of squares:
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2, dtype=np.float32)
    if isclose(ss_tot, 0):
        return 0.0  # zero variance in the ground-truth data
    return 1 - ss_res / ss_tot


def circ_r2_score(y_true, y_pred):
    """Calculate circular R² (the coefficient of determination)

    The best possible score is 1.0, lower values are worse.

    .. versionadded:: 0.7

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.

    Returns
    -------
    z : float
        The R² score

    Notes
    -----
    *  If the ground-truth data has zero variance, R² will be zero.
    *  This is not a symmetric function

    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size != y_pred.size:
        raise ValueError('"y_true" (%d) and "y_pred" (%d) must have the same '
                         'size.' % (y_true.size, y_pred.size))
    if y_true.size < 2:
        raise ValueError('Need at least two data points.')
    # Difference between two angles in [-pi/2, pi/2]:
    err = delta_angle(y_true, y_pred)
    # Use circular variance in `ss_tot`, which divides by len(y_true).
    # Therefore, we also need to divide `ss_res` by len(y_true), which
    # is the same as taking the mean instead of the sum:
    ss_res = np.mean(err ** 2, dtype=np.float32)
    ss_tot = np.asarray(circvar(y_true, low=-np.pi / 2, high=np.pi / 2),
                        dtype=np.float32)
    if isclose(ss_tot, 0):
        return 0.0  # zero variance in the ground-truth data
    return 1 - ss_res / ss_tot
