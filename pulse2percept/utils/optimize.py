"""`bisect`"""
import numpy as np
import logging


def bisect(y_target, func, args=None, kwargs=None, x_lo=0, x_hi=1,
           x_tol=1e-6, y_tol=1e-3, max_iter=100):
    """Binary search (bisection method) to find `x` value that gives `y_target`

    For a function ``y = func(x, *args, **kwargs)``, returns ``x_opt`` for
    which ``func(x_opt, *args, **kwargs)`` is approximately equal to
    ``y_target``.

    Parameters
    ----------
    y_target : float
        Target y value
    func, args, kwargs : optional
        The function to call along with its positional and keyword arguments
    x_lo, x_hi : float, optional
        Lower and upper bounds on ``x``
    x_tol : float, optional
        Search will stop if the range of candidate ``x`` values is smaller
        than ``x_tol``
    y_tol : float, optional
        Search will stop if ``y`` is within ``y_tol`` of ``y_target``
    max_iter : int, optional
        Maximum number of iterations to run

    Returns
    -------
    x_opt : float
        The x value such that func(x_opt) $\\approx$ y_target

    Notes
    -----
    *  Assumes ``func`` is a monotonously increasing function of ``x``.
    *  Does **not** require ``x_lo`` and ``x_hi`` to have opposite signs as in
       the conventional bisection method.

    .. versionadded:: 0.7

    """
    if x_lo >= x_hi:
        raise ValueError("x_lo (%f) must be lower than x_hi "
                         "(%f)." % (x_lo, x_hi))
    if x_tol <= 0:
        raise ValueError("x_tol (%f) must be greater than zero." % x_tol)
    if y_tol <= 0:
        raise ValueError("y_tol (%f) must be greater than zero." % y_tol)
    if max_iter <= 0:
        raise ValueError("max_iter (%d) must be greater than zero." % max_iter)
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    n_iter = 1
    while n_iter < max_iter:
        # New midpoint
        x_mid = (x_lo + x_hi) / 2.0

        # Predict `y_mid` from `x_mid` using ``func``
        y_mid = func(x_mid, *args, **kwargs)

        # `x_mid` or `y_mid` within tolerance: found a solution!
        if np.abs(y_mid - y_target) < y_tol or (x_hi - x_lo) < x_tol:
            return x_mid

        # Increment step counter
        n_iter += 1

        # New interval:
        if y_mid < y_target:
            # Same as sign change in traditional bisection method:
            # Set `x_mid` as new lower bound
            x_lo = x_mid
        else:
            x_hi = x_mid
    logging.getLogger(__name__).warning("max_iter=%d reached" % max_iter)
    return x_mid
