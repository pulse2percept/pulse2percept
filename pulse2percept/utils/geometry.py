"""
`cart2pol`, `pol2cart`, `delta_angle`
"""

import numpy as np

def cart2pol(x, y):
    """Convert Cartesian to polar coordinates

    Parameters
    ----------
    x, y : scalar or array-like
        The x,y Cartesian coordinates

    Returns
    -------
    theta, rho : scalar or array-like
        The transformed polar coordinates
    """
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    """Convert polar to Cartesian coordinates

    Parameters
    ----------
    theta, rho : scalar or array-like
        The polar coordinates

    Returns
    -------
    x, y : scalar or array-like
        The transformed Cartesian coordinates
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def delta_angle(source_angle, target_angle, hi=2 * np.pi):
    """Returns the signed difference between two angles (rad)

    The difference is calculated as target_angle - source_angle.
    The difference will thus be positive if target_angle > source_angle.

    .. versionadded:: 0.7

    Parameters
    ----------
    source_angle, target_angle : array_like
        Input arrays with circular data in the range [0, hi]
    hi : float, optional
        Sets the upper bounds of the range (e.g., 2*np.pi or 360).
        Lower bound is always 0

    Returns
    -------
    The signed difference target_angle - source_angle in [0, hi]

    """
    diff = target_angle - source_angle
    def mod(a, n): return (a % n + n) % n
    return mod(diff + hi / 2, hi) - hi / 2
