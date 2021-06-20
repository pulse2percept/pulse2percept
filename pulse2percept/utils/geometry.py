"""
`Grid2D`, `RetinalCoordTransform`, `Curcio1990Transform`,
`Watson2014Transform`, `Watson2014DisplaceTransform`, `cart2pol`, `pol2cart`, 'delta_angle'

"""
import numpy as np
from abc import ABCMeta, abstractmethod
import scipy.stats as spst
# Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working:
from collections.abc import Sequence
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from .base import PrettyPrint
from .constants import ZORDER


class Grid2D(PrettyPrint):
    """2D spatial grid

    This class generates a two-dimensional mesh grid from a range of x, y
    values and provides an iterator to loop over elements.

    .. versionadded:: 0.6

    Parameters
    ----------
    x_range : (x_min, x_max)
        A tuple indicating the range of x values (includes end points)
    y_range : tuple, (y_min, y_max)
        A tuple indicating the range of y values (includes end points)
    step : int, double, tuple
        Step size. If int or double, the same step will apply to both x and
        y ranges. If a tuple, it is interpreted as (x_step, y_step).
    grid_type : {'rectangular', 'hexagonal'}
        The grid type

    Notes
    -----
    *  The grid uses Cartesian indexing (``indexing='xy'`` for NumPy's
       ``meshgrid`` function). This implies that the grid's shape will be
       (number of y coordinates) x (number of x coordinates).
    *  If a range is zero, the step size is irrelevant.

    Examples
    --------
    You can iterate through a grid as if it were a list:

    >>> grid = Grid2D((0, 1), (2, 3))
    >>> for x, y in grid:
    ...     print(x, y)
    0.0 2.0
    1.0 2.0
    0.0 3.0
    1.0 3.0

    """

    def __init__(self, x_range, y_range, step=1, grid_type='rectangular'):
        self.x_range = x_range
        self.y_range = y_range
        self.step = step
        self.type = grid_type
        # These could also be their own subclasses:
        if grid_type == 'rectangular':
            self._make_rectangular_grid(x_range, y_range, step)
        elif grid_type == 'hexagonal':
            self._make_hexagonal_grid(x_range, y_range, step)
        else:
            raise ValueError("Unknown grid type '%s'." % grid_type)

    def _pprint_params(self):
        """Return dictionary of class arguments to pretty-print"""
        return {'x_range': self.x_range, 'y_range': self.y_range,
                'step': self.step, 'shape': self.shape,
                'type': self.type}

    def _make_rectangular_grid(self, x_range, y_range, step):
        """Creates a rectangular grid"""
        if not isinstance(x_range, (tuple, list, np.ndarray)):
            raise TypeError(("x_range must be a tuple, list or NumPy array, "
                             "not %s.") % type(x_range))
        if not isinstance(y_range, (tuple, list, np.ndarray)):
            raise TypeError(("y_range must be a tuple, list or NumPy array, "
                             "not %s.") % type(y_range))
        if len(x_range) != 2 or len(y_range) != 2:
            raise ValueError("x_range and y_range must have 2 elements.")
        if isinstance(step, (tuple, list, np.ndarray)):
            if len(step) != 2:
                raise ValueError("If 'step' is a tuple, it must provide "
                                 "two values (x_step, y_step), not "
                                 "%d." % len(step))
            x_step = step[0]
            y_step = step[1]
        else:
            x_step = y_step = step
        # Build the grid from `x_range`, `y_range`. If the range is 0, make
        # sure that the number of steps is 1, because linspace(0, 0, num=5)
        # will return a 1x5 array:
        xdiff = np.abs(np.diff(x_range))
        nx = int(np.round(xdiff / x_step) + 1) if xdiff != 0 else 1
        self._xflat = np.linspace(*x_range, num=nx, dtype=np.float32)
        ydiff = np.abs(np.diff(y_range))
        ny = int(np.round(ydiff / y_step) + 1) if ydiff != 0 else 1
        self._yflat = np.linspace(*y_range, num=ny, dtype=np.float32)
        self.x, self.y = np.meshgrid(self._xflat, self._yflat, indexing='xy')
        self.shape = self.x.shape
        self.reset()

    def _make_hexagonal_grid(self, x_range, y_range, step):
        raise NotImplementedError

    def __iter__(self):
        """Iterator"""
        self.reset()
        return self

    def __next__(self):
        it = self._iter
        if it >= self.x.size:
            raise StopIteration
        self._iter += 1
        return self.x.ravel()[it], self.y.ravel()[it]

    def reset(self):
        self._iter = 0

    def plot(self, transform=None, autoscale=True, zorder=None, ax=None):
        """Plot the extension of the grid

        Parameters
        ----------
        transform : function, optional
            A coordinate transform to be applied to the (x,y) coordinates of
            the grid (e.g., :py:meth:`Curcio1990Transform.dva2ret`)
        autoscale : bool, optional
            Whether to adjust the x,y limits of the plot to fit the implant
        zorder : int, optional
            The Matplotlib zorder at which to plot the grid
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            A Matplotlib axes object. If None, will either use the current axes
            (if exists) or create a new Axes object
        """
        if ax is None:
            ax = plt.gca()
        ax.set_aspect('equal')
        if autoscale:
            ax.autoscale(True)
        if zorder is None:
            zorder = ZORDER['background']

        x, y = self.x, self.y
        if transform is not None:
            x, y = transform(self.x), transform(self.y)

        if self.type == 'rectangular':
            xy = []
            for array in (x, y):
                border = []
                # Top row (left to right), not the last element:
                border += list(array[0, :-1])
                # Right column (top to bottom), not the last element:
                border += list(array[:-1, -1])
                # Bottom row (right to left), not the last element:
                border += list(array[-1, :0:-1])
                # Left column (bottom to top), all elements element:
                border += list(array[::-1, 0])
                xy.append(border)
            # Draw border:
            ax.add_patch(Polygon(np.array(xy).T, alpha=0.3, ec='k', fc='gray',
                                 ls='--', zorder=zorder))
            # This is needed in MPL 3.0.X to set the axis limit correctly:
            ax.autoscale_view()
        else:
            raise NotImplementedError
        return ax


class RetinalCoordTransform(object, metaclass=ABCMeta):
    """Base class for a retinal coordinate transform

    A template

    """

    @abstractmethod
    def dva2ret(self):
        """Convert degrees of visual angle (dva) to retinal coords (um)"""
        raise NotImplementedError

    @abstractmethod
    def ret2dva(self):
        """Convert retinal coords (um) to degrees of visual angle (dva)"""
        raise NotImplementedError


class Curcio1990Transform(RetinalCoordTransform):
    """Converts between visual angle and retinal eccentricity [Curcio1990]_"""

    @staticmethod
    def dva2ret(xdva):
        """Convert degrees of visual angle (dva) to retinal eccentricity (um)

        Assumes that one degree of visual angle is equal to 280 um on the
        retina [Curcio1990]_.
        """
        return 280.0 * xdva

    @staticmethod
    def ret2dva(xret):
        """Convert retinal eccentricity (um) to degrees of visual angle (dva)

        Assumes that one degree of visual angle is equal to 280 um on the
        retina [Curcio1990]_
        """
        return xret / 280.0


class Watson2014Transform(RetinalCoordTransform):
    """Converts between visual angle and retinal eccentricity [Watson2014]_"""

    @staticmethod
    def ret2dva(r_um):
        """Converts retinal distances (um) to visual angles (deg)

        This function converts an eccentricity measurement on the retinal
        surface(in micrometers), measured from the optic axis, into degrees
        of visual angle using Eq. A6 in [Watson2014]_.

        Parameters
        ----------
        r_um : double or array-like
            Eccentricity in microns

        Returns
        -------
        r_dva : double or array-like
            Eccentricity in degrees of visual angle (dva)
        """
        sign = np.sign(r_um)
        r_mm = 1e-3 * np.abs(r_um)
        r_deg = 3.556 * r_mm + 0.05993 * r_mm ** 2 - 0.007358 * r_mm ** 3
        r_deg += 3.027e-4 * r_mm ** 4
        return sign * r_deg

    @staticmethod
    def dva2ret(r_deg):
        """Converts visual angles (deg) into retinal distances (um)

        This function converts degrees of visual angle into a retinal distance 
        from the optic axis (um) using Eq. A5 in [Watson2014]_.

        Parameters
        ----------
        r_dva : double or array-like
            Eccentricity in degrees of visual angle (dva)

        Returns
        -------
        r_um : double or array-like
            Eccentricity in microns


        """
        sign = np.sign(r_deg)
        r_deg = np.abs(r_deg)
        r_mm = 0.268 * r_deg + 3.427e-4 * r_deg ** 2 - 8.3309e-6 * r_deg ** 3
        r_um = 1e3 * r_mm
        return sign * r_um


class Watson2014DisplaceTransform(RetinalCoordTransform):
    """Converts between visual angle and retinal eccentricity using RGC
       displacement [Watson2014]_

    Converts from eccentricity (defined as distance from a visual center) in
    degrees of visual angle (dva) to microns on the retina using Eqs. 5, A5,
    and A6 in [Watson2014]_.

    In a central retinal zone, the retinal ganglion cell (RGC) bodies are
    displaced centrifugally some distance from the inner segments of the cones
    to which they are connected through the bipolar cells, and thus from their
    receptive field. The displacement function is described in Eq. 5 of
    [Watson2014]_.

    """

    @staticmethod
    def watson_displacement(r, meridian='temporal'):
        """Ganglion cell displacement function

        Implements the ganglion cell displacement function described in Eq. 5
        of [Watson2014]_.

        Parameters
        ----------
        r : double|array-like
            Eccentricity in degrees of visual angle (dva)
        meridian : 'temporal' or 'nasal'

        Returns
        -------
        The displacement in dva experienced by ganglion cells at eccentricity
        ``r``.

        """
        if (not isinstance(meridian, (np.ndarray, str)) or
                not np.all([m in ['temporal', 'nasal']
                            for m in np.array([meridian]).ravel()])):
            raise ValueError("'meridian' must be either 'temporal' or 'nasal'")
        alpha = np.where(meridian == 'temporal', 1.8938, 2.4607)
        beta = np.where(meridian == 'temporal', 2.4598, 1.7463)
        gamma = np.where(meridian == 'temporal', 0.91565, 0.77754)
        delta = np.where(meridian == 'temporal', 14.904, 15.111)
        mu = np.where(meridian == 'temporal', -0.09386, -0.15933)
        scale = np.where(meridian == 'temporal', 12.0, 10.0)
        # Formula:
        rmubeta = (np.abs(r) - mu) / beta
        numer = delta * gamma * np.exp(-rmubeta ** gamma)
        numer *= rmubeta ** (alpha * gamma - 1)
        denom = beta * spst.gamma.pdf(alpha, 5)
        return numer / denom / scale

    def dva2ret(self, xdva, ydva):
        """Converts dva to retinal coords

        Parameters
        ----------
        xdva, ydva : double or array-like
            x,y coordinates in dva

        Returns
        -------
        xret, yret : double or array-like
            Corresponding x,y coordinates in microns
        """
        if self.eye == 'LE':
            raise NotImplementedError
        # Convert x, y (dva) into polar coordinates:
        theta, rho_dva = utils.cart2pol(xdva, ydva)
        # Add RGC displacement:
        meridian = np.where(xdva < 0, 'temporal', 'nasal')
        rho_dva += self.watson_displacement(rho_dva, meridian=meridian)
        # Convert back to x, y (dva):
        x, y = utils.pol2cart(theta, rho_dva)
        # Convert to retinal coords:
        return dva2ret(x), dva2ret(y)

    def ret2dva(self, xret, yret):
        raise NotImplementedError


def cart2pol(x, y):
    """Convert Cartesian to polar coordinates"""
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    """Convert polar to Cartesian coordinates"""
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
