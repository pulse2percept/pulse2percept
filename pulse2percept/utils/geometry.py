"""`GridXY`, `RetinalCoordTrafo`, `Watson2014Trafo`, `Watson2014DisplaceTrafo`,
   `cart2pol`, `pol2cart`"""
import numpy as np
from abc import ABCMeta, abstractmethod
import scipy.stats as spst
# Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working:
from collections.abc import Sequence
from .base import PrettyPrint


class GridXY(PrettyPrint):
    """2D spatial grid

    This class generates a two-dimensional mesh grid from a range of x, y
    values and provides an iterator to loop over elements.

    Parameters
    ----------
    x_range : tuple
        (x_min, x_max), includes end point
    y_range : tuple
        (y_min, y_max), includes end point
    step : int, double
        Step size, same for x and y
    grid_type : {'rectangular', 'hexagonal'}
        The grid type

    .. note::

        The grid uses Cartesian indexing (``indexing='xy'`` for NumPy's
        ``meshgrid`` function). This implies that the grid's shape will be
        (number of y coordinates) x (number of x coordinates).

    Examples
    --------
    You can iterate through a grid as if it were a list:

    >>> grid = GridXY((0, 1), (2, 3))
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
        if isinstance(step, Sequence):
            raise TypeError("step must be a scalar.")
        # Build the grid from `x_range`, `y_range`. If the range is 0, make
        # sure that the number of steps is 1, because linspace(0, 0, num=5)
        # will return a 1x5 array:
        xdiff = np.abs(np.diff(x_range))
        nx = int(np.ceil((xdiff + 1) / step)) if xdiff != 0 else 1
        self._xflat = np.linspace(*x_range, num=nx, dtype=np.float32)
        ydiff = np.abs(np.diff(y_range))
        ny = int(np.ceil((ydiff + 1) / step)) if ydiff != 0 else 1
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

        This function converts degrees of visual angle into a retinal distance from
        the optic axis (um) using Eq. A5 in [Watson2014]_.

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
