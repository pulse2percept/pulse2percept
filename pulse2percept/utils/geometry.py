"""
`Grid2D`, `VisualFieldMap`, `Curcio1990Map`, `Watson2014Map`,
`Watson2014DisplaceMap`, `cart2pol`, `pol2cart`, `delta_angle`

"""
import numpy as np
from abc import ABCMeta, abstractmethod
import scipy.stats as spst
from scipy.spatial import ConvexHull
# Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working:
from collections.abc import Sequence
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

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
            raise ValueError(f"Unknown grid type '{grid_type}'.")

    def _pprint_params(self):
        """Return dictionary of class arguments to pretty-print"""
        return {'x_range': self.x_range, 'y_range': self.y_range,
                'step': self.step, 'shape': self.shape,
                'type': self.type}

    def _make_rectangular_grid(self, x_range, y_range, step):
        """Creates a rectangular grid"""
        if not isinstance(x_range, (tuple, list, np.ndarray)):
            raise TypeError((f"x_range must be a tuple, list or NumPy array, "
                             f"not {type(x_range)}."))
        if not isinstance(y_range, (tuple, list, np.ndarray)):
            raise TypeError((f"y_range must be a tuple, list or NumPy array, "
                             f"not {type(y_range)}."))
        if len(x_range) != 2 or len(y_range) != 2:
            raise ValueError("x_range and y_range must have 2 elements.")
        if isinstance(step, (tuple, list, np.ndarray)):
            if len(step) != 2:
                raise ValueError(f"If 'step' is a tuple, it must provide "
                                 f"two values (x_step, y_step), not "
                                 f"{len(step)}.")
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

    def plot(self, transform=None, label=None, style='hull', autoscale=True,
             zorder=None, ax=None, figsize=None, fc='gray'):
        """Plot the extension of the grid

        Parameters
        ----------
        transform : function, optional
            A coordinate transform to be applied to the (x,y) coordinates of
            the grid (e.g., :py:meth:`Curcio1990Transform.dva2ret`). It must
            accept two input arguments (x and y) and output two variables (the
            transformed x and y).
        label : str, optional
            A name to be used as the label of the matplotlib plot. This can be used
            to label plots with multiple regions (i.e. call plt.legend after)
        style : {'hull', 'scatter', 'cell'}, optional
            * 'hull': Show the convex hull of the grid (that is, the outline of
              the smallest convex set that contains all grid points).
            * 'scatter': Scatter plot all grid points
            * 'cell': Show the outline of each grid cell as a polygon. Note that
              this can be costly for a high-resolution grid.
        autoscale : bool, optional
            Whether to adjust the x,y limits of the plot to fit the implant
        zorder : int, optional
            The Matplotlib zorder at which to plot the grid
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            A Matplotlib axes object. If None, will either use the current axes
            (if exists) or create a new Axes object
        figsize : (float, float), optional
            Desired (width, height) of the figure in inches
        fc : str or valid matplotlib color, optional
            Facecolor, or edge color if style=scatter, of the plotted region
            Defaults to gray
        """
        if style.lower() not in ['hull', 'scatter', 'cell']:
            raise ValueError(f'Unknown plotting style "{style}". Choose from: '
                             f'"hull", "scatter", "cell"')
        if ax is None:
            ax = plt.gca()
        if figsize is not None:
            ax.figure.set_size_inches(figsize)
        ax.set_aspect('equal')
        if autoscale:
            ax.autoscale(True)
        if zorder is None:
            zorder = ZORDER['background']

        x, y = self.x, self.y
        try:
            # Step might be a tuple:
            x_step, y_step = self.step
        except TypeError:
            x_step = self.step
            y_step = self.step

        if style.lower() == 'cell':
            # Show a polygon for every grid cell that we are simulating:
            if self.type == 'hexagonal':
                raise NotImplementedError
            patches = []
            for xret, yret in zip(x.ravel(), y.ravel()):
                # Outlines of the cell are given by (x,y) and the step size:
                vertices = np.array([
                    [xret - x_step / 2, yret - y_step / 2],
                    [xret - x_step / 2, yret + y_step / 2],
                    [xret + x_step / 2, yret + y_step / 2],
                    [xret + x_step / 2, yret - y_step / 2],
                ])
                if transform is not None:
                    vertices = np.array(transform(*vertices.T)).T
                patches.append(Polygon(vertices, alpha=0.3, ec='k', fc=fc,
                                       ls='--', zorder=zorder))
            ax.add_collection(PatchCollection(patches, match_original=True,
                                              zorder=zorder, label=label))
        else:
            # Show either the convex hull or a scatter plot:
            if transform is not None:
                x, y = transform(self.x, self.y)
            points = np.vstack((x.ravel(), y.ravel()))
            # Remove NaN values from the grid:
            points = points[:, ~np.logical_or(*np.isnan(points))]
            if style.lower() == 'hull':
                hull = ConvexHull(points.T)
                ax.add_patch(Polygon(points[:, hull.vertices].T, alpha=0.3, ec='k',
                                     fc=fc, ls='--', zorder=zorder, label=label))
            elif style.lower() == 'scatter':
                ax.scatter(*points, alpha=0.3, ec=fc, color=fc, marker='+',
                           zorder=zorder, label=label)
        # This is needed in MPL 3.0.X to set the axis limit correctly:
        ax.autoscale_view()
        return ax


class VisualFieldMap(object, metaclass=ABCMeta):
    """Base class for a visual field map (retinotopy)

    A template

    """

    @abstractmethod
    def dva2ret(self, x, y):
        """Convert degrees of visual angle (dva) to retinal coords (um)"""
        raise NotImplementedError

    @abstractmethod
    def ret2dva(self, x, y):
        """Convert retinal coords (um) to degrees of visual angle (dva)"""
        raise NotImplementedError


class Curcio1990Map(VisualFieldMap):
    """Converts between visual angle and retinal eccentricity [Curcio1990]_"""

    @staticmethod
    def dva2ret(xdva, ydva):
        """Convert degrees of visual angle (dva) to retinal eccentricity (um)

        Assumes that one degree of visual angle is equal to 280 um on the
        retina [Curcio1990]_.
        """
        return 280.0 * xdva, 280.0 * ydva

    @staticmethod
    def ret2dva(xret, yret):
        """Convert retinal eccentricity (um) to degrees of visual angle (dva)

        Assumes that one degree of visual angle is equal to 280 um on the
        retina [Curcio1990]_
        """
        return xret / 280.0, yret / 280.0


class Watson2014Map(VisualFieldMap):
    """Converts between visual angle and retinal eccentricity [Watson2014]_"""

    @staticmethod
    def ret2dva(x_um, y_um, coords='cart'):
        """Converts retinal distances (um) to visual angles (deg)

        This function converts an eccentricity measurement on the retinal
        surface(in micrometers), measured from the optic axis, into degrees
        of visual angle using Eq. A6 in [Watson2014]_.

        Parameters
        ----------
        x_um, y_um : double or array-like
            Original x and y coordinates on the retina (microns)
        coords : {'cart', 'polar'}
            Whether to return the result in Cartesian or polar coordinates

        Returns
        -------
        x_dva, y_dva : double or array-like
            Transformed x and y coordinates (degrees of visual angle, dva)
        """
        phi_um, r_um = cart2pol(x_um, y_um)
        sign = np.sign(r_um)
        r_mm = 1e-3 * np.abs(r_um)
        r_deg = 3.556 * r_mm + 0.05993 * r_mm ** 2 - 0.007358 * r_mm ** 3
        r_deg += 3.027e-4 * r_mm ** 4
        r_deg *= sign
        if coords.lower() == 'cart':
            return pol2cart(phi_um, r_deg)
        elif coords.lower() == 'polar':
            return phi_um, r_deg
        raise ValueError(f'Unknown coordinate system "{coords}".')

    @staticmethod
    def dva2ret(x_deg, y_deg, coords='cart'):
        """Converts visual angles (deg) into retinal distances (um)

        This function converts degrees of visual angle into a retinal distance 
        from the optic axis (um) using Eq. A5 in [Watson2014]_.

        Parameters
        ----------
        x_dva, y_dva : double or array-like
            Original x and y coordinates (degrees of visual angle, dva)
        coords : {'cart', 'polar'}
            Whether to return the result in Cartesian or polar coordinates

        Returns
        -------
        x_ret, y_ret : double or array-like
            Transformed x and y coordinates on the retina (microns)

        """
        phi_deg, r_deg = cart2pol(x_deg, y_deg)
        sign = np.sign(r_deg)
        r_deg = np.abs(r_deg)
        r_mm = 0.268 * r_deg + 3.427e-4 * r_deg ** 2 - 8.3309e-6 * r_deg ** 3
        r_um = 1e3 * r_mm * sign
        if coords.lower() == 'cart':
            return pol2cart(phi_deg, r_um)
        elif coords.lower() == 'polar':
            return phi_deg, r_um
        raise ValueError(f'Unknown coordinate system "{coords}".')


class Watson2014DisplaceMap(Watson2014Map):
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
        # Convert x, y (dva) into polar coordinates:
        theta, rho_dva = cart2pol(xdva, ydva)
        # Add RGC displacement:
        meridian = np.where(xdva < 0, 'temporal', 'nasal')
        rho_dva += self.watson_displacement(rho_dva, meridian=meridian)
        # Convert back to x, y (dva):
        x, y = pol2cart(theta, rho_dva)
        return super(Watson2014DisplaceMap, self).dva2ret(x, y)

    def ret2dva(self, xret, yret):
        raise NotImplementedError


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
