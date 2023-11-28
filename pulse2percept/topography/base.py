"""
`Grid2D`, `VisualFieldMap`

"""

import numpy as np
from abc import ABCMeta, abstractmethod
from scipy.spatial import ConvexHull
# Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working:
from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib as mpl


from ..utils.base import PrettyPrint
from ..utils.constants import ZORDER
from ..models import BaseModel

class Grid2D(PrettyPrint):
    """2D spatial grid

    This class generates and stores 2D mesh grids of coordinates across
    different regions (visual field, retina, cortex). The grid is uniform 
    in visual field, and transformed with a retinotopic mapping to 
    obtain the grid in other regions.

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
    You can iterate through a grid as if it were a list.
    Notice, the grid is indexed in (x, y) order, starting in the upper 
    left of the grid (following image convention)

    >>> grid = Grid2D((0, 1), (2, 3))
    >>> for x, y in grid:
    ...     print(x, y)
    0.0 3.0
    1.0 3.0
    0.0 2.0
    1.0 2.0

    """

    all_regions = ['dva', 'ret', 'v1', 'v2', 'v3']
    discontinuous_x = ['v1', 'v2', 'v3']
    discontinuous_y = ['v2', 'v3']

    @staticmethod
    def _register_regions(regions):
        """ Registers helper getters and setters to allow e.g. grid.ret, grid.v1.
            Necessary for backwards compatibility. Static because property attributes are
            tracked at the class level
            
            Note: The list of regions given does NOT need be the regions currently
            being used (can be all valid regions). If a given region does not exist 
            at call time, then a ValueError will be raised (e.g. grid.v1 with a
            retinal visual field map will throw an error).

            Parameters:
            ------------
            regions : list of str
                Names of each region to register
        """
        def getter(regionname):
            def fn(self):
                if regionname in self._grid.keys():
                    return self._grid[regionname]
                else:
                    raise ValueError(f"Region {regionname} not found. Make sure the model is" \
                        " built with the correct retinotopy")
            return fn
        def setter(regionname):
            def fn(self, value):
                self._grid[regionname] = value
            return fn

        for region in regions:
            if not hasattr(Grid2D, region):
                setattr(Grid2D, region, property(fget=getter(region), 
                                                 fset=setter(region)))

    @property
    def x(self):
        return self._grid['dva'].x

    @x.setter
    def x(self, value):
        self._grid['dva'] = self.CoordinateGrid(value, self.y)
    
    @property 
    def y(self):
        return self._grid['dva'].y
    
    @y.setter
    def y(self, value):
        self._grid['dva'] = self.CoordinateGrid(self.x, value)


    def __init__(self, x_range, y_range, step=1, grid_type='rectangular'):
        self.x_range = x_range
        self.y_range = y_range
        self.step = step
        self.type = grid_type
        self.retinotopy = None
        self.regions = []
        self.retinotopy = None
        # Datatype for storing the grid of coordinates
        self.CoordinateGrid = namedtuple("CoordinateGrid", ['x', 'y'])
        # Internally, coordinate grids for each region are stored in _grid
        self._grid = {}
        # Register helper getters and setters for region names. This is slightly
        # wasteful to do with every instance, but it is impossible to do before initialization
        self._register_regions(self.all_regions)

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
        # Create the grid, flip y axis so that it increases from bottom to top:
        self._grid['dva'] = self.CoordinateGrid(
            *np.meshgrid(self._xflat, self._yflat[::-1], indexing='xy'))
        self.shape = self.x.shape
        self.size = self.x.size
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

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._grid[key]
        elif isinstance(key, int):
            return self.CoordinateGrid(self.x.ravel()[key], self.y.ravel()[key])
        else:
            raise ValueError(f"Unknown key: {key}. Must be region name or \
                              integer position")

    def build(self, retinotopy):
        self.retinotopy = retinotopy
        for region, map_fn in retinotopy.from_dva().items():
            self._grid[region] = self.CoordinateGrid(*map_fn(self.x, self.y))
            if region not in self.regions:
                self.regions.append(region)
            # Register the mapping if it wasn't already
            if region not in self.all_regions:
                self._register_regions([region])

    def plot(self, style='hull', autoscale=True, zorder=None, ax=None,
            figsize=None, fc=None, use_dva=False, legend=False):
        """Plot the extension of the grid

        Parameters
        ----------
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
        use_dva : bool, optional
            Whether dva or transformed points should be plotted.  If True, will
            not apply any transformations, and if False, will apply all
            transformations in self.retinotopy
        legend : bool, optional
            Whether to add a plot legend. The legend is always added if there 
            are 2 or more regions. This only applies if there is 1 region.
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

        transforms = [('dva', None)]
        if not use_dva:
            transforms = self.retinotopy.from_dva().items()

        color_map = {
            'ret' : 'gray',
            'dva' : 'gray',
            'v1' : 'red',
            'v2' : 'orange',
            'v3' : 'green'
        }        
        # for tracking legend items when style='cell'
        legends = []
        for idx, (label, transform) in enumerate(transforms):
            if fc is not None:
                color = fc[label] if isinstance(fc, dict) else fc     
            elif label in color_map.keys():
                color = color_map[label]
            else:
                color = 'gray'

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
                    # If region is discontinuous and vertices cross boundary, skip
                    if (transform and
                        label in self.discontinuous_x and 
                        np.sign(vertices[0][0]) != np.sign(vertices[2][0])):
                        continue
                    if (transform and
                        label in self.discontinuous_y and 
                        np.sign(vertices[0][1]) != np.sign(vertices[1][1])):
                        continue
                    # transform the points
                    if transform is not None:
                        vertices = np.array(transform(*vertices.T)).T
                    patches.append(Polygon(vertices, alpha=0.3, ec='k', fc=color,
                                        ls='--', zorder=zorder, label=label))
                legends.append(patches[0])
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
                    if self.retinotopy and self.retinotopy.split_map and not use_dva:
                        # all split maps have an offset for left fovea
                        divide = 0 if use_dva else self.retinotopy.left_offset / 2
                        points_right = points[:, points[0] >= divide]
                        points_left = points[:, points[0] <= divide]
                        if points_right.size > 0:
                            hull_right = ConvexHull(points_right.T)
                            ax.add_patch(Polygon(points_right[:, hull_right.vertices].T, alpha=0.3, ec='k',
                                            fc=color, ls='--', zorder=zorder))
                        if points_left.size > 0:
                            hull_left = ConvexHull(points_left.T)
                            ax.add_patch(Polygon(points_left[:, hull_left.vertices].T, alpha=0.3, ec='k',
                                            fc=color, ls='--', zorder=zorder))
                    else:
                        hull = ConvexHull(points.T)
                        ax.add_patch(Polygon(points[:, hull.vertices].T, alpha=0.3, ec='k',
                                            fc=color, ls='--', zorder=zorder))
                    legends.append(ax.patches[-1])
                elif style.lower() == 'scatter':
                    ax.scatter(*points, alpha=0.4, ec=color, color=color, marker='+',
                            zorder=zorder, label=label)
        
        # This is needed in MPL 3.0.X to set the axis limit correctly:
        ax.autoscale_view()
        # plot boundary between hemispheres if it exists
        # but don't change the plot limits 
        lim = ax.get_xlim()
        if self.retinotopy and self.retinotopy.split_map:
            boundary = self.retinotopy.left_offset / 2
            if use_dva:
                boundary = 0
            if lim[0] < boundary < lim[1]:
                ax.axvline(boundary, linestyle=':', c='gray')

        if len(transforms) > 1 or legend:
            if style in ['cell', 'hull']:
                ax.legend(legends, [t[0] for t in transforms], loc='upper right')
            else:
                ax.legend(loc='upper right')

        return ax


class VisualFieldMap(BaseModel):
    """ Base template class for a visual field map (retinotopy) """

    # If the map is split into left and right hemispheres. 
    split_map = False

    def __init__(self, **params):
        super().__init__(**params)
        # don't need build functionality from BaseModel
        self.is_built = True

    @abstractmethod
    def from_dva(self):
        """ Returns a dict containing the region(s) that this retinotopy maps 
            to, and the corresponding mapping function(s).
        """
        raise NotImplementedError

    def to_dva(self):
        """ Returns a dict containing the region(s) that this retinotopy maps 
            from, and the corresponding inverse mapping function(s). This 
            transform is optional for most models.
        """
        raise NotImplementedError

    def get_default_params(self):
        """Required to inherit from BaseModel"""
        return {}

    def __eq__(self, other):
        """
        Equality operator for VisualFieldMap.

        Parameters
        ----------
        other: VisualFieldMap
            VisualFieldMap to compare against

        Returns
        -------
        bool:
            True if the compared objects have identical attributes, False otherwise.
        """
        if not isinstance(other, self.__class__):
            return False
        if id(self) == id(other):
            return True
        return self.__dict__ == other.__dict__
