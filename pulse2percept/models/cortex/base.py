"""`ScoreboardSpatial`, `ScoreboardModel"""

from ..base import Model, SpatialModel
from ...topography import Polimeni2006Map
from .._beyeler2019 import fast_scoreboard
from ...utils.constants import ZORDER
import numpy as np

class ScoreboardSpatial(SpatialModel):
    """Cortical adaptation of scoreboard model from [Beyeler2019]_

    Implements the scoreboard model described in [Beyeler2019]_, where percepts
    from each electrode are Gaussian blobs. The percepts resulting from different 
    cortical regions (e.g. v1/v2/v3) are added linearly. The `rho` parameter 
    modulates phosphene size.

    .. note ::

        Use this class if you want to combine the spatial model with a temporal
        model.
        Use :py:class:`~pulse2percept.models.cortex.ScoreboardModel` if you want a
        a standalone model.

    Parameters
    ----------
    rho : double, optional
        Exponential decay constant describing phosphene size (microns).
    regions : list of str, optional
        The regions to simulate. Options are 'v1', 'v2', or 'v3'. Default:
        ['v1']
    xrange : (x_min, x_max), optional
        A tuple indicating the range of x values to simulate (in degrees of
        visual angle). In a right eye, negative x values correspond to the
        temporal retina, and positive x values to the nasal retina. In a left
        eye, the opposite is true.
    yrange : tuple, (y_min, y_max), optional
        A tuple indicating the range of y values to simulate (in degrees of
        visual angle). Negative y values correspond to the superior retina,
        and positive y values to the inferior retina.
    xystep : int, double, tuple, optional
        Step size for the range of (x,y) values to simulate (in degrees of
        visual angle). For example, to create a grid with x values [0, 0.5, 1]
        use ``xrange=(0, 1)`` and ``xystep=0.5``.
    grid_type : {'rectangular', 'hexagonal'}, optional
        Whether to simulate points on a rectangular or hexagonal grid
    retinotopy : :py:class:`~pulse2percept.topography..VisualFieldMap`, optional
        An instance of a :py:class:`~pulse2percept.topography.VisualFieldMap`
        object that provides retinotopic mappings.
        By default, :py:class:`~pulse2percept.topography.Polimeni2006Map` is
        used.
    n_gray : int, optional
        The number of gray levels to use. If an integer is given, k-means
        clustering is used to compress the color space of the percept into
        ``n_gray`` bins. If None, no compression is performed.
    noise : float or int, optional
        Adds salt-and-pepper noise to each percept frame. An integer will be
        interpreted as the number of pixels to subject to noise in each frame.
        A float between 0 and 1 will be interpreted as a ratio of pixels to
        subject to noise in each frame.
    n_threads : int, optional
        Number of CPU threads to use during parallelization using OpenMP. 
        Defaults to max number of user CPU cores.

    .. important ::
        If you change important model parameters outside the constructor (e.g.,
        by directly setting ``model.xrange = (-10, 10)``), you will have to call
        ``model.build()`` again for your changes to take effect.

    """
    def __init__(self, **params):
        super(ScoreboardSpatial, self).__init__(**params)

        # Use [Polemeni2006]_ visual field map by default
        if 'retinotopy' not in params.keys():
            self.retinotopy = Polimeni2006Map(regions=self.regions)

    def get_default_params(self):
        """Returns all settable parameters of the scoreboard model"""
        base_params = super(ScoreboardSpatial, self).get_default_params()
        params = {
                    # override xrange and yrange so we don't have points
                    # on the boundary between hemispheres or y=0
                    'xrange' : (-14.99, 15.01),
                    'yrange' : (-14.99, 15.01),
                    # radial current spread
                    'rho': 200,  
                    # Visual field regions to simulate
                    'regions' : ['v1']
                 }
        return {**base_params, **params}

    def _build(self):
        # could potentially just adjust these instead of warning
        for region in self.regions:
            if np.any(self.grid[region].x == 0):
                raise UserWarning("Since the visual cortex is discontinuous " +
                    "across hemispheres, it is recommended to not simulate points " +
                    " at exactly x=0. This can be avoided by adding a small " + 
                    "to both limits of xrange")
            if (region in ['v2', 'v3'] and
                np.any(self.grid[region].y == 0)):
                raise UserWarning(f"Since the {region} is discontinuous " +
                    "across the y axis, it is recommended to not simulate points " +
                    " at exactly y=0. This can be avoided by adding a small " + 
                    "to both limits of yrange")

    def _predict_spatial(self, earray, stim):
        """Predicts the brightness at spatial locations"""
        x_el = np.array([earray[e].x for e in stim.electrodes],
                                        dtype=np.float32)
        y_el = np.array([earray[e].y for e in stim.electrodes],
                                        dtype=np.float32)
        return np.sum([
                fast_scoreboard(stim.data, x_el, y_el,
                                self.grid[region].x.ravel(), self.grid[region].y.ravel(),
                                self.rho, self.thresh_percept, self.n_threads)
                for region in self.regions ],
            axis = 0)


    def plot(self, use_dva=False, style='scatter', autoscale=True, ax=None,
             figsize=None):
        """Plot the model

        Parameters
        ----------
        use_dva : bool, optional
            Uses degrees of visual angle (dva) if True, else retinal
            coordinates (microns)
        style : {'hull', 'scatter', 'cell'}, optional
            Grid plotting style:

            * 'hull': Show the convex hull of the grid (that is, the outline of
              the smallest convex set that contains all grid points).
            * 'scatter': Scatter plot all grid points
            * 'cell': Show the outline of each grid cell as a polygon. Note that
              this can be costly for a high-resolution grid.
        autoscale : bool, optional
            Whether to adjust the x,y limits of the plot to fit the implant
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            A Matplotlib axes object. If None, will either use the current axes
            (if exists) or create a new Axes object.
        figsize : (float, float), optional
            Desired (width, height) of the figure in inches

        Returns
        -------
        ax : ``matplotlib.axes.Axes``
            Returns the axis object of the plot
        """
        if not self.is_built:
            self.build()
        if use_dva:
            ax = self.grid.plot(autoscale=autoscale, ax=ax, style=style,
                                zorder=ZORDER['background'], figsize=figsize)
            ax.set_xlabel('x (dva)')
            ax.set_ylabel('y (dva)')
        else:
            for idx_region, region in enumerate(self.retinotopy.regions):
                transform = self.retinotopy.from_dva()[region]
                if region == 'v1':
                    fc = 'red'
                elif region == 'v2':
                    fc = 'orange'
                elif region == 'v3':
                    fc = 'green'
                ax = self.grid.plot(transform=transform, label=region, ax=ax,
                                    zorder=ZORDER['background'] + 1, style=style,
                                    figsize=figsize, autoscale=autoscale, fc=fc)


            ax.legend(loc='upper right')

            ax.set_xlabel('x (microns)')
            ax.set_ylabel('y (microns)')
        return ax


class ScoreboardModel(Model):
    """Cortical adaptation of scoreboard model from [Beyeler2019]_ (standalone model)

    Implements the scoreboard model described in [Beyeler2019]_, where percepts
    from each electrode are Gaussian blobs. The percepts resulting from different 
    cortical regions (e.g. v1/v2/v3) are added linearly. The `rho` parameter 
    modulates phosphene size.

    .. note ::

        Use this class if you want a standalone model.
        Use :py:class:`~pulse2percept.models.cortex.ScoreboardSpatial` if you want
        to combine the spatial model with a temporal model.

    Parameters
    ----------
    rho : double, optional
        Exponential decay constant describing phosphene size (microns).
    regions : list of str, optional
        The regions to simulate. Options are 'v1', 'v2', or 'v3'. Default:
        ['v1']
    xrange : (x_min, x_max), optional
        A tuple indicating the range of x values to simulate (in degrees of
        visual angle). In a right eye, negative x values correspond to the
        temporal retina, and positive x values to the nasal retina. In a left
        eye, the opposite is true.
    yrange : tuple, (y_min, y_max), optional
        A tuple indicating the range of y values to simulate (in degrees of
        visual angle). Negative y values correspond to the superior retina,
        and positive y values to the inferior retina.
    xystep : int, double, tuple, optional
        Step size for the range of (x,y) values to simulate (in degrees of
        visual angle). For example, to create a grid with x values [0, 0.5, 1]
        use ``xrange=(0, 1)`` and ``xystep=0.5``.
    grid_type : {'rectangular', 'hexagonal'}, optional
        Whether to simulate points on a rectangular or hexagonal grid
    retinotopy : :py:class:`~pulse2percept.topography..VisualFieldMap`, optional
        An instance of a :py:class:`~pulse2percept.topography.VisualFieldMap`
        object that provides retinotopic mappings.
        By default, :py:class:`~pulse2percept.topography.Polimeni2006Map` is
        used.
    n_gray : int, optional
        The number of gray levels to use. If an integer is given, k-means
        clustering is used to compress the color space of the percept into
        ``n_gray`` bins. If None, no compression is performed.
    noise : float or int, optional
        Adds salt-and-pepper noise to each percept frame. An integer will be
        interpreted as the number of pixels to subject to noise in each frame.
        A float between 0 and 1 will be interpreted as a ratio of pixels to
        subject to noise in each frame.
    n_threads : int, optional
        Number of CPU threads to use during parallelization using OpenMP. 
        Defaults to max number of user CPU cores.

    .. important ::
        If you change important model parameters outside the constructor (e.g.,
        by directly setting ``model.xrange = (-10, 10)``), you will have to call
        ``model.build()`` again for your changes to take effect.

    """

    def __init__(self, **params):
        super(ScoreboardModel, self).__init__(spatial=ScoreboardSpatial(**params),
                                              temporal=None,
                                              **params)