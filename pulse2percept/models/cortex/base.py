"""`ScoreboardSpatial`, `ScoreboardModel"""

from ..base import Model, SpatialModel
from ...topography import Polimeni2006Map
from .._beyeler2019 import fast_scoreboard
from ...utils.constants import ZORDER
import warnings
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
    @property
    def regions(self):
        return self._regions
    
    @regions.setter
    def regions(self, regions):
        
        if not isinstance(regions, list):
            regions = [regions]
        self._regions = regions

    def __init__(self, **params):
        self._regions = None
        super(ScoreboardSpatial, self).__init__(**params)

        # Use [Polemeni2006]_ visual field map by default
        if 'retinotopy' not in params.keys():
            self.retinotopy = Polimeni2006Map(regions=self.regions)
        elif 'regions' in params.keys() and \
            set(self.regions) != set(self.retinotopy.regions):
            raise ValueError("Conflicting regions in provided retinotopy and regions")
        else:
            # need to override self.regions
            self.regions = self.retinotopy.regions

        if not isinstance(self.regions, list):
            self.regions = [self.regions]

    def get_default_params(self):
        """Returns all settable parameters of the scoreboard model"""
        base_params = super(ScoreboardSpatial, self).get_default_params()
        params = {
                    'xrange' : (-5, 5),
                    'yrange' : (-5, 5),
                    'xystep' : 0.1,
                    # radial current spread
                    'rho': 200,  
                    # Visual field regions to simulate
                    'regions' : ['v1']
                 }
        return {**base_params, **params}

    def _build(self):
        # warn the user either that they are simulating points at discontinuous boundaries, 
        # or that the points will be moved by a small constant
        if np.any(self.grid['dva'].x == 0):
            if hasattr(self.retinotopy, 'jitter_boundary') and self.retinotopy.jitter_boundary:
                warnings.warn("Since the visual cortex is discontinuous " +
                    "across hemispheres, it is recommended to not simulate points " +
                    " at exactly x=0. Points on the boundary will be moved " +
                    "by a small constant") 
            else:
                warnings.warn("Since the visual cortex is discontinuous " +
                    "across hemispheres, it is recommended to not simulate points " +
                    " at exactly x=0. This can be avoided by adding a small " + 
                    "to both limits of xrange") 
        if (np.any([r in self.regions for r in self.grid.discontinuous_y]) and 
            np.any(self.grid['dva'].y == 0)):
            if hasattr(self.retinotopy, 'jitter_boundary') and self.retinotopy.jitter_boundary:
                warnings.warn("Since some simulated regions are discontinuous " +
                    "across the y axis, it is recommended to not simulate points " +
                    " at exactly y=0.  Points on the boundary will be moved " +
                    "by a small constant") 
            else:
                warnings.warn(f"Since some simulated regions are discontinuous " +
                    "across the y axis, it is recommended to not simulate points " +
                    " at exactly y=0. This can be avoided by adding a small " + 
                    "to both limits of yrange or setting " +
                    "self.retinotopy.jitter_boundary=True")

    def _predict_spatial(self, earray, stim):
        """Predicts the brightness at spatial locations"""
        x_el = np.array([earray[e].x for e in stim.electrodes],
                                        dtype=np.float32)
        y_el = np.array([earray[e].y for e in stim.electrodes],
                                        dtype=np.float32)

        # whether to allow current to spread between hemispheres
        separate = 0
        boundary = 0
        if self.retinotopy.split_map:
            separate = 1
            boundary = self.retinotopy.left_offset/2
        return np.sum([
                fast_scoreboard(stim.data, x_el, y_el,
                                self.grid[region].x.ravel(), self.grid[region].y.ravel(),
                                self.rho, self.thresh_percept, 
                                separate, boundary, 
                                self.n_threads)
                for region in self.regions ],
            axis = 0)

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