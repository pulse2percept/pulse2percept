"""`CortexSpatial`, `ScoreboardSpatial`, `ScoreboardModel`"""

from ..base import Model, SpatialModel, TorchBaseModel
from ...topography import Polimeni2006Map
from .._beyeler2019 import fast_scoreboard, fast_scoreboard_3d
from ...utils.constants import ZORDER
import numpy as np
import torch

class CortexSpatial(SpatialModel):
    """Abstract base class for cortical models
    
    This is an abstract class that cortical models can subclass
    to get cortical implementation of the following features. 
    1) Updated default parameters for cortex
    2) Handling of multiple visual regions via regions property
    3) Plotting, including multiple visual regions, legends, vertical 
       divide at longitudinal fissure, etc.

    Parameters:
    -----------
    regions : list of str, optional
        The regions to simulate. Options are any combination of 'v1', 'v2', 'v3'. 
        Default: ['v1']. 
    rho : double, optional
        Exponential decay constant describing current spread size (microns).
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
    vfmap : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
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
        super(CortexSpatial, self).__init__(**params)

        # Use [Polemeni2006]_ visual field map by default
        if 'vfmap' not in params.keys():
            self.vfmap = Polimeni2006Map(regions=self.regions)
        elif 'regions' in params.keys() and \
            set(self.regions) != set(self.vfmap.regions):
            raise ValueError("Conflicting regions in provided vfmap and user-supplied regions parameter")
        else:
            # need to override self.regions
            self.regions = self.vfmap.regions

        if not isinstance(self.regions, list):
            self.regions = [self.regions]

    def get_default_params(self):
        """Returns all settable parameters of the scoreboard model"""
        base_params = super(CortexSpatial, self).get_default_params()
        params = {
                    'xrange' : (-5, 5),
                    'yrange' : (-5, 5),
                    'xystep' : 0.1,
                    # Visual field regions to simulate
                    'regions' : ['v1']
                 }
        return {**base_params, **params}
    

    def plot(self, use_dva=False, style=None, autoscale=True, ax=None,
             figsize=None, fc=None, **kwargs):
        """Plot the model
        Parameters
        ----------
        use_dva : bool, optional
            Plot points in visual field. If false, simulated points will be 
            plotted in cortex
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
        fc : matplotlib color, optional
            Face color for the grid cells. If None, will use the default
            matplotlib color cycle.
        kwargs : dict, optional
            Additional keyword arguments are passed on to Grid2D.plot()
        Returns
        -------
        ax : ``matplotlib.axes.Axes``
            Returns the axis object of the plot
        """
        if style is None:
            style = 'hull' if use_dva else 'scatter'
        # Model must be built to access cortical coordinates
        if not self.is_built:
            self.build()
        ax = self.grid.plot(style=style, use_dva=use_dva, autoscale=autoscale, 
                            ax=ax, figsize=figsize, fc=fc, 
                            zorder=ZORDER['background'], 
                            legend=True if not use_dva else False)
        if use_dva:
            ax.set_xlabel('x (dva)')
            ax.set_ylabel('y (dva)')
        else:
            ax.set_xticklabels(np.array(ax.get_xticks()) / 1000)
            ax.set_yticklabels(np.array(ax.get_yticks()) / 1000)
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
        return ax

    def plot3D(self, style='scatter', ax=None, **kwargs):
        if not self.is_built:
            self.build()
        ax = self.grid.plot3D(style=style, ax=ax, **kwargs)
        # this is only ever for cortex right now so this is safe
        ax.set_xticklabels(np.array(ax.get_xticks()) / 1000)
        ax.set_yticklabels(np.array(ax.get_yticks()) / 1000)
        ax.set_zticklabels(np.array(ax.get_zticks()) / 1000)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')
        ax.view_init(elev=20, azim=110)
        return ax

class TorchScoreboardSpatial(TorchBaseModel):
    def __init__(self, p2pmodel):
        super().__init__(p2pmodel)
        self.rho = torch.tensor(p2pmodel.rho, device=self.device)
        self.shape = p2pmodel.grid.shape
        self.regions = p2pmodel.regions
        self.thresh_percept = torch.tensor(p2pmodel.thresh_percept, device=self.device)
        # whether to let current spread between regions
        self.separate = 0
        self.boundary = 0
        if p2pmodel.vfmap.split_map:
            self.separate = 1
            self.boundary = p2pmodel.vfmap.left_offset/2
        self.locs = {}
        for region in self.regions:
            x = torch.tensor(p2pmodel.grid[region].x.ravel(),device=self.device)
            y = torch.tensor(p2pmodel.grid[region].y.ravel(),device=self.device)
            if p2pmodel.grid[region].z is not None:
                z = torch.tensor(p2pmodel.grid[region].z.ravel(),device=self.device)
                self.locs[region] = torch.stack([x, y, z], axis=-1)
            else:
                self.locs[region] = torch.stack([x, y], axis=-1)

    def forward(self, amps, e_locs, model_params=None):
        """Predicts the percept
        Parameters
        ----------
        amps: (t, n_elecs) shaped tensor
            Amplitude for each electrode in the implant
        e_locs: (n_elecs, dim) shaped tensor
            electrode location (x, y, [z optional]) for each electrode
        model_params: tensor, optional
            rho parameter for current spread
        """
        rho = self.rho if model_params is None else model_params[0]
        amps = amps.T

        # (npixels, nelecs)
        tot_intensities = 0
        for region in self.regions:
            d2_el = torch.sum((self.locs[region][:, None, :] - e_locs[None, :, :] )**2, axis=-1)
            intensities = amps[:, None, :] * torch.exp(-d2_el / (2 * rho**2)) # generate gaussian blobs for each electrode
            intensities = torch.nan_to_num(intensities)
            if self.separate:
                intensities *= torch.where((e_locs[None,:,0] < self.boundary) == (self.locs[region][:,None,0] < self.boundary), 1, 0) # ensure current cannot spread between hemispheres
            intensities = torch.sum(intensities, axis=-1) # add up all gaussian blobs
            tot_intensities += intensities
        tot_intensities = torch.where(tot_intensities > self.thresh_percept, tot_intensities, 0.0)
        return tot_intensities.T

class ScoreboardSpatial(CortexSpatial):
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
    vfmap : :py:class:`~pulse2percept.topography..VisualFieldMap`, optional
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
        self.torchmodel = None

    def _build(self):
        if self.engine == 'torch':
            self.torchmodel = TorchScoreboardSpatial(self)

    def get_default_params(self):
        """Returns all settable parameters of the scoreboard model"""
        base_params = super(ScoreboardSpatial, self).get_default_params()
        params = {
                    # radial current spread
                    'rho': 200,  
                    'ndim' : [2, 3],
                    'engine': 'cython'
                 }
        return {**base_params, **params}

    def _predict_spatial(self, earray, stim):
        """Predicts the brightness at spatial locations"""
        x_el = np.array([earray[e].x for e in stim.electrodes],
                                        dtype=np.float32)
        y_el = np.array([earray[e].y for e in stim.electrodes],
                                        dtype=np.float32)
        z_el = np.array([earray[e].z for e in stim.electrodes],
                                        dtype=np.float32)

        # whether to allow current to spread between hemispheres
        separate = 0
        boundary = 0
        if self.vfmap.split_map:
            separate = 1
            boundary = self.vfmap.left_offset/2
        if self.engine == "torch":
            if self.vfmap.ndim == 2:
                e_locs = torch.tensor(np.stack([x_el, y_el], axis=-1), device=self.device)
            else:
                e_locs = torch.tensor(np.stack([x_el, y_el, z_el], axis=-1), device=self.device)
            amps = torch.tensor(stim.data, device=self.device)
            return self.torchmodel(amps=amps, e_locs=e_locs).numpy()

        elif self.engine == "cython":
            if self.vfmap.ndim == 3:
                return np.sum([
                    fast_scoreboard_3d(stim.data, x_el, y_el, z_el,
                                    self.grid[region].x.ravel(), 
                                    self.grid[region].y.ravel(),
                                    self.grid[region].z.ravel(),
                                    self.rho, self.thresh_percept, 
                                    separate, boundary, 
                                    self.n_threads)
                    for region in self.regions ],
                axis = 0)
            elif self.vfmap.ndim == 2:
                return np.sum([
                    fast_scoreboard(stim.data, x_el, y_el,
                                    self.grid[region].x.ravel(), self.grid[region].y.ravel(),
                                    self.rho, self.thresh_percept, 
                                    separate, boundary, 
                                    self.n_threads)
                    for region in self.regions ],
                axis = 0)
            else:
                raise ValueError("Invalid dimensionality of visual field map")


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
    vfmap : :py:class:`~pulse2percept.topography..VisualFieldMap`, optional
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