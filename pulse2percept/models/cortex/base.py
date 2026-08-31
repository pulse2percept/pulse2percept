""":py:class:`~pulse2percept.models.cortex.CortexSpatial`, 
   :py:class:`~pulse2percept.models.cortex.ScoreboardSpatial`, 
   :py:class:`~pulse2percept.models.cortex.ScoreboardModel`"""

from ..base import Model, SpatialModel, _blend_meridian, _warn_rho_vs_pitch
from ...topography import Polimeni2006Map
from .._beyeler2019 import fast_scoreboard, fast_scoreboard_3d
from ...units import DimensionMismatchError, dva, um
from ...utils.constants import UM_PER_MM, ZORDER
import numpy as np

class CortexSpatial(SpatialModel):
    """Abstract base class for cortical models
    
    This is an abstract class that cortical models can subclass
    to get cortical implementation of the following features. 

    *  Updated default parameters for cortex
    *  Handling of multiple visual regions via regions property
    *  Plotting, including multiple visual regions, legends, vertical 
       divide at longitudinal fissure, etc.

    Parameters:
    -----------
    implant : :py:class:`~pulse2percept.implants.Implant`
        The implant whose stimulation this model predicts.

        .. versionadded:: 0.11.0

    regions : list of str, optional
        The regions to simulate. Options are any combination of 'v1', 'v2', 'v3'. 
        Default: ['v1']. 
    rho : double, optional
        Exponential decay constant describing current spread size (microns).
    min_current_spread : float, optional
        An electrode is skipped at grid points where its Gaussian current
        spread has decayed below this fraction of its peak. The default
        (1e-8, about 6.1 ``rho`` away) drops the Gaussian *times* the
        stimulus amplitude, summed over the skipped electrodes, so the error
        at a point is bounded by ``min_current_spread`` times the summed
        amplitude across electrodes.
    xrange : (x_min, x_max), optional
        A tuple indicating the range of x values to simulate (in degrees of
        visual angle). In a right eye, negative x values correspond to the
        temporal retina, and positive x values to the nasal retina. In a left
        eye, the opposite is true.
    yrange : tuple, (y_min, y_max), optional
        A tuple indicating the range of y values to simulate (in degrees of
        visual angle). Negative y values correspond to the superior retina,
        and positive y values to the inferior retina.
    step : int, double, tuple, optional
        Step size for the range of (x,y) values to simulate (in degrees of
        visual angle). For example, to create a grid with x values [0, 0.5, 1]
        use ``xrange=(0, 1)`` and ``step=0.5``. Pass a tuple to give the x
        and y axes different step sizes.
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
    n_jobs : int, optional
        Alias for ``n_threads``; ``None`` or ``-1`` uses every core.

    .. important::

        Changing a model parameter outside the constructor (e.g., by directly
        setting ``model.xrange = (-10, 10)``) invalidates the build, and the
        next ``predict_percept`` builds it again.
    """
    @property
    def regions(self):
        return self._regions
    
    @regions.setter
    def regions(self, regions):
        
        if not isinstance(regions, list):
            regions = [regions]
        self._regions = regions

    def __init__(self, implant, **params):
        self._regions = None
        super(CortexSpatial, self).__init__(implant, **params)

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

    def _retinal_range_to_dva(self, name, value):
        """A cortical model has no retinal extent for a length to stand for

        :py:class:`~pulse2percept.models.SpatialModel` decides this from the
        map that is installed at the time of assignment, which here is not yet
        the cortical one: this constructor puts that in place only *after*
        ``super().__init__`` has applied the parameters. Refusing here rather
        than there keeps a length from being read as a retinal extent in the
        window in between.
        """
        raise DimensionMismatchError(
            f"'{name}' is a visual field extent, measured in degrees of "
            f"visual angle. A physical length is shorthand for one only on a "
            f"retinal map, and {type(self).__name__} is cortical. Specify "
            f"'{name}' in dva instead.")

    def get_default_params(self):
        """Returns all settable parameters of the scoreboard model"""
        base_params = super(CortexSpatial, self).get_default_params()
        params = {
                    'xrange' : (-5, 5),
                    'yrange' : (-5, 5),
                    'step' : 0.1,
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
            # Cortical coordinates are stored in microns, plotted in mm:
            ax.set_xticklabels(np.array(ax.get_xticks()) / UM_PER_MM)
            ax.set_yticklabels(np.array(ax.get_yticks()) / UM_PER_MM)
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
        return ax

    def plot3D(self, style='scatter', ax=None, **kwargs):
        if not self.is_built:
            self.build()
        ax = self.grid.plot3D(style=style, ax=ax, **kwargs)
        # this is only ever for cortex right now so this is safe
        ax.set_xticklabels(np.array(ax.get_xticks()) / UM_PER_MM)
        ax.set_yticklabels(np.array(ax.get_yticks()) / UM_PER_MM)
        ax.set_zticklabels(np.array(ax.get_zticks()) / UM_PER_MM)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')
        ax.view_init(elev=20, azim=110)
        return ax


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
    implant : :py:class:`~pulse2percept.implants.Implant`
        The implant whose stimulation this model predicts.

        .. versionadded:: 0.11.0

    rho : double, optional
        Exponential decay constant describing phosphene size (microns).
    min_current_spread : float, optional
        An electrode is skipped at grid points where its Gaussian current
        spread has decayed below this fraction of its peak. The default
        (1e-8, about 6.1 ``rho`` away) drops the Gaussian *times* the
        stimulus amplitude, summed over the skipped electrodes, so the
        error at a point is bounded by ``min_current_spread`` times the
        summed amplitude across electrodes.
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
    step : int, double, tuple, optional
        Step size for the range of (x,y) values to simulate (in degrees of
        visual angle). For example, to create a grid with x values [0, 0.5, 1]
        use ``xrange=(0, 1)`` and ``step=0.5``. Pass a tuple to give the x
        and y axes different step sizes.
    grid_type : {'rectangular', 'hexagonal'}, optional
        Whether to simulate points on a rectangular or hexagonal grid
    meridian_blend : float, optional
        Gaussian standard deviation (dva) for smoothing across the vertical
        meridian. Default: 0.1. Set to 0 to disable.

        .. versionadded:: 0.10.0
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
    n_jobs : int, optional
        Alias for ``n_threads``; ``None`` or ``-1`` uses every core.

    .. important ::
    
        Changing a model parameter outside the constructor (e.g., by directly
        setting ``model.xrange = (-10, 10)``) invalidates the build, and the
        next ``predict_percept`` builds it again.

    """
    def __init__(self, implant, **params):
        super(ScoreboardSpatial, self).__init__(implant, **params)

    def get_default_params(self):
        """Returns all settable parameters of the scoreboard model"""
        base_params = super(ScoreboardSpatial, self).get_default_params()
        params = {
                    # radial current spread
                    'rho': 200,  
                    'ndim' : [2, 3],
                    'meridian_blend' : 0.1
                 }
        return {**base_params, **params}

    def get_param_units(self):
        """Return a dict of the units that parameters are stored in"""
        # Cortical coordinates are stored in microns (see `CorticalMap`), and
        # the current spread is compared against them:
        return {**super().get_param_units(), 'rho': um, 'meridian_blend': dva}

    def _build(self):
        _warn_rho_vs_pitch(self)

    def _postprocess_spatial(self, resp):
        """Blend the percept across the vertical meridian

        On this model rather than on `CortexSpatial`: the seam is a property
        of the split map this one is built on, not of being cortical, and a
        future cortical model without one should not inherit a correction for
        it.
        """
        blended = _blend_meridian(resp, self.grid, 'vertical',
                                  self.meridian_blend)
        if blended is resp:
            return resp
        # Restore percept threshold after blending:
        blended[np.abs(blended) < self.thresh_percept] = 0
        return blended

    def _predict_spatial(self, earray, stim):
        """Predicts the brightness at spatial locations"""
        x_el, y_el, z_el = self._electrode_coords(earray, stim)
        amp = self._stim_values(stim)

        # whether to allow current to spread between hemispheres
        separate = 0
        boundary = 0
        if self.vfmap.split_map:
            separate = 1
            boundary = self.vfmap.left_offset/2
        cutoff_r2 = self._cutoff_r2(self.rho)
        if self.vfmap.ndim == 3:
            return np.sum([
                fast_scoreboard_3d(amp, x_el, y_el, z_el,
                                self.grid[region].x.ravel(),
                                self.grid[region].y.ravel(),
                                self.grid[region].z.ravel(),
                                self.rho, self.thresh_percept, cutoff_r2,
                                separate, boundary,
                                self.n_threads)
                for region in self.regions ],
            axis = 0)
        elif self.vfmap.ndim == 2:
            return np.sum([
                fast_scoreboard(amp, x_el, y_el,
                                self.grid[region].x.ravel(), self.grid[region].y.ravel(),
                                self.rho, self.thresh_percept, cutoff_r2,
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
    implant : :py:class:`~pulse2percept.implants.Implant`
        The implant whose stimulation this model predicts.

        .. versionadded:: 0.11.0

    rho : double, optional
        Exponential decay constant describing phosphene size (microns).
    min_current_spread : float, optional
        An electrode is skipped at grid points where its Gaussian current
        spread has decayed below this fraction of its peak. The default
        (1e-8, about 6.1 ``rho`` away) drops the Gaussian *times* the
        stimulus amplitude, summed over the skipped electrodes, so the
        error at a point is bounded by ``min_current_spread`` times the
        summed amplitude across electrodes.
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
    step : int, double, tuple, optional
        Step size for the range of (x,y) values to simulate (in degrees of
        visual angle). For example, to create a grid with x values [0, 0.5, 1]
        use ``xrange=(0, 1)`` and ``step=0.5``. Pass a tuple to give the x
        and y axes different step sizes.
    grid_type : {'rectangular', 'hexagonal'}, optional
        Whether to simulate points on a rectangular or hexagonal grid
    meridian_blend : float, optional
        Gaussian standard deviation (dva) for smoothing across the vertical
        meridian. Default: 0.1. Set to 0 to disable.

        .. versionadded:: 0.10.0
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
    n_jobs : int, optional
        Alias for ``n_threads``; ``None`` or ``-1`` uses every core.

    .. important ::
        Changing a model parameter outside the constructor (e.g., by directly
        setting ``model.xrange = (-10, 10)``) invalidates the build, and the next
        ``predict_percept`` builds it again.

    """

    def __init__(self, implant, **params):
        super(ScoreboardModel, self).__init__(
            spatial=ScoreboardSpatial(implant, **params), temporal=None,
            **params)