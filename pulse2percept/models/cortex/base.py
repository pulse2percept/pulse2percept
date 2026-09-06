""":py:class:`~pulse2percept.models.cortex.CortexSpatial`"""

from ..base import SpatialModel, _draw_placed_implant
from ...topography.cortex import Polimeni2006Map
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
    grid_type : {'rect', 'hex'}, optional
        Whether to simulate points on a rectangular or hexagonal grid
    visual_field_map : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
        An instance of a :py:class:`~pulse2percept.topography.VisualFieldMap`
        object that provides retinotopic mappings.
        By default, :py:class:`~pulse2percept.topography.cortex.Polimeni2006Map` is
        used.
    n_gray : int, optional
        The number of gray levels to use. If an integer is given, k-means
        clustering is used to compress the color space of the percept into
        ``n_gray`` bins. If None, no compression is performed.
    implant_position : (x, y) or Quantity, optional
        Position of the device-local origin, in tissue coordinates or dva.

        .. versionadded:: 0.11.0

    implant_rotation : float or Quantity, optional
        In-plane rotation (deg), positive counter-clockwise.

        .. versionadded:: 0.11.0

    implant_depth : float or Quantity, optional
        Signed offset (um) along the normal of a 2D tissue map.

        .. versionadded:: 0.11.0

    location_noise : float or None, optional
        Standard deviation of fixed electrode-specific phosphene offsets, in dva.
        Requires an invertible 2D ``visual_field_map``. ``None`` or 0 disables it.
        Location-dependent models may also change phosphene shape or size.

        .. versionadded:: 0.11.0

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

    def __init__(self, implant, *, regions=None, visual_field_map=None,
                 **params):
        self._regions = None
        # `None` means "not given" for both: each default depends on the other.
        if regions is not None:
            params['regions'] = regions
        if visual_field_map is not None:
            params['visual_field_map'] = visual_field_map
        super(CortexSpatial, self).__init__(implant, **params)

        # Use [Polemeni2006]_ visual field map by default
        if visual_field_map is None:
            self.visual_field_map = Polimeni2006Map(regions=self.regions)
        elif regions is not None and \
            set(self.regions) != set(self.visual_field_map.regions):
            raise ValueError("Conflicting regions in provided visual_field_map and user-supplied regions parameter")
        else:
            # need to override self.regions
            self.regions = self.visual_field_map.regions

        if not isinstance(self.regions, list):
            self.regions = [self.regions]

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
             figsize=None, fc=None, show_implant=False, **kwargs):
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
        show_implant : bool, optional
            Draw the implant at its model-side placement. Requires
            ``use_dva=False``.

            .. versionadded:: 0.11.0
        kwargs : dict, optional
            Additional keyword arguments are passed on to Grid2D.plot()
        
        Returns
        -------
        ax : ``matplotlib.axes.Axes``
            Returns the axis object of the plot
        """
        if show_implant and use_dva:
            raise NotImplementedError(
                "show_implant=True is only supported in tissue coordinates; "
                "a nonlinear visual_field_map does not transform device "
                "geometry rigidly.")
        if style is None:
            style = 'hull' if use_dva else 'scatter'
        # Model must be built to access cortical coordinates
        if not self.is_built:
            self.build()
        ax = self.grid.plot(style=style, use_dva=use_dva, autoscale=autoscale, 
                            ax=ax, figsize=figsize, fc=fc, 
                            zorder=ZORDER['background'], 
                            legend=True if not use_dva else False)
        if show_implant:
            _draw_placed_implant(self, ax, autoscale=autoscale)
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

    def plot3d(self, style='scatter', ax=None, **kwargs):
        if not self.is_built:
            self.build()
        ax = self.grid.plot3d(style=style, ax=ax, **kwargs)
        # this is only ever for cortex right now so this is safe
        ax.set_xticklabels(np.array(ax.get_xticks()) / UM_PER_MM)
        ax.set_yticklabels(np.array(ax.get_yticks()) / UM_PER_MM)
        ax.set_zticklabels(np.array(ax.get_zticks()) / UM_PER_MM)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')
        ax.view_init(elev=20, azim=110)
        return ax
