""":py:class:`~pulse2percept.models.cortex.ScoreboardSpatial`,
   :py:class:`~pulse2percept.models.cortex.ScoreboardModel`"""

from ..base import (Model, _blend_meridian, _thread_params,
                    _warn_rho_vs_pitch)
from .._scoreboard import fast_scoreboard, fast_scoreboard_3d
from .base import CortexSpatial
from ...units import dva, um
import numpy as np


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

    .. warning::

        ``rho`` is fixed: this model does not predict pulse-dependent
        phosphene size. Doubling amplitude doubles brightness and leaves the
        phosphene exactly as wide. Use
        :py:class:`~pulse2percept.models.cortex.DynaphosModel` for a cortical
        model whose phosphene size follows the stimulus current.

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
    grid_type : {'rect', 'hex'}, optional
        Whether to simulate points on a rectangular or hexagonal grid
    meridian_blend : float, optional
        Gaussian standard deviation (dva) for smoothing across the vertical
        meridian. Default: 0.1. Set to 0 to disable.

        .. versionadded:: 0.10.0
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

    .. important ::
    
        Changing a model parameter outside the constructor (e.g., by directly
        setting ``model.xrange = (-10, 10)``) invalidates the build, and the
        next ``predict_percept`` builds it again.

    """
    def __init__(self, implant, *, rho=200, regions=None, meridian_blend=0.1,
                 xrange=(-5, 5), yrange=(-5, 5), step=0.1,
                 grid_type='rect', thresh_percept=0,
                 min_current_spread=1e-8, visual_field_map=None, n_gray=None,
                 implant_position=(0, 0), implant_rotation=0,
                 implant_depth=0,
                 location_noise=None,
                 verbose=True, ndim=None, n_threads=None, n_jobs=None):
        super().__init__(
            implant, rho=rho, regions=regions,
            meridian_blend=meridian_blend, xrange=xrange, yrange=yrange,
            step=step, grid_type=grid_type, thresh_percept=thresh_percept,
            min_current_spread=min_current_spread,
            visual_field_map=visual_field_map,
            n_gray=n_gray,
            implant_position=implant_position,
            implant_rotation=implant_rotation,
            implant_depth=implant_depth,
            location_noise=location_noise, verbose=verbose,
            ndim=[2, 3] if ndim is None else ndim,
            **_thread_params(n_threads, n_jobs))

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

    def _predict_spatial(self, electrode_array, stim):
        """Predicts the brightness at spatial locations"""
        amp = self._stim_values(stim)

        # whether to allow current to spread between hemispheres
        separate = 0
        boundary = 0
        if self.visual_field_map.split_map:
            separate = 1
            boundary = self.visual_field_map.left_offset/2
        cutoff_r2 = self._cutoff_r2(self.rho)
        # `location_noise` displaces an electrode in the visual field, so its
        # cortical coordinates are region-specific:
        coords = {region: self._electrode_coords(electrode_array, stim,
                                                 region=region)
                  for region in self.regions}
        if self.visual_field_map.ndim == 3:
            return np.sum([
                fast_scoreboard_3d(amp, *coords[region],
                                self.grid[region].x.ravel(),
                                self.grid[region].y.ravel(),
                                self.grid[region].z.ravel(),
                                self.rho, self.thresh_percept, cutoff_r2,
                                separate, boundary,
                                self.n_threads)
                for region in self.regions ],
            axis = 0)
        elif self.visual_field_map.ndim == 2:
            return np.sum([
                fast_scoreboard(amp, *coords[region][:2],
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

    .. warning::

        ``rho`` is fixed: this model does not predict pulse-dependent
        phosphene size. Doubling amplitude doubles brightness and leaves the
        phosphene exactly as wide. Use
        :py:class:`~pulse2percept.models.cortex.DynaphosModel` for a cortical
        model whose phosphene size follows the stimulus current.

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
    grid_type : {'rect', 'hex'}, optional
        Whether to simulate points on a rectangular or hexagonal grid
    meridian_blend : float, optional
        Gaussian standard deviation (dva) for smoothing across the vertical
        meridian. Default: 0.1. Set to 0 to disable.

        .. versionadded:: 0.10.0
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

    .. important ::
        Changing a model parameter outside the constructor (e.g., by directly
        setting ``model.xrange = (-10, 10)``) invalidates the build, and the next
        ``predict_percept`` builds it again.

    """

    def __init__(self, implant, *, rho=200, regions=None, meridian_blend=0.1,
                 xrange=(-5, 5), yrange=(-5, 5), step=0.1,
                 grid_type='rect', thresh_percept=0,
                 min_current_spread=1e-8, visual_field_map=None, n_gray=None,
                 implant_position=(0, 0), implant_rotation=0,
                 implant_depth=0,
                 location_noise=None,
                 verbose=True, ndim=None, n_threads=None, n_jobs=None):
        super().__init__(
            spatial=ScoreboardSpatial(
                implant, rho=rho, regions=regions,
                meridian_blend=meridian_blend, xrange=xrange, yrange=yrange,
                step=step, grid_type=grid_type,
                thresh_percept=thresh_percept,
                min_current_spread=min_current_spread,
                visual_field_map=visual_field_map,
                n_gray=n_gray,
                implant_position=implant_position,
                implant_rotation=implant_rotation,
                implant_depth=implant_depth,
                location_noise=location_noise, verbose=verbose, ndim=ndim,
                n_threads=n_threads, n_jobs=n_jobs),
            temporal=None)
