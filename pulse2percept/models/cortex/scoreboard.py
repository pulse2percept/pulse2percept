""":py:class:`~pulse2percept.models.cortex.ScoreboardSpatial`,
   :py:class:`~pulse2percept.models.cortex.ScoreboardModel`"""

from ..base import (Model, _blend_meridian, _is_tensor, _thread_params,
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

        ``rho`` is fixed, so phosphene size does not depend on the pulse:
        doubling amplitude doubles brightness at constant width. Use
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
        Fraction of peak Gaussian current spread below which an electrode is
        skipped at a grid point. The default 1e-8 (about 6.1 ``rho``) bounds
        the error at a point by ``min_current_spread`` times the summed
        amplitude across electrodes.
    regions : list of str, optional
        The regions to simulate. Options are 'v1', 'v2', or 'v3'. Default:
        ['v1']
    xrange : (x_min, x_max), optional
        A tuple indicating the range of x values to simulate (in degrees of
        visual angle). Negative x values lie left of fixation, positive x
        values right of it.
    yrange : tuple, (y_min, y_max), optional
        A tuple indicating the range of y values to simulate (in degrees of
        visual angle). Negative y values lie below fixation, positive y
        values above it.
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
    #: Spatial-only use may read dimensionless image or video values as
    #: relative electrode drive, so an encoder is optional.
    _accepts_dimensionless_drive = True

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
        # Cortical coordinates are in microns (see `CorticalMap`):
        return {**super().get_param_units(), 'rho': um, 'meridian_blend': dva}

    def _build(self):
        _warn_rho_vs_pitch(self)

    def _postprocess_spatial(self, resp):
        """Blend the percept across the vertical meridian

        Defined here because the seam comes from the split map, which other
        `CortexSpatial` models may not use.
        """
        blended = _blend_meridian(resp, self.grid, 'vertical',
                                  self.meridian_blend)
        if blended is resp:
            return resp
        # Restore percept threshold after blending:
        if _is_tensor(blended):
            import torch
            return torch.where(blended.abs() >= self.thresh_percept, blended,
                               0.0)
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
        # One tissue location per electrode (displaced through its own region
        # by `location_noise`), spread over every simulated region's grid:
        xyz = self._electrode_coords(electrode_array, stim)
        coords = {region: xyz for region in self.regions}
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

    def _predict_tensor(self, waveform, time):
        """Return the flat Torch response to an ``(n_electrodes, T)`` waveform.

        Same Gaussian, cutoff, hemisphere separation, per-region threshold and
        meridian blending as ``predict_percept``, computed as one
        ``(P, E) @ (E, T)`` product per region. Geometry is fixed.
        """
        import torch
        if self.n_gray is not None:
            # Quantization is discrete and has no exact gradient:
            raise NotImplementedError("Tensor prediction does not support "
                                      "n_gray; set n_gray=None.")
        x_el, y_el, z_el = self._electrode_coords(
            self.implant.electrode_array, None,
            electrodes=self.implant.electrode_names)
        rho = np.float32(self.rho)
        cutoff_r2 = self._cutoff_r2(self.rho)
        resp = 0
        for region in self.regions:
            x_grid = self.grid[region].x.reshape((-1, 1))
            y_grid = self.grid[region].y.reshape((-1, 1))
            # float32, as in `fast_scoreboard`:
            dx = x_grid - x_el
            dy = y_grid - y_el
            r2 = dx * dx + dy * dy
            if self.visual_field_map.ndim == 3:
                # A 2D map ignores electrode z; a 3D one adds depth, as in
                # `fast_scoreboard_3d`:
                dz = self.grid[region].z.reshape((-1, 1)) - z_el
                r2 = r2 + dz * dz
            weights = np.exp(-r2 / (np.float32(2) * rho * rho))
            # Drops pairs beyond the cutoff and unmapped (NaN) grid points:
            drop = ~(r2 <= cutoff_r2)
            if self.visual_field_map.split_map:
                # No current spreads between hemispheres:
                boundary = np.float32(self.visual_field_map.left_offset / 2)
                drop |= (x_grid < boundary) != (x_el < boundary)
            weights[drop] = 0
            weights = torch.as_tensor(weights, dtype=waveform.dtype,
                                      device=waveform.device)
            region_resp = weights @ waveform
            # Each region is thresholded before the sum, as in Cython:
            resp = resp + torch.where(
                region_resp.abs() >= self.thresh_percept, region_resp, 0.0)
        resp = self._postprocess_spatial(resp)
        return self._spatial_response(resp, time, None)


class ScoreboardModel(Model):
    """Cortical adaptation of scoreboard model from [Beyeler2019]_ (standalone model)

    Implements the scoreboard model described in [Beyeler2019]_, where percepts
    from each electrode are Gaussian blobs. The percepts resulting from different 
    cortical regions (e.g. v1/v2/v3) are added linearly. The `rho` parameter 
    modulates phosphene size.

    .. note::

        Use :class:`ScoreboardSpatial` to combine the spatial model with a
        temporal model.

    .. note::

        For this spatial-only model, dimensionless image and video values are
        treated as relative electrode amplitudes, so an implant encoder is not
        required. Physical or pulse-dependent models still require encoded
        stimulation.

    .. warning::

        ``rho`` is fixed, so phosphene size does not depend on the pulse:
        doubling amplitude doubles brightness at constant width. Use
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
        Fraction of peak Gaussian current spread below which an electrode is
        skipped at a grid point. The default 1e-8 (about 6.1 ``rho``) bounds
        the error at a point by ``min_current_spread`` times the summed
        amplitude across electrodes.
    regions : list of str, optional
        The regions to simulate. Options are 'v1', 'v2', or 'v3'. Default:
        ['v1']
    xrange : (x_min, x_max), optional
        A tuple indicating the range of x values to simulate (in degrees of
        visual angle). Negative x values lie left of fixation, positive x
        values right of it.
    yrange : tuple, (y_min, y_max), optional
        A tuple indicating the range of y values to simulate (in degrees of
        visual angle). Negative y values lie below fixation, positive y
        values above it.
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
