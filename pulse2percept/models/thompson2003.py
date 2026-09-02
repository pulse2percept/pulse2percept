""":py:class:`~pulse2percept.models.Thompson2003Model`,
   :py:class:`~pulse2percept.models.Thompson2003Spatial` [Thompson2003]_"""

import numpy as np
import copy
from ..utils import sample
from ..topography import Curcio1990Map
from ..units import um
from ..models import Model, SpatialModel
from .base import _thread_params, _warn_ignores_z
from ._thompson2003 import fast_thompson2003

import warnings


class Thompson2003Spatial(SpatialModel):
    r"""Spatial model of [Thompson2003]_.

    Models each electrode as a circular phosphene with uniform brightness
    inside a fixed radius and zero contribution outside. For electrode
    :math:`e`, let

    .. math::

        r_e(x,y) =
        \sqrt{(x-x_e)^2 + (y-y_e)^2}.

    The spatial response is

    .. math::

        I(x,y,t) =
        \sum_{e \in E}
        [1-D_e(t)]\,A_e(t)\,
        \mathbf{1}\left[r_e(x,y) < R\right],

    where :math:`A_e(t)` is stimulus amplitude, :math:`R` is ``radius``,
    :math:`D_e(t)` is 1 for a dropped electrode and 0 otherwise, and
    :math:`\mathbf{1}` is the indicator function. Contributions from
    overlapping disks add linearly.

    Dropout is resampled independently for each stimulus frame. Electrode
    ``z`` coordinates are ignored.

    Use this class to combine the spatial model with a temporal model. Use
    :py:class:`~pulse2percept.models.Thompson2003Model` for the standalone
    spatial model.

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.Implant`
        Implant whose electrode geometry is modeled.

        .. versionadded:: 0.11.0

    radius : float, Quantity, or None, optional
        Radius of each circular phosphene, in microns. If ``None``, uses ``0.45
        * implant.electrode_array.spacing``, giving a disk diameter equal to
        90% of the electrode spacing. The electrode array must provide a
        ``spacing`` attribute. Default: ``None``.
    dropout : int, float, or None, optional
        Number or fraction of electrodes randomly omitted from each stimulus
        frame. An integer gives the number of dropped electrodes; a float in
        [0, 1] gives their fraction. ``None`` disables dropout.
    xrange : (float, float) or Quantity, optional
        Horizontal visual-field extent in degrees of visual angle. A physical
        retinal extent may instead be resolved through ``visual_field_map``.
    yrange : (float, float) or Quantity, optional
        Vertical visual-field extent in degrees of visual angle. A physical
        retinal extent may instead be resolved through ``visual_field_map``.
    step : float, (float, float), or Quantity, optional
        Grid spacing in degrees of visual angle. A pair specifies separate x
        and y spacing.

        .. versionchanged:: 0.10.0
            Renamed from ``xystep``; ``xystep`` was removed in 0.11.0.

    grid_type : {'rectangular', 'hexagonal'}, optional
        Sampling lattice used for the visual-field grid.
    thresh_percept : float, optional
        Brightness values below this threshold are set to zero.
    min_current_spread : float, optional
        Inherited Gaussian current-spread cutoff. This parameter is not used
        by ``Thompson2003Spatial``.
    visual_field_map : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
        Retinotopic map between visual-field and retinal coordinates. Defaults
        to :py:class:`~pulse2percept.topography.Curcio1990Map`.
    n_gray : int or None, optional
        Number of gray levels in the returned percept. ``None`` disables
        gray-level quantization.
    noise : float, int, or None, optional
        Salt-and-pepper noise applied to each percept frame. An integer gives
        the number of affected pixels; a float in [0, 1] gives their fraction.
    verbose : bool, optional
        Whether to print status messages.
    ndim : list of int, optional
        Dimensionalities of ``visual_field_map`` accepted by the model.
    n_threads : int, optional
        Inherited OpenMP thread count. The Thompson spatial kernel does not
        currently use this parameter.
    n_jobs : int or None, optional
        Alias for ``n_threads``. The Thompson spatial kernel does not currently
        use this parameter.
    """

    def __init__(self, implant, *, radius=None, dropout=None,
                 xrange=(-15, 15), yrange=(-15, 15), step=0.25,
                 grid_type='rectangular', thresh_percept=0,
                 min_current_spread=1e-8, visual_field_map=None, n_gray=None,
                 noise=None,
                 verbose=True, ndim=None, n_threads=None, n_jobs=None):
        super().__init__(
            implant, radius=radius, dropout=dropout, xrange=xrange,
            yrange=yrange, step=step, grid_type=grid_type,
            thresh_percept=thresh_percept,
            min_current_spread=min_current_spread,
            visual_field_map=(Curcio1990Map() if visual_field_map is None else
                              visual_field_map),
            n_gray=n_gray, noise=noise, verbose=verbose,
            ndim=[2] if ndim is None else ndim,
            **_thread_params(n_threads, n_jobs))

    def get_default_params(self):
        """Return default model parameters."""
        base_params = super(Thompson2003Spatial, self).get_default_params()
        params = {'radius': None, 'dropout': None,
                  'visual_field_map': Curcio1990Map()}
        return {**base_params, **params}

    def get_param_units(self):
        """Return units used to store model parameters."""
        return {**super().get_param_units(), 'radius': um}

    def _predict_spatial(self, electrode_array, stim):
        """Predict the spatial response."""
        _warn_ignores_z(self, electrode_array)
        radius = self.radius
        if radius is None:
            if not hasattr(electrode_array, 'spacing'):
                raise NotImplementedError
            radius = 0.45 * electrode_array.spacing
        dropout = np.zeros(stim.shape, dtype=np.uint8)
        if self.dropout is not None:
            for t in range(dropout.shape[1]):
                dropout[sample(np.arange(stim.shape[0]), k=self.dropout),
                        t] = 255
        x_el, y_el, _ = self._electrode_coords(electrode_array, stim)
        return fast_thompson2003(self._stim_values(stim), x_el, y_el,
                                 self.grid.ret.x.ravel(),
                                 self.grid.ret.y.ravel(),
                                 dropout.astype(np.uint8),
                                 radius,
                                 self.thresh_percept)


class Thompson2003Model(Model):
    r"""Standalone spatial model of [Thompson2003]_.

    Uses :py:class:`~pulse2percept.models.Thompson2003Spatial` without a
    temporal component. See that class for the top-hat disk equation and
    dropout model.

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.Implant`
        Implant whose electrode geometry is modeled.

        .. versionadded:: 0.11.0

    radius : float, Quantity, or None, optional
        Radius of each circular phosphene, in microns. If ``None``, uses
        ``0.45 * implant.electrode_array.spacing``. Default: ``None``.
    dropout : int, float, or None, optional
        Number or fraction of electrodes randomly omitted from each stimulus
        frame. ``None`` disables dropout.
    xrange : (float, float) or Quantity, optional
        Horizontal visual-field extent in degrees of visual angle. A physical
        retinal extent may instead be resolved through ``visual_field_map``.
    yrange : (float, float) or Quantity, optional
        Vertical visual-field extent in degrees of visual angle. A physical
        retinal extent may instead be resolved through ``visual_field_map``.
    step : float, (float, float), or Quantity, optional
        Grid spacing in degrees of visual angle. A pair specifies separate x
        and y spacing.

        .. versionchanged:: 0.10.0
            Renamed from ``xystep``; ``xystep`` was removed in 0.11.0.

    grid_type : {'rectangular', 'hexagonal'}, optional
        Sampling lattice used for the visual-field grid.
    thresh_percept : float, optional
        Brightness values below this threshold are set to zero.
    min_current_spread : float, optional
        Inherited Gaussian current-spread cutoff. Not used by the Thompson
        spatial model.
    visual_field_map : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
        Retinotopic map between visual-field and retinal coordinates. Defaults
        to :py:class:`~pulse2percept.topography.Curcio1990Map`.
    n_gray : int or None, optional
        Number of gray levels in the returned percept. ``None`` disables
        gray-level quantization.
    noise : float, int, or None, optional
        Salt-and-pepper noise applied to each percept frame.
    verbose : bool, optional
        Whether to print status messages.
    ndim : list of int, optional
        Dimensionalities of ``visual_field_map`` accepted by the spatial model.
    n_threads : int, optional
        Inherited OpenMP thread count. The Thompson spatial kernel does not
        currently use this parameter.
    n_jobs : int or None, optional
        Alias for ``n_threads``. The Thompson spatial kernel does not currently
        use this parameter.
    """

    def __init__(self, implant, *, radius=None, dropout=None,
                 xrange=(-15, 15), yrange=(-15, 15), step=0.25,
                 grid_type='rectangular', thresh_percept=0,
                 min_current_spread=1e-8, visual_field_map=None, n_gray=None,
                 noise=None,
                 verbose=True, ndim=None, n_threads=None, n_jobs=None):
        super().__init__(
            spatial=Thompson2003Spatial(
                implant, radius=radius, dropout=dropout, xrange=xrange,
                yrange=yrange, step=step, grid_type=grid_type,
                thresh_percept=thresh_percept,
                min_current_spread=min_current_spread,
                visual_field_map=visual_field_map,
                n_gray=n_gray, noise=noise, verbose=verbose, ndim=ndim,
                n_threads=n_threads, n_jobs=n_jobs),
            temporal=None)
