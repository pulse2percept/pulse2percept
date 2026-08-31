""":py:class:`~pulse2percept.models.Thompson2003Model`,
   :py:class:`~pulse2percept.models.Thompson2003Spatial` [Thompson2003]_"""

import numpy as np
import copy
from ..utils import sample
from ..topography import Curcio1990Map
from ..units import um
from ..models import Model, SpatialModel
from .base import _warn_ignores_z
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
    implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`
        Implant whose electrode geometry is modeled.

        .. versionadded:: 0.11.0

    radius : float, Quantity, or None, optional
        Radius of each circular phosphene, in microns. If ``None``, uses
        ``0.45 * implant.earray.spacing``, giving a disk diameter equal to 90%
        of the electrode spacing. The electrode array must provide a
        ``spacing`` attribute. Default: ``None``.
    dropout : int, float, or None, optional
        Number or fraction of electrodes randomly omitted from each stimulus
        frame. An integer gives the number of dropped electrodes; a float in
        [0, 1] gives their fraction. ``None`` disables dropout.
    xrange : (float, float) or Quantity, optional
        Horizontal visual-field extent in degrees of visual angle. A physical
        retinal extent may instead be resolved through ``vfmap``.
    yrange : (float, float) or Quantity, optional
        Vertical visual-field extent in degrees of visual angle. A physical
        retinal extent may instead be resolved through ``vfmap``.
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
    vfmap : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
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
        Dimensionalities of ``vfmap`` accepted by the model.
    n_threads : int, optional
        Inherited OpenMP thread count. The Thompson spatial kernel does not
        currently use this parameter.
    n_jobs : int or None, optional
        Alias for ``n_threads``. The Thompson spatial kernel does not currently
        use this parameter.
    """

    def get_default_params(self):
        """Return default model parameters."""
        base_params = super(Thompson2003Spatial, self).get_default_params()
        params = {'radius': None, 'dropout': None,
                  'vfmap': Curcio1990Map()}
        return {**base_params, **params}

    def get_param_units(self):
        """Return units used to store model parameters."""
        return {**super().get_param_units(), 'radius': um}

    def _predict_spatial(self, earray, stim):
        """Predict the spatial response."""
        _warn_ignores_z(self, earray)
        radius = self.radius
        if radius is None:
            if not hasattr(earray, 'spacing'):
                raise NotImplementedError
            radius = 0.45 * earray.spacing
        dropout = np.zeros(stim.shape, dtype=np.uint8)
        if self.dropout is not None:
            for t in range(dropout.shape[1]):
                dropout[sample(np.arange(stim.shape[0]), k=self.dropout),
                        t] = 255
        x_el, y_el, _ = self._electrode_coords(earray, stim)
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
    implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`
        Implant whose electrode geometry is modeled.

        .. versionadded:: 0.11.0

    radius : float, Quantity, or None, optional
        Radius of each circular phosphene, in microns. If ``None``, uses
        ``0.45 * implant.earray.spacing``. Default: ``None``.
    dropout : int, float, or None, optional
        Number or fraction of electrodes randomly omitted from each stimulus
        frame. ``None`` disables dropout.
    xrange : (float, float) or Quantity, optional
        Horizontal visual-field extent in degrees of visual angle. A physical
        retinal extent may instead be resolved through ``vfmap``.
    yrange : (float, float) or Quantity, optional
        Vertical visual-field extent in degrees of visual angle. A physical
        retinal extent may instead be resolved through ``vfmap``.
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
    vfmap : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
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
        Dimensionalities of ``vfmap`` accepted by the spatial model.
    n_threads : int, optional
        Inherited OpenMP thread count. The Thompson spatial kernel does not
        currently use this parameter.
    n_jobs : int or None, optional
        Alias for ``n_threads``. The Thompson spatial kernel does not currently
        use this parameter.
    """

    def __init__(self, implant, **params):
        super(Thompson2003Model, self).__init__(
            spatial=Thompson2003Spatial(implant), temporal=None, **params)
