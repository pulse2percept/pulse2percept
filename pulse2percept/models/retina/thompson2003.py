""":py:class:`~pulse2percept.models.retina.Thompson2003Model`,
   :py:class:`~pulse2percept.models.retina.Thompson2003Spatial` [Thompson2003]_"""

import numpy as np
from ...utils import sample
from ...topography.retina import Curcio1990Map
from ...units import um
from ..base import Model
from .base import RetinalSpatial, _warn_ignores_z


class Thompson2003Spatial(RetinalSpatial):
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
    :py:class:`~pulse2percept.models.retina.Thompson2003Model` for the standalone
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

    grid_type : {'rect', 'hex'}, optional
        Sampling lattice used for the visual-field grid.
    thresh_percept : float, optional
        Brightness values below this threshold are set to zero.
    visual_field_map : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
        Retinotopic map between visual-field and retinal coordinates. Defaults
        to :py:class:`~pulse2percept.topography.retina.Curcio1990Map`.
    n_gray : int or None, optional
        Number of gray levels in the returned percept. ``None`` disables
        gray-level quantization.
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

    verbose : bool, optional
        Whether to print status messages.
    ndim : list of int, optional
        Dimensionalities of ``visual_field_map`` accepted by the model.

    .. versionchanged:: 0.12.0

        Runs on Torch; ``n_threads`` and ``n_jobs`` were removed.
    """

    def __init__(self, implant, *, radius=None, dropout=None,
                 xrange=(-15, 15), yrange=(-15, 15), step=0.25,
                 grid_type='rect', thresh_percept=0,
                 visual_field_map=None, n_gray=None,
                 implant_position=(0, 0), implant_rotation=0,
                 implant_depth=0,
                 location_noise=None,
                 verbose=True, ndim=None):
        super().__init__(
            implant, radius=radius, dropout=dropout, xrange=xrange,
            yrange=yrange, step=step, grid_type=grid_type,
            thresh_percept=thresh_percept,
            visual_field_map=(Curcio1990Map() if visual_field_map is None else
                              visual_field_map),
            n_gray=n_gray,
            implant_position=implant_position,
            implant_rotation=implant_rotation,
            implant_depth=implant_depth,
            location_noise=location_noise, verbose=verbose,
            ndim=[2] if ndim is None else ndim)

    def get_default_params(self):
        """Return default model parameters."""
        base_params = super(Thompson2003Spatial, self).get_default_params()
        # Prediction runs on Torch's own thread pool:
        del base_params['n_threads'], base_params['n_jobs']
        params = {'radius': None, 'dropout': None,
                  'visual_field_map': Curcio1990Map()}
        return {**base_params, **params}

    def get_param_units(self):
        """Return units used to store model parameters."""
        return {**super().get_param_units(), 'radius': um}

    @property
    def _tensor_exact(self):
        # Dropout is sampled per prepared stimulus frame in `_predict_spatial`:
        return not self.dropout

    def _radius(self, electrode_array):
        """Return the phosphene radius (microns)."""
        if self.radius is not None:
            return self.radius
        if not hasattr(electrode_array, 'spacing'):
            raise NotImplementedError
        return 0.45 * electrode_array.spacing

    def _predict_spatial(self, electrode_array, stim):
        """Predict float32 brightness over the spatial grid."""
        import torch
        _warn_ignores_z(self, electrode_array)
        radius = self._radius(electrode_array)
        dropout = np.zeros(stim.shape, dtype=bool)
        if self.dropout is not None:
            for t in range(dropout.shape[1]):
                dropout[sample(np.arange(stim.shape[0]), k=self.dropout),
                        t] = True
        x_el, y_el, _ = self._electrode_coords(electrode_array, stim)
        waveform = torch.tensor(self._stim_values(stim), dtype=torch.float32)
        with torch.inference_mode():
            return self._predict_thompson_tensor(
                waveform, x_el, y_el, radius,
                dropout=torch.from_numpy(dropout)).numpy()

    def _predict_tensor(self, waveform, time):
        """Return the flat Torch response to an ``(n_electrodes, T)`` waveform.

        Runs the ``predict_percept`` kernel on every implant electrode.
        Geometry is fixed.
        """
        if self.n_gray is not None:
            # Quantization is discrete and has no exact gradient:
            raise NotImplementedError("Tensor prediction does not support "
                                      "n_gray; set n_gray=None.")
        if not self._tensor_exact:
            raise NotImplementedError("Tensor prediction does not support "
                                      "dropout; set dropout=None.")
        electrode_array = self.implant.electrode_array
        _warn_ignores_z(self, electrode_array)
        x_el, y_el, _ = self._electrode_coords(
            electrode_array, None, electrodes=self.implant.electrode_names)
        resp = self._predict_thompson_tensor(waveform, x_el, y_el,
                                             self._radius(electrode_array))
        return self._spatial_response(resp, time, None)

    def _predict_thompson_tensor(self, waveform, x_el, y_el, radius,
                                 dropout=None):
        """Return the flat thresholded ``(P, T)`` response.

        ``waveform`` rows follow the float32 electrode coordinates ``x_el``,
        ``y_el`` (microns). ``dropout`` is an optional boolean
        ``(n_electrodes, T)`` mask of dropped electrodes. Geometry is
        float32; the response has the dtype and device of ``waveform``.
        """
        import torch
        device = waveform.device
        x, y = (torch.as_tensor(np.ravel(c), dtype=torch.float32,
                                device=device)[:, None]
                for c in (self.grid.ret.x, self.grid.ret.y))
        x_el, y_el = (torch.as_tensor(c, dtype=torch.float32, device=device)
                      for c in (x_el, y_el))
        # Python floats holding float32 values keep Torch in float32. Strict
        # `<`: a point exactly at `radius` is outside the disk:
        radius = np.float32(radius)
        inside = (x - x_el) ** 2 + (y - y_el) ** 2 < float(radius * radius)
        # Unmapped grid points are zero:
        inside &= ~(x.isnan() | y.isnan())
        if dropout is not None:
            waveform = torch.where(dropout.to(device), 0.0, waveform)
        resp = inside.to(waveform.dtype) @ waveform
        thresh = float(np.float32(self.thresh_percept))
        # Zeroes only `|resp| < thresh`, so NaN propagates. `+ 0.0` turns -0.0
        # into 0.0:
        return torch.where(resp.abs() < thresh, 0.0, resp) + 0.0


class Thompson2003Model(Model):
    r"""Standalone spatial model of [Thompson2003]_.

    Uses :py:class:`~pulse2percept.models.retina.Thompson2003Spatial` without a
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

    grid_type : {'rect', 'hex'}, optional
        Sampling lattice used for the visual-field grid.
    thresh_percept : float, optional
        Brightness values below this threshold are set to zero.
    visual_field_map : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
        Retinotopic map between visual-field and retinal coordinates. Defaults
        to :py:class:`~pulse2percept.topography.retina.Curcio1990Map`.
    n_gray : int or None, optional
        Number of gray levels in the returned percept. ``None`` disables
        gray-level quantization.
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

    verbose : bool, optional
        Whether to print status messages.
    ndim : list of int, optional
        Dimensionalities of ``visual_field_map`` accepted by the spatial model.

    .. versionchanged:: 0.12.0

        Runs on Torch; ``n_threads`` and ``n_jobs`` were removed.
    """

    def __init__(self, implant, *, radius=None, dropout=None,
                 xrange=(-15, 15), yrange=(-15, 15), step=0.25,
                 grid_type='rect', thresh_percept=0,
                 visual_field_map=None, n_gray=None,
                 implant_position=(0, 0), implant_rotation=0,
                 implant_depth=0,
                 location_noise=None,
                 verbose=True, ndim=None):
        super().__init__(
            spatial=Thompson2003Spatial(
                implant, radius=radius, dropout=dropout, xrange=xrange,
                yrange=yrange, step=step, grid_type=grid_type,
                thresh_percept=thresh_percept,
                visual_field_map=visual_field_map,
                n_gray=n_gray,
                implant_position=implant_position,
                implant_rotation=implant_rotation,
                implant_depth=implant_depth,
                location_noise=location_noise, verbose=verbose, ndim=ndim),
            temporal=None)
