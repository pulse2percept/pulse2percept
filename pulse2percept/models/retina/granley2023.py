""":py:class:`~pulse2percept.models.retina.Granley2023Model` [Granley2023]_"""
import numpy as np

from ...topography.retina import Watson2014Map
from ...units import deg, dva
from .base import RetinalSpatial, _warn_ignores_z
from .beyeler2019 import _AxonBundleMixin
from .granley2021 import _BiphasicModel, _BiphasicSpatialMixin


def _mvg_shape(freq, amp, pdur, *, rho, lam, a0, a1, a2, a3, a4, amp_cutoff):
    """Return per-electrode peak brightness, area (pixels²) and eccentricity.

    Takes Torch tensors of frequency (Hz), amplitude (xTh) and phase duration
    (ms)."""
    import torch
    bright = torch.where(amp > amp_cutoff,
                         a0 * amp.clamp(min=1e-5) ** a1 + a2 * freq, 0.0)
    area = (rho * a3 * amp).clamp(min=1)
    ecc = (lam * (pdur / 0.45) ** a4).clamp(0, 0.99)
    return bright, area, ecc


def _mvg_response(freq, amp, pdur, x, y, x_el, y_el, theta, *,
                  thresh_percept, **params):
    """Return the summed oriented-Gaussian response at each pixel.

    ``freq``, ``amp``, ``pdur``, ``x_el``, ``y_el`` and ``theta`` are Torch
    tensors with one entry per active electrode; ``x``, ``y`` hold the pixel
    coordinates. Coordinates are in pixels with ``y`` pointing down the image
    rows. ``params`` go to ``_mvg_shape``. Differentiable away from the
    amplitude cutoff, area floor and eccentricity clip."""
    import torch
    bright, area, ecc = _mvg_shape(freq, amp, pdur, **params)
    # Variances whose `thresh_percept` contour has this area and eccentricity:
    contour = -2 * np.pi * np.log(thresh_percept)
    minor = torch.sqrt(1 - ecc ** 2)
    var_x = area * minor / contour
    var_y = area / (contour * minor)
    dx = x[:, None] - x_el
    dy = y[:, None] - y_el
    cos, sin = torch.cos(theta), torch.sin(theta)
    # Quadratic form d.T @ inv(R @ diag(var_x, var_y) @ R.T) @ d:
    quad = ((dx * cos + dy * sin) ** 2 / var_x +
            (dy * cos - dx * sin) ** 2 / var_y)
    return (bright * torch.exp(-0.5 * quad)).sum(dim=1)


def _mirrors(visual_field_map, x, y, eps=1.0):
    """Return whether ``ret_to_dva`` reverses orientation at each point (um)"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xr, yr = visual_field_map.ret_to_dva(x + eps, y)
    xl, yl = visual_field_map.ret_to_dva(x - eps, y)
    xu, yu = visual_field_map.ret_to_dva(x, y + eps)
    xd, yd = visual_field_map.ret_to_dva(x, y - eps)
    return (xr - xl) * (yu - yd) - (xu - xd) * (yr - yl) < 0


class _Granley2023Spatial(_BiphasicSpatialMixin, _AxonBundleMixin,
                          RetinalSpatial):
    """Spatial stage of :py:class:`Granley2023Model`."""

    #: No effect models; ``rho`` is used directly.
    _shared_with_effect = {}

    def __init__(self, implant, *, rho=75, lam=0.9, orient_scale=1,
                 a0=0.4733, a1=0.5211, a2=0.016, a3=0.5, a4=-0.2122,
                 amp_cutoff=0.25, thresh_percept=np.exp(-2),
                 xrange=(-15, 15), yrange=(-15, 15), step=0.25,
                 visual_field_map=None, n_gray=None,
                 implant_position=(0, 0), implant_rotation=0,
                 implant_depth=0, location_noise=None, loc_od=(15.5, 1.5),
                 n_axons=1000, axons_range=(-180, 180), n_ax_segments=500,
                 ax_segments_range=(0, 50), verbose=True):
        super().__init__(
            implant, rho=rho, lam=lam, orient_scale=orient_scale, a0=a0,
            a1=a1, a2=a2, a3=a3, a4=a4, amp_cutoff=amp_cutoff,
            thresh_percept=thresh_percept, xrange=xrange, yrange=yrange,
            step=step,
            visual_field_map=(Watson2014Map() if visual_field_map is None else
                              visual_field_map),
            n_gray=n_gray, implant_position=implant_position,
            implant_rotation=implant_rotation, implant_depth=implant_depth,
            location_noise=location_noise, loc_od=loc_od, n_axons=n_axons,
            axons_range=axons_range, n_ax_segments=n_ax_segments,
            ax_segments_range=ax_segments_range, verbose=verbose)
        self._built_eye = None
        # Build-time geometry, in output pixels:
        self._el_index = None
        self._el_x = None
        self._el_y = None
        self._el_theta = None
        self._px = None
        self._py = None

    def get_default_params(self):
        params = {
            'rho': 75,
            'lam': 0.9,
            'orient_scale': 1,
            'a0': 0.4733,
            'a1': 0.5211,
            'a2': 0.016,
            'a3': 0.5,
            'a4': -0.2122,
            'amp_cutoff': 0.25,
            'thresh_percept': np.exp(-2),
            'loc_od': (15.5, 1.5),
            'n_axons': 1000,
            'axons_range': (-180, 180),
            'n_ax_segments': 500,
            'ax_segments_range': (0, 50),
            'visual_field_map': Watson2014Map(),
        }
        return {**super().get_default_params(), **params}

    def get_param_units(self):
        # `rho` is an area in output pixels and has no p2p unit:
        return {**super().get_param_units(), 'loc_od': dva,
                'axons_range': deg}

    def _build(self):
        step = np.unique(np.ravel(self.step))
        if step.size != 1:
            raise ValueError(f"{type(self).__name__} requires square pixels: "
                             f"'rho' is an area in pixels, so x and y 'step' "
                             f"must be equal, not {self.step}.")
        if not 0 < self.thresh_percept < 1:
            raise ValueError(f"'thresh_percept' sets the contour that 'rho' "
                             f"and 'lam' describe and must lie in (0, 1), not "
                             f"{self.thresh_percept}.")
        self._built_eye = self.eye
        self._correct_loc_od()
        names = self.implant.electrode_names
        x_ret, y_ret, _ = self._electrode_coords(self.implant.electrode_array,
                                                 None, electrodes=names)
        tangent = self.calc_bundle_tangent_fast(
            x_ret, y_ret, bundles=self.grow_axon_bundles())
        # The tangent is measured in retinal microns, as in [Granley2023]_;
        # mirror it where the map flips the retina into the visual field:
        tangent = np.where(_mirrors(self.visual_field_map, x_ret, y_ret),
                           -tangent, tangent)
        theta = tangent - np.pi / 2
        theta = np.where(theta < -np.pi / 2, theta + np.pi, theta)
        x_dva, y_dva = self.visual_field_map.ret_to_dva(x_ret, y_ret)
        # Pixel coordinates with y pointing down the image rows:
        step = float(step[0])
        self._el_index = {name: i for i, name in enumerate(names)}
        self._el_x = np.asarray(x_dva, dtype=np.float32) / step
        self._el_y = -np.asarray(y_dva, dtype=np.float32) / step
        self._el_theta = np.asarray(theta, dtype=np.float32)
        self._px = self.grid.x.ravel().astype(np.float32) / step
        self._py = -self.grid.y.ravel().astype(np.float32) / step

    def _predict_spatial(self, electrode_array, stim):
        """Predict the representative spatial percept."""
        import torch
        _warn_ignores_z(self, electrode_array)
        active, elec_params = self._elec_params(stim)
        idx = [self._el_index[name] for name in active]
        freq, amp, pdur = torch.as_tensor(elec_params).T
        with torch.inference_mode():
            return _mvg_response(
                freq, amp, pdur, torch.as_tensor(self._px),
                torch.as_tensor(self._py), torch.as_tensor(self._el_x[idx]),
                torch.as_tensor(self._el_y[idx]),
                torch.as_tensor(self._el_theta[idx]) * self.orient_scale,
                rho=self.rho, lam=self.lam, a0=self.a0, a1=self.a1,
                a2=self.a2, a3=self.a3, a4=self.a4,
                amp_cutoff=self.amp_cutoff,
                thresh_percept=self.thresh_percept).numpy()


class Granley2023Model(_BiphasicModel):
    r"""Oriented Gaussian phosphene model of [Granley2023]_.

    The phosphene model of the human-in-the-loop optimization (HILO) study.
    Each stimulated electrode produces an elliptical Gaussian centered on the
    electrode. Pulse amplitude, frequency, and phase duration set its
    brightness, area, and eccentricity; the local nerve fiber bundle
    [Jansonius2009]_ sets only its orientation. Unlike
    :py:class:`~pulse2percept.models.retina.BiphasicAxonMapModel`, activation
    does not spread along the axon. Returns one representative percept for the
    full pulse train.

    For threshold-scaled amplitude :math:`\tilde{a}`, frequency :math:`f`, and
    phase duration :math:`t` (ms), each electrode contributes

    .. math::

        B &= \begin{cases}
            a_0 \tilde{a}^{a_1} + a_2 f & \tilde{a} > \mathrm{amp\_cutoff} \\
            0 & \text{otherwise}
        \end{cases} \\
        A &= \max(\rho a_3 \tilde{a}, 1) \\
        \epsilon &= \mathrm{clip}\left(\lambda (t / 0.45)^{a_4},
                                       0, 0.99\right)

    The phosphene :math:`B \exp(-\frac{1}{2} d^T \Sigma^{-1} d)` has peak
    :math:`B`. Its covariance :math:`\Sigma` is chosen so the contour at
    ``thresh_percept`` has area :math:`A` (pixels²) and eccentricity
    :math:`\epsilon`, with the major axis along the local bundle. Electrode
    contributions are summed.

    .. important::

        ``rho`` is an area in output pixels, as in [Granley2023]_. A pixel is
        ``step`` x ``step`` dva, so changing ``step`` changes the phosphene's
        angular size.

    Stimuli must retain their cathodic-first pulse-train description: either
    :py:class:`~pulse2percept.stimuli.BiphasicPulseTrain` objects, or a still
    image encoded with the standard biphasic encoder pulse (see
    :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`). Give amplitude in
    multiples of perceptual threshold (:py:data:`~pulse2percept.units.xTh`) or
    provide a threshold calibration for current-valued amplitudes. No
    phase-duration threshold correction is applied. Interphase duration and
    exact pulse timing are ignored. Videos are not supported.

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.retina.RetinalImplant`
        Implant whose electrode geometry and eye are modeled.
    rho : float, optional
        Phosphene area in output pixels² at :math:`\tilde{a} = 1 / a_3`
        (2 xTh by default).
    lam : float, optional
        Phosphene eccentricity at 0.45 ms phase duration, in [0, 0.99].
    orient_scale : float, optional
        Multiplies the orientation angle derived from the bundle tangent.
    a0, a1 : float, optional
        Brightness coefficient and amplitude exponent.
    a2 : float, optional
        Brightness slope per Hz.
    a3 : float, optional
        Area slope per threshold multiple.
    a4 : float, optional
        Phase-duration exponent of eccentricity.
    amp_cutoff : float, optional
        Amplitudes (xTh) at or below this produce no phosphene.
    thresh_percept : float, optional
        Relative brightness of the contour that ``rho`` and ``lam`` describe,
        in (0, 1). Does not threshold the percept.
    xrange : (float, float) or Quantity, optional
        Horizontal visual-field extent in degrees of visual angle.
    yrange : (float, float) or Quantity, optional
        Vertical visual-field extent in degrees of visual angle.
    step : float or Quantity, optional
        Grid spacing in degrees of visual angle. Sets the pixel size of
        ``rho``; x and y spacing must be equal.
    visual_field_map : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
        Retinotopic map between visual-field and retinal coordinates. Defaults
        to :py:class:`~pulse2percept.topography.retina.Watson2014Map`.
    n_gray : int or None, optional
        Number of gray levels in the returned percept. ``None`` disables
        gray-level quantization.
    implant_position : (x, y) or Quantity, optional
        Position of the device-local origin, in tissue coordinates or dva.
    implant_rotation : float or Quantity, optional
        In-plane rotation (deg), positive counter-clockwise.
    implant_depth : float or Quantity, optional
        Signed offset (um) along the normal of a 2D tissue map. Ignored.
    location_noise : float or None, optional
        Standard deviation of fixed electrode-specific phosphene offsets, in
        dva. Moves both the phosphene center and the bundle orientation.
        ``None`` or 0 disables it.
    loc_od : (float, float) or Quantity, optional
        Optic-disc location in degrees of visual angle. Its horizontal sign is
        set from the bound implant's eye.
    n_axons : int, optional
        Number of nerve fiber bundles generated.
    axons_range : (float, float) or Quantity, optional
        Range of initial bundle angles ``phi0`` in the [Jansonius2009]_ model.
    n_ax_segments : int, optional
        Number of radial samples used to generate each bundle.
    ax_segments_range : (float, float), optional
        Radial-coordinate range used to generate each bundle.
    verbose : bool, optional
        Whether to print status messages.

    Notes
    -----
    The [Granley2023]_ experiments used
    ``xrange=(-12, 12)``, ``yrange=(-12, 12)``, ``step=0.5`` and
    ``rho=62.5``.

    .. versionadded:: 0.12.0

    Examples
    --------
    .. code-block:: python

        import pulse2percept as p2p

        implant = p2p.implants.retina.ArgusII()
        model = p2p.models.retina.Granley2023Model(implant)
        train = p2p.stimuli.BiphasicPulseTrain(20, 2 * p2p.units.xTh, 0.45)
        percept = model.predict_percept({'C5': train})
    """

    def __init__(self, implant, *, rho=75, lam=0.9, orient_scale=1,
                 a0=0.4733, a1=0.5211, a2=0.016, a3=0.5, a4=-0.2122,
                 amp_cutoff=0.25, thresh_percept=np.exp(-2),
                 xrange=(-15, 15), yrange=(-15, 15), step=0.25,
                 visual_field_map=None, n_gray=None,
                 implant_position=(0, 0), implant_rotation=0,
                 implant_depth=0, location_noise=None, loc_od=(15.5, 1.5),
                 n_axons=1000, axons_range=(-180, 180), n_ax_segments=500,
                 ax_segments_range=(0, 50), verbose=True):
        super().__init__(
            spatial=_Granley2023Spatial(
                implant, rho=rho, lam=lam, orient_scale=orient_scale, a0=a0,
                a1=a1, a2=a2, a3=a3, a4=a4, amp_cutoff=amp_cutoff,
                thresh_percept=thresh_percept, xrange=xrange, yrange=yrange,
                step=step, visual_field_map=visual_field_map, n_gray=n_gray,
                implant_position=implant_position,
                implant_rotation=implant_rotation,
                implant_depth=implant_depth, location_noise=location_noise,
                loc_od=loc_od, n_axons=n_axons, axons_range=axons_range,
                n_ax_segments=n_ax_segments,
                ax_segments_range=ax_segments_range, verbose=verbose),
            temporal=None)
