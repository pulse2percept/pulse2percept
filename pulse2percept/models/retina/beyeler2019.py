""":py:class:`~pulse2percept.models.retina.AxonMapModel`,
   :py:class:`~pulse2percept.models.retina.AxonMapSpatial` [Beyeler2019]_"""

import os
import numpy as np
import pickle
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from ...units import deg, dimensionless, dva, um
from ...utils.constants import UM_PER_MM, ZORDER
from ...topography.retina import Watson2014Map
from ...implants import ElectrodeArray
from ...stimuli import Stimulus
from ..base import (Model, _blend_meridian, _draw_placed_implant,
                    _is_tensor, _thread_params, _warn_rho_vs_pitch)
from .._scoreboard import fast_scoreboard
from .base import RetinalSpatial, _warn_ignores_z
from ._beyeler2019 import (fast_axon_map, fast_jansonius,
                           fast_find_closest_axon)

import warnings


#: Version of the serialized ``axon_pickle`` payload. Increment when its
#: layout or parameter semantics change.
_AXON_CACHE_VERSION = 3

#: Approximate working-memory target (bytes) per pixel block in
#: ``AxonMapSpatial._predict_tensor``; see ``_axon_blocks``.
_AXON_BLOCK_BYTES = 16 * 2 ** 20


def _is_axon_cache(payload):
    """Return whether ``payload`` uses the current axon-cache format."""
    return (isinstance(payload, tuple) and len(payload) == 4 and
            payload[0] == _AXON_CACHE_VERSION)


def _flatten_bundles(bundles):
    """Flatten bundles while preserving shared bundle references.

    Parameters
    ----------
    bundles : list of (N, 2) arrays
        One bundle per grid point. Repeated references to the same array are
        stored once.

    Returns
    -------
    flat : (M, 2) ndarray
        Distinct bundles concatenated.
    boff : (n_distinct + 1,) ndarray
        Offsets of distinct bundles in ``flat``.
    bundle_id : (len(bundles),) ndarray
        Distinct-bundle index for each input entry."""
    seen = {}
    distinct = []
    bundle_id = np.empty(len(bundles), dtype=np.intp)
    for pos, bundle in enumerate(bundles):
        idx = seen.get(id(bundle))
        if idx is None:
            idx = seen[id(bundle)] = len(distinct)
            distinct.append(bundle)
        bundle_id[pos] = idx
    lens = np.array([len(bundle) for bundle in distinct], dtype=np.intp)
    if np.any(lens == 0):
        raise ValueError("Every bundle must have at least one segment.")
    flat = np.ascontiguousarray(np.concatenate(distinct))
    return flat, np.concatenate(([0], np.cumsum(lens))), bundle_id


def _axon_blocks(counts, n_el, n_time, itemsize, budget):
    """Return ``(p0, p1)`` pixel ranges whose tensor work fits in ``budget``.

    ``counts`` holds the retained segments per pixel. Counts the float32
    Gaussian and its cast to ``itemsize`` (segments x electrodes), the
    segment response and its ``abs`` (segments x T), and the padded ``abs``
    (pixels x longest axon x T). A range holds at least one pixel, so a single
    long axon may exceed ``budget``.
    """
    per_seg = n_el * (4 + itemsize) + 2 * n_time * itemsize
    per_slot = n_time * itemsize
    blocks = []
    p0, n_seg, longest = 0, 0, 1
    for p, count in enumerate(counts.tolist()):
        grown, longer = n_seg + count, max(longest, count)
        if (p > p0 and grown * per_seg + (p - p0 + 1) * longer * per_slot >
                budget):
            blocks.append((p0, p))
            p0, grown, longer = p, count, max(1, count)
        n_seg, longest = grown, longer
    blocks.append((p0, len(counts)))
    return blocks


class ScoreboardSpatial(RetinalSpatial):
    r"""Scoreboard model of [Beyeler2019]_ (spatial module only).

    Models each electrode's percept as a circular Gaussian. Use
    :py:class:`~pulse2percept.models.retina.ScoreboardModel` for a standalone model.

    The spatial response is modeled as a Gaussian centered on each electrode:

    .. math::

        I(x, y) =
        \sum_{e \in E}
        a_e
        \exp\left(
            -\frac{(x-x_e)^2 + (y-y_e)^2}{2\rho^2}
        \right),

    where :math:`a_e` is the drive at site :math:`e`, and :math:`\rho`
    controls the spatial spread of activation. Larger values of
    :math:`\rho` produce broader phosphenes.

    For current-driven implants, :math:`a_e` is current amplitude. For
    photovoltaic implants,
    :py:class:`~pulse2percept.stimuli.PhotovoltaicEncoder` instead provides
    normalized optical drive. Scoreboard visualizes the stimulation pattern;
    it does not model photovoltaic conversion or the retinal response.

    .. warning::

        ``rho`` is fixed, so phosphene size does not depend on the pulse:
        doubling amplitude doubles brightness at constant width, unlike
        [Nanduri2012]_. Use
        :py:class:`~pulse2percept.models.retina.BiphasicScoreboardSpatial` for
        pulse-dependent brightness and size.

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.Implant`
        Implant whose electrode geometry is modeled.

        .. versionadded:: 0.11.0

    rho : float or Quantity, optional
        Gaussian spatial decay constant in microns. Larger values produce
        broader phosphenes. The same ``rho`` value applies to all electrodes.

        .. important::

            Electrode-retina distance (``z``) does not directly affect ``rho``.

    xrange : (float, float) or Quantity, optional
        Horizontal visual-field extent in degrees of visual angle. May also be
        passed as retinal extent using physical units such as ``um``. The
        correspondence is resolved through ``visual_field_map``.
    yrange : (float, float) or Quantity, optional
        Vertical visual-field extent in degrees of visual angle. May also be
        passed as retinal extent using physical units such as ``um``. The
        correspondence is resolved through ``visual_field_map``.
    step : float, (float, float), or Quantity, optional
        Grid spacing in degrees of visual angle. A pair specifies separate x
        and y spacing.

        .. versionchanged:: 0.10.0
            Renamed from ``xystep``; ``xystep`` was removed in 0.11.0.

    grid_type : {'rect', 'hex'}, optional
        Sampling lattice used for the visual-field grid.
    thresh_percept : float, optional
        Brightness values below this threshold are set to zero.
    min_current_spread : float, optional
        Fraction of peak Gaussian current spread below which an electrode may
        be skipped at a grid point. Set to 0 to disable the cutoff.
    visual_field_map : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
        Retinotopic map between visual-field and retinal coordinates. Defaults
        to :py:class:`~pulse2percept.topography.retina.Watson2014Map`.
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
    n_threads : int, optional
        Number of OpenMP threads.
    n_jobs : int or None, optional
        Alias for ``n_threads``. ``None`` and -1 use all available CPU cores."""

    #: Also accepts encoded normalized optical drive from a
    #: PhotovoltaicEncoder.
    extra_stimulus_units = (dimensionless,)

    #: Spatial-only use may read dimensionless image or video values as
    #: relative electrode drive, so an encoder is optional.
    _accepts_dimensionless_drive = True

    def __init__(self, implant, *, rho=100, xrange=(-15, 15),
                 yrange=(-15, 15), step=0.25, grid_type='rect',
                 thresh_percept=0, min_current_spread=1e-8,
                 visual_field_map=None,
                 n_gray=None,
                 implant_position=(0, 0), implant_rotation=0,
                 implant_depth=0,
                 location_noise=None, verbose=True, ndim=None,
                 n_threads=None, n_jobs=None):
        super().__init__(
            implant, rho=rho, xrange=xrange, yrange=yrange, step=step,
            grid_type=grid_type, thresh_percept=thresh_percept,
            min_current_spread=min_current_spread,
            visual_field_map=(Watson2014Map() if visual_field_map is None else
                              visual_field_map),
            n_gray=n_gray,
            implant_position=implant_position,
            implant_rotation=implant_rotation,
            implant_depth=implant_depth,
            location_noise=location_noise, verbose=verbose,
            ndim=[2] if ndim is None else ndim,
            **_thread_params(n_threads, n_jobs))

    def get_default_params(self):
        """Return all settable scoreboard parameters."""
        base_params = super(ScoreboardSpatial, self).get_default_params()
        params = {'rho': 100, 'visual_field_map': Watson2014Map()}
        return {**base_params, **params}

    def get_param_units(self):
        """Return units used to store model parameters."""
        return {**super().get_param_units(), 'rho': um}

    def _build(self):
        _warn_rho_vs_pitch(self)

    def _predict_spatial(self, electrode_array, stim):
        """Predict brightness over the spatial grid."""
        _warn_ignores_z(self, electrode_array)
        x_el, y_el, _ = self._electrode_coords(electrode_array, stim)
        return fast_scoreboard(self._stim_values(stim), x_el, y_el,
                               self.grid.ret.x.ravel(),
                               self.grid.ret.y.ravel(),
                               self.rho,
                               self.thresh_percept,
                               self._cutoff_r2(self.rho),
                               0, 0,  # no current boundaries
                               self.n_threads)

    def _predict_tensor(self, waveform, time):
        """Return the flat Torch response to an ``(n_electrodes, T)`` waveform.

        Same Gaussian, cutoff and threshold as ``_predict_spatial``, computed
        as a ``(P, E) @ (E, T)`` product. Geometry is fixed.
        """
        import torch
        if self.n_gray is not None:
            # Quantization is discrete and has no exact gradient:
            raise NotImplementedError("Tensor prediction does not support "
                                      "n_gray; set n_gray=None.")
        electrode_array = self.implant.electrode_array
        _warn_ignores_z(self, electrode_array)
        x_el, y_el, _ = self._electrode_coords(
            electrode_array, None, electrodes=self.implant.electrode_names)
        # float32, as in `fast_scoreboard`:
        dx = self.grid.ret.x.reshape((-1, 1)) - x_el
        dy = self.grid.ret.y.reshape((-1, 1)) - y_el
        r2 = dx * dx + dy * dy
        rho = np.float32(self.rho)
        weights = np.exp(-r2 / (np.float32(2) * rho * rho))
        # Drops pairs beyond the cutoff and unmapped (NaN) grid points:
        weights[~(r2 <= self._cutoff_r2(self.rho))] = 0
        weights = torch.as_tensor(weights, dtype=waveform.dtype,
                                  device=waveform.device)
        resp = weights @ waveform
        resp = torch.where(resp.abs() >= self.thresh_percept, resp, 0.0)
        return self._spatial_response(resp, time, None)


class ScoreboardModel(Model):
    r"""Scoreboard model of [Beyeler2019]_.

    Models each electrode's percept as a circular Gaussian. Use
    :py:class:`~pulse2percept.models.retina.ScoreboardSpatial` to combine this spatial
    model with a temporal model.

    The spatial response is modeled as a Gaussian centered on each electrode:

    .. math::

        I(x, y) =
        \sum_{e \in E}
        a_e
        \exp\left(
            -\frac{(x-x_e)^2 + (y-y_e)^2}{2\rho^2}
        \right),

    where :math:`a_e` is the drive at site :math:`e`, and :math:`\rho`
    controls the spatial spread of activation. Larger values of
    :math:`\rho` produce broader phosphenes.

    For current-driven implants, :math:`a_e` is current amplitude. For
    photovoltaic implants,
    :py:class:`~pulse2percept.stimuli.PhotovoltaicEncoder` instead
    provides normalized optical drive. Scoreboard visualizes the
    stimulation pattern; it does not model photovoltaic conversion or the
    retinal response.

    .. note::

        For this spatial-only model, dimensionless image and video values are
        treated as relative electrode amplitudes, so an implant encoder is not
        required. Physical or pulse-dependent models still require encoded
        stimulation.

    .. warning::

        ``rho`` is fixed, so phosphene size does not depend on the pulse:
        doubling amplitude doubles brightness at constant width, unlike
        [Nanduri2012]_. Use
        :py:class:`~pulse2percept.models.retina.BiphasicScoreboardModel` for
        pulse-dependent brightness and size.

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.Implant`
        Implant whose electrode geometry is modeled.

        .. versionadded:: 0.11.0

    rho : float or Quantity, optional
        Gaussian spatial decay constant in microns. Larger values produce
        broader phosphenes. The same ``rho`` value applies to all electrodes.

        .. important::

            Electrode-retina distance (``z``) does not directly affect ``rho``.

    xrange : (float, float) or Quantity, optional
        Horizontal visual-field extent in degrees of visual angle. May also be
        passed as retinal extent using physical units such as ``um``. The
        correspondence is resolved through ``visual_field_map``.
    yrange : (float, float) or Quantity, optional
        Vertical visual-field extent in degrees of visual angle. May also be
        passed as retinal extent using physical units such as ``um``. The
        correspondence is resolved through ``visual_field_map``.
    step : float, (float, float), or Quantity, optional
        Grid spacing in degrees of visual angle. A pair specifies separate x
        and y spacing.

        .. versionchanged:: 0.10.0
            Renamed from ``xystep``; ``xystep`` was removed in 0.11.0.

    grid_type : {'rect', 'hex'}, optional
        Sampling lattice used for the visual-field grid.
    thresh_percept : float, optional
        Brightness values below this threshold are set to zero.
    min_current_spread : float, optional
        Fraction of peak Gaussian current spread below which an electrode may
        be skipped at a grid point. Set to 0 to disable the cutoff.
    visual_field_map : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
        Retinotopic map between visual-field and retinal coordinates. Defaults
        to :py:class:`~pulse2percept.topography.retina.Watson2014Map`.
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
    n_threads : int, optional
        Number of OpenMP threads.
    n_jobs : int or None, optional
        Alias for ``n_threads``. ``None`` and -1 use all available CPU cores."""

    def __init__(self, implant, *, rho=100, xrange=(-15, 15),
                 yrange=(-15, 15), step=0.25, grid_type='rect',
                 thresh_percept=0, min_current_spread=1e-8,
                 visual_field_map=None,
                 n_gray=None,
                 implant_position=(0, 0), implant_rotation=0,
                 implant_depth=0,
                 location_noise=None, verbose=True, ndim=None,
                 n_threads=None, n_jobs=None):
        super().__init__(
            spatial=ScoreboardSpatial(
                implant, rho=rho, xrange=xrange, yrange=yrange, step=step,
                grid_type=grid_type, thresh_percept=thresh_percept,
                min_current_spread=min_current_spread,
                visual_field_map=visual_field_map,
                n_gray=n_gray,
                implant_position=implant_position,
                implant_rotation=implant_rotation,
                implant_depth=implant_depth,
                location_noise=location_noise, verbose=verbose, ndim=ndim,
                n_threads=n_threads, n_jobs=n_jobs),
            temporal=None)


class AxonMapSpatial(RetinalSpatial):
    r"""Axon map model of [Beyeler2019]_ (spatial module only).

    Models percepts as activation spread along retinal nerve fiber bundle
    trajectories. Use :py:class:`~pulse2percept.models.retina.AxonMapModel` for a
    standalone model.

    The spatial response extends the scoreboard model by allowing activation
    to spread along retinal nerve fiber bundles [Beyeler2019]_. For an axon
    segment, the contribution of electrode :math:`e` is proportional to

    .. math::

        a_e
        \exp\left(
            -\frac{d_e^2}{2\rho^2}
            -\frac{d_{\mathrm{soma}}^2}{2\lambda^2}
        \right),

    where :math:`d_e` is the distance from the segment to electrode :math:`e`,
    and :math:`d_{\mathrm{soma}}` is the path length along the axon from that
    segment to the ganglion cell body. Thus :math:`\rho` controls spread
    away from the axon, whereas :math:`\lambda` controls spread along it.

    .. important::
    
        ``rho`` and ``lam`` vary substantially across patients [Beyeler2019]_.
        The defaults are representative values, not patient-specific estimates.

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.Implant`
        Implant whose electrode geometry and eye are modeled.

        .. versionadded:: 0.11.0

    rho : float or Quantity, optional
        Gaussian spatial decay constant in microns. Larger values produce
        broader phosphenes. The same ``rho`` value applies to all electrodes.

        .. important::

            Electrode-retina distance (``z``) does not directly affect ``rho``.

    lam : float or Quantity, optional
        Gaussian decay constant along the axon between stimulation site and
        soma, in microns. Larger values lengthen the percept.

        .. versionchanged:: 0.10.0
            Renamed from ``axlambda``; ``axlambda`` was removed in 0.11.0.

    xrange : (float, float) or Quantity, optional
        Horizontal visual-field extent in degrees of visual angle. May also be
        passed as retinal extent using physical units such as ``um``. The
        correspondence is resolved through ``visual_field_map``.
    yrange : (float, float) or Quantity, optional
        Vertical visual-field extent in degrees of visual angle. May also be
        passed as retinal extent using physical units such as ``um``. The
        correspondence is resolved through ``visual_field_map``.
    step : float, (float, float), or Quantity, optional
        Grid spacing in degrees of visual angle. A pair specifies separate x
        and y spacing.

        .. versionchanged:: 0.10.0
            Renamed from ``xystep``; ``xystep`` was removed in 0.11.0.

    grid_type : {'rect', 'hex'}, optional
        Sampling lattice used for the visual-field grid.
    thresh_percept : float, optional
        Brightness values below this threshold are set to zero.
    min_current_spread : float, optional
        Fraction of peak Gaussian current spread below which an electrode may
        be skipped at an axon segment. Set to 0 to disable the cutoff.
    visual_field_map : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
        Retinotopic map between visual-field and retinal coordinates. Defaults
        to :py:class:`~pulse2percept.topography.retina.Watson2014Map`.
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
        Radial-coordinate range used to generate each bundle in the
        [Jansonius2009]_ model.
    min_ax_sensitivity : float, optional
        Minimum relative axon sensitivity retained during precomputation.
    meridian_blend : float or Quantity, optional
        Gaussian standard deviation for blending across the horizontal
        meridian, in degrees of visual angle. Set to 0 to disable.

        .. versionadded:: 0.10.0

    axon_pickle : str, optional
        File used to cache generated axon bundles.
    ignore_pickle : bool, optional
        If True, regenerate axon bundles instead of loading ``axon_pickle``.
    verbose : bool, optional
        Whether to print status messages.
    ndim : list of int, optional
        Dimensionalities of ``visual_field_map`` accepted by the model.
    n_threads : int, optional
        Number of OpenMP threads.
    n_jobs : int or None, optional
        Alias for ``n_threads``. ``None`` and -1 use all available CPU cores.

    Notes
    -----
    ``ax_segments_range`` values above 90 are outside the range for which this
    axon-map construction is considered reliable."""

    #: Spatial-only use may read dimensionless image or video values as
    #: relative electrode drive, so an encoder is optional.
    _accepts_dimensionless_drive = True

    #: The Torch core is 3-7x slower than Cython on CPU (dense segment
    #: Gaussians, padded argmax), so public prediction stays on Cython:
    _composite_tensor = False

    def __init__(self, implant, *, rho=300, lam=500, xrange=(-15, 15),
                 yrange=(-15, 15), step=0.25, grid_type='rect',
                 thresh_percept=0, min_current_spread=1e-8,
                 visual_field_map=None,
                 n_gray=None,
                 implant_position=(0, 0), implant_rotation=0,
                 implant_depth=0,
                 location_noise=None, loc_od=(15.5, 1.5), n_axons=1000,
                 axons_range=(-180, 180), n_ax_segments=500,
                 ax_segments_range=(0, 50), min_ax_sensitivity=1e-3,
                 meridian_blend=1, axon_pickle='axons.pickle',
                 ignore_pickle=False, verbose=True, ndim=None,
                 n_threads=None, n_jobs=None):
        super().__init__(
            implant, rho=rho, lam=lam, xrange=xrange, yrange=yrange, step=step,
            grid_type=grid_type, thresh_percept=thresh_percept,
            min_current_spread=min_current_spread,
            visual_field_map=(Watson2014Map() if visual_field_map is None else
                              visual_field_map),
            n_gray=n_gray,
            implant_position=implant_position,
            implant_rotation=implant_rotation,
            implant_depth=implant_depth,
            location_noise=location_noise, loc_od=loc_od, n_axons=n_axons,
            axons_range=axons_range, n_ax_segments=n_ax_segments,
            ax_segments_range=ax_segments_range,
            min_ax_sensitivity=min_ax_sensitivity,
            meridian_blend=meridian_blend, axon_pickle=axon_pickle,
            ignore_pickle=ignore_pickle, verbose=verbose,
            ndim=[2] if ndim is None else ndim,
            **_thread_params(n_threads, n_jobs))
        self.axon_contrib = None
        self.axon_idx_start = None
        self.axon_idx_end = None
        self._built_eye = None

    @property
    def eye(self):
        """Eye used by the axon map.

        Taken from the bound implant, which must be a
        :py:class:`~pulse2percept.implants.retina.RetinalImplant` because the
        optic disc lies nasal to the fovea of the implanted eye.

        .. versionchanged:: 0.11.0
            ``eye`` is no longer a separate model parameter."""
        if not hasattr(self.implant, 'eye'):
            raise TypeError(
                f"{type(self).__name__} mirrors the optic disc onto the "
                f"implanted eye, so its implant must carry retinal "
                f"laterality, which {type(self.implant).__name__} does not. "
                f"Wrap a custom array in "
                f"pulse2percept.implants.retina.RetinalImplant, e.g. "
                f"RetinalImplant(ElectrodeGrid(...), eye='right').")
        return self.implant.eye

    @property
    def is_built(self):
        """Return whether the axon map matches the implant's current eye."""
        return super().is_built and self._built_eye == self.eye

    def get_default_params(self):
        base_params = super(AxonMapSpatial, self).get_default_params()
        params = {
            'rho': 300,
            'lam': 500,
            'loc_od': (15.5, 1.5),
            'n_axons': 1000,
            'axons_range': (-180, 180),
            'n_ax_segments': 500,
            'ax_segments_range': (0, 50),
            'min_ax_sensitivity': 1e-3,
            'meridian_blend': 1,
            'axon_pickle': 'axons.pickle',
            'ignore_pickle': False,
            'visual_field_map': Watson2014Map()
        }
        return {**base_params, **params}

    def get_param_units(self):
        """Return units used to store model parameters."""
        # ``axons_range`` is an angle; ``ax_segments_range`` is the
        # Jansonius radial coordinate and has no p2p unit declaration.
        return {**super().get_param_units(), 'rho': um, 'lam': um,
                'loc_od': dva, 'meridian_blend': dva, 'axons_range': deg}

    def _jansonius2009(self, phi0, beta_sup=-1.9, beta_inf=0.5, eye='right'):
        """Generate one nerve fiber bundle using [Jansonius2009]_.

        Parameters
        ----------
        phi0 : float
            Initial bundle angle in degrees, in [-180, 180].
        beta_sup : float, optional
            Superior-retina curvature parameter (Eq. 5 in [Jansonius2009]_).
        beta_inf : float, optional
            Inferior-retina curvature parameter (Eq. 6 in [Jansonius2009]_).
        eye : {'right', 'left'}, optional
            Eye for which to generate the bundle.

        Returns
        -------
        ax_pos : (N, 2) ndarray
            Bundle coordinates in degrees of visual angle, ordered away from
            the optic disc.

        Notes
        -----
        [Jansonius2009]_ did not include bundles with ``phi0`` in [-60, 60]
        degrees."""
        loc_od = self.loc_od
        if eye.lower() not in ['left', 'right']:
            e_s = f"Unknown eye string '{eye}': Choose from 'left', 'right'."
            raise ValueError(e_s)
        if eye.lower() == 'left':
            # Jansonius is parameterized for a right eye; mirror left eyes.
            loc_od = (-loc_od[0], loc_od[1])
        if np.abs(phi0) > 180.0:
            raise ValueError('phi0 must be within [-180, 180].')
        if self.n_ax_segments < 1:
            raise ValueError('Number of radial sampling points must be >= 1.')
        if np.any(np.array(self.ax_segments_range) < 0):
            raise ValueError('ax_segments_range cannot be negative.')
        if self.ax_segments_range[0] > self.ax_segments_range[1]:
            raise ValueError('Lower bound on rho cannot be larger than the '
                             ' upper bound.')
        is_superior = phi0 > 0
        rho = np.linspace(*self.ax_segments_range, num=self.n_ax_segments,
                          dtype=np.float32)
        xprime, yprime = fast_jansonius(rho, phi0, beta_sup, beta_inf)
        # Truncate at the first horizontal-meridian crossing:
        if is_superior:
            idx = np.where(yprime < 0)[0]
        else:
            idx = np.where(yprime > 0)[0]
        if idx.size:
            xprime = xprime[:idx[0]]
            yprime = yprime[:idx[0]]
        # Shift the origin from optic disc to fovea:
        xmodel = xprime + loc_od[0]
        ymodel = yprime
        if loc_od[0] > 0:
            # Use Appendix A for a positive optic-disc x coordinate:
            idx = xprime > -loc_od[0]
        else:
            # Mirror the correction for a negative x coordinate:
            idx = xprime < -loc_od[0]
        ymodel[idx] = yprime[idx] + loc_od[1] * (xmodel[idx] / loc_od[0]) ** 2
        # Mirror back to the left eye:
        if eye.lower() == 'left':
            xmodel *= -1
        return np.vstack((xmodel, ymodel)).astype(np.float32).T

    def grow_axon_bundles(self, n_bundles=None, prune=True):
        """Generate nerve fiber bundles from the [Jansonius2009]_ model.

        Parameters
        ----------
        n_bundles : int, optional
            Number of bundles. Defaults to ``n_axons``.
        prune : bool, optional
            If True, discard bundles and segments outside the simulated visual
            field.

        Returns
        -------
        bundles : list of (N, 2) ndarrays
            Bundle coordinates on the retina in microns."""
        if n_bundles is None:
            n_bundles = self.n_axons
        # Sample initial bundle angles uniformly:
        phi = np.linspace(*self.axons_range, num=n_bundles)
        bundles = [self._jansonius2009(p, eye=self.eye) for p in phi]
        bundles = list(filter(lambda x: len(x) > 0, bundles))
        if prune:
            # Prune to the simulated visual field:
            xmin, xmax = self.xrange
            ymin, ymax = self.yrange
            bundles = list(filter(lambda x: (np.max(x[:, 0]) >= xmin and
                                             np.min(x[:, 0]) <= xmax and
                                             np.max(x[:, 1]) >= ymin and
                                             np.min(x[:, 1]) <= ymax),
                                  bundles))
            bundles = list(filter(lambda x: len(x) > 10, bundles))
        # Convert visual-field coordinates to retinal microns:
        bundles = [np.array(self.visual_field_map.dva_to_ret(b[:, 0], b[:,
                                                                        1])).T
                   for b in bundles]
        return bundles

    def find_closest_axon(self, bundles, xret=None, yret=None,
                          return_index=False, return_segment=False):
        """Find the nearest nerve fiber bundle for one or more retinal points.

        Parameters
        ----------
        bundles : list of (N, 2) ndarrays
            Bundle coordinates in microns.
        xret, yret : array_like, optional
            Retinal coordinates in microns. Defaults to the model grid.
        return_index : bool, optional
            Also return the index of the nearest bundle.
        return_segment : bool, optional
            Also return the index of the nearest segment within that bundle.

        Returns
        -------
        axon : ndarray or list of ndarrays
            Nearest bundle for each query point.
        idx_axon : int or ndarray, optional
            Returned when ``return_index`` is True.
        idx_segment : int or ndarray, optional
            Returned when ``return_segment`` is True."""
        if len(bundles) <= 0:
            raise ValueError("bundles must have length greater than zero")
        if xret is None:
            xret = self.grid.ret.x
        if yret is None:
            yret = self.grid.ret.y
        xret = np.asarray(xret, dtype=np.float32)
        yret = np.asarray(yret, dtype=np.float32)
        # ``boff`` maps flat segment indices back to bundles without storing
        # a bundle ID for every segment:
        boff = np.concatenate(([0], np.cumsum([len(ax) for ax in bundles])))
        flat_bundles = np.concatenate(bundles)
        kdtree = cKDTree(flat_bundles, leafsize=60)
        query = np.stack((xret.ravel(), yret.ravel()), axis=1)
        _, closest_seg = kdtree.query(query, workers=max(1, self.n_threads))

        closest_idx = (np.searchsorted(boff, closest_seg, side='right') -
                       1).astype(np.uint32)
        idx_segment = closest_seg - boff[closest_idx]
        if len(closest_idx) == 1:
            closest_idx = closest_idx[0]
            idx_segment = idx_segment[0]
            closest_axon = bundles[closest_idx]
        else:
            closest_axon = [bundles[n] for n in closest_idx]
        if return_index and return_segment:
            return closest_axon, closest_idx, idx_segment
        if return_segment:
            return closest_axon, idx_segment
        if return_index:
            return closest_axon, closest_idx
        return closest_axon

    def calc_axon_sensitivity(self, bundles):
        """Calculate sensitivity along the axon associated with each grid point.

        ``bundles[i]`` is assumed to pass through grid point ``i``. Segments
        beyond the soma are removed, and sensitivity decays with distance from
        the soma according to ``lam``.

        Parameters
        ----------
        bundles : list of (N, 2) ndarrays
            One retinal bundle per grid point, in microns.

        Returns
        -------
        axon_contrib : list of (N, 3) ndarrays
            Retinal x, y, and relative sensitivity for each retained segment."""
        contrib, starts = self._calc_axon_sensitivity_flat(*_flatten_bundles(
            bundles))
        return [contrib[lo:hi] for lo, hi in zip(starts[:-1], starts[1:])]

    def _calc_axon_sensitivity_flat(self, flat, boff, bundle_id, seg=None):
        """Vectorized core of :py:meth:`calc_axon_sensitivity`.

        Parameters
        ----------
        flat : (M, 2) ndarray
            Distinct bundles concatenated.
        boff : (n_bundles + 1,) ndarray
            Bundle offsets in ``flat``.
        bundle_id : (n_points,) ndarray
            Bundle index associated with each grid point.
        seg : (n_points,) ndarray, optional
            Absolute index in ``flat`` of the segment nearest each grid point.
            Computed internally if omitted.

        Returns
        -------
        contrib : (N, 3) ndarray
            Concatenated retinal x, y, and sensitivity values.
        starts : (n_points + 1,) ndarray
            Axon offsets in ``contrib``."""
        lam = self.lam
        xyret = np.column_stack((self.grid.ret.x.ravel(),
                                 self.grid.ret.y.ravel()))
        blens = np.diff(boff)
        # Accumulate arc length in float64; float32 loses appreciable
        # precision over hundreds of segments.
        out_dtype = np.promote_types(flat.dtype, np.float32)
        flat = flat.astype(np.float64, copy=False)
        xyret = xyret.astype(np.float64, copy=False)

        # Arc length from each bundle's first segment, reset at bundle seams:
        step = np.hypot(*(flat[1:] - flat[:-1]).T)
        step[boff[1:-1] - 1] = 0.0  # do not cross bundle seams
        arc = np.empty(len(flat))
        arc[0] = 0.0
        np.cumsum(step, out=arc[1:])
        arc -= np.repeat(arc[boff[:-1]], blens)

        base = boff[bundle_id]
        if seg is None:
            # Flatten all (grid point, segment) pairs for each selected bundle:
            lens = blens[bundle_id]
            pair_off = np.concatenate(([0], np.cumsum(lens)))
            within = np.arange(pair_off[-1]) - np.repeat(pair_off[:-1], lens)
            pairs = np.repeat(base, lens) + within
            d2 = ((flat[pairs, 0] - np.repeat(xyret[:, 0], lens)) ** 2 +
                  (flat[pairs, 1] - np.repeat(xyret[:, 1], lens)) ** 2)
            # Match ``np.argmin`` tie-breaking by choosing the lower index:
            closest = np.repeat(np.minimum.reduceat(d2, pair_off[:-1]), lens)
            cand = np.where(d2 <= closest, within, np.iinfo(np.intp).max)
            seg = base + np.minimum.reduceat(cand, pair_off[:-1])
        else:
            seg = np.asarray(seg, dtype=np.intp)

        # Distance from the soma to segment ``k`` is
        # ``d0 + arc[seg] - arc[k]``. Keep only segments above the sensitivity
        # threshold.
        d0 = np.hypot(flat[seg, 0] - xyret[:, 0], flat[seg, 1] - xyret[:, 1])
        max_d2 = -2.0 * lam ** 2 * np.log(self.min_ax_sensitivity)
        if max_d2 <= 0:
            # No segment meets the sensitivity threshold:
            n_keep = np.zeros(len(bundle_id), dtype=np.intp)
        else:
            # Retained segments form a contiguous run ending at ``seg``.
            # Offset each bundle's arc lengths to make one monotonic array for
            # the vectorized ``searchsorted``.
            span = arc[boff[1:] - 1].max() + 1.0
            offset = bundle_id * span
            lo = np.searchsorted(arc + np.repeat(np.arange(len(blens)) * span,
                                                 blens),
                                 arc[seg] + d0 - np.sqrt(max_d2) + offset,
                                 side='right')
            n_keep = np.maximum(seg - np.maximum(lo, base) + 1, 0)

        starts = np.concatenate(([0], np.cumsum(n_keep)))
        # Gather retained segments backward from the soma:
        gather = (np.repeat(seg, n_keep) -
                  (np.arange(starts[-1]) - np.repeat(starts[:-1], n_keep)))
        dist = np.repeat(d0 + arc[seg], n_keep) - arc[gather]
        contrib = np.empty((starts[-1], 3), dtype=out_dtype)
        contrib[:, :2] = flat[gather]
        contrib[:, 2] = np.exp(-dist ** 2 / (2.0 * lam ** 2))
        return contrib, starts

    def calc_bundle_tangent(self, xc, yc):
        """Calculate the local nerve fiber bundle orientation.

        Parameters
        ----------
        xc, yc : float
            Retinal coordinates in microns.

        Returns
        -------
        tangent : float
            Bundle orientation in radians, restricted to [-pi/2, pi/2]."""
        if isinstance(xc, (list, np.ndarray)):
            raise TypeError("xc must be a scalar")
        if isinstance(yc, (list, np.ndarray)):
            raise TypeError("yc must be a scalar")
        bundles = self.grow_axon_bundles()
        bundle = self.find_closest_axon(bundles, xret=xc, yret=yc)
        idx = np.argmin((bundle[:, 0] - xc) ** 2 + (bundle[:, 1] - yc) ** 2)
        # Use a one-sided difference at bundle endpoints:
        if idx == 0:
            dx = bundle[1, :] - bundle[0, :]
        elif idx == bundle.shape[0] - 1:
            dx = bundle[-1, :] - bundle[-2, :]
        else:
            dx = (bundle[idx + 1, :] - bundle[idx - 1, :]) / 2
        dx[1] *= -1
        tangent = np.arctan2(*dx[::-1])
        # Orientation is axial; wrap to [-pi/2, pi/2]:
        if tangent < np.deg2rad(-90):
            tangent += np.deg2rad(180)
        if tangent > np.deg2rad(90):
            tangent -= np.deg2rad(180)
        return tangent
    

    def calc_bundle_tangent_fast(self, xc, yc, bundles=None):
        """Calculate local bundle orientation for multiple retinal points.

        Reuses a KD-tree search over ``bundles`` and is intended for vectorized
        queries.

        Parameters
        ----------
        xc, yc : array_like
            Retinal coordinates in microns.
        bundles : list of (N, 2) ndarrays, optional
            Precomputed bundles. Generated if omitted.

        Returns
        -------
        tangent : ndarray
            Bundle orientations in radians, shaped like ``xc``."""

        if bundles is None:
            bundles = self.grow_axon_bundles()
        xc = np.asarray(xc, dtype=np.float32)
        yc = np.asarray(yc, dtype=np.float32)
        # Map flattened segments to bundle IDs:
        axon_idx = [[idx] * len(ax) for idx, ax in enumerate(bundles)]
        axon_idx = [item for sublist in axon_idx for item in sublist]
        axon_idx = np.array(axon_idx, dtype=np.uint32)
        flat_bundles = np.concatenate(bundles)
        kdtree = cKDTree(flat_bundles, leafsize=60)
        query = np.stack((xc.ravel(), yc.ravel()), axis=1)
        _, closest_seg = kdtree.query(query)
        segs = axon_idx[closest_seg]
        prev_segs = axon_idx[np.where(closest_seg > 0, closest_seg, 1) - 1]
        next_segs = axon_idx[np.where(closest_seg < len(axon_idx)-2, closest_seg, len(axon_idx)-2) + 1]

        offset_l = np.where(prev_segs == segs, -1, 0)
        offset_r = np.where(next_segs == segs, 1, 0)
        dx = flat_bundles[np.minimum(closest_seg + offset_r, len(flat_bundles)-1)] - flat_bundles[np.maximum(closest_seg + offset_l, 0)]

        dx[:, 1] *= -1
        tangent = np.arctan2(dx[:, 1], dx[:, 0])

        # Orientation is axial; wrap to [-pi/2, pi/2]:
        tangent = np.where(tangent < -np.pi/2, tangent+np.pi, tangent)
        tangent = np.where(tangent > np.pi/2, tangent - np.pi, tangent)
        return tangent.reshape(xc.shape)


    def _warn_placement(self):
        """Warn when the epiretinal axon-map mechanism does not match placement."""
        placement = self.implant.placement
        if placement is None or placement == 'epiretinal':
            return
        warnings.warn(
            f"{type(self).__name__} predicts elongated percepts because an "
            f"epiretinal array stimulates passing nerve fiber bundles. This "
            f"implant is {placement}, where that mechanism does not apply, so "
            f"the streaks below are an artifact of the model rather than a "
            f"prediction about the device. A placement-appropriate "
            f"local-response scoreboard model is a safer phenomenological "
            f"starting point.")

    def _correct_loc_od(self):
        """Place the optic disc on the nasal side of the implanted eye."""
        sign = -1 if self.eye == 'left' else 1
        self.loc_od = (sign * np.abs(self.loc_od[0]), self.loc_od[1])

    def _build(self):
        if self.lam < 10:
            raise ValueError('"lam" < 10 is not supported by this model. '
                             'Consider using ScoreboardModel instead.')
        # Check the eye first so a non-retinal implant fails before warnings:
        self._built_eye = self.eye
        self._warn_placement()
        _warn_rho_vs_pitch(self)
        self._correct_loc_od()
        # Reuse the cache only when format and build parameters match:
        need_axons = False
        cached = None
        if self.ignore_pickle:
            need_axons = True
        else:
            if os.path.isfile(self.axon_pickle):
                params, cached = pickle.load(open(self.axon_pickle, 'rb'))
                # Cache layouts from older versions are regenerated rather than
                # interpreted; pre-0.10 caches also store ``xystep`` metadata.
                if not _is_axon_cache(cached):
                    need_axons = True
                else:
                    for key, value in params.items():
                        if (not hasattr(self, key) or
                                not np.allclose(getattr(self, key), value)):
                            need_axons = True
                            break
            else:
                need_axons = True
        # Generate geometry if the cache cannot be reused:
        if need_axons:
            bundles = self.grow_axon_bundles()
            _, bundle_id, idx_segment = self.find_closest_axon(
                bundles, return_index=True, return_segment=True)
            bundle_id = np.atleast_1d(bundle_id).astype(np.intp)
            idx_segment = np.atleast_1d(idx_segment).astype(np.intp)
            # Keep only bundles selected by at least one grid point:
            used, bundle_id = np.unique(bundle_id, return_inverse=True)
            bundles = [bundles[idx] for idx in used]
            bundle_id = np.ravel(bundle_id).astype(np.intp)
        else:
            _, bundles, bundle_id, idx_segment = cached
        # The Cython kernel consumes concatenated axons plus slice offsets:
        flat, boff, _ = _flatten_bundles(bundles)
        axon_contrib, starts = self._calc_axon_sensitivity_flat(
            flat, boff, bundle_id, seg=boff[bundle_id] + idx_segment)
        self.axon_contrib = np.ascontiguousarray(axon_contrib,
                                                 dtype=np.float32)
        self.axon_idx_start = starts[:-1]
        self.axon_idx_end = starts[1:]
        if need_axons:
            # Cache geometry inputs and generated bundles:
            params = {'loc_od': self.loc_od,
                      'n_axons': self.n_axons, 'axons_range': self.axons_range,
                      'xrange': self.xrange, 'yrange': self.yrange,
                      'step': self.step, 'n_ax_segments': self.n_ax_segments,
                      'ax_segments_range': self.ax_segments_range}
            pickle.dump((params, (_AXON_CACHE_VERSION, bundles, bundle_id,
                                  idx_segment)),
                        open(self.axon_pickle, 'wb'))

    def _predict_spatial(self, electrode_array, stim):
        """Predict brightness over the spatial grid."""
        _warn_ignores_z(self, electrode_array)
        x_el, y_el, _ = self._electrode_coords(electrode_array, stim)
        return fast_axon_map(self._stim_values(stim), x_el, y_el,
                             self.axon_contrib,
                             self.axon_idx_start.astype(np.uint32),
                             self.axon_idx_end.astype(np.uint32),
                             self.rho,
                             self.thresh_percept,
                             self._cutoff_r2(self.rho),
                             self.n_threads)

    def _predict_tensor(self, waveform, time):
        """Return the flat Torch response to an ``(n_electrodes, T)`` waveform.

        Same Gaussian, cutoff, signed brightest-segment selection, thresholds
        and meridian blending as ``predict_percept``. Whole axons are processed
        in blocks of about ``_AXON_BLOCK_BYTES``. Geometry is fixed.

        Until backward, autograd keeps each block's Gaussian (segments x
        electrodes x ``itemsize`` bytes in total) and one int64 index per
        output value.
        """
        import torch
        if self.n_gray is not None:
            # Quantization is discrete and has no exact gradient:
            raise NotImplementedError("Tensor prediction does not support "
                                      "n_gray; set n_gray=None.")
        electrode_array = self.implant.electrode_array
        _warn_ignores_z(self, electrode_array)
        x_el, y_el, _ = self._electrode_coords(
            electrode_array, None, electrodes=self.implant.electrode_names)
        # Electrodes sorted by x, as in `fast_axon_map`:
        order = np.argsort(x_el, kind='stable')
        x_el, y_el = x_el[order], y_el[order]
        waveform = waveform[torch.as_tensor(order, device=waveform.device)]
        rho = np.float32(self.rho)
        cutoff_r2 = self._cutoff_r2(self.rho)
        start, end = self.axon_idx_start, self.axon_idx_end
        n_el, n_time = x_el.size, waveform.shape[1]
        device = waveform.device
        t_idx = torch.arange(n_time, device=device)
        blocks = []
        for p0, p1 in _axon_blocks(end - start, n_el, n_time,
                                   waveform.element_size(),
                                   _AXON_BLOCK_BYTES):
            lo, hi = start[p0], end[p1 - 1]
            seg = self.axon_contrib[lo:hi]
            # float32, as in `fast_axon_map`. The trailing zero row is the
            # response of an axon without segments:
            gauss = np.zeros((hi - lo + 1, n_el), dtype=np.float32)
            dx = seg[:, :1] - x_el
            dy = seg[:, 1:2] - y_el
            r2 = dx * dx + dy * dy
            gauss[:-1] = seg[:, 2:] * np.exp(-r2 / (np.float32(2) * rho * rho))
            # Drops pairs beyond the cutoff and segments without a location:
            gauss[:-1][~(r2 <= cutoff_r2)] = 0
            seg_resp = torch.as_tensor(gauss, dtype=waveform.dtype,
                                       device=device) @ waveform
            counts = end[p0:p1] - start[p0:p1]
            first = start[p0:p1] - lo
            with torch.no_grad():
                # |response| as (pixels, longest axon, T) in axon order;
                # padding is 0:
                pix = np.repeat(np.arange(p1 - p0), counts)
                pos = np.arange(hi - lo) - np.repeat(first, counts)
                packed = seg_resp.new_zeros((p1 - p0, max(counts.max(), 1),
                                             n_time))
                idx = (torch.as_tensor(pix, device=device),
                       torch.as_tensor(pos, device=device))
                packed[idx] = seg_resp[:-1].abs()
                # First largest |response|, matching the strict `>` update in
                # Cython:
                best = packed.argmax(dim=1)
            rows = torch.as_tensor(first, device=device)[:, None] + best
            empty = torch.as_tensor(counts == 0, device=device)[:, None]
            rows = rows.masked_fill(empty, int(hi - lo))
            # Indexing keeps only `rows` for backward, not the segment
            # response (gather would keep it):
            block = seg_resp[rows, t_idx]
            blocks.append(torch.where(block.abs() >= self.thresh_percept,
                                      block, 0.0))
        resp = self._postprocess_spatial(torch.cat(blocks))
        return self._spatial_response(resp, time, None)

    def _postprocess_spatial(self, resp):
        """Blend the response across the horizontal meridian."""
        blended = _blend_meridian(resp, self.grid, 'horizontal',
                                  self.meridian_blend)
        if blended is resp:
            # Preserve the unblended response bit-for-bit.
            return resp
        # Reapply the percept threshold after blending:
        if _is_tensor(blended):
            import torch
            return torch.where(blended.abs() >= self.thresh_percept, blended,
                               0.0)
        blended[np.abs(blended) < self.thresh_percept] = 0
        return blended

    def plot(self, use_dva=False, style='hull', annotate=True, autoscale=True,
             ax=None, figsize=None, show_implant=False):
        """Plot the axon map.

        Parameters
        ----------
        use_dva : bool, optional
            Plot in degrees of visual angle instead of retinal microns.
        style : {'hull', 'scatter', 'cell'}, optional
            Grid plotting style.
        annotate : bool, optional
            Label retinal quadrants.
        autoscale : bool, optional
            Set axis limits to include the simulated region.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Defaults to the current axes.
        figsize : (float, float), optional
            Figure size in inches.
        show_implant : bool, optional
            Draw the implant at its model-side placement. Requires
            ``use_dva=False``.

            .. versionadded:: 0.11.0

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes containing the plot."""
        if show_implant and use_dva:
            raise NotImplementedError(
                "show_implant=True is only supported in tissue coordinates; "
                "a nonlinear visual_field_map does not transform device "
                "geometry rigidly.")
        if ax is None:
            ax = plt.gca()
        if figsize is not None:
            ax.figure.set_size_inches(figsize)
        ax.set_facecolor('white')
        ax.set_aspect('equal')

        self._correct_loc_od()

        axon_bundles = self.grow_axon_bundles(n_bundles=100, prune=False)

        if use_dva:
            units = 'degrees of visual angle'
            # Plot at least +/-18 dva:
            xmin = min(np.floor(self.xrange[0] / 3) * 3, -18)
            xmax = max(np.ceil(self.xrange[1] / 3) * 3, 18)
            ymin = min(np.floor(self.yrange[0] / 3) * 3, -18)
            ymax = max(np.ceil(self.yrange[1] / 3) * 3, 18)
            od_xy = self.loc_od
            od_w = 6.44
            od_h = 6.85
            # Convert bundles to dva:
            axon_bundles = [
                np.array(self.visual_field_map.ret_to_dva(bundle[:, 0],
                                                          bundle[:, 1])).T
                for bundle in axon_bundles]
            labels = ['upper', 'lower', 'left', 'right']
        else:
            units = 'microns'
            # Plot at least +/-5 mm, rounded to whole millimeters:
            xmin, ymin = self.visual_field_map.dva_to_ret(self.xrange[0],
                                                          self.yrange[0])
            xmin = min(np.floor(xmin / UM_PER_MM) * UM_PER_MM, -5000)
            ymin = min(np.floor(ymin / UM_PER_MM) * UM_PER_MM, -5000)
            xmax, ymax = self.visual_field_map.dva_to_ret(self.xrange[1],
                                                          self.yrange[1])
            xmax = max(np.ceil(xmax / UM_PER_MM) * UM_PER_MM, 5000)
            ymax = max(np.ceil(ymax / UM_PER_MM) * UM_PER_MM, 5000)
            od_xy = self.visual_field_map.dva_to_ret(*self.loc_od)
            od_w = 1770
            od_h = 1880
            if self.eye == 'right':
                labels = ['superior', 'inferior', 'temporal', 'nasal']
            else:
                labels = ['superior', 'inferior', 'nasal', 'temporal']

        for bundle in axon_bundles:
            # Break paths outside the plotting window:
            x_idx = np.logical_or(bundle[:, 0] < xmin, bundle[:, 0] > xmax)
            bundle[x_idx, 0] = np.nan
            y_idx = np.logical_or(bundle[:, 1] < ymin, bundle[:, 1] > ymax)
            bundle[y_idx, 1] = np.nan
            ax.plot(bundle[:, 0], bundle[:, 1], c=(0.6, 0.6, 0.6),
                    linewidth=2, zorder=ZORDER['background'])
        # Draw the optic disc:
        ax.add_patch(Ellipse(od_xy, width=od_w, height=od_h, alpha=1,
                             color='white', zorder=ZORDER['background'] + 1))
        if self.is_built:
            self.grid.plot(ax=ax, style=style, zorder=ZORDER['background'] + 2,
                           use_dva=use_dva)
        if show_implant:
            # Keep the anatomical plot window; the implant does not rescale it:
            _draw_placed_implant(self, ax, autoscale=False)
        ax.set_xlabel(f'x ({units})')
        ax.set_ylabel(f'y ({units})')
        if autoscale:
            ax.axis((xmin, xmax, ymin, ymax))
        if annotate:
            ann = ax.inset_axes([0.05, 0.05, 0.2, 0.2],
                                zorder=ZORDER['annotate'])
            ann.annotate('', (0.5, 1), (0.5, 0),
                         arrowprops={'arrowstyle': '<->'})
            ann.annotate('', (1, 0.5), (0, 0.5),
                         arrowprops={'arrowstyle': '<->'})
            positions = [(0.5, 1), (0.5, 0), (0, 0.5), (1, 0.5)]
            valign = ['bottom', 'top', 'center', 'center']
            rots = [0, 0, 90, -90]
            for label, pos, va, rot in zip(labels, positions, valign, rots):
                ann.annotate(label, pos, ha='center', va=va, rotation=rot)
            ann.axis('off')
            ann.set_xticks([])
            ann.set_yticks([])
        return ax


class AxonMapModel(Model):
    r"""Axon map model of [Beyeler2019]_.

    Models percepts as activation spread along retinal nerve fiber bundle
    trajectories. Use :py:class:`~pulse2percept.models.retina.AxonMapSpatial` to
    combine this spatial model with a temporal model.

    The spatial response extends the scoreboard model by allowing activation
    to spread along retinal nerve fiber bundles [Beyeler2019]_. For an axon
    segment, the contribution of electrode :math:`e` is proportional to

    .. math::

        a_e
        \exp\left(
            -\frac{d_e^2}{2\rho^2}
            -\frac{d_{\mathrm{soma}}^2}{2\lambda^2}
        \right),

    where :math:`d_e` is the distance from the segment to electrode :math:`e`,
    and :math:`d_{\mathrm{soma}}` is the path length along the axon from that
    segment to the ganglion cell body. Thus :math:`\rho` controls spread
    away from the axon, whereas :math:`\lambda` controls spread along it.
    Nerve fiber bundle geometry is derived from [Jansonius2009]_.

    .. note::

        Dimensionless image or video values are read as relative electrode
        drive, so an implant ``encoder`` is optional for spatial phosphene
        patterns. An encoder is required for physical amplitude, pulse timing,
        safety constraints, or temporal and pulse-dependent models.

    .. important::

        ``rho`` and ``lam`` vary substantially across patients [Beyeler2019]_.
        The defaults are representative values, not patient-specific estimates.

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.Implant`
        Implant whose electrode geometry and eye are modeled.

        .. versionadded:: 0.11.0

    rho : float or Quantity, optional
        Gaussian spatial decay constant in microns. Larger values produce
        broader phosphenes. The same ``rho`` value applies to all electrodes.

        .. important::

            Electrode-retina distance (``z``) does not directly affect ``rho``.

    lam : float or Quantity, optional
        Gaussian decay constant along the axon between stimulation site and
        soma, in microns. Larger values lengthen the percept.

        .. versionchanged:: 0.10.0
            Renamed from ``axlambda``; ``axlambda`` was removed in 0.11.0.

    xrange : (float, float) or Quantity, optional
        Horizontal visual-field extent in degrees of visual angle. May also be
        passed as retinal extent using physical units such as ``um``. The
        correspondence is resolved through ``visual_field_map``.
    yrange : (float, float) or Quantity, optional
        Vertical visual-field extent in degrees of visual angle. May also be
        passed as retinal extent using physical units such as ``um``. The
        correspondence is resolved through ``visual_field_map``.
    step : float, (float, float), or Quantity, optional
        Grid spacing in degrees of visual angle. A pair specifies separate x
        and y spacing.

        .. versionchanged:: 0.10.0
            Renamed from ``xystep``; ``xystep`` was removed in 0.11.0.

    grid_type : {'rect', 'hex'}, optional
        Sampling lattice used for the visual-field grid.
    thresh_percept : float, optional
        Brightness values below this threshold are set to zero.
    min_current_spread : float, optional
        Fraction of peak Gaussian current spread below which an electrode may
        be skipped at an axon segment. Set to 0 to disable the cutoff.
    visual_field_map : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
        Retinotopic map between visual-field and retinal coordinates. Defaults
        to :py:class:`~pulse2percept.topography.retina.Watson2014Map`.
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
        Radial-coordinate range used to generate each bundle in the
        [Jansonius2009]_ model.
    min_ax_sensitivity : float, optional
        Minimum relative axon sensitivity retained during precomputation.
    meridian_blend : float or Quantity, optional
        Gaussian standard deviation for blending across the horizontal
        meridian, in degrees of visual angle. Set to 0 to disable.

        .. versionadded:: 0.10.0

    axon_pickle : str, optional
        File used to cache generated axon bundles.
    ignore_pickle : bool, optional
        If True, regenerate axon bundles instead of loading ``axon_pickle``.
    verbose : bool, optional
        Whether to print status messages.
    ndim : list of int, optional
        Dimensionalities of ``visual_field_map`` accepted by the model.
    n_threads : int, optional
        Number of OpenMP threads.
    n_jobs : int or None, optional
        Alias for ``n_threads``. ``None`` and -1 use all available CPU cores.

    Notes
    -----
    ``ax_segments_range`` values above 90 are outside the range for which this
    axon-map construction is considered reliable."""

    def __init__(self, implant, *, rho=300, lam=500, xrange=(-15, 15),
                 yrange=(-15, 15), step=0.25, grid_type='rect',
                 thresh_percept=0, min_current_spread=1e-8,
                 visual_field_map=None,
                 n_gray=None,
                 implant_position=(0, 0), implant_rotation=0,
                 implant_depth=0,
                 location_noise=None, loc_od=(15.5, 1.5), n_axons=1000,
                 axons_range=(-180, 180), n_ax_segments=500,
                 ax_segments_range=(0, 50), min_ax_sensitivity=1e-3,
                 meridian_blend=1, axon_pickle='axons.pickle',
                 ignore_pickle=False, verbose=True, ndim=None,
                 n_threads=None, n_jobs=None):
        super().__init__(
            spatial=AxonMapSpatial(
                implant, rho=rho, lam=lam, xrange=xrange, yrange=yrange,
                step=step, grid_type=grid_type,
                thresh_percept=thresh_percept,
                min_current_spread=min_current_spread,
                visual_field_map=visual_field_map,
                n_gray=n_gray,
                implant_position=implant_position,
                implant_rotation=implant_rotation,
                implant_depth=implant_depth,
                location_noise=location_noise, loc_od=loc_od, n_axons=n_axons,
                axons_range=axons_range, n_ax_segments=n_ax_segments,
                ax_segments_range=ax_segments_range,
                min_ax_sensitivity=min_ax_sensitivity,
                meridian_blend=meridian_blend, axon_pickle=axon_pickle,
                ignore_pickle=ignore_pickle, verbose=verbose, ndim=ndim,
                n_threads=n_threads, n_jobs=n_jobs),
            temporal=None)

