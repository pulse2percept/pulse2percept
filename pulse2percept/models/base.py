""":py:class:`~pulse2percept.models.BaseModel`,
   :py:class:`~pulse2percept.models.Model`,
   :py:class:`~pulse2percept.models.SpatialModel`,
   :py:class:`~pulse2percept.models.TemporalModel`"""
import inspect
import warnings
from abc import ABCMeta, abstractmethod
from copy import deepcopy, copy
import numpy as np
import multiprocessing
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from ..implants import ProsthesisSystem
from ..stimuli import ImageStimulus, Stimulus, VideoStimulus
from ..stimuli.base import _describe_unit, _has_time_axis
from ..percepts import Percept
from ..topography import Curcio1990Map, Grid2D, RetinalMap
from ..units import (DimensionMismatchError, Quantity, Unit, as_value, dva, ms,
                     um, uA)
from ..vision import Scene
from ..utils import (PrettyPrint, FreezeError, Frozen, Parametrized,
                     warn_deprecated_params, rename_deprecated_params)
from ..utils.base import _is_constructing
from ..utils.constants import ZORDER


def _n_jobs_alias():
    """Build ``n_jobs`` as an alias for ``n_threads``.

    ``None`` and ``-1`` select all available CPU cores.
    """
    def getter(self):
        return self.n_threads

    def setter(self, val):
        if val is None:
            val = multiprocessing.cpu_count()
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise ValueError(f"n_jobs must be an integer, None, or -1 (all "
                             f"cores), not {val!r}.")
        if val == -1:
            val = multiprocessing.cpu_count()
        if val < 1:
            raise ValueError(f"n_jobs must be >= 1, or -1 for all cores, "
                             f"not {val}.")
        self.n_threads = int(val)

    return property(getter, setter,
                    doc="Number of OpenMP threads to use during "
                        "parallelization. An alias for ``n_threads``: both "
                        "names read and write the same value.")


#: Samples per video frame used when a temporal kernel cannot reduce an
#: interval internally. This fallback is approximate for sub-frame transients.
_FRAME_SUBSAMPLES = 8


def _subsample(t_out, dt, n_sub, start=None):
    """Sample each output interval at up to ``n_sub`` points.

    Used by temporal models that cannot reduce an interval internally.

    Parameters
    ----------
    start : float, optional
        Start of the first interval (ms).

    Returns
    -------
    t : array
        Sample times (ms).
    idx : array
        Start index of each interval in ``t``.
    """
    ticks = np.round(np.asarray(t_out, dtype=np.float64) / dt).astype(np.int64)
    # An interval runs from the previous output point up to and including this
    # one. Brightness is continuous, so the value carried across the boundary
    # is a floor on what the next interval reaches:
    first = ticks[0] if start is None else int(round(float(start) / dt))
    lo = np.concatenate(([min(first, ticks[0])], ticks[:-1]))
    parts = []
    for a, b in zip(lo, ticks):
        span = int(b - a)
        if span <= 0:
            parts.append(np.array([b], dtype=np.int64))
            continue
        # An interval cannot be sampled more often than it has `dt` steps:
        k = min(int(n_sub), span)
        parts.append(a + np.round(
            np.arange(1, k + 1) * span / k).astype(np.int64))
    idx = np.cumsum([0] + [p.size for p in parts[:-1]])
    return np.concatenate(parts) * dt, idx


def _frame_clock(stim, dt, unit=ms):
    """Return percept output times for an encoded video stimulus.

    Encoders record source-frame timing in stimulus metadata. Output
    times are rounded to the model's ``dt`` grid.

    Returns
    -------
    t : array
        Frame-end times in ``unit``.
    start : float
        Start time of the first frame.

    Returns None for stimuli without encoder frame metadata.

    .. versionadded:: 0.10.0
    """
    meta = getattr(stim, 'metadata', None)
    if not isinstance(meta, dict):
        return None
    enc = meta.get('encoder')
    if not isinstance(enc, dict):
        # `Stimulus` files metadata it does not recognize under 'user':
        user = meta.get('user')
        enc = user.get('encoder') if isinstance(user, dict) else None
    if not isinstance(enc, dict) and 'stim' in meta:
        # A `Percept` on its way from the spatial model to the temporal one
        # carries the stimulus it came from, and the frame clock with it:
        return _frame_clock(meta['stim'], dt, unit=unit)
    if not isinstance(enc, dict):
        return None
    try:
        frame_time = np.asarray(enc['frame_time'], dtype=np.float64)
        frame_dur = float(enc['frame_dur'])
    except (KeyError, TypeError, ValueError):
        return None
    if frame_time.size == 0 or not np.isfinite(frame_dur) or frame_dur <= 0:
        return None
    # Encoder frame metadata is stored in milliseconds; convert it to the
    # model's time unit before comparing it with `dt` or `t_percept`.
    if unit != ms:
        frame_time = Quantity(frame_time, ms).to_value(unit)
        frame_dur = Quantity(frame_dur, ms).to_value(unit)
    # Count in whole `dt` steps rather than rounding each frame time, so that
    # the spacing comes out exactly even and every point is exactly a multiple
    # of `dt` (which `predict_percept` insists on):
    step = max(1, int(round(frame_dur / dt)))
    start = int(round(float(frame_time[0]) / dt))
    ends = start + np.arange(1, frame_time.size + 1, dtype=np.int64) * step
    return ends * dt, start * dt


def _vfmap_first(params):
    """Apply ``vfmap`` before parameters whose units depend on that map."""
    if 'vfmap' not in params:
        return params
    return {'vfmap': params['vfmap'],
            **{key: val for key, val in params.items() if key != 'vfmap'}}


def _length_valued(value):
    """Return whether a value or pair contains a physical length."""
    values = value if isinstance(value, (list, tuple)) else [value]
    return any(isinstance(v, (Quantity, Unit)) and v.dimension == um.dimension
               for v in values)


def _require_stim_dimension(model, stim):
    """Require a stimulus with a physical dimension accepted by ``model``.

    Percepts are not checked because they represent model output rather
    than electrical stimulation.
    """
    if not isinstance(stim, Stimulus):
        return
    accepted = (model.stimulus_unit,) + tuple(model.extra_stimulus_units)
    expected = ' or '.join(_describe_unit(unit) for unit in accepted)
    if stim.unit.dimension in {unit.dimension for unit in accepted}:
        if (stim.unit.dimension.is_dimensionless and
                not stim._is_normalized_drive):
            # Dimensionless model input must be encoded drive, not gray levels.
            raise DimensionMismatchError(
                f"{type(model).__name__} expects {expected}, and this "
                f"stimulus is dimensionless but is not a normalized drive: "
                f"gray levels are not stimulation. Encode the picture first, "
                f"or hand the implant the picture and let its encoder do it.")
        return
    raise DimensionMismatchError(
        f"{type(model).__name__} expects {expected}, got "
        f"{_describe_unit(stim.unit)}.")


def _spatial_input(stim):
    """Return the spatial view of a prepared stimulus.

    Encoded stimuli expose frame-level modulation to spatial-only
    models instead of the time-resolved pulse schedule.
    """
    return stim._spatial_view()


def _delivered(stim):
    """Return the delivered pulse train for a prepared stimulus.

    Encoded stimuli carry a frame-level view for spatial-only models. Remove that
    view before a temporal stage integrates the waveform.
    """
    if stim is None or not stim._has_spatial_view:
        return stim
    return Stimulus(stim)


def _device_scene(scene, implant):
    """The visual scene the implant's own input pipeline sees"""
    source = implant._preprocess(scene.source)
    if source is scene.source:
        return scene
    if not isinstance(source, (ImageStimulus, VideoStimulus)):
        raise TypeError(
            f"This implant's 'preprocess' returned a "
            f"{type(source).__name__}, which has no pixels to place in the "
            f"visual field. Preprocessing a scene operates on the picture, so "
            f"it has to give an ImageStimulus or a VideoStimulus back; "
            f"turning gray levels into current is the encoder's job.")

    def refuse(what, before, after):
        raise ValueError(
            f"This implant's 'preprocess' changed the scene's {what} from "
            f"{before} to {after}. A scene's 'fov' describes the geometry of "
            f"the source it was given, so preprocessing may change pixel "
            f"values and channels, but not spatial shape or timing.")

    if isinstance(source, VideoStimulus) != isinstance(scene.source,
                                                       VideoStimulus):
        refuse('kind', type(scene.source).__name__, type(source).__name__)
    device = Scene(source, fov=scene.fov)
    if device.shape != scene.shape:
        refuse('shape', scene.shape, device.shape)
    if scene.time is not None:
        # Same instants, told in whichever unit preprocessing handed back:
        mine = np.asarray(scene.time)
        theirs = np.asarray(as_value(Quantity(np.asarray(device.time),
                                              device.time_unit),
                                     scene.time_unit, 'time'))
        if mine.size != theirs.size:
            refuse('frame count', mine.size, theirs.size)
        if not np.allclose(mine, theirs):
            refuse('frame times', f'{mine} {scene.time_unit}',
                   f'{theirs} {scene.time_unit}')
    return device


def _scene_stim(model, scene, gaze):
    """Prepare electrode stimulation sampled from a scene."""
    if not model.has_space:
        raise ValueError("A scene is registered against the retina, which "
                         "needs a spatial model. This model has only a "
                         "temporal one.")
    model.spatial._require_implant()
    implant = model.implant
    vfmap = getattr(model.spatial, 'vfmap', None)
    if not isinstance(vfmap, RetinalMap):
        raise ValueError(
            f"A scene reaches the electrodes through the model's 'vfmap', "
            f"which has to say where on the retina each degree of visual "
            f"angle lands. This model's is a {type(vfmap).__name__}; "
            f"registering a scene against a cortical map is not implemented.")
    if implant.encoder is None:
        raise ValueError(
            "A scene is a picture, and there is no principled default for "
            "turning a gray level into stimulation. Give the implant an "
            "'encoder' (e.g. an AmplitudeEncoder, or a PRIMAEncoder for a "
            "photovoltaic device) to say how.")
    device_scene = _device_scene(scene, implant)
    xy = implant.earray.coordinates(vfmap.tissue_unit)[:, :2].T
    x_vf, y_vf = vfmap.ret_to_dva(*xy)
    gray = device_scene._device_input(x_vf, y_vf, gaze=gaze)
    if device_scene.time is None:
        # A still scene is sampled as a one-frame movie; a `Stimulus` with no
        # time axis wants that frame axis gone, or it reads the frame as a
        # time point:
        gray = gray[:, 0]
    seen = Stimulus(gray, electrodes=implant.electrode_names,
                    time=device_scene.time,
                    metadata=device_scene.source.metadata)
    # Preprocessing already ran on the scene source. Use a shallow copy with
    # preprocessing disabled for the remaining preparation steps:
    device = copy(implant)
    device.preprocess = False
    return device.prepare_stim(seen._inherit_units(device_scene.source))


def _blend_meridian(resp, grid, meridian, width):
    """Blend a response across a visual-field meridian.

    ``width`` is the Gaussian standard deviation in dva. Blurring is 1D,
    normal to the meridian, and tapered by distance from it. Time points are
    processed independently. A zero width or one-sided grid is a no-op.
    """
    if width is None or width == 0:
        return resp
    width = float(width)
    if width < 0:
        raise ValueError(f"Blend width must be non-negative, not {width}.")
    if meridian == 'vertical':
        dist, axis = grid.x, 1
    elif meridian == 'horizontal':
        dist, axis = grid.y, 0
    else:
        raise ValueError(f"Unknown meridian '{meridian}'; expected 'vertical' "
                         f"or 'horizontal'.")
    # Convert width from dva to samples:
    along = dist[:, 0] if axis == 0 else dist[0, :]
    if along.size < 2 or not (np.any(along < 0) and np.any(along > 0)):
        # Nothing to blend unless the grid straddles the meridian:
        return resp
    spacing = float(np.abs(np.diff(along)).mean())
    # Filter each time point independently:
    work = np.asarray(resp).reshape(dist.shape + (-1,))
    blurred = gaussian_filter1d(work, width / spacing, axis=axis,
                                mode='nearest')
    weight = np.exp(-dist ** 2 / (2.0 * width ** 2))[..., np.newaxis]
    weight = weight.astype(work.dtype, copy=False)
    # `work + weight * (blurred - work)`, accumulated into the buffer
    # `gaussian_filter1d` already returned. Written as one expression it costs
    # three more arrays the size of the whole response:
    np.subtract(blurred, work, out=blurred)
    np.multiply(blurred, weight, out=blurred)
    np.add(blurred, work, out=blurred)
    return blurred.reshape(resp.shape).astype(resp.dtype, copy=False)


#: Parameter names declared by each model class, cached for ``__setattr__``.
_declared = {}


def _declared_params(model):
    """Return cached parameter names declared by ``get_default_params``."""
    cls = type(model)
    names = _declared.get(cls)
    if names is None:
        names = _declared[cls] = frozenset(model.get_default_params())
    return names


def _unchanged(before, after):
    """Return whether an assignment preserves the current value.

    Values that cannot be compared are treated as changed.
    """
    if before is after:
        return True
    try:
        return bool(np.all(before == after))
    except Exception:
        return False


def _electrode_pitch(model):
    """Return median nearest-neighbor electrode spacing in ``space_unit``.

    Distances use the dimensions represented by ``vfmap`` so that pitch matches
    the coordinates used for prediction. Returns ``None`` for fewer than two
    electrodes or zero spacing.
    """
    coords = model.implant.earray.coordinates(model.space_unit)
    coords = coords[:, :model.vfmap.ndim]
    if len(coords) < 2:
        return None
    # The nearest *other* electrode, so the query asks for two:
    distances, _ = cKDTree(coords).query(coords, k=2)
    pitch = float(np.median(distances[:, 1]))
    return pitch if pitch > 0 else None


def _warn_rho_vs_pitch(model):
    """Warn when ``rho`` exceeds the implant's median electrode spacing."""
    pitch = _electrode_pitch(model)
    if pitch is None or model.rho <= pitch:
        return
    overlap = np.exp(-pitch ** 2 / (2 * model.rho ** 2))
    warnings.warn(
        f"rho={model.rho:.0f} um is wider than this implant's electrode "
        f"pitch ({pitch:.0f} um), a ratio of {model.rho / pitch:.2f}. A point "
        f"one pitch away from an electrode still sees {overlap:.0%} of its "
        f"peak, so neighbouring electrodes blur into each other and the "
        f"percept says more about rho than about which electrodes were "
        f"driven.")


def _warn_ignores_z(model, earray):
    """Warn when a model ignores nonzero electrode ``z`` coordinates."""
    if np.allclose([e.z for e in earray.electrode_objects], 0):
        return
    warnings.warn(
        f"{type(model).__name__} does not model electrode-retina distance: "
        f"nonzero z values do not change its response. In a real implant, "
        f"distance is expected to affect stimulation threshold and spatial "
        f"recruitment, but that relationship is not parameterized by this "
        f"model.")


class BaseModel(Parametrized, metaclass=ABCMeta):
    """Abstract base class for computational models.

    Adds build state to :py:class:`~pulse2percept.utils.Parametrized`.
    Changing a declared model parameter invalidates the build; prediction
    rebuilds automatically when needed.

    .. versionchanged:: 0.11.0
        ``predict_percept`` builds automatically after construction or a
        parameter change.
    """

    def __setattr__(self, name, value):
        """Invalidate the build when a declared parameter changes.

        Assignments that preserve the current value leave the build intact.
        """
        if _is_constructing(self) or name not in _declared_params(self):
            super().__setattr__(name, value)
            return
        before = getattr(self, name, None)
        super().__setattr__(name, value)
        if not _unchanged(before, getattr(self, name, None)):
            object.__setattr__(self, '_is_built', False)

    # Numerical kernels receive plain values in these canonical units.
    # They define the model's numerical contract, not user-configurable units.

    #: The unit stimulus values are expressed in
    stimulus_unit = uA
    #: Additional stimulus units accepted by this model
    extra_stimulus_units = ()
    #: The unit spatial coordinates are expressed in
    space_unit = um
    #: The unit time is expressed in
    time_unit = ms

    def __init__(self, **params):
        """Initialize a model from declared parameters.

        Parameters
        ----------
        **params : keyword arguments
            Values for parameters declared by ``get_default_params``.
        """
        super().__init__(**params)
        # This flag will be flipped once the ``build`` method was called
        self._is_built = False

    def _build(self):
        """Customize the building process by implementing this method"""
        pass

    def _stim_unit(self, stim):
        """Return the model unit matching ``stim``."""
        if stim.unit.dimension == self.stimulus_unit.dimension:
            return self.stimulus_unit
        for unit in self.extra_stimulus_units:
            if stim.unit.dimension == unit.dimension:
                return unit
        return self.stimulus_unit

    def _stim_values(self, stim):
        """Return stimulus values in the unit this model reads them in.

        Stimuli are converted at the model boundary; percept values are passed
        through because brightness is not a physical stimulus quantity.
        """
        if not isinstance(stim, Stimulus):
            return stim.data
        _require_stim_dimension(self, stim)
        return stim.values(self._stim_unit(stim))

    def _stim_times(self, stim):
        """Return the time axis in ``time_unit``.

        Applies to both stimuli and percepts.
        """
        if not isinstance(stim, (Stimulus, Percept)):
            return stim.time
        return stim.times(self.time_unit)

    def _to_stim_time(self, t, stim):
        """Convert model-side times to the stimulus time unit."""
        if t is None or not isinstance(stim, (Stimulus, Percept)) \
                or stim.time_unit == self.time_unit:
            return t
        return Quantity(t, self.time_unit).to_value(stim.time_unit)

    def _electrode_coords(self, earray, stim):
        """Return stimulus electrode coordinates in ``space_unit``.

        Coordinates follow ``stim.electrodes`` order and are returned as
        contiguous float32 arrays for the numerical kernels.

        Parameters
        ----------
        earray : :py:class:`~pulse2percept.implants.ElectrodeArray`
            Electrode array containing the named electrodes.
        stim : :py:class:`~pulse2percept.stimuli.Stimulus`
            Stimulus whose electrode ordering is required.

        Returns
        -------
        x, y, z : tuple of ndarray
            Coordinate arrays with shape ``(n_electrodes,)``.
        """
        xyz = earray.coordinates(self.space_unit, electrodes=stim.electrodes)
        return tuple(np.ascontiguousarray(xyz[:, i], dtype=np.float32)
                     for i in range(3))

    def build(self, **build_params):
        """Build the model.

        Runs expensive one-time setup after applying any supplied model
        parameters. ``predict_percept`` builds automatically when needed.

        Parameters
        ----------
        **build_params : keyword arguments
            Declared model parameters to set before building.

        Returns
        -------
        self

        Notes
        -----
        Subclasses should override ``_build``, not this method.
        """
        # Via `set_params`, not a bare `setattr` loop, so that a deprecated or
        # renamed parameter is handled here exactly as it is in the
        # constructor:
        self.set_params(**build_params)
        self._build()
        self._is_built = True
        return self

    @property
    def is_built(self):
        """A read-only flag indicating whether the model has been built"""
        return self._is_built

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        # Guard here as well as in the base implementation: without it, an
        # already-copied model would be rebuilt on every revisit.
        if id(self) in memodict:
            return memodict[id(self)]
        implant = getattr(self, '_implant', None)
        if implant is not None:
            # The implant is model context, not model state. Share it across
            # copies so geometry-dependent build state remains valid.
            memodict.setdefault(id(implant), implant)
        copied = super().__deepcopy__(memodict)
        if self.is_built:
            copied.build()
        return copied


class SpatialModel(BaseModel, metaclass=ABCMeta):
    """Abstract base class for spatial models.

    Spatial models map electrode stimulation to brightness on a sampled
    visual-field grid. Subclasses implement ``_predict_spatial`` and may
    override ``_build`` for precomputation.

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`
        Implant whose electrode geometry is modeled.

        .. versionadded:: 0.11.0

    xrange : (float, float) or Quantity, optional
        Horizontal visual-field extent in degrees of visual angle. On retinal
        maps, a physical retinal extent may be given instead and is resolved
        through ``vfmap``.
    yrange : (float, float) or Quantity, optional
        Vertical visual-field extent in degrees of visual angle. On retinal
        maps, a physical retinal extent may be given instead and is resolved
        through ``vfmap``.
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
        Fraction of peak Gaussian current spread below which an electrode may
        be skipped at a grid point. Set to 0 to disable the cutoff.
    vfmap : VisualFieldMap, optional
        Retinotopic map between visual-field and tissue coordinates.
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
        Number of OpenMP threads.
    n_jobs : int or None, optional
        Alias for ``n_threads``. ``None`` and -1 use all available CPU cores.

    Notes
    -----
    ``xrange`` and ``yrange`` always describe the simulated visual field and
    are stored in degrees of visual angle. A retinal length is only shorthand
    for selecting that extent through ``vfmap``; the resulting grid is still
    uniformly sampled in visual angle. ``step`` therefore only accepts angular
    spacing.

    .. versionadded:: 0.6
    """

    #: ``n_jobs`` is an alias for ``n_threads``; see ``_n_jobs_alias``.
    n_jobs = _n_jobs_alias()

    def __init__(self, implant, **params):
        # `vfmap` first: `xrange`/`yrange` may be given as a retinal extent,
        # which is resolved through the map as it is assigned. See
        # `_vfmap_first`.
        super().__init__(implant=implant, **_vfmap_first(params))
        self.grid = None

    @property
    def implant(self):
        """The prosthesis system whose geometry this model uses.

        Rebinding invalidates the spatial build.

        .. versionadded:: 0.11.0
        """
        return getattr(self, '_implant', None)

    @implant.setter
    def implant(self, implant):
        """The prosthesis system whose geometry this model uses.

        Rebinding the implant invalidates the spatial build.

        .. versionadded:: 0.11.0
        """
        if implant is not None and not isinstance(implant, ProsthesisSystem):
            raise TypeError(f"'implant' must be a ProsthesisSystem object, "
                            f"not {type(implant)}.")
        if implant is not getattr(self, '_implant', None):
            # Spatial build state depends on implant geometry:
            self._is_built = False
        self._implant = implant

    def _require_implant(self):
        """Raise if no implant is bound to the spatial model."""
        if not isinstance(self.implant, ProsthesisSystem):
            raise ValueError(
                f"{type(self).__name__} predicts what a particular implant "
                f"produces, so it needs one: "
                f"{type(self).__name__}(implant=ArgusII()). The stimulus is "
                f"what 'predict_percept' takes.")

    def set_params(self, **params):
        """Set the parameters of this model

        ``vfmap`` is applied before the other parameters, so that a retinal
        extent given for ``xrange``/``yrange`` in the same call is resolved
        through the map the caller asked for. See ``_vfmap_first``.
        """
        super().set_params(**_vfmap_first(params))

    def _normalize_param_value(self, name, value):
        """Normalize a parameter to its stored unit.

        Physical ``xrange`` and ``yrange`` values are resolved through ``vfmap``;
        other unitful parameters use the generic conversion.
        """
        if name in ('xrange', 'yrange') and _length_valued(value):
            return self._retinal_range_to_dva(name, value)
        return super()._normalize_param_value(name, value)

    def _retinal_range_to_dva(self, name, value):
        """Resolve a retinal extent to a visual-field range.

        ``xrange`` is converted along the horizontal retinal meridian and
        ``yrange`` along the vertical meridian. The result is stored in degrees of
        visual angle and is not reinterpreted if ``vfmap`` changes later.

        Parameters
        ----------
        name : {'xrange', 'yrange'}
            Range being assigned.
        value : (min, max)
            Retinal extent.

        Returns
        -------
        tuple of float
            Visual-field extent in increasing order.
        """
        vfmap = getattr(self, 'vfmap', None)
        if not isinstance(vfmap, RetinalMap):
            raise DimensionMismatchError(
                f"'{name}' is a visual field extent, measured in degrees of "
                f"visual angle. A physical length is shorthand for one only "
                f"on a retinal map, and this model's vfmap is a "
                f"{type(vfmap).__name__}. Specify '{name}' in dva instead.")
        # In the unit the map's tissue side is measured in, which is what its
        # inverse transform below expects:
        extent = np.asarray(as_value(value, vfmap.tissue_unit, name),
                            dtype=np.float64).ravel()
        if extent.size != 2:
            raise ValueError(f"'{name}' must be a (min, max) pair, not "
                             f"{value}.")
        lo, hi = extent
        try:
            if name == 'xrange':
                lo_dva, _ = vfmap.ret_to_dva(lo, 0)
                hi_dva, _ = vfmap.ret_to_dva(hi, 0)
            else:
                _, lo_dva = vfmap.ret_to_dva(0, lo)
                _, hi_dva = vfmap.ret_to_dva(0, hi)
        except NotImplementedError:
            raise NotImplementedError(
                f"This visual field map ({type(vfmap).__name__}) cannot infer "
                f"a visual field range from retinal distance. Specify "
                f"'{name}' in dva instead.") from None
        # Sorted, because the retinal y axis points the opposite way from the
        # visual field's, so the two end points can come back swapped:
        return tuple(sorted((float(lo_dva), float(hi_dva))))

    def get_default_params(self):
        """Return a dictionary of default values for all model parameters"""
        params = {
            'implant': None,
            'xrange': (-15, 15),  # dva
            'yrange': (-15, 15),  # dva
            'step': 0.25,  # dva
            'grid_type': 'rectangular',
            'thresh_percept': 0,
            'min_current_spread': 1e-8,
            'vfmap': Curcio1990Map(),
            'n_gray': None,
            'noise': None,
            'verbose': True,
            'ndim' : [2],
            # `n_jobs` writes through to `n_threads`, so it must come last.
            'n_threads': multiprocessing.cpu_count(),
            'n_jobs': None,
        }
        return params

    def get_param_units(self):
        """Return a dict of the units that parameters are stored in

        ``xrange`` and ``yrange`` additionally accept a retinal extent, which
        is not a unit conversion and so does not appear here; see
        ``_retinal_range_to_dva``.
        """
        return {
            **super().get_param_units(),
            # The simulated patch of visual field is specified in degrees of
            # visual angle; the visual field map turns those into tissue
            # coordinates when the grid is built:
            'xrange': dva,
            'yrange': dva,
            'step': dva,
        }

    def _cutoff_r2(self, rho):
        """Return the squared distance where Gaussian spread is negligible.

        For a spread ``exp(-r**2 / (2 * rho**2))``, converts
        ``min_current_spread`` to a squared-distance cutoff. The default 1e-8 is
        about 6.1 ``rho``; 0 disables the cutoff.

        Parameters
        ----------
        rho : float
            Current-spread decay constant in microns.

        Returns
        -------
        np.float32
            Squared distance in microns squared, or ``inf`` if disabled.
        """
        min_spread = self.min_current_spread
        if min_spread is None or min_spread <= 0:
            return np.float32(np.inf)
        if min_spread >= 1:
            raise ValueError(f"min_current_spread must be smaller than 1 (or "
                             f"0 to disable the cutoff), not {min_spread}.")
        return np.float32(-2.0 * rho ** 2 * np.log(min_spread))

    def build(self, **build_params):
        """Build the spatial model.

        Applies any supplied parameters, validates the implant and visual-field
        map, builds the sampling grid, then runs ``_build``.

        Parameters
        ----------
        **build_params : keyword arguments
            Declared model parameters to set before building.

        Returns
        -------
        self
        """
        # See `BaseModel.build`:
        self.set_params(**build_params)
        self._require_implant()
        if self.vfmap.ndim not in self.ndim:
            raise ValueError(f"Model expects one of {self.ndim} dimensions, but "
                             f"visual field map has {self.vfmap.ndim} dimensions.")
        self.grid = Grid2D(self.xrange, self.yrange, step=self.step,
                           grid_type=self.grid_type)
        self.grid.build(self.vfmap)
        self._build()
        self._is_built = True
        return self

    @abstractmethod
    def _predict_spatial(self, earray, stim):
        """Compute the spatial response.

        Parameters
        ----------
        earray : :py:class:`~pulse2percept.implants.ElectrodeArray`
            Electrode array for the bound implant.
        stim : :py:class:`~pulse2percept.stimuli.Stimulus`
            Prepared stimulus with shape electrodes x time.

        Returns
        -------
        np.ndarray
            Brightness with shape grid points x time.
        """
        raise NotImplementedError

    def _postprocess_spatial(self, resp):
        """Hook for spatial-model postprocessing."""
        return resp

    def predict_percept(self, source, t_percept=None):
        """Predict the spatial response.

        Parameters
        ----------
        source : stimulus source
            Anything accepted by
            :py:meth:`~pulse2percept.implants.ProsthesisSystem.prepare_stim`.
        t_percept : float or array-like, optional
            Output times in ``time_unit``. If omitted, use the source time points.
            Unitful times are accepted.

        Returns
        -------
        percept : :py:class:`~pulse2percept.percepts.Percept` or None
            Percept with shape Y x X x T, or ``None`` for an empty source.

        Notes
        -----
        For an encoded image or video, a spatial-only model uses frame-level
        modulation rather than the delivered pulse train. In a composite
        :py:class:`Model` with a temporal stage, the spatial stage receives the
        delivered train so the temporal model can integrate it.

        .. versionchanged:: 0.11.0
            Takes the stimulus source rather than an implant carrying a stimulus.
        """
        if not self.is_built:
            self.build()
        return self._predict_prepared(self.implant.prepare_stim(source),
                                      t_percept=t_percept)

    def _predict_prepared(self, stim, t_percept=None):
        """Predict the spatial response to an already prepared stimulus.

        Composite models use this path to prepare stimulation once before running
        spatial and temporal stages.
        """
        if not self.is_built:
            self.build()
        t_percept = as_value(t_percept, self.time_unit, 't_percept')
        if stim is None:
            # Nothing to see here:
            return None
        source = _spatial_input(stim)
        _require_stim_dimension(self, source)
        if source.time is None and t_percept is not None:
            # Static modulation has no time axis even if its encoded pulse
            # train does:
            what = ("the modulation behind this stimulus"
                    if source is not stim else "stimulus")
            raise ValueError(f"Cannot calculate spatial response at times "
                             f"t_percept={t_percept} because {what} does not "
                             f"have a time component.")
        # Make sure we don't change the user's Stimulus object:
        stim = deepcopy(source)
        # Make sure to operate on the compressed stim:
        if not stim.is_compressed:
            stim.compress()
        if t_percept is None:
            # In `time_unit`, like everything else on this side of the
            # boundary; `_to_stim_time` converts back where the stimulus is
            # indexed by it below:
            t_percept = self._stim_times(stim)
        n_time = 1 if t_percept is None else np.array([t_percept]).size
        if stim.data.size == 0:
            # Stimulus was compressed to zero:
            resp = np.zeros((self.grid.x.size, n_time), dtype=np.float32)
        else:
            # Calculate the Stimulus at requested time points:
            if t_percept is not None:
                # Save electrode parameters
                # np.asarray: indexing a single-electrode stimulus returns a
                # scalar, which has no `reshape`:
                at = self._to_stim_time(t_percept, stim)
                # Preserve the normalized-drive marker through resampling.
                rebuild = type(stim) if stim._is_normalized_drive else Stimulus
                stim = rebuild(
                    np.asarray(stim[:, at]).reshape((-1, n_time)),
                    electrodes=stim.electrodes, time=at
                )._inherit_units(stim)._inherit_metadata(stim)
                # find unique stimulus points
                _, t_unique, inverse = np.unique(stim.data.T, axis=0,
                                                 return_index=True,
                                                 return_inverse=True)
                # np.unique orders rows by value, not time. Restore chronological
                # order and remap `inverse`; flatten it for NumPy 2.x shape
                # differences in axis-wise unique.
                order = np.argsort(t_unique)
                t_unique = t_unique[order]
                rank = np.empty_like(order)
                rank[order] = np.arange(order.size)
                inverse = rank[np.ravel(inverse)]
                uniq_time = stim.time[t_unique]
                if len(uniq_time) == 1:
                    uniq_time = None
                # `_predict_spatial` only ever sees this de-duplicated
                # copy, so the stimulus' metadata has to come along:
                stim_unique = rebuild(
                    stim[:, stim.time[t_unique]], electrodes=stim.electrodes,
                    time=uniq_time
                )._inherit_units(stim)._inherit_metadata(stim)
                resp_unique = self._predict_spatial(self.implant.earray,
                                                    stim_unique)
                # reconstruct original time points, making sure to preserve C ordering
                resp = resp_unique[..., inverse].copy(order='C')
            else:
                resp = self._predict_spatial(self.implant.earray, stim)
        resp = self._postprocess_spatial(resp)
        return Percept(resp.reshape(list(self.grid.x.shape) + [-1]),
                       space=self.grid, time=t_percept,
                       time_unit=self.time_unit,
                       metadata={'stim': stim}, n_gray=self.n_gray, noise=self.noise)

    def plot(self, use_dva=False, style='hull', autoscale=True, ax=None,
             figsize=None):
        """Plot the model

        Parameters
        ----------
        use_dva : bool, optional
            Uses degrees of visual angle (dva) if True, else retinal
            coordinates (microns)
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

        Returns
        -------
        ax : ``matplotlib.axes.Axes``
            Returns the axis object of the plot
        """
        if not self.is_built:
            self.build()

        zorder = ZORDER['background'] + (0 if use_dva else 1)

        ax = self.grid.plot(autoscale=autoscale, ax=ax, style=style, zorder=zorder,
                            figsize=figsize, use_dva=use_dva)

        if use_dva:
            ax.set_xlabel('x (dva)')
            ax.set_ylabel('y (dva)')
        else:
            ax.set_xlabel('x (microns)')
            ax.set_ylabel('y (microns)')
        return ax


class TemporalModel(BaseModel, metaclass=ABCMeta):
    """Abstract base class for temporal models.

    Temporal models map a time-varying stimulus or percept to brightness over
    time. Subclasses implement ``_predict_temporal`` and may override
    ``_build`` for precomputation.

    Parameters
    ----------
    dt : float or Quantity, optional
        Simulation time step. Plain values are interpreted as milliseconds.
    thresh_percept : float, optional
        Brightness values below this threshold are set to zero.
    reduce : {'last', 'peak'}, optional
        How automatically selected output intervals are summarized.
        ``'last'`` reports brightness at the interval endpoint; ``'peak'``
        reports the maximum brightness reached in the interval. Explicit
        ``t_percept`` values always request those exact instants.
    verbose : bool, optional
        Whether to print status messages.
    n_threads : int, optional
        Number of OpenMP threads.
    n_jobs : int or None, optional
        Alias for ``n_threads``. ``None`` and -1 use all available CPU cores.

    Notes
    -----
    Models with ``_reduces_intervals = True`` compute ``'peak'`` within the
    integrator. Other temporal models approximate it by subsampling each
    output interval.

    .. versionadded:: 0.6

    .. versionchanged:: 0.10.0
        Added ``reduce``.
    """

    #: ``n_jobs`` is an alias for ``n_threads``; see ``_n_jobs_alias``.
    n_jobs = _n_jobs_alias()

    #: Polarity that drives brightness: -1 for cathodic, +1 for anodic.
    #: Used when checking stimulus polarity and constructing canonical drives.
    _drive_sign = -1

    #: Whether ``_predict_temporal`` can return an exact peak over each interval
    #: instead of relying on subsampling in ``predict_percept``.
    _reduces_intervals = False

    def get_default_params(self):
        """Return default model parameters."""
        params = {
            'dt': 0.005,  # ms
            'thresh_percept': 0,
            'reduce': 'last',
            'verbose': True,
            'n_threads': multiprocessing.cpu_count(),
            'n_jobs': None,  # Alias for n_threads; must be applied last
        }
        return params

    def get_param_units(self):
        """Return a dict of the units that parameters are stored in"""
        # `dt` is the simulation step, so it counts in whatever the model
        # counts time in -- milliseconds for every model p2p ships:
        return {**super().get_param_units(), 'dt': self.time_unit}

    @abstractmethod
    def _predict_temporal(self, stim, t_percept):
        """Compute the temporal response.

        Parameters
        ----------
        stim : Stimulus or Percept
            Time-varying input.
        t_percept : array-like
            Output times in milliseconds.

        Returns
        -------
        np.ndarray
            Response with shape space x time.

        Notes
        -----
        Models that support exact interval reduction accept a third ``reduce``
        argument and set ``_reduces_intervals = True``.
        """
        raise NotImplementedError

    def predict_percept(self, stim, t_percept=None):
        """Predict the temporal response.

        Parameters
        ----------
        stim : Stimulus or Percept
            Time-varying input. The temporal model is applied independently at
            each spatial location.
        t_percept : float or array-like, optional
            Output times in ``time_unit``. Unitful times are accepted. If omitted,
            encoded video frame times are used when available; otherwise output is
            sampled every 20 ms, with at least one frame for a shorter stimulus.

        Returns
        -------
        percept : :py:class:`~pulse2percept.percepts.Percept` or None
            Percept with shape Y x X x T, or ``None`` if ``stim`` is ``None``.

        Notes
        -----
        Explicit ``t_percept`` values sample brightness at those instants.
        Otherwise ``reduce`` determines whether each output interval reports its
        endpoint or peak. Requested times are sorted and must lie on the ``dt``
        grid.

        .. versionchanged:: 0.10.0
            Automatically selected output times may summarize intervals via
            ``reduce``.
        """
        if not self.is_built:
            self.build()
        if stim is None:
            # Nothing to see here:
            return None
        if not isinstance(stim, (Stimulus, Percept)):
            raise TypeError(f"'stim' must be a Stimulus or Percept object, "
                            f"not {type(stim)}.")
        t_percept = as_value(t_percept, self.time_unit, 't_percept')
        _require_stim_dimension(self, stim)
        if stim.time is None:
            raise ValueError("Cannot calculate temporal response, because "
                             "stimulus/percept does not have a time "
                             "component.")
        # Make sure we don't change the user's Stimulus/Percept object:
        _stim = deepcopy(stim)
        if isinstance(stim, Stimulus):
            # Make sure to operate on the compressed stim:
            if not _stim.is_compressed:
                _stim.compress()
            _space = [len(stim.electrodes), 1]
        elif isinstance(stim, Percept):
            _space = [len(stim.ydva), len(stim.xdva)]
        # In `time_unit`: `_frame_clock`, `dt` and `t_percept` all count in it
        _time = self._stim_times(stim)

        reduce, t_out, sub_idx = 'last', None, None
        if t_percept is None:
            # With automatic output times, `reduce` summarizes each interval.
            reduce = self.reduce
            if reduce not in ('peak', 'last'):
                raise ValueError(f"'reduce' must be 'peak' or 'last', not "
                                 f"{self.reduce!r}.")
            # Prefer encoder frame timing; otherwise report at 50 Hz.
            frames = _frame_clock(stim, self.dt, unit=self.time_unit)
            if frames is None:
                # Convert 20 ms into the model's unit. `nextafter` makes an exact
                # frame boundary inclusive. The minimum one-frame duration keeps
                # sub-frame stimuli visible instead of reporting only t=0.
                frame_dur = as_value(20 * ms, self.time_unit)
                end = np.maximum(frame_dur, _time[-1])
                t_out = np.arange(0, np.nextafter(end, np.inf), frame_dur)
                first = None
            else:
                t_out, first = frames
            t_percept = t_out
            if reduce == 'peak' and not self._reduces_intervals:
                # This model can only be asked for instants, so approximate the
                # peak by asking for several per interval and keeping the
                # largest:
                t_percept, sub_idx = _subsample(t_out, self.dt,
                                                _FRAME_SUBSAMPLES, first)
        # We need to make sure the requested `t_percept` are sorted and
        # multiples of `dt`:
        t_percept = np.sort([t_percept]).flatten()
        remainder = np.mod(t_percept, self.dt) / self.dt
        atol = 1e-3
        within_atol = (remainder < atol) | (np.abs(1 - remainder) < atol)
        if not np.all(within_atol):
            raise ValueError(f"t={t_percept[np.logical_not(within_atol)]} are "
                             f"not multiples of dt={self.dt:.2e}.")
        if _stim.data.size == 0:
            # Stimulus was compressed to zero:
            resp = np.zeros(_space + [t_percept.size], dtype=np.float32)
        elif self._reduces_intervals:
            # This model tracks the peak inside its own integrator, which is
            # exact however coarse the output rate is:
            resp = self._predict_temporal(_stim, t_percept, reduce)
            self._warn_if_blank(_stim, resp)
        else:
            # Calculate the Stimulus at requested time points:
            resp = self._predict_temporal(_stim, t_percept)
            self._warn_if_blank(_stim, resp)
        resp = resp.reshape(_space + [t_percept.size])
        if sub_idx is not None:
            # Preserve pulse-driven peaks rather than averaging them over gaps.
            resp = np.maximum.reduceat(resp, sub_idx, axis=-1)
            t_percept = t_out
        # A temporal model rewrites a spatial percept frame by frame; it does
        # not move it in the visual field, so it hands the grid back on:
        return Percept(resp, space=None, time=t_percept,
                       time_unit=self.time_unit,
                       metadata={'stim': stim})._inherit_space(stim)

    def _warn_if_blank(self, stim, resp):
        """Warn when stimulus polarity explains an all-zero response.
        """
        if np.any(resp) or not np.any(stim.data):
            return
        # Only if *nothing* in the stimulus has the sign the model responds to:
        if np.any(np.sign(stim.data) == self._drive_sign):
            return
        polarity = 'cathodic (negative)' if self._drive_sign < 0 else \
            'anodic (positive)'
        warnings.warn(
            f"{type(self).__name__} produced an all-zero percept: brightness "
            f"in this model is driven by {polarity} current, and the stimulus "
            f"has none. Encoding an image or a video with "
            f"pulse2percept.stimuli.AmplitudeEncoder gives it the right "
            f"polarity; otherwise negate it.")


class Model(Frozen, PrettyPrint):
    """Composite computational model.

    Combines an optional spatial model with an optional temporal model.

    .. code-block:: python

        model = Model(spatial=ScoreboardSpatial(ArgusII()),
                      temporal=Nanduri2012Temporal())

    Parameters
    ----------
    spatial : :py:class:`~pulse2percept.models.SpatialModel` or type, optional
        Spatial model instance or class.
    temporal : :py:class:`~pulse2percept.models.TemporalModel` or type, optional
        Temporal model instance or class.
    implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`, optional
        Implant used by the spatial component. Required only when ``spatial``
        is given as a class.

        .. versionadded:: 0.11.0
    **params : keyword arguments
        Parameters forwarded to the spatial and temporal components. A name
        present on both components is applied to both.

    .. versionadded:: 0.6
    """

    # Composite units come from the component that consumes or emits the
    # quantity. Define them explicitly because `__getattr__` may return both.

    @property
    def stimulus_unit(self):
        """The unit stimulus values are expressed in

        The stimulus goes to the spatial model if there is one, and straight
        to the temporal model otherwise.
        """
        if self.has_space:
            return self.spatial.stimulus_unit
        if self.has_time:
            return self.temporal.stimulus_unit
        return BaseModel.stimulus_unit

    @property
    def extra_stimulus_units(self):
        """Additional stimulus units accepted by the active component"""
        if self.has_space:
            return self.spatial.extra_stimulus_units
        if self.has_time:
            return self.temporal.extra_stimulus_units
        return BaseModel.extra_stimulus_units

    @property
    def space_unit(self):
        """The unit spatial coordinates are expressed in

        The temporal model never sees a coordinate.
        """
        if self.has_space:
            return self.spatial.space_unit
        return BaseModel.space_unit

    @property
    def time_unit(self):
        """Time unit used by the final model stage.

        ``t_percept`` and the returned percept use the temporal model's unit when
        present, otherwise the spatial model's unit.
        """
        if self.has_time:
            return self.temporal.time_unit
        if self.has_space:
            return self.spatial.time_unit
        return BaseModel.time_unit

    def __init__(self, spatial=None, temporal=None, *, implant=None,
                 **params):
        # `implant` is named explicitly so it cannot reach the temporal
        # component through `**params`; only the spatial one has geometry.
        # Normalize renamed parameters once before class construction and
        # subsequent `set_params`, avoiding duplicate warnings.
        for model in (spatial, temporal):
            if isinstance(model, type):
                params = rename_deprecated_params(
                    type(self).__name__, params,
                    getattr(model, '_renamed_params', {}))
        # Set the spatial model:
        if spatial is not None and not isinstance(spatial, SpatialModel):
            if issubclass(spatial, SpatialModel):
                # User should have passed an instance, not a class:
                spatial = spatial(implant, **params)
            else:
                raise TypeError(f"'spatial' must be a SpatialModel instance, "
                                f"not {type(spatial)}.")
        self.spatial = spatial
        # Set the temporal model:
        if temporal is not None and not isinstance(temporal, TemporalModel):
            if issubclass(temporal, TemporalModel):
                # User should have passed an instance, not a class:
                temporal = temporal(**params)
            else:
                raise TypeError(f"'temporal' must be a TemporalModel instance, "
                                f"not {type(temporal)}.")
        self.temporal = temporal
        # Use user-specified parameter values instead of defaults:
        self.set_params(params)
        if implant is not None:
            if not self.has_space:
                raise ValueError(
                    "An implant is where a spatial model puts its electrodes, "
                    "and this model has only a temporal component. A temporal "
                    "model is given the stimulus and nothing else.")
            bound = self.spatial.implant
            if bound is not None and bound is not implant:
                raise ValueError(
                    f"This model was given two different implants: "
                    f"'implant' is {type(implant).__name__} and the spatial "
                    f"model is already bound to "
                    f"{type(bound).__name__}. A model describes one device, "
                    f"so name it once - either here or on the spatial model.")
            self.spatial.implant = implant

    def __getattr__(self, attr):
        """Forward missing attributes to model components.

        If both components define the attribute, return a dictionary with
        ``'spatial'`` and ``'temporal'`` entries.
        """
        # Check the spatial/temporal model:
        try:
            spatial = getattr(self.spatial, attr)
            spatial_valid = True
        except AttributeError:
            spatial_valid = False
        try:
            temporal = getattr(self.temporal, attr)
            temporal_valid = True
        except AttributeError:
            temporal_valid = False
        if not spatial_valid and not temporal_valid:
            # If we are in the constructor, this will be caught later and
            # a new variable will be constructed
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{attr}'.")
        if not spatial_valid:
            return temporal
        if not temporal_valid:
            return spatial
        return {'spatial': spatial, 'temporal': temporal}

    def __setattr__(self, name, value):
        """Set a composite attribute or forward a model parameter.

        Attributes created during construction stay on the composite. Later
        assignments are forwarded to the spatial and/or temporal component.
        """
        if _is_constructing(self):
            # Constructor-owned attributes stay on the composite; user parameters
            # are forwarded explicitly by `set_params`.
            super().__setattr__(name, value)
            return
        self._set_component_param(name, value)

    def _set_component_param(self, name, value):
        """Forward an assignment to the spatial and/or temporal model

        Raises a ``FreezeError`` if neither sub-model knows the name; outside
        the constructor there is nowhere else for it to go.
        """
        found = False
        try:
            self.spatial.__setattr__(name, value)
            found = True
        except (AttributeError, FreezeError):
            pass
        try:
            self.temporal.__setattr__(name, value)
            found = True
        except (AttributeError, FreezeError):
            pass
        if not found:
            err_str = (f"'{name}' not found. You cannot add attributes to "
                       f"{self.__class__.__name__} outside the constructor.")
            raise FreezeError(err_str)

    def __deepcopy__(self, memodict=None):
        """Return a deep copy of the model.
        """
        if memodict is None:
            memodict = {}
        if id(self) in memodict:
            return memodict[id(self)]
        attributes = deepcopy(self.__dict__, memodict)
        # Most Model subclasses create their spatial and temporal models in
        # the constructor, so those cannot be passed in as parameters:
        spatial = attributes.pop('spatial', None)
        temporal = attributes.pop('temporal', None)
        # A subclass builds its own spatial component and so requires the
        # implant, which lives on that component and not in `__dict__`.
        # `Model` itself declares it optional and takes the copy back below.
        implant_param = inspect.signature(
            self.__class__.__init__).parameters.get('implant')
        implant_required = getattr(implant_param, 'default',
                                   None) is inspect.Parameter.empty
        if spatial is not None and implant_required:
            attributes['implant'] = spatial.implant
        result = self.__class__(**attributes)
        # Restore copied sub-models after construction; their parameters do not
        # live in the composite `__dict__`. Bypass forwarding `__setattr__`.
        if spatial is not None:
            object.__setattr__(result, 'spatial', spatial)
        if temporal is not None:
            object.__setattr__(result, 'temporal', temporal)
        if self.is_built:
            result.build()
        memodict[id(self)] = result
        return result

    def __eq__(self, other):
        """Return whether two models have equal spatial and temporal components.
        """
        if not isinstance(other, self.__class__):
            return False
        if id(self) == id(other):
            return True
        return self.temporal == other.temporal and self.spatial == other.spatial

    def __hash__(self):
        # Default python 2.6+ implementation
        return id(self) // 16

    def _pprint_params(self):
        """Return a dictionary of parameters to pretty - print"""
        params = {'spatial': self.spatial, 'temporal': self.temporal}
        # Also display the parameters from the spatial/temporal model:
        if self.has_space:
            params.update(self.spatial._pprint_params())
        if self.has_time:
            params.update(self.temporal._pprint_params())
        return params

    def set_params(self, params):
        """Set component model parameters.

        A parameter present on both spatial and temporal components is applied to
        both.

        Parameters
        ----------
        params : dict
            Parameter values to set.
        """
        # Instance-based composites bypass `BaseModel.__init__`; collect
        # deprecations from both components and warn once.
        specs = {}
        renamed = {}
        for model in (self.spatial, self.temporal):
            specs.update(getattr(model, '_deprecated_params', {}))
            renamed.update(getattr(model, '_renamed_params', {}))
        warn_deprecated_params(type(self).__name__, params, specs)
        params = rename_deprecated_params(type(self).__name__, params, renamed)
        # Apply `vfmap` first, then forward directly to components. During
        # construction, assigning on `self` would create composite attributes.
        for key, val in _vfmap_first(params).items():
            self._set_component_param(key, val)

    def build(self, **build_params):
        """Build all model components.

        Unlike prediction-time auto-building, this rebuilds every component.

        Parameters
        ----------
        **build_params : keyword arguments
            Component parameters to set before building.

        Returns
        -------
        self
        """
        self.set_params(build_params)
        if self.has_space:
            self.spatial.build()
        if self.has_time:
            self.temporal.build()
        return self

    def _build_stale(self):
        """Build only components whose cached state is invalid.

        This avoids rebuilding an unchanged spatial stage during temporal parameter
        sweeps, and vice versa.
        """
        if self.has_space and not self.spatial.is_built:
            self.spatial.build()
        if self.has_time and not self.temporal.is_built:
            self.temporal.build()

    def predict_percept(self, source, t_percept=None, gaze=None, vmax=None,
                        vmin=0):
        """Predict a percept.

        Parameters
        ----------
        source : stimulus source or :py:class:`~pulse2percept.vision.Scene`
            What is presented to the device: anything accepted by
            :py:meth:`~pulse2percept.implants.ProsthesisSystem.prepare_stim`, or a
            visual scene.
        t_percept : float or array-like, optional
            Output times in ``time_unit``. Unitful times are accepted.
        gaze : (x, y) or (n_frames, 2), optional
            Scene location falling on the fovea, in degrees of visual angle.
            Requires ``source`` to be a scene.
        vmax : float, optional
            Percept brightness mapped to white when composing a scene with a
            scotoma. Required for scotoma composition.
        vmin : float, optional
            Percept brightness mapped to black for scotoma composition.

        Returns
        -------
        percept : :py:class:`~pulse2percept.percepts.Percept` or None
            Brightness percept for ordinary prediction. For a scene with a scotoma,
            returns an RGB percept on the scene pixel grid.

        .. versionchanged:: 0.11.0
            ``source`` is now the presented stimulus or scene rather than an implant
            carrying a stimulus.
        """
        # Scene sampling depends on the current spatial build:
        self._build_stale()
        if not isinstance(source, Scene):
            for name, value in (('gaze', gaze), ('vmax', vmax)):
                if value is not None:
                    raise ValueError(
                        f"'{name}' says where an implanted eye is looking in "
                        f"a scene, and this prediction is not about one. Pass "
                        f"a Scene to place one.")
            if vmin != 0:
                raise ValueError("'vmin' maps a percept onto a display, which "
                                 "only happens for a scene with a scotoma.")
            return self._predict_percept(self._prepared(source), t_percept)
        resp = self._predict_percept(_scene_stim(self, source, gaze),
                                     t_percept)
        if source.scotoma is None or resp is None:
            # Nothing is lost, so there is nothing to compose the percept
            # into: what the implant produces is the whole answer.
            return resp
        return source._compose(resp, vmax, vmin=vmin, gaze=gaze)

    def _prepared(self, source):
        """Prepare a source for the bound implant.

        Temporal-only models have no implant and return the source unchanged.
        """
        if not self.has_space:
            return source
        self.spatial._require_implant()
        return self.implant.prepare_stim(source)

    def _predict_percept(self, stim, t_percept=None):
        """Predict the percept a prepared stimulus produces"""
        self._build_stale()
        # The sub-models normalize too; doing it here as well keeps the error
        # message below reading in plain milliseconds:
        t_percept = as_value(t_percept, self.time_unit, 't_percept')
        if stim is None or (not self.has_space and not self.has_time):
            # Nothing to see here:
            return None
        # Spatial-only models validate the spatial view, not the waveform.
        _require_stim_dimension(
            self, _spatial_input(stim) if self.has_space and not self.has_time
            else stim)
        # `_has_time_axis`, not `stim.time`: whether there is a time axis is a
        # question a stimulus can answer from its structure, and asking it for
        # the axis itself would generate the waveform behind it.
        has_time_axis = _has_time_axis(stim)
        if not has_time_axis and t_percept is not None:
            raise ValueError(f"Cannot calculate temporal response at times "
                             f"t_percept={t_percept}, because stimulus/percept does not "
                             f"have a time component.")

        if self.has_space and self.has_time:
            # Custom temporal combiners need the structured stimulus:
            combine = getattr(self.spatial, '_combine_temporal', None)
            resp = self.spatial._predict_prepared(
                stim if combine is not None else _delivered(stim),
                t_percept=None)
            if has_time_axis:
                if resp.time is None and combine is not None:
                    # Allow a spatial model to define custom temporal
                    # combination for a timeless intermediate percept:
                    resp = combine(resp, self.temporal, stim, t_percept)
                else:
                    # Then pass that to the temporal model, which will output
                    # at all `t_percept` time steps:
                    resp = self.temporal.predict_percept(resp,
                                                         t_percept=t_percept)
        elif self.has_space:
            resp = self.spatial._predict_prepared(stim, t_percept=t_percept)
        elif self.has_time:
            resp = self.temporal.predict_percept(stim, t_percept=t_percept)
        return resp

    @property
    def has_space(self):
        """Returns True if the model has a spatial component"""
        return self.spatial is not None

    @property
    def has_time(self):
        """Returns True if the model has a temporal component"""
        return self.temporal is not None

    @property
    def is_built(self):
        """Returns True if the ``build`` model has been called"""
        _is_built = True
        if self.has_space:
            _is_built &= self.spatial.is_built
        if self.has_time:
            _is_built &= self.temporal.is_built
        return _is_built
