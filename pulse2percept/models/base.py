""":py:class:`~pulse2percept.models.BaseModel`,
   :py:class:`~pulse2percept.models.Model`,
   :py:class:`~pulse2percept.models.NotBuiltError`,
   :py:class:`~pulse2percept.models.SpatialModel`,
   :py:class:`~pulse2percept.models.TemporalModel`"""
import warnings
from abc import ABCMeta, abstractmethod
from copy import deepcopy, copy
import numpy as np
import multiprocessing
from scipy.ndimage import gaussian_filter1d

from ..implants import ProsthesisSystem
from ..stimuli import ImageStimulus, Stimulus, VideoStimulus
from ..stimuli.base import _describe_unit, _has_time_axis
from ..percepts import Percept
from ..topography import Curcio1990Map, Grid2D, RetinalMap
from ..units import (DimensionMismatchError, Quantity, Unit, as_value, dva, ms,
                     um, uA)
from ..vision import Scene
from ..utils import (PrettyPrint, FreezeError, Frozen, Parametrized,
                     deprecated_alias, warn_deprecated_params,
                     rename_deprecated_params)
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
    # An encoder records its frame clock in milliseconds (see
    # `pulse2percept.stimuli.StimulusEncoder`), and what comes back has to be
    # in the calling model's own time unit, since that is what it compares
    # `dt` and `t_percept` in. A no-op for every model p2p ships:
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
    if stim.unit.dimension in {unit.dimension for unit in accepted}:
        return
    expected = ' or '.join(_describe_unit(unit) for unit in accepted)
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
    """Return the stimulus as the pulse train the electrodes deliver.

    Strips an encoded stimulus of its frame-level modulation view, so that
    the spatial stage reads the waveform a temporal stage then integrates.
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
    """What the scene delivers to the bound implant's electrodes"""
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
            "turning a gray level into current. Give the implant an "
            "'encoder' (e.g. an AmplitudeEncoder) to say how.")
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
    # Preprocessing already ran, on the picture rather than on the values it
    # samples to; everything else `prepare_stim` does is still wanted, so a
    # stand-in with that one setting off runs it. A shallow copy, because the
    # caller's implant is not this prediction's to change:
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


class NotBuiltError(ValueError, AttributeError):
    """Exception class used to raise if model is used before building

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """


#: Declared parameter names per model class, so that ``__setattr__`` below
#: does not rebuild the default dict on every assignment.
_declared = {}


def _declared_params(model):
    """The names ``get_default_params`` declares, cached per class"""
    cls = type(model)
    names = _declared.get(cls)
    if names is None:
        names = _declared[cls] = frozenset(model.get_default_params())
    return names


def _unchanged(before, after):
    """Whether re-assigning a parameter left it at the same value

    Falls back to "changed" for anything that cannot be compared, which costs
    at most one rebuild.
    """
    if before is after:
        return True
    try:
        return bool(np.all(before == after))
    except Exception:
        return False


def _invalidating(set_attr):
    """Wrap ``__setattr__`` so a new parameter value un-builds the model.

    Which parameters a build depends on is not worth enumerating: the
    expensive ones are the point of building, and re-deriving a grid because
    ``thresh_percept`` moved is cheaper than being wrong. Assignments that
    change nothing keep the build, so ``model.rho = model.rho`` is free.
    """

    def __setattr__(self, name, value):
        if _is_constructing(self) or name not in _declared_params(self):
            set_attr(self, name, value)
            return
        before = getattr(self, name, None)
        set_attr(self, name, value)
        if not _unchanged(before, getattr(self, name, None)):
            object.__setattr__(self, '_is_built', False)

    return __setattr__


class BaseModel(Parametrized, metaclass=ABCMeta):
    """Abstract base class for all models

    Adds the build workflow on top of
    :py:class:`~pulse2percept.utils.Parametrized`, which supplies the
    parameter, pretty-printing, equality and deep-copy machinery:

    *  Build a model (via ``build``) and flip the ``is_built`` switch

    .. versionchanged:: 0.11.0

        Building is automatic. ``predict_percept`` builds a model that is not
        built yet, and giving any parameter a new value un-builds it, so
        ``model.rho = 200`` takes effect on the next prediction.

    .. versionchanged:: 0.10.0

        Everything other than the build workflow moved to
        :py:class:`~pulse2percept.utils.Parametrized`.

    """

    #: A new parameter value invalidates the build; see ``_invalidating``.
    __setattr__ = _invalidating(Parametrized.__setattr__)

    # The units a model's numerical implementation works in. p2p converts a
    # unitful argument to these at the API boundary and hands the kernels
    # ordinary numbers, so these are a statement about what the numbers *are*,
    # not a setting to change: a model that wanted different ones would have to
    # be written against them throughout. A model uses whichever apply to it --
    # a purely temporal model has no use for `space_unit`.
    #
    # Their purpose is to spare a model's internals from knowing where the
    # canonical units are fixed. `earray.coordinates(self.space_unit)` says
    # what it needs; `[e.x for e in ...]` merely happens to be right.

    #: The unit stimulus values are expressed in
    stimulus_unit = uA
    #: Additional stimulus units accepted by this model
    extra_stimulus_units = ()
    #: The unit spatial coordinates are expressed in
    space_unit = um
    #: The unit time is expressed in
    time_unit = ms

    def __init__(self, **params):
        """BaseModel constructor

        Parameters
        ----------
        **params : optional keyword arguments
            All keyword arguments must be listed in ``get_default_params``
        """
        super().__init__(**params)
        # This flag will be flipped once the ``build`` method was called
        self._is_built = False

    def _build(self):
        """Customize the building process by implementing this method"""
        pass

    def _stim_values(self, stim):
        """The stimulus values a kernel consumes, in ``stimulus_unit``

        One half of the numerical boundary (``_electrode_coords`` and
        ``_stim_times`` are the others). A model's implementation asks for
        what it consumes, so that declaring a different ``stimulus_unit``
        actually delivers different numbers rather than silently leaving the
        kernel to remember a factor of a thousand.

        For every model p2p ships this is the identity: ``stimulus_unit`` is
        microamps and :py:class:`~pulse2percept.stimuli.Stimulus`
        canonicalizes to microamps, so ``values`` hands back the stored array
        without touching it.

        A :py:class:`~pulse2percept.percepts.Percept` is passed through: a
        temporal model is applied to one as readily as to a stimulus, and
        brightness is model output rather than a physical quantity.
        """
        if not isinstance(stim, Stimulus):
            return stim.data
        _require_stim_dimension(self, stim)
        return stim.values(self.stimulus_unit)

    def _stim_times(self, stim):
        """The stimulus time axis a kernel consumes, in ``time_unit``

        The time counterpart of :py:meth:`_stim_values`. A stimulus stores
        milliseconds; a model that declared some other ``time_unit`` gets its
        own.

        A :py:class:`~pulse2percept.percepts.Percept` is converted the same
        way. Its *values* are brightness and stay exactly as they are, but its
        time axis is a physical quantity like any other, and this is the
        boundary a spatial model's output crosses on its way into a temporal
        model: the two need not count in the same unit.
        """
        if not isinstance(stim, (Stimulus, Percept)):
            return stim.time
        return stim.times(self.time_unit)

    def _to_stim_time(self, t, stim):
        """Express a model-side time in the stimulus' own unit

        ``t_percept`` arrives in ``time_unit`` and is used both to index the
        stimulus (which counts in *its* unit) and to label the percept. This
        is the one conversion that goes the other way.
        """
        if t is None or not isinstance(stim, (Stimulus, Percept)) \
                or stim.time_unit == self.time_unit:
            return t
        return Quantity(t, self.time_unit).to_value(stim.time_unit)

    def _electrode_coords(self, earray, stim):
        """Where the electrodes a stimulus names are, ready for a kernel

        Returns one contiguous float32 array per axis, expressed in
        ``space_unit``, which is what the Cython kernels take.

        Asks for ``stim.electrodes`` rather than for the array as it stands,
        because a stimulus need not name every electrode of the implant and
        need not name them in array order: what the kernel needs is one
        coordinate per *row of the stimulus*.

        Parameters
        ----------
        earray : :py:class:`~pulse2percept.implants.ElectrodeArray`
            The electrode array to look the coordinates up in.
        stim : :py:class:`~pulse2percept.stimuli.Stimulus`
            The stimulus whose electrodes to look up.

        Returns
        -------
        x, y, z : (n_electrodes,) np.ndarray of np.float32

        """
        xyz = earray.coordinates(self.space_unit, electrodes=stim.electrodes)
        return tuple(np.ascontiguousarray(xyz[:, i], dtype=np.float32)
                     for i in range(3))

    def build(self, **build_params):
        """Build the model

        Every model must have a ```build`` method, which is meant to perform
        all expensive one-time calculations. You must call ``build`` before
        calling ``predict_percept``.

        .. important::

            Don't override this method if you are building your own model.
            Customize ``_build`` instead.

        Parameters
        ----------
        build_params : additional parameters to set
            You can overwrite parameters that are listed in
            ``get_default_params``. Trying to add new class attributes outside
            of that will cause a ``FreezeError``.
            Example: ``model.build(param1=val)``

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
            # The implant is the device a model is pointed at, not part of the
            # model's own state, so a copy describes the same physical
            # implant. Seeding the memo is what shares it: it also keeps
            # `copy.implant is model.implant`, and stops the rebuild below
            # from being invalidated by a freshly copied device.
            memodict.setdefault(id(implant), implant)
        copied = super().__deepcopy__(memodict)
        if self.is_built:
            copied.build()
        return copied


class SpatialModel(BaseModel, metaclass=ABCMeta):
    """Abstract base class for all spatial models

    Provides basic functionality for all spatial models:

    *  ``build``: builds the spatial grid used to calculate the percept.
       You can add your own ``_build`` method (note the underscore) that
       performs additional expensive one-time calculations.
    *  ``predict_percept``: predicts the percepts based on an implant/stimulus.
       Don't customize this method - implement your own ``_predict_spatial``
       instead (see below).
       A user must call ``build`` before calling ``predict_percept``.

    To create your own spatial model, you must subclass ``SpatialModel`` and
    provide an implementation for:

    *  ``_predict_spatial``: This method should accept an ElectrodeArray as well
       as a Stimulus, and compute the brightness at all spatial coordinates of
       ``self.grid``, returned as a 2D NumPy array (space x time).

       .. note ::

           The ``_`` in the method name indicates that this is a private method,
           meaning that it should not be called by the user. Instead, the user
           should call ``predict_percept``, which in turn will call
           ``_predict_spatial``.
           The same logic applies to ``build`` (called by the user; don't touch)
           and ``_build`` (called by ``build``; customize this instead).

    In addition, you can customize the following:

    *  ``__init__``: the constructor can be used to define additional
       parameters (note that you cannot add parameters on-the-fly)
    *  ``get_default_params``: all settable model parameters must be listed by
       this method
    *  ``_build`` (optional): a way to add one-time computations to the build
       process

    .. versionadded:: 0.6

    .. note ::

        You will not be able to add more parameters outside the constructor;
        e.g., ``model.newparam = 1`` will lead to a ``FreezeError``.

    Notes
    -----
    *  ``xrange`` and ``yrange`` say which patch of the **visual field** to
       simulate, in degrees of visual angle, and may be given as plain numbers
       or as unitful quantities (e.g. ``xrange=(-12 * dva, 12 * dva)``).

       .. versionchanged:: 0.10.0

          On a retinal model they may also be given as a physical extent
          (e.g. ``xrange=(-4 * mm, 4 * mm)``), which the model's ``vfmap``
          resolves into the visual field range that piece of retina covers.
          Each range is converted along its own retinal meridian, and the grid
          that results is rectangular and uniformly sampled in dva exactly as
          it always was -- under a nonlinear map it is therefore not the image
          of a retinal rectangle. The range is stored in dva either way, so
          changing ``vfmap`` afterwards does not reinterpret it. This is
          shorthand, not a unit conversion: ``step`` has no such spelling,
          since a grid spaced evenly on the retina is a different grid from
          one spaced evenly in the visual field.

    .. seealso ::

        *  `Basic Concepts > Computational Models > Building your own model
           <topics-models-building-your-own>`
    """

    #: ``n_jobs`` is an alias for ``n_threads``; see ``_n_jobs_alias``.
    n_jobs = _n_jobs_alias()

    #: ``step`` used to be called ``xystep``. The old name still reads and
    #: writes ``step``, with a ``DeprecationWarning``:
    xystep = deprecated_alias('step', deprecated_version='0.10.0',
                              removed_version='0.11.0')

    def __init__(self, **params):
        # `vfmap` first: `xrange`/`yrange` may be given as a retinal extent,
        # which is resolved through the map as it is assigned. See
        # `_vfmap_first`.
        super().__init__(**_vfmap_first(params))
        self.grid = None

    @property
    def implant(self):
        """The prosthesis system this model predicts percepts for

        A spatial model says where in the visual field the electrodes of a
        particular device are seen, so the device is model context rather than
        trial input: it is named once, and
        :py:meth:`~pulse2percept.models.SpatialModel.predict_percept` is then
        given the stimulus. Rebinding invalidates the build.

        .. versionadded:: 0.11.0
        """
        return getattr(self, '_implant', None)

    @implant.setter
    def implant(self, implant):
        """Implant setter (called upon ``self.implant = implant``)"""
        if implant is not None and not isinstance(implant, ProsthesisSystem):
            raise TypeError(f"'implant' must be a ProsthesisSystem object, "
                            f"not {type(implant)}.")
        if implant is not getattr(self, '_implant', None):
            # A build describes one device's geometry, so it does not survive
            # being pointed at another:
            self._is_built = False
        self._implant = implant

    def _require_implant(self):
        """Require the implant a spatial prediction is about"""
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
        """Convert a unitful parameter into the unit it is stored in

        Extends the generic conversion (see
        :py:meth:`~pulse2percept.utils.Parametrized._normalize_param_value`)
        with the one parameter whose unit is not the whole story: a *physical*
        ``xrange``/``yrange`` is shorthand for the visual field range that
        extent covers, which only the model's visual field map can say.
        Everything else, ``step`` included, is converted as usual.
        """
        if name in ('xrange', 'yrange') and _length_valued(value):
            return self._retinal_range_to_dva(name, value)
        return super()._normalize_param_value(name, value)

    def _retinal_range_to_dva(self, name, value):
        """Resolve a retinal extent into the visual field range it spans

        The model simulates a patch of the *visual field*, sampled uniformly in
        degrees of visual angle, and that is what ``xrange``/``yrange`` are and
        stay. Giving them in microns says which patch by naming the piece of
        retina it lands on; the map turns that into degrees here, once, and
        what is stored afterwards is an ordinary pair of dva. Changing
        ``vfmap`` later therefore does not reinterpret a range that has already
        been resolved -- the same rule an implant's coordinates follow.

        Each range is converted along its own retinal meridian: ``xrange``
        through ``ret_to_dva(x, 0)`` and ``yrange`` through
        ``ret_to_dva(0, y)``. The grid built from the result is rectangular and
        uniformly sampled in dva, exactly as it is for a range given in degrees,
        so under a nonlinear map it is not the image of a retinal rectangle.
        Naming retinal lengths says how far to simulate; it does not change what
        the grid is uniform in.

        This is deliberately not a unit conversion, and
        :py:class:`~pulse2percept.units.Quantity` will not do it: how far a
        degree reaches on tissue is what a visual field map is for. It is also
        deliberately not offered for ``step``, since a grid spaced evenly on
        the retina is not the same grid as one spaced evenly in the visual
        field, and only the latter is what :py:class:`Grid2D` builds.

        Parameters
        ----------
        name : {'xrange', 'yrange'}
            Which of the two ranges is being assigned.
        value : (min, max)
            The extent, as a pair of lengths.

        Returns
        -------
        (min_dva, max_dva) : tuple of float
            The same extent in degrees of visual angle, in increasing order.

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
            # The device whose electrodes this model places in the visual
            # field. Required before building or predicting:
            'implant': None,
            # We will be simulating a patch of the visual field (xrange/yrange
            # in degrees of visual angle), at a given spatial resolution (step
            # size):
            'xrange': (-15, 15),  # dva
            'yrange': (-15, 15),  # dva
            'step': 0.25,  # dva
            'grid_type': 'rectangular',
            # Below threshold, percept has brightness zero:
            'thresh_percept': 0,
            # An electrode whose Gaussian current spread at a point has fallen
            # below this is skipped for that point:
            'min_current_spread': 1e-8,
            # Visual field map (retinotopy) to be used:
            'vfmap': Curcio1990Map(),
            # Number of gray levels to use in the percept:
            'n_gray': None,
            # Salt-and-pepper noise on the output:
            'noise': None,
            # True: print status messages, 0: silent
            'verbose': True,
            # default to 2d model. 3d models should override this
            'ndim' : [2],
            # Number of OpenMP threads. `n_jobs` is an alias that writes
            # through to `n_threads`, so it has to be applied *after* it:
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
        """Squared distance at which an electrode stops contributing

        Models with a Gaussian current spread ``exp(-r^2 / (2 rho^2))`` spend
        most of their time on (point, electrode) pairs whose Gaussian has
        already underflowed the result. This converts ``min_current_spread``
        into the squared distance at which that happens, for the spatial
        kernels to compare against.

        .. note::

            The default ``min_current_spread`` of 1e-8 corresponds to a radius
            of about 6.1 ``rho``. The kernels compare the Gaussian against the
            cutoff *before* scaling it by the stimulus amplitude and summing
            over electrodes, so what is dropped at a point is
            ``sum_i gauss_i * amplitude_i`` over the electrodes outside the
            cutoff.

            Set it to 0 to sum over every electrode no matter how distant and get
            the exact result, or raise it to trade more accuracy for speed.

        Parameters
        ----------
        rho : float
            The model's current-spread decay constant (microns).

        Returns
        -------
        cutoff_r2 : np.float32
            Squared distance (microns^2), or ``inf`` if no cutoff applies.
        """
        min_spread = self.min_current_spread
        if min_spread is None or min_spread <= 0:
            return np.float32(np.inf)
        if min_spread >= 1:
            raise ValueError(f"min_current_spread must be smaller than 1 (or "
                             f"0 to disable the cutoff), not {min_spread}.")
        return np.float32(-2.0 * rho ** 2 * np.log(min_spread))

    def build(self, **build_params):
        """Build the model

        Performs expensive one-time calculations, such as building the spatial
        grid used to predict a percept. You must call ``build`` before
        calling ``predict_percept``.

        .. important::

            Don't override this method if you are building your own model.
            Customize ``_build`` instead.

        Parameters
        ----------
        build_params: additional parameters to set
            You can overwrite parameters that are listed in
            ``get_default_params``. Trying to add new class attributes outside
            of that will cause a ``FreezeError``.
            Example: ``model.build(param1=val)``

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
        """Customized spatial response

        Called by the user from ``predict_percept`` after error checking.

        Parameters
        ----------
        earray: :py:class:`~pulse2percept.implants.ElectrodeArray`
            A valid electrode array.
        stim : :py:meth:`~pulse2percept.stimuli.Stimulus`
            A valid stimulus with a 2D data container (n_electrodes, n_time).

        Returns
        -------
        percept: np.ndarray
            A 2D NumPy array that has the same dimensions as the input stimulus
            (n_electrodes, n_time).
        """
        raise NotImplementedError

    def _postprocess_spatial(self, resp):
        """Hook for spatial-model postprocessing."""
        return resp

    def predict_percept(self, source, t_percept=None):
        """Predict the spatial response

        .. important::

            Don't override this method if you are creating your own model.
            Customize ``_predict_spatial`` instead.

        .. note::

            **This method reads modulation frames, not pulses.** Where the
            prepared stimulus was produced by the implant's own
            :py:class:`~pulse2percept.stimuli.StimulusEncoder`, what is read
            is one amplitude per electrode per frame of the source video,
            rather than the pulse train delivering it.

            A pulse train says *when* current flows, and a raster says which
            electrodes are allowed to flow together. Both are facts about
            time, and a model with no temporal component can express neither:
            handed the train, it would report the stimulus one instant at a
            time, so an encoded image would come back as a sequence of raster
            slots instead of as the image. So an image gives one percept
            frame, and a video one percept frame per video frame.

            :py:class:`~pulse2percept.models.Model` hands its spatial stage
            the delivered pulse train instead whenever it also has a temporal
            stage, since integrating those pulses is what that stage is for.

        .. versionchanged:: 0.11.0
            Takes the stimulus source rather than an implant; the implant is
            the one this model is bound to.

        Parameters
        ----------
        source : :py:class:`~pulse2percept.stimuli.Stimulus` source type
            What is presented to the device. Anything
            :py:meth:`~pulse2percept.implants.ProsthesisSystem.prepare_stim`
            accepts, including an image or a video for the implant's encoder.
        t_percept: float or list of floats, optional
            The time points at which to output a percept, counted in this
            model's :py:attr:`~pulse2percept.models.BaseModel.time_unit`
            (milliseconds, for every model p2p ships).
            If None, the time points of the stimulus being read are used --
            the frame times, for an encoded stimulus.
            May be given as a unitful quantity (e.g. ``[0, 20] * ms``); see
            :py:mod:`pulse2percept.units`.

        Returns
        -------
        percept: :py:class:`~pulse2percept.models.Percept`
            A Percept object whose ``data`` container has dimensions Y x X x T,
            and whose time axis is labelled in ``time_unit``.
            Will return None if ``source`` is None or empty.

        """
        if not self.is_built:
            self.build()
        return self._predict_prepared(self.implant.prepare_stim(source),
                                      t_percept=t_percept)

    def _predict_prepared(self, stim, t_percept=None):
        """Predict the spatial response to an already-prepared stimulus

        The half of :py:meth:`predict_percept` that runs after the implant's
        input pipeline, so that a composite model can prepare once and hand
        the same stimulus to both stages. Models that replace the whole
        spatial prediction rather than customizing ``_predict_spatial``
        override this.
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
            # A single-frame source (an image) modulates the electrodes to one
            # steady thing, so there are no times to ask about even though the
            # pulse train delivering it does have a time axis:
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
                stim = Stimulus(
                    np.asarray(stim[:, at]).reshape((-1, n_time)),
                    electrodes=stim.electrodes, time=at
                )._inherit_units(stim)._inherit_metadata(stim)
                # find unique stimulus points
                _, t_unique, inverse = np.unique(stim.data.T, axis=0,
                                                 return_index=True,
                                                 return_inverse=True)
                # np.unique orders what it returns by stimulus value, not by
                # time, so `t_unique` comes back shuffled with respect to the
                # time axis. Sort it back into chronological order and remap
                # `inverse` to match, so that the de-duplicated stimulus below
                # is built with strictly increasing time. The percept is
                # correct either way -- `inverse` undoes whatever order was
                # used -- but a Stimulus with shuffled time warns, and any
                # model that looks at `stim.time` would read it wrong.
                # np.ravel: NumPy has changed the shape of `inverse` for
                # axis-wise calls between 2.x releases, and the remap below
                # needs it flat.
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
                stim_unique = Stimulus(
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
    """Abstract base class for all temporal models

    Provides basic functionality for all temporal models:

    *  ``build``: builds the model in order to calculate the percept.
       You can add your own ``_build`` method (note the underscore) that
       performs additional expensive one-time calculations.
    *  ``predict_percept``: predicts the percepts based on an implant/stimulus.
       You can add your own ``_predict_temporal`` method to customize this
       step. A user must call ``build`` before calling ``predict_percept``.

    To create your own temporal model, you must subclass ``SpatialModel`` and
    provide an implementation for:

    *  ``_predict_temporal``: a method that accepts either a
       :py:class:`~pulse2percept.stimuli.Stimulus` or a
       :py:class:`~pulse2percept.percepts.Percept` object and a list of time
       points at which to calculate the resulting percept, returned as a 2D
       NumPy array (space x time).

    In addition, you can customize the following:

    *  ``__init__``: the constructor can be used to define additional
       parameters (note that you cannot add parameters on-the-fly)
    *  ``get_default_params``: all settable model parameters must be listed by
       this method
    *  ``_build`` (optional): a way to add one-time computations to the build
       process

    Parameters
    ----------
    dt : float, optional
        Sampling time step of the simulation (ms)
    thresh_percept : float, optional
        Below threshold, the percept has brightness zero.
    reduce : {'last', 'peak'}, optional
        How a percept time point summarizes the interval since the previous
        one, when ``predict_percept`` picks the output times itself (that is,
        when ``t_percept`` is None). ``'last'`` reports the brightness at the
        instant the interval ended, which is what every version before 0.10.0
        did and what the published models still default to. ``'peak'`` reports
        the highest brightness reached over the interval.

        Peak is worth reaching for because electrical stimulation is pulsatile:
        the brightness an interval produces rises and falls within it, so the
        closing instant says more about where in the pulse cycle it fell than
        about the interval. Peak rather than mean because what a pulse train
        produces is a flash, and averaging over the gaps that follow would
        scale every interval by its duty cycle instead.
        :py:class:`~pulse2percept.models.FadingTemporal` defaults to it.

        How exactly it is computed depends on the model. One that sets
        ``_reduces_intervals`` tracks the peak across every ``dt`` step inside
        its own integrator, which is exact at any output rate. Any other model
        is sampled at several instants per interval instead, which cannot catch
        a transient shorter than the resulting step; see ``_FRAME_SUBSAMPLES``.

        Naming ``t_percept`` overrides this: an explicit time point is a
        request for that instant, and is always answered with the brightness
        there.

        .. versionadded:: 0.10.0
    n_threads : int, optional
        Number of CPU threads to use during parallelization using OpenMP.
        Defaults to max number of user CPU cores.

    .. versionadded:: 0.6

    .. note ::

        You will not be able to add more parameters outside the constructor;
        e.g., ``model.newparam = 1`` will lead to a ``FreezeError``.

    .. seealso ::

        *  `Basic Concepts > Computational Models > Building your own model
           <topics-models-building-your-own>`
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
            'dt': 0.005,  # Simulation time step (ms)
            'thresh_percept': 0,
            'reduce': 'last',  # How automatically chosen intervals are summarized
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
        """Customized temporal response

        Called by the user from ``predict_percept`` after error checking.

        Parameters
        ----------
        stim : :py:meth:`~pulse2percept.stimuli.Stimulus`
            A valid stimulus with a 2D data container (n_electrodes, n_time).
        t_percept : list of floats
            The time points at which to output a percept (ms).

        Returns
        -------
        percept: np.ndarray
            A 2D NumPy array (space x time) that specifies the percept at each
            spatial location and time step.

        Notes
        -----
        A model that can summarize an interval rather than sample an instant
        takes a third argument, ``reduce`` ('peak' or 'last'), and sets
        ``_reduces_intervals = True``. ``predict_percept`` only passes the
        argument to models that advertise it, so an override with the
        two-argument signature above keeps working.
        """
        raise NotImplementedError

    def predict_percept(self, stim, t_percept=None):
        """Predict the temporal response

        .. important ::

            Don't override this method if you are creating your own model.
            Customize ``_predict_temporal`` instead.

        Parameters
        ----------
        stim: : py: class: `~pulse2percept.stimuli.Stimulus` or
               : py: class: `~pulse2percept.models.Percept`
            Either a Stimulus or a Percept object. The temporal model will be
            applied to each spatial location in the stimulus/percept.
        t_percept : float or list of floats, optional
            The time points at which to output a percept, counted in this
            model's :py:attr:`~pulse2percept.models.BaseModel.time_unit`
            (milliseconds, for every model p2p ships). May be given as a
            unitful quantity (e.g. ``[0, 20] * ms``); see
            :py:mod:`pulse2percept.units`.
            If None, the percept will be output once per frame of the video the
            stimulus was encoded from, or failing that once every 20 ms (50 Hz
            frame rate), starting at zero and stopping at the last frame
            boundary the stimulus reaches.

            .. note ::

                A stimulus shorter than a single frame still gets one frame,
                whose time point therefore falls after the end of the
                stimulus. That is the only case in which the output runs past
                the stimulus, and it is what makes a brief pulse visible at
                all: reporting it only at t=0 would describe it before it had
                had any effect. Name ``t_percept`` to be reported at
                particular instants instead.

        Returns
        -------
        percept : :py:class:`~pulse2percept.models.Percept`
            A Percept object whose ``data`` container has dimensions Y x X x T.
            Will return None if ``stim`` is None.

        Notes
        -----
        *  If a list of time points is provided for ``t_percept``, the values
           will automatically be sorted.

        *  Naming ``t_percept`` asks for the brightness *at those instants*.
           Leaving it None asks the model to pick the output times, and
           ``reduce`` then says what each point reports about the interval
           leading up to it -- the closing instant, or the peak reached over
           it.

           The distinction matters because electrical stimulation is pulsatile.
           A 20 Hz train of 0.46 ms biphasic pulses drives brightness in
           sub-millisecond transients at a 1.8% duty cycle, so an instant
           sampled from it is almost always an instant between pulses. Worse,
           the sampling phase walks: against a 29.97 fps video the frame
           (33.37 ms) and the pulse period (50 ms) are incommensurate, so which
           electrodes a frame catches drifts from frame to frame. Under a
           raster, where each group pulses in its own slot, that shows up as
           groups appearing in the wrong order or not at all.

        .. versionchanged:: 0.10.0

            Output times chosen by the model can summarize their interval
            instead of sampling its final instant. See ``reduce``.

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
            # Nobody asked for a particular instant, so the output times are
            # this model's to choose -- and having chosen them it owes a
            # summary of each interval rather than a sample of one instant out
            # of it. `reduce` says which summary; see the docstring.
            reduce = self.reduce
            if reduce not in ('peak', 'last'):
                raise ValueError(f"'reduce' must be 'peak' or 'last', not "
                                 f"{self.reduce!r}.")
            # A stimulus that came out of an encoder knows the frame rate of
            # the video behind it, and that is the rate worth reporting at:
            # one percept frame per video frame. Failing that, output at a
            # 50 Hz frame rate, starting at zero and stopping at the last
            # frame boundary the stimulus reaches:
            frames = _frame_clock(stim, self.dt, unit=self.time_unit)
            if frames is None:
                # One frame every 20 ms is a 50 Hz frame rate no matter what
                # this model counts in, so the interval is converted rather
                # than written down as the number 20. `nextafter` is what
                # makes `arange`'s half-open end include a stimulus that ends
                # exactly on a frame boundary and stop short of one that does
                # not, without inventing a unit of time to add to it.
                #
                # The floor at `frame_dur` is the one case where the output
                # does run past the end of the stimulus, and it is deliberate:
                # a stimulus shorter than a single frame would otherwise be
                # reported only at t=0, before it had had any effect at all.
                # Brightness outlives the stimulus that caused it, so the one
                # frame containing it is what is worth reporting; ask for
                # something else by naming `t_percept`. Unlike the millisecond
                # of slack this replaced, a frame means the same thing in any
                # `time_unit`.
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
            # Collapse each interval's samples down to the largest. Peak, not
            # mean, because what a pulse train produces is a flash, and
            # averaging it over the gaps that follow would scale every interval
            # by its duty cycle instead:
            resp = np.maximum.reduceat(resp, sub_idx, axis=-1)
            t_percept = t_out
        # A temporal model rewrites a spatial percept frame by frame; it does
        # not move it in the visual field, so it hands the grid back on:
        return Percept(resp, space=None, time=t_percept,
                       time_unit=self.time_unit,
                       metadata={'stim': stim})._inherit_space(stim)

    def _warn_if_blank(self, stim, resp):
        """Point out a percept that came out blank for a polarity reason

        A stimulus of the wrong sign is not an error -- the model integrates it
        and rectifies the result away -- so it otherwise looks exactly like a
        stimulus that was simply too weak to see. Assigning a grayscale image
        or video that was never encoded lands here, because gray levels are
        nonnegative and most temporal models are driven by cathodic current.
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
    """Computational model

    To build your own model, you can mix and match spatial and temporal models
    at will.

    For example, to create a model that combines the scoreboard model described
    in [Beyeler2019]_ with the temporal model cascade described in
    [Nanduri2012]_, use the following:

    .. code-block :: python

        model = Model(spatial=ScoreboardSpatial(),
                      temporal=Nanduri2012Temporal())

    .. seealso ::

        *  `Basic Concepts > Computational Models > Building your own model
           <topics-models-building-your-own>`

    .. versionadded:: 0.6

    Parameters
    ----------
    spatial: :py:class:`~pulse2percept.models.SpatialModel` or None
        The spatial model, which decides where in the visual field a stimulus
        is seen. May be given as a class, which is then constructed from
        ``params``.
    temporal: :py:class:`~pulse2percept.models.TemporalModel` or None
        The temporal model, which decides how the response evolves over time.
        May be given as a class, which is then constructed from ``params``.
    implant: :py:class:`~pulse2percept.implants.ProsthesisSystem`, optional
        The device this model predicts percepts for. Stored on the spatial
        model, which is what needs it, and read back through ``model.implant``;
        there is one implant, not a copy per component. A temporal-only model
        never sees an electrode, and so does not take one.

        .. versionadded:: 0.11.0

    **params:
        Additional keyword arguments(e.g., ``verbose=True``) to be passed to
        either the spatial model, the temporal model, or both.

    """

    # A composite reports the units of whichever component actually consumes
    # the quantity, because those are the units its own arguments are read in
    # and its own percept is written in. Spelled out here rather than left to
    # `__getattr__`, which answers with a *dict* when both components have the
    # attribute; and each one falls back on the canonical default, because a
    # Model with neither component still has to be able to normalize its
    # arguments. See `BaseModel` for what the three of them mean.

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
        """The unit time is expressed in

        ``t_percept`` is read in, and the resulting
        :py:class:`~pulse2percept.percepts.Percept` is written in, the unit of
        the last stage of the pipeline: the temporal model if there is one,
        the spatial model otherwise. The two need not agree -- a spatial model
        counting in seconds hands its percept to a temporal model counting in
        milliseconds and the time axis is converted on the way across.
        """
        if self.has_time:
            return self.temporal.time_unit
        if self.has_space:
            return self.spatial.time_unit
        return BaseModel.time_unit

    def __init__(self, spatial=None, temporal=None, **params):
        # The implant belongs to the spatial model, which is the only stage
        # that knows what an electrode is. Held back from `params` so that it
        # is not offered to a temporal model that has no such parameter:
        implant = params.pop('implant', None)
        # A sub-model passed as a *class* is constructed from `params` below,
        # and `set_params` then hands the same dict to the resulting instance.
        # Both paths rewrite renamed parameters, so an old name reaching this
        # constructor would otherwise be warned about twice. Settle it once,
        # up front, and let the two paths downstream see only the new name:
        for model in (spatial, temporal):
            if isinstance(model, type):
                params = rename_deprecated_params(
                    type(self).__name__, params,
                    getattr(model, '_renamed_params', {}))
        # Set the spatial model:
        if spatial is not None and not isinstance(spatial, SpatialModel):
            if issubclass(spatial, SpatialModel):
                # User should have passed an instance, not a class:
                spatial = spatial(**params)
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
        """Called when the default attr access fails with an AttributeError

        This method is called when the user tries to access an attribute(e.g.,
        ``model.a``), but ``a`` could not be found(either because it is part
        of the spatial / temporal model or because it doesn't exist).

        Returns
        -------
        attr: any
            Checks both spatial and temporal models and:

            *  returns the attribute if found.
            *  if the attribute exists in both spatial / temporal model,
               returns a dictionary ``{'spatial': attr, 'temporal': attr}``.
            *  if the attribtue is not found, raises an AttributeError.

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
        """Called when an attribute is set

        This method is called when a new attribute is set(e.g.,
        ``model.a=2``). This is allowed in the constructor, but will raise a
        ``FreezeError`` elsewhere.

        ``model.a = X`` can be used as a shorthand to set ``model.spatial.a``
        and / or ``model.temporal.a``.

        """
        if _is_constructing(self):
            # `self.spatial = ...`, and whatever attributes a subclass
            # constructor creates, belong to the composite object. User
            # parameters do not: `set_params` forwards those explicitly, so
            # that `rho` passed to the constructor still reaches the
            # sub-models rather than shadowing them here.
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
        """
        Perform a deep copy of the Model object.

        Parameters
        ----------
        memodict: dict
            Dictionary of objects already copied during the current copying pass.

        Returns
            Deep copy of the object
        -------

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
        result = self.__class__(**attributes)
        # Whatever the constructor made, replace it with our copies. Model
        # parameters (e.g. `rho`) are forwarded to the sub-models by
        # `__setattr__`, so they live in `spatial`/`temporal`, not in
        # `self.__dict__` -- reconstructing from the constructor alone would
        # silently reset them to their defaults. This bypasses
        # `Model.__setattr__`, which outside the constructor forwards
        # attributes to the sub-models; it is the assignment __init__ makes.
        if spatial is not None:
            object.__setattr__(result, 'spatial', spatial)
        if temporal is not None:
            object.__setattr__(result, 'temporal', temporal)
        if self.is_built:
            result.build()
        memodict[id(self)] = result
        return result

    def __eq__(self, other):
        """
        Equality operator for Model.

        Parameters
        ----------
        other: Model
            Model to compare against

        Returns
        -------
        bool:
            True if the compared objects have identical attributes, False otherwise.
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
        """Set model parameters

        This is a convenience function to set parameters that might be part of
        the spatial model, the temporal model, or both.

        Alternatively, you can set the parameter directly, e.g.
        ``model.spatial.verbose = True``.

        .. note::

            If a parameter exists in both spatial and temporal models(e.g.,
            ``verbose``), both models will be updated.

        Parameters
        ----------
        params: dict
            A dictionary of parameters to set.
        """
        # A Model built from *instances* never routes through
        # ``BaseModel.__init__``, so the deprecated names have to be caught
        # here too. Collect both sides first so a parameter deprecated on the
        # spatial *and* temporal model only warns once.
        specs = {}
        renamed = {}
        for model in (self.spatial, self.temporal):
            specs.update(getattr(model, '_deprecated_params', {}))
            renamed.update(getattr(model, '_renamed_params', {}))
        warn_deprecated_params(type(self).__name__, params, specs)
        params = rename_deprecated_params(type(self).__name__, params, renamed)
        # Each parameter is forwarded to the sub-models one at a time, so the
        # order they are applied in is decided here rather than by
        # `SpatialModel.set_params`. See `_vfmap_first`:
        #
        # Forwarding directly rather than via `setattr`: `set_params` also
        # runs from inside `__init__`, where an assignment to `self` would
        # land on the composite object instead of the sub-models.
        for key, val in _vfmap_first(params).items():
            self._set_component_param(key, val)

    def build(self, **build_params):
        """Build the model

        Performs expensive one-time calculations, such as building the spatial
        grid used to predict a percept.

        Parameters
        ----------
        build_params: additional parameters to set
            You can overwrite parameters that are listed in
            ``get_default_params``. Trying to add new class attributes outside
            of that will cause a ``FreezeError``.
            Example: ``model.build(param1=val)``

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

    def predict_percept(self, source, t_percept=None, gaze=None, vmax=None,
                        vmin=0):
        """Predict a percept

        .. important ::

            You must call ``build`` before calling ``predict_percept``.

        Given an ordinary stimulus source, this predicts what the bound
        implant delivers for it (see
        :py:meth:`~pulse2percept.implants.ProsthesisSystem.prepare_stim`).
        Given a :py:class:`~pulse2percept.vision.Scene`, the model is the
        glue: it follows each electrode out through its own ``vfmap`` to the
        place in the scene that electrode sees, hands those values to the
        implant's ``encoder``, and predicts the percept that results.

        If the scene also has a
        :py:class:`~pulse2percept.vision.Scotoma`, the result is what the
        person actually sees: intact native vision outside the lost region,
        and the prosthetic percept inside it, as one RGB percept on the
        scene's own pixel grid.

        .. versionchanged:: 0.11.0

            Takes what is presented to the device -- a stimulus source or a
            scene -- rather than an implant carrying a stimulus. ``gaze``,
            ``vmax`` and ``vmin`` are what a scene needs, and are rejected
            without one.

        Parameters
        ----------
        source : :py:class:`~pulse2percept.stimuli.Stimulus` source type or
                 :py:class:`~pulse2percept.vision.Scene`
            What is presented to the device: anything
            :py:meth:`~pulse2percept.implants.ProsthesisSystem.prepare_stim`
            accepts, or a scene the implanted eye is looking at.
        t_percept: float or list of floats, optional
            The time points at which to output a percept, counted in this
            model's :py:attr:`~pulse2percept.models.BaseModel.time_unit`
            (milliseconds, for every model p2p ships).
            If None, the time points of the prepared stimulus are used.
            May be given as a unitful quantity (e.g. ``[0, 20] * ms``); see
            :py:mod:`pulse2percept.units`.
        gaze : (x, y) or (n_frames, 2), optional
            Where the eye is pointing: the scene location that currently falls
            on the fovea, in degrees of visual angle (e.g. ``(5, 0) * dva``).
            Defaults to the origin. One pair fixates; one pair per frame moves
            the eye between the frames of a video scene. The implant does not
            move when gaze does, and neither does an eye-centered scotoma --
            the scene moves past them. Requires a scene.

            .. versionadded:: 0.11.0

        vmax : float, optional
            The perceived brightness that displays as white. Required when the
            scene has a scotoma, because the result is then a picture rather
            than model output: a percept is in arbitrary units, so nothing
            here can guess the transfer function.

            .. versionadded:: 0.11.0

        vmin : float, optional
            The perceived brightness that displays as black. Brightness maps
            linearly onto [0, 1] between the two, and is clipped outside them.

            .. versionadded:: 0.11.0

        Returns
        -------
        percept: :py:class:`~pulse2percept.models.Percept`
            Without a scene, or with one that has no scotoma: a brightness
            percept whose ``data`` has dimensions Y x X x T, and None if
            ``source`` is None or empty. With a scene that has a scotoma: an
            RGB percept of dimensions Y x X x 3 x T on the scene's pixel grid.

        """
        # Before the scene is sampled and a whole stimulus encoded from it,
        # so that a build the caller never asked for does not happen twice:
        if not self.is_built:
            self.build()
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
        """Run a source through the bound implant's input pipeline

        A temporal-only model has no implant and no electrodes, so what it is
        given is already the stimulus it integrates.
        """
        if not self.has_space:
            return source
        self.spatial._require_implant()
        return self.implant.prepare_stim(source)

    def _predict_percept(self, stim, t_percept=None):
        """Predict the percept a prepared stimulus produces"""
        if not self.is_built:
            self.build()
        # The sub-models normalize too; doing it here as well keeps the error
        # message below reading in plain milliseconds:
        t_percept = as_value(t_percept, self.time_unit, 't_percept')
        if stim is None or (not self.has_space and not self.has_time):
            # Nothing to see here:
            return None
        _require_stim_dimension(self, stim)
        # `_has_time_axis`, not `stim.time`: whether there is a time axis is a
        # question a stimulus can answer from its structure, and asking it for
        # the axis itself would generate the waveform behind it.
        has_time_axis = _has_time_axis(stim)
        if not has_time_axis and t_percept is not None:
            raise ValueError(f"Cannot calculate temporal response at times "
                             f"t_percept={t_percept}, because stimulus/percept does not "
                             f"have a time component.")

        if self.has_space and self.has_time:
            # Need to calculate the spatial response at all stimulus points
            # (i.e., whenever the stimulus changes). The delivered pulse train
            # rather than the modulation behind it: integrating those pulses is
            # what the temporal stage is for.
            resp = self.spatial._predict_prepared(_delivered(stim),
                                                  t_percept=None)
            if has_time_axis:
                combine = getattr(self.spatial, '_combine_temporal', None)
                if resp.time is None and combine is not None:
                    # A spatial model hands over a percept with no time axis,
                    # so the spatial model decides what to do with it:
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
