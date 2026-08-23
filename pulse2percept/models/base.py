""":py:class:`~pulse2percept.models.BaseModel`, 
   :py:class:`~pulse2percept.models.Model`,
   :py:class:`~pulse2percept.models.NotBuiltError`,
   :py:class:`~pulse2percept.models.SpatialModel`,
   :py:class:`~pulse2percept.models.TemporalModel`"""
import sys
import warnings
from abc import ABCMeta, abstractmethod
from copy import deepcopy, copy
import numpy as np
import multiprocessing
from scipy.ndimage import gaussian_filter1d

from ..implants import ProsthesisSystem
from ..stimuli import Stimulus
from ..stimuli.base import _describe_unit, _has_time_axis
from ..percepts import Percept
from ..topography import Curcio1990Map, Grid2D, RetinalMap
from ..units import (DimensionMismatchError, Quantity, Unit, as_value, dva, ms,
                     um, uA)
from ..utils import (PrettyPrint, FreezeError, Parametrized, bisect,
                     deprecated_alias, warn_deprecated_params,
                     rename_deprecated_params)
from ..utils.constants import ZORDER


def _n_jobs_alias():
    """Build the ``n_jobs`` property, an alias for ``n_threads``

    Both names refer to the same OpenMP thread count, and both read and write
    the same storage, so they can never drift apart. ``None`` and ``-1`` mean
    "use every core", following the scikit-learn convention.
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


#: How many instants inside each video frame a percept is sampled at before
#: being reduced to one value for that frame. Electrical stimulation is
#: pulsatile, so the brightness a frame produces rises and falls within it; a
#: single instant lands wherever the pulse cycle happens to be and can differ
#: from its neighbours by two orders of magnitude for no reason a viewer would
#: recognize. Sampling across the frame and keeping the peak reports what the
#: frame actually did.
#:
#: This is the fallback for models whose kernel cannot track the peak itself
#: (``_reduces_intervals``), and it is only approximate: dynamics much faster
#: than ``frame_dur`` divided by this are under-reported, because no finite
#: sampling can summarize a transient shorter than its own step. A 0.92 ms
#: pulse against a 33.4 ms frame needs 37 samples to be caught reliably; eight
#: of them catch it about one frame in seven, which is enough for a model whose
#: output is already smooth on the millisecond scale and not enough for one
#: whose output is not.
_FRAME_SUBSAMPLES = 8


def _subsample(t_out, dt, n_sub, start=None):
    """Sample the interval leading up to each output point ``n_sub`` times

    The fallback for a model that cannot summarize an interval inside its own
    integrator (see ``_FRAME_SUBSAMPLES``): ask it for several instants per
    interval and keep the largest. Samples are evenly spaced and land exactly
    on the output point, so the value there is always among them -- the peak of
    an interval is then never below the instant it ends on, whatever the
    sampling does.

    Parameters
    ----------
    start : float, optional
        Where the interval summarized by the *first* output point begins (ms).
        A frame clock puts its output points on frame *ends*, so the first one
        summarizes an interval reaching back to the start of frame 0. If None,
        the first output point has no interval and stands for its own instant.

    Returns
    -------
    t : array
        The instants to evaluate at (ms), strictly increasing.
    idx : array
        Where each output point's samples begin in ``t``, ready to pass
        straight to ``np.maximum.reduceat``.

    .. versionadded:: 0.10.0

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
    """When to sample a percept for the video an encoded stimulus came from

    An encoder separates the clock that decides *when the picture changes* from
    the clock that decides *when the electrodes pulse*, and records the former
    in the stimulus' metadata. That is the rate at which a percept is worth
    reporting: one frame in, one frame out. The pulse train's own time points
    are far finer and carry no extra picture.

    A percept point lands on the end of each frame. Everything is rounded onto
    the model's ``dt`` grid (a 29.97 fps frame is 33.3667 ms, which is not a
    multiple of the default dt=0.005 ms). It is the frame *interval* that is
    rounded rather than each frame time on its own, so that the percept keeps
    an evenly spaced time axis -- ``play`` and ``save`` infer a frame rate from
    it and refuse a ragged one.

    Returns
    -------
    t : (n_frames,) array
        The time at which each frame ends, expressed in ``unit``.
    start : float
        The time (in ``unit``) at which the first frame begins, which is where
        the interval summarized by ``t[0]`` starts.

    Returns None for anything that did not come out of an encoder.

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
    """Order parameters so that ``vfmap`` is applied before the others

    A retinal extent assigned to ``xrange``/``yrange`` is resolved through the
    model's visual field map at assignment time (see
    :py:meth:`SpatialModel._retinal_range_to_dva`), so which map is installed
    when that happens decides what the range comes out as. Parameters are
    otherwise applied in the order they were given, and

    .. code-block:: python

        AxonMapModel(xrange=(-2.8 * mm, 2.8 * mm), vfmap=Curcio1990Map())

    has to use the map the caller asked for rather than whichever one was
    already there. Only ``vfmap`` is moved; everything else keeps its order,
    so nothing else becomes order-sensitive.
    """
    if 'vfmap' not in params:
        return params
    return {'vfmap': params['vfmap'],
            **{key: val for key, val in params.items() if key != 'vfmap'}}


def _length_valued(value):
    """Whether a value, or either half of a pair, is a physical length

    What tells a retinal extent apart from a visual one is the *dimension* of
    what was passed, not the unit: ``mm``, ``um`` and ``m`` are all the same
    shorthand, and a bare number is no shorthand at all.
    """
    values = value if isinstance(value, (list, tuple)) else [value]
    return any(isinstance(v, (Quantity, Unit)) and v.dimension == um.dimension
               for v in values)


def _require_stim_dimension(model, stim):
    """Refuse a stimulus that is not the physical quantity a model reads

    Only the *dimension* has to match; a stimulus in a compatible unit is
    converted rather than refused. In practice there is nothing left for a
    model to convert, because :py:class:`~pulse2percept.stimuli.Stimulus`
    canonicalizes to microamps when it is built -- an amplitude given as
    ``0.05 * mA`` is already 50 by the time any model sees it.

    The dimension is another matter, and is the model's to insist on. Gray
    levels are not small currents, so a model that multiplied them by a
    Gaussian current spread and called the result brightness would be doing
    exactly the silent reinterpretation ``stimulus_unit`` exists to declare
    away. A picture becomes a stimulus by being encoded (see
    :py:class:`~pulse2percept.stimuli.StimulusEncoder`), which is where the
    gray levels are given a current to stand for.

    A :py:class:`~pulse2percept.percepts.Percept` is not checked: it is
    brightness, the output of a spatial model, and carries no unit.
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


def _spatial_input(implant):
    """The stimulus a model with no temporal component reads off an implant

    A pulse train says *when* current flows and a raster says which electrodes
    may flow at once. Both are facts about time that such a model has no
    machinery to express -- handed the delivered train, it would report an
    encoded image as a sequence of raster slots. So an encoded stimulus is
    read for the modulation it realizes instead: one amplitude per electrode
    per frame of the source. Anything else is its own answer.
    """
    return implant.stim._spatial_view()


def _rescale(stim, scale):
    """A copy of ``stim`` with every amplitude multiplied by ``scale``

    Rebuilt from the data rather than scaled through the operator, because a
    stimulus with no time component picks one up on the way through: that is
    what lets a temporal model be handed one at all, and
    :py:meth:`TemporalModel.find_threshold` has always relied on it.

    The metadata is carried across: it is what tells ``predict_percept`` which
    video the stimulus was encoded from, and hence when to report a percept.
    Without it every trial of a ``find_threshold`` search would be evaluated
    on a different time base than the caller's own ``predict_percept`` uses.
    """
    return Stimulus(scale * stim.data, electrodes=stim.electrodes,
                    time=stim.time,
                    metadata=deepcopy(stim.metadata))._inherit_units(stim)


def _rescaled_implant(implant, amp):
    """A copy of ``implant`` whose stimulus peaks at ``amp``

    What ``find_threshold`` varies from trial to trial. Scaling the stimulus
    scales every description of it at once: an encoded one is still the
    schedule it was, delivering less current, and what a spatial model reads
    off it moves with what a temporal one does. A search run on only one of
    them would not be a search for the threshold of what the caller's own
    ``predict_percept`` reports.
    """
    trial = deepcopy(implant)
    trial.stim = implant.stim * (amp / implant.stim.data.max())
    return trial


def _delivered(implant):
    """The same implant, made to hand a spatial model the pulse train

    The other side of :py:func:`_spatial_input`: when a temporal stage follows,
    integrating the pulses is the point, so they are what has to get through.
    An ordinary :py:class:`~pulse2percept.stimuli.Stimulus` is its own spatial
    view, so wrapping in one is how the delivered pulses are asked for -- and
    the wrapper stays unmaterialized until something reads them.
    """
    if implant.stim is None or not implant.stim._has_spatial_view:
        return implant
    stand_in = copy(implant)
    stand_in._stim = Stimulus(implant.stim)
    return stand_in


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
    blended = weight * blurred + (1 - weight) * work
    return blended.reshape(resp.shape).astype(resp.dtype, copy=False)


class NotBuiltError(ValueError, AttributeError):
    """Exception class used to raise if model is used before building

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """


class BaseModel(Parametrized, metaclass=ABCMeta):
    """Abstract base class for all models

    Adds the build workflow on top of
    :py:class:`~pulse2percept.utils.Parametrized`, which supplies the
    parameter, pretty-printing, equality and deep-copy machinery:

    *  Build a model (via ``build``) and flip the ``is_built`` switch

    .. versionchanged:: 0.10.0

        Everything other than the build workflow moved to
        :py:class:`~pulse2percept.utils.Parametrized`.

    """

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
        self.is_built = True
        return self

    @property
    def is_built(self):
        """A flag indicating whether the model has been built"""
        return self._is_built

    @is_built.setter
    def is_built(self, val):
        """This flag can only be set in the constructor or ``build``"""
        # getframe(0) is '_is_built', getframe(1) is 'set_attr'.
        # getframe(2) is the one we are looking for, and has to be either the
        # construct or ``build``:
        f_caller_2 = sys._getframe(2).f_code.co_name
        f_caller_3 = sys._getframe(3).f_code.co_name
        if f_caller_2 in ["__init__", "build"] or \
           f_caller_3 in ["__init__", "build"]:
            self._is_built = val
        else:
            err_s = (f"The attribute `is_built` can only be set in the "
                     f"constructor or in ``build``, not in ``{f_caller_2}``.")
            raise AttributeError(err_s)

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        # Guard here as well as in the base implementation: without it, an
        # already-copied model would be rebuilt on every revisit.
        if id(self) in memodict:
            return memodict[id(self)]
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
        if self.vfmap.ndim not in self.ndim:
            raise ValueError(f"Model expects one of {self.ndim} dimensions, but "
                             f"visual field map has {self.vfmap.ndim} dimensions.")
        self.grid = Grid2D(self.xrange, self.yrange, step=self.step,
                           grid_type=self.grid_type)
        self.grid.build(self.vfmap)
        self._build()
        self.is_built = True
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

    def predict_percept(self, implant, t_percept=None):
        """Predict the spatial response

        .. important::

            Don't override this method if you are creating your own model.
            Customize ``_predict_spatial`` instead.

        .. note::

            **This method reads modulation frames, not pulses.** Where
            ``implant.stim`` was produced by the implant's own
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
            Models that replace ``predict_percept`` outright rather than
            customizing ``_predict_spatial`` --
            :py:class:`~pulse2percept.models.BiphasicAxonMapSpatial` and
            :py:class:`~pulse2percept.models.cortex.DynaphosModel` -- read
            ``implant.stim`` directly and are unaffected by any of this.

        Parameters
        ----------
        implant: :py:class:`~pulse2percept.implants.ProsthesisSystem`
            A valid prosthesis system. A stimulus can be passed via
            :py:meth:`~pulse2percept.implants.ProsthesisSystem.stim`.
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
            Will return None if ``implant.stim`` is None.

        """
        if not self.is_built:
            raise NotBuiltError("Yout must call ``build`` first.")
        if not isinstance(implant, ProsthesisSystem):
            raise TypeError(f"'implant' must be a ProsthesisSystem object, "
                            f"not {type(implant)}.")
        t_percept = as_value(t_percept, self.time_unit, 't_percept')
        if implant.stim is None:
            # Nothing to see here:
            return None
        source = _spatial_input(implant)
        _require_stim_dimension(self, source)
        if source.time is None and t_percept is not None:
            # A single-frame source (an image) modulates the electrodes to one
            # steady thing, so there are no times to ask about even though the
            # pulse train delivering it does have a time axis:
            what = ("the modulation behind this stimulus"
                    if source is not implant.stim else "stimulus")
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
                resp_unique = self._predict_spatial(implant.earray, stim_unique)
                # reconstruct original time points, making sure to preserve C ordering
                resp = resp_unique[..., inverse].copy(order='C')
            else:
                resp = self._predict_spatial(implant.earray, stim)
        resp = self._postprocess_spatial(resp)
        return Percept(resp.reshape(list(self.grid.x.shape) + [-1]),
                       space=self.grid, time=t_percept,
                       time_unit=self.time_unit,
                       metadata={'stim': stim}, n_gray=self.n_gray, noise=self.noise)

    def find_threshold(self, implant, bright_th, amp_range=(0, 999), amp_tol=1,
                       bright_tol=0.1, max_iter=100):
        """Find the threshold current for a certain stimulus

        Estimates ``amp_th`` such that the output of
        ``model.predict_percept(stim(amp_th))`` is approximately ``bright_th``.

        Parameters
        ----------
        implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`
            The implant and its stimulus to use. Stimulus amplitude will be
            up and down regulated until ``amp_th`` is found.
        bright_th : float
            Model output (brightness) that's considered "at threshold".
        amp_range : (amp_lo, amp_hi), optional
            Range of amplitudes to search, counted in this model's
            :py:attr:`~pulse2percept.models.BaseModel.stimulus_unit`
            (microamps, for every model p2p ships).
        amp_tol : float, optional
            Search will stop if candidate range of amplitudes is within
            ``amp_tol``, in ``stimulus_unit``
        bright_tol : float, optional
            Search will stop if model brightness is within ``bright_tol`` of
            ``bright_th``
        max_iter : int, optional
            Search will stop after ``max_iter`` iterations

        Returns
        -------
        amp_th : float
            Threshold current, in ``stimulus_unit``, estimated so that the
            output of ``model.predict_percept(stim(amp_th))`` is within
            ``bright_tol`` of ``bright_th``.

        Notes
        -----
        *  ``amp_range`` and ``amp_tol`` may be given as unitful quantities
           (e.g. ``amp_range=(0, 1 * mA)``); the answer comes back as a plain
           number of microamps. ``bright_th`` and ``bright_tol`` are model
           output, which is not a physical quantity and carries no unit. See
           :py:mod:`pulse2percept.units`.

        """
        if not isinstance(implant, ProsthesisSystem):
            raise TypeError(f"'implant' must be a ProsthesisSystem, not "
                            f"{type(implant)}.")
        amp_range = as_value(amp_range, self.stimulus_unit, 'amp_range')
        amp_tol = as_value(amp_tol, self.stimulus_unit, 'amp_tol')

        def inner_predict(amp, fnc_predict, implant):
            return fnc_predict(_rescaled_implant(implant, amp)).data.max()

        return bisect(bright_th, inner_predict,
                      args=[self.predict_percept, implant],
                      x_lo=amp_range[0], x_hi=amp_range[1], x_tol=amp_tol,
                      y_tol=bright_tol, max_iter=max_iter)

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
            raise NotBuiltError("Yout must call ``build`` first.")
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
        return Percept(resp, space=None, time=t_percept,
                       time_unit=self.time_unit, metadata={'stim': stim})

    def _warn_if_blank(self, stim, resp):
        """Point out a percept that came out blank for a polarity reason

        A stimulus of the wrong sign is not an error -- the model integrates it
        and rectifies the result away -- so it otherwise looks exactly like a
        stimulus that was simply too weak to see. Assigning a grayscale image
        or video straight to ``implant.stim`` lands here, because gray levels
        are nonnegative and most temporal models are driven by cathodic
        current.
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

    def find_threshold(self, stim, bright_th, amp_range=(0, 999), amp_tol=1,
                       bright_tol=0.1, max_iter=100, t_percept=None):
        """Find the threshold current for a certain stimulus

        Estimates ``amp_th`` such that the output of
        ``model.predict_percept(stim(amp_th))`` is approximately ``bright_th``.

        Parameters
        ----------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus`
            The stimulus to use. Stimulus amplitude will be up and down
            regulated until ``amp_th`` is found.
        bright_th : float
            Model output (brightness) that's considered "at threshold".
        amp_range : (amp_lo, amp_hi), optional
            Range of amplitudes to search, counted in this model's
            :py:attr:`~pulse2percept.models.BaseModel.stimulus_unit`
            (microamps, for every model p2p ships).
        amp_tol : float, optional
            Search will stop if candidate range of amplitudes is within
            ``amp_tol``, in ``stimulus_unit``
        bright_tol : float, optional
            Search will stop if model brightness is within ``bright_tol`` of
            ``bright_th``
        max_iter : int, optional
            Search will stop after ``max_iter`` iterations
        t_percept: float or list of floats, optional
            The time points at which to output a percept, counted in this
            model's :py:attr:`~pulse2percept.models.BaseModel.time_unit`
            (milliseconds, for every model p2p ships).
            If None, ``implant.stim.time`` is used.
            May be given as a unitful quantity (e.g. ``[0, 20] * ms``); see
            :py:mod:`pulse2percept.units`.

        Returns
        -------
        amp_th : float
            Threshold current, in ``stimulus_unit``, estimated so that the
            output of ``model.predict_percept(stim(amp_th))`` is within
            ``bright_tol`` of ``bright_th``.

        Notes
        -----
        *  ``amp_range``, ``amp_tol`` and ``t_percept`` may be given as unitful
           quantities; the answer comes back as a plain number of microamps.
           ``bright_th`` and ``bright_tol`` are model output, which is not a
           physical quantity and carries no unit. See
           :py:mod:`pulse2percept.units`.

        """
        if not isinstance(stim, Stimulus):
            raise TypeError(f"'stim' must be a Stimulus, not {type(stim)}.")
        amp_range = as_value(amp_range, self.stimulus_unit, 'amp_range')
        amp_tol = as_value(amp_tol, self.stimulus_unit, 'amp_tol')
        t_percept = as_value(t_percept, self.time_unit, 't_percept')

        def inner_predict(amp, fnc_predict, stim, **kwargs):
            _stim = _rescale(stim, amp / stim.data.max())
            return fnc_predict(_stim, **kwargs).data.max()

        return bisect(bright_th, inner_predict,
                      args=[self.predict_percept, stim],
                      kwargs={'t_percept': t_percept},
                      x_lo=amp_range[0], x_hi=amp_range[1], x_tol=amp_tol,
                      y_tol=bright_tol, max_iter=max_iter)


class Model(PrettyPrint):
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
        blah
    temporal: :py:class:`~pulse2percept.models.TemporalModel` or None
        blah
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
        if sys._getframe(1).f_code.co_name == '__init__':
            # Allow setting new attributes in the constructor:
            if isinstance(sys._getframe(1).f_locals['self'], self.__class__):
                super().__setattr__(name, value)
                return
        # Outside the constructor, we cannot add new attributes (FreezeError).
        # But, we have to check whether the attribute is part of the spatial
        # model, the temporal model, or both:
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
        for key, val in _vfmap_first(params).items():
            setattr(self, key, val)

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

    def predict_percept(self, implant, t_percept=None):
        """Predict a percept

        .. important ::

            You must call ``build`` before calling ``predict_percept``.

        Parameters
        ----------
        implant: :py:class:`~pulse2percept.implants.ProsthesisSystem`
            A valid prosthesis system. A stimulus can be passed via
            :py:meth:`~pulse2percept.implants.ProsthesisSystem.stim`.
        t_percept: float or list of floats, optional
            The time points at which to output a percept, counted in this
            model's :py:attr:`~pulse2percept.models.BaseModel.time_unit`
            (milliseconds, for every model p2p ships).
            If None, ``implant.stim.time`` is used.
            May be given as a unitful quantity (e.g. ``[0, 20] * ms``); see
            :py:mod:`pulse2percept.units`.

        Returns
        -------
        percept: :py:class:`~pulse2percept.models.Percept`
            A Percept object whose ``data`` container has dimensions Y x X x T.
            Will return None if ``implant.stim`` is None.
        """
        if not self.is_built:
            raise NotBuiltError("Yout must call ``build`` first.")
        if not isinstance(implant, ProsthesisSystem):
            raise TypeError(f"'implant' must be a ProsthesisSystem object, not "
                            f"{type(implant)}.")
        # The sub-models normalize too; doing it here as well keeps the error
        # message below reading in plain milliseconds:
        t_percept = as_value(t_percept, self.time_unit, 't_percept')
        if implant.stim is None or (not self.has_space and not self.has_time):
            # Nothing to see here:
            return None
        _require_stim_dimension(self, implant.stim)
        # `_has_time_axis`, not `stim.time`: whether there is a time axis is a
        # question a stimulus can answer from its structure, and asking it for
        # the axis itself would generate the waveform behind it.
        has_time_axis = _has_time_axis(implant.stim)
        if not has_time_axis and t_percept is not None:
            raise ValueError(f"Cannot calculate temporal response at times "
                             f"t_percept={t_percept}, because stimulus/percept does not "
                             f"have a time component.")

        if self.has_space and self.has_time:
            # Need to calculate the spatial response at all stimulus points
            # (i.e., whenever the stimulus changes)
            resp = self.spatial.predict_percept(_delivered(implant),
                                                t_percept=None)
            if has_time_axis:
                combine = getattr(self.spatial, '_combine_temporal', None)
                if resp.time is None and combine is not None:
                    # A spatial model hands over a percept with no time axis,
                    # so the spatial model decides what to do with it:
                    resp = combine(resp, self.temporal, implant.stim,
                                   t_percept)
                else:
                    # Then pass that to the temporal model, which will output
                    # at all `t_percept` time steps:
                    resp = self.temporal.predict_percept(resp,
                                                         t_percept=t_percept)
        elif self.has_space:
            resp = self.spatial.predict_percept(implant, t_percept=t_percept)
        elif self.has_time:
            resp = self.temporal.predict_percept(implant.stim,
                                                 t_percept=t_percept)
        return resp

    def find_threshold(self, implant, bright_th, amp_range=(0, 999), amp_tol=1,
                       bright_tol=0.1, max_iter=100, t_percept=None):
        """Find the threshold current for a certain stimulus

        Estimates ``amp_th`` such that the output of
        ``model.predict_percept(stim(amp_th))`` is approximately ``bright_th``.

        Parameters
        ----------
        implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`
            The implant and its stimulus to use. Stimulus amplitude will be
            up and down regulated until ``amp_th`` is found.
        bright_th : float
            Model output (brightness) that's considered "at threshold".
        amp_range : (amp_lo, amp_hi), optional
            Range of amplitudes to search, counted in this model's
            :py:attr:`~pulse2percept.models.BaseModel.stimulus_unit`
            (microamps, for every model p2p ships).
        amp_tol : float, optional
            Search will stop if candidate range of amplitudes is within
            ``amp_tol``
        bright_tol : float, optional
            Search will stop if model brightness is within ``bright_tol`` of
            ``bright_th``
        max_iter : int, optional
            Search will stop after ``max_iter`` iterations
        t_percept: float or list of floats, optional
            The time points at which to output a percept, counted in this
            model's :py:attr:`~pulse2percept.models.BaseModel.time_unit`
            (milliseconds, for every model p2p ships).
            If None, ``implant.stim.time`` is used.
            May be given as a unitful quantity (e.g. ``[0, 20] * ms``); see
            :py:mod:`pulse2percept.units`.

        Returns
        -------
        amp_th : float
            Threshold current, in ``stimulus_unit``, estimated so that the
            output of ``model.predict_percept(stim(amp_th))`` is within
            ``bright_tol`` of ``bright_th``.

        Notes
        -----
        *  ``amp_range``, ``amp_tol`` and ``t_percept`` may be given as unitful
           quantities; the answer comes back as a plain number of microamps.
           ``bright_th`` and ``bright_tol`` are model output, which is not a
           physical quantity and carries no unit. See
           :py:mod:`pulse2percept.units`.

        """
        if not isinstance(implant, ProsthesisSystem):
            raise TypeError(f"'implant' must be a ProsthesisSystem, not "
                            f"{type(implant)}.")
        amp_range = as_value(amp_range, self.stimulus_unit, 'amp_range')
        amp_tol = as_value(amp_tol, self.stimulus_unit, 'amp_tol')
        t_percept = as_value(t_percept, self.time_unit, 't_percept')

        def inner_predict(amp, fnc_predict, implant, **kwargs):
            return fnc_predict(_rescaled_implant(implant, amp),
                               **kwargs).data.max()

        return bisect(bright_th, inner_predict,
                      args=[self.predict_percept, implant],
                      kwargs={'t_percept': t_percept},
                      x_lo=amp_range[0], x_hi=amp_range[1], x_tol=amp_tol,
                      y_tol=bright_tol, max_iter=max_iter)

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
