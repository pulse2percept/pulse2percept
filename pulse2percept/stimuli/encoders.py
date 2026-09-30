""":py:class:`~pulse2percept.stimuli.Encoder`,
   :py:class:`~pulse2percept.stimuli.ImplantEncoder`,
   :py:class:`~pulse2percept.stimuli.PulseEncoder`,
   :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`,
   :py:class:`~pulse2percept.stimuli.FrequencyEncoder`,
   :py:class:`~pulse2percept.stimuli.TraceEncoder`,
   :py:class:`~pulse2percept.stimuli.PhotovoltaicEncoder`,
   :py:class:`~pulse2percept.stimuli.PRIMAEncoder`"""
from abc import ABCMeta, abstractmethod
import math
import numpy as np
from copy import deepcopy
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .base import ImageStimulus, Stimulus, VideoStimulus, _adoptable
from .pulses import BiphasicPulse
from ..units import (DimensionMismatchError, Hz, Quantity, as_value,
                     dimensionless, dva, mW, mm, ms, nm, uA, xTh)
from ..utils import PrettyPrint, frame_interval
# Point encoder warnings at the caller.
from ..utils.deprecation import _warn_external
from ..utils.constants import DT, MS_PER_S

# Warn when encoding an unsampled pixel grid would create a huge stimulus.
_BIG_STIM = 5e7

# Warn before an encoded stimulus grows to an impractical number of time points.
_BIG_TIME = 20000

# Duration (ms) of a source that has no time axis of its own, such as an image:
_DEFAULT_FRAME_DUR = 500.0


def _finite(name, value):
    """Raise ValueError for NaN or infinity (which pass ``<`` checks)"""
    if not np.all(np.isfinite(np.asarray(value, dtype=np.float64))):
        raise ValueError(f"'{name}' must be finite, not {value}.")


def _all_equal(a):
    """Return True if all elements of ``a`` are equal (True if empty)"""
    return a.size == 0 or bool(np.all(a == a.flat[0]))


def _fps(metadata):
    """Return the frame rate stored in stimulus metadata, or None

    Looks at the top level (``VideoStimulus``) and under ``'user'``
    (stimuli derived from it).
    """
    if not isinstance(metadata, dict):
        return None
    if 'fps' in metadata:
        return metadata['fps']
    user = metadata.get('user')
    return user.get('fps') if isinstance(user, dict) else None


class _EncodedStimulus(Stimulus):
    """A resolved encoder schedule, expanded into a waveform on demand"""
    #: See `Stimulus._is_parametric`:
    _is_parametric = True

    #: whether the stimulus has a special spatial view
    _has_spatial_view = True

    __slots__ = ('_amp', '_ticks', '_sched', '_onsets', '_frames',
                 '_pulse_ticks', '_pulse_vals', '_total', '_freq',
                 '_frame_time', '_frame_dur', '_time', '_phase_dur',
                 '_cathodic_first', '_source_time', '_source_dur')

    def __init__(self, electrodes, amp, ticks, sched, onsets, frames,
                 pulse_ticks, pulse_vals, total, freq, frame_time,
                 frame_dur, cycle, amp_unit=uA, phase_dur=None,
                 cathodic_first=True, source_time=None, source_dur=None):
        self._amp = self._own(amp, amp.dtype)
        self._ticks = self._own(ticks, ticks.dtype)
        self._sched = self._own(sched, sched.dtype)
        self._onsets = tuple(self._own(o, o.dtype) for o in onsets)
        self._frames = tuple(self._own(f, f.dtype) for f in frames)
        self._pulse_ticks = self._own(pulse_ticks, pulse_ticks.dtype)
        self._pulse_vals = self._own(pulse_vals, pulse_vals.dtype)
        self._total = float(total)
        # Realized Hz after clock/raster quantization:
        self._freq = self._own(freq, np.float64)
        self._frame_time = self._own(frame_time, frame_time.dtype)
        self._frame_dur = float(frame_dur)
        self._phase_dur = None if phase_dur is None else float(phase_dur)
        self._cathodic_first = bool(cathodic_first)
        # Source-video frame onsets and duration (ms); None for a still or
        # a video retimed by the encoder's `frame_dur`:
        self._source_time = (None if source_time is None else
                             self._own(source_time, np.float64))
        self._source_dur = None if source_dur is None else float(source_dur)
        # Built lazily without rendering the waveform:
        self._time = None
        self._defer(electrodes, unit=amp_unit)
        self.metadata['encoder'] = {'frame_time': self._frame_time,
                                    'frame_dur': self._frame_dur,
                                    'cycle': cycle, **self._source_clock()}

    def _source_clock(self):
        """Source-video frame onsets and duration (ms); empty if none"""
        if self._source_time is None:
            return {}
        return {'source_frame_time': self._source_time,
                'source_frame_dur': self._source_dur}

    @property
    def _firing(self):
        """Electrode-frames whose pulse clock is running"""
        return self._freq > 0

    def _spatial_view(self):
        """One column per frame of the source and one row per electrode"""
        data = np.where(self._firing, self._amp, np.float32(0)).astype(
            np.float32)
        # The spatial view represents a single frame as timeless:
        if self._frame_time.size > 1:
            stim = Stimulus(data, electrodes=self.electrodes,
                            time=self._frame_time)
        else:
            stim = Stimulus(data.ravel(), electrodes=self.electrodes)
        stim.metadata['encoder'] = {'frame_time': self._frame_time,
                                    'frame_dur': self._frame_dur,
                                    **self._source_clock()}
        return stim._inherit_units(self)

    def _rebuilt(self, electrodes, amp, sched, freq, amp_unit=None):
        """Return this schedule with new electrodes or amplitudes

        ``amp_unit`` defaults to the current unit.
        """
        rebuilt = _EncodedStimulus(
            electrodes, amp, self._ticks, sched, self._onsets, self._frames,
            self._pulse_ticks, self._pulse_vals, self._total, freq,
            self._frame_time, self._frame_dur,
            self.metadata['encoder']['cycle'],
            amp_unit=self.unit if amp_unit is None else amp_unit,
            phase_dur=self._phase_dur,
            cathodic_first=self._cathodic_first,
            source_time=self._source_time, source_dur=self._source_dur)
        rebuilt.metadata['user'] = deepcopy(self.metadata.get('user'))
        return rebuilt

    def _biphasic_params(self):
        """Return one realized biphasic condition per driven electrode.

        Each entry is ``(electrode, freq, amp, phase_dur, stim_dur,
        cathodic_first)``. Returns None for custom pulses and rejects
        multi-frame schedules.
        """
        if self._phase_dur is None:
            return None
        if self._amp.shape[1] != 1:
            raise NotImplementedError(
                f"A schedule of {self._amp.shape[1]} frames describes a "
                f"different pulse train on each frame, so it has no single "
                f"(freq, amp, phase_dur) per electrode. Encode a still image, "
                f"or drive the electrodes with BiphasicPulseTrain objects.")
        return [(name, float(f), float(a), self._phase_dur, self._total,
                 self._cathodic_first)
                for name, a, f in zip(self.electrodes, self._amp[:, 0],
                                      self._freq[:, 0])
                if a != 0 and f > 0]

    def _with_thresholds(self, thresholds):
        """Return an xTh schedule calibrated to uA, without rendering"""
        if self.unit != xTh:
            return self
        driven = np.any(self._firing & (self._amp != 0), axis=1)
        names = [n for n, d in zip(self.electrodes, driven) if d]
        missing = sorted(n for n in names if n not in thresholds)
        if len(missing) == len(names):
            return self
        if missing:
            raise DimensionMismatchError(
                f"Calibrating only some electrodes would leave "
                f"{', '.join(missing)} measured in threshold multiples and "
                f"the rest in uA. Give every driven electrode a threshold, or "
                f"none of them.")
        # Undriven electrodes need no threshold:
        scale = np.array([thresholds.get(n, 1.0) for n in self.electrodes],
                         dtype=np.float32)[:, np.newaxis]
        return self._rebuilt(self.electrodes, self._amp * scale, self._sched,
                             self._freq, amp_unit=uA)

    def _scaled(self, factor):
        """Return this schedule with amplitudes scaled by ``factor``"""
        return self._rebuilt(self.electrodes, self._amp * factor, self._sched,
                             self._freq)

    def _without_electrodes(self, electrodes):
        """Return this schedule without ``electrodes``"""
        keep = self._keep_mask(electrodes)
        return self._rebuilt(self.electrodes[keep], self._amp[keep],
                             self._sched[keep], self._freq[keep])

    @property
    def duration(self):
        """Duration of the stimulus (ms)"""
        return self._total

    @property
    def time(self):
        """Time points of the stimulus (ms)"""
        if self._time is None:
            time = self._ticks * DT
            time[-1] = self._total
            self._time = self._own(time, np.float64)
        return self._time

    def _render(self):
        """Expand the schedule into pulse trains."""
        data = np.zeros((len(self.electrodes), self._ticks.size),
                        dtype=np.float32)
        for s, (onset, frame) in enumerate(zip(self._onsets, self._frames)):
            rows = np.flatnonzero(self._sched == s)
            if rows.size == 0 or onset.size == 0:
                continue
            wave = PulseEncoder._sample(onset, self._pulse_ticks,
                                        self._pulse_vals, self._ticks)
            # Which pulse each time point belongs to:
            at = np.searchsorted(onset, self._ticks, side='right') - 1
            np.clip(at, 0, onset.size - 1, out=at)
            data[rows] = self._amp[rows][:, frame[at]] * wave
        # ``data`` is newly allocated and can be adopted without copying.
        return {'data': _adoptable(data), 'electrodes': self.electrodes,
                'time': self.time}

    def _pprint_params(self):
        """Return a dict of class attributes to pretty-print"""
        return {'electrodes': self.electrodes,
                'n_frames': self._amp.shape[1],
                'n_time': self._ticks.size,
                'n_schedules': len(self._onsets),
                'duration': self._total,
                'metadata': self.metadata}


def _sampled_frames(source, implant=None, frame_dur=None):
    """Reduce a source to one gray level per stimulation site per frame.

    Images and videos are sampled at the implant's electrode locations; any
    other dimensionless stimulus is used as is. Without an implant, every
    pixel is its own stimulation site.

    Parameters
    ----------
    frame_dur : float, optional
        Frame duration (ms) to impose on the source. If None, a source
        with a time axis keeps its own frame timing and one without a
        time axis is presented for ``_DEFAULT_FRAME_DUR`` ms.

    Returns
    -------
    gray : ``(n_electrodes, n_frames)`` array
        Gray levels clipped to [0, 1].
    electrodes : array
        Electrode names.
    frame_time : (n_frames,) array
        Frame onset times (ms).
    frame_dur : float
        Frame duration (ms).
    """
    if not isinstance(source, Stimulus):
        raise TypeError(f"'source' must be a Stimulus object, not "
                        f"{type(source)}.")
    # Encoders accept gray levels, not already-physical stimulation.
    if not source.unit.dimension.is_dimensionless:
        raise DimensionMismatchError(
            f"An encoder turns gray levels into stimulation, so its "
            f"source must be dimensionless, not "
            f"{source.unit.dimension.name} ({source.unit}). Pass an "
            f"ImageStimulus or a VideoStimulus.")
    # Read frame rate before sampling at implant coordinates.
    fps = _fps(source.metadata)
    stim = source
    if (implant is not None and
            isinstance(stim, (ImageStimulus, VideoStimulus))):
        # Sample images/videos at implant coordinates and convert RGB to gray.
        stim = implant.reshape_stim(stim)
    # Modulation operates on dimensionless gray levels in [0, 1].
    gray = np.clip(np.asarray(stim.values(dimensionless),
                              dtype=np.float32), 0, 1)
    if stim.time is None:
        # Static images use the default presentation duration.
        gray = gray.reshape((-1, 1))
        frame_dur = (_DEFAULT_FRAME_DUR if frame_dur is None
                     else frame_dur)
        frame_time = np.zeros(1, dtype=np.float64)
    elif frame_dur is None:
        # Preserve the source frame interval.
        frame_dur = frame_interval(np.asarray(stim.time), fps=fps)
        frame_time = np.asarray(stim.time, dtype=np.float64)
    else:
        # Explicit frame_dur replaces source frame timing.
        frame_time = np.arange(gray.shape[1], dtype=np.float64) * frame_dur
    return gray, stim.electrodes, frame_time, frame_dur


class Encoder(PrettyPrint, metaclass=ABCMeta):
    """Base class for encoders.

    An encoder converts a desired visual target into stimulation for a
    prosthetic system. Each subclass defines the target representation it
    accepts and implements :py:meth:`encode`.

    Only an :py:class:`~pulse2percept.stimuli.ImplantEncoder` can be installed
    on :py:attr:`Implant.encoder <pulse2percept.implants.Implant.encoder>`.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.Implant`, optional
        The implant to encode for. Once bound, an encoder cannot be rebound.
    """
    __slots__ = ('_implant',)

    def __init__(self, implant=None):
        self._implant = None
        if implant is not None:
            self._bind(implant)

    @property
    def implant(self):
        """The implant this encoder is bound to, or None (read-only)"""
        return self._implant

    def _bind(self, implant):
        """Bind to ``implant``; rebinding to a different implant fails"""
        # Imported here because `implants` imports this module:
        from ..implants.base import Implant
        if not isinstance(implant, Implant):
            raise TypeError(f"'implant' must be an Implant object, not "
                            f"{type(implant)}.")
        if self._implant is None:
            self._implant = implant
        elif self._implant is not implant:
            raise ValueError(
                f"This {type(self).__name__} is already bound to another "
                f"{type(self._implant).__name__}. Construct a separate "
                f"encoder for each implant.")
        return self

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        # Class name only: the implant pretty-prints its own encoder.
        return {'implant': (None if self.implant is None else
                            type(self.implant).__name__)}

    @abstractmethod
    def encode(self, source):
        """Encode a visual target as stimulation

        Parameters
        ----------
        source
            The visual target to encode, in the representation the subclass
            accepts.

        Returns
        -------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus`
            The encoded stimulus, in the unit that drives the device.
        """
        raise NotImplementedError


class ImplantEncoder(Encoder):
    """Base class for encoders installable on an implant.

    An implant encoder accepts the dimensionless
    :py:class:`~pulse2percept.stimuli.Stimulus` (image or video gray levels)
    that :py:meth:`Implant.prepare_stim
    <pulse2percept.implants.Implant.prepare_stim>` produces, and can
    therefore be assigned to :py:attr:`Implant.encoder
    <pulse2percept.implants.Implant.encoder>`. Assigning an unbound encoder
    binds it to that implant.

    .. versionadded:: 0.11.0

    """
    __slots__ = ()


class PulseEncoder(ImplantEncoder):
    """Abstract base class for electrical pulse-train encoders.

    Maps image or video gray levels onto the parameters of repeated
    electrical pulses. If the encoder has an implant, the source is sampled
    at its electrode locations and scheduled using its raster pattern.

    Subclasses implement :meth:`_modulate`, which maps gray levels to
    pulse amplitude and frequency.

    .. versionadded:: 0.10.0

    .. versionchanged:: 0.11.0
        Renamed from ``StimulusEncoder``. The implant is passed to the
        constructor rather than to :py:meth:`encode`.

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.Implant`, optional
        The implant to encode for. Its electrode locations are used to
        sample images and videos, its electrode names label the result, and
        its :py:attr:`~pulse2percept.implants.Implant.raster` decides which
        electrodes may pulse when. If None, every pixel of the source is its
        own electrode and every electrode fires on the same schedule.
    phase_dur : float, optional
        Duration of each pulse phase (ms).
    interphase_dur : float, optional
        Gap between cathodic and anodic phases (ms).
    cathodic_first : bool, optional
        If True, deliver the cathodic phase first.
    pulse : :class:`~pulse2percept.stimuli.Stimulus`, optional
        Pulse shape to repeat. Its amplitude is normalized away.
    clock : float, optional
        Stimulator clock period (ms). Pulse periods and raster offsets
        are rounded to whole clock cycles.
    n_levels : int, optional
        Number of gray levels used before modulation.
    frame_dur : float, optional
        Frame duration (ms). If None, infer it from the source.
    stretch : bool, optional
        If True, stretch source gray levels to [0, 1].

    Notes
    -----
    Plain numbers use the units documented above; unitful values are
    converted automatically. See :mod:`pulse2percept.units`.
    """
    __slots__ = ('phase_dur', 'interphase_dur', 'cathodic_first', 'pulse',
                 'clock', 'n_levels', 'frame_dur', 'stretch')

    #: Unit of amplitudes returned by _modulate
    amp_unit = uA

    def __init__(self, implant=None, phase_dur=0.46, interphase_dur=0,
                 cathodic_first=True, pulse=None, clock=None, n_levels=None,
                 frame_dur=None, stretch=False):
        super().__init__(implant)
        # Normalize timing inputs; a custom pulse contributes shape only.
        phase_dur = as_value(phase_dur, ms, 'phase_dur')
        interphase_dur = as_value(interphase_dur, ms, 'interphase_dur')
        clock = as_value(clock, ms, 'clock')
        frame_dur = as_value(frame_dur, ms, 'frame_dur')
        _finite('phase_dur', phase_dur)
        if phase_dur <= DT:
            raise ValueError(f"'phase_dur' must be greater than DT={DT} ms.")
        _finite('interphase_dur', interphase_dur)
        if interphase_dur < 0:
            raise ValueError("'interphase_dur' cannot be negative.")
        if pulse is not None:
            if not isinstance(pulse, Stimulus):
                raise TypeError(f"'pulse' must be a Stimulus object, not "
                                f"{type(pulse)}.")
            if pulse.time is None:
                raise ValueError("'pulse' must have a time component.")
            if pulse.shape[0] != 1:
                raise ValueError(f"'pulse' must be a single-electrode "
                                 f"stimulus, not one with {pulse.shape[0]} "
                                 f"electrodes.")
        if clock is not None:
            _finite('clock', clock)
            if clock < DT:
                raise ValueError(f"'clock' cannot be finer than the simulation "
                                 f"time step DT={DT} ms.")
        if n_levels is not None:
            _finite('n_levels', n_levels)
            # ``n_levels`` is a count.
            if int(n_levels) != n_levels:
                raise ValueError(f"'n_levels' must be a whole number, not "
                                 f"{n_levels}.")
            if n_levels < 2:
                raise ValueError("'n_levels' must be at least 2.")
        if frame_dur is not None:
            _finite('frame_dur', frame_dur)
            if frame_dur <= 0:
                raise ValueError("'frame_dur' must be positive.")
        self.phase_dur = phase_dur
        self.interphase_dur = interphase_dur
        self.cathodic_first = cathodic_first
        self.pulse = pulse
        self.clock = clock
        self.n_levels = n_levels
        self.frame_dur = frame_dur
        self.stretch = stretch

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super()._pprint_params()
        params.update({'phase_dur': self.phase_dur,
                       'interphase_dur': self.interphase_dur,
                       'cathodic_first': self.cathodic_first,
                       'pulse': self.pulse, 'clock': self.clock,
                       'n_levels': self.n_levels,
                       'frame_dur': self.frame_dur, 'stretch': self.stretch})
        return params

    @abstractmethod
    def _modulate(self, gray):
        """Map gray levels to pulse amplitude and frequency.

        Parameters
        ----------
        gray : ``(n_electrodes, n_frames)`` array
            Gray levels in [0, 1].

        Returns
        -------
        amp : array
            Nonnegative pulse amplitudes, measured in
            :py:attr:`amp_unit`, broadcastable to ``gray.shape``.
        freq : array
            Pulse frequencies (Hz), broadcastable to ``gray.shape``.
        """
        raise NotImplementedError

    @staticmethod
    def _ticks(t):
        """Round a time (ms) onto the simulation's ``DT`` grid

        ``Stimulus`` cannot hold time points closer than ``DT`` anyway, and
        integer ticks make merging electrode time axes exact.
        """
        return np.round(np.asarray(t, dtype=np.float64) / DT).astype(np.int64)

    def _unit_pulse(self):
        """Return the pulse to repeat as (ticks, values), peak amplitude 1

        Values peak at -1 (cathodic first) or +1. Ticks start at zero.
        """
        if self.pulse is None:
            pulse = BiphasicPulse(1, self.phase_dur,
                                  interphase_dur=self.interphase_dur,
                                  cathodic_first=self.cathodic_first)
            values = pulse.data.ravel().astype(np.float32)
        else:
            pulse = self.pulse
            # Make sure we don't change the user's pulse:
            values = pulse.data.ravel().astype(np.float32).copy()
            peak = np.abs(values).max()
            if peak > 0:
                values /= peak
        ticks = self._ticks(pulse.time)
        if np.any(np.diff(ticks) < 1):
            raise ValueError(f"'pulse' has time points closer together than "
                             f"DT={DT} ms, which the simulation cannot "
                             f"resolve. Lengthen 'phase_dur'.")
        # Only the pulse shape is used, so start it at t=0:
        ticks = ticks - ticks[0]
        # Nonzero end points would fill the gaps between tiled pulses:
        if values[0] != 0 or values[-1] != 0:
            raise ValueError("'pulse' must start and end at zero amplitude, "
                             "since it is repeated to fill a train.")
        return ticks, values

    def _periods(self, freq, pulse_len):
        """Pulse period for every electrode and frame

        The period is a fractional number of ticks; only each onset is rounded
        to ``DT``. Rounding the period would accumulate error (a 30 Hz period
        is 33333.33 ticks, drifting 1/3 tick per pulse).

        Returns
        -------
        firing : ``(n_electrodes, n_frames)`` bool array
            Whether the pulse clock is running at all.
        period : ``(n_electrodes, n_frames)`` float array
            The period, in ticks, or 0 where the clock is not running.

        """
        firing = freq > 0
        period = np.zeros(freq.shape, dtype=np.float64)
        # Hz to a period in ms, and ms to ticks of the DT grid:
        period[firing] = MS_PER_S / freq[firing] / DT
        # A clocked stimulator requires periods of whole clock cycles (this
        # also reduces the number of time points). Round the period up:
        if self.clock is not None:
            tick = self.clock / DT
            period[firing] = tick * np.maximum(
                1.0, np.ceil(period[firing] / tick - 1e-9))
        if np.any(period[firing] < pulse_len):
            too_fast = MS_PER_S / (np.min(period[firing]) * DT)
            raise ValueError(f"A pulse (dur={pulse_len * DT:.3f} ms) does not "
                             f"fit into the pulse train window of a "
                             f"{too_fast:.1f} Hz train. Shorten 'phase_dur' "
                             f"or lower the frequency.")
        return firing, period

    def _raster_grid(self, electrodes, period, firing, pulse_len, raster):
        """The slot each electrode may pulse in, and the raster sweep

        The sweep must fit inside the shortest pulse period. Without
        ``group_dur``, the sweep equals that period, split evenly between the
        groups; otherwise it is ``n_groups * group_dur``.

        Returns
        -------
        offset : ``(n_electrodes,)`` float array
            How far behind group 0 (in ticks) each electrode may start a pulse.
        cycle : float or None
            The sweep in ticks. ``_assemble`` quantizes differing periods onto
            it so groups cannot drift together; a shared period is unchanged.
            None if there is nothing to multiplex.

        """
        zero = np.zeros(len(electrodes), dtype=np.float64)
        if raster is None or raster.n_groups < 2 or not np.any(firing):
            return zero, None
        # Use the requested frequencies, not the amplitudes, so raster
        # feasibility does not depend on the video content:
        fastest = float(np.min(period[firing]))
        group = np.asarray(raster.groups(electrodes), dtype=np.int64)
        if (group.min(initial=0) < 0 or
                group.max(initial=0) >= raster.n_groups):
            raise ValueError(f"'groups' must be in 0..{raster.n_groups - 1}.")
        slot = raster.slot_dur(fastest * DT) / DT
        if self.clock is not None:
            tick = self.clock / DT
            if raster.group_dur is not None:
                # Round the explicit slot onto the clock; the cycle is built
                # from it:
                slot = max(1.0, round(slot / tick)) * tick
            else:
                # Pulses can only start on a clock edge. Use the largest whole
                # number of clock cycles that fits every group into the period:
                slot = np.floor(slot / tick + 1e-9) * tick
                if slot < tick:
                    raise ValueError(
                        f"A {fastest * DT:.3f} ms pulse period holds only "
                        f"{int(fastest / tick)} clock cycle(s) of "
                        f"clock={self.clock:g} ms, which is not enough to give "
                        f"each of {raster.n_groups} raster groups its own turn. "
                        f"Use fewer groups, a finer 'clock', or a lower "
                        f"frequency.")
        # With no explicit slot the groups divide the pulse period:
        cycle = fastest if raster.group_dur is None else raster.n_groups * slot
        # Check the clock-rounded slot (e.g., 5.1 ms on a 1 ms clock is 5 ms,
        # two of which fit into a 10 ms period):
        if cycle > fastest * (1 + 1e-9):
            raise ValueError(
                f"A raster of {raster.n_groups} groups {slot * DT:.3f} ms "
                f"apart takes {cycle * DT:.3f} ms to get through, which does "
                f"not fit into the {fastest * DT:.3f} ms pulse period. Shorten "
                f"'group_dur', use fewer groups, or lower the frequency.")
        offset = group.astype(np.float64) * slot
        # Each group's turn must fit a pulse plus one tick:
        edges = np.unique(np.round(offset))
        if edges.size < np.unique(group).size:
            raise ValueError(
                f"Two raster groups were given the same {slot * DT:.3f} ms "
                f"turn, so they would pulse together. Use fewer groups, a "
                f"finer 'clock', or a lower frequency.")
        edges = np.append(edges, np.round(cycle))
        gap = float(np.min(np.diff(edges), initial=cycle))
        if gap < pulse_len + 1:
            raise ValueError(
                f"A raster group gets a {gap * DT:.3f} ms turn, which has no "
                f"room for a {pulse_len * DT:.3f} ms pulse. Use fewer groups, "
                f"shorten 'phase_dur', or lower the pulse frequency (a faster "
                f"pulse train leaves each group less time).")
        return offset, cycle

    @staticmethod
    def _onsets(start, period, active, frame_ticks, last, grid):
        """When one schedule pulses, and which frame each pulse belongs to

        Parameters
        ----------
        start : float
            The first tick at which this schedule may pulse.
        last : int
            The last tick at which a pulse may *begin*.
        grid : float
            Onset spacing: the raster sweep, else the stimulator clock. A
            schedule that goes silent resumes on this grid.

        Returns
        -------
        onset : (n_pulses,) int array
            The tick each pulse begins at.
        frame : (n_pulses,) int array
            The frame whose modulation parameters each pulse carries.

        """
        n_frames = frame_ticks.size
        empty = np.zeros(0, dtype=np.int64)
        # Fast path for a single period (amplitude modulation): onsets are an
        # arithmetic sequence, minus inactive frames:
        step = float(period[0])
        if step > 0 and np.all(period == step):
            if start > last:
                return empty, empty
            n = int(np.floor((last - start) / step + 1e-9)) + 1
            onset = np.round(start + np.arange(n) * step).astype(np.int64)
            frame = np.searchsorted(frame_ticks, onset, side='right') - 1
            np.clip(frame, 0, n_frames - 1, out=frame)
            keep = active[frame]
            return onset[keep], frame[keep]
        # Frequency modulation: the rate is piecewise constant over frames, so
        # track the pulse-clock phase. Phase advances at 1/period; a pulse
        # fires when it reaches 1:
        onset, frame = [], []
        # Start at full phase, so the first pulse lands at the start of its
        # slot. Start in the frame that contains that slot:
        phase, t = 1.0, float(start)
        k = int(np.searchsorted(frame_ticks, round(t), side='right')) - 1
        k = min(max(k, 0), n_frames - 1)
        prev = -np.inf
        while k < n_frames and t <= last:
            # How far this frame reaches, and how much phase it can supply:
            edge = float(frame_ticks[k + 1]) if k + 1 < n_frames else np.inf
            rate = 1.0 / period[k] if period[k] > 0 else 0.0
            if rate == 0.0:
                # A stopped clock (0 Hz) accumulates no phase; carry the
                # current phase to the next frame:
                if not np.isfinite(edge):
                    break
                t, k = edge, k + 1
                continue
            due = phase + (edge - t) * rate
            if due < 1.0:
                # No pulse due in this frame; carry the phase to the next one:
                if not np.isfinite(edge):
                    break
                phase, t, k = due, edge, k + 1
                continue
            # The pulse is due in this frame. Snap it forward onto the grid
            # (raster cycle or clock), so it is never delivered earlier (at a
            # higher rate) than requested:
            cross = t + (1.0 - phase) / rate
            tick = int(round(start + grid * np.ceil(
                (cross - start) / grid - 1e-9)))
            if tick <= prev:
                # Onsets must strictly increase:
                tick = int(round(prev + grid))
            if tick > last:
                break
            # Snapping can move the pulse into the next frame, so look up the
            # frame at the snapped onset:
            j = int(np.searchsorted(frame_ticks, tick, side='right')) - 1
            j = min(max(j, 0), n_frames - 1)
            if period[j] <= 0:
                # The pulse landed in a 0 Hz frame. Hold it for the next frame
                # that restarts the clock:
                phase, t, k = 1.0, float(tick), j
                continue
            if active[j]:
                onset.append(tick)
                frame.append(j)
            prev, phase, t = tick, 0.0, float(tick)
            k = j
        return (np.asarray(onset, dtype=np.int64).reshape(-1),
                np.asarray(frame, dtype=np.int64).reshape(-1))

    @staticmethod
    def _sample(onset, pulse_ticks, pulse_vals, ticks):
        """Sample one schedule's pulse train onto the stimulus' time axis"""
        t = (onset[:, np.newaxis] + pulse_ticks[np.newaxis, :]).ravel()
        v = np.tile(pulse_vals, onset.size)
        # Back-to-back pulses share a (zero) end point:
        t, keep = np.unique(t, return_index=True)
        return np.interp(ticks, t, v[keep])

    def _assemble(self, amp, freq, electrodes, frame_time, frame_dur,
                  timed=False):
        """Build the pulse trains for every electrode and frame

        Electrodes that pulse at the same times share a waveform, scaled by
        their amplitude. One waveform is built per distinct schedule, which
        keeps frequency modulation tractable (thousands of electrode-frames,
        a few dozen schedules).

        The time axis is global: pulses are placed at absolute times.
        """
        n_el, n_frames = len(electrodes), frame_time.size
        shape = (n_el, n_frames)
        amp = np.ascontiguousarray(
            np.broadcast_to(np.asarray(amp, dtype=np.float32), shape))
        freq = np.ascontiguousarray(
            np.broadcast_to(np.asarray(freq, dtype=np.float64), shape))
        pulse_ticks, pulse_vals = self._unit_pulse()
        pulse_len = int(pulse_ticks[-1])
        frame_ticks = self._ticks(frame_time)
        total = float(frame_time[-1] + frame_dur)
        # The stimulus lasts as long as the source. Floor (with an epsilon for
        # binary rounding) to leave at least one tick between the last pulse
        # and the end point:
        end = int(np.floor(total / DT + 1e-9))
        last = end - 1 - pulse_len
        if last < 0:
            raise ValueError(f"A pulse (dur={pulse_len * DT:.3f} ms) does not "
                             f"fit into a stimulus of {total:.3f} ms. Shorten "
                             f"'phase_dur' or lengthen the source.")
        firing, period = self._periods(freq, pulse_len)
        # Zero-amplitude electrodes keep their clock running ("firing") to stay
        # in phase, but add no pulses or time points:
        active = firing & (amp != 0)
        # The implant defines the electrode schedule:
        raster = getattr(self.implant, 'raster', None)
        offset, cycle = self._raster_grid(electrodes, period, firing, pulse_len,
                                          raster)
        if cycle is not None and not _all_equal(period[firing]):
            # Different periods drift, so groups would eventually coincide.
            # Round every period up to a whole number of raster cycles:
            period[firing] = cycle * np.maximum(
                1.0, np.ceil(period[firing] / cycle - 1e-9))

        # An electrode's schedule is fixed by its slot, periods, and activity:
        key = np.concatenate([offset[:, np.newaxis], period,
                              active.astype(np.float64)], axis=1)
        uniq, sched = np.unique(key, axis=0, return_inverse=True)
        # The shape of `return_inverse` differs between NumPy 2.x releases:
        sched = np.ravel(sched)
        origin = float(frame_ticks[0])
        # Onset grid: raster cycle, else stimulator clock, else DT:
        grid = (cycle if cycle is not None else
                (self.clock / DT if self.clock is not None else 1.0))
        onsets, frames = [], []
        for row in uniq:
            onset, frame = self._onsets(
                origin + row[0], row[1:1 + n_frames],
                row[1 + n_frames:].astype(bool), frame_ticks, last, grid)
            onsets.append(onset)
            frames.append(frame)
        # Warn about frames that get no pulse (pulse rate below frame rate):
        hit = np.zeros(n_frames, dtype=bool)
        for f in frames:
            hit[f] = True
        missed = np.count_nonzero(~hit & active.any(axis=0))
        if missed:
            fps = MS_PER_S / frame_dur
            _warn_external(
                f"{missed} of {n_frames} frames deliver no pulse at all, "
                f"because the pulse period is longer than a frame "
                f"({fps:.2f} fps). Their gray levels are never sampled; "
                f"raise the frequency to see them.", category=UserWarning)

        ticks = np.unique(np.concatenate(
            [np.array([0, end], dtype=np.int64)] +
            [(o[:, np.newaxis] + pulse_ticks[np.newaxis, :]).ravel()
             for o in onsets if o.size]))
        n_time = ticks.size
        if n_time > _BIG_TIME:
            _warn_external(
                f"This stimulus has {n_time} time points, which every model "
                f"downstream will pay for. Coarsening 'clock' is the lever "
                f"that helps most, since it confines every pulse onset to the "
                f"same grid; a 'raster' does the same. 'n_levels' helps far "
                f"less on its own, because two electrodes on the same gray "
                f"level still pulse at different times.",
                category=UserWarning)
        if n_el * n_time > _BIG_STIM and self.implant is None:
            _warn_external(
                f"Encoding {n_el} electrodes x {n_time} time points will "
                f"allocate {n_el * n_time * 4 / 1e9:.1f} GB. Construct the "
                f"encoder with an 'implant' to encode at electrode "
                f"resolution instead.", category=UserWarning)

        # Defer expanding the schedule into an n_el x n_time matrix:
        realized = np.zeros(shape, dtype=np.float64)
        realized[firing] = MS_PER_S / (period[firing] * DT)
        return _EncodedStimulus(
            electrodes, amp, ticks, sched, onsets, frames, pulse_ticks,
            pulse_vals, total, realized, frame_time, frame_dur,
            None if cycle is None else cycle * DT, amp_unit=self.amp_unit,
            phase_dur=None if self.pulse is not None else self.phase_dur,
            cathodic_first=self.cathodic_first,
            source_time=frame_time if timed else None,
            source_dur=frame_dur if timed else None)

    def _modulation(self, source):
        """Per-electrode, per-frame amplitude and frequency

        Reduces the source to one gray level per electrode per frame, applies
        ``stretch`` and ``n_levels``, and calls ``_modulate``. Pulse timing
        and rasters are handled by :py:meth:`_assemble`.

        Returns the positional arguments of :py:meth:`_assemble`.
        """
        gray, electrodes, frame_time, frame_dur = _sampled_frames(
            source, self.implant, self.frame_dur)
        if self.stretch:
            gray = gray - gray.min()
            peak = gray.max()
            if peak > 0:
                gray = gray / peak
        if self.n_levels is not None:
            # Quantize gray levels before modulating, so `n_levels` means the
            # same thing for every subclass:
            steps = self.n_levels - 1
            gray = np.round(gray * steps) / steps
        amp, freq = self._modulate(gray)
        return amp, freq, electrodes, frame_time, frame_dur

    def encode(self, source):
        """Encode an image or a video as a train of electrical pulses

        Parameters
        ----------
        source : :py:class:`~pulse2percept.stimuli.Stimulus`
            The image or video to encode, dimensionless, with gray levels in
            [0, 1] (as produced by
            :py:class:`~pulse2percept.stimuli.ImageStimulus` and
            :py:class:`~pulse2percept.stimuli.VideoStimulus`).

        Returns
        -------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus`
            The encoded stimulus. Amplitudes are in uA, or in ``xTh`` if the
            encoder's amplitude parameters were; time is in ms.

        Raises
        ------
        :py:class:`~pulse2percept.units.DimensionMismatchError`
            If ``source`` is not dimensionless.

        """
        modulation = self._modulation(source)
        # Frames keep the source clock unless `frame_dur` retimes them:
        timed = source.time is not None and self.frame_dur is None
        return self._assemble(*modulation, timed=timed)


class AmplitudeEncoder(PulseEncoder):
    """Encode gray levels as pulse amplitudes

    Every electrode emits a pulse train of the same fixed frequency; the gray
    level at the electrode sets the pulse amplitude. This is how most retinal
    prostheses encode a video.

    Because all electrodes share one pulse period, a raster does not quantize
    ``freq``. Without ``group_dur``, the groups divide the period evenly;
    with ``group_dur``, they are packed into a shorter sweep at the start of
    every period. No two groups are active at the same time.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.Implant`, optional
        The implant to encode for. See
        :py:class:`~pulse2percept.stimuli.PulseEncoder`.
    amp_range : (min_amp, max_amp), optional
        Range of pulse amplitudes, in uA or in multiples of perceptual
        threshold (``xTh``). A gray level of 0 maps onto ``min_amp`` and a
        gray level of 1 onto ``max_amp``.

        Bare numbers mean uA. Both endpoints must have the same dimension,
        e.g. ``(0 * xTh, 3 * xTh)``; ``(0, 3 * xTh)`` is rejected. An ``xTh``
        range gives an ``xTh`` stimulus, which an implant converts to current
        if it has :py:attr:`~pulse2percept.implants.Implant.thresholds` for
        every driven electrode.

        .. versionchanged:: 0.11.0
            Accepts ``xTh`` as well as current.
    freq : float, optional
        Pulse train frequency (Hz), the same for every electrode and
        independent of the frame rate. A raster does not quantize it; only
        ``clock`` can lower it, by rounding the period up to whole clock
        cycles.

        .. note::

           With a frequency below the frame rate, some frames receive no pulse
           and their gray levels are dropped. Encoding warns when this happens.

    phase_dur, interphase_dur, cathodic_first, frame_dur, stretch
        See :py:class:`~pulse2percept.stimuli.PulseEncoder`.

    Notes
    -----
    *  Arguments may be given as plain numbers in the units documented above,
       or as unitful quantities (e.g. ``0.05 * mA``, ``460 * us``,
       ``0.02 * kHz``), which are converted to those units. See
       :py:mod:`pulse2percept.units`.

    Examples
    --------
    Encode a movie for Argus II, mapping gray levels onto 0-50 uA at 20 Hz:

    >>> import numpy as np
    >>> import pulse2percept as p2p
    >>> video = p2p.stimuli.VideoStimulus(np.random.rand(16, 20, 30),
    ...                                   metadata={'fps': 20})
    >>> implant = p2p.implants.retina.ArgusII()
    >>> encoder = p2p.stimuli.AmplitudeEncoder(implant, amp_range=(0, 50))
    >>> stim = encoder.encode(video)

    The same thing, with the implant encoding its own input:

    >>> implant.encoder = p2p.stimuli.AmplitudeEncoder(amp_range=(0, 50))
    >>> stim = implant.prepare_stim(video)

    """
    __slots__ = ('amp_range', 'freq', 'amp_unit')

    def __init__(self, implant=None, amp_range=(0, 50), freq=20, **kwargs):
        super().__init__(implant, **kwargs)
        amp_unit = self._amp_range_unit(amp_range)
        # `amp_range` is converted element-wise, so its endpoints may use
        # different (compatible) units:
        amp_range = as_value(amp_range, amp_unit, 'amp_range')
        freq = as_value(freq, Hz, 'freq')
        if np.size(amp_range) != 2:
            raise ValueError(f"'amp_range' must be a (min_amp, max_amp) "
                             f"tuple, not {amp_range}.")
        _finite('amp_range', amp_range)
        if np.any(np.asarray(amp_range) < 0):
            raise ValueError(f"'amp_range' cannot be negative: the sign of "
                             f"the pulse is set by 'cathodic_first', not by "
                             f"the amplitude. Got {amp_range}.")
        _finite('freq', freq)
        if freq < 0:
            raise ValueError("'freq' cannot be negative.")
        self.amp_range = amp_range
        self.amp_unit = amp_unit
        self.freq = freq

    @staticmethod
    def _amp_range_unit(amp_range):
        """Return the unit of amp_range (uA or xTh); mixtures raise an error"""
        dim = getattr(amp_range, 'dimension', None)
        if dim is not None:
            dims = [dim]
        else:
            try:
                dims = [getattr(a, 'dimension', uA.dimension)
                        for a in amp_range]
            except TypeError:
                # Not a pair; the shape check reports the error:
                return uA
        if all(d == xTh.dimension for d in dims):
            return xTh
        if any(d == xTh.dimension for d in dims):
            raise DimensionMismatchError(
                f"'amp_range' mixes threshold multiples with current. Give "
                f"both endpoints in xTh, as in (0 * xTh, 3 * xTh), or both in "
                f"current. Got {amp_range}.")
        return uA

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super()._pprint_params()
        params.update({'amp_range': self.amp_range, 'amp_unit': self.amp_unit,
                       'freq': self.freq})
        return params

    def _modulate(self, gray):
        """Gray level in [0, 1] -> amplitude in ``amp_range``"""
        amp_lo, amp_hi = self.amp_range
        return amp_lo + gray * (amp_hi - amp_lo), self.freq


class FrequencyEncoder(PulseEncoder):
    """Encode gray levels as pulse train frequencies

    Every electrode emits pulses of the same fixed amplitude; the gray level
    at the electrode sets the pulse frequency.

    .. important::

       Frequency modulation is far more expensive to simulate than amplitude
       modulation: electrodes at different rates pulse at different times, so
       the stimulus needs a time point at every pulse edge of every electrode.

       ``clock`` (the stimulator's time base) reduces this. Encoding a
       94-frame clip for Argus II at frequencies in (0, 300] Hz:

       =======================  ===========
       setting                  time points
       =======================  ===========
       (amplitude modulation)           442
       no quantization              143,771
       ``clock=1``                   21,505
       ``clock=2``                   10,893
       ``n_levels=8``               127,327
       ``clock=1, n_levels=8``       20,917
       =======================  ===========

       ``clock`` reduces frequency resolution, most at the top of the range:
       with ``freq_range=(0, 300)``, ``clock=1`` delivers the brightest pixels
       at 250 Hz and ``clock=2`` at 200 Hz.

       ``n_levels`` helps little: the pulse clock keeps its phase across
       frames, so electrodes on the same gray level still pulse at different
       times unless their whole history matches.

       A raster also reduces the cost by confining onsets to the raster grid.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.Implant`, optional
        The implant to encode for. See
        :py:class:`~pulse2percept.stimuli.PulseEncoder`.
    freq_range : (min_freq, max_freq), optional
        Range of pulse train frequencies (Hz). A gray level of 0 maps onto
        ``min_freq`` and a gray level of 1 onto ``max_freq``. A frequency of 0
        means no pulse at all.

        .. note::

           Realizable frequencies are quantized by ``clock`` and, with a
           raster, onto the raster sweep: every period becomes a whole number
           of sweeps, so the realizable rates are ``1000 / (m * sweep)`` Hz.

           With ``group_dur=None``, the sweep is the shortest requested
           period, so the fastest electrode keeps its rate. With an explicit
           ``group_dur``, the sweep is ``n_groups * group_dur``, so even the
           fastest electrode is generally rounded: with a six-group 1 ms
           sweep, 100 Hz (10 ms) is delivered as 83.3 Hz (12 ms, two sweeps).

           The period is always rounded up, so an electrode is never driven
           faster than requested: with a 10 ms sweep, 67 Hz becomes 50 Hz.
           Shorten ``group_dur`` for a finer grid.
    amp : float, optional
        Pulse amplitude (uA), the same for every electrode.
    phase_dur, interphase_dur, cathodic_first, pulse, clock, n_levels, \
frame_dur, stretch
        See :py:class:`~pulse2percept.stimuli.PulseEncoder`.

    Notes
    -----
    *  Arguments may be given as plain numbers in the units documented above,
       or as unitful quantities (e.g. ``0.05 * mA``, ``460 * us``,
       ``0.02 * kHz``), which are converted to those units. See
       :py:mod:`pulse2percept.units`.

    Examples
    --------
    Encode a movie for Argus II at 50 uA, mapping gray levels onto 0-300 Hz on
    a 1 ms stimulator clock. A 300 Hz period (3.3 ms) does not fit Argus II's
    six-group 2 ms raster sweep, so the raster is disabled:

    >>> import numpy as np
    >>> import pulse2percept as p2p
    >>> video = p2p.stimuli.VideoStimulus(np.random.rand(16, 20, 30),
    ...                                   metadata={'fps': 30})
    >>> implant = p2p.implants.retina.ArgusII(raster=None)
    >>> encoder = p2p.stimuli.FrequencyEncoder(implant, freq_range=(0, 300),
    ...                                        amp=50, clock=1)
    >>> stim = encoder.encode(video)

    """
    __slots__ = ('freq_range', 'amp')

    def __init__(self, implant=None, freq_range=(0, 300), amp=50, **kwargs):
        super().__init__(implant, **kwargs)
        # See `PulseEncoder.__init__`:
        freq_range = as_value(freq_range, Hz, 'freq_range')
        amp = as_value(amp, uA, 'amp')
        if np.size(freq_range) != 2:
            raise ValueError(f"'freq_range' must be a (min_freq, max_freq) "
                             f"tuple, not {freq_range}.")
        _finite('freq_range', freq_range)
        if np.any(np.asarray(freq_range) < 0):
            raise ValueError(f"'freq_range' cannot be negative: {freq_range}.")
        _finite('amp', amp)
        if amp < 0:
            raise ValueError(f"'amp' cannot be negative: the sign of the "
                             f"pulse is set by 'cathodic_first', not by the "
                             f"amplitude. Got {amp}.")
        self.freq_range = freq_range
        self.amp = amp

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super()._pprint_params()
        params.update({'freq_range': self.freq_range, 'amp': self.amp})
        return params

    def _modulate(self, gray):
        """Gray level in [0, 1] -> frequency in ``freq_range``"""
        freq_lo, freq_hi = self.freq_range
        return self.amp, freq_lo + gray * (freq_hi - freq_lo)


class _GrayFrames(Stimulus):
    """Dimensionless gray levels, one row per electrode and column per frame"""
    _default_unit = dimensionless

    __slots__ = ()


def _pixel_graph(pixels):
    """Return the 8-connected neighbor lists of (row, col) ``pixels``"""
    index = {tuple(p): i for i, p in enumerate(pixels)}
    neighbors = [[] for _ in pixels]
    for i, (r, c) in enumerate(pixels):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                j = index.get((r + dr, c + dc))
                if j is None or j == i:
                    continue
                # Skip diagonals bridged by an orthogonal pixel, or every
                # right-angle bend would look like a branch:
                bridged = (r + dr, c) in index or (r, c + dc) in index
                if dr and dc and bridged:
                    continue
                neighbors[i].append(j)
    return neighbors


def _prune_spurs(skeleton, mask):
    """Return ``skeleton`` without the spurs skeletonization adds at corners

    At a sharp corner of a thick stroke, the skeleton continues past the
    bend into the corner, reaching ``r / sin(angle / 2)`` from the junction,
    where ``r`` is the junction's distance to the background of ``mask``. An
    end branch whose tip lies within ``3 r`` is removed if it is the only such
    branch at its junction. A short real branch is removed too.
    """
    pixels = np.argwhere(skeleton)
    neighbors = _pixel_graph(pixels)
    degree = np.array([len(n) for n in neighbors])
    radius = distance_transform_edt(mask)
    spurs = {}
    for end in np.flatnonzero(degree == 1):
        branch, prev = [end], -1
        while degree[branch[-1]] == 2 or len(branch) == 1:
            nxt = [j for j in neighbors[branch[-1]] if j != prev]
            if not nxt:
                break
            prev = branch[-1]
            branch.append(nxt[0])
        junction = branch[-1]
        tip = np.hypot(*(pixels[end] - pixels[junction]))
        if degree[junction] > 2 and tip <= 3 * radius[tuple(pixels[junction])]:
            spurs.setdefault(junction, []).append(branch[:-1])
    pruned = skeleton.copy()
    for found in spurs.values():
        # Two short end branches at one junction are a drawn fork:
        if len(found) == 1:
            pruned[tuple(pixels[found[0]].T)] = False
    return pruned


def _ordered_path(skeleton):
    """Return the (row, col) pixels of a one-pixel-wide open path, in order

    Starts at the endpoint that comes first in row-major order. Rejects empty,
    disconnected, branched, and closed skeletons.
    """
    pixels = np.argwhere(skeleton)
    if not len(pixels):
        raise ValueError("The image contains no trace. Trace pixels must be "
                         "brighter than 'threshold'.")
    neighbors = _pixel_graph(pixels)
    degree = np.array([len(n) for n in neighbors])
    # Connected components, by flood fill from pixel 0:
    seen = {0}
    todo = [0]
    while todo:
        for j in neighbors[todo.pop()]:
            if j not in seen:
                seen.add(j)
                todo.append(j)
    if len(seen) != len(pixels):
        raise ValueError("The image contains more than one disconnected "
                         "trace. TraceEncoder requires a single stroke.")
    if np.any(degree > 2):
        r, c = pixels[np.argmax(degree > 2)]
        raise ValueError(f"The trace branches at pixel (row={r}, col={c}). "
                         f"TraceEncoder requires a single nonbranching "
                         f"stroke.")
    if len(pixels) == 1:
        return pixels
    ends = np.flatnonzero(degree == 1)
    if len(ends) != 2:
        raise ValueError("The trace is a closed loop, which has no start "
                         "point. Pass an (N, 2) trajectory instead.")
    path = [ends[0]]
    prev = -1
    while len(path) < len(pixels):
        nxt = [j for j in neighbors[path[-1]] if j != prev]
        if not nxt:
            break
        prev = path[-1]
        path.append(nxt[0])
    if len(set(path)) != len(pixels) or path[-1] != ends[1]:
        raise ValueError("The trace could not be ordered as a single path.")
    return pixels[path]


class TraceEncoder(Encoder):
    """Encode a visual trajectory as sequential single-electrode stimulation.

    ``TraceEncoder`` maps an ordered trajectory in the visual field to physical
    electrodes using the model's visual-field map and implant placement. The
    selected electrodes are stimulated one at a time for ``step_dur`` each.

    Unlike an :py:class:`~pulse2percept.stimuli.ImplantEncoder`,
    ``TraceEncoder`` is not attached to
    :py:attr:`~pulse2percept.implants.Implant.encoder`. Call
    :py:meth:`encode` directly.

    Input
    -----
    The target may be either:

    * An ``(N, 2)`` array of ordered ``(x, y)`` positions in dva.
    * A grayscale :py:class:`~pulse2percept.stimuli.ImageStimulus` containing
      one bright stroke on a dark background. ``extent`` places the image in
      the visual field.

    Image traces are thresholded and skeletonized to a one-pixel centerline.
    The centerline must form one open, nonbranching path. Its direction starts
    at the endpoint that appears first in row-major order (topmost, then
    leftmost).

    Thick corners can produce short spurs during skeletonization. Setting
    ``prune_spurs=True`` removes short endpoint spurs before topology is
    checked, but may also remove a short real branch. Use
    :py:meth:`trajectory` to inspect the resulting path, or pass an explicit
    ``(N, 2)`` trajectory when direction must be controlled.

    Mapping
    -------
    Each trajectory sample is mapped independently to the nearest active
    electrode with a finite inverse visual-field location. ``region`` selects
    the target region when the visual-field map contains more than one.

    Consecutive samples that map to the same electrode are collapsed into a
    single stimulation step. Revisiting that electrode later in the trajectory
    produces a new step.

    The resulting electrode sequence depends on retinotopy and implant
    placement, so its path across the physical array need not resemble the
    trajectory in visual space.

    This encoder is inspired by the dynamic letter-tracing paradigm of
    [Beauchamp2020]_. It uses physical electrodes only and does not implement
    current steering, virtual electrodes, or the stimulation protocol from
    that study.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    model : model
        Model used to map visual-field positions to physical electrodes.
        Requires an ``implant``, an invertible ``visual_field_map``
        (``from_dva`` and ``to_dva``), and implant placement parameters.
        The encoder uses ``model.implant``.
    amp : float or Quantity, optional
        Pulse amplitude in uA, or in ``xTh`` for threshold-relative
        stimulation.
    freq : float or Quantity, optional
        Pulse-train frequency (Hz).
    phase_dur : float or Quantity, optional
        Duration of each pulse phase (ms).
    step_dur : float or Quantity, optional
        Stimulation duration for each trajectory step (ms).
    interphase_dur : float or Quantity, optional
        Gap between cathodic and anodic phases (ms).
    cathodic_first : bool, optional
        If True, deliver the cathodic phase first.
    clock : float or Quantity, optional
        Stimulator clock period (ms). See
        :py:class:`~pulse2percept.stimuli.PulseEncoder`.
    region : str, optional
        Region returned by ``visual_field_map.from_dva()`` to use, e.g.
        ``'v1'``. Required when the map contains more than one region.
    threshold : float, optional
        Gray level in ``[0, 1]`` above which image pixels belong to the trace.
    prune_spurs : bool, optional
        If True, remove short endpoint spurs introduced by skeletonization
        before topology validation. This may also remove a short real branch.
        If False, any branch is rejected.

    Notes
    -----
    * Pulse trains are generated with
      :py:class:`~pulse2percept.stimuli.AmplitudeEncoder` using
      ``frame_dur=step_dur``, so the implant's raster is respected.
    * Sparse trajectory vertices are not interpolated. Sample intermediate
      points explicitly when tracing a line. Image traces contain one sample
      per skeleton pixel, so image resolution determines sampling density.
    * :py:class:`~pulse2percept.models.cortex.DynaphosModel` simulates encoded
      stimulation using its own ``freq`` and ``p_dur``. Set ``freq`` and
      ``phase_dur`` to match these values.
    * Models with ``location_noise`` are not supported.

    Examples
    --------
    Trace a horizontal line at 2 dva below fixation using Orion in the right
    hemisphere:

    >>> import numpy as np
    >>> import pulse2percept as p2p
    >>> from pulse2percept.units import mm, uA, ms
    >>> implant = p2p.implants.cortex.Orion()
    >>> model = p2p.models.cortex.DynaphosModel(
    ...     implant, implant_position=(20, -5) * mm)
    >>> encoder = p2p.stimuli.TraceEncoder(
    ...     model,
    ...     amp=100 * uA,
    ...     freq=model.freq,
    ...     phase_dur=model.p_dur,
    ...     step_dur=50 * ms)
    >>> x = np.linspace(-6, -2, 41)
    >>> trace = np.column_stack([x, np.full_like(x, -2)])
    >>> encoder.electrode_sequence(trace)
    ['41', '37', '38']
    >>> stim = encoder.encode(trace)

    The same line can be supplied as an image with one pixel per 0.1 dva:

    >>> from pulse2percept.units import dva
    >>> line = np.zeros((9, 41))
    >>> line[4, :] = 1
    >>> target = p2p.stimuli.ImageStimulus(line)
    >>> extent = (-6.05, -1.95, -2.45, -1.55) * dva
    >>> encoder.trajectory(target, extent=extent)[[0, -1]]
    array([[-6., -2.],
           [-2., -2.]])
    >>> encoder.electrode_sequence(target, extent=extent)
    ['41', '37', '38']
    >>> stim = encoder.encode(target, extent=extent)
    """
    __slots__ = ('model', 'amp', 'amp_unit', 'freq', 'phase_dur', 'step_dur',
                 'interphase_dur', 'cathodic_first', 'clock', 'region',
                 'threshold', 'prune_spurs')

    def __init__(self, model, *, amp=100 * uA, freq=300 * Hz,
                 phase_dur=0.17 * ms, step_dur=50 * ms, interphase_dur=0 * ms,
                 cathodic_first=True, clock=None, region=None, threshold=0.5,
                 prune_spurs=False):
        for attr in ('implant', 'visual_field_map', '_electrode_coords'):
            if getattr(model, attr, None) is None:
                raise TypeError(
                    f"TraceEncoder requires a model with '{attr}', which "
                    f"{type(model).__name__} does not provide. For a "
                    f"composite Model, pass its spatial model.")
        super().__init__(model.implant)
        self.model = model
        self.amp_unit = (xTh if getattr(amp, 'dimension', None) ==
                         xTh.dimension else uA)
        self.amp = as_value(amp, self.amp_unit, 'amp')
        self.freq = as_value(freq, Hz, 'freq')
        self.phase_dur = as_value(phase_dur, ms, 'phase_dur')
        self.step_dur = as_value(step_dur, ms, 'step_dur')
        self.interphase_dur = as_value(interphase_dur, ms, 'interphase_dur')
        self.cathodic_first = cathodic_first
        self.clock = as_value(clock, ms, 'clock')
        self.region = region
        for name in ('amp', 'freq'):
            value = getattr(self, name)
            _finite(name, value)
            if np.size(value) != 1 or value <= 0:
                raise ValueError(f"'{name}' must be a positive scalar, not "
                                 f"{value}.")
        _finite('threshold', threshold)
        if np.size(threshold) != 1 or not 0 <= threshold <= 1:
            raise ValueError(f"'threshold' must be a scalar in [0, 1], not "
                             f"{threshold}.")
        self.threshold = float(threshold)
        self.prune_spurs = bool(prune_spurs)
        self._check_model()
        # Validates the pulse parameters:
        self._pulse_encoder()

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super()._pprint_params()
        # Class name only: the model holds the implant, which holds encoders.
        params.update({'model': type(self.model).__name__, 'amp': self.amp,
                       'amp_unit': self.amp_unit, 'freq': self.freq,
                       'phase_dur': self.phase_dur, 'step_dur': self.step_dur,
                       'interphase_dur': self.interphase_dur,
                       'cathodic_first': self.cathodic_first,
                       'clock': self.clock, 'region': self.region,
                       'threshold': self.threshold,
                       'prune_spurs': self.prune_spurs})
        return params

    def _pulse_encoder(self):
        """Return the AmplitudeEncoder that turns one-hot frames into pulses"""
        return AmplitudeEncoder(
            self.implant, amp_range=(0 * self.amp_unit,
                                     self.amp * self.amp_unit),
            freq=self.freq, phase_dur=self.phase_dur,
            interphase_dur=self.interphase_dur,
            cathodic_first=self.cathodic_first, clock=self.clock,
            frame_dur=self.step_dur)

    def _check_model(self):
        """Reject a rebound implant or a model with location noise"""
        model = self.model
        if model.implant is not self.implant:
            raise ValueError(
                f"model.implant was rebound after this TraceEncoder was "
                f"bound to a {type(self.implant).__name__}. Construct a new "
                f"TraceEncoder for the new implant.")
        noise = getattr(model, 'location_noise', None)
        if noise is not None and noise != 0:
            raise NotImplementedError(
                f"TraceEncoder uses canonical retinotopy and implant "
                f"placement. location_noise={noise} dva adds random "
                f"phosphene offsets, which are not calibration data. Set "
                f"location_noise=None.")
        if model.visual_field_map is None:
            raise ValueError(f"{type(model).__name__} has no "
                             f"'visual_field_map'.")

    def _transforms(self):
        """Return the region name and its (dva -> tissue, tissue -> dva)"""
        vfmap = self.model.visual_field_map
        name = type(vfmap).__name__
        try:
            forward = vfmap.from_dva()
        except NotImplementedError:
            raise NotImplementedError(
                f"{name} does not map dva onto tissue.") from None
        region = self.region
        if region is None:
            if len(forward) != 1:
                raise ValueError(f"{name} maps onto regions {list(forward)}. "
                                 f"Pass 'region' to select one.")
            region = next(iter(forward))
        elif region not in forward:
            raise ValueError(f"Region {region!r} is not available in {name}, "
                             f"which maps onto {list(forward)}.")
        try:
            inverse = vfmap.to_dva()[region]
        except (NotImplementedError, KeyError):
            raise NotImplementedError(
                f"TraceEncoder requires an invertible map, but {name} does "
                f"not map region {region!r} back to dva.") from None
        return region, forward[region], inverse

    @staticmethod
    def _pointwise(transform, coords, n_out):
        """Apply a map transform one point at a time; returns (N, n_out)

        Some maps are batch-dependent (Polimeni2006Map shifts x=0 samples
        toward the batch mean), so each point is mapped on its own.
        """
        out = np.empty((len(coords), n_out))
        for i, point in enumerate(coords):
            mapped = transform(*[np.array([c]) for c in point])
            out[i] = [np.asarray(c, dtype=np.float64).ravel()[0]
                      for c in mapped[:n_out]]
        return out

    def _image_trajectory(self, image, extent):
        """Return the ordered centerline of a one-stroke image in dva"""
        if extent is None:
            raise ValueError("An ImageStimulus has no angular size. Pass "
                             "'extent' as (left, right, bottom, top) in dva.")
        extent = np.array(as_value(extent, dva, 'extent'), dtype=np.float64)
        if extent.shape != (4,) or not np.all(np.isfinite(extent)):
            raise ValueError(f"'extent' must be four finite numbers (left, "
                             f"right, bottom, top) in dva, not {extent}.")
        left, right, bottom, top = extent
        if not (left < right and bottom < top):
            raise ValueError(f"'extent' requires left < right and bottom < "
                             f"top, not {tuple(extent)}.")
        if len(image.img_shape) != 2:
            raise ValueError(f"TraceEncoder requires a grayscale image, not "
                             f"shape {image.img_shape}. Construct it with "
                             f"ImageStimulus(image, as_gray=True).")
        if image.data.size != np.prod(image.img_shape):
            raise ValueError("TraceEncoder requires a dense ImageStimulus. "
                             "Construct it with compress=False.")
        n_rows, n_cols = image.img_shape
        mask = np.asarray(image.data).reshape(image.img_shape) > self.threshold
        skeleton = skeletonize(mask)
        if self.prune_spurs:
            skeleton = _prune_spurs(skeleton, mask)
        row, col = _ordered_path(skeleton).T
        # Pixel centers, as in Scene.pixel_to_dva; row 0 is the top:
        return np.column_stack([
            left + (col + 0.5) * (right - left) / n_cols,
            top - (row + 0.5) * (top - bottom) / n_rows])

    def trajectory(self, source, *, extent=None):
        """Return the ordered visual-field trajectory of a target

        Parameters
        ----------
        source : (N, 2) array_like, Quantity, or ImageStimulus
            Ordered ``(x, y)`` samples in dva (bare numbers are dva), or a
            grayscale image of a single stroke.
        extent : ``(left, right, bottom, top)`` Quantity, optional
            Outer edges of the image in dva, as in
            :py:attr:`Scene.extent <pulse2percept.vision.Scene.extent>`.
            Required for an image; not accepted for an ``(N, 2)`` trajectory.

        Returns
        -------
        xy : (N, 2) np.ndarray
            Ordered ``(x, y)`` samples in dva. For an image, one sample per
            skeleton pixel center.

        """
        if isinstance(source, ImageStimulus):
            return self._image_trajectory(source, extent)
        if isinstance(source, Stimulus):
            raise TypeError(f"TraceEncoder encodes an (N, 2) trajectory in "
                            f"dva or an ImageStimulus, not a "
                            f"{type(source).__name__}.")
        if extent is not None:
            raise ValueError("'extent' applies to an ImageStimulus only. An "
                             "(N, 2) trajectory is already in dva.")
        xy = np.array(as_value(source, dva, 'source'), dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] == 0:
            raise ValueError(f"The trajectory must have shape (N, 2) with "
                             f"N >= 1, not {xy.shape}.")
        _finite('source', xy)
        return xy

    def _target_tissue(self, xy, region, forward):
        """Return dva trace samples as (N, ndim) tissue coordinates"""
        vfmap = self.model.visual_field_map
        try:
            tissue = self._pointwise(forward, xy, vfmap.ndim)
        except NotImplementedError:
            raise NotImplementedError(
                f"{type(vfmap).__name__} does not map dva onto region "
                f"{region!r}.") from None
        lost = np.flatnonzero(~np.all(np.isfinite(tissue), axis=1))
        if lost.size:
            raise ValueError(
                f"{type(vfmap).__name__} does not map trajectory sample(s) "
                f"{lost[:5].tolist()} (e.g., {xy[lost[0]].tolist()} dva) onto "
                f"tissue.")
        return tissue

    def _candidates(self, region, inverse):
        """Return names and tissue coordinates of selectable electrodes

        Selectable: activated, with a finite ``to_dva`` location. Maps return
        NaN for tissue outside the region.
        """
        vfmap = self.model.visual_field_map
        array = self.implant.electrode_array
        names = [n for n, e in array.electrodes.items() if e.activated]
        if not names:
            raise ValueError(f"{type(self.implant).__name__} has no activated "
                             f"electrodes.")
        xyz = np.column_stack(self.model._electrode_coords(
            array, None, electrodes=names))[:, :vfmap.ndim]
        xyz = Quantity(xyz.astype(np.float64), self.model.space_unit
                       ).to_value(vfmap.tissue_unit)
        try:
            ok = np.all(np.isfinite(self._pointwise(inverse, xyz, 2)), axis=1)
        except NotImplementedError:
            raise NotImplementedError(
                f"TraceEncoder requires an invertible map, but "
                f"{type(vfmap).__name__} does not map region {region!r} "
                f"back to dva.") from None
        if not np.any(ok):
            raise ValueError(
                f"None of the {len(names)} activated electrodes of "
                f"{type(self.implant).__name__} lies in region {region!r} of "
                f"{type(vfmap).__name__} at this implant placement.")
        return [n for n, keep in zip(names, ok) if keep], xyz[ok]

    def _sequence(self, source, extent=None):
        """Return candidate electrode names and the index of each trace step"""
        xy = self.trajectory(source, extent=extent)
        self._check_model()
        region, forward, inverse = self._transforms()
        names, xyz = self._candidates(region, inverse)
        target = self._target_tissue(xy, region, forward)
        _, nearest = cKDTree(xyz).query(target)
        # Collapse consecutive duplicates only; later revisits are kept:
        keep = np.r_[True, nearest[1:] != nearest[:-1]]
        return names, nearest[keep]

    def electrode_sequence(self, source, *, extent=None):
        """Return the electrodes a target stimulates, in order

        Parameters
        ----------
        source, extent :
            Target, as in :py:meth:`trajectory`.

        Returns
        -------
        names : list
            One electrode name per ``step_dur`` step, with consecutive
            duplicates collapsed.

        """
        names, steps = self._sequence(source, extent)
        return [names[i] for i in steps]

    def encode(self, source, *, extent=None):
        """Encode a traced target as sequential pulse trains

        Parameters
        ----------
        source, extent :
            Target, as in :py:meth:`trajectory`.

        Returns
        -------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus`
            Pulse trains in uA (or ``xTh``), one ``step_dur`` step per
            selected electrode, lasting ``n_steps * step_dur`` ms. Rows are
            the visited electrodes in electrode-array order.

        """
        names, steps = self._sequence(source, extent)
        rows, which = np.unique(steps, return_inverse=True)
        frames = np.zeros((rows.size, steps.size), dtype=np.float32)
        frames[np.ravel(which), np.arange(steps.size)] = 1
        frames = _GrayFrames(frames, electrodes=[names[i] for i in rows],
                             time=np.arange(steps.size) * self.step_dur)
        return self._pulse_encoder().encode(frames)


#: Output unit of optical encoders
_IRRADIANCE = mW / mm ** 2


class _NormalizedStimulus(Stimulus):
    """Dimensionless encoded drive for spatial models."""
    _default_unit = dimensionless
    _is_normalized_drive = True

    __slots__ = ()


class _OpticalStimulus(Stimulus):
    """Lazy pulsed-illumination schedule.

    Stores per-pixel ON duration for each pulse period, peak irradiance, and
    repetition rate. Waveform samples are generated on demand.
    """
    #: See `Stimulus._is_parametric`:
    _is_parametric = True

    #: offers a normalized time-averaged view to spatial-only models
    _has_spatial_view = True

    __slots__ = ('_dur', '_ticks', '_onsets', '_irradiance', '_freq',
                 '_wavelength', '_grayscale', '_total', '_ref_drive',
                 '_static', '_frame_time', '_frame_dur', '_time',
                 '_source_time', '_source_dur')

    def __init__(self, electrodes, dur, ticks, onsets, irradiance, freq,
                 wavelength, grayscale, total, static, frame_time, frame_dur,
                 ref_drive, source_time=None, source_dur=None):
        irradiance = float(irradiance)
        # Also checked for rebuilt/scaled schedules:
        if not math.isfinite(irradiance) or irradiance < 0:
            raise ValueError(f"'irradiance' must be a finite, nonnegative "
                             f"power density, not {irradiance}.")
        # ON duration (ms) per pixel and pulse period:
        self._dur = self._own(dur, np.float64)
        self._ticks = self._own(ticks, np.int64)
        # Onset (ticks) of every pulse period:
        self._onsets = self._own(onsets, np.int64)
        self._irradiance = irradiance
        self._freq = float(freq)
        self._wavelength = float(wavelength)
        self._grayscale = bool(grayscale)
        self._total = float(total)
        # Whether the source had a time axis of its own:
        self._static = bool(static)
        # The time-averaged irradiance `_spatial_view` calls 1.0:
        self._ref_drive = float(ref_drive)
        # Projector clock (pulse periods), not the source-video clock:
        self._frame_time = self._own(frame_time, np.float64)
        self._frame_dur = float(frame_dur)
        # Source-video frame onsets and duration (ms); None for a still:
        self._source_time = (None if source_time is None else
                             self._own(source_time, np.float64))
        self._source_dur = None if source_dur is None else float(source_dur)
        # Built lazily without rendering the waveform:
        self._time = None
        self._defer(electrodes, unit=_IRRADIANCE)
        # Metadata stores frame timing; optical settings remain schedule state.
        self.metadata['encoder'] = {'frame_time': self._frame_time,
                                    'frame_dur': self._frame_dur,
                                    **self._source_clock()}

    def _source_clock(self):
        """Source-video frame onsets and duration (ms); empty for a still"""
        if self._source_time is None:
            return {}
        return {'source_frame_time': self._source_time,
                'source_frame_dur': self._source_dur}

    @property
    def wavelength(self):
        """Wavelength (nm) of the projected light"""
        return self._wavelength

    @property
    def irradiance(self):
        """Peak irradiance (mW/mm^2) while a pixel is on"""
        return self._irradiance

    @property
    def freq(self):
        """Pulse repetition rate (Hz)"""
        return self._freq

    @property
    def pulse_dur(self):
        """ON duration (ms) of every pixel, one column per pulse period"""
        return self._dur

    @property
    def duty_cycle(self):
        """Fraction of each pulse period every pixel spends on"""
        return self._dur * self._freq / MS_PER_S

    @property
    def pulse_time(self):
        """Onset (ms) of every pulse period, one per ``pulse_dur`` column"""
        return self._onsets * DT

    @property
    def grayscale(self):
        """Whether gray levels were pulse-width modulated, not binarized"""
        return self._grayscale

    @property
    def duration(self):
        """Duration of the stimulus (ms)"""
        return self._total

    @property
    def time(self):
        """Time points of the stimulus (ms)"""
        if self._time is None:
            time = self._ticks * DT
            time[-1] = self._total
            self._time = self._own(time, np.float64)
        return self._time

    def _spatial_view(self):
        """Return normalized time-averaged optical drive per frame."""
        drive = (self._irradiance * self.duty_cycle / self._ref_drive).astype(
            np.float32)
        if self._static:
            # Collapse repeated pulse periods for a static source.
            stim = _NormalizedStimulus(drive[:, 0].ravel(),
                                       electrodes=self.electrodes)
        else:
            stim = _NormalizedStimulus(drive, electrodes=self.electrodes,
                                       time=self._frame_time)
        stim.metadata['encoder'] = {'frame_time': self._frame_time,
                                    'frame_dur': self._frame_dur,
                                    **self._source_clock()}
        return stim

    def _rebuilt(self, electrodes, dur, irradiance):
        """Return this schedule with new pixels or irradiance"""
        rebuilt = _OpticalStimulus(
            electrodes, dur, self._ticks, self._onsets, irradiance, self._freq,
            self._wavelength, self._grayscale, self._total, self._static,
            self._frame_time, self._frame_dur, self._ref_drive,
            self._source_time, self._source_dur)
        rebuilt.metadata['user'] = deepcopy(self.metadata.get('user'))
        return rebuilt

    def _scaled(self, factor):
        """Return this schedule with irradiance scaled by ``factor``."""
        if not np.isscalar(factor):
            return None
        factor = float(factor)
        if not math.isfinite(factor) or factor < 0:
            raise ValueError(f"Scaling an optical stimulus by {factor} would "
                             f"ask for a negative or undefined irradiance. "
                             f"Only nonnegative, finite factors describe "
                             f"light.")
        return self._rebuilt(self.electrodes, self._dur,
                             self._irradiance * factor)

    def _without_electrodes(self, electrodes):
        """Return this schedule without ``electrodes``"""
        keep = self._keep_mask(electrodes)
        return self._rebuilt(self.electrodes[keep], self._dur[keep],
                             self._irradiance)

    def _render(self):
        """Expand the schedule into rectangular pulses."""
        ticks = np.asarray(self._ticks)
        n_frames = self._onsets.size
        # Map stored time points to pulse periods.
        at = np.searchsorted(self._onsets, ticks, side='right') - 1
        np.clip(at, 0, n_frames - 1, out=at)
        # Keep durations per frame to avoid an n_electrodes x n_time int64 array.
        dur = np.round(self._dur / DT).astype(np.int64)
        data = np.zeros((dur.shape[0], ticks.size), dtype=np.float32)
        # Each pulse period occupies a contiguous time span.
        bounds = np.searchsorted(at, np.arange(n_frames + 1))
        irradiance = np.float32(self._irradiance)
        for j in range(n_frames):
            lo, hi = bounds[j], bounds[j + 1]
            if hi <= lo:
                continue
            # Time since the current pulse-period onset.
            since = ticks[lo:hi] - self._onsets[j]
            # Match the one-DT rise/fall convention used by other stimuli.
            np.copyto(data[:, lo:hi], irradiance,
                      where=(since >= 1) & (since <= dur[:, j, None] - 1))
        # ``data`` is newly allocated and can be adopted without copying.
        return {'data': _adoptable(data), 'electrodes': self.electrodes,
                'time': self.time}

    def _pprint_params(self):
        """Return a dict of class attributes to pretty-print"""
        return {'electrodes': self.electrodes,
                'n_frames': self._dur.shape[1],
                'n_time': self._ticks.size,
                'irradiance': self._irradiance,
                'freq': self._freq,
                'duration': self._total,
                'metadata': self.metadata}


class PhotovoltaicEncoder(ImplantEncoder):
    """Encode image/video gray levels as pulsed optical stimulation

    Photovoltaic subretinal arrays are driven by pulsed near-infrared light.
    This encoder samples an image or video at the implant's pixel locations
    and returns irradiance in ``mW/mm^2``.
    Peak irradiance is fixed; gray level sets the ON duration within each
    pulse period.

    .. versionadded:: 0.11.0

    .. versionchanged:: 0.11.0
        The implant is passed to the constructor rather than to
        :py:meth:`encode`. The optical parameters are keyword-only.

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.Implant`, optional
        The implant to encode for. Its pixel locations are used to sample
        images and videos and its pixel names label the result. If None,
        every pixel of the source is its own photovoltaic pixel.
    irradiance : float or Quantity
        Peak irradiance (mW/mm^2) while a pixel is on.
    freq : float or Quantity
        Pulse repetition rate (Hz).
    pulse_dur : float or Quantity
        ON duration (ms) of a fully lit pixel. Must fit into ``1000 / freq``.
    wavelength : float or Quantity
        Wavelength (nm) of the illumination. Photovoltaic pixel response is
        wavelength dependent, so there is no generic default.
    grayscale : bool, optional
        If True (default), map gray levels to ON duration. If False, use
        binary off/on encoding.
    threshold : float, optional
        Binary-mode threshold. The default 0.5 is a pulse2percept convention.

    Notes
    -----
    *  Grayscale mode scales ON duration linearly, ``gray * pulse_dur``. This
       is a pulse2percept convention: the source studies report
       fixed-duration pulses and no grayscale transfer function. Continuous
       durations produce more time points than a device that quantizes them
       (see :py:class:`~pulse2percept.stimuli.PRIMAEncoder`).
    *  Videos are sampled at the pulse rate using zero-order hold.
    *  ``_spatial_view`` returns normalized time-averaged optical drive, where
       1.0 is a fully lit pixel at these settings (``ref_drive``). It is
       neither retinal current nor perceptual brightness.
    *  Photovoltaic conversion to retinal current is not modeled.

    Examples
    --------
    >>> from pulse2percept.implants.retina import Lorach2015Array
    >>> from pulse2percept.stimuli import PhotovoltaicEncoder, samples
    >>> encoder = PhotovoltaicEncoder(Lorach2015Array(), irradiance=4,
    ...                               freq=40, pulse_dur=4, wavelength=915)
    >>> encoder.encode(samples.logo_bvl()).unit
    mW/mm^2

    """
    __slots__ = ('irradiance', 'freq', 'pulse_dur', 'wavelength', 'grayscale',
                 'threshold')

    def __init__(self, implant=None, *, irradiance, freq, pulse_dur,
                 wavelength, grayscale=True, threshold=0.5):
        super().__init__(implant)
        irradiance = as_value(irradiance, _IRRADIANCE, 'irradiance')
        freq = as_value(freq, Hz, 'freq')
        pulse_dur = as_value(pulse_dur, ms, 'pulse_dur')
        wavelength = as_value(wavelength, nm, 'wavelength')
        threshold = as_value(threshold, dimensionless, 'threshold')
        _finite('irradiance', irradiance)
        if irradiance <= 0:
            raise ValueError("'irradiance' must be positive.")
        _finite('freq', freq)
        if freq <= 0:
            raise ValueError("'freq' must be positive.")
        _finite('wavelength', wavelength)
        if wavelength <= 0:
            raise ValueError("'wavelength' must be positive.")
        _finite('pulse_dur', pulse_dur)
        if pulse_dur < 0:
            raise ValueError("'pulse_dur' cannot be negative.")
        if pulse_dur > 0:
            self._check_pulse_dur(pulse_dur, freq)
        _finite('threshold', threshold)
        if not 0 <= threshold <= 1:
            raise ValueError(f"'threshold' must be a gray level in [0, 1], "
                             f"not {threshold}.")
        self.irradiance = irradiance
        self.freq = freq
        self.pulse_dur = pulse_dur
        self.wavelength = wavelength
        self.grayscale = bool(grayscale)
        self.threshold = threshold

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super()._pprint_params()
        params.update({'irradiance': self.irradiance, 'freq': self.freq,
                       'pulse_dur': self.pulse_dur,
                       'wavelength': self.wavelength,
                       'grayscale': self.grayscale,
                       'threshold': self.threshold})
        return params

    def _check_pulse_dur(self, pulse_dur, freq):
        """Reject a pulse (ms) that does not fit into one period"""
        period = MS_PER_S / freq
        if pulse_dur >= period:
            raise ValueError(
                f"A {pulse_dur:g} ms pulse does not fit into the "
                f"{period:.3f} ms period of a {freq:g} Hz pulse train. "
                f"Shorten 'pulse_dur' or lower 'freq'.")

    @property
    def period(self):
        """Pulse period (ms)"""
        return MS_PER_S / self.freq

    @property
    def ref_drive(self):
        """Time-averaged irradiance (mW/mm^2) that maps to 1.0 in the
        normalized view

        A fully lit pixel at these settings. Devices with a documented
        projector maximum use that maximum instead.
        """
        drive = self.irradiance * self.pulse_dur * self.freq / MS_PER_S
        # Avoid dividing by zero; a dark schedule has zero drive anyway:
        return drive if drive > 0 else 1.0

    def _durations(self, gray):
        """Map gray levels in [0, 1] to ON durations (ms)

        Binary mode lights a pixel for the full ``pulse_dur``; grayscale mode
        scales it linearly, which is a pulse2percept convention.
        """
        # Use float64 so durations survive rounding onto the DT time grid.
        gray = np.asarray(gray, dtype=np.float64)
        if not self.grayscale:
            return np.where(gray >= self.threshold, self.pulse_dur, 0.0)
        # Gray levels are already clipped to [0, 1].
        return gray * self.pulse_dur

    def encode(self, source):
        """Encode an image or a video as near-infrared irradiance

        Parameters
        ----------
        source : :py:class:`~pulse2percept.stimuli.Stimulus`
            The image or video to encode, dimensionless, with gray levels in
            [0, 1] (as produced by
            :py:class:`~pulse2percept.stimuli.ImageStimulus` and
            :py:class:`~pulse2percept.stimuli.VideoStimulus`).

        Returns
        -------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus`
            The projected irradiance (``mW/mm^2``), time in ms. The waveform
            is generated lazily.

        Raises
        ------
        :py:class:`~pulse2percept.units.DimensionMismatchError`
            If ``source`` is not dimensionless.

        """
        period = self.period
        gray, electrodes, frame_time, frame_dur = _sampled_frames(
            source, self.implant)
        n_el = len(electrodes)
        static = frame_time.size == 1 and getattr(source, 'time', None) is None
        # Preserve source start time and duration.
        start = float(frame_time[0])
        total = float(frame_time[-1] + frame_dur)
        n_periods = max(1, int(np.ceil((total - start) / period - 1e-9)))
        onset_ms = start + np.arange(n_periods, dtype=np.float64) * period
        # Sample source frames at pulse onsets using zero-order hold.
        at = np.searchsorted(frame_time, onset_ms, side='right') - 1
        np.clip(at, 0, frame_time.size - 1, out=at)
        dur = self._durations(gray[:, at])

        # Round absolute onsets to DT to avoid accumulated period error.
        onsets = np.round(onset_ms / DT).astype(np.int64)
        end = int(np.round(total / DT))
        dur_ticks = np.round(dur / DT).astype(np.int64)
        # Drop final pulses that do not fit without truncation or off-grid timing.
        fits = dur_ticks <= (end - onsets)[np.newaxis, :]
        dur = np.where(fits, dur, 0.0)
        dur_ticks = np.where(fits, dur_ticks, 0)
        edges = [np.array([0, end], dtype=np.int64)]
        for j in range(n_periods):
            levels = np.unique(dur_ticks[:, j])
            levels = levels[levels > 0]
            if levels.size == 0:
                # No edges for a dark frame.
                continue
            on = onsets[j]
            edges.append(np.concatenate(([on, on + 1],
                                         levels + (on - 1), levels + on)))
        ticks = np.unique(np.concatenate(edges))
        ticks = ticks[(ticks >= 0) & (ticks <= end)]

        n_time = ticks.size
        if n_time > _BIG_TIME:
            _warn_external(
                f"This stimulus has {n_time} time points, which every model "
                f"downstream will pay for. A lower 'freq' or a shorter source "
                f"is the lever that helps most; so is a source with fewer "
                f"distinct gray levels, since each one needs its own ON "
                f"duration.", category=UserWarning)
        if n_el * n_time > _BIG_STIM and self.implant is None:
            _warn_external(
                f"Encoding {n_el} pixels x {n_time} time points will allocate "
                f"{n_el * n_time * 4 / 1e9:.1f} GB. Construct the encoder "
                f"with an 'implant' to encode at pixel resolution instead.",
                category=UserWarning)

        # Keep the pulse schedule lazy; render waveform samples on demand.
        return _OpticalStimulus(
            electrodes, dur, ticks, onsets, self.irradiance, self.freq,
            self.wavelength, self.grayscale, total, static,
            np.zeros(1) if static else onset_ms,
            total if static else period, self.ref_drive,
            None if static else frame_time, None if static else frame_dur)


class PRIMAEncoder(PhotovoltaicEncoder):
    """Encode image/video gray levels for the PRIMA projector.

    PRIMA uses 880 nm illumination rather than injected current. The projector
    uses fixed peak irradiance and pulse-width modulation [Palanker2020]_,
    [Holz2026]_. This encoder returns irradiance in ``mW/mm^2``.

    Unlike :py:class:`~pulse2percept.stimuli.PhotovoltaicEncoder`, ON
    durations are quantized onto the projector's 0.7 ms grid, and normalized
    drive is referenced to the projector maximum.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.Implant`, optional
        The implant to encode for. See
        :py:class:`~pulse2percept.stimuli.PhotovoltaicEncoder`.
    irradiance : float or Quantity, optional
        Peak irradiance (mW/mm^2) while a pixel is on.
    freq : float or Quantity, optional
        Projector frame rate (Hz).
    pulse_dur : float or Quantity, optional
        Maximum ON duration (ms). Must lie on the ``pulse_step`` grid and not
        exceed ``max_pulse_dur``.
    grayscale : bool, optional
        If True (default), map gray levels to pulse duration. If False, use
        binary off/on encoding.
    threshold : float, optional
        Binary-mode threshold. The default 0.5 is a pulse2percept convention.

    Notes
    -----
    *  Pivotal-system defaults are 3.5 mW/mm^2, 30 Hz, and 14 nonzero ON
       durations from 0.7 to 9.8 ms.
    *  Grayscale mode maps normalized intensity linearly to these duration
       levels. The clinical camera-to-pulse-duration transfer function is not
       published.
    *  Videos are sampled at the projector clock using zero-order hold.
    *  ``_spatial_view`` returns normalized time-averaged optical drive for
       spatial models. It is neither retinal current nor perceptual
       brightness.
    *  Clinical image preprocessing is outside this encoder.

    Examples
    --------
    >>> from pulse2percept.implants.retina import PRIMAPivotal
    >>> from pulse2percept.stimuli import PRIMAEncoder, samples
    >>> PRIMAEncoder(PRIMAPivotal()).encode(samples.logo_bvl()).unit
    mW/mm^2

    """
    #: Smallest nonzero ON duration (ms); all durations are multiples of it
    pulse_step = 0.7

    #: Longest documented ON duration (ms), i.e. 14 steps
    max_pulse_dur = 9.8

    #: Wavelength (nm) the projector illuminates at
    projector_wavelength = 880.0

    #: Peak irradiance (mW/mm^2) of the pivotal-trial projector
    max_irradiance = 3.5

    #: Frame rate (Hz) of the pivotal-trial projector
    max_freq = 30.0

    #: Largest documented duty cycle, ``max_freq * max_pulse_dur``
    max_duty_cycle = max_freq * max_pulse_dur / MS_PER_S

    #: Time-averaged irradiance (mW/mm^2) that maps to 1.0 in the normalized
    #: spatial view (projector maximum)
    ref_drive = max_irradiance * max_duty_cycle

    __slots__ = ()

    def __init__(self, implant=None, irradiance=3.5 * mW / mm ** 2,
                 freq=30 * Hz, pulse_dur=9.8 * ms, grayscale=True,
                 threshold=0.5):
        # Wavelength is fixed by the projector:
        super().__init__(implant, irradiance=irradiance, freq=freq,
                         pulse_dur=pulse_dur,
                         wavelength=self.projector_wavelength * nm,
                         grayscale=grayscale, threshold=threshold)

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super()._pprint_params()
        del params['wavelength']
        return params

    def _check_pulse_dur(self, pulse_dur, freq):
        """Also require a duration on the projector's grid (no rounding)"""
        steps = pulse_dur / self.pulse_step
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError(
                f"'pulse_dur' must be a whole multiple of "
                f"{self.pulse_step} ms, the step the projector modulates "
                f"in, not {pulse_dur:g} ms.")
        if pulse_dur > self.max_pulse_dur + 1e-9:
            raise ValueError(
                f"'pulse_dur' cannot exceed {self.max_pulse_dur} ms, the "
                f"longest documented ON duration, not {pulse_dur:g} ms.")
        super()._check_pulse_dur(pulse_dur, freq)

    @property
    def n_levels(self):
        """Number of nonzero ON durations available up to ``pulse_dur``"""
        return int(round(self.pulse_dur / self.pulse_step))

    def _durations(self, gray):
        """Map gray levels in [0, 1] to ON durations (ms)

        Binary mode uses the full ``pulse_dur``; grayscale mode rounds onto
        the projector's duration grid.
        """
        # Use float64 so durations land exactly on the hardware grid.
        gray = np.asarray(gray, dtype=np.float64)
        if not self.grayscale:
            return np.where(gray >= self.threshold, self.pulse_dur, 0.0)
        # Gray levels are already clipped to [0, 1].
        return np.round(gray * self.n_levels) * self.pulse_step
