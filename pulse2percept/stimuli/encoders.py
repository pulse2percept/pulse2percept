""":py:class:`~pulse2percept.stimuli.Encoder`,
   :py:class:`~pulse2percept.stimuli.StimulusEncoder`,
   :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`,
   :py:class:`~pulse2percept.stimuli.FrequencyEncoder`,
   :py:class:`~pulse2percept.stimuli.PRIMAEncoder`"""
from abc import ABCMeta, abstractmethod
import math
import numpy as np
from copy import deepcopy

from .base import Stimulus, _adoptable
from .images import ImageStimulus
from .pulses import BiphasicPulse
from .videos import VideoStimulus
from ..units import (DimensionMismatchError, Hz, as_value, dimensionless, mW,
                     mm, ms, uA, xTh)
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
    """Reject NaN and infinity, which slip through every ``<`` comparison"""
    if not np.all(np.isfinite(np.asarray(value, dtype=np.float64))):
        raise ValueError(f"'{name}' must be finite, not {value}.")


def _all_equal(a):
    """Whether every element of ``a`` is the same value

    Empty counts as equal: there is nothing there to differ.
    """
    return a.size == 0 or bool(np.all(a == a.flat[0]))


def _fps(metadata):
    """Frame rate recorded in a (possibly wrapped) stimulus metadata dict

    ``Stimulus`` stores metadata it does not recognize under a ``'user'`` key,
    so the frame rate that ``VideoStimulus`` picked up from the movie file sits
    at the top level on the video itself but one level down on anything derived
    from it. Returns None if neither carries one.
    """
    if not isinstance(metadata, dict):
        return None
    if 'fps' in metadata:
        return metadata['fps']
    user = metadata.get('user')
    return user.get('fps') if isinstance(user, dict) else None


class _EncodedStimulus(Stimulus):
    """A resolved encoder schedule, expanded into a waveform on demand"""
    #: whether the stimulus is described by its schedule rather than by its
    #: samples
    _is_parametric = True

    #: whether the stimulus has a special spatial view
    _has_spatial_view = True

    __slots__ = ('_amp', '_ticks', '_sched', '_onsets', '_frames',
                 '_pulse_ticks', '_pulse_vals', '_total', '_freq',
                 '_frame_time', '_frame_dur', '_time', '_phase_dur',
                 '_cathodic_first')

    def __init__(self, electrodes, amp, ticks, sched, onsets, frames,
                 pulse_ticks, pulse_vals, total, freq, frame_time,
                 frame_dur, cycle, amp_unit=uA, phase_dur=None,
                 cathodic_first=True):
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
        # Built lazily without rendering the waveform:
        self._time = None
        self._defer(electrodes, unit=amp_unit)
        self.metadata['encoder'] = {'frame_time': self._frame_time,
                                    'frame_dur': self._frame_dur,
                                    'cycle': cycle}

    @property
    def _firing(self):
        """Electrode-frames whose pulse clock is running"""
        return self._freq > 0

    def _spatial_view(self):
        """One column per frame of the source and one row per electrode"""
        data = np.where(self._firing, self._amp, np.float32(0)).astype(
            np.float32)
        # A source with a single frame has no time axis of its own
        if self._frame_time.size > 1:
            stim = Stimulus(data, electrodes=self.electrodes,
                            time=self._frame_time)
        else:
            stim = Stimulus(data.ravel(), electrodes=self.electrodes)
        stim.metadata['encoder'] = {'frame_time': self._frame_time,
                                    'frame_dur': self._frame_dur}
        return stim._inherit_units(self)

    def _rebuilt(self, electrodes, amp, sched, freq, amp_unit=None):
        """This schedule, driving different electrodes or amplitudes

        ``amp_unit`` defaults to the unit this schedule already uses; only
        threshold calibration changes it.
        """
        rebuilt = _EncodedStimulus(
            electrodes, amp, self._ticks, sched, self._onsets, self._frames,
            self._pulse_ticks, self._pulse_vals, self._total, freq,
            self._frame_time, self._frame_dur,
            self.metadata['encoder']['cycle'],
            amp_unit=self.unit if amp_unit is None else amp_unit,
            phase_dur=self._phase_dur,
            cathodic_first=self._cathodic_first)
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
        """Calibrate a threshold-relative schedule to uA without rendering."""
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
        """This schedule, delivering amplitudes scaled by ``factor``"""
        return self._rebuilt(self.electrodes, self._amp * factor, self._sched,
                             self._freq)

    def _without_electrodes(self, electrodes):
        """This schedule, no longer driving ``electrodes``"""
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
            wave = StimulusEncoder._sample(onset, self._pulse_ticks,
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


class Encoder(PrettyPrint, metaclass=ABCMeta):
    """Base class for image and video encoders.

    Encoders map dimensionless visual input to the physical quantity driving an
    implant. Subclasses implement :py:meth:`encode`.

    .. versionadded:: 0.11.0
    """
    __slots__ = ()

    @abstractmethod
    def encode(self, source, implant=None):
        """Encode an image or a video as stimulation

        Parameters
        ----------
        source : :py:class:`~pulse2percept.stimuli.Stimulus`
            The image or video to encode. Must be dimensionless.
        implant : :py:class:`~pulse2percept.implants.Implant`, optional
            The implant to encode for. If None, every pixel of the source is
            treated as its own stimulation site.

        Returns
        -------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus`
            The encoded stimulus, in the unit that drives the device.
        """
        raise NotImplementedError

    def _as_frames(self, source, implant=None, frame_dur=None):
        """Reduce a source to one gray level per electrode per frame.

        Parameters
        ----------
        frame_dur : float, optional
            Frame duration (ms) to impose on the source. If None, a source
            with a time axis keeps its own frame timing and one without a
            time axis is presented for ``_DEFAULT_FRAME_DUR`` ms.

        Returns
        -------
        gray : (n_electrodes, n_frames) array
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


class StimulusEncoder(Encoder):
    """Abstract base class for stimulus encoders.

    Encoders map image or video gray levels to electrical pulse trains.
    If an implant is supplied, the source is sampled at its electrode
    locations and scheduled using its raster pattern.

    Subclasses implement :meth:`_modulate`, which maps gray levels to
    pulse amplitude and frequency.

    .. versionadded:: 0.10.0

    Parameters
    ----------
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

    def __init__(self, phase_dur=0.46, interphase_dur=0,
                 cathodic_first=True, pulse=None, clock=None, n_levels=None,
                 frame_dur=None, stretch=False):
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
        return {'phase_dur': self.phase_dur,
                'interphase_dur': self.interphase_dur,
                'cathodic_first': self.cathodic_first, 'pulse': self.pulse,
                'clock': self.clock, 'n_levels': self.n_levels,
                'frame_dur': self.frame_dur, 'stretch': self.stretch}

    @abstractmethod
    def _modulate(self, gray):
        """Map gray levels to pulse amplitude and frequency.

        Parameters
        ----------
        gray : (n_electrodes, n_frames) array
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

        Every time point in the assembled stimulus is an integer number of
        ``DT``. That is not a loss of precision -- ``Stimulus`` already refuses
        to hold two time points closer together than ``DT`` -- and it makes the
        union of several electrodes' time axes an exact integer operation
        rather than a float comparison against a tolerance.
        """
        return np.round(np.asarray(t, dtype=np.float64) / DT).astype(np.int64)

    def _unit_pulse(self):
        """The pulse to repeat, as (ticks, values) with unit peak amplitude

        The values peak at -1 (cathodic first) or +1, so that multiplying by an
        amplitude in uA yields the pulse to deliver. The ticks start at zero,
        whatever the supplied pulse's time axis did.
        """
        if self.pulse is None:
            pulse = BiphasicPulse(1, self.phase_dur,
                                  interphase_dur=self.interphase_dur,
                                  cathodic_first=self.cathodic_first)
            values = pulse.data.ravel().astype(np.float32)
        else:
            pulse = self.pulse
            # `Stimulus.data` hands out the container itself, so copy before
            # normalizing or the user's pulse is rescaled along with ours:
            values = pulse.data.ravel().astype(np.float32).copy()
            peak = np.abs(values).max()
            if peak > 0:
                values /= peak
        ticks = self._ticks(pulse.time)
        if np.any(np.diff(ticks) < 1):
            raise ValueError(f"'pulse' has time points closer together than "
                             f"DT={DT} ms, which the simulation cannot "
                             f"resolve. Lengthen 'phase_dur'.")
        # `Stimulus` only requires a time axis to be ordered, not to start at
        # zero. What is borrowed from the supplied pulse is its shape, so
        # anchor it at zero rather than rendering every copy of it that far
        # into the train:
        ticks = ticks - ticks[0]
        # A pulse is tiled into a train, and the gaps between the copies carry
        # whatever its end points carry. Anything but zero would smear across
        # the whole train:
        if values[0] != 0 or values[-1] != 0:
            raise ValueError("'pulse' must start and end at zero amplitude, "
                             "since it is repeated to fill a train.")
        return ticks, values

    def _periods(self, freq, pulse_len):
        """Pulse period for every electrode and frame

        The period is carried as a (possibly fractional) number of ticks rather
        than being rounded onto the ``DT`` grid, and each pulse onset is
        rounded only when it is placed. Rounding the period instead would let
        the error accumulate: a 30 Hz period is 33333.33 ticks, and stepping by
        33333 of them drifts a third of a tick per pulse, which is enough to
        walk the train off the frame it belongs to over the course of a video.

        Returns
        -------
        firing : (n_electrodes, n_frames) bool array
            Whether the pulse clock is running at all.
        period : (n_electrodes, n_frames) float array
            The period, in ticks, or 0 where the clock is not running.

        """
        firing = freq > 0
        period = np.zeros(freq.shape, dtype=np.float64)
        # Hz to a period in ms, and ms to ticks of the DT grid:
        period[firing] = MS_PER_S / freq[firing] / DT
        # A clocked stimulator can only realize a period that is a whole number
        # of clock cycles, which is what keeps the number of distinct schedules
        # (and hence of time points) down. Round the period *up*:
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

        The sweep has to fit inside the shortest pulse period anyone asked for.
        With no explicit ``group_dur`` it *is* that period, split evenly between
        the groups; with one, it is ``n_groups * group_dur`` and generally much
        shorter than the period.

        Returns
        -------
        offset : (n_electrodes,) float array
            How far behind group 0 (in ticks) each electrode may start a pulse.
        cycle : float or None
            The sweep in ticks. Periods that differ from one another are
            quantized onto it by ``_assemble``, so that groups cannot drift
            together; a period they all share is left exactly as asked. None
            when there is nothing to multiplex.

        """
        zero = np.zeros(len(electrodes), dtype=np.float64)
        if raster is None or raster.n_groups < 2 or not np.any(firing):
            return zero, None
        # Derived from the requested frequencies, not from the amplitudes:
        # whether a raster is a workable schedule is a property of the device,
        # not of how bright today's video happens to be.
        fastest = float(np.min(period[firing]))
        group = np.asarray(raster.groups(electrodes), dtype=np.int64)
        if (group.min(initial=0) < 0 or
                group.max(initial=0) >= raster.n_groups):
            raise ValueError(f"'groups' must be in 0..{raster.n_groups - 1}.")
        slot = raster.slot_dur(fastest * DT) / DT
        if self.clock is not None:
            tick = self.clock / DT
            if raster.group_dur is not None:
                # An explicit slot is the primitive the cycle is made of, so
                # round *it* onto the clock and rebuild the cycle from the
                # result:
                slot = max(1.0, round(slot / tick)) * tick
            else:
                # Splitting the cycle evenly generally lands the group
                # boundaries between clock edges, and a stimulator can only
                # start a pulse on one. Take the largest whole number of clock
                # cycles that still fits every group into the period (floor):
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
        # Check the slot the hardware will actually use, not the one that was
        # asked for: a 5.1 ms slot on a 1 ms clock is a 5 ms slot, and two of
        # those do fit into a 10 ms period even though two of 5.1 ms do not.
        if cycle > fastest * (1 + 1e-9):
            raise ValueError(
                f"A raster of {raster.n_groups} groups {slot * DT:.3f} ms "
                f"apart takes {cycle * DT:.3f} ms to get through, which does "
                f"not fit into the {fastest * DT:.3f} ms pulse period. Shorten "
                f"'group_dur', use fewer groups, or lower the frequency.")
        offset = group.astype(np.float64) * slot
        # Every group's turn has to be long enough to finish a pulse in, and
        # each pulse has to clear the next group's turn by a whole tick:
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
            The spacing of the onsets this schedule is allowed to use -- the
            raster sweep, or failing that the stimulator's clock. A schedule
            that goes silent has to come back onto it.

        Returns
        -------
        onset : (n_pulses,) int array
            The tick each pulse begins at.
        frame : (n_pulses,) int array
            The frame whose modulation parameters each pulse carries.

        """
        n_frames = frame_ticks.size
        empty = np.zeros(0, dtype=np.int64)
        # Fast path: one period for the whole stimulus, which is what amplitude
        # modulation always produces. The onsets are then an arithmetic
        # sequence, and only the frames that deliver nothing drop out of it:
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
        # Frequency modulation: the rate is piecewise constant over the video's
        # frames, so track the *phase* of the pulse clock rather than jumping
        # straight to the next pulse. Phase advances at 1/period, which changes
        # the instant a frame boundary goes by; a pulse fires whenever it
        # reaches 1.
        onset, frame = [], []
        # A full phase to begin with, so that a schedule's first pulse lands at
        # the start of its slot rather than one period into it. Start counting
        # in the frame that actually contains that slot:
        phase, t = 1.0, float(start)
        k = int(np.searchsorted(frame_ticks, round(t), side='right')) - 1
        k = min(max(k, 0), n_frames - 1)
        prev = -np.inf
        while k < n_frames and t <= last:
            # How far this frame reaches, and how much phase it can supply:
            edge = float(frame_ticks[k + 1]) if k + 1 < n_frames else np.inf
            rate = 1.0 / period[k] if period[k] > 0 else 0.0
            if rate == 0.0:
                # A stopped clock supplies no phase, so nothing can come due
                # here however long the frame is. Whatever phase had built up
                # waits for the frame that starts the clock again:
                if not np.isfinite(edge):
                    break
                t, k = edge, k + 1
                continue
            due = phase + (edge - t) * rate
            if due < 1.0:
                # The frame runs out before the next pulse comes due, so carry
                # the phase across the boundary and pick the new rate up there:
                if not np.isfinite(edge):
                    break
                phase, t, k = due, edge, k + 1
                continue
            # The pulse comes due inside this frame. Snap it forward onto the
            # grid this schedule is allowed to use (i.e., the raster cycle, or
            # the stimulator's clock. Forward rather than to the nearest point,
            # because a grid is a timing constraint and no timing constraint may
            # deliver a pulse earlier (and so at a higher rate) than asked for.
            cross = t + (1.0 - phase) / rate
            tick = int(round(start + grid * np.ceil(
                (cross - start) / grid - 1e-9)))
            if tick <= prev:
                # Never let the grid stall or reverse the train:
                tick = int(round(prev + grid))
            if tick > last:
                break
            # Which frame the pulse lands in is decided by where it actually
            # goes, not where it came due: snapping can carry it over a boundary
            # into a frame that wants something else entirely.
            j = int(np.searchsorted(frame_ticks, tick, side='right')) - 1
            j = min(max(j, 0), n_frames - 1)
            if period[j] <= 0:
                # The pulse landed in a frame whose clock is stopped. Hold it
                # rather than spending it there: a frame at 0 Hz should neither
                # be given a pulse it did not ask for nor swallow one, and the
                # frame that starts the clock again should come up at its full
                # rate rather than a period short.
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
        # Back-to-back pulses share an end point, which both copies of the
        # pulse put at zero:
        t, keep = np.unique(t, return_index=True)
        return np.interp(ticks, t, v[keep])

    def _assemble(self, amp, freq, electrodes, frame_time, frame_dur,
                  implant=None):
        """Build the pulse trains for every electrode and frame

        Electrodes that pulse at the same times share the shape of their
        waveform; only the amplitude that scales it differs. So rather than
        building one waveform per electrode, this builds one per distinct
        schedule and indexes into them, which is what keeps frequency
        modulation (thousands of electrode-frames, a few dozen schedules)
        tractable.

        The time axis is global rather than per-frame. Pulses live at absolute
        times and no two frames' pulses coincide.
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
        # The stimulus lasts exactly as long as the source did. Flooring (with
        # an epsilon so a duration that is a whole number of ticks does not
        # lose one to binary rounding) leaves at least one tick between the
        # last pulse and the end point that pins the duration:
        end = int(np.floor(total / DT + 1e-9))
        last = end - 1 - pulse_len
        if last < 0:
            raise ValueError(f"A pulse (dur={pulse_len * DT:.3f} ms) does not "
                             f"fit into a stimulus of {total:.3f} ms. Shorten "
                             f"'phase_dur' or lengthen the source.")
        firing, period = self._periods(freq, pulse_len)
        # An electrode delivering no current has nothing to schedule. Its clock
        # keeps running ("firing"), so it stays in phase with its neighbors,
        # but it costs no pulses and no time points:
        active = firing & (amp != 0)
        # The implant is the one source of truth for how the device schedules
        # its electrodes:
        offset, cycle = self._raster_grid(electrodes, period, firing, pulse_len,
                                          getattr(implant, 'raster', None))
        if cycle is not None and not _all_equal(period[firing]):
            # Electrodes on different periods drift relative to one another,
            # and two groups would eventually land on the same instant. Pinning
            # every period to a whole number of raster cycles is what stops
            # that. Round the period up rather than to the nearest cycle:
            period[firing] = cycle * np.maximum(
                1.0, np.ceil(period[firing] / cycle - 1e-9))

        # Everything about when an electrode pulses is fixed by its slot and by
        # the period/activity it carries through the frames:
        key = np.concatenate([offset[:, np.newaxis], period,
                              active.astype(np.float64)], axis=1)
        uniq, sched = np.unique(key, axis=0, return_inverse=True)
        # NumPy has changed the shape `return_inverse` comes back with between
        # 2.x releases, and it indexes rows below:
        sched = np.ravel(sched)
        origin = float(frame_ticks[0])
        # The grid a schedule's onsets live on: the raster cycle if there is
        # one, else the stimulator's clock, else the simulation's own step.
        grid = (cycle if cycle is not None else
                (self.clock / DT if self.clock is not None else 1.0))
        onsets, frames = [], []
        for row in uniq:
            onset, frame = self._onsets(
                origin + row[0], row[1:1 + n_frames],
                row[1 + n_frames:].astype(bool), frame_ticks, last, grid)
            onsets.append(onset)
            frames.append(frame)
        # A frame whose gray levels never reach an electrode is content thrown
        # away, which is what asking for a pulse rate below the frame rate
        # does. It is no longer a limit on the rate itself:
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
        if n_el * n_time > _BIG_STIM and implant is None:
            _warn_external(
                f"Encoding {n_el} electrodes x {n_time} time points will "
                f"allocate {n_el * n_time * 4 / 1e9:.1f} GB. Pass 'implant' "
                f"to encode at electrode resolution instead.",
                category=UserWarning)

        # The schedule is settled. Expanding it into an n_el x n_time matrix
        # is the expensive half, and the half nothing needs until somebody
        # asks for samples:
        realized = np.zeros(shape, dtype=np.float64)
        realized[firing] = MS_PER_S / (period[firing] * DT)
        return _EncodedStimulus(
            electrodes, amp, ticks, sched, onsets, frames, pulse_ticks,
            pulse_vals, total, realized, frame_time, frame_dur,
            None if cycle is None else cycle * DT, amp_unit=self.amp_unit,
            phase_dur=None if self.pulse is not None else self.phase_dur,
            cathodic_first=self.cathodic_first)

    def _modulation(self, source, implant=None):
        """What the source asks each electrode for, frame by frame

        The first half of encoding, and the half with no time resolution in
        it: the source is reduced to one gray level per electrode per frame,
        stretched and quantized as asked, and ``_modulate`` turns those gray
        levels into the amplitude and frequency each electrode is to run at.
        There is no waveform here, no pulse clock and no raster -- those are
        :py:meth:`_assemble`, and they are what makes the result a *train*
        rather than a description of one.

        Returns the arguments :py:meth:`_assemble` takes, in the order it
        takes them.
        """
        gray, electrodes, frame_time, frame_dur = self._as_frames(
            source, implant, self.frame_dur)
        if self.stretch:
            gray = gray - gray.min()
            peak = gray.max()
            if peak > 0:
                gray = gray / peak
        if self.n_levels is not None:
            # Quantize before modulating rather than after, so that the same
            # parameter means the same thing however a subclass modulates:
            steps = self.n_levels - 1
            gray = np.round(gray * steps) / steps
        amp, freq = self._modulate(gray)
        return amp, freq, electrodes, frame_time, frame_dur

    def encode(self, source, implant=None):
        """Encode an image or a video as a train of electrical pulses

        Parameters
        ----------
        source : :py:class:`~pulse2percept.stimuli.Stimulus`
            The image or video to encode. Gray levels are expected in [0, 1],
            which is what :py:class:`~pulse2percept.stimuli.ImageStimulus` and
            :py:class:`~pulse2percept.stimuli.VideoStimulus` produce. It must
            be dimensionless: this method is the boundary at which a picture
            becomes stimulation, so an electrical stimulus is not a valid
            source for it.
        implant : :py:class:`~pulse2percept.implants.Implant`, optional
            The implant to encode for. Its electrode locations are used to
            sample the source, its electrode names label the resulting
            stimulus, and its
            :py:attr:`~pulse2percept.implants.Implant.raster` decides
            which electrodes may pulse when. If None, every pixel of the source
            is treated as its own electrode and every electrode fires on the
            same schedule.

            .. versionchanged:: 0.10.0
                The implant is named here rather than owned by the encoder, so
                that one encoder can be used for several implants and so that
                the implant is the only place device scheduling is described.

        Returns
        -------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus`
            The encoded stimulus, ready for the implant to deliver. Its
            amplitudes are in microamps, or in threshold multiples (``xTh``)
            if the encoder's amplitude parameters were, and its time axis is
            in milliseconds, whatever units the encoder's own parameters were
            given in.

        Raises
        ------
        :py:class:`~pulse2percept.units.DimensionMismatchError`
            If ``source`` is not dimensionless.

        """
        return self._assemble(*self._modulation(source, implant),
                              implant=implant)


class AmplitudeEncoder(StimulusEncoder):
    """Encode gray levels as pulse amplitudes

    Every electrode emits a pulse train of the same fixed frequency, and the
    gray level of the pixel it sees sets the amplitude of those pulses. This is
    how most retinal prostheses encode a video.

    Because every electrode shares one pulse period, a raster costs no
    frequency here: the groups hold fixed offsets from one another and so can
    never drift together, which means nothing has to be quantized and ``freq``
    is delivered exactly. With no explicit ``group_dur`` the groups also divide
    that period evenly, one turn each per pulse; an explicit ``group_dur``
    packs them into a shorter sweep at the start of every period instead.
    Either way no two groups are ever active at the same instant.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    amp_range : (min_amp, max_amp), optional
        Range of pulse amplitudes, in uA or in multiples of perceptual
        threshold (``xTh``). A gray level of 0 maps onto ``min_amp`` and a
        gray level of 1 onto ``max_amp``.

        Bare numbers mean uA. Both endpoints must carry the same dimension, so
        a threshold-relative range is spelled ``(0 * xTh, 3 * xTh)``; the half
        spelling ``(0, 3 * xTh)`` is rejected rather than guessed at. A
        threshold-relative range encodes to a ``xTh`` stimulus, which an
        implant converts to current when it has
        :py:attr:`~pulse2percept.implants.Implant.thresholds` for
        every driven electrode.

        .. versionchanged:: 0.11.0
            Accepts ``xTh`` as well as current.
    freq : float, optional
        Pulse train frequency (Hz), the same for every electrode. The pulse
        clock runs independently of the video, so the frame rate has no say in
        the rate delivered. Because every electrode shares this one period, a
        raster does not quantize it either: the groups keep a fixed offset from
        one another and cannot drift together. Only ``clock`` can lower it, by
        rounding the period up to a whole number of cycles.

        .. note::

           A frequency below the frame rate is realizable, but wasteful: some
           frames then receive no pulse at all and their gray levels are never
           delivered. Encoding warns when this happens.

    phase_dur, interphase_dur, cathodic_first, frame_dur, stretch
        See :py:class:`~pulse2percept.stimuli.StimulusEncoder`.

    Notes
    -----
    *  Arguments may be given as plain numbers in the units documented above,
       or as unitful quantities (e.g. ``0.05 * mA``, ``460 * us``,
       ``0.02 * kHz``), which are converted to those units. See
       :py:mod:`pulse2percept.units`.

    Examples
    --------
    Encode a movie for Argus II, mapping gray levels onto 0-50 uA at 20 Hz:

    >>> import pulse2percept as p2p
    >>> implant = p2p.implants.ArgusII()
    >>> implant.encoder = p2p.stimuli.AmplitudeEncoder(amp_range=(0, 50))
    >>> stim = implant.prepare_stim(p2p.stimuli.BostonTrain())

    The same thing spelled out, for an implant that is not to keep the encoder:

    >>> encoder = p2p.stimuli.AmplitudeEncoder(amp_range=(0, 50))
    >>> stim = encoder.encode(p2p.stimuli.BostonTrain(), implant=implant)

    """
    __slots__ = ('amp_range', 'freq', 'amp_unit')

    def __init__(self, amp_range=(0, 50), freq=20, **kwargs):
        super().__init__(**kwargs)
        amp_unit = self._amp_range_unit(amp_range)
        # See `StimulusEncoder.__init__`. `amp_range` is converted element by
        # element, so its two endpoints may be given in different units:
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
        """Return the uA or xTh unit of amp_range, rejecting mixtures."""
        dim = getattr(amp_range, 'dimension', None)
        if dim is not None:
            dims = [dim]
        else:
            try:
                dims = [getattr(a, 'dimension', uA.dimension)
                        for a in amp_range]
            except TypeError:
                # Not a pair at all; the shape check reports that.
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


class FrequencyEncoder(StimulusEncoder):
    """Encode gray levels as pulse train frequencies

    Every electrode emits pulses of the same fixed amplitude, and the gray
    level of the pixel it sees sets how often they come.

    .. important::

       Frequency modulation is far more expensive to simulate than amplitude
       modulation, because electrodes pulsing at different rates do not pulse
       at the same *times*: the stimulus needs a time point wherever any
       electrode's pulse has an edge, rather than the handful of time points
       that amplitude modulation shares between all of them.

       ``clock`` is the lever that cuts that down, and it is physically
       motivated: real stimulators have a time base. Encoding the 94-frame
       ``BostonTrain`` for Argus II at frequencies in (0, 300] Hz:

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

       ``clock`` is not free, though: it buys those time points with frequency
       resolution, and it spends it at the top of the range where the periods
       are shortest. Against ``freq_range=(0, 300)``, ``clock=1`` delivers the
       brightest pixels at 250 Hz rather than 300, and ``clock=2`` at 200 Hz.
       Pick it against the fastest train you actually need.

       ``n_levels`` is a much weaker lever here than the numbers above might
       suggest, and only worth reaching for once ``clock`` is set. Because the
       pulse clock keeps its phase across frames, two electrodes quantized onto
       the same gray level still pulse at different *times* unless their whole
       history matches; quantizing gray levels no longer collapses them onto a
       shared schedule the way it would if every frame restarted the train.

       A raster cuts the cost too, and for the same reason a clock does: it
       confines every onset to the raster grid.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    freq_range : (min_freq, max_freq), optional
        Range of pulse train frequencies (Hz). A gray level of 0 maps onto
        ``min_freq`` and a gray level of 1 onto ``max_freq``. A frequency of 0
        means no pulse at all.

        .. note::

           Realizable frequencies are quantized by ``clock``, and, when a
           raster is in play, onto the raster sweep -- which under frequency
           modulation is the usual case, since the electrodes are by
           construction on differing rates. Every period becomes a whole number
           of sweeps, so the realizable rates are ``1000 / (m * sweep)`` Hz.

           How coarse that grid is depends on how the sweep was set. With
           ``group_dur=None`` the sweep is the shortest period asked for, so
           the fastest electrode keeps its rate and pulses once per sweep while
           slower ones pulse every *m*-th. With an explicit ``group_dur`` the
           sweep is ``n_groups * group_dur`` and unrelated to any requested
           rate, so even the fastest electrode is generally rounded: against a
           six-group 1 ms sweep, a requested 100 Hz (10 ms) is delivered as
           83.3 Hz (12 ms, two sweeps).

           Quantizing onto the sweep always rounds the *period* up, so an
           electrode is never driven faster than it was asked for: against a
           10 ms sweep, 67 Hz comes back as 50 Hz rather than 100 Hz. Rounding
           to the nearest sweep instead would deliver up to twice the charge
           the caller asked for. Shorten ``group_dur`` for a finer grid.
    amp : float, optional
        Pulse amplitude (uA), the same for every electrode.
    phase_dur, interphase_dur, cathodic_first, pulse, clock, n_levels, \
frame_dur, stretch
        See :py:class:`~pulse2percept.stimuli.StimulusEncoder`.

    Notes
    -----
    *  Arguments may be given as plain numbers in the units documented above,
       or as unitful quantities (e.g. ``0.05 * mA``, ``460 * us``,
       ``0.02 * kHz``), which are converted to those units. See
       :py:mod:`pulse2percept.units`.

    Examples
    --------
    Encode a movie for Argus II at 50 uA, mapping gray levels onto 0-300 Hz on
    a 1 ms stimulator clock. A 300 Hz period is 3.3 ms, which Argus II's own
    six-group 2 ms raster sweep does not fit into, so this device drives every
    electrode at once:

    >>> import pulse2percept as p2p
    >>> implant = p2p.implants.ArgusII(raster=None)
    >>> implant.encoder = p2p.stimuli.FrequencyEncoder(freq_range=(0, 300),
    ...                                                amp=50, clock=1)
    >>> stim = implant.prepare_stim(p2p.stimuli.BostonTrain())

    """
    __slots__ = ('freq_range', 'amp')

    def __init__(self, freq_range=(0, 300), amp=50, **kwargs):
        super().__init__(**kwargs)
        # See `StimulusEncoder.__init__`:
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


#: The unit an optical encoder measures its output in
_IRRADIANCE = mW / mm ** 2


class _NormalizedStimulus(Stimulus):
    """Dimensionless encoded drive for spatial models."""
    _default_unit = dimensionless
    _is_normalized_drive = True

    __slots__ = ()


class _OpticalStimulus(Stimulus):
    """Lazy PRIMA projector schedule.

    Stores per-pixel ON duration for each projector frame, peak irradiance, and
    frame rate. Waveform samples are generated on demand.
    """
    #: described by its schedule rather than by its samples
    _is_parametric = True

    #: offers a normalized time-averaged view to spatial-only models
    _has_spatial_view = True

    __slots__ = ('_dur', '_ticks', '_onsets', '_irradiance', '_freq',
                 '_wavelength', '_grayscale', '_total', '_ref_drive',
                 '_static', '_frame_time', '_frame_dur', '_time')

    def __init__(self, electrodes, dur, ticks, onsets, irradiance, freq,
                 wavelength, grayscale, total, static, frame_time, frame_dur,
                 ref_drive):
        irradiance = float(irradiance)
        # Rebuilt/scaled schedules must also have physical irradiance.
        if not math.isfinite(irradiance) or irradiance < 0:
            raise ValueError(f"'irradiance' must be a finite, nonnegative "
                             f"power density, not {irradiance}.")
        # ON duration (ms) per pixel and projector frame:
        self._dur = self._own(dur, np.float64)
        self._ticks = self._own(ticks, np.int64)
        # Onset (ticks) of every projector frame:
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
        self._frame_time = self._own(frame_time, np.float64)
        self._frame_dur = float(frame_dur)
        # Built lazily without rendering the waveform:
        self._time = None
        self._defer(electrodes, unit=_IRRADIANCE)
        # Metadata stores frame timing; optical settings remain schedule state.
        self.metadata['encoder'] = {'frame_time': self._frame_time,
                                    'frame_dur': self._frame_dur}

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
        """Projector frame rate (Hz)"""
        return self._freq

    @property
    def pulse_dur(self):
        """ON duration (ms) of every pixel, one column per projector frame"""
        return self._dur

    @property
    def duty_cycle(self):
        """Fraction of each projector period every pixel spends on"""
        return self._dur * self._freq / MS_PER_S

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
            # Collapse repeated projector periods for a static source.
            stim = _NormalizedStimulus(drive[:, 0].ravel(),
                                       electrodes=self.electrodes)
        else:
            stim = _NormalizedStimulus(drive, electrodes=self.electrodes,
                                       time=self._frame_time)
        stim.metadata['encoder'] = {'frame_time': self._frame_time,
                                    'frame_dur': self._frame_dur}
        return stim

    def _rebuilt(self, electrodes, dur, irradiance):
        """This schedule, driving different pixels or at a different power"""
        rebuilt = _OpticalStimulus(
            electrodes, dur, self._ticks, self._onsets, irradiance, self._freq,
            self._wavelength, self._grayscale, self._total, self._static,
            self._frame_time, self._frame_dur, self._ref_drive)
        rebuilt.metadata['user'] = deepcopy(self.metadata.get('user'))
        return rebuilt

    def _scaled(self, factor):
        """Return this schedule with irradiance scaled by ``factor``."""
        if not np.isscalar(factor):
            return None
        factor = float(factor)
        if not math.isfinite(factor) or factor < 0:
            raise ValueError(f"Scaling an optical stimulus by {factor} would "
                             f"ask the projector for a negative or undefined "
                             f"irradiance. Only nonnegative, finite factors "
                             f"describe light.")
        return self._rebuilt(self.electrodes, self._dur,
                             self._irradiance * factor)

    def _without_electrodes(self, electrodes):
        """This schedule, no longer illuminating ``electrodes``"""
        keep = self._keep_mask(electrodes)
        return self._rebuilt(self.electrodes[keep], self._dur[keep],
                             self._irradiance)

    def _render(self):
        """Expand the schedule into rectangular pulses."""
        ticks = np.asarray(self._ticks)
        n_frames = self._onsets.size
        # Map stored time points to projector frames.
        at = np.searchsorted(self._onsets, ticks, side='right') - 1
        np.clip(at, 0, n_frames - 1, out=at)
        # Keep durations per frame to avoid an n_electrodes x n_time int64 array.
        dur = np.round(self._dur / DT).astype(np.int64)
        data = np.zeros((dur.shape[0], ticks.size), dtype=np.float32)
        # Each projector frame occupies a contiguous time span.
        bounds = np.searchsorted(at, np.arange(n_frames + 1))
        irradiance = np.float32(self._irradiance)
        for j in range(n_frames):
            lo, hi = bounds[j], bounds[j + 1]
            if hi <= lo:
                continue
            # Time since the current projector-frame onset.
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


class PRIMAEncoder(Encoder):
    """Encode image/video gray levels for the PRIMA projector.

    PRIMA uses 880 nm illumination rather than injected current. The projector
    uses fixed peak irradiance and pulse-width modulation [Palanker2020]_,
    [Holz2026]_. This encoder returns irradiance in ``mW/mm^2``.

    .. versionadded:: 0.11.0

    Parameters
    ----------
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
       spatial models. It is neither retinal current nor perceptual brightness.
    *  Clinical image preprocessing is outside this encoder.

    Examples
    --------
    >>> from pulse2percept.implants import PRIMAPivotal
    >>> from pulse2percept.stimuli import LogoBVL, PRIMAEncoder
    >>> PRIMAEncoder().encode(LogoBVL(), implant=PRIMAPivotal()).unit
    mW/mm^2

    """
    #: Wavelength (nm) of the projected near-infrared light
    wavelength = 880.0

    #: Smallest nonzero ON duration (ms) the projector can produce; every other
    #: duration is a whole multiple of it
    pulse_step = 0.7

    #: Longest documented ON duration (ms), i.e. 14 steps
    max_pulse_dur = 9.8

    #: Peak irradiance (mW/mm^2) of the pivotal-trial projector
    max_irradiance = 3.5

    #: Frame rate (Hz) of the pivotal-trial projector
    max_freq = 30.0

    #: Largest documented duty cycle, ``max_freq * max_pulse_dur``
    max_duty_cycle = max_freq * max_pulse_dur / MS_PER_S

    #: Time-averaged irradiance (mW/mm^2) the normalized spatial view calls 1.0
    ref_drive = max_irradiance * max_duty_cycle

    __slots__ = ('irradiance', 'freq', 'pulse_dur', 'grayscale', 'threshold')

    def __init__(self, irradiance=3.5 * mW / mm ** 2, freq=30 * Hz,
                 pulse_dur=9.8 * ms, grayscale=True, threshold=0.5):
        irradiance = as_value(irradiance, _IRRADIANCE, 'irradiance')
        freq = as_value(freq, Hz, 'freq')
        pulse_dur = as_value(pulse_dur, ms, 'pulse_dur')
        threshold = as_value(threshold, dimensionless, 'threshold')
        _finite('irradiance', irradiance)
        if irradiance <= 0:
            raise ValueError("'irradiance' must be positive.")
        _finite('freq', freq)
        if freq <= 0:
            raise ValueError("'freq' must be positive.")
        _finite('pulse_dur', pulse_dur)
        if pulse_dur < 0:
            raise ValueError("'pulse_dur' cannot be negative.")
        if pulse_dur > 0:
            # Require exact hardware-grid durations; do not round silently.
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
            period = MS_PER_S / freq
            if pulse_dur >= period:
                raise ValueError(
                    f"A {pulse_dur:g} ms pulse does not fit into the "
                    f"{period:.3f} ms period of a {freq:g} Hz projector. "
                    f"Shorten 'pulse_dur' or lower 'freq'.")
        _finite('threshold', threshold)
        if not 0 <= threshold <= 1:
            raise ValueError(f"'threshold' must be a gray level in [0, 1], "
                             f"not {threshold}.")
        self.irradiance = irradiance
        self.freq = freq
        self.pulse_dur = pulse_dur
        self.grayscale = bool(grayscale)
        self.threshold = threshold

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        return {'irradiance': self.irradiance, 'freq': self.freq,
                'pulse_dur': self.pulse_dur, 'grayscale': self.grayscale,
                'threshold': self.threshold}

    @property
    def period(self):
        """Projector period (ms)"""
        return MS_PER_S / self.freq

    @property
    def n_levels(self):
        """Number of nonzero ON durations available up to ``pulse_dur``"""
        return int(round(self.pulse_dur / self.pulse_step))

    def _durations(self, gray):
        """Map gray levels in [0, 1] to ON durations (ms)

        Binary mode lights a pixel for the full ``pulse_dur``; grayscale mode
        pulse-width modulates onto the projector's own duration grid.
        """
        # Use float64 so durations land exactly on the hardware grid.
        gray = np.asarray(gray, dtype=np.float64)
        if not self.grayscale:
            return np.where(gray >= self.threshold, self.pulse_dur, 0.0)
        # Gray levels are already clipped to [0, 1].
        return np.round(gray * self.n_levels) * self.pulse_step

    def encode(self, source, implant=None):
        """Encode an image or a video as near-infrared irradiance

        Parameters
        ----------
        source : :py:class:`~pulse2percept.stimuli.Stimulus`
            The image or video to encode. Gray levels are expected in [0, 1],
            which is what :py:class:`~pulse2percept.stimuli.ImageStimulus` and
            :py:class:`~pulse2percept.stimuli.VideoStimulus` produce. It must
            be dimensionless.
        implant : :py:class:`~pulse2percept.implants.Implant`, optional
            The implant to encode for. Its pixel locations are used to sample
            the source and its pixel names label the resulting stimulus. If
            None, every pixel of the source is treated as its own photovoltaic
            pixel.

        Returns
        -------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus`
            The projected irradiance, in ``mW/mm^2``, with a time axis in
            milliseconds. The waveform is generated only when samples are
            asked for.

        Raises
        ------
        :py:class:`~pulse2percept.units.DimensionMismatchError`
            If ``source`` is not dimensionless.

        """
        period = self.period
        gray, electrodes, frame_time, frame_dur = self._as_frames(source,
                                                                  implant)
        n_el = len(electrodes)
        static = frame_time.size == 1 and getattr(source, 'time', None) is None
        # Preserve source start time and duration.
        start = float(frame_time[0])
        total = float(frame_time[-1] + frame_dur)
        n_periods = max(1, int(np.ceil((total - start) / period - 1e-9)))
        onset_ms = start + np.arange(n_periods, dtype=np.float64) * period
        # Sample source frames at projector onsets using zero-order hold.
        at = np.searchsorted(frame_time, onset_ms, side='right') - 1
        np.clip(at, 0, frame_time.size - 1, out=at)
        dur = self._durations(gray[:, at])

        # Round absolute onsets to DT to avoid accumulated 30 Hz period error.
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
                f"is the lever that helps most; in grayscale mode, so is a "
                f"shorter 'pulse_dur', which offers fewer distinct durations.",
                category=UserWarning)
        if n_el * n_time > _BIG_STIM and implant is None:
            _warn_external(
                f"Encoding {n_el} pixels x {n_time} time points will allocate "
                f"{n_el * n_time * 4 / 1e9:.1f} GB. Pass 'implant' to encode "
                f"at pixel resolution instead.", category=UserWarning)

        # Keep the projector schedule lazy; render waveform samples on demand.
        return _OpticalStimulus(
            electrodes, dur, ticks, onsets, self.irradiance, self.freq,
            self.wavelength, self.grayscale, total, static,
            np.zeros(1) if static else onset_ms,
            total if static else period, self.ref_drive)
