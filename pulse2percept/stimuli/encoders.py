""":py:class:`~pulse2percept.stimuli.Encoder`,
   :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`,
   :py:class:`~pulse2percept.stimuli.FrequencyEncoder`"""
from abc import ABCMeta, abstractmethod
import warnings
import numpy as np

from .base import Stimulus
from .images import ImageStimulus
from .pulses import BiphasicPulse
from .videos import VideoStimulus
from ..utils import PrettyPrint, frame_interval
from ..utils.constants import DT

# Encoding a source that still has one row per *pixel* rather than one per
# electrode produces a data container this many elements large before anyone
# notices. Warn past that, because the fix (pass ``implant``) is a one-liner:
_BIG_STIM = 5e7

# Every model downstream of the encoder allocates something proportional to the
# number of time points in the stimulus -- a spatial model, one float per grid
# point per time point. Warn past this many, because the fixes (``clock``,
# ``n_levels``) are not obvious:
_BIG_TIME = 20000

# Duration (ms) of a source that has no time axis of its own, such as an image:
_DEFAULT_FRAME_DUR = 500.0


def _finite(name, value):
    """Reject NaN and infinity, which slip through every ``<`` comparison"""
    if not np.all(np.isfinite(np.asarray(value, dtype=np.float64))):
        raise ValueError(f"'{name}' must be finite, not {value}.")


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


class Encoder(PrettyPrint, metaclass=ABCMeta):
    """Abstract base class for all stimulus encoders

    An encoder translates the gray levels of an image or a video into the
    electrical stimulus that a retinal implant would actually deliver: each
    electrode emits a train of biphasic pulses, and the gray level of the pixel
    that the electrode sees determines some property of that train.

    Three clocks are involved, and they are deliberately independent of one
    another:

    *  The **frame clock** belongs to the video. It says when the modulation
       parameters update -- a new frame is a new gray level, and hence a new
       amplitude or a new frequency. It is also the rate at which a percept is
       worth reporting, which is why it is recorded in the encoded stimulus'
       metadata for :py:meth:`~pulse2percept.models.Model.predict_percept` to
       pick up. It takes no part in the timing of the pulses themselves.

    *  The **pulse clock** belongs to ``freq``. It runs continuously for the
       whole stimulus rather than restarting at every frame, so a requested
       frequency is the frequency actually delivered. A pulse takes the
       modulation parameters of the frame its *onset* falls into, so it is
       never cut in half by a frame boundary.

    *  The **raster cycle** belongs to the
       :py:class:`~pulse2percept.implants.Raster`, and says which electrodes
       may pulse when. Pulse periods are whole multiples of it, so electrodes
       in different raster groups provably never pulse at the same instant.

    .. versionchanged:: 0.10.0

        Pulse timing no longer restarts at every video frame. Before, a frame
        that was not a whole number of pulse periods long silently changed the
        pulse rate: at 29.97 fps, ``freq=50`` delivered 59.94 pulses per second.

    All encoders share the same two-step structure:

    1.  Reduce the source to one gray level per electrode per frame. If the
        encoder was given an ``implant``, the source is first sampled at the
        electrode locations, so that everything downstream works at electrode
        resolution.
    2.  Map those gray levels onto pulse train parameters (``_modulate``), then
        assemble the pulse trains (``_assemble``).

    Subclasses only implement ``_modulate``; everything else is provided here.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`, optional
        The implant to encode for. Its electrode locations are used to sample
        the source, and its electrode names label the resulting stimulus.
        If None, every pixel of the source is treated as its own electrode,
        which is rarely what you want for anything but an already downsampled
        image.
    phase_dur : float, optional
        Duration (ms) of the cathodic/anodic phase of each pulse.
    interphase_dur : float, optional
        Duration (ms) of the gap between the cathodic and anodic phases.
    cathodic_first : bool, optional
        If True, the cathodic phase of each pulse is delivered first. Retinal
        ganglion cells are most sensitive to cathodic current, and the temporal
        models in :py:mod:`pulse2percept.models` treat cathodic current as
        brightness-increasing, so bright pixels map onto cathodic-first pulses.
    pulse : :py:class:`~pulse2percept.stimuli.Stimulus`, optional
        A single pulse to repeat, in place of the symmetric biphasic pulse
        built from ``phase_dur``, ``interphase_dur`` and ``cathodic_first``
        (which are then ignored). Only its *shape* is used: its amplitude is
        normalized away, since that is what the encoder sets, and its time axis
        is shifted to start at zero. It must start and end at zero amplitude,
        since it is tiled into a train.
    clock : float, optional
        Period (ms) of the stimulator's time base. Pulse periods and raster
        offsets are rounded to a whole number of clock cycles, as they would be
        on real hardware. If None, they are placed at the full resolution of
        the simulation (``DT`` = 1e-3 ms).

        .. important::

           This is the main lever on how expensive an encoded stimulus is to
           simulate. Electrodes that end up on the same pulse schedule share a
           time axis, so a coarse clock keeps the number of distinct time
           points in the stimulus small. It does nothing for amplitude
           modulation (where every electrode is already on the same schedule)
           but is decisive for frequency modulation; see
           :py:class:`~pulse2percept.stimuli.FrequencyEncoder`.
    n_levels : int, optional
        Number of gray levels the encoder can distinguish, mimicking the
        resolution of the device's input stage. Gray levels are rounded onto
        ``n_levels`` values evenly spaced over [0, 1] before being modulated.
        If None, they are taken at full precision.
    raster : :py:class:`~pulse2percept.implants.Raster`, optional
        How the stimulator takes turns between electrodes it cannot drive at
        the same time. Each raster group gets its own slot within a repeating
        raster cycle, and pulse periods are quantized onto that cycle, so no
        two groups are ever active at once. If None, the ``implant``'s own
        raster is used, and failing that every electrode fires on the same
        schedule.
    frame_dur : float, optional
        Duration (ms) of a single frame. If None, it is inferred from the
        source's frame rate (or, failing that, from its time axis). A source
        without a time axis, such as an
        :py:class:`~pulse2percept.stimuli.ImageStimulus`, is treated as a
        single frame lasting 500 ms.
    stretch : bool, optional
        If True, the gray levels of the source are stretched to fill [0, 1]
        before they are modulated, so that the darkest pixel maps onto the
        bottom of the modulation range and the brightest onto the top.
        If False (the default), gray levels are taken at face value: a gray
        level of 0.5 always maps onto the middle of the range no matter how
        bright the rest of the source is.

        .. note::

           Stretching makes the encoding depend on the content of the source.
           A uniform image has no range to stretch, and encodes to a stimulus
           of zero amplitude everywhere.

    """
    __slots__ = ('implant', 'phase_dur', 'interphase_dur', 'cathodic_first',
                 'pulse', 'clock', 'n_levels', 'raster', 'frame_dur',
                 'stretch')

    def __init__(self, implant=None, phase_dur=0.46, interphase_dur=0,
                 cathodic_first=True, pulse=None, clock=None, n_levels=None,
                 raster=None, frame_dur=None, stretch=False):
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
            # A fractional `n_levels` would quietly become a fractional step
            # size, and the number of levels you got back would not be the
            # number you asked for:
            if int(n_levels) != n_levels:
                raise ValueError(f"'n_levels' must be a whole number, not "
                                 f"{n_levels}.")
            if n_levels < 2:
                raise ValueError("'n_levels' must be at least 2.")
        if frame_dur is not None:
            _finite('frame_dur', frame_dur)
            if frame_dur <= 0:
                raise ValueError("'frame_dur' must be positive.")
        self.implant = implant
        self.phase_dur = phase_dur
        self.interphase_dur = interphase_dur
        self.cathodic_first = cathodic_first
        self.pulse = pulse
        self.clock = clock
        self.n_levels = n_levels
        self.raster = raster
        self.frame_dur = frame_dur
        self.stretch = stretch

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        return {'implant': self.implant, 'phase_dur': self.phase_dur,
                'interphase_dur': self.interphase_dur,
                'cathodic_first': self.cathodic_first, 'pulse': self.pulse,
                'clock': self.clock, 'n_levels': self.n_levels,
                'raster': self.raster, 'frame_dur': self.frame_dur,
                'stretch': self.stretch}

    @abstractmethod
    def _modulate(self, gray):
        """Map gray levels onto pulse train parameters

        Parameters
        ----------
        gray : (n_electrodes, n_frames) array
            Gray levels in [0, 1], one per electrode per frame.

        Returns
        -------
        amp : array
            Pulse amplitude (uA), broadcastable to ``gray.shape``. The sign is
            supplied by ``cathodic_first``, so this should be nonnegative.
        freq : array
            Pulse train frequency (Hz), broadcastable to ``gray.shape``.

        """
        raise NotImplementedError

    def _as_frames(self, source):
        """Reduce the source to one gray level per electrode per frame

        Returns
        -------
        gray : (n_electrodes, n_frames) array
            Gray levels in [0, 1]. Values outside that range (an edge filter
            can produce them) are clipped.
        electrodes : array
            Electrode names, one per row of ``gray``.
        frame_time : (n_frames,) array
            The time (ms) at which each frame starts.
        frame_dur : float
            The duration (ms) of a single frame.

        """
        if not isinstance(source, Stimulus):
            raise TypeError(f"'source' must be a Stimulus object, not "
                            f"{type(source)}.")
        # Read the frame rate off the *source*: sampling at electrode locations
        # does not necessarily carry it along.
        fps = _fps(source.metadata)
        stim = source
        if (self.implant is not None and
                isinstance(stim, (ImageStimulus, VideoStimulus))):
            # Sample the source at the electrode locations. This is the same
            # step that assigning an image or a video to `implant.stim` would
            # perform, done here so that the pulse trains below are built at
            # electrode resolution rather than at pixel resolution. It is also
            # where RGB becomes gray. Row count is not a usable test of whether
            # a source is already in electrode coordinates -- a 10x6 image and
            # an RGB 4x5 image both have exactly as many rows as Argus II has
            # electrodes -- so always reshape:
            stim = self.implant.reshape_stim(stim)
        gray = np.clip(np.asarray(stim.data, dtype=np.float32), 0, 1)
        if stim.time is None:
            # A single frame, which has no duration of its own:
            gray = gray.reshape((-1, 1))
            frame_dur = (_DEFAULT_FRAME_DUR if self.frame_dur is None
                         else self.frame_dur)
            frame_time = np.zeros(1, dtype=np.float64)
        elif self.frame_dur is None:
            # `frame_interval` rejects a time axis whose frames are not all
            # the same length, so the frames tile the source exactly:
            frame_dur = frame_interval(np.asarray(stim.time), fps=fps)
            frame_time = np.asarray(stim.time, dtype=np.float64)
        else:
            # An explicit `frame_dur` re-times the source: the frames keep
            # their order and their content, but each one now lasts
            # `frame_dur` ms. Keeping the source's own frame times here would
            # let neighboring frames overlap:
            frame_dur = self.frame_dur
            frame_time = np.arange(gray.shape[1], dtype=np.float64) * frame_dur
        return gray, stim.electrodes, frame_time, frame_dur

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
        period[firing] = 1000.0 / freq[firing] / DT
        # A clocked stimulator can only realize a period that is a whole number
        # of clock cycles, which is what keeps the number of distinct schedules
        # (and hence of time points) down:
        if self.clock is not None:
            tick = self.clock / DT
            period[firing] = tick * np.maximum(
                1.0, np.round(period[firing] / tick))
        if np.any(period[firing] < pulse_len):
            too_fast = 1000.0 / (np.min(period[firing]) * DT)
            raise ValueError(f"A pulse (dur={pulse_len * DT:.3f} ms) does not "
                             f"fit into the pulse train window of a "
                             f"{too_fast:.1f} Hz train. Shorten 'phase_dur' "
                             f"or lower the frequency.")
        return firing, period

    def _raster_grid(self, electrodes, period, firing, pulse_len):
        """The slot each electrode may pulse in, and the raster cycle

        The cycle a raster has to get through is the shortest pulse period
        anyone asked for. Under amplitude modulation that is *the* period, so
        every group gets its turn exactly once per pulse and the requested
        frequency is delivered exactly. Under frequency modulation it is set by
        the fastest electrode, and slower ones pulse every m-th cycle.

        Returns
        -------
        offset : (n_electrodes,) float array
            How far into a raster cycle (in ticks) each electrode may start a
            pulse.
        cycle : float or None
            The raster cycle in ticks, onto which pulse periods are quantized.
            None when there is nothing to multiplex.

        """
        # An explicit raster wins over the implant's own, so that a raster can
        # be tried out without modifying the implant:
        raster = self.raster
        if raster is None:
            raster = getattr(self.implant, 'raster', None)
        zero = np.zeros(len(electrodes), dtype=np.float64)
        if raster is None or raster.n_groups < 2 or not np.any(firing):
            return zero, None
        # Derived from the requested frequencies, not from the amplitudes:
        # whether a raster is a workable schedule is a property of the device,
        # not of how bright today's video happens to be.
        fastest = float(np.min(period[firing]))
        offset = np.asarray(raster.offsets(electrodes, fastest * DT),
                            dtype=np.float64) / DT
        cycle = (fastest if raster.group_dur is None else
                 raster.n_groups * raster.group_dur / DT)
        if self.clock is not None:
            tick = self.clock / DT
            offset = tick * np.round(offset / tick)
            cycle = tick * max(1.0, round(cycle / tick))
        # Every group's turn has to be long enough to finish a pulse in. The
        # gap to the *next* group's turn is what a collision would show up as,
        # so measure those rather than the nominal slot -- rounding a slot onto
        # the clock grid can shorten one gap while lengthening another, and two
        # groups snapping onto the same offset shows up here as a gap of zero.
        # A pulse also has to clear the next group's slot by a whole tick, since
        # each onset is rounded onto the DT grid when it is placed:
        edges = np.append(np.unique(np.round(offset)), np.round(cycle))
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
            raster cycle, or failing that the stimulator's clock. A schedule
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
        # Frequency modulation: the period changes from frame to frame, and the
        # next pulse is scheduled from the previous one rather than from the
        # frame boundary, so that the pulse clock keeps its phase. `t` stays
        # unrounded for the same reason the period does -- rounding it here
        # would let the error accumulate from pulse to pulse:
        onset, frame, t = [], [], float(start)
        while t <= last:
            tick = int(round(t))
            k = int(np.searchsorted(frame_ticks, tick, side='right')) - 1
            k = min(max(k, 0), n_frames - 1)
            if active[k]:
                onset.append(tick)
                frame.append(k)
            if period[k] > 0:
                # The clock runs even where the amplitude is zero, so a dark
                # frame does not reset the phase of the frames around it:
                t += float(period[k])
            elif k + 1 < n_frames:
                # Nothing to stay in phase with, so pick the schedule back up
                # at the next frame -- but on this schedule's own grid, or a
                # silent stretch would knock every pulse after it off the
                # stimulator's clock and multiply the time points needed:
                nxt = float(frame_ticks[k + 1])
                t = start + grid * np.ceil((nxt - start) / grid)
            else:
                break
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

    def _assemble(self, amp, freq, electrodes, frame_time, frame_dur):
        """Build the pulse trains for every electrode and frame

        Electrodes that pulse at the same *times* share the shape of their
        waveform; only the amplitude that scales it differs. So rather than
        building one waveform per electrode, this builds one per distinct
        schedule and indexes into them, which is what keeps frequency
        modulation (thousands of electrode-frames, a few dozen schedules)
        tractable.

        The time axis is global rather than per-frame. Pulses live at absolute
        times and no two frames' pulses coincide, so this costs no more than
        laying each frame out separately did -- and it lets a pulse straddle a
        frame boundary, which is what keeps the pulse clock free of the frame
        clock.
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
        # an epsilon so a duration that *is* a whole number of ticks does not
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
        # keeps running (`firing`), so it stays in phase with its neighbors,
        # but it costs no pulses and no time points:
        active = firing & (amp != 0)
        offset, cycle = self._raster_grid(electrodes, period, firing, pulse_len)
        if cycle is not None:
            # A period that is a whole number of raster cycles is what keeps
            # two groups from ever drifting onto the same instant:
            period[firing] = cycle * np.maximum(
                1.0, np.round(period[firing] / cycle))

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
            warnings.warn(f"{missed} of {n_frames} frames deliver no pulse at "
                          f"all, because the pulse period is longer than a "
                          f"frame ({1000.0 / frame_dur:.2f} fps). Their gray "
                          f"levels are never sampled; raise the frequency to "
                          f"see them.")

        ticks = np.unique(np.concatenate(
            [np.array([0, end], dtype=np.int64)] +
            [(o[:, np.newaxis] + pulse_ticks[np.newaxis, :]).ravel()
             for o in onsets if o.size]))
        n_time = ticks.size
        if n_time > _BIG_TIME:
            warnings.warn(f"This stimulus has {n_time} time points, which "
                          f"every model downstream will pay for. Coarsening "
                          f"'clock' is the lever that helps most, since it "
                          f"confines every pulse onset to the same grid; a "
                          f"'raster' does the same. 'n_levels' helps far less "
                          f"on its own, because two electrodes on the same "
                          f"gray level still pulse at different times.")
        if n_el * n_time > _BIG_STIM and self.implant is None:
            warnings.warn(f"Encoding {n_el} electrodes x {n_time} time points "
                          f"will allocate {n_el * n_time * 4 / 1e9:.1f} GB. "
                          f"Pass 'implant' to encode at electrode resolution "
                          f"instead.")

        # One unit-amplitude waveform per schedule, scaled by the amplitude of
        # whichever frame each pulse belongs to. Sampling a schedule onto the
        # time axis is exact: its own time points are a subset of that axis,
        # and everything between two of them is either a plateau or a gap, both
        # of which linear interpolation reproduces.
        data = np.zeros((n_el, n_time), dtype=np.float32)
        for s, (onset, frame) in enumerate(zip(onsets, frames)):
            rows = np.flatnonzero(sched == s)
            if rows.size == 0 or onset.size == 0:
                continue
            wave = self._sample(onset, pulse_ticks, pulse_vals, ticks)
            # Which pulse -- and hence which frame's amplitude -- each time
            # point belongs to. Points before the first pulse and in the gaps
            # between pulses carry a zero waveform, so what they pick up here
            # never reaches the output:
            at = np.searchsorted(onset, ticks, side='right') - 1
            np.clip(at, 0, onset.size - 1, out=at)
            data[rows] = amp[rows][:, frame[at]] * wave
        time = ticks * DT
        time[-1] = total
        stim = Stimulus(data, electrodes=electrodes, time=time)
        # Provenance, not something a model should compute from: the stimulus
        # is a plain `Stimulus` and every consumer reads its data container.
        # `frame_time` is what lets a percept be reported on the video's own
        # clock rather than on the pulse train's:
        stim.metadata['encoder'] = {'kind': type(self).__name__,
                                    'frame_dur': frame_dur,
                                    'n_frames': n_frames,
                                    'frame_time': frame_time,
                                    'cycle': None if cycle is None else
                                    cycle * DT,
                                    'n_schedules': len(uniq)}
        return stim

    def encode(self, source):
        """Encode an image or a video as a train of electrical pulses

        Parameters
        ----------
        source : :py:class:`~pulse2percept.stimuli.Stimulus`
            The image or video to encode. Gray levels are expected in [0, 1],
            which is what :py:class:`~pulse2percept.stimuli.ImageStimulus` and
            :py:class:`~pulse2percept.stimuli.VideoStimulus` produce.

        Returns
        -------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus`
            The encoded stimulus, ready to assign to ``implant.stim``.

        """
        gray, electrodes, frame_time, frame_dur = self._as_frames(source)
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
        return self._assemble(amp, freq, electrodes, frame_time, frame_dur)


class AmplitudeEncoder(Encoder):
    """Encode gray levels as pulse amplitudes

    Every electrode emits a pulse train of the same fixed frequency, and the
    gray level of the pixel it sees sets the amplitude of those pulses. This is
    how most retinal prostheses encode a video.

    Because every electrode shares one pulse period, a raster splits that
    period evenly between its groups: each group pulses exactly once per period
    in its own slot, ``freq`` is delivered exactly, and no two groups are ever
    active at the same instant.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`, optional
        The implant to encode for; see
        :py:class:`~pulse2percept.stimuli.Encoder`.
    amp_range : (min_amp, max_amp), optional
        Range of pulse amplitudes (uA). A gray level of 0 maps onto
        ``min_amp`` and a gray level of 1 onto ``max_amp``.
    freq : float, optional
        Pulse train frequency (Hz), the same for every electrode. The pulse
        clock runs independently of the video, so this is the rate actually
        delivered whatever the frame rate is.

        .. note::

           A frequency below the frame rate is realizable, but wasteful: some
           frames then receive no pulse at all and their gray levels are never
           delivered. Encoding warns when this happens.

    phase_dur, interphase_dur, cathodic_first, frame_dur, stretch
        See :py:class:`~pulse2percept.stimuli.Encoder`.

    Examples
    --------
    Encode a movie for Argus II, mapping gray levels onto 0-50 uA at 20 Hz:

    >>> import pulse2percept as p2p
    >>> implant = p2p.implants.ArgusII()
    >>> encoder = p2p.stimuli.AmplitudeEncoder(implant, amp_range=(0, 50))
    >>> implant.stim = encoder.encode(p2p.stimuli.BostonTrain())

    """
    __slots__ = ('amp_range', 'freq')

    def __init__(self, implant=None, amp_range=(0, 50), freq=20, **kwargs):
        super().__init__(implant=implant, **kwargs)
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
        self.freq = freq

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super()._pprint_params()
        params.update({'amp_range': self.amp_range, 'freq': self.freq})
        return params

    def _modulate(self, gray):
        """Gray level in [0, 1] -> amplitude in ``amp_range``"""
        amp_lo, amp_hi = self.amp_range
        return amp_lo + gray * (amp_hi - amp_lo), self.freq


class FrequencyEncoder(Encoder):
    """Encode gray levels as pulse train frequencies

    Every electrode emits pulses of the same fixed amplitude, and the gray
    level of the pixel it sees sets how often they come.

    .. note::

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
       no quantization              142,704
       ``clock=1``                   21,519
       ``clock=2``                   10,886
       ``n_levels=8``                88,748
       ``clock=1, n_levels=8``       20,987
       =======================  ===========

       Start with ``clock=1``, which costs nothing in frequency range, and
       coarsen it from there.

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
    implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`, optional
        The implant to encode for; see
        :py:class:`~pulse2percept.stimuli.Encoder`.
    freq_range : (min_freq, max_freq), optional
        Range of pulse train frequencies (Hz). A gray level of 0 maps onto
        ``min_freq`` and a gray level of 1 onto ``max_freq``. A frequency of 0
        means no pulse at all.

        .. note::

           Realizable frequencies are quantized by ``clock``, and, when a
           raster is in play, onto the raster cycle: the fastest electrode
           pulses once per cycle and slower ones every m-th cycle, so the
           realizable rates are ``max_freq / m``. Multiplexing a fast train
           across many groups asks for a lot of a stimulator -- six groups of
           0.92 ms pulses need 5.5 ms per cycle, or at most 181 Hz -- and
           encoding raises rather than quietly delivering something else.
    amp : float, optional
        Pulse amplitude (uA), the same for every electrode.
    phase_dur, interphase_dur, cathodic_first, pulse, clock, n_levels, \
frame_dur, stretch
        See :py:class:`~pulse2percept.stimuli.Encoder`.

    Examples
    --------
    Encode a movie for Argus II at 50 uA, mapping gray levels onto 0-300 Hz on
    a 1 ms stimulator clock:

    >>> import pulse2percept as p2p
    >>> implant = p2p.implants.ArgusII()
    >>> encoder = p2p.stimuli.FrequencyEncoder(implant, freq_range=(0, 300),
    ...                                        amp=50, clock=1)
    >>> implant.stim = encoder.encode(p2p.stimuli.BostonTrain())

    """
    __slots__ = ('freq_range', 'amp')

    def __init__(self, implant=None, freq_range=(0, 300), amp=50, **kwargs):
        super().__init__(implant=implant, **kwargs)
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
