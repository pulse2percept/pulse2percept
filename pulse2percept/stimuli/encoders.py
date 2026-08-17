""":py:class:`~pulse2percept.stimuli.StimulusEncoder`,
   :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`,
   :py:class:`~pulse2percept.stimuli.FrequencyEncoder`"""
from abc import ABCMeta, abstractmethod
import numpy as np

from .base import Stimulus
from .images import ImageStimulus
from .pulses import BiphasicPulse
from .videos import VideoStimulus
from ..units import (DimensionMismatchError, Hz, as_value, dimensionless, ms,
                     uA)
from ..utils import PrettyPrint, frame_interval
# Every warning below is about the *caller's* choice of source, frequency, or
# implant, so it has to point at their line rather than at this file:
from ..utils.deprecation import _warn_external
from ..utils.constants import DT, MS_PER_S

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


class StimulusEncoder(PrettyPrint, metaclass=ABCMeta):
    """Abstract base class for all stimulus encoders

    An encoder translates the gray levels of an image or a video into the
    electrical stimulus that a retinal implant would actually deliver: each
    electrode emits a train of biphasic pulses, and the gray level of the pixel
    that the electrode sees determines some property of that train.

    Three clocks are involved, and they are deliberately independent of one
    another:

    *  The **frame clock** belongs to the video. It says when the modulation
       parameters update; that is, a new frame is a new gray level, and hence
       a new amplitude or a new frequency. It is also the rate at which a
       percept is worth reporting, which is why it is recorded in the encoded
       stimulus' metadata for 
       :py:meth:`~pulse2percept.models.Model.predict_percept` to
       pick up. It takes no part in the timing of the pulses themselves.

    *  The **pulse clock** belongs to ``freq``. It runs continuously for the
       whole stimulus rather than restarting at every frame, so the frame rate
       has no say in the rate delivered. A pulse takes the modulation
       parameters of the frame its *onset* falls into, so it is never cut in
       half by a frame boundary.

       The rate can still come out below the one requested, but only where the
       hardware you described cannot express it: ``clock``, and a raster with
       electrodes on differing rates, both round a pulse period *up*. Neither
       ever rounds down, so an electrode is never driven faster, and so never
       given more charge, than was asked for.

    *  The **raster sweep** belongs to the
       :py:class:`~pulse2percept.implants.Raster`, and says which electrodes
       may pulse when, so that no two raster groups are ever active at the same
       instant.

    All encoders share the same two-step structure, and the seam between the
    steps is where time enters:

    1.  **Modulation.** Reduce the source to one gray level per electrode per
        frame -- sampled at the electrode locations, if :py:meth:`encode` was
        given an ``implant`` -- and map those gray levels onto the amplitude
        and frequency each electrode is to run at (``_modulate``). Nothing
        here is time-resolved. :py:meth:`encode_spatial` stops at this point.
    2.  **Realization.** Turn those parameters into an actual train of pulses
        (``_assemble``), which is where the pulse shape, the pulse clock and
        the raster come in. :py:meth:`encode` runs both steps.

    The split is not an implementation detail: what the two halves produce are
    two different things a reader may want. A model with a temporal component
    integrates pulses and must have the train; one without a temporal
    component has no way to express a pulse train at all, and reads the
    modulation instead.

    An encoder describes a *modulation scheme*, not a device: it holds no
    implant of its own, and the implant it encodes for is named at
    :py:meth:`encode` time (or, more usually, by assigning the encoder to
    :py:attr:`~pulse2percept.implants.ProsthesisSystem.encoder` and letting the
    implant do it). Everything about the device -- which electrodes there are,
    where they sit, and how they take turns -- therefore comes from that one
    implant, including the :py:class:`~pulse2percept.implants.Raster`.

    Subclasses only implement ``_modulate``; everything else is provided here.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    phase_dur : float, optional
        Duration (ms) of the cathodic/anodic phase of each pulse.
    interphase_dur : float, optional
        Duration (ms) of the gap between the cathodic and anodic phases.
    cathodic_first : bool, optional
        If True, the cathodic phase of each pulse is delivered first. Most
        temporal models in :py:mod:`pulse2percept.models` treat cathodic 
        current as brightness-increasing, so bright pixels map onto 
        cathodic-first pulses.
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

           Every timing constraint here (the clock, and the raster sweep) may
           *lower* the rate an electrode ends up on, and none of them may
           raise it. Rounding a period down would deliver more charge than was
           asked for, so a time base that cannot represent a rate exactly gives
           back the nearest slower one it can.

           That makes a coarse clock expensive in frequency: realizable periods
           are ``clock``, ``2*clock``, ``3*clock``, ... , so with ``clock=1`` 
           a requested 300 Hz (3.33 ms) is delivered as 250 Hz (4 ms), and with
           ``clock=3`` as 166.7 Hz.
           Choose it against the top of your frequency range rather than in the
           abstract.
    n_levels : int, optional
        Number of gray levels the encoder can distinguish, mimicking the
        resolution of the device's input stage. Gray levels are rounded onto
        ``n_levels`` values evenly spaced over [0, 1] before being modulated.
        If None, they are taken at full precision.
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

    Notes
    -----
    *  Arguments may be given as plain numbers in the units documented above,
       or as unitful quantities (e.g. ``0.05 * mA``, ``460 * us``,
       ``0.02 * kHz``), which are converted to those units. See
       :py:mod:`pulse2percept.units`.
    *  ``pulse`` is the exception: only its shape is borrowed and its
       amplitude is normalized away, so what that amplitude was measured in
       does not matter. A dimensionless waveform is a perfectly good template.

    """
    __slots__ = ('phase_dur', 'interphase_dur', 'cathodic_first', 'pulse',
                 'clock', 'n_levels', 'frame_dur', 'stretch')

    def __init__(self, phase_dur=0.46, interphase_dur=0,
                 cathodic_first=True, pulse=None, clock=None, n_levels=None,
                 frame_dur=None, stretch=False):
        # Strip the units first; every schedule computed below is in plain
        # milliseconds, as it has always been. `pulse` is deliberately not
        # checked: only its shape is borrowed, and its amplitude is normalized
        # away in `_unit_pulse`, so what that amplitude was measured in does
        # not enter the encoding.
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

    def _as_frames(self, source, implant=None):
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
        # This is where a picture becomes stimulation, so what goes in has to
        # be a picture. An electrical stimulus read as gray levels would have
        # its microamps clipped to [0, 1] below and silently re-modulated into
        # a different current entirely:
        if not source.unit.dimension.is_dimensionless:
            raise DimensionMismatchError(
                f"An encoder turns gray levels into stimulation, so its "
                f"source must be dimensionless, not "
                f"{source.unit.dimension.name} ({source.unit}). Pass an "
                f"ImageStimulus or a VideoStimulus.")
        # Read the frame rate off the *source*: sampling at electrode locations
        # does not necessarily carry it along.
        fps = _fps(source.metadata)
        stim = source
        if (implant is not None and
                isinstance(stim, (ImageStimulus, VideoStimulus))):
            # Sample the source at the electrode locations. This is the same
            # step that assigning an image or a video to `implant.stim` would
            # perform, done here so that the pulse trains below are built at
            # electrode resolution rather than at pixel resolution. It is also
            # where RGB becomes gray. Row count is not a usable test of whether
            # a source is already in electrode coordinates, so always reshape:
            stim = implant.reshape_stim(stim)
        # `values` rather than `data`, to say out loud that these are the
        # dimensionless numbers the modulation below is a function of:
        gray = np.clip(np.asarray(stim.values(dimensionless),
                                  dtype=np.float32), 0, 1)
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

        # One unit-amplitude waveform per schedule, scaled by the amplitude of
        # whichever frame each pulse belongs to:
        data = np.zeros((n_el, n_time), dtype=np.float32)
        for s, (onset, frame) in enumerate(zip(onsets, frames)):
            rows = np.flatnonzero(sched == s)
            if rows.size == 0 or onset.size == 0:
                continue
            wave = self._sample(onset, pulse_ticks, pulse_vals, ticks)
            # Which pulse each time point belongs to:
            at = np.searchsorted(onset, ticks, side='right') - 1
            np.clip(at, 0, onset.size - 1, out=at)
            data[rows] = amp[rows][:, frame[at]] * wave
        time = ticks * DT
        time[-1] = total
        stim = Stimulus(data, electrodes=electrodes, time=time)
        stim.metadata['encoder'] = {'kind': type(self).__name__,
                                    'modulation': False,
                                    'frame_dur': frame_dur,
                                    'n_frames': n_frames,
                                    'frame_time': frame_time,
                                    'cycle': None if cycle is None else
                                    cycle * DT,
                                    'n_schedules': len(uniq)}
        return stim

    def _modulation(self, source, implant=None):
        """What the source asks each electrode for, frame by frame

        The first half of encoding, and the half with no time resolution in
        it: the source is reduced to one gray level per electrode per frame,
        stretched and quantized as asked, and ``_modulate`` turns those gray
        levels into the amplitude and frequency each electrode is to run at.
        There is no waveform here, no pulse clock and no raster -- those are
        :py:meth:`_assemble`, and they are what makes the result a *train*
        rather than a description of one.

        Returns the arguments both halves of the split take, in the order
        :py:meth:`_assemble` and :py:meth:`_as_spatial` accept them.
        """
        gray, electrodes, frame_time, frame_dur = self._as_frames(source,
                                                                  implant)
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

    def _as_spatial(self, amp, freq, electrodes, frame_time, frame_dur):
        """The modulation parameters as a Stimulus, with no waveform in them

        One column per frame of the source and one row per electrode, holding
        the current that electrode is modulated to over that frame. It is what
        the encoder *asks the device for*, before the device's own timing --
        the pulse shape, the pulse clock, the raster -- settles when any of it
        is actually delivered.

        An electrode whose train never fires delivers no current at all, so a
        zero frequency reads as zero amplitude here however high an amplitude
        it was handed. Past that, rate does not survive into this
        representation: what a rate means is a fact about time, and a reader
        with no clock of its own has no way to express it. It is recorded in
        the metadata for readers that do.
        """
        shape = (len(electrodes), frame_time.size)
        amp = np.broadcast_to(np.asarray(amp, dtype=np.float32), shape)
        freq = np.ascontiguousarray(
            np.broadcast_to(np.asarray(freq, dtype=np.float64), shape))
        data = np.where(freq > 0, amp, np.float32(0)).astype(np.float32)
        # A source with a single frame has no time axis of its own, and gets
        # none here either: what it asks for is one steady thing, and saying so
        # is what lets a spatial model report a single picture rather than a
        # sequence of one. Flattened, because that is how `Stimulus` is told
        # a stimulus has no time component -- an (n, 1) container with
        # `time=None` is given a time axis of [0] instead.
        if frame_time.size > 1:
            stim = Stimulus(data, electrodes=electrodes, time=frame_time)
        else:
            stim = Stimulus(data.ravel(), electrodes=electrodes)
        stim.metadata['encoder'] = {'kind': type(self).__name__,
                                    'modulation': True,
                                    'frame_dur': frame_dur,
                                    'n_frames': int(frame_time.size),
                                    'frame_time': frame_time,
                                    'freq': freq}
        return stim

    def _encode_both(self, source, implant=None):
        """Both representations of one source, sharing the modulation step

        What an implant wants when a picture is assigned to it: sampling the
        source at the electrodes is the expensive half of encoding, and doing
        it once for :py:meth:`encode` and again for :py:meth:`encode_spatial`
        would be paying for it twice.
        """
        mod = self._modulation(source, implant)
        return self._as_spatial(*mod), self._assemble(*mod, implant=implant)

    def encode_spatial(self, source, implant=None):
        """Encode an image or a video as the modulation it asks each electrode
        for

        The *spatial* half of :py:meth:`encode`: one amplitude per electrode
        per frame of the source, and no pulse train. Nothing that belongs to
        the device's timing is in it -- no pulse shape, no pulse clock, no
        raster -- because none of those can be read by anything that has no
        clock of its own.

        This is what a purely spatial model consumes (see
        :py:meth:`~pulse2percept.models.SpatialModel.predict_percept`). Handing
        such a model the delivered pulse train instead would have it report the
        stimulus one raster slot at a time, which is a picture of the schedule
        rather than of the image.

        Parameters
        ----------
        source : :py:class:`~pulse2percept.stimuli.Stimulus`
            The image or video to encode; see :py:meth:`encode`.
        implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`, optional
            The implant to encode for; see :py:meth:`encode`. Its raster plays
            no part here, since a raster is a fact about time.

        Returns
        -------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus`
            One row per electrode and one column per frame, in microamps. A
            single-frame source (an image) comes back without a time axis. The
            per-electrode pulse frequency is recorded under
            ``metadata['encoder']['freq']``, for readers that can express a
            rate.

        .. versionadded:: 0.10.0

        """
        return self._as_spatial(*self._modulation(source, implant))

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
        implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`, optional
            The implant to encode for. Its electrode locations are used to
            sample the source, its electrode names label the resulting
            stimulus, and its
            :py:attr:`~pulse2percept.implants.ProsthesisSystem.raster` decides
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
            The encoded stimulus, ready to assign to ``implant.stim``.
            Its amplitudes are in microamps and its time axis in
            milliseconds, whatever units the encoder's own parameters were
            given in.

        Raises
        ------
        :py:class:`~pulse2percept.units.DimensionMismatchError`
            If ``source`` is not dimensionless.

        See Also
        --------
        encode_spatial : the same source as the modulation it asks for, with
                         none of the device's timing in it.

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
        Range of pulse amplitudes (uA). A gray level of 0 maps onto
        ``min_amp`` and a gray level of 1 onto ``max_amp``.
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
    >>> implant.stim = p2p.stimuli.BostonTrain()

    The same thing spelled out, for an implant that is not to keep the encoder:

    >>> encoder = p2p.stimuli.AmplitudeEncoder(amp_range=(0, 50))
    >>> stim = encoder.encode(p2p.stimuli.BostonTrain(), implant=implant)

    """
    __slots__ = ('amp_range', 'freq')

    def __init__(self, amp_range=(0, 50), freq=20, **kwargs):
        super().__init__(**kwargs)
        # See `StimulusEncoder.__init__`. `amp_range` is converted element by
        # element, so its two endpoints may be given in different units:
        amp_range = as_value(amp_range, uA, 'amp_range')
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
    >>> implant.stim = p2p.stimuli.BostonTrain()

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
