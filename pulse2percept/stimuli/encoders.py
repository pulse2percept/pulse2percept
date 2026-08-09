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
    electrical stimulus that a retinal implant would actually deliver: for
    every frame, each electrode emits a train of biphasic pulses that lasts one
    frame period, and the gray level of the pixel that the electrode sees
    determines some property of that train.

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
        (which are then ignored). Its amplitude is normalized away, since that
        is what the encoder sets; only its shape is used. It must start and end
        at zero, since it is tiled into a train.
    clock : float, optional
        Period (ms) of the stimulator's time base. Pulse periods and raster
        delays are rounded to a whole number of clock cycles, as they would be
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
        the same time; electrodes in later raster groups start their pulses
        later in the frame. If None, the ``implant``'s own raster is used, and
        failing that every electrode fires at the start of the frame.
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
        if phase_dur <= DT:
            raise ValueError(f"'phase_dur' must be greater than DT={DT} ms.")
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
        if clock is not None and clock < DT:
            raise ValueError(f"'clock' cannot be finer than the simulation "
                             f"time step DT={DT} ms.")
        if n_levels is not None and n_levels < 2:
            raise ValueError("'n_levels' must be at least 2.")
        if frame_dur is not None and frame_dur <= 0:
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

    def _delays(self, electrodes, frame_dur):
        """Per-electrode delay (ms) into the frame window

        By default every electrode fires at the start of each frame. A
        stimulator that cannot drive them all at once staggers them instead,
        which is what a :py:class:`~pulse2percept.implants.Raster` describes.

        Returns
        -------
        delay : (n_electrodes,) array
            Delay (ms) between the start of a frame and this electrode's first
            pulse.
        """
        # An explicit raster wins over the implant's own, so that a raster can
        # be tried out without modifying the implant:
        raster = self.raster
        if raster is None:
            raster = getattr(self.implant, 'raster', None)
        if raster is None:
            return np.zeros(len(electrodes), dtype=np.float32)
        return np.asarray(raster.delays(electrodes, frame_dur),
                          dtype=np.float32)

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
        fps = _fps(source.metadata)
        stim = source
        if (self.implant is not None and
                isinstance(stim, (ImageStimulus, VideoStimulus)) and
                len(stim.electrodes) != self.implant.n_electrodes):
            # Sample the source at the electrode locations. This is the same
            # step that assigning an image or a video to `implant.stim` would
            # perform, done here so that the pulse trains below are built at
            # electrode resolution rather than at pixel resolution:
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
        amplitude in uA yields the pulse to deliver.
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
        # A pulse is tiled into a train, and the gaps between the copies carry
        # whatever its end points carry. Anything but zero would smear across
        # the whole frame:
        if values[0] != 0 or values[-1] != 0:
            raise ValueError("'pulse' must start and end at zero amplitude, "
                             "since it is repeated to fill a frame.")
        return ticks, values

    def _schedule(self, freq, delay, last, pulse_ticks):
        """When each electrode pulses during a frame

        Parameters
        ----------
        last : int
            The last tick of the frame that a pulse may occupy.

        Returns
        -------
        group : (n_electrodes, n_frames) int array
            Which schedule each electrode is on during each frame. Electrodes
            on the same schedule pulse at the same times, and so differ only by
            the amplitude that scales their waveform.
        onsets : list of arrays
            The pulse onsets (in ticks since the start of the frame) of each
            schedule, indexed by group.

        """
        pulse_len = pulse_ticks[-1] - pulse_ticks[0]
        # Pulse period. A clocked stimulator can only realize a period that is
        # a whole number of clock cycles, which is what keeps the number of
        # distinct schedules (and hence of time points) down:
        window = np.zeros(freq.shape, dtype=np.int64)
        firing = freq > 0
        window[firing] = self._ticks(1000.0 / freq[firing])
        if self.clock is not None:
            cycle = max(1, int(self._ticks(self.clock)))
            window[firing] = cycle * np.maximum(
                1, np.round(window[firing] / cycle).astype(np.int64))
            delay = cycle * np.round(delay / cycle).astype(np.int64)
        if np.any(window[firing] < pulse_len):
            too_fast = 1000.0 / (np.min(window[firing]) * DT)
            raise ValueError(f"A pulse (dur={pulse_len * DT:.3f} ms) does not "
                             f"fit into the pulse train window of a "
                             f"{too_fast:.1f} Hz train. Shorten 'phase_dur' "
                             f"or lower the frequency.")
        # Only whole pulses: a stimulator does not begin a pulse it cannot
        # finish before the frame is over.
        room = last - delay - pulse_len
        n_pulses = np.where(firing & (room >= 0),
                            room // np.maximum(window, 1) + 1, 0)
        # An electrode that was asked to fire but has no room left to do so has
        # been rastered out of its own frame, which would otherwise drop its
        # stimulus without saying so:
        silent = firing & (n_pulses == 0)
        if np.any(silent):
            worst = np.max(np.broadcast_to(delay, freq.shape)[silent])
            raise ValueError(
                f"{np.count_nonzero(silent.any(axis=1))} electrode(s) start "
                f"their turn as late as {worst * DT:.3f} ms into the frame, "
                f"leaving no room for a {pulse_len * DT:.3f} ms pulse before "
                f"it ends. Use fewer raster groups, or a shorter "
                f"'phase_dur'.")
        # Everything about a schedule is fixed by these three numbers:
        key = np.stack([window.ravel(), n_pulses.ravel(),
                        np.broadcast_to(delay, freq.shape).ravel()], axis=1)
        uniq, group = np.unique(key, axis=0, return_inverse=True)
        onsets = [d + np.arange(n, dtype=np.int64) * w for w, n, d in uniq]
        return np.reshape(group, freq.shape), onsets

    @staticmethod
    def _sample(onset, pulse_ticks, pulse_vals, ticks):
        """Sample one schedule's pulse train onto a frame's time axis"""
        t = (onset[:, np.newaxis] + pulse_ticks[np.newaxis, :]).ravel()
        v = np.tile(pulse_vals, onset.size)
        # Back-to-back pulses share an end point, which both copies of the
        # pulse put at zero:
        t, keep = np.unique(t, return_index=True)
        return np.interp(ticks, t, v[keep])

    def _assemble(self, amp, freq, delay, electrodes, frame_time, frame_dur):
        """Build the pulse trains for every electrode and frame

        Electrodes on the same schedule share the *shape* of their within-frame
        waveform; only the amplitude that scales it differs. So rather than
        building one waveform per electrode, this builds one per distinct
        schedule and indexes into them, which is what keeps frequency
        modulation (thousands of electrode-frames, a few dozen schedules)
        tractable.

        Each frame gets its own time axis -- the union of the times the
        schedules *it* uses need -- rather than one axis shared by the whole
        video. That matters when there are many schedules but each frame draws
        on only a few of them, which is exactly the unquantized frequency
        modulation case: a shared axis costs 20 times more there.
        """
        n_el, n_frames = len(electrodes), frame_time.size
        shape = (n_el, n_frames)
        amp = np.broadcast_to(np.asarray(amp, dtype=np.float32), shape)
        freq = np.broadcast_to(np.asarray(freq, dtype=np.float64), shape)
        delay = self._ticks(delay).reshape((-1, 1))
        # The last tick a frame may occupy. It has to leave at least DT before
        # the next frame starts, and `frame_dur` is generally not a whole
        # number of ticks (a 29.97 fps frame is 33.3667 ms), so this floors
        # rather than rounds. The epsilon keeps a frame duration that *is* a
        # whole number of ticks from losing one to binary rounding:
        last = int(np.floor((frame_dur - DT) / DT + 1e-9))
        pulse_ticks, pulse_vals = self._unit_pulse()
        if pulse_ticks[-1] > last:
            raise ValueError(f"A pulse (dur={pulse_ticks[-1] * DT:.3f} ms) "
                             f"does not fit into a frame "
                             f"(dur={frame_dur:.3f} ms). Shorten 'phase_dur' "
                             f"or lower the frame rate.")
        slowest = np.min(freq[freq > 0], initial=np.inf)
        if slowest < 1000.0 / frame_dur:
            warnings.warn(f"freq={slowest:.2f} Hz is slower than the frame "
                          f"rate ({1000.0 / frame_dur:.2f} Hz), so each frame "
                          f"still receives one pulse. The effective pulse "
                          f"rate is the frame rate.")
        group, onsets = self._schedule(freq, delay, last, pulse_ticks)

        # Lay each frame out on just the time points its own schedules need,
        # rather than on the union over the whole video. A frame in which every
        # electrode is dark then costs two time points instead of hundreds.
        # Frames often draw on the same handful of schedules, so the layout is
        # cached by which ones they use:
        layout, cache = [], {}
        for k in range(n_frames):
            present, local = np.unique(group[:, k], return_inverse=True)
            # NumPy has changed the shape `return_inverse` comes back with
            # between 2.x releases, and it indexes rows below:
            local = np.ravel(local)
            key = present.tobytes()
            if key not in cache:
                # Tick 0 and the last tick pin the frame to its full duration,
                # whatever the schedules do in between:
                cache[key] = np.unique(np.concatenate(
                    [np.array([0, last], dtype=np.int64)] +
                    [(onsets[g][:, np.newaxis] +
                      pulse_ticks[np.newaxis, :]).ravel()
                     for g in present if onsets[g].size]))
            layout.append((present, local, cache[key]))
        n_time = sum(ticks.size for _, _, ticks in layout)
        if n_time > _BIG_TIME:
            warnings.warn(f"This stimulus has {n_time} time points, which "
                          f"every model downstream will pay for. Coarsening "
                          f"'clock' puts more electrodes on the same pulse "
                          f"schedule; 'n_levels' reduces how many schedules "
                          f"there are.")
        if n_el * n_time > _BIG_STIM and self.implant is None:
            warnings.warn(f"Encoding {n_el} electrodes x {n_time} time points "
                          f"will allocate {n_el * n_time * 4 / 1e9:.1f} GB. "
                          f"Pass 'implant' to encode at electrode resolution "
                          f"instead.")

        # One unit-amplitude waveform per schedule present in the frame, scaled
        # by each electrode's amplitude. Sampling a schedule onto the frame's
        # time axis is exact: its own time points are a subset of that axis,
        # and everything between two of them is either a plateau or a gap, both
        # of which linear interpolation reproduces.
        data = np.empty((n_el, n_time), dtype=np.float32)
        time = np.empty(n_time, dtype=np.float64)
        col = 0
        for k, (present, local, ticks) in enumerate(layout):
            wave = np.zeros((present.size, ticks.size), dtype=np.float32)
            for idx, g in enumerate(present):
                if onsets[g].size:
                    wave[idx] = self._sample(onsets[g], pulse_ticks,
                                             pulse_vals, ticks)
            block = data[:, col:col + ticks.size]
            np.multiply(amp[:, k:k + 1], wave[local], out=block)
            time[col:col + ticks.size] = frame_time[k] + ticks * DT
            col += ticks.size
        # There is no frame after the last one to keep clear of, so let the
        # stimulus last exactly as long as the source did:
        time[-1] = frame_time[-1] + frame_dur
        stim = Stimulus(data, electrodes=electrodes, time=time)
        # Provenance, not something a model should compute from: the stimulus
        # is a plain `Stimulus` and every consumer reads its data container:
        stim.metadata['encoder'] = {'kind': type(self).__name__,
                                    'frame_dur': frame_dur,
                                    'n_frames': n_frames,
                                    'n_schedules': len(onsets)}
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
        delay = self._delays(electrodes, frame_dur)
        return self._assemble(amp, freq, delay, electrodes, frame_time,
                              frame_dur)


class AmplitudeEncoder(Encoder):
    """Encode gray levels as pulse amplitudes

    Every electrode emits a pulse train of the same fixed frequency, and the
    gray level of the pixel it sees sets the amplitude of those pulses. This is
    how most retinal prostheses encode a video.

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
        Pulse train frequency (Hz), the same for every electrode.

        .. note::

           A frequency below the frame rate cannot be realized: a frame that is
           shorter than one pulse train cycle still receives a single pulse, so
           the effective pulse rate is the frame rate. Encoding warns when this
           happens.

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
        if np.any(np.asarray(amp_range) < 0):
            raise ValueError(f"'amp_range' cannot be negative: the sign of "
                             f"the pulse is set by 'cathodic_first', not by "
                             f"the amplitude. Got {amp_range}.")
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

       Both ``clock`` and ``n_levels`` cut that down, and both are physically
       motivated -- real stimulators have a time base, and real encoders have
       an input resolution. Encoding the 94-frame ``BostonTrain`` for Argus II
       at frequencies in (0, 300] Hz, and predicting a 65x97 percept from it:

       ===================  ===========  ==========  ==============
       setting              time points  percept     predict_percept
       ===================  ===========  ==========  ==============
       (amplitude modulat.)         752       17 MB          2.7 s
       no quantization          121,494      3.0 GB              --
       ``clock=1``               18,056      453 MB         11.8 s
       ``clock=2``               10,083      252 MB          7.4 s
       ``n_levels=8``            17,537      440 MB         10.5 s
       ``clock=1, n_levels=8``   13,674      343 MB          9.2 s
       ===================  ===========  ==========  ==============

       Start with ``clock=1``, which costs nothing in frequency range.

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

           A frame can only hold a whole number of pulses, so at video frame
           rates the realizable frequencies are coarsely spaced no matter what
           range is asked for: a 33 ms frame fits at most ten 300 Hz pulses,
           and so can express about ten levels of gray.
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
        if np.any(np.asarray(freq_range) < 0):
            raise ValueError(f"'freq_range' cannot be negative: {freq_range}.")
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
