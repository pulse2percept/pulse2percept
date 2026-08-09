""":py:class:`~pulse2percept.stimuli.Encoder`,
   :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`"""
from abc import ABCMeta, abstractmethod
import warnings
import numpy as np

from .base import Stimulus
from .images import ImageStimulus
from .pulses import BiphasicPulse
from .pulse_trains import PulseTrain
from .videos import VideoStimulus
from ..utils import PrettyPrint, frame_interval
from ..utils.constants import DT

# Encoding a source that still has one row per *pixel* rather than one per
# electrode produces a data container this many elements large before anyone
# notices. Warn past that, because the fix (pass ``implant``) is a one-liner:
_BIG_STIM = 5e7

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
        resolution. This matters: a 240x426 movie has 102,240 pixels but Argus
        II has 60 electrodes, and pulse trains are two orders of magnitude
        wider than the frames they are built from.
    2.  Map those gray levels onto pulse train parameters (``_modulate``), then
        assemble the pulse trains (``_assemble``).

    Subclasses only implement ``_modulate``; everything else is provided here.

    .. versionadded:: 0.9.2

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
        is what the encoder sets; only its shape is used.
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
                 'pulse', 'frame_dur', 'stretch')

    def __init__(self, implant=None, phase_dur=0.46, interphase_dur=0,
                 cathodic_first=True, pulse=None, frame_dur=None,
                 stretch=False):
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
        if frame_dur is not None and frame_dur <= 0:
            raise ValueError("'frame_dur' must be positive.")
        self.implant = implant
        self.phase_dur = phase_dur
        self.interphase_dur = interphase_dur
        self.cathodic_first = cathodic_first
        self.pulse = pulse
        self.frame_dur = frame_dur
        self.stretch = stretch

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        return {'implant': self.implant, 'phase_dur': self.phase_dur,
                'interphase_dur': self.interphase_dur,
                'cathodic_first': self.cathodic_first, 'pulse': self.pulse,
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

    def _delays(self, n_electrodes):
        """Per-electrode delay (ms) into the frame window

        All electrodes fire at the start of each frame. Implants that cannot
        drive every electrode at once stagger them instead, which is what a
        raster group is; an encoder for such an implant overrides this.

        Returns
        -------
        delay : (n_electrodes,) array
            Delay (ms) between the start of a frame and this electrode's first
            pulse.
        """
        return np.zeros(n_electrodes, dtype=np.float32)

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
            frame_time = np.zeros(1, dtype=np.float32)
        elif self.frame_dur is None:
            # `frame_interval` rejects a time axis whose frames are not all
            # the same length, so the frames tile the source exactly:
            frame_dur = frame_interval(np.asarray(stim.time), fps=fps)
            frame_time = np.asarray(stim.time, dtype=np.float32)
        else:
            # An explicit `frame_dur` re-times the source: the frames keep
            # their order and their content, but each one now lasts
            # `frame_dur` ms. Keeping the source's own frame times here would
            # let neighboring frames overlap:
            frame_dur = self.frame_dur
            frame_time = (np.arange(gray.shape[1], dtype=np.float32) *
                          np.float32(frame_dur))
        return gray, stim.electrodes, frame_time, frame_dur

    def _waveform(self, freq, frame_dur):
        """Unit-amplitude pulse train that fills a single frame window

        Returns
        -------
        data, time : 1-D arrays
            The waveform and its time axis. The waveform peaks at -1 (cathodic
            first) or +1, so that multiplying it by an amplitude in uA yields
            the pulse train to deliver.

        """
        if self.pulse is None:
            pulse = BiphasicPulse(1, self.phase_dur,
                                  interphase_dur=self.interphase_dur,
                                  cathodic_first=self.cathodic_first)
        else:
            # Normalize to unit peak. `Stimulus.data` hands out the container
            # itself, so copy before scaling or the user's pulse is rescaled
            # along with ours:
            data = self.pulse.data.copy()
            peak = np.abs(data).max()
            if peak > 0:
                data /= peak
            pulse = Stimulus(data, time=self.pulse.time)
        if pulse.time[-1] > frame_dur - DT:
            raise ValueError(f"A pulse (dur={pulse.time[-1]:.3f} ms) does not "
                             f"fit into a frame (dur={frame_dur:.3f} ms). "
                             f"Shorten 'phase_dur' or lower the frame rate.")
        if freq > 0 and 1000.0 / freq > frame_dur:
            warnings.warn(f"freq={freq:.2f} Hz is slower than the frame rate "
                          f"({1000.0 / frame_dur:.2f} Hz), so each frame "
                          f"still receives one pulse. The effective pulse "
                          f"rate is the frame rate.")
        # Each frame's train stops DT short of the frame boundary, so that its
        # last time point does not collide with the next frame's first one:
        train = PulseTrain(freq, pulse, stim_dur=frame_dur - DT)
        return train.data.ravel(), np.asarray(train.time)

    def _assemble(self, amp, freq, delay, electrodes, frame_time, frame_dur):
        """Build the pulse trains for every electrode and frame

        Electrodes that share a frequency and a delay share the *shape* of
        their within-frame waveform; only its scale differs. When they all do
        -- amplitude modulation without rastering -- the whole stimulus is the
        outer product of the per-frame amplitudes with that one waveform, which
        is what makes it cheap to build.

        Frequency modulation and raster groups both break that: electrodes then
        pulse at different times, so there is no single within-frame waveform
        and the frames have to be assembled on a union time axis instead. This
        is where that path goes.
        """
        freq = np.unique(np.asarray(freq, dtype=np.float32))
        if freq.size > 1 or np.any(delay != 0):
            raise NotImplementedError(
                "Electrodes that do not share a pulse train frequency and "
                "delay do not share a time axis, so the stimulus cannot be "
                "assembled as an outer product. Frequency modulation and "
                "raster groups need the general (union time axis) path, which "
                "is not implemented yet.")
        wave, wave_time = self._waveform(freq[0], frame_dur)
        amp = np.broadcast_to(np.asarray(amp, dtype=np.float32),
                              (len(electrodes), frame_time.size))
        n_el, n_frames = amp.shape
        n_elem = n_el * n_frames * wave.size
        if n_elem > _BIG_STIM and self.implant is None:
            warnings.warn(f"Encoding {n_el} electrodes x {n_frames} frames "
                          f"into pulse trains of {wave.size} samples each "
                          f"will allocate {n_elem * 4 / 1e9:.1f} GB. Pass "
                          f"'implant' to encode at electrode resolution "
                          f"instead.")
        # Amplitude times waveform, for every frame, concatenated in time:
        data = (amp[:, :, np.newaxis] * wave[np.newaxis, np.newaxis, :])
        data = data.reshape((n_el, n_frames * wave.size))
        time = (frame_time[:, np.newaxis] + wave_time[np.newaxis, :]).ravel()
        # There is no frame after the last one to keep clear of, so let the
        # stimulus last exactly as long as the source did:
        time[-1] = frame_time[-1] + frame_dur
        stim = Stimulus(data, electrodes=electrodes, time=time)
        # Provenance, not something a model should compute from: the stimulus
        # is a plain `Stimulus` and every consumer reads its data container:
        stim.metadata['encoder'] = {'kind': type(self).__name__,
                                    'freq': float(freq[0]),
                                    'frame_dur': frame_dur,
                                    'n_frames': n_frames}
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
        amp, freq = self._modulate(gray)
        delay = self._delays(len(electrodes))
        return self._assemble(amp, freq, delay, electrodes, frame_time,
                              frame_dur)


class AmplitudeEncoder(Encoder):
    """Encode gray levels as pulse amplitudes

    Every electrode emits a pulse train of the same fixed frequency, and the
    gray level of the pixel it sees sets the amplitude of those pulses. This is
    how most retinal prostheses encode a video.

    .. versionadded:: 0.9.2

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
