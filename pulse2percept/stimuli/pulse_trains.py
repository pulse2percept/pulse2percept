""":py:class:`~pulse2percept.stimuli.PulseTrain`, 
   :py:class:`~pulse2percept.stimuli.BiphasicPulseTrain`, 
   :py:class:`~pulse2percept.stimuli.AsymmetricBiphasicPulseTrain`"""
import numpy as np
from copy import deepcopy
from math import isclose

# DT: Sampling time step (ms); defines the duration of the signal edge
# transitions:
from .base import Stimulus
from .pulses import (AsymmetricBiphasicPulse, BiphasicPulse,
                     MonophasicPulse, _electrode_names)
from ..units import Hz, as_value, ms, uA
from ..utils.constants import DT, MS_PER_S


def _tile_pulse(pulse, shift, n_pulses):
    """Concatenate ``n_pulses`` copies of ``pulse``, spaced by ``shift`` ms

    Vectorized equivalent of repeatedly calling
    ``pt = pt.append(pulse >> shift)``, which copies the ever-growing data
    container once per pulse (and is therefore quadratic in ``n_pulses``).

    Parameters
    ----------
    pulse : :py:class:`~pulse2percept.stimuli.Stimulus`
        A stimulus containing a single pulse, with a time component.
    shift : float
        Time (ms) by which each copy is shifted with respect to the previous
        one, in addition to the duration of the pulse itself.
    n_pulses : int
        Number of copies to concatenate.

    Returns
    -------
    data, time : np.ndarray
        The data container and time axis of the concatenated pulse train.
    """
    time, data = pulse.time, pulse.data
    # The time axis of each appended copy, i.e. of ``pulse >> shift``:
    shifted = time + shift
    if shifted[0] < 0:
        raise NotImplementedError("Appending a stimulus with a negative "
                                  "time axis is currently not supported.")
    # ``append`` offsets copy k by the last time point of copy k-1, so the
    # offsets follow the recurrence last[k] = shifted[-1] + last[k-1], seeded
    # with last[0] = time[-1]. The cumsum accumulates in exactly the same
    # order (and therefore rounds identically), which matters because temporal
    # models resolve stimulus edges on a fixed simulation grid:
    steps = np.full(n_pulses, shifted[-1], dtype=np.float64)
    steps[0] = time[-1]
    offsets = np.cumsum(steps, dtype=np.float64)[:-1, np.newaxis]
    if isclose(shifted[0], 0, abs_tol=DT):
        # The last time point of one copy coincides with the first time point
        # of the next, so the two are merged into one - but only if they carry
        # the same amplitude(s):
        if not np.allclose(data[:, 0], data[:, -1]):
            raise ValueError(f"Data mismatch: Cannot append other stimulus "
                             f"because other[t=0] != this[t={time[-1]}ms]. "
                             f"You may need to shift the other stimulus in "
                             f"time by at least {DT:.1e} ms.")
        new_time = np.concatenate((time, (shifted[1:] + offsets).ravel()))
        new_data = np.hstack((data, np.tile(data[:, 1:], n_pulses - 1)))
    else:
        new_time = np.concatenate((time, (shifted + offsets).ravel()))
        new_data = np.tile(data, n_pulses)
    return new_data, new_time


class PulseTrain(Stimulus):
    """Generic pulse train

    Can be used to concatenate single pulses into a pulse train.

    .. seealso ::

        * :py:class:`~pulse2percept.stimuli.BiphasicPulseTrain`
        * :py:class:`~pulse2percept.stimuli.AsymmetricBiphasicPulseTrain`

    .. versionadded:: 0.6

    Parameters
    ----------
    freq : float
        Pulse train frequency (Hz).
    pulse : :py:class:`~pulse2percept.stimuli.Stimulus`
        A Stimulus object containing a single pulse that will be concatenated.
    n_pulses : int
        Number of pulses requested in the pulse train. If None, the entire
        stimulation window (``stim_dur``) is filled.
    stim_dur : float, optional
        Total stimulus duration (ms). The pulse train will be trimmed to make
        the stimulus last ``stim_dur`` ms overall.
    electrode : { int | string }, optional
        Optionally, you can provide your own electrode name.
    metadata : dict
        A dictionary of meta-data

    Notes
    -----
    *  Only pulses that fit whole are delivered. If the pulse train frequency
       does not exactly divide ``stim_dur``, the number of pulses is therefore
       rounded down: a 30 Hz train in a 33.37 ms window has one pulse, not one
       and a fraction of a second. A partial pulse would leave the train with a
       net current.
    *  A frequency slower than ``1000 / stim_dur`` cannot be realized, since
       the window still holds one pulse. Pass ``freq=0`` for a silent train.
    *  Arguments may be given as plain numbers in the units documented above,
       or as unitful quantities (e.g. ``0.02 * kHz``, ``1 * s``), which are
       converted to those units. See :py:mod:`pulse2percept.units`.
    *  The train is measured in whatever ``pulse`` was measured in: tiling an
       electrical pulse gives a train in microamps, and tiling a dimensionless
       one gives a dimensionless train.

    """
    #: Defined by the pulse it repeats rather than by its samples
    #: (see `Stimulus._is_parametric`):
    _is_parametric = True

    __slots__ = ('_freq', '_pulse', '_n_pulses', '_n_pulses_asked',
                 '_stim_dur')

    def __init__(self, freq, pulse, n_pulses=None, stim_dur=1000.0,
                 electrode=None, metadata=None):
        # Strip the units first; everything below is plain numbers in Hz and
        # ms, exactly as it has always been:
        freq = as_value(freq, Hz, 'freq')
        stim_dur = as_value(stim_dur, ms, 'stim_dur')
        if not isinstance(pulse, Stimulus):
            raise TypeError(f"'pulse' must be a Stimulus object, not "
                            f"{type(pulse)}.")
        n_rows = len(pulse.electrodes)
        if n_rows == 0:
            raise ValueError(f"'pulse' has invalid shape "
                             f"({pulse.shape[0]}, {pulse.shape[1]}).")
        # Every parameter-backed stimulus in the library is a pulse, and a
        # pulse always has a time axis. So this only has to ask a raw
        # stimulus, whose waveform is already there to be asked:
        if not pulse._is_parametric and pulse.time is None:
            raise ValueError("'pulse' does not have a time component.")
        # `duration` rather than `time[-1]`: a parametric pulse knows how long
        # it is from its parameters, so none of the arithmetic below has to
        # generate a waveform to find out.
        pulse_dur = pulse.duration

        # How many pulses fit into stim dur. `freq` counts cycles per second
        # and `stim_dur` counts milliseconds, so this is the one place the two
        # clocks have to be reconciled:
        n_max_pulses = freq * stim_dur / MS_PER_S
        # Kept as it was asked for, alongside the count it resolves to below.
        # The two are not interchangeable: this guard measures a request
        # against `n_max_pulses`, while the default counts whole pulses from
        # t=0 and can legitimately come out one higher. Rebuilding a train
        # (see `_scaled`) has to pass the request back, not the result.
        self._n_pulses_asked = n_pulses
        # The requested number of pulses cannot be greater than max pulses:
        if n_pulses is not None:
            n_pulses = int(n_pulses)
            if n_pulses > n_max_pulses:
                raise ValueError(f"stim_dur={stim_dur:.2f} cannot fit more than "
                                 f"{n_max_pulses} pulses.")
        elif freq <= 0:
            n_pulses = 0
        else:
            # Only whole pulses: a pulse that cannot finish before `stim_dur`
            # is over is not delivered at all. Starting one and cutting it
            # short would leave the train with a net current -- a 30 Hz train
            # in a 33.37 ms window used to end on half a cathodic phase, and
            # so was not charge-balanced.
            n_pulses = int(np.floor((stim_dur - pulse_dur) /
                                    (MS_PER_S / freq) + 1e-9)) + 1
        # 0 Hz is allowed, and so is a pulse too long to fit even once. A
        # silent train is a single row of zeros, whatever the pulse looked
        # like, which is what this class has always produced:
        if n_pulses <= 0:
            n_rows = 1
        else:
            # Window duration (ms) is the inverse of pulse train frequency.
            # Asked here rather than in `_render`, so that a pulse too long to
            # fit is refused where the caller can see why:
            window_dur = MS_PER_S / freq
            if pulse_dur > window_dur:
                raise ValueError(f"Pulse (dur={pulse_dur:.2f} ms) does not fit into "
                                 f"pulse train window (dur={window_dur:.2f} "
                                 f"ms)")
        if electrode is None:
            names = np.arange(n_rows)
        else:
            names = np.array([electrode]).ravel()
            if len(names) != n_rows:
                raise ValueError(f"Number of electrodes provided "
                                 f"({len(names)}) does not match the number "
                                 f"of electrodes in the pulse ({n_rows}).")
        self._freq = freq
        # A snapshot, not the caller's object: tiling used to copy the pulse's
        # values into the train there and then, so a pulse the caller goes on
        # to replace must not change a train already built from it. Immutable
        # waveform state makes this share arrays rather than duplicate them.
        self._pulse = deepcopy(pulse)
        self._n_pulses = n_pulses
        self._stim_dur = stim_dur
        # This class tiles whatever pulse it is handed, and the tiled numbers
        # mean whatever that pulse's did -- including the zeros of a silent
        # train. Without this the result would fall back to the default
        # (current) reading of them:
        self._defer(names, unit=pulse.unit, time_unit=pulse.time_unit)
        self.metadata = {'user': metadata}

    @property
    def freq(self):
        """Pulse train frequency (Hz)"""
        return self._freq

    @property
    def pulse(self):
        """The single pulse this train repeats"""
        return self._pulse

    @property
    def n_pulses(self):
        """Number of pulses delivered

        Resolved at construction: ``n_pulses=None`` means as many whole
        pulses as fit into ``stim_dur``.
        """
        return self._n_pulses

    @property
    def stim_dur(self):
        """Total stimulus duration (ms)"""
        return self._stim_dur

    @property
    def pulse_type(self):
        """Name of the class the repeated pulse came from"""
        return self._pulse.__class__.__name__

    def _render(self):
        """Tile the pulse into the train the parameters above describe"""
        pulse, freq = self._pulse, self._freq
        n_pulses, stim_dur = self._n_pulses, self._stim_dur
        if n_pulses <= 0:
            time = np.array([0, stim_dur], dtype=np.float64)
            data = np.array([[0, 0]], dtype=np.float32)
        else:
            # Window duration (ms) is the inverse of pulse train frequency:
            window_dur = MS_PER_S / freq
            shift = np.maximum(0, window_dur - pulse.duration)
            data, time = _tile_pulse(pulse, shift, n_pulses)
        if time[-1] > stim_dur + DT:
            # If stimulus is longer than the requested `stim_dur`, trim it.
            # Make sure to interpolate the end point:
            last_col = [np.interp(stim_dur, time, row) for row in data]
            last_col = np.array(last_col).reshape((-1, 1))
            t_idx = time < stim_dur
            # The interpolated end point has to stay at least DT away from the
            # last point it follows, or the time axis is no longer strictly
            # increasing:
            kept = np.flatnonzero(t_idx)
            if kept.size and time[kept[-1]] > stim_dur - DT:
                t_idx[kept[-1]] = False
            data = np.hstack((data[:, t_idx], last_col))
            time = np.append(time[t_idx], stim_dur)
        elif time[-1] < stim_dur - DT:
            # If stimulus is shorter than the requested `stim_dur`, add a zero:
            data = np.hstack((data, np.zeros((data.shape[0], 1))))
            time = np.append(time, stim_dur)
        return {'data': data, 'electrodes': self.electrodes, 'time': time}

    def _scaled(self, factor):
        """This train, tiling a pulse whose amplitudes were scaled

        Tiling copies the pulse without doing arithmetic on it, so scaling
        every sample of the train is the same thing as scaling the pulse it
        repeats -- and the train's own parameters (frequency, pulse count,
        window) are untouched by either.
        """
        return PulseTrain(self.freq, self.pulse * factor,
                          n_pulses=self._n_pulses_asked,
                          stim_dur=self.stim_dur,
                          electrode=self.electrodes,
                          metadata=deepcopy(self.metadata.get('user')))

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print

        The parameters that define the train rather than the waveform they
        describe, so that printing one does not generate it.
        """
        return {'freq': self.freq, 'pulse_type': self.pulse_type,
                'n_pulses': self.n_pulses, 'stim_dur': self.stim_dur,
                'electrodes': self.electrodes, 'metadata': self.metadata}


class BiphasicPulseTrain(Stimulus):
    """Symmetric biphasic pulse train

    A train of symmetric biphasic pulses.

    .. versionadded:: 0.6

    Parameters
    ----------
    freq : float
        Pulse train frequency (Hz).
    amp : float
        Current amplitude (uA). Negative currents: cathodic, positive: anodic.
        The sign will be converted automatically depending on
        ``cathodic_first``.
    phase_dur : float
        Duration (ms) of the cathodic/anodic phase.
    interphase_dur : float, optional, default: 0
        Duration (ms) of the gap between cathodic and anodic phases.
    delay_dur : float
        Delay duration (ms). Zeros will be inserted at the beginning of the
        stimulus to deliver the first pulse phase after ``delay_dur`` ms.
    n_pulses : int
        Number of pulses requested in the pulse train. If None, the entire
        stimulation window (``stim_dur``) is filled.
    stim_dur : float, optional, default: 1000 ms
        Total stimulus duration (ms). The pulse train will be trimmed to make
        the stimulus last ``stim_dur`` ms overall.
    cathodic_first : bool, optional, default: True
        If True, will deliver the cathodic pulse phase before the anodic one.
    electrode : { int | string }, optional, default: 0
        Optionally, you can provide your own electrode name.
    metadata : dict
        A dictionary of meta-data

    Notes
    -----
    *  Each cycle ("window") of the pulse train consists of a symmetric
       biphasic pulse, created with
       :py:class:`~pulse2percept.stimuli.BiphasicPulse`.
    *  The order and sign of the two phases (cathodic/anodic) of each pulse
       in the train is automatically adjusted depending on the
       ``cathodic_first`` flag.
    *  A pulse train will be considered "charge-balanced" if its net current is
       smaller than 10 picoamps.
    *  Arguments may be given as plain numbers in the units documented above,
       or as unitful quantities (e.g. ``0.05 * mA``, ``450 * us``), which are
       converted to those units. See :py:mod:`pulse2percept.units`.

    """
    #: Defined by the pulse it repeats rather than by its samples
    #: (see `Stimulus._is_parametric`):
    _is_parametric = True

    __slots__ = ('_train',)

    def __init__(self, freq, amp, phase_dur, interphase_dur=0, delay_dur=0,
                 n_pulses=None, stim_dur=1000.0, cathodic_first=True,
                 electrode=None, metadata=None):
        # See `PulseTrain.__init__`. Normalizing here rather than leaving it to
        # `BiphasicPulse` is what keeps the metadata below in the units a model
        # reading it back expects:
        freq = as_value(freq, Hz, 'freq')
        amp = as_value(amp, uA, 'amp')
        phase_dur = as_value(phase_dur, ms, 'phase_dur')
        interphase_dur = as_value(interphase_dur, ms, 'interphase_dur')
        delay_dur = as_value(delay_dur, ms, 'delay_dur')
        stim_dur = as_value(stim_dur, ms, 'stim_dur')
        # Create the individual pulse:
        pulse = BiphasicPulse(amp, phase_dur, delay_dur=delay_dur,
                              interphase_dur=interphase_dur,
                              cathodic_first=cathodic_first,
                              electrode=electrode)
        # Concatenate the pulses. Built here rather than in `_render`, so that
        # every argument is still checked at construction; neither object
        # generates a waveform until one is asked for.
        self._train = PulseTrain(freq, pulse, n_pulses=n_pulses,
                                 stim_dur=stim_dur)
        self._defer(_electrode_names(electrode))

        # Store metadata for BiphasicAxonMapModel. `amp` is stored as a
        # magnitude, because that is all of it that reaches the data:
        # `BiphasicPulse` takes `np.abs(amp)` and reads the polarity off
        # `cathodic_first`. Storing the sign the caller happened to type would
        # have two identical waveforms predict two different percepts, since
        # the models are functions of `amp` and not of `abs(amp)`.
        self.metadata = {'freq': freq,
                         'amp': abs(amp),
                         'phase_dur': phase_dur,
                         'delay_dur': delay_dur,
                         'user': metadata}

    @property
    def freq(self):
        """Pulse train frequency (Hz)"""
        return self._train.freq

    @property
    def n_pulses(self):
        """Number of pulses delivered"""
        return self._train.n_pulses

    @property
    def stim_dur(self):
        """Total stimulus duration (ms)"""
        return self._train.stim_dur

    @property
    def amp(self):
        """Magnitude (uA) of both phases of each pulse"""
        return self._train.pulse.amp

    @property
    def phase_dur(self):
        """Duration (ms) of the cathodic/anodic phase"""
        return self._train.pulse.phase_dur

    @property
    def interphase_dur(self):
        """Duration (ms) of the gap between the two phases"""
        return self._train.pulse.interphase_dur

    @property
    def delay_dur(self):
        """Delay (ms) before the first phase of each pulse"""
        return self._train.pulse.delay_dur

    @property
    def cathodic_first(self):
        """Whether the cathodic phase is delivered first"""
        return self._train.pulse.cathodic_first

    def _render(self):
        """The tiled train the parameters above describe"""
        return {'data': self._train.data, 'electrodes': self.electrodes,
                'time': self._train.time}

    def _scaled(self, factor):
        """This train with every amplitude multiplied by ``factor``

        Built from the parameters rather than from the samples: ``amp`` is
        canonical state now, so twice this train is the train at twice its
        amplitude -- not the float32 waveform cache doubled and its rounding
        promoted to truth. The two agree to within a float32 ulp.
        """
        return BiphasicPulseTrain(
            self.freq, self.amp * abs(factor), self.phase_dur,
            interphase_dur=self.interphase_dur, delay_dur=self.delay_dur,
            n_pulses=self._train._n_pulses_asked,
            stim_dur=self.stim_dur,
            # A negative factor swaps the two phases, which is exactly what
            # this flag says:
            cathodic_first=(self.cathodic_first if factor >= 0
                            else not self.cathodic_first),
            electrode=self.electrodes[0],
            # The compatibility metadata is rebuilt by the constructor, from
            # the new amplitude. Only what the user put there is theirs to
            # carry across:
            metadata=deepcopy(self.metadata.get('user')))

    @classmethod
    def _rescale_params(cls, metadata, factor):
        """Keep the pulse parameters in sync with the data

        :py:class:`~pulse2percept.models.BiphasicAxonMapModel` and
        :py:class:`~pulse2percept.models.cortex.DynaphosModel` read amplitude,
        frequency and phase duration off the metadata rather than off the data.
        An operation that rewrites the data but leaves the metadata behind
        makes the two disagree: ``pt * 2`` delivers twice the current, and the
        model would go on predicting the very same percept.

        Scaling is the one operation that leaves a biphasic pulse train a
        biphasic pulse train, so it scales ``amp`` (a negative factor only
        swaps the two phases, which does not change the magnitude). Anything
        else -- a DC offset, an appended second train, a non-finite factor --
        leaves something that is no longer one biphasic pulse train at one
        amplitude and frequency, so the pulse parameters are dropped: a model
        asking for them then rejects the stimulus rather than predicting from
        numbers that no longer describe it. What the user put in ``metadata``
        is theirs, and survives either way.
        """
        if 'amp' not in metadata:
            # Parameters an earlier operation has already dropped
            return metadata
        if factor is None:
            return {'user': metadata.get('user')}
        return dict(metadata, amp=abs(metadata['amp'] * factor))

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print

        The parameters that define the train rather than the waveform they
        describe, so that printing one does not generate it.
        """
        return {'freq': self.freq, 'amp': self.amp,
                'phase_dur': self.phase_dur,
                'interphase_dur': self.interphase_dur,
                'delay_dur': self.delay_dur, 'stim_dur': self.stim_dur,
                'cathodic_first': self.cathodic_first,
                'electrodes': self.electrodes, 'metadata': self.metadata}


class AsymmetricBiphasicPulseTrain(Stimulus):
    """Asymmetric biphasic pulse

    A simple stimulus consisting of a single biphasic pulse: a cathodic and an
    anodic phase, optionally separated by an interphase gap.
    The two pulse phases can have different amplitudes and duration
    ("asymmetric").
    The order of the two phases is given by the ``cathodic_first`` flag.

    .. versionadded:: 0.6

    Parameters
    ----------
    freq : float
        Pulse train frequency (Hz).
    amp1, amp2 : float
        Current amplitude (uA) of the first and second pulse phases.
        Negative currents: cathodic, positive: anodic.
        The signs will be converted automatically depending on
        ``cathodic_first``.
    phase_dur1, phase_dur2 : float
        Duration (ms) of the first and second pulse phases.
    interphase_dur : float, optional, default: 0
        Duration (ms) of the gap between cathodic and anodic phases.
    delay_dur : float
        Delay duration (ms). Zeros will be inserted at the beginning of the
        stimulus to deliver the first pulse phase after ``delay_dur`` ms.
    n_pulses : int
        Number of pulses requested in the pulse train. If None, the entire
        stimulation window (``stim_dur``) is filled.
    stim_dur : float, optional, default: 1000 ms
        Total stimulus duration (ms). Zeros will be inserted at the end of the
        stimulus to make the the stimulus last ``stim_dur`` ms overall.
    cathodic_first : bool, optional, default: True
        If True, will deliver the cathodic pulse phase before the anodic one.
    electrode : { int | string }, optional, default: 0
        Optionally, you can provide your own electrode name.
    metadata : dict
        A dictionary of meta-data

    Notes
    -----
    *  Arguments may be given as plain numbers in the units documented above,
       or as unitful quantities (e.g. ``0.05 * mA``, ``450 * us``), which are
       converted to those units. See :py:mod:`pulse2percept.units`.

    """
    #: Defined by the pulse it repeats rather than by its samples
    #: (see `Stimulus._is_parametric`):
    _is_parametric = True

    __slots__ = ('_train',)

    def __init__(self, freq, amp1, amp2, phase_dur1, phase_dur2,
                 interphase_dur=0, delay_dur=0, n_pulses=None, stim_dur=1000.0,
                 cathodic_first=True, electrode=None, metadata=None):
        # See `PulseTrain.__init__`:
        freq = as_value(freq, Hz, 'freq')
        amp1 = as_value(amp1, uA, 'amp1')
        amp2 = as_value(amp2, uA, 'amp2')
        phase_dur1 = as_value(phase_dur1, ms, 'phase_dur1')
        phase_dur2 = as_value(phase_dur2, ms, 'phase_dur2')
        interphase_dur = as_value(interphase_dur, ms, 'interphase_dur')
        delay_dur = as_value(delay_dur, ms, 'delay_dur')
        stim_dur = as_value(stim_dur, ms, 'stim_dur')
        # Create the individual pulse:
        pulse = AsymmetricBiphasicPulse(amp1, amp2, phase_dur1, phase_dur2,
                                        delay_dur=delay_dur,
                                        interphase_dur=interphase_dur,
                                        cathodic_first=cathodic_first,
                                        electrode=electrode)
        # Concatenate the pulses (see `BiphasicPulseTrain.__init__`):
        self._train = PulseTrain(freq, pulse, n_pulses=n_pulses,
                                 stim_dur=stim_dur)
        self._defer(_electrode_names(electrode))
        self.metadata = {'user': metadata}

    @property
    def freq(self):
        """Pulse train frequency (Hz)"""
        return self._train.freq

    @property
    def n_pulses(self):
        """Number of pulses delivered"""
        return self._train.n_pulses

    @property
    def stim_dur(self):
        """Total stimulus duration (ms)"""
        return self._train.stim_dur

    @property
    def amp1(self):
        """Magnitude (uA) of the first phase of each pulse"""
        return self._train.pulse.amp1

    @property
    def amp2(self):
        """Magnitude (uA) of the second phase of each pulse"""
        return self._train.pulse.amp2

    @property
    def phase_dur1(self):
        """Duration (ms) of the first pulse phase"""
        return self._train.pulse.phase_dur1

    @property
    def phase_dur2(self):
        """Duration (ms) of the second pulse phase"""
        return self._train.pulse.phase_dur2

    @property
    def interphase_dur(self):
        """Duration (ms) of the gap between the two phases"""
        return self._train.pulse.interphase_dur

    @property
    def delay_dur(self):
        """Delay (ms) before the first phase of each pulse"""
        return self._train.pulse.delay_dur

    @property
    def cathodic_first(self):
        """Whether the cathodic phase is delivered first"""
        return self._train.pulse.cathodic_first

    def _render(self):
        """The tiled train the parameters above describe"""
        return {'data': self._train.data, 'electrodes': self.electrodes,
                'time': self._train.time}

    def _scaled(self, factor):
        """This train with both phase magnitudes multiplied by ``factor``

        See :py:meth:`BiphasicPulseTrain._scaled`.
        """
        return AsymmetricBiphasicPulseTrain(
            self.freq, self.amp1 * abs(factor), self.amp2 * abs(factor),
            self.phase_dur1, self.phase_dur2,
            interphase_dur=self.interphase_dur, delay_dur=self.delay_dur,
            n_pulses=self._train._n_pulses_asked,
            stim_dur=self.stim_dur,
            cathodic_first=(self.cathodic_first if factor >= 0
                            else not self.cathodic_first),
            electrode=self.electrodes[0],
            metadata=deepcopy(self.metadata.get('user')))

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print

        The parameters that define the train rather than the waveform they
        describe, so that printing one does not generate it.
        """
        return {'freq': self.freq, 'amp1': self.amp1, 'amp2': self.amp2,
                'phase_dur1': self.phase_dur1,
                'phase_dur2': self.phase_dur2,
                'interphase_dur': self.interphase_dur,
                'delay_dur': self.delay_dur, 'stim_dur': self.stim_dur,
                'cathodic_first': self.cathodic_first,
                'electrodes': self.electrodes, 'metadata': self.metadata}


class BiphasicTripletTrain(Stimulus):
    """Biphasic pulse triplets

    A train of symmetric biphasic pulse triplets.

    .. versionadded:: 0.6

    Parameters
    ----------
    freq : float
        Pulse train frequency (Hz).
    amp : float
        Current amplitude (uA). Negative currents: cathodic, positive: anodic.
        The sign will be converted automatically depending on
        ``cathodic_first``.
    phase_dur : float
        Duration (ms) of the cathodic/anodic phase.
    interphase_dur : float, optional, default: 0
        Duration (ms) of the gap between cathodic and anodic phases.
    delay_dur : float
        Delay duration (ms). Zeros will be inserted at the beginning of the
        stimulus to deliver the first pulse phase after ``delay_dur`` ms.
    interpulse_dur : float, optional, default: 0
        Delay duration (ms) between each biphasic pulse within the train. Note,
        this delay is also applied after the third biphasic pulse
    n_pulses : int
        Number of pulses requested in the pulse train. If None, the entire
        stimulation window (``stim_dur``) is filled.
    stim_dur : float, optional, default: 1000 ms
        Total stimulus duration (ms). The pulse train will be trimmed to make
        the stimulus last ``stim_dur`` ms overall.
    cathodic_first : bool, optional, default: True
        If True, will deliver the cathodic pulse phase before the anodic one.
    electrode : { int | string }, optional, default: 0
        Optionally, you can provide your own electrode name.
    metadata : dict
        A dictionary of meta-data

    Notes
    -----
    *  Each cycle ("window") of the pulse train consists of three biphasic
       pulses, created with
       :py:class:`~pulse2percept.stimuli.BiphasicPulse`.
    *  The order and sign of the two phases (cathodic/anodic) of each pulse
       in the train is automatically adjusted depending on the
       ``cathodic_first`` flag.
    *  A pulse train will be considered "charge-balanced" if its net current is
       smaller than 10 picoamps.
    *  Arguments may be given as plain numbers in the units documented above,
       or as unitful quantities (e.g. ``0.05 * mA``, ``450 * us``), which are
       converted to those units. See :py:mod:`pulse2percept.units`.

    """
    #: Defined by the pulse it repeats rather than by its samples
    #: (see `Stimulus._is_parametric`):
    _is_parametric = True

    __slots__ = ('_train', '_pulse', '_interpulse_dur')

    def __init__(self, freq, amp, phase_dur, interphase_dur=0, interpulse_dur=0,
                 delay_dur=0, n_pulses=None, stim_dur=1000.0, cathodic_first=True,
                 electrode=None, metadata=None):
        # See `PulseTrain.__init__`:
        freq = as_value(freq, Hz, 'freq')
        amp = as_value(amp, uA, 'amp')
        phase_dur = as_value(phase_dur, ms, 'phase_dur')
        interphase_dur = as_value(interphase_dur, ms, 'interphase_dur')
        interpulse_dur = as_value(interpulse_dur, ms, 'interpulse_dur')
        delay_dur = as_value(delay_dur, ms, 'delay_dur')
        stim_dur = as_value(stim_dur, ms, 'stim_dur')
        # Create the pulse:
        pulse = BiphasicPulse(amp, phase_dur, interphase_dur=interphase_dur,
                              delay_dur=delay_dur,
                              cathodic_first=cathodic_first,
                              electrode=electrode)
        # The pulse the triplet is made of, kept for the parameters below.
        # The triplet itself is assembled once, here, rather than at render
        # time: it is three copies of a handful of time points, and building
        # it is what checks that they fit together at all.
        self._pulse = pulse
        self._interpulse_dur = interpulse_dur
        if interpulse_dur != 0:
            # Create an interpulse 'delay' pulse. It has to sit on the same
            # electrode as `pulse`, or the two cannot be appended:
            delay_pulse = MonophasicPulse(0, interpulse_dur, electrode=electrode)
            pulse = pulse.append(delay_pulse)
        # Create the pulse triplet. Three copies of one pulse is not three
        # protocols, so this keeps the waveform rather than the sequence
        # `append` would otherwise hand back:
        triplet = Stimulus(pulse.append(pulse).append(pulse))
        # Create the triplet train (see `BiphasicPulseTrain.__init__`):
        self._train = PulseTrain(freq, triplet, n_pulses=n_pulses,
                                 stim_dur=stim_dur)
        self._defer(_electrode_names(electrode))
        self.metadata = {'user': metadata}

    @property
    def freq(self):
        """Pulse train frequency (Hz)"""
        return self._train.freq

    @property
    def n_pulses(self):
        """Number of pulses delivered"""
        return self._train.n_pulses

    @property
    def stim_dur(self):
        """Total stimulus duration (ms)"""
        return self._train.stim_dur

    @property
    def amp(self):
        """Magnitude (uA) of both phases of each pulse"""
        return self._pulse.amp

    @property
    def phase_dur(self):
        """Duration (ms) of the cathodic/anodic phase"""
        return self._pulse.phase_dur

    @property
    def interphase_dur(self):
        """Duration (ms) of the gap between the two phases"""
        return self._pulse.interphase_dur

    @property
    def interpulse_dur(self):
        """Delay (ms) after each pulse of the triplet"""
        return self._interpulse_dur

    @property
    def delay_dur(self):
        """Delay (ms) before the first phase of each pulse"""
        return self._pulse.delay_dur

    @property
    def cathodic_first(self):
        """Whether the cathodic phase is delivered first"""
        return self._pulse.cathodic_first

    def _render(self):
        """The tiled train the parameters above describe"""
        return {'data': self._train.data, 'electrodes': self.electrodes,
                'time': self._train.time}

    def _scaled(self, factor):
        """This train with every phase magnitude multiplied by ``factor``

        See :py:meth:`BiphasicPulseTrain._scaled`. The triplet structure --
        three pulses per window, ``interpulse_dur`` apart -- is untouched.
        """
        return BiphasicTripletTrain(
            self.freq, self.amp * abs(factor), self.phase_dur,
            interphase_dur=self.interphase_dur,
            interpulse_dur=self.interpulse_dur, delay_dur=self.delay_dur,
            n_pulses=self._train._n_pulses_asked,
            stim_dur=self.stim_dur,
            cathodic_first=(self.cathodic_first if factor >= 0
                            else not self.cathodic_first),
            electrode=self.electrodes[0],
            metadata=deepcopy(self.metadata.get('user')))

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print

        The parameters that define the train rather than the waveform they
        describe, so that printing one does not generate it.
        """
        return {'freq': self.freq, 'amp': self.amp,
                'phase_dur': self.phase_dur,
                'interphase_dur': self.interphase_dur,
                'interpulse_dur': self.interpulse_dur,
                'delay_dur': self.delay_dur, 'stim_dur': self.stim_dur,
                'cathodic_first': self.cathodic_first,
                'electrodes': self.electrodes, 'metadata': self.metadata}
