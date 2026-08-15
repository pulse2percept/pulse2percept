""":py:class:`~pulse2percept.stimuli.PulseTrain`, 
   :py:class:`~pulse2percept.stimuli.BiphasicPulseTrain`, 
   :py:class:`~pulse2percept.stimuli.AsymmetricBiphasicPulseTrain`"""
import numpy as np
from math import isclose

# DT: Sampling time step (ms); defines the duration of the signal edge
# transitions:
from .base import Stimulus
from .pulses import BiphasicPulse, AsymmetricBiphasicPulse, MonophasicPulse
from ..utils.constants import DT


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

    """
    __slots__ = ('freq', 'pulse_type')

    def __init__(self, freq, pulse, n_pulses=None, stim_dur=1000.0,
                 electrode=None, metadata=None):
        if not isinstance(pulse, Stimulus):
            raise TypeError(f"'pulse' must be a Stimulus object, not "
                            f"{type(pulse)}.")
        if pulse.shape[0] == 0:
            raise ValueError(f"'pulse' has invalid shape "
                             f"({pulse.shape[0]}, {pulse.shape[1]}).")
        if pulse.time is None:
            raise ValueError("'pulse' does not have a time component.")

        # How many pulses fit into stim dur:
        n_max_pulses = freq * stim_dur / 1000.0
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
            n_pulses = int(np.floor((stim_dur - pulse.time[-1]) /
                                    (1000.0 / freq) + 1e-9)) + 1
        # 0 Hz is allowed, and so is a pulse too long to fit even once:
        if n_pulses <= 0:
            time = np.array([0, stim_dur], dtype=np.float64)
            data = np.array([[0, 0]], dtype=np.float32)
        else:
            # Window duration is the inverse of pulse train frequency:
            window_dur = 1000.0 / freq
            if pulse.time[-1] > window_dur:
                raise ValueError(f"Pulse (dur={pulse.time[-1]:.2f} ms) does not fit into "
                                 f"pulse train window (dur={window_dur:.2f} "
                                 f"ms)")
            shift = np.maximum(0, window_dur - pulse.time[-1])
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
            data = np.hstack((data, np.zeros((pulse.data.shape[0], 1))))
            time = np.append(time, stim_dur)
        super().__init__(data, time=time, electrodes=electrode, metadata=None,
                         compress=False)
        self.freq = freq
        self.pulse_type = pulse.__class__.__name__
        self.metadata = {'user': metadata}

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super()._pprint_params()
        params.update({'freq': self.freq,
                       'pulse_type': self.pulse_type})
        return params


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

    """
    __slots__ = ('freq', 'cathodic_first')

    def __init__(self, freq, amp, phase_dur, interphase_dur=0, delay_dur=0,
                 n_pulses=None, stim_dur=1000.0, cathodic_first=True,
                 electrode=None, metadata=None):
        # Create the individual pulse:
        pulse = BiphasicPulse(amp, phase_dur, delay_dur=delay_dur,
                              interphase_dur=interphase_dur,
                              cathodic_first=cathodic_first,
                              electrode=electrode)
        # Concatenate the pulses:
        pt = PulseTrain(freq, pulse, n_pulses=n_pulses, stim_dur=stim_dur)
        super().__init__(pt.data, time=pt.time, electrodes=electrode,
                         compress=False)
        self.freq = freq
        self.cathodic_first = cathodic_first

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

    def _rescale_metadata(self, factor):
        """Keep this train's own parameters, and its polarity, in sync"""
        if factor == 1:
            return
        self.metadata = self._rescale_params(self.metadata, factor)
        if factor is not None and factor < 0:
            # The two phases swapped places. `BiphasicPulse` reads the polarity
            # off this flag, so that is where the sign belongs:
            self.cathodic_first = not self.cathodic_first

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super()._pprint_params()
        params.update({'cathodic_first': self.cathodic_first,
                       'freq': self.freq})
        return params


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

    """
    __slots__ = ('freq', 'cathodic_first')

    def __init__(self, freq, amp1, amp2, phase_dur1, phase_dur2,
                 interphase_dur=0, delay_dur=0, n_pulses=None, stim_dur=1000.0,
                 cathodic_first=True, electrode=None, metadata=None):
        # Create the individual pulse:
        pulse = AsymmetricBiphasicPulse(amp1, amp2, phase_dur1, phase_dur2,
                                        delay_dur=delay_dur,
                                        interphase_dur=interphase_dur,
                                        cathodic_first=cathodic_first,
                                        electrode=electrode)
        # Concatenate the pulses:
        pt = PulseTrain(freq, pulse, n_pulses=n_pulses, stim_dur=stim_dur)
        super().__init__(pt.data, time=pt.time, electrodes=electrode,
                         compress=False)
        self.freq = freq
        self.cathodic_first = cathodic_first
        self.metadata = {'user': metadata}

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super()._pprint_params()
        params.update({'cathodic_first': self.cathodic_first,
                       'freq': self.freq})
        return params


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

    """
    __slots__ = ('freq', 'cathodic_first')

    def __init__(self, freq, amp, phase_dur, interphase_dur=0, interpulse_dur=0,
                 delay_dur=0, n_pulses=None, stim_dur=1000.0, cathodic_first=True,
                 electrode=None, metadata=None):
        # Create the pulse:
        pulse = BiphasicPulse(amp, phase_dur, interphase_dur=interphase_dur,
                              delay_dur=delay_dur,
                              cathodic_first=cathodic_first,
                              electrode=electrode)
        if interpulse_dur != 0:
            # Create an interpulse 'delay' pulse. It has to sit on the same
            # electrode as `pulse`, or the two cannot be appended:
            delay_pulse = MonophasicPulse(0, interpulse_dur, electrode=electrode)
            pulse = pulse.append(delay_pulse)
        # Create the pulse triplet:
        triplet = pulse.append(pulse).append(pulse)
        # Create the triplet train:
        pt = PulseTrain(freq, triplet, n_pulses=n_pulses, stim_dur=stim_dur)
        # Set up the Stimulus object through the constructor:
        super().__init__(pt.data, time=pt.time, electrodes=electrode,
                         compress=False)
        self.freq = freq
        self.cathodic_first = cathodic_first
        self.metadata = {'user': metadata}

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super()._pprint_params()
        params.update({'cathodic_first': self.cathodic_first,
                       'freq': self.freq})
        return params
