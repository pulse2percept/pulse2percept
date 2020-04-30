"""`TimeSeries`, `MonophasicPulse`, `BiphasicPulse`, `PulseTrain`"""
import numpy as np
import copy
import logging
from scipy.interpolate import interp1d


from . import MIN_AMP
from .base import TimeSeries, Stimulus
from .pulses import BiphasicPulse, AsymmetricBiphasicPulse, LegacyBiphasicPulse
from ..utils import deprecated


class BiphasicPulseTrain(Stimulus):
    """Symmetric biphasic pulse train

    .. versionadded:: 0.6

    A train of symmetric biphasic pulses.

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
    stim_dur : float, optional, default: 1000 ms
        Total stimulus duration (ms). The pulse train will be trimmed to make
        the stimulus last ``stim_dur`` ms overall.
    cathodic_first : bool, optional, default: True
        If True, will deliver the cathodic pulse phase before the anodic one.
    dt : float, optional, default: 1e-3 ms
        Sampling time step (ms); defines the duration of the signal edge
        transitions.

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
    *  A pulse train will be trimmed to the right length given by ``stim_dur``.
       Consequently, it's possible to cut off parts of a pulse and thus create
       a pulse train that is not charge-balanced.

    """

    def __init__(self, freq, amp, phase_dur, interphase_dur=0, delay_dur=0,
                 stim_dur=1000.0, cathodic_first=True, dt=1e-3):
        # 0 Hz is allowed:
        if np.isclose(freq, 0):
            time = [0, stim_dur]
            data = [[0, 0]]
        else:
            # Window duration is the inverse of pulse train frequency. If it is
            # too small for phase_dur/interphase_dur/delay_dur (or negative),
            # the BiphasicPulse constructor will catch it:
            window_dur = 1000.0 / freq
            pulse = BiphasicPulse(amp, phase_dur, delay_dur=delay_dur, dt=dt,
                                  interphase_dur=interphase_dur,
                                  cathodic_first=cathodic_first,
                                  stim_dur=window_dur)
            # How many pulses depends on stim_dur:
            n_pulses = int(np.ceil(freq * stim_dur / 1000.0))
            data = []
            time = []
            for i in range(n_pulses):
                d_win = pulse.data
                t_win = pulse.time + i * window_dur
                if t_win[-1] > stim_dur:
                    # Interpolate the data point at t=stim_dur:
                    last_d = interp1d(t_win, d_win)(stim_dur)[0]
                    # Keep only the data points < stim_dur, but append the
                    # interpolated point:
                    idx = t_win <= stim_dur
                    d_win = np.append(d_win[:, idx], [[last_d]], axis=1)
                    t_win = np.append(t_win[idx], stim_dur)
                    data.append(d_win)
                    time.append(t_win)
                else:
                    data.append(d_win)
                    time.append(t_win)
            data = np.concatenate(data, axis=1)
            time = np.concatenate(time, axis=0)
            # There is an edge case for delay_dur=0: There will be two
            # identical `time` entries, which messes with the SciPy
            # interpolation function.
            # Thus retain only the unique time points:
            time, idx = np.unique(time, return_index=True)
            data = data[:, idx]
        super().__init__(data, time=time, compress=False)
        self.freq = freq
        self.cathodic_first = cathodic_first
        self.charge_balanced = np.isclose(np.trapz(data, time)[0], 0,
                                          atol=MIN_AMP)

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super(BiphasicPulseTrain, self)._pprint_params()
        params.update({'cathodic_first': self.cathodic_first,
                       'charge_balanced': self.charge_balanced,
                       'freq': self.freq})
        return params


class AsymmetricBiphasicPulseTrain(Stimulus):
    """Asymmetric biphasic pulse

    .. versionadded:: 0.6

    A simple stimulus consisting of a single biphasic pulse: a cathodic and an
    anodic phase, optionally separated by an interphase gap.
    The two pulse phases can have different amplitudes and duration
    ("asymmetric").
    The order of the two phases is given by the ``cathodic_first`` flag.

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
    stim_dur : float, optional, default: 1000 ms
        Total stimulus duration (ms). Zeros will be inserted at the end of the
        stimulus to make the the stimulus last ``stim_dur`` ms overall.
    cathodic_first : bool, optional, default: True
        If True, will deliver the cathodic pulse phase before the anodic one.
    dt : float, optional, default: 1e-3 ms
        Sampling time step (ms); defines the duration of the signal edge
        transitions.

    """

    def __init__(self, freq, amp1, amp2, phase_dur1, phase_dur2,
                 interphase_dur=0, delay_dur=0, stim_dur=1000.0,
                 cathodic_first=True, dt=1e-3):
        # 0 Hz is allowed:
        if np.isclose(freq, 0):
            time = [0, stim_dur]
            data = [[0, 0]]
        else:
            # Window duration is 1/f:
            window_dur = 1000.0 / freq
            pulse = AsymmetricBiphasicPulse(amp1, amp2, phase_dur1, phase_dur2,
                                            delay_dur=delay_dur, dt=dt,
                                            interphase_dur=interphase_dur,
                                            cathodic_first=cathodic_first,
                                            stim_dur=window_dur)
            # How many pulses depends on stim_dur:
            n_pulses = int(np.ceil(freq * stim_dur / 1000.0))
            data = []
            time = []
            for i in range(n_pulses):
                d_win = pulse.data
                t_win = pulse.time + i * window_dur
                if t_win[-1] > stim_dur:
                    # Interpolate the data point at t=stim_dur:
                    last_d = interp1d(t_win, d_win)(stim_dur)[0]
                    # Keep only the data points < stim_dur, but append the
                    # interpolated point:
                    idx = t_win <= stim_dur
                    d_win = np.append(d_win[:, idx], [[last_d]], axis=1)
                    t_win = np.append(t_win[idx], stim_dur)
                    data.append(d_win)
                    time.append(t_win)
                else:
                    data.append(d_win)
                    time.append(t_win)
            data = np.concatenate(data, axis=1)
            time = np.concatenate(time, axis=0)
            # There is an edge case for delay_dur=0: There will be two
            # identical `time` entries, which messes with the SciPy
            # interpolation function.
            # Thus retain only the unique time points:
            time, idx = np.unique(time, return_index=True)
            data = data[:, idx]
        super().__init__(data, time=time, compress=False)
        self.cathodic_first = cathodic_first
        self.charge_balanced = np.isclose(np.trapz(data, time)[0], 0)
        self.freq = freq

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super(AsymmetricBiphasicPulseTrain, self)._pprint_params()
        params.update({'cathodic_first': self.cathodic_first,
                       'charge_balanced': self.charge_balanced,
                       'freq': self.freq})
        return params


@deprecated(alt_func='BiphasicPulseTrain', deprecated_version='0.6',
            removed_version='0.7')
class PulseTrain(TimeSeries):
    """A train of biphasic pulses

    Parameters
    ----------
    tsample: float
        Sampling time step(milliseconds).
    freq: float, optional, default: 20 Hz
        Frequency of the pulse envelope(Hz).
    amp: float, optional, default: 20 uA
        Max amplitude of the pulse train in micro - amps.
    dur: float, optional, default: 0.5 seconds
        Stimulus duration in milliseconds.
    delay: float, optional, default: 0
        Delay until stimulus on - set in milliseconds.
    pulse_dur: float, optional, default: 0.45 ms
        Single - pulse duration in milliseconds.
    interphase_duration: float, optional, default: 0.45 ms
        Single - pulse interphase duration(the time between the positive
        and negative phase) in milliseconds.
    pulsetype: str, optional, default: 'cathodicfirst'
        Pulse type {'cathodicfirst' | 'anodicfirst'}, where
        'cathodicfirst' has the negative phase first.
    pulseorder: str, optional, default: 'pulsefirst'
        Pulse order {'gapfirst' | 'pulsefirst'}, where
        'pulsefirst' has the pulse first, followed by the gap.
        'gapfirst' has it the other way round.
    """
    __slots__ = ()

    def __init__(self, tsample, freq=20, amp=20, dur=0.5, delay=0,
                 pulse_dur=0.45 / 1000, interphase_dur=0.45 / 1000,
                 pulsetype='cathodicfirst',
                 pulseorder='pulsefirst'):
        if tsample <= 0:
            raise ValueError("tsample must be a non-negative float.")

        # Stimulus size given by `dur`
        stim_size = int(np.round(float(dur) / tsample))

        # Make sure input is non-trivial, else return all zeros
        if np.isclose(freq, 0) or np.isclose(amp, 0):
            TimeSeries.__init__(self, tsample, np.zeros(stim_size))
            return

        # Envelope size (single pulse + gap) given by `freq`
        # Note that this can be larger than `stim_size`, but we will trim
        # the stimulus to proper length at the very end.
        envelope_size = int(np.round(1.0 / float(freq) / tsample))
        if envelope_size > stim_size:
            debug_s = ("Envelope size (%d) clipped to "
                       "stimulus size (%d) for freq=%f" % (envelope_size,
                                                           stim_size,
                                                           freq))
            logging.getLogger(__name__).debug(debug_s)
            envelope_size = stim_size

        # Delay given by `delay`
        delay_size = int(np.round(float(delay) / tsample))

        if delay_size < 0:
            raise ValueError("Delay cannot be negative.")
        delay = np.zeros(delay_size)

        # Single pulse given by `pulse_dur`
        pulse = amp * LegacyBiphasicPulse(pulsetype, pulse_dur, tsample,
                                          interphase_dur).data
        pulse_size = pulse.size
        if pulse_size < 0:
            raise ValueError("Single pulse must fit within 1/freq interval.")

        # Then gap is used to fill up what's left
        gap_size = envelope_size - (delay_size + pulse_size)
        if gap_size < 0:
            logging.error("Envelope (%d) can't fit pulse (%d) + delay (%d)" %
                          (envelope_size, pulse_size, delay_size))
            raise ValueError("Pulse and delay must fit within 1/freq "
                             "interval.")
        gap = np.zeros(gap_size)

        pulse_train = np.array([])
        for j in range(int(np.ceil(dur * freq))):
            if pulseorder == 'pulsefirst':
                pulse_train = np.concatenate((pulse_train, delay, pulse,
                                              gap), axis=0)
            elif pulseorder == 'gapfirst':
                pulse_train = np.concatenate((pulse_train, delay, gap,
                                              pulse), axis=0)
            else:
                raise ValueError("Acceptable values for `pulseorder` are "
                                 "'pulsefirst' or 'gapfirst'")

        # If `freq` is not a nice number, the resulting pulse train might not
        # have the desired length
        if pulse_train.size < stim_size:
            fill_size = stim_size - pulse_train.shape[-1]
            pulse_train = np.concatenate((pulse_train, np.zeros(fill_size)),
                                         axis=0)

        # Trim to correct length (takes care of too long arrays, too)
        pulse_train = pulse_train[:stim_size]

        super(PulseTrain, self).__init__(tsample, pulse_train)
