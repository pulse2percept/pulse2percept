"""`PulseTrain`, `BiphasicPulseTrain`, `AsymmetricBiphasicPulseTrain`"""
import numpy as np
import copy
import logging

from . import MIN_AMP
from .base import Stimulus
from .pulses import BiphasicPulse, AsymmetricBiphasicPulse


class PulseTrain(Stimulus):
    """Generic pulse train

    .. versionadded:: 0.6

    Can be used to concatenate single pulses into a pulse train.

    .. seealso ::

        * :py:class:`~pulse2percept.stimuli.BiphasicPulseTrain`
        * :py:class:`~pulse2percept.stimuli.AsymmetricBiphasicPulseTrain`

    Parameters
    ----------
    freq : float
        Pulse train frequency (Hz).
    pulse : :py:class:`~pulse2percept.stimuli.Stimulus`
        A Stimulus object containing a single pulse that will be concatenated.
    n_pulses : int
        Number of pulses requested in the pulse train. If None, the entire
        stimulation window (``stim_dur``) is filled.
    stim_dur : float, optional, default: 1000 ms
        Total stimulus duration (ms). The pulse train will be trimmed to make
        the stimulus last ``stim_dur`` ms overall.
    electrode : { int | string }, optional, default: 0
        Optionally, you can provide your own electrode name.
    dt : float, optional, default: 1e-3 ms
        Sampling time step (ms); defines the duration of the signal edge
        transitions.
    metadata : dict
        A dictionary of meta-data

    Notes
    -----
    *  If the pulse train frequency does not exactly divide ``stim_dur``, the
       number of pulses will be rounded down. For example, when trying to fit
       a 11 Hz pulse train into a 100 ms window, there will be 9 pulses.

    """

    def __init__(self, freq, pulse, n_pulses=None, stim_dur=1000.0, dt=1e-3,
                 electrode=None, metadata=None):
        if not isinstance(pulse, Stimulus):
            raise TypeError("'pulse' must be a Stimulus object, not "
                            "%s." % type(pulse))
        if pulse.shape[0] == 0:
            raise ValueError("'pulse' has invalid shape "
                             "(%d, %d)." % (pulse.shape[0], pulse.shape[1]))
        if pulse.time is None:
            raise ValueError("'pulse' does not have a time component.")
        # How many pulses fit into stim dur:
        n_max_pulses = int(freq * stim_dur / 1000.0)
        if n_pulses is not None:
            n_pulses = int(n_pulses)
            if n_pulses > n_max_pulses:
                raise ValueError("stim_dur=%.2f cannot fit more than "
                                 "%d pulses." % (stim_dur, n_max_pulses))
        else:
            n_pulses = n_max_pulses
        # 0 Hz is allowed:
        if n_pulses <= 0:
            time = [0, stim_dur]
            data = [[0, 0]]
        else:
            # Window duration is the inverse of pulse train frequency:
            window_dur = 1000.0 / freq
            if pulse.time[-1] > window_dur:
                raise ValueError("Pulse (dur=%.2f ms) does not fit into "
                                 "pulse train window (dur=%.2f "
                                 "ms)" % (pulse.time[-1], window_dur))
            pulse_data = pulse.data
            pulse_time = pulse.time
            # We have to be careful not to create duplicate time points, which
            # will be trimmed (and produce artifacts) upon compression:
            if np.isclose(pulse_time[-1], window_dur):
                pulse_time[-1] -= dt
            # Concatenate the pulses:
            data = []
            time = []
            for i in range(n_pulses):
                data.append(pulse_data)
                time.append(pulse_time + i * window_dur)
            # Make sure the last point in the stimulus is at `stim_dur`:
            if time[-1][-1] < stim_dur:
                data.append(np.zeros((pulse.data.shape[0], 1)))
                time.append([stim_dur])
            data = np.concatenate(data, axis=1)
            time = np.concatenate(time, axis=0)
        super().__init__(data, time=time, electrodes=electrode, metadata=None,
                         compress=False)
        self.freq = freq
        self.pulse_type = pulse.__class__.__name__
        self.charge_balanced = np.isclose(np.trapz(data, time)[0], 0,
                                          atol=MIN_AMP)

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super(PulseTrain, self)._pprint_params()
        params.update({'freq': self.freq,
                       'pulse_type': self.pulse_type,
                       'charge_balanced': self.charge_balanced})
        return params


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
    dt : float, optional, default: 1e-3 ms
        Sampling time step (ms); defines the duration of the signal edge
        transitions.
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

    def __init__(self, freq, amp, phase_dur, interphase_dur=0, delay_dur=0,
                 n_pulses=None, stim_dur=1000.0, cathodic_first=True, dt=1e-3,
                 electrode=None, metadata=None):
        # Create the individual pulse:
        pulse = BiphasicPulse(amp, phase_dur, delay_dur=delay_dur, dt=dt,
                              interphase_dur=interphase_dur,
                              cathodic_first=cathodic_first,
                              electrode=electrode)
        # Concatenate the pulses:
        pt = PulseTrain(freq, pulse, n_pulses=n_pulses, stim_dur=stim_dur,
                        dt=dt, metadata=metadata)
        super().__init__(pt.data, time=pt.time, compress=False)
        self.freq = freq
        self.cathodic_first = cathodic_first
        self.charge_balanced = pt.charge_balanced

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
    dt : float, optional, default: 1e-3 ms
        Sampling time step (ms); defines the duration of the signal edge
        transitions.
    metadata : dict
        A dictionary of meta-data

    """

    def __init__(self, freq, amp1, amp2, phase_dur1, phase_dur2,
                 interphase_dur=0, delay_dur=0, n_pulses=None, stim_dur=1000.0,
                 cathodic_first=True, dt=1e-3, electrode=None, metadata=None):
        # Create the individual pulse:
        pulse = AsymmetricBiphasicPulse(amp1, amp2, phase_dur1, phase_dur2,
                                        delay_dur=delay_dur, dt=dt,
                                        interphase_dur=interphase_dur,
                                        cathodic_first=cathodic_first,
                                        electrode=electrode)
        # Concatenate the pulses:
        pt = PulseTrain(freq, pulse, n_pulses=n_pulses, stim_dur=stim_dur,
                        dt=dt, metadata=metadata)
        super().__init__(pt.data, time=pt.time, compress=False)
        self.freq = freq
        self.cathodic_first = cathodic_first
        self.charge_balanced = pt.charge_balanced

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super(AsymmetricBiphasicPulseTrain, self)._pprint_params()
        params.update({'cathodic_first': self.cathodic_first,
                       'charge_balanced': self.charge_balanced,
                       'freq': self.freq})
        return params


class BiphasicTripletTrain(Stimulus):
    """Biphasic pulse triplets

    .. versionadded:: 0.6

    A train of symmetric biphasic pulse triplets.

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
    dt : float, optional, default: 1e-3 ms
        Sampling time step (ms); defines the duration of the signal edge
        transitions.
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

    def __init__(self, freq, amp, phase_dur, interphase_dur=0, delay_dur=0,
                 n_pulses=None, stim_dur=1000.0, cathodic_first=True, dt=1e-3,
                 electrode=None, metadata=None):
        # Create the pulse:
        pulse = BiphasicPulse(amp, phase_dur, interphase_dur=interphase_dur,
                              delay_dur=delay_dur, dt=dt,
                              cathodic_first=cathodic_first,
                              electrode=electrode)
        # Create the pulse triplet:
        triplet_dur = 3 * (2 * phase_dur + interphase_dur + delay_dur + dt)
        triplet = PulseTrain(3 * 1000.0 / triplet_dur, pulse, n_pulses=3,
                             stim_dur=triplet_dur)
        # Create the triplet train:
        pt = PulseTrain(freq, triplet, n_pulses=n_pulses, stim_dur=stim_dur)
        # Set up the Stimulus object through the constructor:
        super().__init__(pt.data, time=pt.time, compress=False)
        self.freq = freq
        self.cathodic_first = cathodic_first
        self.charge_balanced = pt.charge_balanced

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super(BiphasicPulseTrain, self)._pprint_params()
        params.update({'cathodic_first': self.cathodic_first,
                       'charge_balanced': self.charge_balanced,
                       'freq': self.freq})
        return params
