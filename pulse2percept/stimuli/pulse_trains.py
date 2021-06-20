"""`PulseTrain`, `BiphasicPulseTrain`, `AsymmetricBiphasicPulseTrain`"""
import numpy as np
import logging

# DT: Sampling time step (ms); defines the duration of the signal edge
# transitions:
from .base import Stimulus
from .pulses import BiphasicPulse, AsymmetricBiphasicPulse, MonophasicPulse
from ..utils.constants import DT


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
    *  If the pulse train frequency does not exactly divide ``stim_dur``, the
       number of pulses will be rounded down. For example, when trying to fit
       a 11 Hz pulse train into a 100 ms window, there will be 9 pulses.

    """

    def __init__(self, freq, pulse, n_pulses=None, stim_dur=1000.0,
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
        n_max_pulses = freq * stim_dur / 1000.0
        # The requested number of pulses cannot be greater than max pulses:
        if n_pulses is not None:
            n_pulses = int(n_pulses)
            if n_pulses > n_max_pulses:
                raise ValueError("stim_dur=%.2f cannot fit more than "
                                 "%d pulses." % (stim_dur, n_max_pulses))
        else:
            # `freq` might not perfectly divide `stim_dur`, so we will create
            # one extra pulse and trim to the right length:
            n_pulses = int(np.ceil(n_max_pulses))
        # 0 Hz is allowed:
        if n_pulses <= 0:
            time = np.array([0, stim_dur], dtype=np.float32)
            data = np.array([[0, 0]], dtype=np.float32)
        else:
            # Window duration is the inverse of pulse train frequency:
            window_dur = 1000.0 / freq
            if pulse.time[-1] > window_dur:
                raise ValueError("Pulse (dur=%.2f ms) does not fit into "
                                 "pulse train window (dur=%.2f "
                                 "ms)" % (pulse.time[-1], window_dur))
            shift = np.maximum(0, window_dur - pulse.time[-1])
            pt = pulse
            for i in range(1, n_pulses):
                pt = pt.append(pulse >> shift)
            data = pt.data
            time = pt.time
        if time[-1] > stim_dur + DT:
            # If stimulus is longer than the requested `stim_dur`, trim it.
            # Make sure to interpolate the end point:
            last_col = [np.interp(stim_dur, time, row) for row in data]
            last_col = np.array(last_col).reshape((-1, 1))
            t_idx = time < stim_dur
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
        params = super(PulseTrain, self)._pprint_params()
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
        super().__init__(pt.data, time=pt.time, compress=False)
        self.freq = freq
        self.cathodic_first = cathodic_first

        # Store metadata for BiphasicAxonMapModel
        self.metadata = {'freq': freq,
                         'amp': amp,
                         'phase_dur': phase_dur,
                         'delay_dur': delay_dur,
                         'user': metadata}

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super(BiphasicPulseTrain, self)._pprint_params()
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
        super().__init__(pt.data, time=pt.time, compress=False)
        self.freq = freq
        self.cathodic_first = cathodic_first
        self.metadata = {'user': metadata}

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super(AsymmetricBiphasicPulseTrain, self)._pprint_params()
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

    def __init__(self, freq, amp, phase_dur, interphase_dur=0, interpulse_dur=0,
                 delay_dur=0, n_pulses=None, stim_dur=1000.0, cathodic_first=True,
                 electrode=None, metadata=None):
        # Create the pulse:
        pulse = BiphasicPulse(amp, phase_dur, interphase_dur=interphase_dur,
                              delay_dur=delay_dur,
                              cathodic_first=cathodic_first,
                              electrode=electrode)
        if interpulse_dur != 0:
            # Create an interpulse 'delay' pulse:
            delay_pulse = MonophasicPulse(0, interpulse_dur)
            pulse = pulse.append(delay_pulse)
        # Create the pulse triplet:
        triplet = pulse.append(pulse).append(pulse)
        # Create the triplet train:
        pt = PulseTrain(freq, triplet, n_pulses=n_pulses, stim_dur=stim_dur)
        # Set up the Stimulus object through the constructor:
        super(BiphasicTripletTrain, self).__init__(pt.data, time=pt.time,
                                                   compress=False)
        self.freq = freq
        self.cathodic_first = cathodic_first
        self.metadata = {'user': metadata}

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super(BiphasicTripletTrain, self)._pprint_params()
        params.update({'cathodic_first': self.cathodic_first,
                       'freq': self.freq})
        return params
