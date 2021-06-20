"""`MonophasicPulse`, `BiphasicPulse`, `AsymmetricBiphasicPulse`"""
import numpy as np

# DT: Sampling time step (ms); defines the duration of the signal edge
# transitions:
from .base import Stimulus
from ..utils.constants import DT


class MonophasicPulse(Stimulus):
    """Monophasic pulse

    A simple stimulus consisting of a single monophasic pulse (either
    cathodic/negative or anodic/positive).

    .. versionadded:: 0.6

    Parameters
    ----------
    amp : float
        Current amplitude (uA). Negative currents: cathodic, positive: anodic.
    phase_dur : float
        Duration (ms) of the cathodic or anodic phase.
    delay_dur : float
        Delay duration (ms). Zeros will be inserted at the beginning of the
        stimulus to deliver the pulse after ``delay_dur`` ms.
    stim_dur : float, optional
        Total stimulus duration (ms). Zeros will be inserted at the end of the
        stimulus to make the stimulus last ``stim_dur`` ms overall.
    electrode : { int | string }, optional
        Optionally, you can provide your own electrode name.

    Notes
    -----
    *  The sign of ``amp`` will determine whether the pulse is cathodic
       (negative current) or anodic (positive current).
    *  A regular monophasic pulse is not considered "charge-balanced". However,
       if ``amp`` is small enough, the pulse can be considered
       "charge-balanced" if its net current is smaller than 10 picoamps.

    Examples
    --------
    A single cathodic pulse (1ms phase duration at 20uA) delivered after
    2ms and embedded in a stimulus that lasts 10ms overall:

    >>> from pulse2percept.stimuli import MonophasicPulse
    >>> pulse = MonophasicPulse(-20, 1, delay_dur=2, stim_dur=10)

    """

    def __init__(self, amp, phase_dur, delay_dur=0, stim_dur=None,
                 electrode=None):
        if phase_dur <= DT:
            raise ValueError("'phase_dur' must be greater than DT=%ems." % DT)
        if delay_dur < 0:
            raise ValueError("'delay_dur' cannot be negative.")
        # The minimum stimulus duration is given by the pulse, IPG, and delay:
        min_dur = phase_dur + delay_dur
        if stim_dur is None:
            stim_dur = min_dur
        else:
            if stim_dur < min_dur:
                raise ValueError("'stim_dur' must be at least %.3f ms, not "
                                 "%.3f ms." % (min_dur, stim_dur))
        # We only need to store the time points at which the stimulus changes.
        time = [0]
        data = [0]
        if delay_dur > DT:
            time += [delay_dur]
            data += [0]
        # The mono-phase has data[t=delay_dur] = 0, then rises to amp in DT
        # and is back to zero at t=delya_dur+phase_dur:
        time += [delay_dur + DT, delay_dur + phase_dur - DT,
                 delay_dur + phase_dur]
        data += [amp, amp, 0]
        if stim_dur - time[-1] > DT:
            # If the stimulus extends beyond the second pulse, add another data
            # point:
            time += [stim_dur]
            data += [0]
        else:
            # But, if the end point is close enough to `stim_dur`, update the
            # last time point so that the stimulus is exactly `stim_dur` long:
            time[-1] = stim_dur
        data = np.array(data, dtype=np.float32).reshape((1, -1))
        time = np.array(time, dtype=np.float32)
        super().__init__(data, electrodes=electrode, time=time, compress=False)
        self.cathodic = amp <= 0

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super(MonophasicPulse, self)._pprint_params()
        params.update({'cathodic': self.cathodic})
        return params


class BiphasicPulse(Stimulus):
    """Symmetric biphasic pulse

    A simple stimulus consisting of a single biphasic pulse: a cathodic and an
    anodic phase, optionally separated by an interphase gap.
    Both cathodic and anodic phases have the same duration ("symmetric").

    .. versionadded:: 0.6

    Parameters
    ----------
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
    stim_dur : float, optional, default:
               ``2*phase_dur+interphase_dur+delay_dur``
        Total stimulus duration (ms). Zeros will be inserted at the end of the
        stimulus to make the stimulus last ``stim_dur`` ms overall.
    cathodic_first : bool, optional, default: True
        If True, will deliver the cathodic pulse phase before the anodic one.
    electrode : { int | string }, optional, default: 0
        Optionally, you can provide your own electrode name.

    Notes
    -----
    *  The order of the two phases is given by the ``cathodic_first`` flag.
    *  A biphasic pulse created with this class will always be considered
       "charge-balanced".

    Examples
    --------
    A cathodic-first pulse (1ms phase duration at 20uA, no interphase gap)
    delivered after 2ms and embedded in a stimulus that lasts 10ms overall:

    >>> from pulse2percept.stimuli import BiphasicPulse
    >>> pulse = BiphasicPulse(-20, 1, delay_dur=2, stim_dur=10)

    """

    def __init__(self, amp, phase_dur, interphase_dur=0, delay_dur=0,
                 stim_dur=None, cathodic_first=True, electrode=None):
        if phase_dur <= DT:
            raise ValueError("'phase_dur' must be greater than DT=%ems." % DT)
        if interphase_dur < 0:
            raise ValueError("'interphase_dur' cannot be negative.")
        if delay_dur < 0:
            raise ValueError("'delay_dur' cannot be negative.")
        # The minimum stimulus duration is given by the pulse, IPG, and delay:
        min_dur = 2 * phase_dur + interphase_dur + delay_dur
        if stim_dur is None:
            stim_dur = min_dur
        else:
            if stim_dur < min_dur:
                raise ValueError("'stim_dur' must be at least %.3f ms, not "
                                 "%.3f ms." % (min_dur, stim_dur))
        amp = -np.abs(amp) if cathodic_first else np.abs(amp)
        # We only need to store the time points at which the stimulus changes.
        time = [0]
        data = [0]
        if delay_dur > DT:
            time += [delay_dur]
            data += [0]
        # The first phase has data[t=delay_dur] = 0, then rises to amp in DT
        # and is back to zero at t=delya_dur+phase_dur:
        time += [delay_dur + DT, delay_dur + phase_dur - DT,
                 delay_dur + phase_dur]
        data += [amp, amp, 0]
        if interphase_dur > 0:
            time += [delay_dur + phase_dur + interphase_dur]
            data += [0]
        time += [delay_dur + phase_dur + interphase_dur + DT,
                 delay_dur + 2 * phase_dur + interphase_dur - DT,
                 delay_dur + 2 * phase_dur + interphase_dur]
        data += [-amp, -amp, 0]
        if stim_dur - time[-1] > DT:
            # If the stimulus extends beyond the second pulse, add another data
            # point:
            time += [stim_dur]
            data += [0]
        else:
            # But, if the end point is close enough to `stim_dur`, update the
            # last time point so that the stimulus is exactly `stim_dur` long:
            time[-1] = stim_dur
        data = np.array(data, dtype=np.float32).reshape((1, -1))
        time = np.array(time, dtype=np.float32)
        super().__init__(data, electrodes=electrode, time=time, compress=False)
        self.cathodic_first = cathodic_first

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super(BiphasicPulse, self)._pprint_params()
        params.update({'cathodic_first': self.cathodic_first})
        return params


class AsymmetricBiphasicPulse(Stimulus):
    """Asymmetric biphasic pulse

    A simple stimulus consisting of a single biphasic pulse: a cathodic and an
    anodic phase, optionally separated by an interphase gap.
    The two pulse phases can have different amplitudes and duration
    ("asymmetric").

    .. versionadded:: 0.6

    Parameters
    ----------
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
    stim_dur : float, optional, default:
               ``2*phase_dur+interphase_dur+delay_dur``
        Total stimulus duration (ms). Zeros will be inserted at the end of the
        stimulus to make the stimulus last ``stim_dur`` ms overall.
    cathodic_first : bool, optional, default: True
        If True, will deliver the cathodic pulse phase before the anodic one.
    electrode : { int | string }, optional, default: 0
        Optionally, you can provide your own electrode name.

    Notes
    -----
    *  The order of the two phases is given by the ``cathodic_first`` flag.
    *  The sign of ``amp`` will be automatically adjusted depending on the
       ``cathodic_first`` flag.
    *  A pulse will be considered "charge-balanced" if its net current is
       smaller than 10 picoamps.

    Examples
    --------
    An asymmetric cathodic-first pulse (first phase: -40uA, 1ms; second phase:
    10uA, 4ms; 1ms interphase-gap) delivered after 2ms and embedded in a
    stimulus that lasts 15ms overall:

    >>> from pulse2percept.stimuli import AsymmetricBiphasicPulse
    >>> pulse = AsymmetricBiphasicPulse(-40, 10, 1, 4, interphase_dur=1,
    ...                                 delay_dur=2, stim_dur=15)

    """

    def __init__(self, amp1, amp2, phase_dur1, phase_dur2, interphase_dur=0,
                 delay_dur=0, stim_dur=None, cathodic_first=True,
                 electrode=None):
        if phase_dur1 <= 0:
            raise ValueError("'phase_dur1' must be greater than 0.")
        if phase_dur2 <= 0:
            raise ValueError("'phase_dur1' must be greater than 0.")
        if interphase_dur < 0:
            raise ValueError("'interphase_dur' cannot be negative.")
        if delay_dur < 0:
            raise ValueError("'delay_dur' cannot be negative.")
        # The minimum stimulus duration is given by the pulse, IPG, and delay:
        min_dur = phase_dur1 + phase_dur2 + interphase_dur + delay_dur
        if stim_dur is None:
            stim_dur = min_dur
        else:
            if stim_dur < min_dur:
                raise ValueError("'stim_dur' must be at least %.3f ms, not "
                                 "%.3f ms." % (min_dur, stim_dur))
        if cathodic_first:
            amp1 = -np.abs(amp1)
            amp2 = np.abs(amp2)
        else:
            amp1 = np.abs(amp1)
            amp2 = -np.abs(amp2)
        # We only need to store the time points at which the stimulus changes.
        time = [0]
        data = [0]
        if delay_dur > DT:
            time += [delay_dur]
            data += [0]
        # The first phase has data[t=delay_dur] = 0, then rises to amp in DT
        # and is back to zero at t=delya_dur+phase_dur:
        time += [delay_dur + DT, delay_dur + phase_dur1 - DT,
                 delay_dur + phase_dur1]
        data += [amp1, amp1, 0]
        if interphase_dur > 0:
            time += [delay_dur + phase_dur1 + interphase_dur]
            data += [0]
        time += [delay_dur + phase_dur1 + interphase_dur + DT,
                 delay_dur + phase_dur1 + interphase_dur + phase_dur2 - DT,
                 delay_dur + phase_dur1 + interphase_dur + phase_dur2]
        data += [amp2, amp2, 0]
        if stim_dur - time[-1] > DT:
            # If the stimulus extends beyond the second pulse, add another data
            # point:
            time += [stim_dur]
            data += [0]
        else:
            # But, if the end point is close enough to `stim_dur`, update the
            # last time point so that the stimulus is exactly `stim_dur` long:
            time[-1] = stim_dur
        data = np.array(data, dtype=np.float32).reshape((1, -1))
        time = np.array(time, dtype=np.float32)
        super().__init__(data, electrodes=electrode, time=time, compress=False)
        self.cathodic_first = cathodic_first

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super(AsymmetricBiphasicPulse, self)._pprint_params()
        params.update({'cathodic_first': self.cathodic_first})
        return params
