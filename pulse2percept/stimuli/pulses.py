"""`MonophasicPulse`, `BiphasicPulse`, `AsymmetricBiphasicPulse`"""
import numpy as np

from . import MIN_AMP
from .base import Stimulus


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
    stim_dur : float, optional, default: ``phase_dur+delay_dur``
        Total stimulus duration (ms). Zeros will be inserted at the end of the
        stimulus to make the stimulus last ``stim_dur`` ms overall.
    electrode : { int | string }, optional, default: 0
        Optionally, you can provide your own electrode name.
    dt : float, optional, default: 1e-6 ms
        Sampling time step (ms); defines the duration of the signal edge
        transitions.

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
                 electrode=None, dt=1e-6):
        if phase_dur <= 0:
            raise ValueError("'phase_dur' must be greater than 0.")
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
        # For example, the pulse amplitude is zero at time=`delay_dur` and
        # `amp` at time=`delay_dur+dt`:
        data = np.array([0, 0, amp, amp, 0, 0]).reshape((1, -1))
        time = np.array([0, delay_dur,
                         delay_dur + dt, delay_dur + phase_dur - dt,
                         delay_dur + phase_dur, stim_dur])

        # There is an edge case for delay_dur=0: There will be two identical
        # `time` entries, which messes with the SciPy interpolation function.
        # Thus retain only the unique time points:
        time, idx = np.unique(time, return_index=True)
        data = data[:, idx]
        super().__init__(data, electrodes=electrode, time=time, compress=False)
        self.cathodic = amp <= 0
        self.charge_balanced = np.isclose(np.trapz(data, time)[0], 0,
                                          atol=MIN_AMP)

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super(MonophasicPulse, self)._pprint_params()
        params.update({'cathodic': self.cathodic,
                       'charge_balanced': self.charge_balanced})
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
    dt : float, optional, default: 1e-6 ms
        Sampling time step (ms); defines the duration of the signal edge
        transitions.

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
                 stim_dur=None, cathodic_first=True, electrode=None, dt=1e-6):
        if phase_dur <= 0:
            raise ValueError("'phase_dur' must be greater than 0.")
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
        # We only need to store the time points at which the stimulus changes.
        amp = -np.abs(amp) if cathodic_first else np.abs(amp)
        data = [0, 0, amp, amp, 0, 0, -amp, -amp, 0, 0]
        time = [0, delay_dur,
                delay_dur + dt, delay_dur + phase_dur - dt,
                delay_dur + phase_dur,
                delay_dur + phase_dur + interphase_dur,
                delay_dur + phase_dur + interphase_dur + dt,
                delay_dur + 2 * phase_dur + interphase_dur - dt,
                delay_dur + 2 * phase_dur + interphase_dur,
                stim_dur]
        data = np.array(data).reshape((1, -1))
        time = np.array(time)
        # There is an edge case for delay_dur=0: There will be two identical
        # `time` entries, which messes with the SciPy interpolation function.
        # Thus retain only the unique time points:
        time, idx = np.unique(time, return_index=True)
        data = data[:, idx]
        super().__init__(data, electrodes=electrode, time=time, compress=False)
        self.cathodic_first = cathodic_first
        self.charge_balanced = np.isclose(np.trapz(data, time)[0], 0,
                                          atol=MIN_AMP)

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super(BiphasicPulse, self)._pprint_params()
        params.update({'cathodic_first': self.cathodic_first,
                       'charge_balanced': self.charge_balanced})
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
    dt : float, optional, default: 1e-6 ms
        Sampling time step (ms); defines the duration of the signal edge
        transitions.

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
                 electrode=None, dt=1e-6):
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
        # We only need to store the time points at which the stimulus changes.
        if cathodic_first:
            amp1 = -np.abs(amp1)
            amp2 = np.abs(amp2)
        else:
            amp1 = np.abs(amp1)
            amp2 = -np.abs(amp2)
        data = [0, 0, amp1, amp1, 0, 0, amp2, amp2, 0, 0]
        time = [0, delay_dur,
                delay_dur + dt, delay_dur + phase_dur1 - dt,
                delay_dur + phase_dur1,
                delay_dur + phase_dur1 + interphase_dur,
                delay_dur + phase_dur1 + interphase_dur + dt,
                delay_dur + phase_dur1 + interphase_dur + phase_dur2 - dt,
                delay_dur + phase_dur1 + interphase_dur + phase_dur2,
                stim_dur]
        data = np.array(data).reshape((1, -1))
        time = np.array(time)
        # There is an edge case for delay_dur=0: There will be two identical
        # `time` entries, which messes with the SciPy interpolation function.
        # Thus retain only the unique time points:
        time, idx = np.unique(time, return_index=True)
        data = data[:, idx]
        super().__init__(data, electrodes=electrode, time=time, compress=False)
        self.cathodic_first = cathodic_first
        self.charge_balanced = np.isclose(np.trapz(data, time)[0], 0,
                                          atol=MIN_AMP)

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super(AsymmetricBiphasicPulse, self)._pprint_params()
        params.update({'cathodic_first': self.cathodic_first,
                       'charge_balanced': self.charge_balanced})
        return params
