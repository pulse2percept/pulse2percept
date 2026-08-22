""":py:class:`~pulse2percept.stimuli.MonophasicPulse`,
   :py:class:`~pulse2percept.stimuli.BiphasicPulse`,
   :py:class:`~pulse2percept.stimuli.AsymmetricBiphasicPulse`"""
import numpy as np

# DT: Sampling time step (ms); defines the duration of the signal edge
# transitions:
from .base import Stimulus
from ..units import as_value, ms, uA
from ..utils.constants import DT


def _electrode_names(electrode):
    """The name a single-electrode pulse is filed under

    ``Stimulus`` numbers electrodes 0..N-1 where the source did not name them,
    so an unnamed pulse is electrode 0. A pulse drives exactly one electrode,
    which is checked here rather than left to the waveform: the constructor
    is where a caller finds out that they named two.
    """
    if electrode is None:
        return [0]
    names = np.array([electrode]).ravel()
    if len(names) != 1:
        raise ValueError(f"A pulse is delivered to a single electrode, but "
                         f"{len(names)} names were given ({electrode}).")
    return names


def _pad_to_stim_dur(time, data, stim_dur):
    """Close a pulse waveform off at ``stim_dur``

    Either by adding a final zero, or -- where the pulse already ends within a
    time step of it -- by moving its last point onto it, so that the stimulus
    is exactly ``stim_dur`` long either way.
    """
    if stim_dur - time[-1] > DT:
        # If the stimulus extends beyond the second pulse, add another data
        # point:
        time += [stim_dur]
        data += [0]
    else:
        # But, if the end point is close enough to `stim_dur`, update the
        # last time point so that the stimulus is exactly `stim_dur` long:
        time[-1] = stim_dur
    return (np.array(data, dtype=np.float32).reshape((1, -1)),
            np.array(time, dtype=np.float64))


class MonophasicPulse(Stimulus):
    """Monophasic pulse

    A simple stimulus consisting of a single monophasic pulse (either
    cathodic/negative or anodic/positive).

    .. versionadded:: 0.6

    .. versionchanged:: 0.10.0
        The pulse retains the parameters that define it, and generates its
        sampled waveform only when one is asked for.

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
    *  Arguments may be given as plain numbers in the units documented above,
       or as unitful quantities (e.g. ``0.05 * mA``, ``450 * us``), which are
       converted to those units. See :py:mod:`pulse2percept.units`.
    *  The parameters above are read-only. A pulse with different parameters
       is a different pulse; build another one.

    Examples
    --------
    A single cathodic pulse (1ms phase duration at 20uA) delivered after
    2ms and embedded in a stimulus that lasts 10ms overall:

    >>> from pulse2percept.stimuli import MonophasicPulse
    >>> pulse = MonophasicPulse(-20, 1, delay_dur=2, stim_dur=10)

    """
    #: Defined by the parameters above rather than by its samples
    #: (see `Stimulus._is_parametric`):
    _is_parametric = True

    __slots__ = ('_amp', '_phase_dur', '_delay_dur', '_stim_dur')

    def __init__(self, amp, phase_dur, delay_dur=0, stim_dur=None,
                 electrode=None):
        # Strip the units first; everything below is plain numbers in uA and
        # ms, exactly as it has always been:
        amp = as_value(amp, uA, 'amp')
        phase_dur = as_value(phase_dur, ms, 'phase_dur')
        delay_dur = as_value(delay_dur, ms, 'delay_dur')
        stim_dur = as_value(stim_dur, ms, 'stim_dur')
        if phase_dur <= DT:
            raise ValueError(f"'phase_dur' must be greater than DT={DT}ms.")
        if delay_dur < 0:
            raise ValueError("'delay_dur' cannot be negative.")
        # The minimum stimulus duration is given by the pulse, IPG, and delay:
        min_dur = phase_dur + delay_dur
        if stim_dur is None:
            stim_dur = min_dur
        else:
            if stim_dur < min_dur:
                raise ValueError(f"'stim_dur' must be at least {min_dur:.3f} ms, not "
                                 f"{stim_dur:.3f} ms.")
        self._amp = amp
        self._phase_dur = phase_dur
        self._delay_dur = delay_dur
        self._stim_dur = stim_dur
        self._defer(_electrode_names(electrode))

    @property
    def amp(self):
        """Current amplitude (uA); negative is cathodic"""
        return self._amp

    @property
    def phase_dur(self):
        """Duration (ms) of the cathodic or anodic phase"""
        return self._phase_dur

    @property
    def delay_dur(self):
        """Delay (ms) before the pulse is delivered"""
        return self._delay_dur

    @property
    def stim_dur(self):
        """Total stimulus duration (ms)"""
        return self._stim_dur

    @property
    def duration(self):
        """Stimulus duration (ms)

        The waveform is built to end exactly at ``stim_dur``, so this is
        known without generating one.
        """
        return self._stim_dur

    @property
    def cathodic(self):
        """Whether the pulse delivers a negative current"""
        return self._amp <= 0

    def _render(self):
        """Build the waveform from the parameters above"""
        amp, phase_dur = self._amp, self._phase_dur
        delay_dur = self._delay_dur
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
        data, time = _pad_to_stim_dur(time, data, self._stim_dur)
        return {'data': data, 'electrodes': self.electrodes, 'time': time}

    def _scaled(self, factor):
        """This pulse with its amplitude multiplied by ``factor``

        ``amp`` carries the polarity here, so a negative factor needs nothing
        else said about it.
        """
        return MonophasicPulse(self.amp * factor, self.phase_dur,
                               delay_dur=self.delay_dur,
                               stim_dur=self.stim_dur,
                               electrode=self.electrodes[0])

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print

        The defining parameters rather than the waveform, so that printing a
        pulse does not generate one.
        """
        return {'amp': self.amp, 'phase_dur': self.phase_dur,
                'delay_dur': self.delay_dur, 'stim_dur': self.stim_dur,
                'cathodic': self.cathodic, 'electrodes': self.electrodes,
                'metadata': self.metadata}


class BiphasicPulse(Stimulus):
    """Symmetric biphasic pulse

    A simple stimulus consisting of a single biphasic pulse: a cathodic and an
    anodic phase, optionally separated by an interphase gap.
    Both cathodic and anodic phases have the same duration ("symmetric").

    .. versionadded:: 0.6

    .. versionchanged:: 0.10.0
        The pulse retains the parameters that define it, and generates its
        sampled waveform only when one is asked for.

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
    *  Arguments may be given as plain numbers in the units documented above,
       or as unitful quantities (e.g. ``0.05 * mA``, ``450 * us``), which are
       converted to those units. See :py:mod:`pulse2percept.units`.
    *  The parameters above are read-only. A pulse with different parameters
       is a different pulse; build another one.
    *  ``amp`` reads back as a magnitude, because that is all of it that
       reaches the waveform: the constructor takes ``np.abs(amp)`` and gets
       the polarity from ``cathodic_first``.

    Examples
    --------
    A cathodic-first pulse (1ms phase duration at 20uA, no interphase gap)
    delivered after 2ms and embedded in a stimulus that lasts 10ms overall:

    >>> from pulse2percept.stimuli import BiphasicPulse
    >>> pulse = BiphasicPulse(-20, 1, delay_dur=2, stim_dur=10)

    """
    #: Defined by the parameters above rather than by its samples
    #: (see `Stimulus._is_parametric`):
    _is_parametric = True

    __slots__ = ('_amp', '_phase_dur', '_interphase_dur', '_delay_dur',
                 '_stim_dur', '_cathodic_first')

    def __init__(self, amp, phase_dur, interphase_dur=0, delay_dur=0,
                 stim_dur=None, cathodic_first=True, electrode=None):
        # See `MonophasicPulse.__init__`:
        amp = as_value(amp, uA, 'amp')
        phase_dur = as_value(phase_dur, ms, 'phase_dur')
        interphase_dur = as_value(interphase_dur, ms, 'interphase_dur')
        delay_dur = as_value(delay_dur, ms, 'delay_dur')
        stim_dur = as_value(stim_dur, ms, 'stim_dur')
        if phase_dur <= DT:
            raise ValueError(f"'phase_dur' must be greater than DT={DT}ms.")
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
                raise ValueError(f"'stim_dur' must be at least {min_dur:.3f} ms, not "
                                 f"{stim_dur:.3f} ms.")
        # Only the magnitude is stored; `cathodic_first` is where the sign of
        # each phase comes from (see `_render`):
        self._amp = abs(amp)
        self._phase_dur = phase_dur
        self._interphase_dur = interphase_dur
        self._delay_dur = delay_dur
        self._stim_dur = stim_dur
        self._cathodic_first = cathodic_first
        self._defer(_electrode_names(electrode))

    @property
    def amp(self):
        """Magnitude (uA) of both pulse phases"""
        return self._amp

    @property
    def phase_dur(self):
        """Duration (ms) of the cathodic/anodic phase"""
        return self._phase_dur

    @property
    def interphase_dur(self):
        """Duration (ms) of the gap between the two phases"""
        return self._interphase_dur

    @property
    def delay_dur(self):
        """Delay (ms) before the first phase is delivered"""
        return self._delay_dur

    @property
    def stim_dur(self):
        """Total stimulus duration (ms)"""
        return self._stim_dur

    @property
    def duration(self):
        """Stimulus duration (ms)

        The waveform is built to end exactly at ``stim_dur``, so this is
        known without generating one.
        """
        return self._stim_dur

    @property
    def cathodic_first(self):
        """Whether the cathodic phase is delivered first"""
        return self._cathodic_first

    def _render(self):
        """Build the waveform from the parameters above"""
        amp = -self._amp if self._cathodic_first else self._amp
        phase_dur, interphase_dur = self._phase_dur, self._interphase_dur
        delay_dur = self._delay_dur
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
        data, time = _pad_to_stim_dur(time, data, self._stim_dur)
        return {'data': data, 'electrodes': self.electrodes, 'time': time}

    def _scaled(self, factor):
        """This pulse with both phase magnitudes multiplied by ``factor``

        Only the magnitude is stored, so a negative factor is expressed by
        swapping the two phases -- which is exactly what ``cathodic_first``
        says.
        """
        return BiphasicPulse(self.amp * abs(factor), self.phase_dur,
                             interphase_dur=self.interphase_dur,
                             delay_dur=self.delay_dur,
                             stim_dur=self.stim_dur,
                             cathodic_first=(self.cathodic_first
                                             if factor >= 0
                                             else not self.cathodic_first),
                             electrode=self.electrodes[0])

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print

        The defining parameters rather than the waveform, so that printing a
        pulse does not generate one.
        """
        return {'amp': self.amp, 'phase_dur': self.phase_dur,
                'interphase_dur': self.interphase_dur,
                'delay_dur': self.delay_dur, 'stim_dur': self.stim_dur,
                'cathodic_first': self.cathodic_first,
                'electrodes': self.electrodes, 'metadata': self.metadata}


class AsymmetricBiphasicPulse(Stimulus):
    """Asymmetric biphasic pulse

    A simple stimulus consisting of a single biphasic pulse: a cathodic and an
    anodic phase, optionally separated by an interphase gap.
    The two pulse phases can have different amplitudes and duration
    ("asymmetric").

    .. versionadded:: 0.6

    .. versionchanged:: 0.10.0
        The pulse retains the parameters that define it, and generates its
        sampled waveform only when one is asked for.

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
    *  Arguments may be given as plain numbers in the units documented above,
       or as unitful quantities (e.g. ``0.05 * mA``, ``450 * us``), which are
       converted to those units. See :py:mod:`pulse2percept.units`.
    *  The parameters above are read-only. A pulse with different parameters
       is a different pulse; build another one.
    *  ``amp1`` and ``amp2`` read back as magnitudes, because that is all of
       them that reaches the waveform: the constructor takes ``np.abs`` of
       each and gets the polarity from ``cathodic_first``.

    Examples
    --------
    An asymmetric cathodic-first pulse (first phase: -40uA, 1ms; second phase:
    10uA, 4ms; 1ms interphase-gap) delivered after 2ms and embedded in a
    stimulus that lasts 15ms overall:

    >>> from pulse2percept.stimuli import AsymmetricBiphasicPulse
    >>> pulse = AsymmetricBiphasicPulse(-40, 10, 1, 4, interphase_dur=1,
    ...                                 delay_dur=2, stim_dur=15)

    """
    #: Defined by the parameters above rather than by its samples
    #: (see `Stimulus._is_parametric`):
    _is_parametric = True

    __slots__ = ('_amp1', '_amp2', '_phase_dur1', '_phase_dur2',
                 '_interphase_dur', '_delay_dur', '_stim_dur',
                 '_cathodic_first')

    def __init__(self, amp1, amp2, phase_dur1, phase_dur2, interphase_dur=0,
                 delay_dur=0, stim_dur=None, cathodic_first=True,
                 electrode=None):
        # See `MonophasicPulse.__init__`:
        amp1 = as_value(amp1, uA, 'amp1')
        amp2 = as_value(amp2, uA, 'amp2')
        phase_dur1 = as_value(phase_dur1, ms, 'phase_dur1')
        phase_dur2 = as_value(phase_dur2, ms, 'phase_dur2')
        interphase_dur = as_value(interphase_dur, ms, 'interphase_dur')
        delay_dur = as_value(delay_dur, ms, 'delay_dur')
        stim_dur = as_value(stim_dur, ms, 'stim_dur')
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
                raise ValueError(f"'stim_dur' must be at least {min_dur:.3f} ms, not "
                                 f"{stim_dur:.3f} ms.")
        # Only the magnitudes are stored; `cathodic_first` is where the sign
        # of each phase comes from (see `_render`):
        self._amp1 = abs(amp1)
        self._amp2 = abs(amp2)
        self._phase_dur1 = phase_dur1
        self._phase_dur2 = phase_dur2
        self._interphase_dur = interphase_dur
        self._delay_dur = delay_dur
        self._stim_dur = stim_dur
        self._cathodic_first = cathodic_first
        self._defer(_electrode_names(electrode))

    @property
    def amp1(self):
        """Magnitude (uA) of the first pulse phase"""
        return self._amp1

    @property
    def amp2(self):
        """Magnitude (uA) of the second pulse phase"""
        return self._amp2

    @property
    def phase_dur1(self):
        """Duration (ms) of the first pulse phase"""
        return self._phase_dur1

    @property
    def phase_dur2(self):
        """Duration (ms) of the second pulse phase"""
        return self._phase_dur2

    @property
    def interphase_dur(self):
        """Duration (ms) of the gap between the two phases"""
        return self._interphase_dur

    @property
    def delay_dur(self):
        """Delay (ms) before the first phase is delivered"""
        return self._delay_dur

    @property
    def stim_dur(self):
        """Total stimulus duration (ms)"""
        return self._stim_dur

    @property
    def duration(self):
        """Stimulus duration (ms)

        The waveform is built to end exactly at ``stim_dur``, so this is
        known without generating one.
        """
        return self._stim_dur

    @property
    def cathodic_first(self):
        """Whether the cathodic phase is delivered first"""
        return self._cathodic_first

    def _render(self):
        """Build the waveform from the parameters above"""
        if self._cathodic_first:
            amp1, amp2 = -self._amp1, self._amp2
        else:
            amp1, amp2 = self._amp1, -self._amp2
        phase_dur1, phase_dur2 = self._phase_dur1, self._phase_dur2
        interphase_dur, delay_dur = self._interphase_dur, self._delay_dur
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
        data, time = _pad_to_stim_dur(time, data, self._stim_dur)
        return {'data': data, 'electrodes': self.electrodes, 'time': time}

    def _scaled(self, factor):
        """This pulse with both phase magnitudes multiplied by ``factor``

        See :py:meth:`BiphasicPulse._scaled`; the two phases keep their order,
        and only which of them is cathodic changes.
        """
        return AsymmetricBiphasicPulse(
            self.amp1 * abs(factor), self.amp2 * abs(factor),
            self.phase_dur1, self.phase_dur2,
            interphase_dur=self.interphase_dur, delay_dur=self.delay_dur,
            stim_dur=self.stim_dur,
            cathodic_first=(self.cathodic_first if factor >= 0
                            else not self.cathodic_first),
            electrode=self.electrodes[0])

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print

        The defining parameters rather than the waveform, so that printing a
        pulse does not generate one.
        """
        return {'amp1': self.amp1, 'amp2': self.amp2,
                'phase_dur1': self.phase_dur1,
                'phase_dur2': self.phase_dur2,
                'interphase_dur': self.interphase_dur,
                'delay_dur': self.delay_dur, 'stim_dur': self.stim_dur,
                'cathodic_first': self.cathodic_first,
                'electrodes': self.electrodes, 'metadata': self.metadata}
