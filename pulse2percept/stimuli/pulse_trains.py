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
from ..units import Hz, as_value, ms, uA, xTh
from ..utils.constants import DT, MS_PER_S

def _as_threshold_amp(threshold_amp, name='threshold_amp'):
    """Normalize a threshold to a plain, strictly positive value in uA"""
    threshold_amp = as_value(threshold_amp, uA, name)
    if threshold_amp is None:
        return None
    threshold_amp = float(threshold_amp)
    if not np.isfinite(threshold_amp) or threshold_amp <= 0:
        raise ValueError(f"'{name}' must be a positive, finite current, not "
                         f"{threshold_amp}.")
    return threshold_amp


def _is_threshold_relative(amp):
    """Return True if an amplitude is given as a multiple of threshold"""
    return getattr(amp, 'dimension', None) == xTh.dimension


def _tile_pulse(pulse, shift, n_pulses):
    """Tile ``pulse`` ``n_pulses`` times with ``shift`` ms between copies.

    This is the vectorized equivalent of repeated ``Stimulus.append``.

    Returns
    -------
    data, time : np.ndarray
        Waveform and time axis of the tiled train.
    """
    time, data = pulse.time, pulse.data
    # The time axis of each appended copy, i.e. of ``pulse >> shift``:
    shifted = time + shift
    if shifted[0] < 0:
        raise NotImplementedError("Appending a stimulus with a negative "
                                  "time axis is currently not supported.")
    # ``append`` offsets copy k by the last time point of copy k-1:
    # last[k] = shifted[-1] + last[k-1], last[0] = time[-1]. The cumsum
    # accumulates in the same order, so it rounds identically (temporal models
    # resolve stimulus edges on a fixed simulation grid):
    steps = np.full(n_pulses, shifted[-1], dtype=np.float64)
    steps[0] = time[-1]
    offsets = np.cumsum(steps, dtype=np.float64)[:-1, np.newaxis]
    if isclose(shifted[0], 0, abs_tol=DT):
        # The last point of one copy coincides with the first point of the
        # next. Merge them if their amplitudes match:
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
    *  Only whole pulses are delivered: the number of pulses is rounded down
       (e.g., a 30 Hz train in a 33.37 ms window has one pulse). A partial
       pulse would leave a net current.
    *  A frequency below ``1000 / stim_dur`` Hz still delivers one pulse. Pass
       ``freq=0`` for a silent train.
    *  Arguments may be given as plain numbers in the units documented above,
       or as unitful quantities (e.g. ``0.02 * kHz``, ``1 * s``), which are
       converted to those units. See :py:mod:`pulse2percept.units`.
    *  The train uses the unit of ``pulse`` (e.g., uA or dimensionless).

    """
    #: See `Stimulus._is_parametric`:
    _is_parametric = True

    __slots__ = ('_freq', '_pulse', '_n_pulses', '_n_pulses_asked',
                 '_stim_dur')

    def __init__(self, freq, pulse, n_pulses=None, stim_dur=1000.0,
                 electrode=None, metadata=None):
        # Normalize frequency and duration.
        freq = as_value(freq, Hz, 'freq')
        stim_dur = as_value(stim_dur, ms, 'stim_dur')
        if not isinstance(pulse, Stimulus):
            raise TypeError(f"'pulse' must be a Stimulus object, not "
                            f"{type(pulse)}.")
        n_rows = len(pulse.electrodes)
        if n_rows == 0:
            raise ValueError(f"'pulse' has invalid shape "
                             f"({pulse.shape[0]}, {pulse.shape[1]}).")
        # Raw pulses must carry a time axis.
        if not pulse._is_parametric and pulse.time is None:
            raise ValueError("'pulse' does not have a time component.")
        # ``duration`` avoids rendering a parametric pulse.
        pulse_dur = pulse.duration

        # How many pulses fit into stim dur (`freq` in Hz, `stim_dur` in ms):
        n_max_pulses = freq * stim_dur / MS_PER_S
        # Store the requested count for `_scaled`. The resolved default below
        # counts whole pulses from t=0 and can exceed `n_max_pulses` by one:
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
            # Only whole pulses; a truncated pulse would leave a net current:
            n_pulses = int(np.floor((stim_dur - pulse_dur) /
                                    (MS_PER_S / freq) + 1e-9)) + 1
        # A silent train (0 Hz, or no pulse fits) is a single row of zeros:
        if n_pulses <= 0:
            n_rows = 1
        else:
            # Window duration (ms) is the inverse of pulse train frequency.
            # Checked at construction rather than in `_render`:
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
        # Snapshot the pulse so later caller changes cannot affect this train.
        self._pulse = deepcopy(pulse)
        self._n_pulses = n_pulses
        self._stim_dur = stim_dur
        # Inherit the pulse's units (also for a silent train), instead of the
        # default uA:
        self._defer(names, unit=pulse.unit, time_unit=pulse.time_unit)
        self.metadata = {'user': metadata}

    @property
    def freq(self):
        """Pulse train frequency (Hz)"""
        return self._freq

    @property
    def pulse(self):
        """The single pulse this train repeats"""
        return deepcopy(self._pulse)

    def __deepcopy__(self, memo):
        train = super().__deepcopy__(memo)
        train._pulse = deepcopy(self._pulse, memo)
        return train

    @property
    def n_pulses(self):
        """Number of pulses delivered"""
        return self._n_pulses

    @property
    def stim_dur(self):
        """Total stimulus duration (ms)"""
        return self._stim_dur

    @property
    def pulse_type(self):
        """Name of the class the repeated pulse came from"""
        return self._pulse.__class__.__name__

    @property
    def duration(self):
        """Stimulus duration (ms)"""
        return self._stim_dur

    def _render(self):
        """Return the tiled pulse train"""
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
            # Keep the interpolated end point at least DT after the previous
            # point, so the time axis stays strictly increasing:
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
        """Return this train with the pulse amplitudes scaled by ``factor``"""
        return PulseTrain(self.freq, self._pulse * factor,
                          n_pulses=self._n_pulses_asked,
                          stim_dur=self.stim_dur,
                          electrode=self.electrodes,
                          metadata=deepcopy(self.metadata.get('user')))

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
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

        May also be given as a multiple of perceptual threshold, e.g.
        ``2 * xTh`` (see :py:data:`~pulse2percept.units.xTh`). The current is
        ``amp * threshold_amp``; without a threshold, the train stays in
        ``xTh``.
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
    threshold_amp : float, optional
        Perceptual threshold (uA) of the stimulated electrode. Converts
        between :py:attr:`amp_factor` and current.

        .. versionadded:: 0.10.0

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
    *  ``unit`` is uA or ``xTh``. An ``xTh`` train without a threshold cannot
       be delivered by an implant until a threshold is assigned.

    Examples
    --------
    Twice threshold, on an electrode whose threshold has been measured:

    >>> from pulse2percept.stimuli import BiphasicPulseTrain
    >>> from pulse2percept.units import uA, xTh
    >>> pt = BiphasicPulseTrain(20, 2 * xTh, 0.45, threshold_amp=80 * uA)
    >>> pt.amp, pt.amp_factor
    (160.0, 2.0)

    The same stimulation, given as a current:

    >>> pt = BiphasicPulseTrain(20, 160.0 * uA, 0.45, threshold_amp=80 * uA)
    >>> pt.amp, pt.amp_factor
    (160.0, 2.0)

    Without a threshold, the train stays in xTh:

    >>> pt = BiphasicPulseTrain(20, 2.0 * xTh, 0.45)
    >>> pt.amp_factor, pt.unit, pt.threshold_amp
    (2.0, xTh, None)

    """
    #: See `Stimulus._is_parametric`:
    _is_parametric = True

    __slots__ = ('_train', '_amp_relative', '_explicit_threshold_amp',
                 '_threshold_override')

    def __init__(self, freq, amp, phase_dur, interphase_dur=0, delay_dur=0,
                 n_pulses=None, stim_dur=1000.0, cathodic_first=True,
                 electrode=None, metadata=None, threshold_amp=None):
        # Convert to plain numbers in Hz, uA, ms:
        freq = as_value(freq, Hz, 'freq')
        # Preserve whether amp was specified as current or threshold multiple:
        self._explicit_threshold_amp = _as_threshold_amp(threshold_amp)
        # Keep an implant override separate so it can be cleared later.
        self._threshold_override = None
        self._amp_relative = _is_threshold_relative(amp)
        if self._amp_relative:
            amp = as_value(amp, xTh, 'amp')
            if self.threshold_amp is not None:
                amp = amp * self.threshold_amp
        else:
            amp = as_value(amp, uA, 'amp')
        unit = xTh if self._amp_relative and self.threshold_amp is None else uA
        phase_dur = as_value(phase_dur, ms, 'phase_dur')
        interphase_dur = as_value(interphase_dur, ms, 'interphase_dur')
        delay_dur = as_value(delay_dur, ms, 'delay_dur')
        stim_dur = as_value(stim_dur, ms, 'stim_dur')
        # Create the individual pulse:
        pulse = BiphasicPulse(amp, phase_dur, delay_dur=delay_dur,
                              interphase_dur=interphase_dur,
                              cathodic_first=cathodic_first,
                              electrode=electrode)
        # Concatenate the pulses. Built here so arguments are checked at
        # construction; the waveform is still generated lazily:
        self._train = PulseTrain(freq, pulse, n_pulses=n_pulses,
                                 stim_dur=stim_dur)
        self._defer(_electrode_names(electrode), unit=unit)
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
    def duration(self):
        """Stimulus duration (ms)"""
        return self._train.stim_dur

    @property
    def amp(self):
        """Magnitude of both phases of each pulse, in :py:attr:`unit`"""
        return self._train._pulse.amp

    @property
    def threshold_amp(self):
        """Perceptual threshold (uA) of this train

        The implant calibration if set, else the constructor's
        ``threshold_amp``, else None.

        .. versionadded:: 0.10.0
        """
        if self._threshold_override is not None:
            return self._threshold_override
        return self._explicit_threshold_amp

    @property
    def amp_factor(self):
        """Amplitude as a multiple of :py:attr:`threshold_amp`

        None for a current amplitude without a threshold. An ``xTh``
        amplitude is returned as-is.

        .. versionadded:: 0.10.0
        """
        threshold_amp = self.threshold_amp
        if threshold_amp is None:
            return self.amp if self._amp_relative else None
        return self.amp / threshold_amp

    @property
    def phase_dur(self):
        """Duration (ms) of the cathodic/anodic phase"""
        return self._train._pulse.phase_dur

    @property
    def interphase_dur(self):
        """Duration (ms) of the gap between the two phases"""
        return self._train._pulse.interphase_dur

    @property
    def delay_dur(self):
        """Delay (ms) before the first phase of each pulse"""
        return self._train._pulse.delay_dur

    @property
    def cathodic_first(self):
        """Whether the cathodic phase is delivered first"""
        return self._train._pulse.cathodic_first

    def _render(self):
        """Return the tiled pulse train"""
        return {'data': self._train.data, 'electrodes': self.electrodes,
                'time': self._train.time}

    def _rebuilt(self, amp, threshold_amp, cathodic_first=None):
        """Return a copy of this train with a new amplitude and threshold"""
        if cathodic_first is None:
            cathodic_first = self.cathodic_first
        train = BiphasicPulseTrain(
            self.freq, amp, self.phase_dur,
            interphase_dur=self.interphase_dur, delay_dur=self.delay_dur,
            n_pulses=self._train._n_pulses_asked,
            stim_dur=self.stim_dur, cathodic_first=cathodic_first,
            electrode=self.electrodes[0],
            metadata=deepcopy(self.metadata.get('user')),
            threshold_amp=threshold_amp)
        train._explicit_threshold_amp = self._explicit_threshold_amp
        train._threshold_override = self._threshold_override
        return train

    def _with_threshold(self, override):
        """Return this train calibrated to threshold ``override`` (uA)"""
        if override == self._threshold_override:
            return self
        amp = self.amp_factor * xTh if self._amp_relative else self.amp
        resolved = (override if override is not None
                    else self._explicit_threshold_amp)
        train = self._rebuilt(amp, resolved)
        train._threshold_override = override
        return train

    def _scaled(self, factor):
        """Return this train with every amplitude multiplied by ``factor``

        Keeps the amplitude unit: doubling ``2 * xTh`` gives ``4 * xTh``.
        """
        if self._amp_relative:
            amp = self.amp_factor * abs(factor) * xTh
        else:
            amp = self.amp * abs(factor)
        # A negative factor swaps the two phases:
        return self._rebuilt(amp, self.threshold_amp,
                             cathodic_first=(self.cathodic_first if factor >= 0
                                             else not self.cathodic_first))

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        # Print the amplitude as given (xTh or uA), since the two recalibrate
        # differently (see `_with_threshold`):
        amp = self.amp_factor * xTh if self._amp_relative else self.amp
        params = {'freq': self.freq, 'amp': amp,
                  'phase_dur': self.phase_dur,
                  'interphase_dur': self.interphase_dur,
                  'delay_dur': self.delay_dur, 'stim_dur': self.stim_dur,
                  'cathodic_first': self.cathodic_first,
                  'electrodes': self.electrodes, 'metadata': self.metadata}
        if self.threshold_amp is not None:
            params['threshold_amp'] = self.threshold_amp
        return params


class AsymmetricBiphasicPulseTrain(Stimulus):
    """Asymmetric biphasic pulse train

    A train of asymmetric biphasic pulses, each with a cathodic and an
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
        stimulus to make the stimulus last ``stim_dur`` ms overall.
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
    #: See `Stimulus._is_parametric`:
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
    def duration(self):
        """Stimulus duration (ms)"""
        return self._train.stim_dur

    @property
    def amp1(self):
        """Magnitude (uA) of the first phase of each pulse"""
        return self._train._pulse.amp1

    @property
    def amp2(self):
        """Magnitude (uA) of the second phase of each pulse"""
        return self._train._pulse.amp2

    @property
    def phase_dur1(self):
        """Duration (ms) of the first pulse phase"""
        return self._train._pulse.phase_dur1

    @property
    def phase_dur2(self):
        """Duration (ms) of the second pulse phase"""
        return self._train._pulse.phase_dur2

    @property
    def interphase_dur(self):
        """Duration (ms) of the gap between the two phases"""
        return self._train._pulse.interphase_dur

    @property
    def delay_dur(self):
        """Delay (ms) before the first phase of each pulse"""
        return self._train._pulse.delay_dur

    @property
    def cathodic_first(self):
        """Whether the cathodic phase is delivered first"""
        return self._train._pulse.cathodic_first

    def _render(self):
        """Return the tiled pulse train"""
        return {'data': self._train.data, 'electrodes': self.electrodes,
                'time': self._train.time}

    def _scaled(self, factor):
        """Return this train with both phases multiplied by ``factor``"""
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
        """Return a dict of class arguments to pretty-print"""
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
    #: See `Stimulus._is_parametric`:
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
        pulse = BiphasicPulse(amp, phase_dur, interphase_dur=interphase_dur,
                              delay_dur=delay_dur,
                              cathodic_first=cathodic_first,
                              electrode=electrode)
        self._pulse = pulse
        self._interpulse_dur = interpulse_dur
        if interpulse_dur != 0:
            delay_pulse = MonophasicPulse(0, interpulse_dur, electrode=electrode)
            pulse = pulse.append(delay_pulse)
        triplet = pulse.append(pulse).append(pulse)
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
    def duration(self):
        """Stimulus duration (ms)"""
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
        """Return the tiled pulse train"""
        return {'data': self._train.data, 'electrodes': self.electrodes,
                'time': self._train.time}

    def _scaled(self, factor):
        """Return this train with every phase multiplied by ``factor``"""
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
        """Return a dict of class arguments to pretty-print"""
        return {'freq': self.freq, 'amp': self.amp,
                'phase_dur': self.phase_dur,
                'interphase_dur': self.interphase_dur,
                'interpulse_dur': self.interpulse_dur,
                'delay_dur': self.delay_dur, 'stim_dur': self.stim_dur,
                'cathodic_first': self.cathodic_first,
                'electrodes': self.electrodes, 'metadata': self.metadata}
