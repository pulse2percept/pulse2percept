"""`TimeSeries`, `MonophasicPulse`, `BiphasicPulse`, `PulseTrain`"""
import numpy as np
import copy
import logging
from scipy.interpolate import interp1d


from .base import TimeSeries, Stimulus
from .pulses import BiphasicPulse, AsymmetricBiphasicPulse


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
        pulse = amp * BiphasicPulse(pulsetype, pulse_dur, tsample,
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
