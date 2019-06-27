import numpy as np
import copy
import logging
from scipy import interpolate as spi


class TimeSeries(object):

    def __init__(self, tsample, data):
        """Container for time series data

        Provides a container for time series data. Every time series has a
        sampling step `tsample`, and some `data` sampled at that rate.

        Parameters
        ----------
        tsample : float
            Sampling time step (seconds).
        data : array_like
            Time series data sampled at every `tsample` seconds.
        """
        self.data = data
        self.tsample = tsample
        self.duration = self.data.shape[-1] * tsample
        self.shape = data.shape

    def __getitem__(self, y):
        return TimeSeries(self.tsample, self.data[y])

    def append(self, other):
        """Appends the data of another TimeSeries object (in time)

        This function concatenates the data of another TimeSeries object to
        the current object, along the last dimension (time). To make this work,
        all but the last dimension of the two objects must be the same.

        If the two objects have different time sampling steps, the other object
        is resampled to fit the current `tsample`.

        Parameters
        ----------
        other : TimeSeries
            A TimeSeries object whose content should be appended.

        Examples
        --------
        >>> from pulse2percept import stimuli
        >>> pt = stimuli.TimeSeries(1.0, np.zeros((2, 2, 10)))
        >>> num_frames = pt.shape[-1]
        >>> pt.append(pt)
        >>> pt.shape[-1] == 2 * num_frames
        True
        """
        # Make sure type is correct
        if not isinstance(other, TimeSeries):
            raise TypeError("Other object must be of type "
                            "p2p.stimuli.TimeSeries.")

        # Make sure size is correct for all but the last dimension (number
        # of frames)
        if self.shape[:-1] != other.shape[:-1]:
            raise ValueError("Shape mismatch: ", self.shape[:-1], " vs. ",
                             other.shape[:-1])

        # Then resample the other to current `tsample`
        resampled = other.resample(self.tsample)

        # Then concatenate the two
        self.data = np.concatenate((self.data, resampled.data), axis=-1)
        self.duration = self.data.shape[-1] * self.tsample
        self.shape = self.data.shape

    def max(self):
        """Returns the time and value of the largest data point

        This function returns the the largest value in the TimeSeries data,
        as well as the time at which it occurred.

        Returns
        -------
        t : float
            time (s) at which max occurred
        val : float
            max value
        """
        # Find index and value of largest element
        idx = self.data.argmax()
        val = self.data.max()

        # Find frame that contains the brightest data point using `unravel`,
        # which maps the flat index `idx_px` onto the high-dimensional
        # indices (x,y,z).
        # What we want is index `z` (i.e., the frame index), given by the last
        # dimension in the return argument.
        idx_frame = np.unravel_index(idx, self.data.shape)[-1]

        # Convert index to time
        t = idx_frame * self.tsample

        return t, val

    def max_frame(self):
        """Returns the time frame that contains the largest data point

        This function returns the time frame in the TimeSeries data that
        contains the largest data point, as well as the time at which
        it occurred.

        Returns
        -------
        t : float
            time (s) at which max occurred
        frame : TimeSeries
            A TimeSeries object.
        """
        # Find index and value of largest element
        idx = self.data.argmax()

        # Find frame that contains the brightest data point using `unravel`,
        # which maps the flat index `idx_px` onto the high-dimensional
        # indices (x,y,z).
        # What we want is index `z` (i.e., the frame index), given by the last
        # dimension in the return argument.
        idx_frame = np.unravel_index(idx, self.data.shape)[-1]
        t = idx_frame * self.tsample
        frame = self.data[..., idx_frame]

        return t, TimeSeries(self.tsample, copy.deepcopy(frame))

    def resample(self, tsample_new):
        """Returns data sampled according to new time step

        This function returns a TimeSeries object whose data points were
        resampled according to a new time step `tsample_new`. New values
        are found using linear interpolation.

        Parameters
        ----------
        tsample_new : float
            New sampling time step (s)

        Returns
        -------
        TimeSeries object
        """
        if tsample_new is None or tsample_new == self.tsample:
            return TimeSeries(self.tsample, self.data)

        # Try to avoid rounding errors in arr size by making sure `t_old` is
        # too long at first, then cutting it to the right size
        y_old = self.data
        t_old = np.arange(0, self.duration + self.tsample, self.tsample)
        t_old = t_old[:y_old.shape[-1]]
        f = spi.interp1d(t_old, y_old, axis=-1, fill_value='extrapolate')

        t_new = np.arange(0, self.duration, tsample_new)
        y_new = f(t_new)

        return TimeSeries(tsample_new, y_new)


class MonophasicPulse(TimeSeries):

    def __init__(self, ptype, pdur, tsample, delay_dur=0, stim_dur=None):
        """A pulse with a single phase

        Parameters
        ----------
        ptype : {'anodic', 'cathodic'}
            Pulse type. Anodic pulses have positive current amplitude,
            cathodic pulses have negative amplitude.
        pdur : float
            Pulse duration (s).
        tsample : float
            Sampling time step (s).
        delay_dur : float, optional
            Pulse delay (s). Pulse will be zero-padded (prepended) to deliver
            the pulse only after `delay_dur` milliseconds. Default: 0.
        stim_dur : float, optional
            Stimulus duration (ms). Pulse will be zero-padded (appended) to fit
            the stimulus duration. Default: No additional zero padding,
            `stim_dur` is `pdur`+`delay_dur`.
        """
        if tsample <= 0:
            raise ValueError("tsample must be a non-negative float.")

        if stim_dur is None:
            stim_dur = pdur + delay_dur

        # Convert durations to number of samples
        pulse_size = int(np.round(pdur / tsample))
        delay_size = int(np.round(delay_dur / tsample))
        stim_size = int(np.round(stim_dur / tsample))

        if ptype == 'cathodic':
            pulse = -np.ones(pulse_size)
        elif ptype == 'anodic':
            pulse = np.ones(pulse_size)
        else:
            raise ValueError("Acceptable values for `ptype` are 'anodic', "
                             "'cathodic'.")

        pulse = np.concatenate((np.zeros(delay_size), pulse,
                                np.zeros(stim_size)))
        TimeSeries.__init__(self, tsample, pulse[:stim_size])


class BiphasicPulse(TimeSeries):

    def __init__(self, ptype, pdur, tsample, interphase_dur=0):
        """A charge-balanced pulse with a cathodic and anodic phase

        A single biphasic pulse with duration `pdur` per phase,
        separated by `interphase_dur` is returned.

        Parameters
        ----------
        ptype : {'cathodicfirst', 'anodicfirst'}
            A cathodic-first pulse has the negative phase first, whereas an
            anodic-first pulse has the positive phase first.
        pdur : float
            Duration of single (positive or negative) pulse phase in seconds.
        tsample : float
            Sampling time step in seconds.
        interphase_dur : float, optional
            Duration of inter-phase interval (between positive and negative
            pulse) in seconds. Default: 0.
        """
        if tsample <= 0:
            raise ValueError("tsample must be a non-negative float.")

        # Get the two monophasic pulses
        on = MonophasicPulse('anodic', pdur, tsample, 0, pdur)
        off = MonophasicPulse('cathodic', pdur, tsample, 0, pdur)

        # Insert interphase gap if necessary
        gap = np.zeros(int(round(interphase_dur / tsample)))

        # Order the pulses
        if ptype == 'cathodicfirst':
            # has negative current first
            pulse = np.concatenate((off.data, gap), axis=0)
            pulse = np.concatenate((pulse, on.data), axis=0)
        elif ptype == 'anodicfirst':
            pulse = np.concatenate((on.data, gap), axis=0)
            pulse = np.concatenate((pulse, off.data), axis=0)
        else:
            raise ValueError("Acceptable values for `type` are "
                             "'anodicfirst' or 'cathodicfirst'")
        TimeSeries.__init__(self, tsample, pulse)


class PulseTrain(TimeSeries):

    def __init__(self, tsample, freq=20, amp=20, dur=0.5, delay=0,
                 pulse_dur=0.45 / 1000, interphase_dur=0.45 / 1000,
                 pulsetype='cathodicfirst',
                 pulseorder='pulsefirst'):
        """A train of biphasic pulses

        Parameters
        ----------
        tsample : float
            Sampling time step (seconds).
        freq : float, optional, default: 20 Hz
            Frequency of the pulse envelope (Hz).
        amp : float, optional, default: 20 uA
            Max amplitude of the pulse train in micro-amps.
        dur : float, optional, default: 0.5 seconds
            Stimulus duration in seconds.
        delay : float, optional, default: 0
            Delay until stimulus on-set in seconds.
        pulse_dur : float, optional, default: 0.45 ms
            Single-pulse duration in seconds.
        interphase_duration : float, optional, default: 0.45 ms
            Single-pulse interphase duration (the time between the positive
            and negative phase) in seconds.
        pulsetype : str, optional, default: 'cathodicfirst'
            Pulse type {'cathodicfirst' | 'anodicfirst'}, where
            'cathodicfirst' has the negative phase first.
        pulseorder : str, optional, default: 'pulsefirst'
            Pulse order {'gapfirst' | 'pulsefirst'}, where
            'pulsefirst' has the pulse first, followed by the gap.
            'gapfirst' has it the other way round.
        """
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

        TimeSeries.__init__(self, tsample, pulse_train)
