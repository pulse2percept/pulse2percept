# -*implants -*-
"""

Functions for creating retinal implants

"""
import numpy as np
import six
import copy
import logging

from pulse2percept import utils
from pulse2percept import implants
from pulse2percept import files

# Rather than trying to import these all over, try once and then remember
# by setting a flag.
try:
    import skimage.io as sio
    import skimage.transform as sit
    import skimage.color as sic
    has_skimage = True
except (ImportError, AttributeError):
    # Might also raise "dict object has no attribute 'io'"
    has_skimage = False


class MonophasicPulse(utils.TimeSeries):

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
        utils.TimeSeries.__init__(self, tsample, pulse[:stim_size])


class BiphasicPulse(utils.TimeSeries):

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
        utils.TimeSeries.__init__(self, tsample, pulse)


class PulseTrain(utils.TimeSeries):

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
            utils.TimeSeries.__init__(self, tsample, np.zeros(stim_size))
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

        utils.TimeSeries.__init__(self, tsample, pulse_train)


def image2pulsetrain(img, implant, coding='amplitude', valrange=[0, 50],
                     max_contrast=False, const_val=20, invert=False,
                     tsample=0.005 / 1000, dur=0.5, pulsedur=0.5 / 1000.,
                     interphasedur=0.5 / 1000., pulsetype='cathodicfirst'):
    """Converts an image into a series of pulse trains

    This function creates an input stimulus from an RGB or grayscale image.
    The image is down-sampled to fit the spatial layout of the implant
    (currently supported are ArgusI and ArgusII arrays).
    Requires Scikit-Image.

    Parameters
    ----------
    img : str|array_like
        An input image, either a valid filename (string) or a numpy array
        (row x col x channels).
    implant : p2p.implants.ElectrodeArray
        An ElectrodeArray object that describes the implant.
    coding : {'amplitude', 'frequency'}, optional
        A string describing the coding scheme:
        - 'amplitude': Image intensity is linearly converted to a current
                       amplitude between `valrange[0]` and `valrange[1]`.
                       Frequency is held constant at `const_freq`.
        - 'frequency': Image intensity is linearly converted to a pulse
                       frequency between `valrange[0]` and `valrange[1]`.
                       Amplitude is held constant at `const_amp`.
        Default: 'amplitude'
    valrange : list, optional
        Range of stimulation values to be used (If `coding` is 'amplitude',
        specifies min and max current; if `coding` is 'frequency', specifies
        min and max frequency).
        Default: [0, 50]
    max_contrast : bool, optional
        Flag wether to maximize image contrast (True) or not (False).
        Default: False
    const_val : float, optional
        For frequency coding: The constant amplitude value to be used for all
        pulse trains. For amplitude coding: The constant frequency value to
        be used for all pulse trains.
        Default: 20
    invert : bool, optional
        Flag whether to invert the grayscale values of the image (True) or
        not (False).
        Default: False
    tsample : float, optional
        Sampling time step (seconds). Default: 0.005 / 1000 seconds.
    dur : float, optional
        Stimulus duration (seconds). Default: 0.5 seconds.
    pulsedur : float, optional
        Duration of single (positive or negative) pulse phase in seconds.
    interphasedur : float, optional
        Duration of inter-phase interval (between positive and negative
        pulse) in seconds.
    pulsetype : {'cathodicfirst', 'anodicfirst'}, optional
        A cathodic-first pulse has the negative phase first, whereas an
        anodic-first pulse has the positive phase first.

    Returns
    -------
    pulses : list
        A list of p2p.stimuli.PulseTrain objects, one for each electrode in
        the implant.

    """
    if not has_skimage:
        # We don't want to repeatedly import Scikit-Image. This would (e.g.)
        # unnecessarily slow down `video2pulsetrain`.
        raise ImportError("You do not have scikit-image installed. "
                          "You can install it via $ pip install scikit-image.")

    # Make sure range of values is valid
    assert len(valrange) == 2 and valrange[1] > valrange[0]

    isargus1 = isinstance(implant, implants.ArgusI)
    isargus2 = isinstance(implant, implants.ArgusII)
    isalphaims = isinstance(implant, implants.AlphaIMS)
    if not isargus1 and not isargus2 and not isalphaims:
        raise TypeError("For now, implant must be of type implants.ArgusI or "
                        "implants.ArgusII.")

    if isinstance(img, six.string_types):
        # Load image from filename
        img_orig = sio.imread(img, as_grey=True).astype(np.float32)
        logging.getLogger(__name__).info("Loaded file '%s'." % img)
    else:
        if img.ndim == 2:
            # Grayscale
            img_orig = img.astype(np.float32)
        else:
            # Assume RGB, convert to grayscale
            assert img.shape[-1] == 3
            img_orig = sic.rgb2gray(np.array(img)).astype(np.float32)

    # Make sure all pixels are between 0 and 1
    if img_orig.max() > 1.0:
        img_orig /= 255.0

    # Let Scikit-Image do the resampling: Downscale image to fit array
    # Use mode 'reflect' for np.pad: Pads with the reflection of the vector
    # mirrored on the first and last values of the vector along each axis.
    if isargus1:
        img_stim = sit.resize(img_orig, (4, 4), mode='reflect')
    elif isargus2:
        img_stim = sit.resize(img_orig, (6, 10), mode='reflect')
    elif isalphaims:
        img_stim = sit.resize(img_orig, (37, 37), mode='reflect')

    # If specified, invert the mapping of grayscale values:
    if invert:
        img_stim = 1.0 - img_stim

    # If specified, maximize the contrast in the image:
    if max_contrast:
        img_stim -= img_stim.min()
        if img_stim.max() > 0:
            img_stim /= img_stim.max()

    # With all pixels between 0 and 1, now scale to valrange
    assert np.all(img_stim >= 0.0) and np.all(img_stim <= 1.0)
    img_stim = img_stim * np.diff(valrange) + valrange[0]
    assert np.all(img_stim >= valrange[0]) and np.all(img_stim <= valrange[1])

    stim = []
    for _, px in np.ndenumerate(img_stim):
        if coding == 'amplitude':
            amp = px
            freq = const_val
        elif coding == 'frequency':
            amp = const_val
            freq = px
        else:
            e_s = "Acceptable values for `coding` are 'amplitude' or"
            e_s += "'frequency'."
            raise ValueError(e_s)

        pt = PulseTrain(tsample, freq=freq, amp=amp, dur=dur,
                        pulse_dur=pulsedur,
                        interphase_dur=interphasedur,
                        pulsetype=pulsetype)
        stim.append(pt)

    return stim


def video2pulsetrain(filename, implant, framerate=20,
                     coding='amplitude', valrange=[0, 50],
                     max_contrast=False, const_val=20, invert=False,
                     tsample=0.005 / 1000, pulsedur=0.5 / 1000.,
                     interphasedur=0.5 / 1000., pulsetype='cathodicfirst',
                     ffmpeg_path=None, libav_path=None):
    """Converts a video into a series of pulse trains

    This function creates an input stimulus from a video.
    Every frame of the video is passed to `image2pulsetrain`, where it is
    down-sampled to fit the spatial layout of the implant (currently supported
    are ArgusI and ArgusII arrays).
    In this mapping, rows of the image correspond to rows in the implant
    (top row, Argus I: A1 B1 C1 D1, Argus II: A1 A2 ... A10).

    Requires Scikit-Image and Scikit-Video.

    Parameters
    ----------
    img : str|array_like
        An input image, either a valid filename (string) or a numpy array
        (row x col x channels).
    implant : p2p.implants.ElectrodeArray
        An ElectrodeArray object that describes the implant.
    coding : {'amplitude', 'frequency'}, optional
        A string describing the coding scheme:
        - 'amplitude': Image intensity is linearly converted to a current
                       amplitude between `valrange[0]` and `valrange[1]`.
                       Frequency is held constant at `const_freq`.
        - 'frequency': Image intensity is linearly converted to a pulse
                       frequency between `valrange[0]` and `valrange[1]`.
                       Amplitude is held constant at `const_amp`.
        Default: 'amplitude'
    valrange : list, optional
        Range of stimulation values to be used (If `coding` is 'amplitude',
        specifies min and max current; if `coding` is 'frequency', specifies
        min and max frequency).
        Default: [0, 50]
    max_contrast : bool, optional
        Flag wether to maximize image contrast (True) or not (False).
        Default: False
    const_val : float, optional
        For frequency coding: The constant amplitude value to be used for all
        pulse trains. For amplitude coding: The constant frequency value to
        be used for all pulse trains.
        Default: 20
    invert : bool, optional
        Flag whether to invert the grayscale values of the image (True) or
        not (False).
        Default: False
    tsample : float, optional
        Sampling time step (seconds). Default: 0.005 / 1000 seconds.
    dur : float, optional
        Stimulus duration (seconds). Default: 0.5 seconds.
    pulsedur : float, optional
        Duration of single (positive or negative) pulse phase in seconds.
    interphasedur : float, optional
        Duration of inter-phase interval (between positive and negative
        pulse) in seconds.
    pulsetype : {'cathodicfirst', 'anodicfirst'}, optional
        A cathodic-first pulse has the negative phase first, whereas an
        anodic-first pulse has the positive phase first.

    Returns
    -------
    pulses : list
        A list of p2p.stimuli.PulseTrain objects, one for each electrode in
        the implant.

    """

    # Load generator to read video frame-by-frame
    reader = files.load_video_generator(filename, ffmpeg_path, libav_path)

    # Temporarily increase logger level to suppress info messages
    current_level = logging.getLogger(__name__).getEffectiveLevel()
    logging.getLogger(__name__).setLevel(logging.WARN)

    # Convert the desired framerate to a duration (seconds)
    dur = 1.0 / framerate

    # Read one frame at a time, and append to previous frames
    video = []
    for img in reader.nextFrame():
        frame = image2pulsetrain(img, implant, coding=coding,
                                 valrange=valrange, max_contrast=max_contrast,
                                 const_val=const_val, invert=invert,
                                 tsample=tsample, dur=dur, pulsedur=pulsedur,
                                 interphasedur=interphasedur,
                                 pulsetype=pulsetype)
        if video:
            # List of pulse trains: Append new frame to each element
            [v.append(f) for v, f in zip(video, frame)]
        else:
            # Initialize with a list of pulse trains
            video = frame

    # Restore logger level
    logging.getLogger(__name__).setLevel(current_level)

    return video


def parse_pulse_trains(stim, implant):
    """Parse input stimulus and convert to list of pulse trains

    Parameters
    ----------
    stim : utils.TimeSeries|list|dict
        There are several ways to specify an input stimulus:

        - For a single-electrode array, pass a single pulse train; i.e., a
          single utils.TimeSeries object.
        - For a multi-electrode array, pass a list of pulse trains, where
          every pulse train is a utils.TimeSeries object; i.e., one pulse
          train per electrode.
        - For a multi-electrode array, specify all electrodes that should
          receive non-zero pulse trains by name in a dictionary. The key
          of each element is the electrode name, the value is a pulse train.
          Example: stim = {'E1': pt, 'stim': pt}, where 'E1' and 'stim' are
          electrode names, and `pt` is a utils.TimeSeries object.
    implant : p2p.implants.ElectrodeArray
        A p2p.implants.ElectrodeArray object that describes the implant.

    Returns
    -------
    A list of pulse trains; one pulse train per electrode.
    """
    # Parse input stimulus
    if isinstance(stim, utils.TimeSeries):
        # `stim` is a single object: This is only allowed if the implant
        # has only one electrode
        if implant.num_electrodes > 1:
            e_s = "More than 1 electrode given, use a list of pulse trains"
            raise ValueError(e_s)
        pt = [copy.deepcopy(stim)]
    elif isinstance(stim, dict):
        # `stim` is a dictionary: Look up electrode names and assign pulse
        # trains, fill the rest with zeros

        # Get right size from first dict element, then generate all zeros
        idx0 = list(stim.keys())[0]
        pt_zero = utils.TimeSeries(stim[idx0].tsample,
                                   np.zeros_like(stim[idx0].data))
        pt = [pt_zero] * implant.num_electrodes

        # Iterate over dictionary and assign non-zero pulse trains to
        # corresponding electrodes
        for key, value in stim.items():
            el_idx = implant.get_index(key)
            if el_idx is not None:
                pt[el_idx] = copy.deepcopy(value)
            else:
                e_s = "Could not find electrode with name '%s'" % key
                raise ValueError(e_s)
    else:
        # Else, `stim` must be a list of pulse trains, one for each electrode
        if len(stim) != implant.num_electrodes:
            e_s = "Number of pulse trains must match number of electrodes"
            raise ValueError(e_s)
        pt = copy.deepcopy(stim)

    return pt
