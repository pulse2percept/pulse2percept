# distutils: language = c++
# ^ needed for bool

from ..utils._fast_math cimport c_isclose, float32, int32
from libc.math cimport(fabs as c_abs, fmax as c_max)
from libcpp cimport bool
import numpy as np
cimport numpy as cnp

from .base import Stimulus
from .pulses import BiphasicPulse
from ..utils import unique

cpdef fast_encode(stim, int32 amp_min, int32 amp_max, pulse=None):
    """Encode image using amplitude modulation

    Encodes the image as a series of pulses, where the gray levels of the
    image are interpreted as the amplitude of a pulse with values in
    ``amp_range``.

    By default, a single biphasic pulse is used for each pixel, with 0.46ms
    phase duration, and 500ms total stimulus duration.

    Parameters
    ----------
    stim : VideoStimulus
        The VideoStimulus that will be encoded.
    amp_min, amp_max :
        Range of amplitude values to use for the encoding. The image's
        gray levels will be scaled such that the smallest value is mapped
        onto ``min_amp`` and the largest onto ``max_amp``.
    pulse : :py:class:`~pulse2percept.stimuli.Stimulus`, optional
        A valid pulse or pulse train to be used for the encoding.
        If None given, a :py:class:`~pulse2percept.stimuli.BiphasicPulse`
        (0.46 ms phase duration, stimulus duration inferred from the movie
        frame rate) will be used.

    Returns
    -------
    stim : :py:class:`~pulse2percept.stimuli.Stimulus`
        Encoded stimulus

    """
    cdef:
        float32 frame_dur
        float32 amp

    if pulse is not None:
        if not isinstance(pulse, Stimulus):
            raise TypeError("'pulse' must be a Stimulus object.")
        if pulse.time is None:
            raise ValueError("'pulse' must have a time component.")
    # Set frame rate, either from metadata or inferred from stim.time:
    try:
        frame_dur = 1000.0 / stim.metadata['fps']
    except KeyError:
        t_diff = unique(np.diff(stim.time))
        if len(t_diff) > 1:
            raise NotImplementedError
        frame_dur = 1000.0 / t_diff[0]
    # Normalize the range of pixel values:
    vid_data = stim.data - stim.data.min()
    if not c_isclose(np.abs(vid_data).max(), 0):
        vid_data /= np.abs(vid_data).max()
    # If no pulse is provided, create a default pulse
    # This pulse will be scaled to provide pixel grayscale levels
    if pulse is None:
        pulse = BiphasicPulse(1, 0.46, stim_dur=frame_dur)
    # Make sure the provided pulse has max amp 1:
    enc_data = pulse.data
    if not c_isclose(np.abs(enc_data).max(), 0):
        enc_data /= np.abs(enc_data).max()
    out_stim = {}
    for px_data, e in zip(vid_data, stim.electrodes):
        px_stim = None
        # For each pixel, we get a list of grayscale values (over time):
        for px in px_data:
            # Amplitude modulation:
            amp = px * (amp_max - amp_min) + amp_min
            s = Stimulus(amp * enc_data, time=pulse.time, electrodes=e)
            if px_stim is None:
                px_stim = s
            else:
                px_stim = px_stim.append(s)
        out_stim.update({e: px_stim})
    return Stimulus(out_stim)