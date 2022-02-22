# distutils: language = c++
# ^ needed for bool

from cython.parallel import prange
from libc.math cimport(fabs as c_abs, fmax as c_max)
from libcpp cimport bool
import numpy as np
cimport numpy as cnp

from .base import Stimulus
from .pulses import BiphasicPulse
from ..utils import unique
from ..utils._fast_math cimport c_isclose, float32, int32

#use memoryiews for enc_data and pulse_time

ctypedef fused electrode_t:
    long
    unicode
    bytes

ctypedef fused time_t:
    int32
    float32

cpdef fast_encode( float32 [:, :] vid_data, electrode_t [:] electrodes, float32 [:] enc_data, time_t [:] pulse_time, int32 amp_min, int32 amp_max ):
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

    #Not working on BostonTrain, maybe because Pulse has multiple dimensions?

    cdef:
        float32 amp, px
        float32 [:] px_stim, new_data
        time_t [:] px_stim_time, new_time
        int32 elec_num

    out_stim = {}
    stim_time = np.empty([1,0])
    for px_data, e in zip(vid_data, electrodes):
        px_stim = None
        # For each pixel, we get a list of grayscale values (over time):
        for px in px_data:
            # Amplitude modulation:
            amp = px * (amp_max - amp_min) + amp_min
            if px_stim is None:
                px_stim = enc_data.copy()
                for i in range(px_stim.shape[0]):
                    px_stim[i] = <float32> (px_stim[i] * amp)
                stim_time = pulse_time.copy()
            else:
                new_data = enc_data.copy()
                new_time = pulse_time.copy()
                for i in range(len(new_time)):
                    new_time[i] += stim_time[len(stim_time)-1]
                for i in range(new_data.shape[0]):
                    new_data[i] = <float32> (new_data[i] * amp)
                if c_isclose(enc_data[len(enc_data)-1], new_data[0]):
                    px_stim = np.hstack([px_stim, new_data[1:]])
                    stim_time = np.concatenate([stim_time, new_time[1:]])
                else:
                    px_stim = np.hstack([px_stim, new_data])
                    stim_time = np.concatenate([stim_time, new_time])
        out_stim.update({e: px_stim})
    return out_stim, stim_time