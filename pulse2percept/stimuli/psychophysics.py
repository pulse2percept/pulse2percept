"""`GratingStimulus`"""

import numpy as np

from .videos import VideoStimulus
from ..utils import radial_mask


class GratingStimulus(VideoStimulus):
    """Drifting sinusoidal grating

    .. versionadded:: 0.7

    Parameters
    ----------
    shape : (height, width)
        A tuple specifying the desired height and the width of the grating
        stimulus.

    electrodes : int, string or list thereof; optional, default: None
        Optionally, you can provide your own electrode names. If none are
        given, electrode names will be numbered 0..N.

    time :

    direction :

        spatial_freq :

        temporal_freq :

        phase :

        contrast :

    mask : {'gauss', 'circle', None}

    metadata : dict, optional, default: None
        Additional stimulus metadata can be stored in a dictionary.

    """

    def __init__(self, shape, time=None, electrodes=None, direction=0,
                 spatial_freq=0.1, temporal_freq=0.001, phase=0, contrast=1,
                 mask='gauss', metadata=None):

        direction = np.deg2rad(direction)
        phase = np.deg2rad(phase)

        height, width = shape
        x = np.arange(width) - np.ceil(width / 2.0)
        y = np.arange(height) - np.ceil(height / 2.0)
        if time is None:
            t = np.arange(0, 1001, 20)
        elif isinstance(time, (list, np.ndarray)):
            t = np.asarray(time)
        else:
            t = np.arange(0, time + 1, 20)

        X, Y, T = np.meshgrid(x, y, t, indexing='xy')

        channel = np.cos(-2 * np.pi * spatial_freq * np.cos(direction) * X
                         + 2 * np.pi * spatial_freq *
                         np.sin(direction) * Y
                         + 2 * np.pi * temporal_freq * T
                         + 2 * np.pi * phase)
        if mask is not None:
            mask = radial_mask((height, width), mask=mask)
            channel *= mask[..., np.newaxis]

        channel = contrast * channel / 2 + 0.5

        # Call VideoStimulus constructor:
        super(GratingStimulus, self).__init__(channel, as_gray=True,
                                              time=t,
                                              electrodes=electrodes,
                                              metadata=metadata,
                                              compress=False)
