"""`BarStimulus`, `GratingStimulus`"""

import numpy as np

from .videos import VideoStimulus
from ..utils import radial_mask


class GratingStimulus(VideoStimulus):
    """Drifting sinusoidal grating

    A drifting sinusoidal grating of a given spatial and temporal frequency.

    .. versionadded:: 0.7

    Parameters
    ----------
    shape : (height, width)
        A tuple specifying the desired height (pixels) and the width (pixels)
        of the grating stimulus.

    direction : scalar in [0, 360) degrees, optional
        Drift direction of the grating.

    spatial_freq : scalar (cycles/pixel), optional
        Spatial frequency of the grating in cycles per pixel

    temporal_freq : scalar (cycles/frame), optional
        Temporal frequency of the grating in cycles per frame

    phase : scalar (degrees), optional
        The initial phase of the grating in degrees

    contrast : scalar in [0, 1], optional
        Stimulus contrast between 0 and 1

    time : scalar, array-like, or None; optional
        The time points at which to evaluate the drifting grating:

        -  If a scalar, ``time`` is interpreted as the end time (in
           milliseconds) of a time series with 50 Hz frame rate.
        -  If array-like, ``time`` is interpreted as the exact time points (in
           milliseconds) at which to draw the grating (end point included).
        -  If None, ``time`` defaults to a 1-second time series at 50 Hz frame
           rate (end point included).

    mask : {'gauss', 'circle', None}
        Stimulus mask:

        -  "gauss": a 2D Gaussian designed such that the border of the image
           lies at 3 standard deviations
        -  "circle": a circle that fits into the ``shape`` of the stimulus
        -  None: no mask

    electrodes : int, string or list thereof; optional, default: None
        Optionally, you can provide your own electrode names. If none are
        given, electrode names will be numbered 0..N.

    metadata : dict, optional, default: None
        Additional stimulus metadata can be stored in a dictionary.

    """

    def __init__(self, shape, direction=0, spatial_freq=0.1,
                 temporal_freq=0.001, phase=0, contrast=1, time=None,
                 mask=None, electrodes=None, metadata=None):
        direction = np.deg2rad(direction)
        phase = np.deg2rad(phase)
        height, width = shape
        x = np.arange(width) - np.ceil(width / 2.0)
        y = np.arange(height) - np.ceil(height / 2.0)
        if time is None:
            time = np.arange(0, 1001, 20)
        elif isinstance(time, (list, np.ndarray)):
            time = np.asarray(time)
        else:
            time = np.arange(0, time + 1, 20)

        # Since `temporal_freq` is in cycles/frame, we need to pass the frame
        # indices as time, not the actual time points:
        X, Y, T = np.meshgrid(x, y, np.arange(len(time)), indexing='xy')
        channel = np.cos(-2 * np.pi * spatial_freq * np.cos(direction) * X +
                         2 * np.pi * spatial_freq * np.sin(direction) * Y +
                         2 * np.pi * temporal_freq * T +
                         phase)
        if mask is not None:
            mask = radial_mask((height, width), mask=mask)
            channel *= mask[..., np.newaxis]

        channel = contrast * channel / 2.0 + 0.5

        # Call VideoStimulus constructor:
        super(GratingStimulus, self).__init__(channel, as_gray=True,
                                              time=time,
                                              electrodes=electrodes,
                                              metadata=metadata,
                                              compress=False)


class BarStimulus(VideoStimulus):
    """Drifting bar

    A drifting bar stimulus.

    .. versionadded:: 0.7

    Parameters
    ----------
    shape : (height, width)
        A tuple specifying the desired height (pixels) and the width (pixels)
        of the grating stimulus.

    direction : scalar in [0, 360) degrees, optional
        Drift direction of the bar.

    speed : scalar in pixels/frame, optional
        Drift speed of the bar.

    bar_width : scalar in pixels, optional
        The width of the center of the bar.

    edge_width : scalar in pixels, optional
        The width of the cosine edges of the bar. An edge of width `edge_width`
        will be tacked onto both sides of the bar, so the total width will be
        `bar_width` + 2 * `edge_width`

    px_btw_bars : scalar in pixels, optional
        The number of pixels between the bars in the stimulus.

    start_pos : scalar in pixels, optional
        The starting position of the first bar. The coordinate system is a line
        lying along the direction of the bar motion passing through the center
        of the stimulus. The point 0 is the center of the stimulus.

    contrast : scalar in [0, 1], optional
        Stimulus contrast between 0 and 1

    time : scalar, array-like, or None; optional
        The time points at which to evaluate the drifting bar:

        -  If a scalar, ``time`` is interpreted as the end time (in
           milliseconds) of a time series with 50 Hz frame rate.
        -  If array-like, ``time`` is interpreted as the exact time points (in
           milliseconds) at which to draw the bar (end point included).
        -  If None, ``time`` defaults to a 1-second time series at 50 Hz frame
           rate (end point included).

    mask : {'gauss', 'circle', None}
        Stimulus mask:

        -  "gauss": a 2D Gaussian designed such that the border of the image
           lies at 3 standard deviations
        -  "circle": a circle that fits into the ``shape`` of the stimulus
        -  None: no mask

    electrodes : int, string or list thereof; optional, default: None
        Optionally, you can provide your own electrode names. If none are
        given, electrode names will be numbered 0..N.

    metadata : dict, optional, default: None
        Additional stimulus metadata can be stored in a dictionary.

    """

    def __init__(self, shape, direction=0, speed=0.1, bar_width=1,
                 edge_width=3, px_btw_bars=None, start_pos=0, contrast=1,
                 time=None, mask=None, electrodes=None, metadata=None):
        height, width = shape
        if px_btw_bars is None:
            px_btw_bars = width
        half_width = bar_width / 2.0

        # A bar is basically a single period of a sinusoidal grating. We don't
        # apply the mask and contrast here, but later:
        spatial_freq = 1.0 / px_btw_bars
        temporal_freq = spatial_freq * speed
        phase = start_pos * spatial_freq * 360  # deg
        grating = GratingStimulus(shape, time=time, direction=direction,
                                  contrast=1.0, mask=None, phase=phase,
                                  spatial_freq=spatial_freq,
                                  temporal_freq=temporal_freq)

        bar = grating.data.reshape(grating.vid_shape)
        for i in range(bar.shape[-1]):
            frame = bar[..., i]
            # There are 3 regions:
            # - where stim should be one (center of the bar)
            # - where stim should be zero (outside the bar)
            # - where stim should be in between (edges of the bar)
            bar_inner_th = np.cos(2 * np.pi * spatial_freq * half_width)
            bar_outer_th = np.cos(2 * np.pi * spatial_freq * (half_width +
                                                              edge_width))
            bar_one = frame >= bar_inner_th
            bar_edge = np.logical_and(frame < bar_inner_th,
                                      frame > bar_outer_th)
            bar_zero = frame <= bar_outer_th
            # Set the regions to the appropriate level:
            frame[bar_one] = 1.0
            frame[bar_zero] = 0.0
            # Adjust the range to [0, 2*pi):
            frame[bar_edge] = np.arccos(frame[bar_edge])
            # Adjust the range to [0, 1] spatial period:
            frame[bar_edge] = frame[bar_edge] / (2 * np.pi * spatial_freq)
            frame[bar_edge] = 0.5 * np.pi * (frame[bar_edge] - half_width)
            frame[bar_edge] /= edge_width
            frame[bar_edge] = np.cos(frame[bar_edge])
            bar[..., i] = frame

        # Adjust to range [-1, 1]:
        bar = 2.0 * bar - 1.0
        # Apply mask:
        if mask is not None:
            mask = radial_mask((height, width), mask=mask)
            bar *= mask[..., np.newaxis]
        # Apply contrast:
        bar = contrast * bar / 2.0 + 0.5

        # Call VideoStimulus constructor:
        super(BarStimulus, self).__init__(bar, as_gray=True,
                                          time=grating.time,
                                          electrodes=electrodes,
                                          metadata=metadata,
                                          compress=False)
