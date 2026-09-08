"""Procedurally generated visual stimuli in physical visual units.

The generators return :py:class:`~pulse2percept.vision.Scene` objects and are
accessed through :mod:`pulse2percept.stimuli.psychophysics`.

:py:class:`~pulse2percept.stimuli.GratingStimulus` and
:py:class:`~pulse2percept.stimuli.BarStimulus` are deprecated legacy APIs and
will be removed in v0.12.
"""

import warnings

import numpy as np

from .base import ImageStimulus, VideoStimulus
from ..units import as_value, deg, dimensionless, dva, ms, s, Hz
from ..utils import radial_mask
from ..utils.constants import MS_PER_S
from ..utils.deprecation import deprecated

__all__ = [
    'bar',
    'BarStimulus',
    'grating',
    'GratingStimulus',
    'landolt_c',
    'tumbling_e',
]


@deprecated(alt_func='pulse2percept.stimuli.psychophysics.grating',
            deprecated_version='0.11.0', removed_version='0.12.0',
            extra_msg="The replacement returns a Scene and is parametrized "
                      "in physical units (cycles/dva, Hz, explicit sample "
                      "times in ms) rather than cycles/pixel, cycles/frame, "
                      "and an implicit 50 Hz frame rate, so parameter values "
                      "do not carry over unchanged.")
class GratingStimulus(VideoStimulus):
    """Drifting sinusoidal grating

    A drifting sinusoidal grating of a given spatial and temporal frequency.

    .. versionadded:: 0.7

    Parameters
    ----------
    shape : (height, width)
        A tuple specifying the desired height (pixels) and the width (pixels)
        of the grating stimulus.

    direction : scalar in [0, 360) degrees or Quantity, optional
        Drift direction of the grating.

    spatial_freq : scalar (cycles/pixel), optional
        Spatial frequency of the grating in cycles per pixel

    temporal_freq : scalar (cycles/frame), optional
        Temporal frequency of the grating in cycles per frame

    phase : scalar (degrees) or Quantity, optional
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
        given, each pixel is named after its place in the image: a letter for
        the row, a number for the column, and a suffix for the color channel
        (e.g. 'A1', 'C12', 'A1_R').

    metadata : dict, optional, default: None
        Additional stimulus metadata can be stored in a dictionary.

    """
    __slots__ = ()

    def __init__(self, shape, direction=0, spatial_freq=0.1,
                 temporal_freq=0.001, phase=0, contrast=1, time=None,
                 mask=None, electrodes=None, metadata=None):
        # `time` is a point (or an end point) in time, so it may be given as
        # a quantity; everything below works on plain milliseconds:
        time = as_value(time, ms, 'time')
        direction = np.deg2rad(as_value(direction, deg, 'direction'))
        phase = np.deg2rad(as_value(phase, deg, 'phase'))
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
        super().__init__(channel, as_gray=True,
                                              time=time,
                                              electrodes=electrodes,
                                              metadata=metadata,
                                              compress=False)


@deprecated(alt_func='pulse2percept.stimuli.psychophysics.bar',
            deprecated_version='0.11.0', removed_version='0.12.0',
            extra_msg="The replacement returns a Scene and is parametrized "
                      "in physical units (dva, dva/s, explicit sample times "
                      "in ms) rather than pixels, pixels/frame, and an "
                      "implicit 50 Hz frame rate; it also draws a single bar, "
                      "so 'px_btw_bars' has no counterpart.")
class BarStimulus(VideoStimulus):
    """Drifting bar

    A drifting bar stimulus.

    .. versionadded:: 0.7

    Parameters
    ----------
    shape : (height, width)
        A tuple specifying the desired height (pixels) and the width (pixels)
        of the grating stimulus.

    direction : scalar in [0, 360) degrees or Quantity, optional
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
        given, each pixel is named after its place in the image: a letter for
        the row, a number for the column, and a suffix for the color channel
        (e.g. 'A1', 'C12', 'A1_R').

    metadata : dict, optional, default: None
        Additional stimulus metadata can be stored in a dictionary.

    """
    __slots__ = ()

    def __init__(self, shape, direction=0, speed=0.1, bar_width=1,
                 edge_width=3, px_btw_bars=None, start_pos=0, contrast=1,
                 time=None, mask=None, electrodes=None, metadata=None):
        # See `GratingStimulus.__init__`:
        time = as_value(time, ms, 'time')
        height, width = shape
        if px_btw_bars is None:
            px_btw_bars = width
        half_width = bar_width / 2.0

        # A bar is basically a single period of a sinusoidal grating. We don't
        # apply the mask and contrast here, but later:
        spatial_freq = 1.0 / px_btw_bars
        temporal_freq = spatial_freq * speed
        phase = start_pos * spatial_freq * 360  # deg
        # The caller already got this class's deprecation notice; building
        # the grating internally must not warn a second time:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            grating = GratingStimulus(shape, time=time, direction=direction,
                                      contrast=1.0, mask=None, phase=phase,
                                      spatial_freq=spatial_freq,
                                      temporal_freq=temporal_freq)

        # A copy, because the loop below rewrites the grating frame by frame
        # and a stimulus does not hand out a buffer anyone can write into:
        bar = grating.data.reshape(grating.vid_shape).copy()
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
        super().__init__(bar, as_gray=True,
                                          time=grating.time,
                                          electrodes=electrodes,
                                          metadata=metadata,
                                          compress=False)


#: Landolt-C proportions, in multiples of the gap width. The stroke width is
#: half the difference, and therefore one gap wide as well.
_INNER_DIAMETER, _OUTER_DIAMETER = 3.0, 5.0

#: Tumbling-E proportions: the glyph spans five stroke widths either way, with
#: one-stroke bars separated by one-stroke gaps.
_E_EXTENT = 5.0

#: Output rows rasterized at a time, which caps how much of the finer grid
#: has to exist at once
_BLOCK_ROWS = 64

#: Supersampling used for optotype rasterization.
_SUPERSAMPLE = 4

#: Fewest pixels across a bar that still rasterize it as a bar, not a line
_MIN_BAR_PX = 2

#: Minimum output pixels across an optotype's critical feature. The 3-pixel
#: floor and 4x supersampling keep worst-case width error below 10% in the
#: validation sweep; see ``test_optotype_raster_floor``.
_MIN_OPTOTYPE_PX = 3


def _check_shape(shape):
    """Return ``shape`` as a positive integer ``(rows, cols)``"""
    shape = np.asarray(shape)
    if shape.shape != (2,) or not np.issubdtype(shape.dtype, np.integer):
        raise ValueError(f"'shape' must be a (rows, cols) pair of integers, "
                         f"not {shape.tolist()}.")
    if np.any(shape < 1):
        raise ValueError(f"'shape' must be positive, not {shape.tolist()}.")
    return int(shape[0]), int(shape[1])


def _resolve_shape_fov(shape, fov):
    """Return ``((n_rows, n_cols), (width, height))`` for a raster"""
    # Local import: `vision` imports `stimuli`, so this cannot be top-level.
    from ..vision.scene import _resolve_fov
    n_rows, n_cols = _check_shape(shape)
    return (n_rows, n_cols), _resolve_fov(fov, n_rows, n_cols)


def _visual_grid(shape, fov):
    """Return ``(x, y, (width, height))`` for a procedural stimulus

    ``x`` and ``y`` hold the pixel centers in visual-field coordinates,
    following the :py:class:`~pulse2percept.vision.Scene` convention: ``fov``
    is the outer extent of the frame, and row 0 holds the largest ``y``.
    """
    (n_rows, n_cols), (width, height) = _resolve_shape_fov(shape, fov)
    cols, rows = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    x = (cols + 0.5) * (width / n_cols) - width / 2
    y = height / 2 - (rows + 0.5) * (height / n_rows)
    return x, y, (width, height)


def _rasterize(glyph, shape, fov):
    """Supersample an analytic mask and area-average it onto the output raster.

    The raster is processed in row blocks to bound memory use.
    """
    n_rows, n_cols = shape
    n_fine_cols = n_cols * _SUPERSAMPLE
    pitch_x = fov[0] / n_fine_cols
    pitch_y = fov[1] / (n_rows * _SUPERSAMPLE)
    x = ((np.arange(n_fine_cols, dtype=np.float32) + 0.5) * pitch_x -
         fov[0] / 2)[np.newaxis, :]
    coverage = np.empty((n_rows, n_cols), dtype=np.float32)
    for first in range(0, n_rows, _BLOCK_ROWS):
        last = min(first + _BLOCK_ROWS, n_rows)
        fine = np.arange(first * _SUPERSAMPLE, last * _SUPERSAMPLE,
                         dtype=np.float32)
        y = (fov[1] / 2 - (fine + 0.5) * pitch_y)[:, np.newaxis]
        block = glyph(x, y).astype(np.float32)
        coverage[first:last] = block.reshape(last - first, _SUPERSAMPLE,
                                             n_cols, _SUPERSAMPLE).mean((1, 3))
    return coverage


def _check_raster(size, name, feature, fov, shape, minimum):
    """Require ``size`` to span at least ``minimum`` output pixels."""
    px = max(fov[0] / shape[1], fov[1] / shape[0])
    if size / px < minimum:
        raise ValueError(
            f"A {name} of {size:g} dva is {size / px:.2g} pixels across at "
            f"fov={fov} dva and shape={shape}, which does not resolve the "
            f"{feature}. At least {minimum} pixels are required: increase "
            f"'shape' or reduce 'fov'.")


def _check_contrast(contrast):
    """Return ``contrast`` as a Michelson contrast in [0, 1]"""
    contrast = float(as_value(contrast, dimensionless, 'contrast'))
    if not np.isfinite(contrast) or contrast < 0 or contrast > 1:
        raise ValueError(f"'contrast' is a Michelson contrast around mean "
                         f"gray and must lie in [0, 1], not {contrast}.")
    return contrast


def _check_angles(**angles):
    """Return the named angles as finite floats in degrees"""
    values = []
    for name, angle in angles.items():
        angle = float(as_value(angle, deg, name))
        if not np.isfinite(angle):
            raise ValueError(f"'{name}' must be a finite angle in degrees, "
                             f"not {angle}.")
        values.append(angle)
    return values


def _time_points(time):
    """Return explicit sample times in ms, or None for a static image.

    Scalar durations are rejected because these generators assume no frame rate.
    """
    time = as_value(time, ms, 'time')
    if time is None:
        return None
    time = np.asarray(time, dtype=float)
    if time.ndim == 0:
        raise ValueError(
            f"'time' is either None (a static image) or the explicit sample "
            f"times of a video, in ms, not the scalar {time.item():g}. These "
            f"generators assume no frame rate, so pass the time points "
            f"themselves, e.g. np.arange(0, 1000, 20).")
    if time.ndim != 1 or time.size == 0:
        raise ValueError(f"'time' must be a non-empty 1-D array of sample "
                         f"times in ms, not an array of shape {time.shape}.")
    if not np.all(np.isfinite(time)):
        raise ValueError("'time' must hold finite sample times in ms.")
    return time


def _elapsed_s(time):
    """Return sample times in seconds, using t=0 for a static stimulus."""
    return np.zeros(1) if time is None else time / MS_PER_S


def _check_motion_sampled(name, value, unit, time):
    """Reject nonzero motion without explicit sample times."""
    if value != 0 and time is None:
        raise ValueError(
            f"'{name}' is {value:g} {unit}, but 'time' is None, which "
            f"rasterizes a single static frame at t = 0 and would drop the "
            f"motion. Pass the sample times in ms, e.g. "
            f"np.arange(0, 500, 10), or set '{name}' to 0 for a static "
            f"stimulus.")


def _aperture(x, y, mask, fov):
    """Radial aperture in [0, 1], measured in visual-field coordinates

    Isotropic in dva, unlike :py:func:`~pulse2percept.utils.radial_mask`,
    which normalizes each pixel axis separately and so becomes an ellipse on a
    non-square frame. The aperture is the largest circle that fits the field,
    i.e. half its shorter side in radius; ``'gauss'`` puts 3 standard
    deviations at that radius.
    """
    if mask is None:
        return None
    radius = np.hypot(x, y) / (min(fov) / 2.0)
    if mask == 'circle':
        return (radius <= 1).astype(float)
    if mask == 'gauss':
        return np.exp(-4.5 * radius ** 2)
    raise ValueError(f'Unknown mask "{mask}". Choose either "circle" or '
                     f'"gauss".')


def _to_gray(pattern, contrast, window):
    """Map a pattern in [-1, 1] onto gray levels around mean gray 0.5

    ``window`` (an aperture in [0, 1], or None) multiplies the pattern rather
    than the gray levels, so apertured regions fade to mean gray, not black.
    """
    if window is not None:
        pattern = pattern * window[..., np.newaxis]
    return (contrast * pattern / 2.0 + 0.5).astype(np.float32)


def _raster_source(gray, time, metadata):
    """Wrap a raster as an ImageStimulus or VideoStimulus."""
    if time is None:
        return ImageStimulus(gray[..., 0], metadata=metadata, compress=False)
    return VideoStimulus(gray, time=time, metadata=metadata, compress=False)


def _check_spatial_nyquist(spatial_freq, direction, fov, shape):
    """Raise unless the raster resolves the grating along both pixel axes

    A grating of frequency ``fs`` at angle ``theta`` has components
    ``fs |cos(theta)|`` and ``fs |sin(theta)|`` along x and y, each of which
    has to stay strictly below the Nyquist frequency of its own angular pixel
    pitch. Direction matters: a grating that varies only vertically is
    resolved by a frame of wide, short pixels. Equality is not enough either:
    at exactly two samples per cycle the phase is unrecoverable, and a
    quadrature-phase grating rasterizes as a uniform field.
    """
    n_rows, n_cols = shape
    theta = np.deg2rad(direction)
    for axis, component, pitch in (('horizontal', abs(np.cos(theta)),
                                    fov[0] / n_cols),
                                   ('vertical', abs(np.sin(theta)),
                                    fov[1] / n_rows)):
        freq, nyquist = spatial_freq * component, 0.5 / pitch
        if freq >= nyquist:
            raise ValueError(
                f"A grating of {spatial_freq:g} cycles/dva at "
                f"direction={direction:g} deg has a {axis} component of "
                f"{freq:g} cycles/dva, at or above the {nyquist:g} "
                f"cycles/dva Nyquist frequency of a {pitch:g} dva pixel, so "
                f"it would alias. Increase 'shape', or reduce 'fov' or "
                f"'spatial_freq'.")


def _check_temporal_nyquist(temporal_freq, time):
    """Raise unless consecutive samples resolve the drift

    Adjacent frames must advance the grating by less than half a temporal
    cycle. At or past that, the sampled sequence is indistinguishable from a
    slower or a reversed drift, which is the confusion that dropping
    cycles/frame was meant to end.
    """
    if temporal_freq == 0 or time is None or time.size < 2:
        return
    # The widest gap decides: one long gap aliases a sequence that is densely
    # sampled everywhere else.
    step = float(np.max(np.abs(np.diff(time)))) / MS_PER_S
    if temporal_freq * step >= 0.5:
        raise ValueError(
            f"A drift rate of {temporal_freq:g} Hz advances the grating by "
            f"{temporal_freq * step:g} cycles across the widest gap in "
            f"'time' ({step * MS_PER_S:g} ms), at or past the half cycle "
            f"where the sampled drift becomes ambiguous, i.e. past the "
            f"temporal Nyquist limit. Sample at most every "
            f"{0.5 / temporal_freq * MS_PER_S:g} ms, or reduce "
            f"'temporal_freq'.")


def grating(spatial_freq=1, temporal_freq=0, direction=0, phase=0, contrast=1,
            fov=10, shape=(256, 256), time=None, mask=None):
    """Sinusoidal grating

    Rasterize a sinusoidal luminance grating of a given spatial frequency,
    drift rate, and direction, and place it in a
    :py:class:`~pulse2percept.vision.Scene`.

    The gray level at visual-field coordinate ``(x, y)`` and time ``t`` is
    ``0.5 + contrast / 2 * cos(2 pi fs u - 2 pi ft t + phase)``, where
    ``u = x cos(direction) + y sin(direction)`` is the distance along the
    drift axis in dva, ``fs`` is ``spatial_freq`` in cycles/dva, and ``ft`` is
    ``temporal_freq`` in Hz. The pattern therefore drifts along ``direction``
    at ``temporal_freq / spatial_freq`` dva/s.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    spatial_freq : float or Quantity, optional
        Spatial frequency in cycles per degree of visual angle (e.g.
        ``2 / dva``). One cycle is ``1 / spatial_freq`` dva wide, whatever
        ``shape`` is. Its components along x and y must both stay strictly
        below the Nyquist frequency of the corresponding angular pixel pitch.
    temporal_freq : float or Quantity, optional
        Drift rate in Hz (e.g. ``4 * Hz``), non-negative: ``direction`` alone
        says which way the grating drifts. 0 leaves the pattern static; any
        other rate requires ``time``.
    direction : float or Quantity, optional
        Drift direction, in degrees counterclockwise from the positive x axis
        (e.g. ``90 * deg``): 0 right, 90 up, 180 left, 270 down. The bars run
        perpendicular to it, so ``direction=0`` gives vertical bars drifting
        rightwards.
    phase : float or Quantity, optional
        Spatial phase in degrees, at fixation and at ``t = 0``. 0 puts a
        luminance peak at the center of the frame.
    contrast : float, optional
        Michelson contrast in [0, 1] around mean gray 0.5: the pattern spans
        ``0.5 +/- contrast / 2``.
    fov : float or (width, height), optional
        How much of the visual field the scene covers, in dva. A scalar is the
        horizontal extent, and the vertical one follows from ``shape``.
    shape : (rows, cols), optional
        Size of the rasterized frame, in pixels. Raster resolution only: it
        does not enter the spatial frequency.
    time : array_like or None, optional
        Sample times in ms (e.g. ``np.arange(0, 500, 10)``), which the scene's
        source carries as a
        :py:class:`~pulse2percept.stimuli.VideoStimulus`. None gives a static
        :py:class:`~pulse2percept.stimuli.ImageStimulus` instead, and is only
        allowed when ``temporal_freq`` is 0. There is no default frame rate,
        so a scalar duration is rejected. Consecutive samples must advance the
        drift by less than half a temporal cycle.
    mask : {'gauss', 'circle', None}, optional
        Radial aperture applied to the pattern, which fades to mean gray
        outside it. Isotropic in dva and centered on fixation:

        -  ``'gauss'``: a 2D Gaussian whose 3rd standard deviation lies at the
           aperture radius
        -  ``'circle'``: the largest circle that fits into ``fov``
        -  None: no aperture

    Returns
    -------
    scene : :py:class:`~pulse2percept.vision.Scene`

    Examples
    --------
    A static 2 cycles/dva grating across 10 degrees:

    >>> from pulse2percept.stimuli import psychophysics
    >>> from pulse2percept.units import dva
    >>> scene = psychophysics.grating(spatial_freq=2 / dva, fov=10 * dva)
    >>> type(scene.source).__name__
    'ImageStimulus'

    The same grating drifting upwards at 4 Hz, sampled every 10 ms:

    >>> import numpy as np
    >>> from pulse2percept.units import deg, Hz
    >>> scene = psychophysics.grating(spatial_freq=2 / dva,
    ...                               temporal_freq=4 * Hz,
    ...                               direction=90 * deg, fov=10 * dva,
    ...                               time=np.arange(0, 500, 10))
    >>> type(scene.source).__name__
    'VideoStimulus'

    """
    # Local import: `vision` imports `stimuli`, so this cannot be top-level.
    from ..vision.scene import Scene
    spatial_freq = float(as_value(spatial_freq, dva ** -1, 'spatial_freq'))
    if not np.isfinite(spatial_freq) or spatial_freq <= 0:
        raise ValueError(f"'spatial_freq' is a spatial frequency in "
                         f"cycles/dva and must be finite and positive, not "
                         f"{spatial_freq}.")
    temporal_freq = float(as_value(temporal_freq, Hz, 'temporal_freq'))
    if not np.isfinite(temporal_freq) or temporal_freq < 0:
        raise ValueError(f"'temporal_freq' is a drift rate in Hz and must be "
                         f"finite and non-negative, not {temporal_freq}. "
                         f"'direction' alone says which way the grating "
                         f"drifts, so use direction + 180 rather than a "
                         f"negative rate.")
    direction, phase = _check_angles(direction=direction, phase=phase)
    contrast = _check_contrast(contrast)
    time = _time_points(time)
    _check_motion_sampled('temporal_freq', temporal_freq, 'Hz', time)
    x, y, fov = _visual_grid(shape, fov)
    _check_spatial_nyquist(spatial_freq, direction, fov, x.shape)
    _check_temporal_nyquist(temporal_freq, time)
    window = _aperture(x, y, mask, fov)
    theta = np.deg2rad(direction)
    # Signed distance along the drift axis, in dva:
    u = x * np.cos(theta) + y * np.sin(theta)
    pattern = np.cos(2 * np.pi * spatial_freq * u[..., np.newaxis] -
                     2 * np.pi * temporal_freq * _elapsed_s(time) +
                     np.deg2rad(phase))
    metadata = {'generator': 'grating', 'spatial_freq': spatial_freq,
                'temporal_freq': temporal_freq, 'direction': direction,
                'phase': phase, 'contrast': contrast, 'mask': mask,
                'fov': fov}
    source = _raster_source(_to_gray(pattern, contrast, window), time,
                            metadata)
    return Scene(source, fov=fov)


def bar(width=1, direction=0, speed=0, offset=0, edge_width=0, contrast=1,
        fov=10, shape=(256, 256), time=None, mask=None):
    """Moving bar

    Rasterize a single bright bar of a given angular width, moving at a given
    speed and direction, and place it in a
    :py:class:`~pulse2percept.vision.Scene`.

    The bar is a stripe perpendicular to the direction of motion, at
    ``0.5 + contrast / 2`` on a ``0.5 - contrast / 2`` background. Its center
    sits at ``offset + speed * t`` along the motion axis, measured from
    fixation, so ``offset`` is where it is at ``t = 0`` and a given timestamp
    puts it in the same place no matter how the video is sampled.

    This generator draws one bar; use :func:`grating` for periodic patterns.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    width : float or Quantity, optional
        Angular width of the bar's plateau, in dva (e.g. ``0.5 * dva``),
        measured across the direction of motion. Must resolve to at least two
        pixels.
    direction : float or Quantity, optional
        Direction of motion, in degrees counterclockwise from the positive x
        axis (e.g. ``90 * deg``): 0 right, 90 up, 180 left, 270 down. The bar
        itself is perpendicular to it, so ``direction=0`` is a vertical bar
        moving rightwards.
    speed : float or Quantity, optional
        Speed along ``direction``, in dva/s (e.g. ``5 * dva / s``),
        non-negative: ``direction`` alone says which way the bar moves. 0
        leaves the bar stationary; any other speed requires ``time``.
    offset : float or Quantity, optional
        Signed position of the bar's center at ``t = 0``, in dva along the
        motion axis, measured from fixation.
    edge_width : float or Quantity, optional
        Width in dva of the raised-cosine ramp added to either side of the
        plateau, over which the bar falls off to the background. The bar is
        ``width + 2 * edge_width`` across in total. 0 gives hard edges.
    contrast : float, optional
        Michelson contrast in [0, 1] around mean gray 0.5: bar and background
        sit at ``0.5 +/- contrast / 2``.
    fov : float or (width, height), optional
        How much of the visual field the scene covers, in dva. A scalar is the
        horizontal extent, and the vertical one follows from ``shape``.
    shape : (rows, cols), optional
        Size of the rasterized frame, in pixels. Raster resolution only: it
        does not enter the bar's width or speed.
    time : array_like or None, optional
        Sample times in ms (e.g. ``np.arange(0, 500, 10)``), which the scene's
        source carries as a
        :py:class:`~pulse2percept.stimuli.VideoStimulus`. None gives a static
        :py:class:`~pulse2percept.stimuli.ImageStimulus` instead, and is only
        allowed when ``speed`` is 0. There is no default frame rate, so a
        scalar duration is rejected.
    mask : {'gauss', 'circle', None}, optional
        Radial aperture applied to the pattern, which fades to mean gray
        outside it. Isotropic in dva and centered on fixation:

        -  ``'gauss'``: a 2D Gaussian whose 3rd standard deviation lies at the
           aperture radius
        -  ``'circle'``: the largest circle that fits into ``fov``
        -  None: no aperture

    Returns
    -------
    scene : :py:class:`~pulse2percept.vision.Scene`

    Examples
    --------
    A static 1-degree bar, 3 degrees left of fixation:

    >>> from pulse2percept.stimuli import psychophysics
    >>> from pulse2percept.units import dva
    >>> scene = psychophysics.bar(width=1 * dva, offset=-3 * dva,
    ...                           fov=10 * dva)
    >>> type(scene.source).__name__
    'ImageStimulus'

    The same bar sweeping rightwards at 10 dva/s, sampled every 10 ms:

    >>> import numpy as np
    >>> from pulse2percept.units import s
    >>> scene = psychophysics.bar(width=1 * dva, offset=-3 * dva,
    ...                           speed=10 * dva / s, fov=10 * dva,
    ...                           time=np.arange(0, 600, 10))
    >>> type(scene.source).__name__
    'VideoStimulus'

    """
    # Local import: `vision` imports `stimuli`, so this cannot be top-level.
    from ..vision.scene import Scene
    width = float(as_value(width, dva, 'width'))
    if not np.isfinite(width) or width <= 0:
        raise ValueError(f"'width' is an angular width and must be finite "
                         f"and positive, not {width}.")
    edge_width = float(as_value(edge_width, dva, 'edge_width'))
    if not np.isfinite(edge_width) or edge_width < 0:
        raise ValueError(f"'edge_width' is an angular width and must be "
                         f"finite and non-negative, not {edge_width}.")
    speed = float(as_value(speed, dva / s, 'speed'))
    if not np.isfinite(speed) or speed < 0:
        raise ValueError(f"'speed' is a speed in dva/s and must be finite and "
                         f"non-negative, not {speed}. 'direction' alone says "
                         f"which way the bar moves, so use direction + 180 "
                         f"rather than a negative speed.")
    offset = float(as_value(offset, dva, 'offset'))
    if not np.isfinite(offset):
        raise ValueError(f"'offset' is a position in dva along the motion "
                         f"axis and must be finite, not {offset}.")
    direction, = _check_angles(direction=direction)
    contrast = _check_contrast(contrast)
    time = _time_points(time)
    _check_motion_sampled('speed', speed, 'dva/s', time)
    x, y, fov = _visual_grid(shape, fov)
    # A bar under two pixels wide rasterizes as an aliased line, not a bar:
    _check_raster(width, 'bar width', 'bar', fov, x.shape, _MIN_BAR_PX)
    window = _aperture(x, y, mask, fov)
    theta = np.deg2rad(direction)
    # Signed distance along the motion axis, in dva, and where the bar's
    # center sits on that axis at each sample time:
    u = x * np.cos(theta) + y * np.sin(theta)
    center = offset + speed * _elapsed_s(time)
    dist = np.abs(u[..., np.newaxis] - center)
    # Plateau, then the raised-cosine ramp; `profile` is in [0, 1], with 0 the
    # background:
    half = width / 2.0
    profile = (dist <= half).astype(float)
    if edge_width > 0:
        ramp = (dist - half) / edge_width
        edge = (dist > half) & (ramp <= 1)
        profile[edge] = 0.5 * (1 + np.cos(np.pi * ramp[edge]))
    metadata = {'generator': 'bar', 'width': width, 'edge_width': edge_width,
                'direction': direction, 'speed': speed, 'offset': offset,
                'contrast': contrast, 'mask': mask, 'fov': fov}
    source = _raster_source(_to_gray(2.0 * profile - 1.0, contrast, window),
                            time, metadata)
    return Scene(source, fov=fov)


def _landolt_mask(x, y, gap, position, orientation):
    """Boolean mask of the C: an annulus with a gap-wide slot cut out of it"""
    theta = np.deg2rad(orientation)
    # Coordinates relative to the optotype's center, then rotated so that the
    # opening always points along +u:
    dx, dy = x - position[0], y - position[1]
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    radius = np.hypot(u, v)
    annulus = ((radius >= _INNER_DIAMETER / 2 * gap) &
               (radius <= _OUTER_DIAMETER / 2 * gap))
    # The opening is a slot of width `gap` measured across the gap direction,
    # which is what "gap size" means for a Landolt C:
    slot = (u > 0) & (np.abs(v) <= gap / 2)
    return annulus & ~slot


def landolt_c(gap=1, position=(0, 0), orientation=0, fov=10, polarity='dark',
              shape=(512, 512)):
    """Landolt C optotype

    Rasterize a Landolt C and place it in a
    :py:class:`~pulse2percept.vision.Scene`.

    The C uses standard proportions: the stroke and gap are equal, with inner
    and outer diameters of ``3 * gap`` and ``5 * gap``. The analytic glyph is
    supersampled and area-averaged onto the requested raster; ``gap`` must
    span at least three output pixels.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    gap : float or Quantity, optional
        Angular width of the critical opening, in degrees of visual angle
        (e.g. ``0.5 * dva``).
    position : (x, y), optional
        Center of the optotype in visual-field coordinates, in dva. ``y``
        grows upwards.
    orientation : float or Quantity, optional
        Direction the opening points, in degrees counterclockwise from the
        positive x axis (e.g. ``90 * deg``): 0 right, 90 up, 180 left, 270
        down. Any finite angle is accepted.
    fov : float or (width, height), optional
        How much of the visual field the scene covers, in dva. A scalar is the
        horizontal extent, and the vertical one follows from ``shape``.
    polarity : {'dark', 'light'}, optional
        ``'dark'`` draws a black C on white, ``'light'`` a white C on black.
    shape : (rows, cols), optional
        Size of the rasterized frame, in pixels.

    Returns
    -------
    scene : :py:class:`~pulse2percept.vision.Scene`

    Examples
    --------
    A 0.5-degree gap pointing up, five degrees to the right of fixation:

    >>> from pulse2percept.stimuli import psychophysics
    >>> from pulse2percept.units import deg, dva
    >>> scene = psychophysics.landolt_c(gap=0.5 * dva,
    ...                                 position=(5, 0) * dva,
    ...                                 orientation=90 * deg, fov=15 * dva)
    >>> scene.fov
    (15.0, 15.0)

    """
    # Local import: `vision` imports `stimuli`, so this cannot be top-level.
    from ..vision.scene import Scene
    gap = float(as_value(gap, dva, 'gap'))
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError(f"'gap' is an angular width and must be finite and "
                         f"positive, not {gap}.")
    center = np.asarray(as_value(position, dva, 'position'), dtype=float)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError(f"'position' must be a finite (x, y) pair in dva, "
                         f"not {position!r}.")
    orientation = float(as_value(orientation, deg, 'orientation'))
    if not np.isfinite(orientation):
        raise ValueError(f"'orientation' must be a finite angle in degrees, "
                         f"not {orientation}.")
    if polarity not in ('dark', 'light'):
        raise ValueError(f"'polarity' is either 'dark' (black C on white) or "
                         f"'light' (white C on black), not {polarity!r}.")
    shape, (width, height) = _resolve_shape_fov(shape, fov)

    # Cropping a C changes the task rather than the picture, so refuse it:
    radius = _OUTER_DIAMETER / 2 * gap
    for name, offset, extent in (('horizontally', center[0], width),
                                 ('vertically', center[1], height)):
        if abs(offset) + radius > extent / 2:
            raise ValueError(
                f"A Landolt C with gap={gap:g} dva at position="
                f"{center.tolist()} dva reaches {abs(offset) + radius:g} dva "
                f"{name} from fixation, past the {extent / 2:g} dva half-FOV. "
                f"Increase 'fov', or move the optotype closer to fixation.")
    # An opening narrower than a couple of pixels rasterizes as a closed ring,
    # i.e. as a different optotype:
    _check_raster(gap, 'gap', 'opening', (width, height), shape,
                  _MIN_OPTOTYPE_PX)
    coverage = _rasterize(
        lambda x, y: _landolt_mask(x, y, gap, center, orientation), shape,
        (width, height))
    ink, paper = (0.0, 1.0) if polarity == 'dark' else (1.0, 0.0)
    img = (paper + (ink - paper) * coverage).astype(np.float32)
    metadata = {'generator': 'landolt_c', 'gap': gap,
                'position': (float(center[0]), float(center[1])),
                'orientation': orientation, 'polarity': polarity,
                'fov': (width, height)}
    return Scene(ImageStimulus(img, metadata=metadata), fov=(width, height))


def _tumbling_e_mask(x, y, stroke, position, orientation):
    """Boolean mask of the E: a spine column plus three full-width bars"""
    theta = np.deg2rad(orientation)
    # Coordinates relative to the optotype's center, then rotated so that the
    # bars always point along +u:
    dx, dy = x - position[0], y - position[1]
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    half = _E_EXTENT / 2 * stroke
    within = (np.abs(u) <= half) & (np.abs(v) <= half)
    # The spine is the leftmost one-stroke column, spanning the full height:
    spine = u <= -(half - stroke)
    # Top, middle, and bottom bars, each one stroke thick and full width:
    bars = (np.abs(v) <= stroke / 2) | (np.abs(v) >= half - stroke)
    return within & (spine | bars)


def tumbling_e(stroke=1, position=(0, 0), orientation=0, fov=10,
               polarity='dark', shape=(512, 512)):
    """Tumbling E optotype

    Rasterize a Tumbling E and place it in a
    :py:class:`~pulse2percept.vision.Scene`.

    The E uses the standard 5x5 construction: bars and gaps are one ``stroke``
    wide and the full optotype is ``5 * stroke`` across. The glyph is
    supersampled and area-averaged; ``stroke`` must span at least three
    output pixels.

    The cardinal orientations form the conventional Tumbling-E task, although
    any finite angle is accepted.

    .. note::
       The Tumbling E and the Landolt C
       (:py:func:`~pulse2percept.stimuli.psychophysics.landolt_c`) are
       different optotypes measured with different tasks (bar direction vs.
       gap direction). Thresholds obtained with one are not numerically
       interchangeable with the other.

    The analytic E is supersampled and area-averaged onto the requested
    raster, so edge pixels carry the fraction of the glyph they cover. This
    keeps bar and gap widths from depending on where the pixel grid happens to
    fall, and matters most at off-cardinal orientations. ``stroke`` must still
    span at least three output pixels.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    stroke : float or Quantity, optional
        Angular width of a bar, in degrees of visual angle (e.g.
        ``0.5 * dva``). The whole E is ``5 * stroke`` across.
    position : (x, y), optional
        Center of the optotype in visual-field coordinates, in dva. ``y``
        grows upwards.
    orientation : float or Quantity, optional
        Direction the bars point, in degrees counterclockwise from the
        positive x axis (e.g. ``90 * deg``): 0 right, 90 up, 180 left, 270
        down. Any finite angle is accepted.
    fov : float or (width, height), optional
        How much of the visual field the scene covers, in dva. A scalar is the
        horizontal extent, and the vertical one follows from ``shape``.
    polarity : {'dark', 'light'}, optional
        ``'dark'`` draws a black E on white, ``'light'`` a white E on black.
    shape : (rows, cols), optional
        Size of the rasterized frame, in pixels.

    Returns
    -------
    scene : :py:class:`~pulse2percept.vision.Scene`

    Examples
    --------
    A 0.5-degree stroke pointing up, five degrees to the right of fixation:

    >>> from pulse2percept.stimuli import psychophysics
    >>> from pulse2percept.units import deg, dva
    >>> scene = psychophysics.tumbling_e(stroke=0.5 * dva,
    ...                                  position=(5, 0) * dva,
    ...                                  orientation=90 * deg, fov=15 * dva)
    >>> scene.fov
    (15.0, 15.0)

    """
    # Local import: `vision` imports `stimuli`, so this cannot be top-level.
    from ..vision.scene import Scene
    stroke = float(as_value(stroke, dva, 'stroke'))
    if not np.isfinite(stroke) or stroke <= 0:
        raise ValueError(f"'stroke' is an angular width and must be finite "
                         f"and positive, not {stroke}.")
    center = np.asarray(as_value(position, dva, 'position'), dtype=float)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError(f"'position' must be a finite (x, y) pair in dva, "
                         f"not {position!r}.")
    orientation = float(as_value(orientation, deg, 'orientation'))
    if not np.isfinite(orientation):
        raise ValueError(f"'orientation' must be a finite angle in degrees, "
                         f"not {orientation}.")
    if polarity not in ('dark', 'light'):
        raise ValueError(f"'polarity' is either 'dark' (black E on white) or "
                         f"'light' (white E on black), not {polarity!r}.")
    shape, (width, height) = _resolve_shape_fov(shape, fov)

    # Cropping an E changes the task rather than the picture, so refuse it.
    # The glyph is a square, so off-cardinal angles need the axis-aligned
    # extent of the rotated square, not its half-width:
    theta = np.deg2rad(orientation)
    half = _E_EXTENT / 2 * stroke
    extent = half * (abs(np.cos(theta)) + abs(np.sin(theta)))
    for name, offset, fov_size in (('horizontally', center[0], width),
                                   ('vertically', center[1], height)):
        if abs(offset) + extent > fov_size / 2:
            raise ValueError(
                f"A Tumbling E with stroke={stroke:g} dva at position="
                f"{center.tolist()} dva and orientation={orientation:g} deg "
                f"reaches {abs(offset) + extent:g} dva {name} from fixation, "
                f"past the {fov_size / 2:g} dva half-FOV. Increase 'fov', or "
                f"move the optotype closer to fixation.")
    # Bars narrower than a couple of pixels merge with their gaps, i.e. turn
    # the E into a filled square:
    _check_raster(stroke, 'stroke', 'bars', (width, height), shape,
                  _MIN_OPTOTYPE_PX)
    coverage = _rasterize(
        lambda x, y: _tumbling_e_mask(x, y, stroke, center, orientation),
        shape, (width, height))
    ink, paper = (0.0, 1.0) if polarity == 'dark' else (1.0, 0.0)
    img = (paper + (ink - paper) * coverage).astype(np.float32)
    metadata = {'generator': 'tumbling_e', 'stroke': stroke,
                'position': (float(center[0]), float(center[1])),
                'orientation': orientation, 'polarity': polarity,
                'fov': (width, height)}
    return Scene(ImageStimulus(img, metadata=metadata), fov=(width, height))
