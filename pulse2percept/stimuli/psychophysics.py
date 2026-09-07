""":py:class:`~pulse2percept.stimuli.BarStimulus`,
   :py:class:`~pulse2percept.stimuli.GratingStimulus`,
   :py:func:`~pulse2percept.stimuli.psychophysics.landolt_c`,
   :py:func:`~pulse2percept.stimuli.psychophysics.tumbling_e`

Procedurally generated visual stimuli.

Nothing here is loaded from a bundled file (see
:py:mod:`pulse2percept.stimuli.samples` for those): every pattern is
rasterized from its parameters. The optotype generators are reached
through the module rather than the top-level namespace::

    from pulse2percept.stimuli import psychophysics
    from pulse2percept.units import dva
    scene = psychophysics.landolt_c(gap=0.5 * dva)
"""

import numpy as np

from .images import ImageStimulus
from .videos import VideoStimulus
from ..units import as_value, deg, dva, ms
from ..utils import radial_mask

__all__ = [
    'BarStimulus',
    'GratingStimulus',
    'landolt_c',
    'tumbling_e',
]


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
        (e.g. 'A1', 'C12', 'A1_R'). See
        :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

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
        (e.g. 'A1', 'C12', 'A1_R'). See
        :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

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

#: Fewest pixels across an optotype's critical feature that still rasterize it
_MIN_FEATURE_PX = 2


def _check_shape(shape):
    """Return ``shape`` as a positive integer ``(rows, cols)``"""
    shape = np.asarray(shape)
    if shape.shape != (2,) or not np.issubdtype(shape.dtype, np.integer):
        raise ValueError(f"'shape' must be a (rows, cols) pair of integers, "
                         f"not {shape.tolist()}.")
    if np.any(shape < 1):
        raise ValueError(f"'shape' must be positive, not {shape.tolist()}.")
    return int(shape[0]), int(shape[1])


def _optotype_grid(shape, fov):
    """Return ``(x, y, (width, height))`` for a procedural optotype

    ``x`` and ``y`` hold the pixel centers in visual-field coordinates,
    following the :py:class:`~pulse2percept.vision.Scene` convention: ``fov``
    is the outer extent of the frame, and row 0 holds the largest ``y``.
    """
    # Local import: `vision` imports `stimuli`, so this cannot be top-level.
    from ..vision.scene import _resolve_fov
    n_rows, n_cols = _check_shape(shape)
    width, height = _resolve_fov(fov, n_rows, n_cols)
    cols, rows = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    x = (cols + 0.5) * (width / n_cols) - width / 2
    y = height / 2 - (rows + 0.5) * (height / n_rows)
    return x, y, (width, height)


def _check_raster(size, name, feature, fov, shape):
    """Raise unless ``size`` spans ``_MIN_FEATURE_PX`` pixels of the raster

    ``size`` is the angular size of parameter ``name``, which rasterizes as
    ``feature`` (the C's opening, the E's bars). Measured on the coarser of
    the two angular pixel sizes, so neither axis may under-resolve it.
    """
    px = max(fov[0] / shape[1], fov[1] / shape[0])
    if size / px < _MIN_FEATURE_PX:
        raise ValueError(
            f"A {name} of {size:g} dva is {size / px:.2g} pixels across at "
            f"fov={fov} dva and shape={shape}, which does not resolve the "
            f"{feature}. At least {_MIN_FEATURE_PX} pixels are required: "
            f"increase 'shape' or reduce 'fov'.")


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

    Rasterize a Landolt C at a given angular size, eccentricity, and gap
    orientation, and place it in a :py:class:`~pulse2percept.vision.Scene`.

    The C follows the standard proportions, all expressed in multiples of the
    gap width ``gap``: stroke width ``gap``, inner diameter ``3 * gap``, outer
    diameter ``5 * gap``. ``gap`` is therefore the critical feature size,
    which is what an acuity task varies; ``position`` moves the optotype
    through the visual field without changing that size.

    The image is binary (gray levels 0 and 1), not antialiased.

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
    x, y, (width, height) = _optotype_grid(shape, fov)

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
    _check_raster(gap, 'gap', 'opening', (width, height), x.shape)
    mask = _landolt_mask(x, y, gap, center, orientation)
    ink, paper = (0.0, 1.0) if polarity == 'dark' else (1.0, 0.0)
    img = np.where(mask, ink, paper).astype(np.float32)
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

    Rasterize a Tumbling E at a given angular size, eccentricity, and
    orientation, and place it in a :py:class:`~pulse2percept.vision.Scene`.

    The E follows the standard 5x5 construction, all expressed in multiples of
    the stroke width ``stroke``: overall width and height ``5 * stroke``, bars
    and the gaps between them one ``stroke`` each. ``stroke`` is therefore the
    critical feature size, which is what an acuity task varies; ``position``
    moves the optotype through the visual field without changing that size.

    The four cardinal orientations are the conventional Tumbling-E task,
    although any finite angle is accepted here.

    .. note::
       The Tumbling E and the Landolt C
       (:py:func:`~pulse2percept.stimuli.psychophysics.landolt_c`) are
       different optotypes measured with different tasks (bar direction vs.
       gap direction). Thresholds obtained with one are not numerically
       interchangeable with the other.

    The image is binary (gray levels 0 and 1), not antialiased.

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
    x, y, (width, height) = _optotype_grid(shape, fov)

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
    _check_raster(stroke, 'stroke', 'bars', (width, height), x.shape)
    mask = _tumbling_e_mask(x, y, stroke, center, orientation)
    ink, paper = (0.0, 1.0) if polarity == 'dark' else (1.0, 0.0)
    img = np.where(mask, ink, paper).astype(np.float32)
    metadata = {'generator': 'tumbling_e', 'stroke': stroke,
                'position': (float(center[0]), float(center[1])),
                'orientation': orientation, 'polarity': polarity,
                'fov': (width, height)}
    return Scene(ImageStimulus(img, metadata=metadata), fov=(width, height))
