""":py:class:`~pulse2percept.percepts.metrics.FrameMetrics`,
   :py:class:`~pulse2percept.percepts.metrics.PerceptMetrics`,
   :py:func:`~pulse2percept.percepts.metrics.measure_percept`

Measurements of the phosphenes in a percept. Not imported by
:py:mod:`pulse2percept.percepts` itself, so that predicting a percept never
loads them; :py:meth:`~pulse2percept.percepts.Percept.measure` pulls them in
on demand.
"""
from dataclasses import dataclass

import numpy as np
from skimage.measure import label


def _pixel_spacing(coords, name):
    """Spacing (dva) between neighboring pixel centers along one axis"""
    coords = np.asarray(coords, dtype=np.float64).ravel()
    if coords.size < 2:
        raise ValueError(f"Phosphene measurements need at least two pixels "
                         f"along '{name}' to know how much of the visual "
                         f"field a pixel covers.")
    # Grid2D lays its axes down with `linspace`, so the end points give the
    # exact spacing even when the stored coordinates are float32:
    return abs(coords[-1] - coords[0]) / (coords.size - 1)


@dataclass(frozen=True)
class FrameMetrics:
    """Phosphene measurements for a single frame of a brightness percept

    Produced by :py:func:`measure_percept` (usually through
    :py:meth:`~pulse2percept.percepts.Percept.measure`); not meant to be
    constructed directly.

    All measurements consider only positive brightness: the frame is clipped
    at zero first, so negative model output does not count as phosphene
    brightness. "Support" means the pixels at or above ``threshold`` times the
    frame's own positive maximum (half maximum by default), and every shape
    measurement below describes the *entire* support, even if it falls into
    several blobs.

    Geometry is measured on the support as a set of pixels, not weighted by
    the brightness inside it: brightness enters only through where the
    threshold falls. So for a circular phosphene the support is a disk, and
    :py:attr:`major_axis`, :py:attr:`minor_axis` and :py:attr:`diameter` all
    agree.

    A frame with no positive brightness has no phosphene: its brightness,
    area, and component count are zero, and the quantities that would describe
    the position or shape of a phosphene are ``NaN`` rather than a phantom
    phosphene at the origin.

    .. versionadded:: 0.11.0

    Attributes
    ----------
    integrated_brightness : float
        Positive brightness integrated over the visual field, in brightness
        units x dva^2. This is a pixel sum scaled by the area of a pixel, so
        it approximates an integral rather than counting pixels: sampling the
        same percept more finely leaves it approximately unchanged, up to
        discretization.
    max_brightness : float
        Largest positive brightness in the frame, in arbitrary brightness
        units.
    area : float
        Area of the support (dva^2).
    centroid : (float, float)
        Center ``(x, y)`` of the support, in dva: the mean position of its
        pixels.
    diameter : float
        Diameter (dva) of the circle with the same area as the support,
        ``2 * sqrt(area / pi)``. For a sufficiently sampled circular Gaussian
        and the default half-maximum threshold, this approximates its FWHM;
        the support is a set of whole pixels, so the agreement is limited by
        how finely the grid samples the phosphene.
    major_axis, minor_axis : float
        Axis lengths (dva) of the ellipse with the same second moments as the
        support, ``4 * sqrt(eigenvalue)``. For a uniformly filled ellipse this
        recovers its actual axes, so a circular support gives
        ``major_axis == minor_axis == diameter``.
    elongation : float
        ``major_axis / minor_axis``; 1 for a circular phosphene.
    n_components : int
        Number of connected components in the support, using full 2D
        connectivity so diagonally adjacent pixels count as connected.
        Descriptive only: the measurements above still describe the combined
        support.
    touches_edge : bool
        Whether the support reaches the first or last row or column. If True,
        the percept may extend past the simulated visual field, and the
        measurements describe only the visible part of it.

    """
    integrated_brightness: float
    max_brightness: float
    area: float
    centroid: tuple[float, float]
    diameter: float
    major_axis: float
    minor_axis: float
    elongation: float
    n_components: int
    touches_edge: bool


# What a frame without any positive brightness measures. Shared because
# ``FrameMetrics`` is immutable:
_NO_PHOSPHENE = FrameMetrics(integrated_brightness=0.0, max_brightness=0.0,
                             area=0.0, centroid=(np.nan, np.nan),
                             diameter=np.nan, major_axis=np.nan,
                             minor_axis=np.nan, elongation=np.nan,
                             n_components=0, touches_edge=False)


@dataclass(frozen=True)
class PerceptMetrics:
    """Phosphene measurements for every frame of a brightness percept

    Produced by :py:func:`measure_percept` (usually through
    :py:meth:`~pulse2percept.percepts.Percept.measure`); not meant to be
    constructed directly. Frames are measured independently of one another;
    :py:attr:`~pulse2percept.percepts.Percept.time` remains the source of
    timing information.

    Each measurement of :py:class:`FrameMetrics` is also available as an array
    over frames, so ``metrics.diameter`` is the diameter of every frame and
    ``metrics.peak.diameter`` is the diameter of the brightest one.

    .. versionadded:: 0.11.0

    Attributes
    ----------
    frames : tuple of FrameMetrics
        One measurement per frame, in the order the frames are stored.
    threshold : float
        Fraction of each frame's own positive maximum that defined its
        support, recorded so a result says what was measured.

    """
    frames: tuple[FrameMetrics, ...]
    threshold: float

    def __repr__(self):
        return (f"PerceptMetrics(n_frames={len(self.frames)}, "
                f"threshold={self.threshold:g}, "
                f"peak_frame={self.peak_frame})")

    def _column(self, name, dtype=float):
        return np.array([getattr(frame, name) for frame in self.frames],
                        dtype=dtype)

    @property
    def peak_frame(self):
        """Index of the frame with the largest ``integrated_brightness``

        Ties go to the earliest frame, as with ``np.argmax``.
        """
        return int(np.argmax(self.integrated_brightness))

    @property
    def peak(self):
        """:py:class:`FrameMetrics` of the frame at :py:attr:`peak_frame`"""
        return self.frames[self.peak_frame]

    @property
    def integrated_brightness(self):
        """Integrated positive brightness of each frame, (T,)"""
        return self._column('integrated_brightness')

    @property
    def max_brightness(self):
        """Largest positive brightness of each frame, (T,)"""
        return self._column('max_brightness')

    @property
    def area(self):
        """Support area (dva^2) of each frame, (T,)"""
        return self._column('area')

    @property
    def diameter(self):
        """Equivalent-circle diameter (dva) of each frame, (T,)"""
        return self._column('diameter')

    @property
    def centroid(self):
        """Support ``(x, y)`` center (dva) of each frame, (T, 2)"""
        return np.array([frame.centroid for frame in self.frames],
                        dtype=float).reshape((len(self.frames), 2))

    @property
    def major_axis(self):
        """Equivalent-ellipse major axis (dva) of each frame, (T,)"""
        return self._column('major_axis')

    @property
    def minor_axis(self):
        """Equivalent-ellipse minor axis (dva) of each frame, (T,)"""
        return self._column('minor_axis')

    @property
    def elongation(self):
        """Axis ratio of each frame, (T,)"""
        return self._column('elongation')

    @property
    def n_components(self):
        """Number of connected support components in each frame, (T,)"""
        return self._column('n_components', dtype=int)

    @property
    def touches_edge(self):
        """Whether each frame's support reaches the field border, (T,)"""
        return self._column('touches_edge', dtype=bool)


def _measure_frame(frame, x, y, dx, dy, threshold):
    """Measure one (Y, X) frame on the 1D column/row coordinates ``x``/``y``"""
    frame = np.asarray(frame, dtype=np.float64)
    # Before clipping, which would turn a -inf into an innocent-looking 0:
    if not np.all(np.isfinite(frame)):
        raise ValueError("Percept data must be finite to be measured.")
    positive = np.clip(frame, 0, None)
    peak = positive.max()
    pixel_area = dx * dy
    if peak <= 0:
        return _NO_PHOSPHENE
    # The threshold is relative to this frame's own maximum, so scaling a
    # frame's brightness leaves its support -- and every shape measure -- put:
    support = positive >= threshold * peak
    n_rows, n_cols = support.shape
    # Only the support pixels carry coordinates, so index the 1D axes rather
    # than building two full-field coordinate images:
    rows, cols = np.nonzero(support)
    xs, ys = x[cols], y[rows]
    area = rows.size * pixel_area
    cx, cy = xs.mean(), ys.mean()
    off_x, off_y = xs - cx, ys - cy
    # Pixels are cells, not point samples: without the variance of a uniform
    # cell added in, a support only a pixel or two across measures zero width.
    cov_xx = (off_x ** 2).mean() + dx ** 2 / 12
    cov_yy = (off_y ** 2).mean() + dy ** 2 / 12
    cov_xy = (off_x * off_y).mean()
    # `eigvalsh` returns the two eigenvalues in ascending order; clip because
    # roundoff can push a near-degenerate one slightly negative:
    eigvals = np.clip(np.linalg.eigvalsh([[cov_xx, cov_xy],
                                          [cov_xy, cov_yy]]), 0, None)
    minor_axis = 4 * np.sqrt(eigvals[0])
    major_axis = 4 * np.sqrt(eigvals[1])
    touches_edge = bool(rows.min() == 0 or rows.max() == n_rows - 1 or
                        cols.min() == 0 or cols.max() == n_cols - 1)
    return FrameMetrics(
        integrated_brightness=float(positive.sum() * pixel_area),
        max_brightness=float(peak),
        area=float(area),
        centroid=(float(cx), float(cy)),
        diameter=float(2 * np.sqrt(area / np.pi)),
        major_axis=float(major_axis),
        minor_axis=float(minor_axis),
        elongation=float(major_axis / minor_axis),
        n_components=int(label(support, connectivity=2).max()),
        touches_edge=touches_edge)


def measure_percept(percept, threshold=0.5):
    """Measure the phosphenes in a brightness percept

    Measures every frame of ``percept`` independently and returns the results
    as a :py:class:`PerceptMetrics`. See :py:class:`FrameMetrics` for what is
    measured and in what units.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    percept : :py:class:`~pulse2percept.percepts.Percept`
        A (Y, X, T) percept of perceived brightness built on a real
        :py:class:`~pulse2percept.topography.Grid2D`. Positions and sizes are
        reported in degrees of visual angle, which a percept built without a
        grid does not have (its coordinates are pixel indices).
    threshold : float, optional
        Fraction of a frame's own positive maximum at or above which a pixel
        counts as part of the phosphene. Must lie in (0, 1]. The default of
        0.5 makes :py:attr:`FrameMetrics.diameter` approximately the full
        width at half maximum of a sufficiently sampled circular Gaussian
        phosphene. More generally, a Gaussian of standard deviation ``sigma``
        has support diameter ``2 * sigma * sqrt(-2 * log(threshold))``.

    Returns
    -------
    metrics : :py:class:`PerceptMetrics`

    Notes
    -----
    These are measurements of the model's output image. They describe how big
    and how bright a modeled phosphene is, not how well an observer could
    resolve or tell apart what they saw.

    """
    if getattr(percept, 'is_rgb', False):
        raise ValueError("Phosphene measurements are defined on model-"
                         "produced perceived brightness, not on the RGB "
                         "display values this percept holds. Measure the "
                         "brightness percept a model produced instead.")
    if not getattr(percept, '_has_space', False):
        raise ValueError("Phosphene measurements are reported in degrees of "
                         "visual angle, but this percept was built without a "
                         "Grid2D, so its 'xdva'/'ydva' are pixel indices. "
                         "Pass 'space' when building the percept.")
    threshold = float(threshold)
    if not 0 < threshold <= 1:
        raise ValueError(f"'threshold' is a fraction of a frame's own maximum "
                         f"brightness and must lie in (0, 1], not "
                         f"{threshold}.")
    # Left in the dtype it was stored in; `_measure_frame` promotes, and
    # checks, one frame at a time, so a long percept is never duplicated:
    data = percept.data
    dx = _pixel_spacing(percept.xdva, 'xdva')
    dy = _pixel_spacing(percept.ydva, 'ydva')
    # Row 0 of a percept is drawn at the *top* of the visual field, so the row
    # coordinates run the other way from the stored (ascending) 'ydva':
    x = np.asarray(percept.xdva, dtype=np.float64)
    y = np.asarray(percept.ydva, dtype=np.float64)[::-1]
    frames = tuple(_measure_frame(data[..., t], x, y, dx, dy, threshold)
                   for t in range(data.shape[-1]))
    return PerceptMetrics(frames=frames, threshold=threshold)
