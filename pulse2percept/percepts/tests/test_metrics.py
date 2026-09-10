import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.percepts import Percept
from pulse2percept.percepts.metrics import (FrameMetrics, PerceptMetrics,
                                            measure_percept)
from pulse2percept.topography import Grid2D

# Full width at half maximum of a Gaussian with unit standard deviation:
FWHM = 2 * np.sqrt(2 * np.log(2))


def gaussian(grid, x0=0, y0=0, sigma_x=1, sigma_y=None, amplitude=1,
             angle=0):
    """A Gaussian blob sampled on ``grid``, as a one-frame (Y, X, 1) array

    ``angle`` rotates the blob counterclockwise (radians) about ``(x0, y0)``.
    """
    sigma_y = sigma_x if sigma_y is None else sigma_y
    dx, dy = grid.x - x0, grid.y - y0
    # Rotate the sampling coordinates into the blob's own frame:
    u = dx * np.cos(angle) + dy * np.sin(angle)
    v = -dx * np.sin(angle) + dy * np.cos(angle)
    frame = amplitude * np.exp(-0.5 * ((u / sigma_x) ** 2 +
                                       (v / sigma_y) ** 2))
    return frame[..., np.newaxis]


def test_measure_percept_centroid():
    # A blob away from the origin in both x and y, on a field whose x and y
    # ranges differ, so a transposed or flipped axis cannot go unnoticed:
    grid = Grid2D((-8, 8), (-4, 4), step=0.05)
    percept = Percept(gaussian(grid, x0=2.5, y0=-1.5, sigma_x=0.6), space=grid)
    metrics = measure_percept(percept)
    npt.assert_almost_equal(metrics.peak.centroid, (2.5, -1.5), decimal=2)
    npt.assert_equal(metrics.centroid.shape, (1, 2))
    npt.assert_almost_equal(metrics.centroid[0], (2.5, -1.5), decimal=2)


def test_measure_percept_circular_gaussian():
    grid = Grid2D((-5, 5), (-5, 5), step=0.05)
    sigma = 1.2
    percept = Percept(gaussian(grid, sigma_x=sigma), space=grid)
    metrics = measure_percept(percept, threshold=0.5)
    # At half maximum, the equivalent-circle diameter approximates the FWHM:
    npt.assert_allclose(metrics.peak.diameter, FWHM * sigma, rtol=0.02)
    npt.assert_allclose(metrics.peak.area, np.pi * (FWHM * sigma / 2) ** 2,
                        rtol=0.03)
    npt.assert_allclose(metrics.peak.elongation, 1, rtol=0.02)
    npt.assert_allclose(metrics.peak.major_axis, metrics.peak.minor_axis,
                        rtol=0.02)
    # Geometry is measured on the support alone, so the second-moment axes of
    # a sampled disk approach its equivalent-circle diameter:
    npt.assert_allclose(metrics.peak.major_axis, metrics.peak.diameter,
                        rtol=0.02)
    npt.assert_allclose(metrics.peak.minor_axis, metrics.peak.diameter,
                        rtol=0.02)
    npt.assert_equal(metrics.peak.n_components, 1)
    npt.assert_equal(metrics.peak.touches_edge, False)
    npt.assert_almost_equal(metrics.peak.max_brightness, 1)


def test_measure_percept_threshold():
    grid = Grid2D((-5, 5), (-5, 5), step=0.05)
    sigma = 1.2
    percept = Percept(gaussian(grid, sigma_x=sigma), space=grid)
    # A Gaussian falls to `threshold` of its peak at a radius of
    # sigma * sqrt(-2 * log(threshold)):
    for threshold in (np.exp(-0.5), 0.25, 0.8):
        metrics = measure_percept(percept, threshold=threshold)
        npt.assert_equal(metrics.threshold, threshold)
        npt.assert_allclose(metrics.peak.diameter,
                            2 * sigma * np.sqrt(-2 * np.log(threshold)),
                            rtol=0.02)
    # exp(-0.5) puts the support edge exactly one standard deviation out:
    at_sigma = measure_percept(percept, threshold=np.exp(-0.5)).peak
    npt.assert_allclose(at_sigma.diameter, 2 * sigma, rtol=0.02)
    # A stricter threshold keeps strictly less of the same frame:
    half = measure_percept(percept, threshold=0.5).peak
    npt.assert_equal(at_sigma.area < half.area, True)
    npt.assert_equal(measure_percept(percept).threshold, 0.5)


def test_measure_percept_anisotropic_pixels():
    # Pixels twice as wide as they are tall: this fails if (dx, dy) collapses
    # to a single step size, or if the within-cell variance is dropped.
    grid = Grid2D((-1, 1), (-1, 1), step=(0.5, 0.25))
    dx, dy = 0.5, 0.25
    npt.assert_almost_equal(np.diff(grid.x[0, :]), dx)
    npt.assert_almost_equal(np.abs(np.diff(grid.y[:, 0])), dy)
    # A support of exactly one pixel, whose cell is its entire extent:
    frame = np.zeros(grid.shape)
    frame[3, 2] = 1.0
    metrics = measure_percept(Percept(frame[..., np.newaxis], space=grid)).peak
    npt.assert_almost_equal(metrics.area, dx * dy)
    npt.assert_almost_equal(metrics.diameter, 2 * np.sqrt(dx * dy / np.pi))
    npt.assert_equal(np.isfinite([metrics.major_axis, metrics.minor_axis]),
                     [True, True])
    npt.assert_equal(min(metrics.major_axis, metrics.minor_axis) > 0, True)
    # A uniform cell has variance (side ** 2) / 12 along each side:
    npt.assert_almost_equal(metrics.major_axis, 4 * np.sqrt(dx ** 2 / 12))
    npt.assert_almost_equal(metrics.minor_axis, 4 * np.sqrt(dy ** 2 / 12))
    npt.assert_almost_equal(metrics.elongation, dx / dy)
    npt.assert_almost_equal(metrics.centroid, (grid.x[3, 2], grid.y[3, 2]))
    npt.assert_almost_equal(metrics.integrated_brightness, dx * dy)


def test_measure_percept_anisotropic_gaussian():
    grid = Grid2D((-6, 6), (-6, 6), step=0.05)
    wide = measure_percept(Percept(gaussian(grid, sigma_x=1.5, sigma_y=0.5),
                                   space=grid)).peak
    tall = measure_percept(Percept(gaussian(grid, sigma_x=0.5, sigma_y=1.5),
                                   space=grid)).peak
    npt.assert_equal(wide.major_axis > wide.minor_axis, True)
    npt.assert_allclose(wide.elongation, 3, rtol=0.03)
    # Rotating the blob by 90 deg must not change its axis lengths:
    npt.assert_allclose(tall.major_axis, wide.major_axis, rtol=1e-6)
    npt.assert_allclose(tall.minor_axis, wide.minor_axis, rtol=1e-6)


def test_measure_percept_rotated_gaussian():
    # A blob at 90 deg leaves the off-diagonal moment zero; these angles do
    # not, so they are what actually exercises `cov_xy`:
    grid = Grid2D((-6, 6), (-6, 6), step=0.05)
    kwargs = dict(sigma_x=1.5, sigma_y=0.5)
    upright = measure_percept(Percept(gaussian(grid, **kwargs),
                                      space=grid)).peak
    for angle in (np.pi / 6, np.pi / 4, np.pi / 3):
        turned = measure_percept(Percept(gaussian(grid, angle=angle,
                                                  **kwargs), space=grid)).peak
        # Axis lengths and elongation are rotation invariant; only the
        # orientation of the ellipse changed:
        npt.assert_allclose(turned.major_axis, upright.major_axis, rtol=0.02)
        npt.assert_allclose(turned.minor_axis, upright.minor_axis, rtol=0.02)
        npt.assert_allclose(turned.elongation, upright.elongation, rtol=0.02)
        npt.assert_allclose(turned.area, upright.area, rtol=0.02)
        npt.assert_almost_equal(turned.centroid, (0, 0), decimal=2)
    # A 45 deg blob really is diagonal: its support spans x and y equally,
    # which a measurement that ignored `cov_xy` would report as circular.
    diagonal = measure_percept(Percept(gaussian(grid, angle=np.pi / 4,
                                                **kwargs), space=grid)).peak
    rows, cols = np.nonzero(gaussian(grid, angle=np.pi / 4,
                                     **kwargs)[..., 0] >= 0.5)
    npt.assert_allclose(np.ptp(cols), np.ptp(rows), rtol=0.05)
    npt.assert_allclose(diagonal.elongation, 3, rtol=0.03)


def test_measure_percept_amplitude_scaling():
    grid = Grid2D((-5, 5), (-5, 5), step=0.1)
    base = measure_percept(Percept(gaussian(grid), space=grid)).peak
    scaled = measure_percept(Percept(gaussian(grid, amplitude=7.5),
                                     space=grid)).peak
    # Brightness scales, but the threshold is relative to each frame's own
    # maximum, so the support and everything derived from it is untouched:
    npt.assert_allclose(scaled.max_brightness, 7.5 * base.max_brightness)
    npt.assert_allclose(scaled.integrated_brightness,
                        7.5 * base.integrated_brightness)
    npt.assert_allclose(scaled.area, base.area)
    npt.assert_allclose(scaled.diameter, base.diameter)
    npt.assert_allclose(scaled.elongation, base.elongation)
    npt.assert_allclose(scaled.centroid, base.centroid)


def test_measure_percept_resolution_invariance():
    # The same continuous Gaussian, sampled twice as finely:
    coarse = Grid2D((-5, 5), (-5, 5), step=0.1)
    fine = Grid2D((-5, 5), (-5, 5), step=0.05)
    args = dict(x0=1.0, y0=-0.75, sigma_x=1.1)
    coarse_metrics = measure_percept(Percept(gaussian(coarse, **args),
                                             space=coarse)).peak
    fine_metrics = measure_percept(Percept(gaussian(fine, **args),
                                           space=fine)).peak
    npt.assert_allclose(fine_metrics.integrated_brightness,
                        coarse_metrics.integrated_brightness, rtol=0.01)
    npt.assert_allclose(fine_metrics.area, coarse_metrics.area, rtol=0.02)
    npt.assert_allclose(fine_metrics.diameter, coarse_metrics.diameter,
                        rtol=0.01)
    npt.assert_allclose(fine_metrics.centroid, coarse_metrics.centroid,
                        atol=0.01)


def test_measure_percept_nonsquare_field():
    grid = Grid2D((-10, 2), (-1, 5), step=0.05)
    percept = Percept(gaussian(grid, x0=-4, y0=3, sigma_x=1.2, sigma_y=0.4),
                      space=grid)
    metrics = measure_percept(percept).peak
    npt.assert_almost_equal(metrics.centroid, (-4, 3), decimal=2)
    npt.assert_allclose(metrics.elongation, 3, rtol=0.03)
    npt.assert_allclose(metrics.diameter, FWHM * np.sqrt(1.2 * 0.4), rtol=0.02)


def test_measure_percept_empty():
    grid = Grid2D((-2, 2), (-2, 2), step=0.1)
    for frame in (np.zeros(grid.shape), -np.ones(grid.shape)):
        metrics = measure_percept(Percept(frame[..., np.newaxis], space=grid))
        npt.assert_equal(metrics.peak_frame, 0)
        npt.assert_equal(metrics.peak.integrated_brightness, 0)
        npt.assert_equal(metrics.peak.max_brightness, 0)
        npt.assert_equal(metrics.peak.area, 0)
        npt.assert_equal(metrics.peak.n_components, 0)
        npt.assert_equal(metrics.peak.touches_edge, False)
        # No phosphene means no place and no shape, not one at the origin:
        npt.assert_equal(np.isnan(metrics.peak.centroid), (True, True))
        for name in ('diameter', 'major_axis', 'minor_axis', 'elongation'):
            npt.assert_equal(np.isnan(getattr(metrics.peak, name)), True)


def test_measure_percept_touches_edge():
    grid = Grid2D((-5, 5), (-5, 5), step=0.1)
    centered = Percept(gaussian(grid, sigma_x=1), space=grid)
    npt.assert_equal(measure_percept(centered).peak.touches_edge, False)
    # Pushed far enough that the half-maximum support runs off the field:
    clipped = Percept(gaussian(grid, x0=4.8, sigma_x=1), space=grid)
    npt.assert_equal(measure_percept(clipped).peak.touches_edge, True)


def test_measure_percept_multiple_components():
    grid = Grid2D((-8, 8), (-8, 8), step=0.05)
    blob = gaussian(grid, x0=-4, sigma_x=0.5) + gaussian(grid, x0=4,
                                                         sigma_x=0.5)
    metrics = measure_percept(Percept(blob, space=grid)).peak
    npt.assert_equal(metrics.n_components, 2)
    # Both blobs are measured, not just the first one found:
    one = measure_percept(Percept(gaussian(grid, x0=-4, sigma_x=0.5),
                                  space=grid)).peak
    npt.assert_allclose(metrics.area, 2 * one.area, rtol=0.02)
    # The two are symmetric about x=0, so their combined centroid sits there:
    npt.assert_almost_equal(metrics.centroid[0], 0, decimal=6)
    npt.assert_equal(metrics.major_axis > one.major_axis, True)


def test_measure_percept_temporal():
    grid = Grid2D((-4, 4), (-4, 4), step=0.1)
    amplitudes = [0.5, 2.0, 1.0]
    frames = np.concatenate([gaussian(grid, amplitude=amp)
                             for amp in amplitudes], axis=-1)
    metrics = measure_percept(Percept(frames, space=grid, time=[0, 1, 2]))
    npt.assert_equal(len(metrics.frames), 3)
    npt.assert_equal(isinstance(metrics.frames[0], FrameMetrics), True)
    npt.assert_equal(isinstance(metrics, PerceptMetrics), True)
    npt.assert_allclose(metrics.max_brightness, amplitudes)
    integrated = metrics.integrated_brightness
    npt.assert_allclose(integrated / integrated[2], amplitudes)
    npt.assert_equal(metrics.peak_frame, 1)
    npt.assert_equal(metrics.peak, metrics.frames[1])
    # The relative threshold makes every frame the same size:
    npt.assert_allclose(metrics.diameter, [metrics.diameter[0]] * 3)
    for name, shape in [('area', (3,)), ('elongation', (3,)),
                        ('major_axis', (3,)), ('minor_axis', (3,)),
                        ('n_components', (3,)), ('touches_edge', (3,)),
                        ('centroid', (3, 2))]:
        npt.assert_equal(getattr(metrics, name).shape, shape)
    npt.assert_equal(metrics.n_components.dtype, np.dtype(int))
    npt.assert_equal(metrics.touches_edge.dtype, np.dtype(bool))


def test_measure_percept_peak_frame_ties():
    grid = Grid2D((-4, 4), (-4, 4), step=0.2)
    frames = np.concatenate([gaussian(grid)] * 3, axis=-1)
    npt.assert_equal(measure_percept(Percept(frames, space=grid)).peak_frame,
                     0)


def test_measure_percept_immutable():
    grid = Grid2D((-4, 4), (-4, 4), step=0.2)
    metrics = measure_percept(Percept(gaussian(grid), space=grid))
    with pytest.raises(Exception):
        metrics.frames[0].area = 1
    with pytest.raises(Exception):
        metrics.frames = ()


def test_measure_percept_invalid():
    grid = Grid2D((-4, 4), (-4, 4), step=0.2)
    # RGB percepts are display values, not perceived brightness:
    rgb = Percept(np.zeros(grid.shape + (3, 1)), space=grid)
    with pytest.raises(ValueError):
        measure_percept(rgb)
    # Without a Grid2D, 'xdva'/'ydva' are pixel indices, not dva:
    with pytest.raises(ValueError):
        measure_percept(Percept(np.zeros((10, 10, 1))))
    for threshold in (0, -0.5, 1.5, np.nan):
        with pytest.raises(ValueError):
            measure_percept(Percept(gaussian(grid), space=grid),
                            threshold=threshold)
    for bad in (np.nan, np.inf, -np.inf):
        frame = gaussian(grid)
        frame[0, 0, 0] = bad
        with pytest.raises(ValueError):
            measure_percept(Percept(frame, space=grid))
    # A single-pixel axis says nothing about how much field a pixel covers:
    with pytest.raises(ValueError):
        measure_percept(Percept(np.ones((5, 1, 1)),
                                space=Grid2D((0, 0), (-2, 2), step=1)))


def test_Percept_measure():
    grid = Grid2D((-5, 5), (-5, 5), step=0.1)
    percept = Percept(gaussian(grid, x0=1, y0=-2, sigma_x=1.3), space=grid)
    metrics = percept.measure()
    npt.assert_equal(isinstance(metrics, PerceptMetrics), True)
    npt.assert_equal(metrics.threshold, 0.5)
    # The method is a thin delegation, so it must agree with the function:
    npt.assert_equal(metrics, measure_percept(percept))
    npt.assert_equal(metrics is measure_percept(percept), False)


def test_Percept_measure_threshold():
    grid = Grid2D((-5, 5), (-5, 5), step=0.1)
    percept = Percept(gaussian(grid, sigma_x=1.3), space=grid)
    strict = percept.measure(threshold=0.25)
    npt.assert_equal(strict.threshold, 0.25)
    npt.assert_equal(strict, measure_percept(percept, threshold=0.25))
    # A lower threshold keeps more of the frame than the default does:
    npt.assert_equal(strict.peak.area > percept.measure().peak.area, True)


def test_Percept_measure_not_cached():
    grid = Grid2D((-5, 5), (-5, 5), step=0.1)
    percept = Percept(gaussian(grid, sigma_x=1.3), space=grid)
    first = percept.measure()
    # Brighten in place; a cached result would not reflect this:
    percept.data[:] *= 2
    second = percept.measure()
    npt.assert_equal(second is first, False)
    npt.assert_allclose(second.integrated_brightness,
                        2 * first.integrated_brightness)
    npt.assert_allclose(second.max_brightness, 2 * first.max_brightness)
    # The threshold is relative, so the geometry is unchanged:
    npt.assert_allclose(second.area, first.area)
    npt.assert_allclose(second.diameter, first.diameter)
    npt.assert_allclose(second.centroid, first.centroid)


def test_Percept_measure_rejects_rgb():
    # One smoke check that validation is reached through the method; the full
    # matrix is tested against `measure_percept` itself:
    grid = Grid2D((-2, 2), (-2, 2), step=0.5)
    with pytest.raises(ValueError):
        Percept(np.zeros(grid.shape + (3, 1)), space=grid).measure()
