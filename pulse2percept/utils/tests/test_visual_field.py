import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.units import DimensionMismatchError, deg, dva
from pulse2percept.utils._visual_field import (_ray_span, meridian_angles,
                                               ring_radii, visible_band)


@pytest.mark.parametrize('r_max, expected', [
    (1, []), (1.25, [1.25]), (20, [1.25, 2.5, 5, 10, 20]),
    (79.9, [1.25, 2.5, 5, 10, 20, 40]), (80, [1.25, 2.5, 5, 10, 20, 40, 80]),
])
def test_default_rings_double_from_1_25_dva(r_max, expected):
    npt.assert_almost_equal(ring_radii(True, r_max), expected)


def test_rings_off_spacing_and_explicit():
    for off in (False, None):
        npt.assert_equal(ring_radii(off, 20).size, 0)
    npt.assert_almost_equal(ring_radii(15, 40), [15, 30])
    npt.assert_almost_equal(ring_radii(15 * dva, 40), [15, 30])
    # Explicit rings are drawn as given, sorted, even beyond the field:
    npt.assert_almost_equal(ring_radii([20, 2.5, 50], 10), [2.5, 20, 50])
    # Automatic rings (only) respect a lower bound:
    npt.assert_almost_equal(ring_radii(True, 20, r_min=3), [5, 10, 20])
    npt.assert_almost_equal(ring_radii(5, 20, r_min=5), [10, 15, 20])
    npt.assert_almost_equal(ring_radii([1, 2], 20, r_min=5), [1, 2])


@pytest.mark.parametrize('rings', [0, -1, np.nan, np.inf, [], [1, -1],
                                   [np.inf]])
def test_bad_rings(rings):
    with pytest.raises(ValueError):
        ring_radii(rings, 20)


def test_meridian_angles():
    for off in (False, None):
        npt.assert_equal(meridian_angles(off).size, 0)
    npt.assert_almost_equal(meridian_angles(True), np.arange(0, 360, 45))
    npt.assert_almost_equal(meridian_angles(30), np.arange(0, 360, 30))
    npt.assert_almost_equal(meridian_angles(90 * deg), [0, 90, 180, 270])
    # Explicit angles wrap into [0, 360), sorted and deduplicated:
    npt.assert_almost_equal(meridian_angles([90, -90, 360, 0]), [0, 90, 270])
    # Polar angle is geometric deg, not dva:
    with pytest.raises(DimensionMismatchError):
        meridian_angles(45 * dva)


@pytest.mark.parametrize('meridians', [0, -45, np.nan, np.inf, [],
                                       [0, np.nan]])
def test_bad_meridians(meridians):
    with pytest.raises(ValueError):
        meridian_angles(meridians)


def test_ray_span():
    extent = (-15, 5, -4, 10)
    for angle, far in ((0, 5), (90, 10), (180, 15), (270, 4)):
        npt.assert_almost_equal(_ray_span((0, 0), angle, extent), (0, far))
    # From outside, a ray enters and leaves, or misses entirely:
    outside = (-15, -3, -2, 10)
    npt.assert_almost_equal(_ray_span((0, 0), 180, outside), (3, 15))
    npt.assert_equal(_ray_span((0, 0), 0, outside), None)
    npt.assert_equal(_ray_span((0, 0), 90, outside), None)


def test_visible_band():
    # Fovea inside: up to the nearest edge
    npt.assert_almost_equal(visible_band((0, 0), (-15, 5, -4, 10)), (0, 4))
    # Fovea outside: the eccentricities the field covers
    npt.assert_almost_equal(visible_band((0, 0), (-15, -3, -2, 10)),
                            (3, np.hypot(15, 10)))
