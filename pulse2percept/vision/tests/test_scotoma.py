import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.units import dva, ms
from pulse2percept.units import DimensionMismatchError
from pulse2percept.vision import Scotoma


def test_Scotoma_circle():
    scotoma = Scotoma.circle(5 * dva)
    # Complete loss inside, intact vision outside, and the rim counts as lost:
    npt.assert_almost_equal(scotoma(0, 0), 1)
    npt.assert_almost_equal(scotoma(5, 0), 1)
    npt.assert_almost_equal(scotoma(0, -5), 1)
    npt.assert_almost_equal(scotoma(5.001, 0), 0)
    npt.assert_almost_equal(scotoma(4, 4), 0)
    # Symmetric about the fovea, and it is a circle rather than a square:
    npt.assert_almost_equal(scotoma(3, 4), 1)
    npt.assert_almost_equal(scotoma(-3, -4), 1)


def test_Scotoma_circle_off_center():
    scotoma = Scotoma.circle(2, center=(6, -3))
    npt.assert_almost_equal(scotoma(6, -3), 1)
    npt.assert_almost_equal(scotoma(0, 0), 0)
    npt.assert_almost_equal(scotoma(6, -1), 1)
    npt.assert_almost_equal(scotoma(6, 0), 0)


def test_Scotoma_ellipse():
    scotoma = Scotoma.ellipse(6 * dva, 2 * dva)
    npt.assert_almost_equal(scotoma(6, 0), 1)
    npt.assert_almost_equal(scotoma(0, 2), 1)
    # A circle of radius 6 would have swallowed this, an ellipse does not:
    npt.assert_almost_equal(scotoma(0, 3), 0)
    npt.assert_almost_equal(scotoma(4, 1.5), 0)


def test_Scotoma_broadcasts_over_a_grid():
    scotoma = Scotoma.circle(5)
    x, y = np.meshgrid(np.linspace(-10, 10, 21), np.linspace(-10, 10, 21))
    loss = scotoma(x, y)
    npt.assert_equal(loss.shape, x.shape)
    # The mask is the disk it says it is:
    npt.assert_array_equal(loss, (x ** 2 + y ** 2 <= 25).astype(float))
    # A scalar and a grid agree:
    npt.assert_almost_equal(scotoma(x, y)[10, 10], scotoma(0, 0))


def test_Scotoma_takes_a_callable():
    """The seam a measured or graded defect arrives through"""
    scotoma = Scotoma(lambda x, y: np.clip(np.abs(x) / 10, 0, 1),
                      name='graded')
    npt.assert_almost_equal(scotoma(0, 0), 0)
    npt.assert_almost_equal(scotoma(5, 0), 0.5)
    npt.assert_almost_equal(scotoma(20, 0), 1)
    npt.assert_equal('graded' in repr(scotoma), True)


def test_Scotoma_rejects_a_mask_that_is_not_a_loss_fraction():
    """0 is intact and 1 is total, so there is nothing outside [0, 1]"""
    for bad in (lambda x, y: x * 0 + 1.5, lambda x, y: x * 0 - 0.1,
                lambda x, y: x * 0 + np.nan):
        with pytest.raises(ValueError):
            Scotoma(bad)(0, 0)
    with pytest.raises(TypeError):
        Scotoma(0.5)


@pytest.mark.parametrize('radius', [0, -3, np.inf, np.nan])
def test_Scotoma_rejects_a_radius_that_is_not_a_radius(radius):
    with pytest.raises(ValueError):
        Scotoma.circle(radius)
    with pytest.raises(ValueError):
        Scotoma.ellipse(radius, 2)


def test_Scotoma_is_unit_aware():
    """dva at the boundary, plain numbers behind it"""
    npt.assert_almost_equal(Scotoma.circle(5 * dva)(3 * dva, 4 * dva),
                            Scotoma.circle(5)(3, 4))
    npt.assert_almost_equal(Scotoma.circle(5, center=(2, 0) * dva)(7, 0), 1)
    with pytest.raises(DimensionMismatchError):
        Scotoma.circle(5 * ms)
    with pytest.raises(DimensionMismatchError):
        Scotoma.circle(5)(3 * ms, 0)


def test_Scotoma_does_not_move_with_gaze():
    """A scotoma is eye-centered: it is asked about the visual field only

    Nothing in its API takes a gaze or a scene coordinate, which is what keeps
    it fixed relative to the fovea (and therefore to the implant) while the
    scene moves past both.
    """
    scotoma = Scotoma.circle(5)
    npt.assert_equal(scotoma(0, 0), scotoma(0, 0))
    with pytest.raises(TypeError):
        scotoma(0, 0, gaze=(5, 0))


@pytest.mark.parametrize('center', [(np.nan, 0), (0, np.inf), (-np.inf, 2)])
def test_Scotoma_rejects_a_non_finite_center(center):
    """A NaN center compares false everywhere, so it would read as intact

    The quietest possible failure: no error, no scotoma, an entirely healthy
    visual field.
    """
    with pytest.raises(ValueError):
        Scotoma.circle(5, center=center)
    with pytest.raises(ValueError):
        Scotoma.ellipse(5, 2, center=center)


@pytest.mark.parametrize('coord', [np.nan, np.inf, -np.inf])
def test_Scotoma_rejects_non_finite_coordinates(coord):
    """Same trap on the way in: the mask would turn NaN into 0 loss"""
    scotoma = Scotoma.circle(5)
    with pytest.raises(ValueError):
        scotoma(coord, 0)
    with pytest.raises(ValueError):
        scotoma(0, coord)
    with pytest.raises(ValueError):
        scotoma(np.array([0.0, coord]), 0)
