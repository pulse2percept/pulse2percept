import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.utils import (center_image, shift_image, scale_image,
                                 trim_image)


def test_shift_image():
    img = np.zeros((5, 5), dtype=np.uint8)
    img[2, :] = 255
    # Top row:
    top = shift_image(img, 0, -2)
    npt.assert_almost_equal(top[0, :], 255)
    npt.assert_almost_equal(top[1:, :], 0)
    # Bottom row:
    bottom = shift_image(img, 0, 2)
    npt.assert_almost_equal(bottom[:4, :], 0)
    npt.assert_almost_equal(bottom[4, :], 255)
    # Bottom right pixel:
    bottom = shift_image(img, 4, 2)
    npt.assert_almost_equal(bottom[4, 4], 255)
    npt.assert_almost_equal(bottom[:4, :], 0)
    npt.assert_almost_equal(bottom[:, :4], 0)
    with pytest.raises(ValueError):
        shift_image(np.ones(10), 1, 1)
    with pytest.raises(ValueError):
        shift_image(np.ones((3, 4, 5, 6)), 1, 1)


def test_center_image():
    # Create a horizontal bar:
    img = np.zeros((5, 5), dtype=np.uint8)
    img[2, :] = 255
    # Center phosphene:
    npt.assert_almost_equal(img, center_image(img), decimal=3)
    npt.assert_almost_equal(img, center_image(shift_image(img, 0, 2)))
    with pytest.raises(ValueError):
        center_image(np.ones(10))
    with pytest.raises(ValueError):
        center_image(np.ones((3, 4, 5, 6)))


def test_scale_image():
    # Create a horizontal bar:
    img = np.zeros((5, 5), dtype=np.uint8)
    img[2, :] = 255
    # Scale phosphene:
    npt.assert_almost_equal(img, scale_image(img, 1))
    npt.assert_almost_equal(scale_image(img, 0.1).ravel()[12], 255)
    npt.assert_almost_equal(scale_image(img, 0.1).ravel()[:12], 0)
    npt.assert_almost_equal(scale_image(img, 0.1).ravel()[13:], 0)
    with pytest.raises(ValueError):
        scale_image(np.ones(10), 1)
    with pytest.raises(ValueError):
        scale_image(np.ones((3, 4, 5, 6)), 1)
    with pytest.raises(ValueError):
        scale_image(img, 0)


@pytest.mark.parametrize('n_channels', (1, 3))
def test_trim_image(n_channels):
    img = np.squeeze(np.zeros((13, 29, n_channels), dtype=float))
    shape = img.shape
    img[1:-1, 1:-1, ...] = 0.1
    img[2:-2, 2:-2, ...] = 0.2
    npt.assert_equal(trim_image(img).shape[:2], (shape[0] - 2, shape[1] - 2))
    npt.assert_equal(trim_image(img, tol=0.05).shape[:2],
                     (shape[0] - 2, shape[1] - 2))
    npt.assert_equal(trim_image(img, tol=0.1).shape[:2],
                     (shape[0] - 4, shape[1] - 4))
    npt.assert_equal(trim_image(img, tol=0.2).shape[:2], (1, 0))
    npt.assert_equal(trim_image(img, tol=0.1).shape[:2],
                     trim_image(trim_image(img), tol=0.1).shape[:2])
    trimmed, rows, cols = trim_image(img, return_coords=True)
    npt.assert_almost_equal(trimmed, trim_image(img))
    npt.assert_equal(rows, (1, 12))
    npt.assert_equal(cols, (1, 28))

    with pytest.raises(ValueError):
        trim_image(np.ones(10))
    with pytest.raises(ValueError):
        trim_image(np.ones((3, 4, 5, 6)))
    with pytest.raises(ValueError):
        trim_image(img, tol=-1)
