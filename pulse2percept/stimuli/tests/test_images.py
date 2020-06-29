import os
import numpy as np
import numpy.testing as npt
import pytest

from skimage.io import imsave

from pulse2percept.stimuli import ImageStimulus, LogoBVL, LogoUCSB


def test_ImageStimulus():
    # Create a dummy image:
    fname = 'test.png'
    shape = (25, 37)
    ndarray = np.random.rand(*shape)
    imsave(fname, (255 * ndarray).astype(np.uint8))

    # Make sure ImageStimulus loaded is identical to dummy image:
    stim = ImageStimulus(fname)
    npt.assert_equal(stim.shape, (np.prod(shape), 1))
    npt.assert_almost_equal(stim.data, ndarray.reshape((-1, 1)), decimal=2)
    npt.assert_equal(stim.metadata['source'], fname)
    npt.assert_equal(stim.metadata['source_shape'], shape)
    npt.assert_equal(stim.time, None)
    npt.assert_equal(stim.electrodes, np.arange(np.prod(shape)))
    os.remove(fname)

    # Resize the dummy image:
    ndarray = np.ones(shape)
    imsave(fname, (255 * ndarray).astype(np.uint8))
    resize = (12, 18)
    stim = ImageStimulus(fname, resize=resize)
    npt.assert_equal(stim.shape, (np.prod(resize), 1))
    npt.assert_almost_equal(stim.data, np.ones((np.prod(resize), 1)),
                            decimal=2)
    npt.assert_equal(stim.metadata['source'], fname)
    npt.assert_equal(stim.metadata['source_shape'], shape)
    os.remove(fname)


def test_ImageStimulus_invert():
    # Create a dummy image:
    fname = 'test.png'
    shape = (25, 37)
    gray = 1 / 255.0
    ndarray = np.ones(shape) * gray
    imsave(fname, (255 * ndarray).astype(np.uint8))
    # Gray levels are between 0 and 1, and can be inverted:
    stim = ImageStimulus(fname)
    npt.assert_almost_equal(stim.data, gray)
    npt.assert_almost_equal(stim.invert().data, 1 - gray)
    # Inverting does not change the original object:
    npt.assert_almost_equal(stim.data, gray)
    os.remove(fname)

    # Cannot invert RGB:
    fname = 'test.png'
    imsave(fname, (255 * np.ones((25, 37, 4))).astype(np.uint8))
    with pytest.raises(ValueError):
        ImageStimulus(fname).invert()


def test_ImageStimulus_rgb2gray():
    # Create a dummy image:
    fname = 'test.png'
    shape = (25, 37, 3)
    gray = 1 / 255.0
    ndarray = np.ones(shape) * gray
    imsave(fname, (255 * ndarray).astype(np.uint8))
    # Gray levels are between 0 and 1, and can be inverted:
    stim_rgb = ImageStimulus(fname)
    stim_gray = stim_rgb.rgb2gray()
    npt.assert_almost_equal(stim_gray.data, gray)
    npt.assert_equal(stim_gray.img_shape, shape[:2])
    # Original stim unchanged:
    npt.assert_equal(stim_rgb.img_shape, shape)
    os.remove(fname)


def test_ImageStimulus_threshold():
    # Create a dummy image:
    fname = 'test.png'
    shape = (25, 37, 3)
    gray = 129 / 255.0
    ndarray = np.ones(shape) * gray
    imsave(fname, (255 * ndarray).astype(np.uint8))
    # Gray levels are between 0 and 1, and can be inverted:
    stim = ImageStimulus(fname).rgb2gray()
    npt.assert_almost_equal(stim.data, gray)
    npt.assert_equal(stim.img_shape, shape[:2])
    os.remove(fname)


def test_LogoBVL():
    logo = LogoBVL()
    npt.assert_equal(logo.img_shape, (576, 720, 4))
    npt.assert_almost_equal(logo.data.min(), 0)
    npt.assert_almost_equal(logo.data.max(), 1)


def test_LogoUCSB():
    logo = LogoUCSB()
    npt.assert_equal(logo.img_shape, (324, 727))
    npt.assert_almost_equal(logo.data.min(), 0)
    npt.assert_almost_equal(logo.data.max(), 1)
