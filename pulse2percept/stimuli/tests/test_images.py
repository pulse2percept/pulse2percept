import os
import numpy as np
import numpy.testing as npt
import pytest

from skimage.io import imsave

from pulse2percept.stimuli import (ImageStimulus, LogoBVL, LogoUCSB,
                                   SnellenChart)


def create_dummy_img(fname, shape, mode, gray=1.0, return_data=False):
    if mode == 'ones':
        ndarray = np.ones(shape) * gray
    elif mode == 'zeros':
        ndarray = np.zeros(shape)
    elif mode == 'rand':
        ndarray = np.random.rand(*shape) * gray
    elif mode == 'custom':
        ndarray = shape
    imsave(fname, (255 * ndarray).astype(np.uint8))
    if return_data:
        return ndarray


def test_ImageStimulus():
    # Create a dummy image:
    fname = 'test.png'
    shape = (25, 37, 4)
    ndarray = create_dummy_img(fname, shape, 'rand', return_data=True)

    # Make sure ImageStimulus loaded is identical to dummy image:
    stim = ImageStimulus(fname)
    npt.assert_equal(stim.shape, (np.prod(shape), 1))
    npt.assert_almost_equal(stim.data, ndarray.reshape((-1, 1)), decimal=2)
    npt.assert_equal(stim.metadata['source'], fname)
    npt.assert_equal(stim.metadata['source_shape'], shape)
    npt.assert_equal(stim.time, None)
    npt.assert_equal(stim.electrodes, np.arange(np.prod(shape)))
    os.remove(fname)


def test_ImageStimulus_invert():
    # Create a dummy image:
    fname = 'test.png'
    shape = (25, 37)
    gray = 1 / 255.0
    create_dummy_img(fname, shape, 'ones', gray=gray)
    # Gray levels are between 0 and 1, and can be inverted:
    stim = ImageStimulus(fname)
    npt.assert_almost_equal(stim.data, gray)
    npt.assert_almost_equal(stim.invert().data, 1 - gray)
    # Inverting does not change the original object:
    npt.assert_almost_equal(stim.data, gray)
    os.remove(fname)


def test_ImageStimulus_rgb2gray():
    # Create a dummy image:
    fname = 'test.png'
    shape = (25, 37, 3)
    gray = 1 / 255.0
    create_dummy_img(fname, shape, 'ones', gray=gray)
    # Gray levels are between 0 and 1, and can be inverted:
    stim_rgb = ImageStimulus(fname)
    stim_gray = stim_rgb.rgb2gray()
    npt.assert_almost_equal(stim_gray.data, gray)
    npt.assert_equal(stim_gray.img_shape, shape[:2])
    # Original stim unchanged:
    npt.assert_equal(stim_rgb.img_shape, shape)
    os.remove(fname)


def test_ImageStimulus_resize():
    fname = 'test.png'
    shape = (25, 37, 3)
    gray = 129 / 255.0
    create_dummy_img(fname, shape, 'ones', gray=gray)
    # Gray levels are between 0 and 1, and can be inverted:
    stim = ImageStimulus(fname)
    npt.assert_almost_equal(stim.data, gray)
    npt.assert_equal(stim.resize((13, -1)).img_shape, (13, 19, 3))
    # Resize with one dimension -1:
    npt.assert_equal(stim.resize((-1, 24)).img_shape, (16, 24, 3))
    with pytest.raises(ValueError):
        stim.resize((-1, -1))
    os.remove(fname)


def test_ImageStimulus_trim():
    shape = (13, 29)
    ndarray = np.zeros(shape)
    ndarray[1:-1, 1:-1] = 0.1
    ndarray[2:-2, 2:-2] = 0.2
    stim = ImageStimulus(ndarray)
    npt.assert_equal(stim.trim().img_shape, (shape[0] - 2, shape[1] - 2))
    npt.assert_equal(stim.trim(tol=0.05).img_shape,
                     (shape[0] - 2, shape[1] - 2))
    npt.assert_equal(stim.trim(tol=0.1).img_shape,
                     (shape[0] - 4, shape[1] - 4))
    npt.assert_equal(stim.trim(tol=0.2).img_shape, (1, 0))
    npt.assert_equal(stim.trim(tol=0.1).img_shape,
                     stim.trim().trim(tol=0.1).img_shape)


def test_ImageStimulus_threshold():
    # Create a dummy image:
    fname = 'test.png'
    shape = (25, 37, 3)
    gray = 129 / 255.0
    create_dummy_img(fname, shape, 'ones', gray=gray)
    # Gray levels are between 0 and 1, and can be inverted:
    stim = ImageStimulus(fname, as_gray=True)
    stim_th = stim.threshold(0.5)
    npt.assert_almost_equal(stim.data, gray)
    npt.assert_equal(stim.img_shape, shape[:2])
    os.remove(fname)


def test_ImageStimulus_rotate():
    # Create a horizontal bar:
    fname = 'test.png'
    shape = (5, 5)
    ndarray = np.zeros(shape, dtype=np.uint8)
    ndarray[2, :] = 255
    imsave(fname, ndarray)
    stim = ImageStimulus(fname)
    # Vertical line:
    vert = stim.rotate(90, mode='reflect')
    npt.assert_almost_equal(vert.data.reshape(stim.img_shape)[:, 0], 0)
    npt.assert_almost_equal(vert.data.reshape(stim.img_shape)[:, 1], 0)
    npt.assert_almost_equal(vert.data.reshape(stim.img_shape)[:, 2], 1)
    npt.assert_almost_equal(vert.data.reshape(stim.img_shape)[:, 3], 0)
    npt.assert_almost_equal(vert.data.reshape(stim.img_shape)[:, 4], 0)
    # Diagonal, bottom-left to top-right:
    diag = stim.rotate(45, mode='reflect')
    npt.assert_almost_equal(diag.data.reshape(stim.img_shape)[0, 4], 1)
    npt.assert_almost_equal(diag.data.reshape(stim.img_shape)[2, 2], 1)
    npt.assert_almost_equal(diag.data.reshape(stim.img_shape)[4, 0], 1)
    npt.assert_almost_equal(diag.data.reshape(stim.img_shape)[0, 0], 0)
    npt.assert_almost_equal(diag.data.reshape(stim.img_shape)[4, 4], 0)
    # Diagonal, top-left to bottom-right:
    diag = stim.rotate(-45, mode='reflect')
    npt.assert_almost_equal(diag.data.reshape(stim.img_shape)[0, 0], 1)
    npt.assert_almost_equal(diag.data.reshape(stim.img_shape)[2, 2], 1)
    npt.assert_almost_equal(diag.data.reshape(stim.img_shape)[4, 4], 1)
    npt.assert_almost_equal(diag.data.reshape(stim.img_shape)[0, 4], 0)
    npt.assert_almost_equal(diag.data.reshape(stim.img_shape)[4, 0], 0)
    os.remove(fname)


def test_ImageStimulus_shift():
    # Create a horizontal bar:
    fname = 'test.png'
    shape = (5, 5)
    ndarray = np.zeros(shape, dtype=np.uint8)
    ndarray[2, :] = 255
    imsave(fname, ndarray)
    stim = ImageStimulus(fname)
    # Top row:
    top = stim.shift(0, -2)
    npt.assert_almost_equal(top.data.reshape(stim.img_shape)[0, :], 1)
    npt.assert_almost_equal(top.data.reshape(stim.img_shape)[1:, :], 0)
    # Bottom row:
    bottom = stim.shift(0, 2)
    npt.assert_almost_equal(bottom.data.reshape(stim.img_shape)[:4, :], 0)
    npt.assert_almost_equal(bottom.data.reshape(stim.img_shape)[4, :], 1)
    # Bottom right pixel:
    bottom = stim.shift(4, 2)
    npt.assert_almost_equal(bottom.data.reshape(stim.img_shape)[4, 4], 1)
    npt.assert_almost_equal(bottom.data.reshape(stim.img_shape)[:4, :], 0)
    npt.assert_almost_equal(bottom.data.reshape(stim.img_shape)[:, :4], 0)
    os.remove(fname)


def test_ImageStimulus_center():
    # Create a horizontal bar:
    fname = 'test.png'
    shape = (5, 5)
    ndarray = np.zeros(shape, dtype=np.uint8)
    ndarray[2, :] = 255
    imsave(fname, ndarray)
    # Center phosphene:
    stim = ImageStimulus(fname)
    npt.assert_almost_equal(stim.data, stim.center().data)
    npt.assert_almost_equal(stim.data, stim.shift(0, 2).center().data)
    os.remove(fname)


def test_ImageStimulus_scale():
    # Create a horizontal bar:
    fname = 'test.png'
    shape = (5, 5)
    ndarray = np.zeros(shape, dtype=np.uint8)
    ndarray[2, :] = 255
    imsave(fname, ndarray)
    # Scale phosphene:
    stim = ImageStimulus(fname)
    npt.assert_almost_equal(stim.data, stim.scale(1).data)
    npt.assert_almost_equal(stim.scale(0.1)[12], 1)
    npt.assert_almost_equal(stim.scale(0.1)[:12], 0)
    npt.assert_almost_equal(stim.scale(0.1)[13:], 0)
    with pytest.raises(ValueError):
        stim.scale(0)
    os.remove(fname)


def test_ImageStimulus_filter():
    # Create a dummy image:
    fname = 'test.png'
    shape = (25, 37)
    create_dummy_img(fname, shape, 'rand')
    stim = ImageStimulus(fname)

    for filt in ['sobel', 'scharr', 'canny', 'median']:
        filt_stim = stim.filter(filt)
        npt.assert_equal(filt_stim.shape, stim.shape)
        npt.assert_equal(filt_stim.img_shape, stim.img_shape)
        npt.assert_equal(filt_stim.electrodes, stim.electrodes)
        npt.assert_equal(filt_stim.time, None)

    # Invalid filter name:
    with pytest.raises(TypeError):
        stim.filter({'invalid'})
    with pytest.raises(ValueError):
        stim.filter('invalid')

    os.remove(fname)


def test_ImageStimulus_encode():
    stim = ImageStimulus(np.random.rand(4, 5))

    # Amplitude encoding in default range:
    enc = stim.encode()
    npt.assert_almost_equal(enc.time[-1], 500)
    npt.assert_almost_equal(enc.data.max(axis=1).min(), 0)
    npt.assert_almost_equal(enc.data.max(axis=1).max(), 50)

    # Amplitude encoding in custom range:
    enc = stim.encode(amp_range=(2, 43))
    npt.assert_almost_equal(enc.time[-1], 500)
    npt.assert_almost_equal(enc.data.max(axis=1).min(), 2)
    npt.assert_almost_equal(enc.data.max(axis=1).max(), 43)

    with pytest.raises(TypeError):
        stim.encode(pulse={'invalid': 1})
    with pytest.raises(ValueError):
        stim.encode(pulse=LogoUCSB())


def test_ImageStimulus_plot():
    # Create a horizontal bar:
    fname = 'test.png'
    shape = (5, 5)
    ndarray = np.zeros(shape, dtype=np.uint8)
    ndarray[2, :] = 255
    imsave(fname, ndarray)
    stim = ImageStimulus(fname)
    ax = stim.plot()
    npt.assert_equal(ax.axis(), (-0.5, 4.5, 4.5, -0.5))
    os.remove(fname)


def test_ImageStimulus_save():
    # Create a horizontal bar:
    fname = 'test.png'
    shape = (5, 5)
    ndarray = np.zeros(shape, dtype=np.uint8)
    ndarray[2, :] = 255
    imsave(fname, ndarray)
    stim = ImageStimulus(fname)
    fname2 = 'test2.png'
    stim.save(fname2)
    npt.assert_almost_equal(stim.data, ImageStimulus(fname2).data)
    os.remove(fname)
    os.remove(fname2)


@pytest.mark.parametrize('show_annotations', (True, False))
def test_SnellenChart(show_annotations):
    width = 840 if show_annotations else 444
    snellen = SnellenChart(show_annotations=show_annotations)
    npt.assert_equal(snellen.img_shape, (1348, width))
    npt.assert_equal(snellen.time, None)
    npt.assert_almost_equal(snellen.data.max(), 1)
    npt.assert_almost_equal(snellen.data.min(), 0)

    snellen = SnellenChart(row=1, show_annotations=show_annotations)
    npt.assert_equal(snellen.img_shape, (255, width))

    with pytest.raises(ValueError):
        SnellenChart(row=0)
    with pytest.raises(ValueError):
        SnellenChart(row=12)
    with pytest.raises(ValueError):
        SnellenChart(row=[1, 3])


def test_LogoBVL():
    logo = LogoBVL()
    npt.assert_equal(logo.img_shape, (576, 720, 4))
    npt.assert_equal(logo.time, None)
    npt.assert_almost_equal(logo.data.min(), 0)
    npt.assert_almost_equal(logo.data.max(), 1)


def test_LogoUCSB():
    logo = LogoUCSB()
    npt.assert_equal(logo.img_shape, (324, 727))
    npt.assert_equal(logo.time, None)
    npt.assert_almost_equal(logo.data.min(), 0)
    npt.assert_almost_equal(logo.data.max(), 1)
