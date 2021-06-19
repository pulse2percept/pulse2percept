import os
import numpy as np
import numpy.testing as npt
import pytest
from imageio import mimwrite
from matplotlib.animation import FuncAnimation
from skimage.io import imsave

from pulse2percept.stimuli import VideoStimulus, BostonTrain


def test_VideoStimulus():
    # Create a dummy video:
    fname = 'test.mp4'
    shape = (10, 32, 48)
    ndarray = np.random.rand(*shape)
    fps = 1
    mimwrite(fname, (255 * ndarray).astype(np.uint8), fps=fps)
    stim = VideoStimulus(fname, as_gray=True)
    npt.assert_equal(stim.shape, (np.prod(shape[1:]), shape[0]))
    npt.assert_almost_equal(stim.data,
                            ndarray.reshape((shape[0], -1)).transpose(),
                            decimal=1)
    npt.assert_equal(stim.metadata['source'], fname)
    npt.assert_equal(stim.metadata['source_size'], (shape[2], shape[1]))
    npt.assert_almost_equal(stim.time, np.arange(shape[0]) * 1000.0 / fps)
    npt.assert_equal(stim.electrodes, np.arange(np.prod(shape[1:])))
    os.remove(fname)

    # Resize the video:
    ndarray = np.ones(shape)
    mimwrite(fname, (255 * ndarray).astype(np.uint8), fps=fps)
    resize = (16, 32)
    stim = VideoStimulus(fname, as_gray=True, resize=resize)
    npt.assert_equal(stim.shape, (np.prod(resize), shape[0]))
    npt.assert_almost_equal(stim.data,
                            np.ones((np.prod(resize), shape[0])), decimal=1)
    npt.assert_equal(stim.metadata['source'], fname)
    npt.assert_equal(stim.metadata['source_size'], (shape[2], shape[1]))
    npt.assert_almost_equal(stim.time, np.arange(shape[0]) * 1000 / fps)
    npt.assert_equal(stim.electrodes, np.arange(np.prod(resize)))
    os.remove(fname)


def test_VideoStimulus_invert():
    fname = 'test.mp4'
    shape = (10, 32, 48, 3)
    gray = 129 / 255.0
    ndarray = np.ones(shape) * gray
    mimwrite(fname, (255 * ndarray).astype(np.uint8), fps=1)
    stim = VideoStimulus(fname)
    npt.assert_almost_equal(stim.data, gray)
    npt.assert_almost_equal(stim.invert().data, 1 - gray)
    # Inverting does not change the original object:
    npt.assert_almost_equal(stim.data, gray)
    os.remove(fname)


def test_VideoStimulus_rgb2gray():
    fname = 'test.mp4'
    shape = (10, 32, 48, 3)
    gray = 129 / 255.0
    ndarray = np.ones(shape) * gray
    mimwrite(fname, (255 * ndarray).astype(np.uint8), fps=1)
    stim = VideoStimulus(fname, as_gray=True)

    # Gray levels are between 0 and 1, and can be inverted:
    stim_rgb = VideoStimulus(fname)
    stim_gray = stim_rgb.rgb2gray()
    npt.assert_almost_equal(stim_gray.data, gray)
    npt.assert_equal(stim_gray.vid_shape, (shape[1], shape[2], shape[0]))
    # Original stim unchanged:
    npt.assert_equal(stim_rgb.vid_shape,
                     (shape[1], shape[2], shape[3], shape[0]))
    os.remove(fname)


def test_VideoStimulus_resize():
    fname = 'test.mp4'
    shape = (10, 32, 48)
    gray = 129 / 255.0
    ndarray = np.ones(shape) * gray
    mimwrite(fname, (255 * ndarray).astype(np.uint8), fps=1)
    # Gray levels are between 0 and 1, and can be inverted:
    stim = VideoStimulus(fname)
    npt.assert_almost_equal(stim.data, gray)
    npt.assert_equal(stim.resize((13, -1)).vid_shape, (13, 19, 3, 10))
    # Resize with one dimension -1:
    npt.assert_equal(stim.resize((-1, 24)).vid_shape, (16, 24, 3, 10))
    with pytest.raises(ValueError):
        stim.resize((-1, -1))
    os.remove(fname)


def test_VideoStimulus_rotate():
    # Create a horizontal bar:
    shape = (5, 5, 3)
    ndarray = np.zeros(shape, dtype=np.uint8)
    ndarray[2, :, :] = 255
    stim = VideoStimulus(ndarray)
    # Vertical line:
    vert = stim.rotate(90, mode='constant')
    data = vert.data.reshape(vert.vid_shape)
    for i in range(data.shape[-1]):
        npt.assert_almost_equal(data[:, 0, i], 0)
        npt.assert_almost_equal(data[:, 1, i], 0)
        npt.assert_almost_equal(data[:, 2, i], 1)
        npt.assert_almost_equal(data[:, 3, i], 0)
        npt.assert_almost_equal(data[:, 4, i], 0)
    # Diagonal, bottom-left to top-right:
    diag = stim.rotate(45, mode='constant')
    data = diag.data.reshape(diag.vid_shape)
    for i in range(data.shape[-1]):
        npt.assert_almost_equal(data[1, 3, i], 1)
        npt.assert_almost_equal(data[2, 2, i], 1)
        npt.assert_almost_equal(data[3, 1, i], 1)
        npt.assert_almost_equal(data[0, 0, i], 0)
        npt.assert_almost_equal(data[4, 4, i], 0)
    # Diagonal, top-left to bottom-right:
    diag = stim.rotate(-45, mode='constant')
    data = diag.data.reshape(diag.vid_shape)
    for i in range(data.shape[-1]):
        npt.assert_almost_equal(data[1, 1, i], 1)
        npt.assert_almost_equal(data[2, 2, i], 1)
        npt.assert_almost_equal(data[3, 3, i], 1)
        npt.assert_almost_equal(data[0, 4, i], 0)
        npt.assert_almost_equal(data[4, 0, i], 0)


def test_VideoStimulus_shift():
    # Create a horizontal bar:
    shape = (5, 5, 3)
    ndarray = np.zeros(shape, dtype=np.uint8)
    ndarray[2, :, :] = 255
    stim = VideoStimulus(ndarray)
    # Top row:
    top = stim.shift(0, -2)
    data = top.data.reshape(top.vid_shape)
    for i in range(data.shape[-1]):
        npt.assert_almost_equal(top.data.reshape(stim.vid_shape)[0, :, i], 1)
        npt.assert_almost_equal(top.data.reshape(stim.vid_shape)[1:, :, i], 0)
    # Bottom row:
    bottom = stim.shift(0, 2)
    data = bottom.data.reshape(bottom.vid_shape)
    for i in range(data.shape[-1]):
        npt.assert_almost_equal(bottom.data.reshape(stim.vid_shape)[:4, :, i],
                                0)
        npt.assert_almost_equal(bottom.data.reshape(stim.vid_shape)[4, :, i],
                                1)
    # Bottom right pixel:
    bottom = stim.shift(4, 2)
    data = bottom.data.reshape(bottom.vid_shape)
    for i in range(data.shape[-1]):
        npt.assert_almost_equal(bottom.data.reshape(stim.vid_shape)[4, 4, i],
                                1)
        npt.assert_almost_equal(bottom.data.reshape(stim.vid_shape)[:4, :, i],
                                0)
        npt.assert_almost_equal(bottom.data.reshape(stim.vid_shape)[:, :4, i],
                                0)


def test_ImageStimulus_center():
    # Create a horizontal bar:
    ndarray = np.zeros((5, 5, 3), dtype=np.uint8)
    ndarray[2, :, :] = 255
    # Center phosphene:
    stim = VideoStimulus(ndarray)
    npt.assert_almost_equal(stim.data, stim.center().data)
    npt.assert_almost_equal(stim.data, stim.shift(0, 2).center().data)


def test_ImageStimulus_scale():
    # Create a horizontal bar:
    ndarray = np.zeros((5, 5, 3), dtype=np.uint8)
    ndarray[2, :, :] = 255
    stim = VideoStimulus(ndarray)
    npt.assert_almost_equal(stim.data, stim.scale(1).data)
    for i in range(stim.shape[-1]):
        npt.assert_almost_equal(stim.scale(0.1)[12, i], 1)
        npt.assert_almost_equal(stim.scale(0.1)[:12, i], 0)
        npt.assert_almost_equal(stim.scale(0.1)[13:, i], 0)
    with pytest.raises(ValueError):
        stim.scale(0)


def test_VideoStimulus_filter():
    fname = 'test.mp4'
    shape = (10, 32, 48)
    gray = 129 / 255.0
    ndarray = np.ones(shape) * gray
    mimwrite(fname, (255 * ndarray).astype(np.uint8), fps=1)
    stim = VideoStimulus(fname, as_gray=True)

    for filt in ['sobel', 'scharr', 'canny', 'median']:
        filt_stim = stim.filter(filt)
        npt.assert_equal(filt_stim.shape, stim.shape)
        npt.assert_equal(filt_stim.vid_shape, stim.vid_shape)
        npt.assert_equal(filt_stim.electrodes, stim.electrodes)
        npt.assert_equal(filt_stim.time, stim.time)

    # Invalid filter name:
    with pytest.raises(TypeError):
        stim.filter({'invalid'})
    with pytest.raises(ValueError):
        stim.filter('invalid')

    # Cannot apply filter to RGB video:
    shape = (10, 32, 48, 3)
    ndarray = np.ones(shape) * gray
    mimwrite(fname, (255 * ndarray).astype(np.uint8), fps=1)
    stim = VideoStimulus(fname)
    with pytest.raises(ValueError):
        stim.filter('sobel')

    os.remove(fname)


def test_VideoStimulus_encode():
    stim = VideoStimulus(np.random.rand(4, 5, 6))

    # Amplitude encoding in default range:
    enc = stim.encode()
    npt.assert_almost_equal(enc.time[-1], 6000)
    npt.assert_almost_equal(enc.data[:, 4::7].min(), 0)
    npt.assert_almost_equal(enc.data[:, 4::7].max(), 50)

    # Amplitude encoding in custom range:
    enc = stim.encode(amp_range=(2, 43))
    npt.assert_almost_equal(enc.time[-1], 6000)
    npt.assert_almost_equal(enc.data[:, 4::7].min(), 2)
    npt.assert_almost_equal(enc.data[:, 4::7].max(), 43)

    with pytest.raises(TypeError):
        stim.encode(pulse={'invalid': 1})
    with pytest.raises(ValueError):
        stim.encode(pulse=BostonTrain())


def test_VideoStimulus_apply():
    fname = 'test.mp4'
    shape = (10, 32, 48)
    gray = 129 / 255.0
    ndarray = np.ones(shape) * gray
    mimwrite(fname, (255 * ndarray).astype(np.uint8), fps=1)
    stim = VideoStimulus(fname, as_gray=True)

    applied = stim.apply(lambda x: 0.5 * x)
    npt.assert_almost_equal(applied.data, stim.data * 0.5)


@pytest.mark.parametrize('n_frames', (2, 3, 10, 14))
def test_VideoStimulus_play(n_frames):
    ndarray = np.random.rand(2, 4, n_frames)
    video = VideoStimulus(ndarray)
    ani = video.play()
    npt.assert_equal(isinstance(ani, FuncAnimation), True)
    npt.assert_equal(len(list(ani.frame_seq)), n_frames)


def test_BostonTrain():
    video = BostonTrain()
    npt.assert_equal(video.vid_shape, (240, 426, 3, 94))
    npt.assert_almost_equal(video.data.min(), 0)
    npt.assert_almost_equal(video.data.max(), 1)

    # Grayscale:
    video = BostonTrain(as_gray=True)
    npt.assert_equal(video.vid_shape, (240, 426, 94))
    npt.assert_almost_equal(video.data.min(), 0)
    npt.assert_almost_equal(video.data.max(), 1)

    # Resize:
    video = BostonTrain(resize=(32, 32))
    npt.assert_equal(video.vid_shape, (32, 32, 3, 94))
    npt.assert_almost_equal(video.data.min(), 0.0039, decimal=3)
    npt.assert_almost_equal(video.data.max(), 0.9843, decimal=3)
