from pulse2percept.stimuli import (AmplitudeEncoder, VideoStimulus,
                                   BostonTrain, GirlPool)
from skimage.io import imsave
from matplotlib.animation import FuncAnimation
import numpy as np
import numpy.testing as npt
import pytest
from imageio import mimwrite
import matplotlib
matplotlib.use('Agg')


def test_VideoStimulus(tmp_path):
    # Create a dummy video:
    fname = str(tmp_path / 'test.mp4')
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
    # One electrode per pixel, named after its place in the frame (a letter
    # for the row, a number for the column). Frames are the time component,
    # so they do not enter the name:
    npt.assert_equal(len(stim.electrodes), np.prod(shape[1:]))
    npt.assert_equal(stim.electrodes[0], 'A1')
    npt.assert_equal(stim.electrodes[-1], 'AF48')
    npt.assert_equal(stim.electrodes.index('C12'),
                     np.ravel_multi_index((2, 11), shape[1:]))

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
    npt.assert_equal(len(stim.electrodes), np.prod(resize))
    npt.assert_equal(stim.electrodes[0], 'A1')
    npt.assert_equal(stim.electrodes[-1], 'P32')


def test_VideoStimulus_invert(tmp_path):
    fname = str(tmp_path / 'test.mp4')
    shape = (10, 32, 48, 3)
    gray = 129 / 255.0
    ndarray = np.ones(shape) * gray
    mimwrite(fname, (255 * ndarray).astype(np.uint8), fps=1)
    stim = VideoStimulus(fname)
    npt.assert_almost_equal(stim.data, gray, decimal=2)
    npt.assert_almost_equal(stim.invert().data, 1 - gray, decimal=2)
    # Inverting does not change the original object:
    npt.assert_almost_equal(stim.data, gray, decimal=2)


def test_VideoStimulus_rgb2gray(tmp_path):
    fname = str(tmp_path / 'test.mp4')
    shape = (10, 32, 48, 3)
    gray = 129 / 255.0
    ndarray = np.ones(shape) * gray
    mimwrite(fname, (255 * ndarray).astype(np.uint8), fps=1)
    stim = VideoStimulus(fname, as_gray=True)

    # Gray levels are between 0 and 1, and can be inverted:
    stim_rgb = VideoStimulus(fname)
    stim_gray = stim_rgb.rgb2gray()
    npt.assert_almost_equal(stim_gray.data, gray, decimal=2)
    npt.assert_equal(stim_gray.vid_shape, (shape[1], shape[2], shape[0]))
    # Original stim unchanged:
    npt.assert_equal(stim_rgb.vid_shape,
                     (shape[1], shape[2], shape[3], shape[0]))


def test_VideoStimulus_resize(tmp_path):
    fname = str(tmp_path / 'test.mp4')
    shape = (10, 32, 48)
    gray = 129 / 255.0
    ndarray = np.ones(shape) * gray
    mimwrite(fname, (255 * ndarray).astype(np.uint8), fps=1)
    # Gray levels are between 0 and 1, and can be inverted:
    stim = VideoStimulus(fname)
    npt.assert_almost_equal(stim.data, gray, decimal=2)
    npt.assert_equal(stim.resize((13, -1)).vid_shape, (13, 19, 3, 10))
    # Resize with one dimension -1:
    npt.assert_equal(stim.resize((-1, 24)).vid_shape, (16, 24, 3, 10))
    with pytest.raises(ValueError):
        stim.resize((-1, -1))


def test_VideoStimulus_crop(tmp_path):
    fname = str(tmp_path / 'test.mp4')
    shape = (10, 48, 32)
    ndarray = np.random.rand(*shape)
    fps = 1
    mimwrite(fname, (255 * ndarray).astype(np.uint8), fps=fps)
    stim = VideoStimulus(fname, as_gray=True)
    stim_cropped = stim.crop(idx_time=[3, 9], idx_space=[6, 10, 36, 30])
    npt.assert_equal(stim_cropped.vid_shape, (30, 20, 6))
    npt.assert_equal(stim_cropped.data.reshape(stim_cropped.vid_shape)[3, 7, 2],
                     stim.data.reshape(stim.vid_shape)[9, 17, 5])
    npt.assert_equal(stim_cropped.data.reshape(stim_cropped.vid_shape)[10, 18, 5],
                     stim.data.reshape(stim.vid_shape)[16, 28, 8])
    npt.assert_equal(stim_cropped.time, stim.time[3:9])
    npt.assert_equal(stim.electrodes.reshape(48, 32)[9, 17],
                     stim_cropped.electrodes.reshape(30, 20)[3, 7])
    npt.assert_equal(stim.electrodes.reshape(48, 32)[16, 28],
                     stim_cropped.electrodes.reshape(30, 20)[10, 18])

    stim_cropped2 = stim.crop(front=5, back=2, left=10,
                              right=8, top=6, bottom=7)
    npt.assert_equal(stim_cropped2.vid_shape, (35, 14, 3))
    npt.assert_equal(stim_cropped2.data.reshape(stim_cropped2.vid_shape)[3, 7, 2],
                     stim.data.reshape(stim.vid_shape)[9, 17, 7])
    npt.assert_equal(stim_cropped2.data.reshape(stim_cropped2.vid_shape)[10, 9, 1],
                     stim.data.reshape(stim.vid_shape)[16, 19, 6])
    npt.assert_equal(stim_cropped2.time, stim.time[5:8])

    # crop-time and crop-length (start, end) cannot be existed at the same time
    with pytest.raises(ValueError):
        stim.crop(idx_time=[0, 1], front=3)
    with pytest.raises(ValueError):
        stim.crop(idx_time=[3, 9], back=4)
    # Crop time is invalid. It should be [t1, t2], where t1 is the starting
    # frame and t2 is the ending frame
    with pytest.raises(TypeError):
        stim.crop(idx_time=[0, 1, 2])
    with pytest.raises(ValueError):
        stim.crop(idx_time=[5, 4])
    #"crop-length(start, end) cannot be negative"
    with pytest.raises(ValueError):
        stim.crop(front=-1)
    with pytest.raises(ValueError):
        stim.crop(back=-1)
    # crop-length(start, end) should be smaller than the duration of the video
    with pytest.raises(ValueError):
        stim.crop(front=5, back=6)
    # crop-indices and crop-width (left, right, up, down) cannot exist at the
    # same time
    with pytest.raises(Exception):
        stim.crop(idx_space=[5, 10, 25], left=10)
    with pytest.raises(Exception):
        stim.crop(idx_space=[5, 10, 25, 30], left=10)
    with pytest.raises(Exception):
        stim.crop(idx_space=[5, 10, 25, 30], right=8)
    with pytest.raises(Exception):
        stim.crop(idx_space=[5, 10, 25, 30], top=6)
    with pytest.raises(Exception):
        stim.crop(idx_space=[5, 10, 25, 30], bottom=7)
    # "crop-width(left, right, up, down) cannot be negative"
    with pytest.raises(ValueError):
        stim.crop(left=-1)
    with pytest.raises(ValueError):
        stim.crop(right=-1)
    with pytest.raises(ValueError):
        stim.crop(top=-1)
    with pytest.raises(ValueError):
        stim.crop(bottom=-1)
    # "crop-width should be smaller than the shape of the video frame"
    with pytest.raises(ValueError):
        stim.crop(left=14, right=20)
    with pytest.raises(ValueError):
        stim.crop(top=12, bottom=38)
    # "crop-indices must be on the video frame"
    with pytest.raises(ValueError):
        stim.crop(idx_space=[-1, 10, 25, 30])
    with pytest.raises(ValueError):
        stim.crop(idx_space=[5, -1, 25, 30])
    with pytest.raises(ValueError):
        stim.crop(idx_space=[5, 10, 50, 30])
    with pytest.raises(ValueError):
        stim.crop(idx_space=[5, 10, 25, 51])
    # crop-indices is invalid. It should be [y1,x1,y2,x2], where (y1,x1) is
    # upperleft and (y2,x2) is bottom-right
    with pytest.raises(ValueError):
        stim.crop(idx_space=[5, 10, 4, 30])
    with pytest.raises(ValueError):
        stim.crop(idx_space=[5, 10, 25, 9])


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


def test_VideoStimulus_filter(tmp_path):
    fname = str(tmp_path / 'test.mp4')
    shape = (10, 32, 48)
    gray = 129 / 255.0
    ndarray = np.ones(shape) * gray
    mimwrite(fname, (255 * ndarray).astype(np.uint8), fps=1)
    stim = VideoStimulus(fname, as_gray=True)

    for filt in ['sobel', 'scharr', 'canny', 'median']:
        filt_stim = stim.filter(filt)
        npt.assert_equal(filt_stim.shape, stim.shape)
        npt.assert_equal(filt_stim.vid_shape, stim.vid_shape)
        npt.assert_equal(np.asarray(filt_stim.electrodes),
                         np.asarray(stim.electrodes))
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


def test_VideoStimulus_encode():
    # 6 frames, 1 ms apart, so the encoded stimulus lasts 6 ms. Note that the
    # frame duration is the time between frames; before v0.9.2 it was taken to
    # be `1000 / that`, which made this stimulus 6000 ms long:
    stim = VideoStimulus(np.random.rand(4, 5, 6))
    enc = stim.encode(freq=1000)
    npt.assert_almost_equal(enc.time[-1], 6, decimal=3)
    npt.assert_equal(enc.shape[0], stim.shape[0])
    # Gray levels map onto the amplitude range absolutely, so the brightest
    # pixel of the video reaches the top of the range and the rest fall short
    # of it in proportion to how dark they are:
    npt.assert_almost_equal(np.abs(enc.data).max(axis=1),
                            50 * stim.data.max(axis=1), decimal=4)

    # Amplitude encoding in custom range:
    enc = stim.encode(amp_range=(2, 43), freq=1000)
    npt.assert_almost_equal(np.abs(enc.data).max(axis=1),
                            2 + 41 * stim.data.max(axis=1), decimal=4)

    # `encode` is a shorthand for AmplitudeEncoder, and forwards to it:
    npt.assert_almost_equal(stim.encode(freq=1000).data,
                            AmplitudeEncoder(freq=1000).encode(stim).data)
    with pytest.raises(TypeError):
        stim.encode(pulse={'invalid': 1})
    with pytest.raises(ValueError):
        stim.encode(pulse=BostonTrain())


def test_VideoStimulus_apply(tmp_path):
    fname = str(tmp_path / 'test.mp4')
    shape = (10, 32, 48)
    gray = 129 / 255.0
    ndarray = np.ones(shape) * gray
    mimwrite(fname, (255 * ndarray).astype(np.uint8), fps=1)
    stim = VideoStimulus(fname, as_gray=True)

    applied = stim.apply(lambda x: 0.5 * x)
    npt.assert_almost_equal(applied.data, stim.data * 0.5)


@pytest.mark.parametrize('n_frames', (1, 2, 3, 10, 14))
def test_VideoStimulus_play(n_frames):
    ndarray = np.random.rand(2, 4, n_frames)
    video = VideoStimulus(ndarray)
    ani = video.play()
    npt.assert_equal(isinstance(ani, FuncAnimation), True)
    npt.assert_equal(len(list(ani.frame_seq)), n_frames)
    # The animation renders as a self-contained HTML player:
    html = ani.to_jshtml()
    npt.assert_equal('p2p-anim' in html, True)
    npt.assert_equal(f'"n": {n_frames}' in html, True)
    npt.assert_equal(f't = {video.time[-1]:.2f} ms' in html, True)
    # Color videos are played back in color:
    rgb = VideoStimulus(np.random.rand(2, 4, 3, n_frames))
    npt.assert_equal('p2p-anim' in rgb.play().to_jshtml(), True)


def test_VideoStimulus_play_compressed():
    """Compression changes the number of frames, and 'vid_shape' must follow

    A video whose pixels are all nonzero survives spatial compression intact,
    but runs of identical frames are still dropped from the time axis. The
    player is handed a dense (Y, X, T) array, so a stale frame count in
    'vid_shape' makes that reshape fail.
    """
    frame = np.random.rand(4, 5) * 0.5 + 0.5
    other = np.random.rand(4, 5) * 0.5 + 0.5
    ndarray = np.stack([frame] * 4 + [other] * 4, axis=-1)
    video = VideoStimulus(ndarray, time=np.arange(8), compress=True)
    # Four of the eight time points are redundant and have been dropped:
    npt.assert_equal(video.data.shape[-1], 4)
    npt.assert_equal(video.vid_shape, (4, 5, 4))
    # The compressed time axis is no longer homogeneous, hence the explicit fps:
    html = video.play(fps=10).to_jshtml()
    npt.assert_equal('"n": 4' in html, True)
    npt.assert_equal(f't = {video.time[-1]:.2f} ms' in html, True)
    # An all-zero pixel is dropped instead, and no shape can describe what is
    # left, so playback fails with an explanation rather than a reshape error:
    sparse = np.zeros((4, 5, 6))
    sparse[1, 1, :] = np.linspace(0, 1, 6)
    with pytest.raises(ValueError):
        VideoStimulus(sparse, time=np.arange(6), compress=True).play()


def test_VideoStimulus_play_rgba():
    # A four-channel video is RGBA (see the class docstring), not RGB:
    ndarray = np.random.rand(4, 5, 4, 3)
    video = VideoStimulus(ndarray, time=np.arange(3))
    npt.assert_equal(video.vid_shape, (4, 5, 4, 3))
    for fmt in ('png', 'jpg'):
        npt.assert_equal('p2p-anim' in video.play(fmt=fmt).to_jshtml(), True)


def test_VideoStimulus_play_fmt():
    video = VideoStimulus(np.random.rand(8, 8, 4))
    npt.assert_equal('data:image/jpeg;base64,' in video.play().to_jshtml(),
                     True)
    npt.assert_equal('data:image/jpeg;base64,' in
                     video.play(fmt='png').to_jshtml(), False)
    with pytest.raises(ValueError):
        video.play(fmt='gif')


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
    npt.assert_almost_equal(video.data.min(), 0.0056, decimal=2)
    npt.assert_almost_equal(video.data.max(), 0.9871, decimal=2)


def test_GirlPool():
    video = GirlPool()
    npt.assert_equal(video.vid_shape, (240, 426, 3, 91))
    npt.assert_almost_equal(video.data.min(), 0)
    npt.assert_almost_equal(video.data.max(), 1)

    # Grayscale:
    video = GirlPool(as_gray=True)
    npt.assert_equal(video.vid_shape, (240, 426, 91))
    npt.assert_almost_equal(video.data.min(), 0)
    npt.assert_almost_equal(video.data.max(), 0.9983, decimal=2)

    # Resize:
    video = GirlPool(resize=(32, 32))
    npt.assert_equal(video.vid_shape, (32, 32, 3, 91))
    npt.assert_almost_equal(video.data.min(), 0.0001, decimal=2)
    npt.assert_almost_equal(video.data.max(), 0.9988, decimal=2)


def test_VideoStimulus_data_is_contiguous(tmp_path):
    """Video data must reach the Stimulus constructor C-contiguous.

    Frames are decoded frame-first and then transposed so that time is the
    last axis. Taking that transpose lazily leaves the array non-contiguous
    all the way through the conversion to float, and the constructor then has
    to copy it at four times the size.
    """
    fname = str(tmp_path / 'test.mp4')
    ndarray = np.random.rand(12, 32, 48)
    mimwrite(fname, (255 * ndarray).astype(np.uint8), fps=5)
    for kwargs in ({}, {'as_gray': True}, {'resize': (16, 24)}):
        stim = VideoStimulus(fname, **kwargs)
        npt.assert_equal(stim.data.flags['C_CONTIGUOUS'], True)
