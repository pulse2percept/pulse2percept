from pulse2percept.stimuli import (AmplitudeEncoder, VideoStimulus,
                                   BostonTrain, GirlPool)
from pulse2percept.stimuli.videos import _frame_index
from pulse2percept.units import (DimensionMismatchError, Hz, kHz, ms, s,
                                 uA)
from skimage.color import rgb2gray
from skimage.io import imsave
from skimage.transform import resize as vid_resize
from matplotlib.animation import FuncAnimation
import re
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


def test_VideoStimulus_resize_kwargs():
    """Keyword arguments reach scikit-image (Issue #501)"""
    # A white square on black. Nearest-neighbor interpolation keeps the video
    # binary; the default (bilinear, with anti-aliasing on the way down) does
    # not, which is what makes the two distinguishable:
    ndarray = np.zeros((8, 8, 3), dtype=np.float32)
    ndarray[2:6, 2:6] = 1
    stim = VideoStimulus(ndarray)
    nearest = stim.resize((4, 4), order=0, anti_aliasing=False)
    npt.assert_equal(np.isin(nearest.data, [0, 1]).all(), True)
    npt.assert_equal(np.isin(stim.resize((4, 4)).data, [0, 1]).all(), False)
    # An unknown keyword argument is scikit-image's to reject, not ours:
    with pytest.raises(TypeError):
        stim.resize((4, 4), not_a_skimage_kwarg=0)


@pytest.fixture
def clip_source(tmp_path):
    """A 20-frame RGB movie at 10 fps (100 ms per frame), plus its frames"""
    fname = str(tmp_path / 'clip.mp4')
    # 16x16 keeps ffmpeg from resizing to its macro block size, which would
    # make the file's frames a different shape than the ones written here:
    frames = (255 * np.random.default_rng(0).random((20, 16, 16, 3)))
    frames = frames.astype(np.uint8)
    mimwrite(fname, frames, fps=10)
    return fname, frames


def test_VideoStimulus_stop_time(clip_source):
    fname, _ = clip_source
    full = VideoStimulus(fname)
    clip = VideoStimulus(fname, stop_time=500)
    npt.assert_equal(clip.vid_shape, (*full.vid_shape[:-1], 5))
    npt.assert_almost_equal(clip.data, full.data[:, :5])
    # The interval is half-open, so the frame starting at 500 ms is not in it:
    npt.assert_almost_equal(clip.time, np.arange(5) * 100.0)


def test_VideoStimulus_start_and_stop_time(clip_source):
    fname, _ = clip_source
    full = VideoStimulus(fname)
    clip = VideoStimulus(fname, start_time=500, stop_time=1000)
    npt.assert_almost_equal(clip.data, full.data[:, 5:10])
    # A clip starts at t=0 wherever it was cut from, but keeps the frame
    # interval of the source:
    npt.assert_almost_equal(clip.time, np.arange(5) * 100.0)


def test_VideoStimulus_clip_units(clip_source):
    """Bare milliseconds and unitful times name the same frames"""
    fname, _ = clip_source
    bare = VideoStimulus(fname, start_time=500, stop_time=1500)
    unitful = VideoStimulus(fname, start_time=0.5 * s, stop_time=1.5 * s)
    npt.assert_almost_equal(unitful.data, bare.data)
    npt.assert_almost_equal(unitful.time, bare.time)


def test_VideoStimulus_clip_rgb2gray_and_resize(clip_source):
    fname, frames = clip_source
    clip = VideoStimulus(fname, start_time=300, stop_time=600, as_gray=True,
                         resize=(8, 8))
    npt.assert_equal(clip.vid_shape, (8, 8, 3))
    npt.assert_equal(clip.shape, (64, 3))
    expected = vid_resize(rgb2gray(frames[3:6] / 255.0), (3, 8, 8))
    npt.assert_almost_equal(clip.data, expected.reshape((3, -1)).transpose(),
                            decimal=1)


def test_VideoStimulus_stop_time_past_eof(clip_source):
    """A stop time past the end of the file yields what is there"""
    fname, _ = clip_source
    npt.assert_almost_equal(VideoStimulus(fname, stop_time=60 * s).data,
                            VideoStimulus(fname).data)


@pytest.mark.parametrize('kwargs, err', [
    ({'start_time': -100}, ValueError),
    ({'start_time': np.inf}, ValueError),
    ({'stop_time': np.nan}, ValueError),
    ({'start_time': 500, 'stop_time': 500}, ValueError),
    ({'start_time': 500, 'stop_time': 200}, ValueError),
    ({'stop_time': 1 * uA}, DimensionMismatchError),
    # Past the end of the file, so nothing was loaded:
    ({'start_time': 60 * s}, ValueError),
    # Falls between two frame starts (100 and 200 ms), so it names no frame:
    ({'start_time': 110, 'stop_time': 150}, ValueError),
])
def test_VideoStimulus_clip_invalid(clip_source, kwargs, err):
    fname, _ = clip_source
    with pytest.raises(err):
        VideoStimulus(fname, **kwargs)


@pytest.mark.parametrize('source', ['array', 'stimulus'])
def test_VideoStimulus_clip_rejects_in_memory_source(clip_source, source):
    """An in-memory video is shortened by crop(), not while decoding"""
    fname, _ = clip_source
    src = VideoStimulus(fname)
    if source == 'array':
        src = src.data.reshape(src.vid_shape)
    with pytest.raises(ValueError):
        VideoStimulus(src, stop_time=500)


def test_frame_index_on_frame_boundaries():
    """A frame's own start time names that frame, not the one after it

    29.97 fps makes ``i * 1000 / fps * fps / 1000`` land just off ``i``, which
    is what the tolerance in ``_frame_index`` is for. Every other clipping test
    here runs at 10 fps, where the arithmetic happens to be exact.
    """
    fps = 29.97
    for i in (1, 10, 100, 1000):
        npt.assert_equal(_frame_index(i * 1000 / fps, fps), i)


def test_VideoStimulus_clip_does_not_decode_the_whole_file(monkeypatch):
    """Clipping must bound the read, not slice a fully decoded movie"""
    requested = []

    class FakeReader:
        """Hands out 1000 frames of a 10 fps movie, recording each request"""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True
            return False

        def get_meta_data(self):
            return {'fps': 10, 'source_size': (4, 4)}

        def set_image_index(self, index):
            self.index = index

        def get_next_data(self):
            if self.index >= 1000:
                raise IndexError(self.index)
            requested.append(self.index)
            self.index += 1
            return np.full((4, 4, 3), self.index - 1, dtype=np.uint8)

    reader = FakeReader()
    reader.index = 0
    reader.closed = False
    monkeypatch.setattr('pulse2percept.stimuli.videos.video_reader',
                        lambda *args, **kwargs: reader)

    stim = VideoStimulus('fake.mp4', start_time=2 * s, stop_time=2.5 * s)
    npt.assert_equal(requested, list(range(20, 25)))
    npt.assert_equal(stim.vid_shape, (4, 4, 3, 5))
    npt.assert_equal(reader.closed, True)


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


@pytest.mark.parametrize('shape', [(5, 5, 3), (5, 5, 3, 3)])
def test_VideoStimulus_rotate_kwargs(shape):
    """Keyword arguments reach scikit-image (Issue #501)

    A grayscale video is rotated in one pass and a color one frame by frame,
    so both paths have to forward what they are given.
    """
    ndarray = np.zeros(shape, dtype=np.float32)
    ndarray[2] = 1
    stim = VideoStimulus(ndarray)
    # Nearest-neighbor interpolation keeps the bar binary, bilinear does not:
    npt.assert_equal(np.isin(stim.rotate(45, order=0).data, [0, 1]).all(), True)
    npt.assert_equal(np.isin(stim.rotate(45, order=1).data, [0, 1]).all(),
                     False)
    # 'cval' fills the corners the rotation leaves empty:
    rot = stim.rotate(45, order=0, cval=0.3)
    npt.assert_almost_equal(rot.data.reshape(rot.vid_shape)[0, 0], 0.3)
    # 'resize' grows each frame, so the result is named after its own grid
    # rather than inheriting names it has no room for:
    grown = stim.rotate(45, resize=True)
    npt.assert_equal(grown.vid_shape, (7, 7, *shape[2:]))
    npt.assert_equal(grown.shape[0], np.prod(grown.vid_shape[:-1]))
    npt.assert_equal(grown.electrodes[0], 'A1' if len(shape) == 3 else 'A1_R')
    # Rotating in place keeps every pixel's name, and the time axis is intact:
    same = stim.rotate(45)
    npt.assert_equal(same.vid_shape, shape)
    npt.assert_equal(np.asarray(same.electrodes), np.asarray(stim.electrodes))
    npt.assert_almost_equal(same.time, stim.time)


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
    # A shape-preserving function keeps every pixel's name:
    npt.assert_equal(np.asarray(applied.electrodes),
                     np.asarray(stim.electrodes))
    npt.assert_equal(applied.vid_shape, stim.vid_shape)

    # A function that changes the resolution is allowed, and the result is
    # named after its own pixel grid (Issue #500):
    resized = stim.apply(vid_resize, (16, 24))
    npt.assert_equal(resized.vid_shape, (16, 24, shape[0]))
    npt.assert_equal(resized.shape, (16 * 24, shape[0]))
    npt.assert_equal(resized.electrodes[0], 'A1')
    npt.assert_equal(resized.electrodes[-1], 'P24')
    npt.assert_almost_equal(resized.time, stim.time)
    # Positional and keyword arguments both make it through:
    npt.assert_equal(stim.apply(vid_resize, (8, 12), order=0).vid_shape,
                     (8, 12, shape[0]))
    npt.assert_equal(stim.apply(vid_resize, output_shape=(8, 12)).vid_shape,
                     (8, 12, shape[0]))
    # Names can be given explicitly:
    named = stim.apply(vid_resize, (1, 2), electrodes=['a', 'b'])
    npt.assert_equal(list(named.electrodes), ['a', 'b'])
    with pytest.raises(ValueError):
        stim.apply(vid_resize, (1, 2), electrodes=['a', 'b', 'c'])
    # Dropping the color channels changes the pixel count too:
    rgb = VideoStimulus(np.random.rand(6, 8, 3, 4).astype(np.float32))
    npt.assert_equal(rgb.apply(rgb2gray).vid_shape, (6, 8, 4))


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


def test_VideoStimulus_play_fps_units():
    """A frame rate is a frequency, however it is spelled

    .. versionadded:: 0.10.0
    """
    video = VideoStimulus(np.random.rand(2, 4, 5))

    def interval(fps):
        """The frame delay (ms) the HTML player was configured with"""
        html = video.play(fps=fps).to_jshtml()
        return float(re.search(r'"interval": ([0-9.]+)', html).group(1))

    npt.assert_almost_equal(interval(30), 1000 / 30, decimal=6)
    for spelling in (30 * Hz, 0.03 * kHz):
        npt.assert_almost_equal(interval(spelling), interval(30), decimal=12)
    for wrong in (30 * ms, 30 * uA):
        with pytest.raises(DimensionMismatchError):
            video.play(fps=wrong)


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
    # Compressing at construction time and compressing afterwards must leave
    # the stimulus in the same state:
    eager = VideoStimulus(ndarray, time=np.arange(8), compress=True)
    lazy = VideoStimulus(ndarray, time=np.arange(8))
    npt.assert_equal(lazy.vid_shape, (4, 5, 8))
    lazy.compress()
    for video in (eager, lazy):
        # Four of the eight time points are redundant and have been dropped:
        npt.assert_equal(video.data.shape[-1], 4)
        npt.assert_equal(video.vid_shape, (4, 5, 4))
        # The compressed time axis is no longer homogeneous, hence the fps:
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


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.uint8])
def test_VideoStimulus_owns_its_data(dtype):
    # See `test_ImageStimulus_owns_its_data`: float32 is the dtype that
    # `img_as_float32` passes through untouched.
    arr = (np.linspace(0, 1, 60).reshape((4, 5, 3)) if dtype != np.uint8
           else np.arange(60, dtype=np.uint8).reshape((4, 5, 3)))
    arr = np.ascontiguousarray(arr, dtype=dtype)
    stim = VideoStimulus(arr)
    before = stim.data.copy()
    arr[...] = 0
    npt.assert_array_equal(stim.data, before)
    npt.assert_equal(np.shares_memory(arr, stim.data), False)
    npt.assert_equal(stim.data.flags.writeable, False)
    # Freezing what the stimulus took must not reach back into what the
    # caller kept:
    npt.assert_equal(arr.flags.writeable, True)


def test_VideoStimulus_does_not_alias_another_stimulus():
    first = VideoStimulus(np.linspace(0, 1, 60, dtype=np.float32)
                          .reshape((4, 5, 3)))
    second = VideoStimulus(first)
    npt.assert_equal(np.shares_memory(first.data, second.data), False)
