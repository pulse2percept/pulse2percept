import numpy as np
import numpy.testing as npt
import pytest
try:
    # Python 3
    from unittest import mock
except ImportError:
    # Python 2
    import mock

try:
    # Python 3
    from imp import reload
except ImportError:
    pass

from .. import files
from .. import utils


@pytest.mark.skip(reason='ffmpeg dependency')
def test_set_skvideo_path():
    # Smoke-test
    files.set_skvideo_path('/usr/bin')
    files.set_skvideo_path(libav_path='/usr/bin')


@pytest.mark.skip(reason='ffmpeg dependency')
def test_load_video_metadata():
    # Load a test example
    reload(files)
    with pytest.raises(OSError):
        metadata = files.load_video_metadata('nothing_there.mp4')

    from skvideo import datasets
    metadata = files.load_video_metadata(datasets.bikes())
    npt.assert_equal(metadata['@codec_name'], 'h264')
    npt.assert_equal(metadata['@duration_ts'], '128000')
    npt.assert_equal(metadata['@r_frame_rate'], '25/1')

    # Trigger an import error
    with mock.patch.dict("sys.modules", {"skvideo": {}, "skvideo.utils": {}}):
        with pytest.raises(ImportError):
            reload(files)
            files.load_video_metadata(datasets.bikes())


@pytest.mark.skip(reason='ffmpeg dependency')
def test_load_framerate():
    # Load a test example
    reload(files)
    with pytest.raises(OSError):
        files.load_video_metadata('nothing_there.mp4')

    from skvideo import datasets
    fps = files.load_video_framerate(datasets.bikes())
    npt.assert_equal(fps, 25)

    # Trigger an import error
    with mock.patch.dict("sys.modules", {"skvideo": {}, "skvideo.utils": {}}):
        with pytest.raises(ImportError):
            reload(files)
            files.load_video_framerate(datasets.bikes())


@pytest.mark.skip(reason='ffmpeg dependency')
def test_load_video():
    reload(files)
    # Load a test example
    from skvideo import datasets

    # Load with default values
    video = files.load_video(datasets.bikes(), as_timeseries=False)
    npt.assert_equal(isinstance(video, np.ndarray), True)
    npt.assert_equal(video.shape, [250, 272, 640, 3])

    # Load as grayscale
    video = files.load_video(datasets.bikes(), as_timeseries=False,
                             as_gray=True)
    npt.assert_equal(isinstance(video, np.ndarray), True)
    npt.assert_equal(video.shape, [250, 272, 640, 1])

    # Load as TimeSeries
    video = files.load_video(datasets.bikes(), as_timeseries=True)
    fps = files.load_video_framerate(datasets.bikes())
    npt.assert_equal(isinstance(video, utils.TimeSeries), True)
    npt.assert_almost_equal(video.tsample, 1.0 / fps)
    npt.assert_equal(video.shape, [272, 640, 3, 250])

    # Trigger an import error
    with mock.patch.dict("sys.modules", {"skvideo": {}, "skvideo.utils": {}}):
        with pytest.raises(ImportError):
            reload(files)
            files.load_video('invalid.avi')


@pytest.mark.skip(reason='ffmpeg dependency')
def test_load_video_generator():
    # Load a test example
    reload(files)

    from skvideo import datasets
    reader = files.load_video_generator(datasets.bikes())
    for frame in reader.nextFrame():
        npt.assert_equal(frame.shape, [272, 640, 3])

    # Trigger an import error
    with mock.patch.dict("sys.modules", {"skvideo": {}, "skvideo.utils": {}}):
        with pytest.raises(ImportError):
            reload(files)
            files.load_video_generator('invalid.avi')


@pytest.mark.skip(reason='ffmpeg dependency')
def test_save_video():
    # Load a test example
    reload(files)
    from skvideo import datasets

    # There and back again: ndarray
    videoin = files.load_video(datasets.bikes(), as_timeseries=False)
    fpsin = files.load_video_framerate(datasets.bikes())
    files.save_video(videoin, 'myvideo.mp4', fps=fpsin)
    videout = files.load_video('myvideo.mp4', as_timeseries=False)
    npt.assert_equal(videoin.shape, videout.shape)
    npt.assert_almost_equal(videout / 255.0, videoin / 255.0, decimal=0)

    # Write to file with different frame rate, widths, and heights
    fpsout = 15
    files.save_video(videoin, 'myvideo.mp4', width=100, fps=fpsout)
    npt.assert_equal(files.load_video_framerate('myvideo.mp4'), fpsout)
    videout = files.load_video('myvideo.mp4', as_timeseries=False)
    npt.assert_equal(videout.shape[2], 100)
    files.save_video(videoin, 'myvideo.mp4', height=20, fps=fpsout)
    videout = files.load_video('myvideo.mp4', as_timeseries=False)
    npt.assert_equal(videout.shape[1], 20)
    videout = None

    # There and back again: TimeSeries
    tsamplein = 1.0 / float(fpsin)
    tsampleout = 1.0 / float(fpsout)
    rollaxes = np.roll(range(videoin.ndim), -1)
    tsin = utils.TimeSeries(tsamplein, np.transpose(videoin, rollaxes))
    files.save_video(tsin, 'myvideo.mp4', fps=fpsout)
    npt.assert_equal(tsin.tsample, tsamplein)
    tsout = files.load_video('myvideo.mp4', as_timeseries=True)
    npt.assert_equal(files.load_video_framerate('myvideo.mp4'), fpsout)
    npt.assert_equal(isinstance(tsout, utils.TimeSeries), True)
    npt.assert_almost_equal(tsout.tsample, tsampleout)

    # Also verify the actual data
    tsres = tsin.resample(tsampleout)
    npt.assert_equal(tsout.shape, tsres.shape)
    npt.assert_almost_equal(tsout.data / 255.0, tsres.data / tsres.data.max(),
                            decimal=0)

    with pytest.raises(TypeError):
        files.save_video([2, 3, 4], 'invalid.avi')

    # Trigger an import error
    with mock.patch.dict("sys.modules", {"skvideo": {}}):
        with pytest.raises(ImportError):
            reload(files)
            files.save_video(videoin, 'invalid.avi')
    with mock.patch.dict("sys.modules", {"skimage": {}}):
        with pytest.raises(ImportError):
            reload(files)
            files.save_video(videoin, 'invalid.avi')


@pytest.mark.skip(reason='ffmpeg dependency')
def test_save_video_sidebyside():
    reload(files)
    from skvideo import datasets
    videoin = files.load_video(datasets.bikes(), as_timeseries=False)
    fps = files.load_video_framerate(datasets.bikes())
    tsample = 1.0 / float(fps)
    rollaxes = np.roll(range(videoin.ndim), -1)
    percept = utils.TimeSeries(tsample, np.transpose(videoin, rollaxes))

    files.save_video_sidebyside(datasets.bikes(), percept, 'mymovie.mp4',
                                fps=fps)
    videout = files.load_video('mymovie.mp4', as_timeseries=False)
    npt.assert_equal(videout.shape[0], videoin.shape[0])
    npt.assert_equal(videout.shape[1], videoin.shape[1])
    npt.assert_equal(videout.shape[2], videoin.shape[2] * 2)
    npt.assert_equal(videout.shape[3], videoin.shape[3])

    with pytest.raises(TypeError):
        files.save_video_sidebyside(datasets.bikes(), [2, 3, 4], 'invalid.avi')

    with mock.patch.dict("sys.modules", {"skvideo": {}}):
        with pytest.raises(ImportError):
            reload(files)
            files.save_video_sidebyside(datasets.bikes(), percept,
                                        'invalid.avi')
    with mock.patch.dict("sys.modules", {"skimage": {}}):
        with pytest.raises(ImportError):
            reload(files)
            files.save_video_sidebyside(datasets.bikes(), percept,
                                        'invalid.avi')


def test_find_files_like():
    # Not sure how to check a valid pattern match, because we
    # don't know in which directory the test suite is
    # executed... so smoke test
    files.find_files_like(".", ".*")

    # Or fail to find an unlikely file name
    filenames = files.find_files_like(".", r"theresnowaythisexists\.xyz")
    npt.assert_equal(filenames, [])
