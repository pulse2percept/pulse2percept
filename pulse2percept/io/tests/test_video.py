import os
import numpy as np
import numpy.testing as npt
import pytest
from unittest import mock
from imp import reload

# Import the whole module so we can reload it:
from pulse2percept.io import video
from pulse2percept.io import image
from pulse2percept import stimuli
from pulse2percept import implants


@pytest.mark.skip(reason='ffmpeg dependency')
def test_video2stim():
    reload(image)
    reload(video)
    # Smoke-test example video
    from skvideo import datasets
    implant = implants.ArgusI()
    video.video2stim(datasets.bikes(), implant)
    with pytest.raises(OSError):
        video.video2stim('no-such-file.avi', implant)


@pytest.mark.skip(reason='ffmpeg dependency')
def test__set_skvideo_path():
    # Smoke-test
    video._set_skvideo_path('/usr/bin')
    video._set_skvideo_path(libav_path='/usr/bin')


@pytest.mark.skip(reason='ffmpeg dependency')
def test_load_video_metadata():
    # Load a test example
    reload(video)
    with pytest.raises(OSError):
        metadata = video.load_video_metadata('nothing_there.mp4')

    from skvideo import datasets
    metadata = video.load_video_metadata(datasets.bikes())
    npt.assert_equal(metadata['@codec_name'], 'h264')
    npt.assert_equal(metadata['@duration_ts'], '128000')
    npt.assert_equal(metadata['@r_frame_rate'], '25/1')

    # Trigger an import error
    with mock.patch.dict("sys.modules", {"skvideo": {}, "skvideo.utils": {}}):
        with pytest.raises(ImportError):
            reload(video)
            video.load_video_metadata(datasets.bikes())


@pytest.mark.skip(reason='ffmpeg dependency')
def test_load_framerate():
    # Load a test example
    reload(video)
    with pytest.raises(OSError):
        video.load_video_metadata('nothing_there.mp4')

    from skvideo import datasets
    fps = video.load_video_framerate(datasets.bikes())
    npt.assert_equal(fps, 25)

    # Trigger an import error
    with mock.patch.dict("sys.modules", {"skvideo": {}, "skvideo.utils": {}}):
        with pytest.raises(ImportError):
            reload(video)
            video.load_video_framerate(datasets.bikes())


@pytest.mark.skip(reason='ffmpeg dependency')
def test_load_video():
    reload(video)
    # Load a test example
    from skvideo import datasets

    # Load with default values
    movie = video.load_video(datasets.bikes(), as_timeseries=False)
    npt.assert_equal(isinstance(movie, np.ndarray), True)
    npt.assert_equal(movie.shape, [250, 272, 640, 3])

    # Load as grayscale
    movie = video.load_video(datasets.bikes(), as_timeseries=False,
                             as_gray=True)
    npt.assert_equal(isinstance(movie, np.ndarray), True)
    npt.assert_equal(movie.shape, [250, 272, 640, 1])

    # Load as TimeSeries
    movie = video.load_video(datasets.bikes(), as_timeseries=True)
    fps = video.load_video_framerate(datasets.bikes())
    npt.assert_equal(isinstance(movie, stimuli.TimeSeries), True)
    npt.assert_almost_equal(movie.tsample, 1.0 / fps)
    npt.assert_equal(movie.shape, [272, 640, 3, 250])

    # Trigger an import error
    with mock.patch.dict("sys.modules", {"skvideo": {}, "skvideo.utils": {}}):
        with pytest.raises(ImportError):
            reload(video)
            video.load_video('invalid.avi')


@pytest.mark.skip(reason='ffmpeg dependency')
def test_load_video_generator():
    # Load a test example
    reload(video)

    from skvideo import datasets
    reader = video.load_video_generator(datasets.bikes())
    for frame in reader.nextFrame():
        npt.assert_equal(frame.shape, [272, 640, 3])

    # Trigger an import error
    with mock.patch.dict("sys.modules", {"skvideo": {}, "skvideo.utils": {}}):
        with pytest.raises(ImportError):
            reload(video)
            video.load_video_generator('invalid.avi')


@pytest.mark.skip(reason='ffmpeg dependency')
def test_save_video():
    # Load a test example
    reload(video)
    from skvideo import datasets

    # There and back again: ndarray
    videoin = video.load_video(datasets.bikes(), as_timeseries=False)
    fpsin = video.load_video_framerate(datasets.bikes())
    video.save_video(videoin, 'myvideo.mp4', fps=fpsin)
    videout = video.load_video('myvideo.mp4', as_timeseries=False)
    npt.assert_equal(videoin.shape, videout.shape)
    npt.assert_almost_equal(videout / 255.0, videoin / 255.0, decimal=0)

    # Write to file with different frame rate, widths, and heights
    fpsout = 15
    video.save_video(videoin, 'myvideo.mp4', width=100, fps=fpsout)
    npt.assert_equal(video.load_video_framerate('myvideo.mp4'), fpsout)
    videout = video.load_video('myvideo.mp4', as_timeseries=False)
    npt.assert_equal(videout.shape[2], 100)
    video.save_video(videoin, 'myvideo.mp4', height=20, fps=fpsout)
    videout = video.load_video('myvideo.mp4', as_timeseries=False)
    npt.assert_equal(videout.shape[1], 20)
    videout = None

    # There and back again: TimeSeries
    tsamplein = 1.0 / float(fpsin)
    tsampleout = 1.0 / float(fpsout)
    rollaxes = np.roll(range(videoin.ndim), -1)
    tsin = stimuli.TimeSeries(tsamplein, np.transpose(videoin, rollaxes))
    video.save_video(tsin, 'myvideo.mp4', fps=fpsout)
    npt.assert_equal(tsin.tsample, tsamplein)
    tsout = video.load_video('myvideo.mp4', as_timeseries=True)
    npt.assert_equal(video.load_video_framerate('myvideo.mp4'), fpsout)
    npt.assert_equal(isinstance(tsout, stimuli.TimeSeries), True)
    npt.assert_almost_equal(tsout.tsample, tsampleout)

    # Also verify the actual data
    tsres = tsin.resample(tsampleout)
    npt.assert_equal(tsout.shape, tsres.shape)
    npt.assert_almost_equal(tsout.data / 255.0, tsres.data / tsres.data.max(),
                            decimal=0)
    os.remove('myvideo.mp4')

    with pytest.raises(TypeError):
        video.save_video([2, 3, 4], 'invalid.avi')

    # Trigger an import error
    with mock.patch.dict("sys.modules", {"skvideo": {}}):
        with pytest.raises(ImportError):
            reload(video)
            video.save_video(videoin, 'invalid.avi')
    with mock.patch.dict("sys.modules", {"skimage": {}}):
        with pytest.raises(ImportError):
            reload(video)
            video.save_video(videoin, 'invalid.avi')


@pytest.mark.skip(reason='ffmpeg dependency')
def test_save_video_sidebyside():
    reload(video)
    from skvideo import datasets
    videoin = video.load_video(datasets.bikes(), as_timeseries=False)
    fps = video.load_video_framerate(datasets.bikes())
    tsample = 1.0 / float(fps)
    rollaxes = np.roll(range(videoin.ndim), -1)
    percept = stimuli.TimeSeries(tsample, np.transpose(videoin, rollaxes))

    video.save_video_sidebyside(datasets.bikes(), percept, 'mymovie.mp4',
                                fps=fps)
    videout = video.load_video('mymovie.mp4', as_timeseries=False)
    npt.assert_equal(videout.shape[0], videoin.shape[0])
    npt.assert_equal(videout.shape[1], videoin.shape[1])
    npt.assert_equal(videout.shape[2], videoin.shape[2] * 2)
    npt.assert_equal(videout.shape[3], videoin.shape[3])
    os.remove('mymovie.mp4')

    with pytest.raises(TypeError):
        video.save_video_sidebyside(datasets.bikes(), [2, 3, 4], 'invalid.avi')

    with mock.patch.dict("sys.modules", {"skvideo": {}}):
        with pytest.raises(ImportError):
            reload(video)
            video.save_video_sidebyside(datasets.bikes(), percept,
                                        'invalid.avi')
    with mock.patch.dict("sys.modules", {"skimage": {}}):
        with pytest.raises(ImportError):
            reload(video)
            video.save_video_sidebyside(datasets.bikes(), percept,
                                        'invalid.avi')
