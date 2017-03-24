import numpy as np
import numpy.testing as npt
import pytest
import os
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

from pulse2percept import files
from pulse2percept import utils


def test_set_skvideo_path():
    # Smoke-test
    files.set_skvideo_path('/usr/bin')
    files.set_skvideo_path(libav_path='/usr/bin')


def test_load_video():
    # Load a test example
    from skvideo import datasets
    video = files.load_video(datasets.bikes())
    npt.assert_equal(video.shape, [250, 272, 640, 3])

    # Trigger an import error
    with mock.patch.dict("sys.modules", {"skvideo": {}, "skvideo.utils": {}}):
        with pytest.raises(ImportError):
            reload(files)
            files.load_video('invalid.avi')


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


def test_save_video():
    # Load a test example
    reload(files)
    from skvideo import datasets
    video = files.load_video(datasets.bikes())

    # Smoke-test
    files.save_video('myvideo.avi', video)
    files.save_video('myvideo.mp4', video)

    # Trigger an import error
    with mock.patch.dict("sys.modules", {"skvideo": {}, "skvideo.utils": {}}):
        with pytest.raises(ImportError):
            reload(files)
            files.save_video('invalid.avi', video)


def test_save_percept():
    # Smoke-test
    reload(files)
    pt = utils.TimeSeries(1, np.zeros((10, 8, 3)))
    files.save_percept('mypercept.avi', pt)
    files.save_percept('mypercept.mp4', pt, max_contrast=False)

    # Trigger an import error
    with mock.patch.dict("sys.modules", {"skvideo": {}, "skvideo.utils": {}}):
        with pytest.raises(ImportError):
            reload(files)
            files.save_percept('invalid.avi', pt)

    # Trigger a TypeError
    reload(files)
    with pytest.raises(TypeError):
        files.save_percept('invalid.avi', np.zeros((10, 8, 3)))


def test_savemoviefiles():
    # This function is deprecated

    if os.name != 'nt':
        # If not on Windows, this should break
        with pytest.raises(OSError):
            files.savemoviefiles("invalid.avi", np.zeros(10), path='./')
    else:
        # Trigger an import error
        with mock.patch.dict("sys.modules", {"PIL": {}}):
            with pytest.raises(ImportError):
                files.savemoviefiles("invalid.avi", np.zeros(10), path='./')

        # smoke test
        with pytest.warns(UserWarning):
            files.savemoviefiles("invalid.avi", np.zeros(10), path='./')


def test_npy2movie():
    # This function is deprecated

    if os.name != 'nt':
        # If not on Windows, this should break
        with pytest.raises(OSError):
            files.npy2movie("invalid.avi", np.zeros(10), rate=30)
    else:
        # Trigger an import error
        with mock.patch.dict("sys.modules", {"PIL": {}}):
            with pytest.raises(ImportError):
                files.npy2movie("invalid.avi", np.zeros(10), rate=30)

        # smoke test
        with pytest.raises(UserWarning):
            files.npy2movie("invalid.avi", np.zeros(10), rate=30)


def test_scale():
    # This function is deprecated
    inarray = np.random.rand(100)
    for newmin in [0.0, -0.5]:
        for newmax in [1.0, 10.0]:
            scaled = files.scale(inarray, newmin=newmin, newmax=newmax)
            npt.assert_almost_equal(scaled.min(), newmin)
            npt.assert_almost_equal(scaled.max(), newmax)


def test_find_files_like():
    # Not sure how to check a valid pattern match, because we
    # don't know in which directory the test suite is
    # executed... so smoke test
    files.find_files_like(".", ".*")

    # Or fail to find an unlikely file name
    filenames = files.find_files_like(".", "theresnowaythisexists\.xyz")
    npt.assert_equal(filenames, [])
