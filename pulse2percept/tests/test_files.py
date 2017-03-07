import numpy as np
import numpy.testing as npt
import pytest
import os
try:
    from unittest import mock
except ImportError:
    import mock

from pulse2percept import files


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
        files.savemoviefiles("invalid.avi", np.zeros(10), path='./')


def test_npy2movie():
    # This function is deprecated

    if os.name != 'nt':
        # If not on Windows, this should break
        with pytest.raises(OSError):
            files.npy2movie("invalid.avi", np.zeros(10))
    else:
        # Trigger an import error
        with mock.patch.dict("sys.modules", {"PIL": {}}):
            with pytest.raises(ImportError):
                files.npy2movie("invalid.avi", np.zeros(10), path='./')

        # smoke test
        files.npy2movie("invalid.avi", np.zeros(10))


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
