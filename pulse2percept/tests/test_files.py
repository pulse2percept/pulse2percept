from pulse2percept import files
import numpy.testing as npt


def test_find_files_like():
    # Not sure how to check a valid pattern match, because we
    # don't know in which directory the test suite is
    # executed... so smoke test
    files.find_files_like(".", ".*")

    # Or fail to find an unlikely file name
    filenames = files.find_files_like(".", "theresnowaythisexists\.xyz")
    npt.assert_equal(filenames, [])
