import numpy.testing as npt
import os
from shutil import rmtree
import pytest

from pulse2percept.datasets import get_data_dir, clear_data_dir, fetch_url


def _remove_dir(path):
    if os.path.isdir(path):
        rmtree(path)


@pytest.fixture(scope="module")
def tmp_data_dir(tmpdir_factory):
    tmp_file = str(tmpdir_factory.mktemp("p2p_tmp_data_dir"))
    yield tmp_file
    _remove_dir(tmp_file)


def test_data_dir(tmp_data_dir):
    # Create a temporary data directory:
    data_dir = get_data_dir(data_dir=tmp_data_dir)
    npt.assert_equal(data_dir, tmp_data_dir)
    npt.assert_equal(os.path.exists(data_dir), True)

    # Delete both the content and the folder itself:
    clear_data_dir(data_dir=data_dir)
    npt.assert_equal(os.path.exists(data_dir), False)

    # If the folder is missing, it will be created again:
    data_dir = get_data_dir(data_dir=data_dir)
    npt.assert_equal(os.path.exists(data_dir), True)


def test_fetch_url(tmp_data_dir):
    url1 = 'https://www.nature.com/articles/s41598-019-45416-4.pdf'
    file_path1 = os.path.join(tmp_data_dir, 'paper1.pdf')
    paper_checksum1 = 'e8a2db25916cdd15a4b7be75081ef3e57328fa5f335fb4664d1fb7090dcd6842'
    fetch_url(url1, file_path1, remote_checksum=paper_checksum1)
    npt.assert_equal(os.path.exists(file_path1), True)

    url2 = 'https://bionicvisionlab.org/publication/2019-optimal-surgical-placement/2019-optimal-surgical-placement.pdf'
    file_path2 = os.path.join(tmp_data_dir, 'paper2.pdf')
    paper_checksum2 = 'e2d0cbecc9c2826f66f60576b44fe18ad6a635d394ae02c3f528b89cffcd9450'
    # Use wrong checksum:
    with pytest.raises(IOError):
        fetch_url(url2, file_path2, remote_checksum=paper_checksum1)
    # Use correct checksum:
    fetch_url(url2, file_path2, remote_checksum=paper_checksum2)
    npt.assert_equal(os.path.exists(file_path2), True)
