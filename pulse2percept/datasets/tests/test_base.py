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
    url = 'https://bionicvisionlab.org/publications/2017-pulse2percept/2017-pulse2percept.pdf'
    file_path = os.path.join(tmp_data_dir, 'paper2.pdf')
    paper_checksum = '21fd40c6a3f6ae4f09838dc972b5caa5a7d5448bdced454285d2a5fa6cf0cf49'
    # Use wrong checksum:
    with pytest.raises(IOError):
        fetch_url(url, file_path, remote_checksum='abcdef')
    # Use correct checksum:
    fetch_url(url, file_path, remote_checksum=paper_checksum)
    npt.assert_equal(os.path.exists(file_path), True)
