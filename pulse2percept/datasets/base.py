"""`get_data_dir`, `clear_data_dir`, `fetch_url`"""
import sys
from os import environ, makedirs
from os.path import exists, expanduser, join
from shutil import rmtree
import hashlib
from urllib.request import urlretrieve


def get_data_dir(data_dir=None):
    """Return the path of the pulse2percept data directory

    This directory is used to store the datasets retrieved by the data fetch
    utility functions to avoid downloading the data several times.

    By default, this is set to a directory called 'pulse2percept_data' in the
    user home directory.
    Alternatively, it can be set by a ``PULSE2PERCEPT_DATA`` environment
    variable or set programmatically by specifying a path.

    If the directory does not already exist, it is automatically created.

    .. versionadded:: 0.6

    Parameters
    ----------
    data_dir : str or None
        The path to the pulse2percept data directory.

    """
    if data_dir is None:
        data_dir = environ.get('PULSE2PERCEPT_DATA',
                               join('~', 'pulse2percept_data'))
    data_dir = expanduser(data_dir)
    if not exists(data_dir):
        makedirs(data_dir)
    return data_dir


def clear_data_dir(data_dir=None):
    """Delete all content in the data directory

    By default, this is set to a directory called 'pulse2percept_data' in the
    user home directory.
    Alternatively, it can be set by a ``PULSE2PERCEPT_DATA`` environment
    variable or set programmatically by specifying a path.

    .. versionadded:: 0.6

    Parameters
    ----------
    data_dir : str or None
        The path to the pulse2percept data directory.

    """
    data_dir = get_data_dir(data_dir)
    rmtree(data_dir)


def _sha256(path):
    """Calculate the sha256 hash of the file at path."""
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(path, "rb") as f:
        while True:
            buffer = f.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)
    return sha256hash.hexdigest()


def _report_hook(count, block_size, total_size):
    """Display a progress bar for ``urlretrieve``"""
    progress_size = int(count * block_size)
    percent = min(100, int(count * block_size * 100 / total_size))
    sys.stdout.write(f"\rDownloading {progress_size / (1024 * 1024)}"
                     f"/{total_size / (1024 * 1024)} MB ({percent}%)"
    sys.stdout.flush()


def fetch_url(url, file_path, progress_bar=_report_hook, remote_checksum=None):
    """Download a remote file

    Fetch a dataset pointed to by ``url``, check its SHA-256 checksum for
    integrity, and save it to ``file_path``.

    .. versionadded:: 0.6

    Parameters
    ----------
    url : string
        URL of file to download
    file_path: string
        Path to the local file that will be created
    progress_bar : func callback, optional
        A callback to a function ``func(count, block_size, total_size)`` that
        will display a progress bar.
    remote_checksum : str, optional
        The expected SHA-256 checksum of the file.

    """
    urlretrieve(url, file_path, progress_bar)
    checksum = _sha256(file_path)
    if remote_checksum != None and remote_checksum != checksum:
        raise IOError(f"{file_path} has an SHA256 checksum ({checksum}) "
                      f"differing from expected ({remote_checksum}), "
                      f"file may be corrupted.")
