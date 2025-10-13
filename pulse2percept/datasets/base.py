""":py:class:`~pulse2percept.datasets.get_data_dir`, 
   :py:class:`~pulse2percept.datasets.clear_data_dir`, 
   :py:class:`~pulse2percept.datasets.has_network`, 
   :py:class:`~pulse2percept.datasets.osf_is_reachable`,
   :py:class:`~pulse2percept.datasets.download_from_osf`,
   :py:class:`~pulse2percept.datasets.fetch_url`"""
import sys
from os import environ, makedirs
from os.path import exists, expanduser, join
from shutil import rmtree
import hashlib
import ssl
from urllib.request import urlretrieve, urlopen, Request
import socket
import re


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


def has_network(timeout=3.0):
    """Return True if we appear to have general network connectivity.

    Tries a few common endpoints; success on ANY is considered 'online'.
    Keeps it simple and fast; no DNS resolution required for the IP checks.
    """
    checks = [
        ("1.1.1.1", 53),     # Cloudflare DNS (TCP)
        ("8.8.8.8", 53),     # Google DNS (TCP)
        ("example.com", 80), # Plain HTTP
        ("osf.io", 443),     # OSF over HTTPS
    ]
    for host, port in checks:
        try:
            sock = socket.create_connection((host, port), timeout=timeout)
            sock.close()
            return True
        except OSError:
            continue
    return False


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
    sys.stdout.write(f"\rDownloading {progress_size / (1024 * 1024):.1f}"
                     f"/{total_size / (1024 * 1024):.1f} MB ({percent}%)")
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
    # Hacky way to keep using ulretrieve without SSL verification:
    ssl._create_default_https_context = ssl._create_unverified_context
    urlretrieve(url, file_path, progress_bar)
    checksum = _sha256(file_path)
    if remote_checksum != None and remote_checksum != checksum:
        raise IOError(f"{file_path} has an SHA256 checksum ({checksum}) "
                      f"differing from expected ({remote_checksum}), "
                      f"file may be corrupted.")


def osf_is_reachable(test_url="https://osf.io/rduj4", timeout=5.0):
    # Convert bare GUID to /download if needed
    m = re.match(r"^https?://(?:www\.)?osf\.io/([a-z0-9]+)/?$", test_url, flags=re.I)
    if m:
        test_url = "https://osf.io/%s/download" % m.group(1)

    try:
        # HEAD first
        req = Request(test_url, method="HEAD")
        with urlopen(req, timeout=timeout) as resp:
            code = getattr(resp, "status", 200)
            if 200 <= code < 400:
                return True
    except Exception:
        pass

    # Fallback: GET a few bytes
    try:
        with urlopen(test_url, timeout=timeout) as resp:
            return len(resp.read(64)) > 0
    except Exception:
        return False


def _normalize_osf_download(osf_id_or_url):
    """Return a direct OSF download URL from a GUID or URL."""
    if re.match(r"^https?://", osf_id_or_url, flags=re.I):
        m = re.match(r"^(https?://(?:www\.)?osf\.io/[a-z0-9]+)/?$", osf_id_or_url, flags=re.I)
        if m:
            return m.group(1) + "/download"
        return osf_id_or_url  # may already include /download
    return f"https://osf.io/{osf_id_or_url}/download"


def download_from_osf(osf_id_or_url, filename, checksum=None,
                      data_path=None, download_if_missing=True,
                      progress_bar=_report_hook):
    """Download once from OSF into the data dir, via fetch_url."""
    data_path = get_data_dir(data_path)
    file_path = join(data_path, filename)

    if exists(file_path):
        return file_path
    if not download_if_missing:
        raise IOError(f"No local file {file_path} found")

    # quick preflight checks (fast + minimal)
    if not has_network():
        raise IOError("No internet connection.")
    if not osf_is_reachable():  # probes your tiny OSF file
        raise IOError("OSF downloads appear unavailable right now.")

    url = _normalize_osf_download(osf_id_or_url)
    fetch_url(url, file_path, progress_bar=progress_bar, remote_checksum=checksum)
    return file_path
