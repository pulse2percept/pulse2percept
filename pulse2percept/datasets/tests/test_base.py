import numpy.testing as npt
import os
from shutil import rmtree
import pytest
import socket
import io
from urllib.error import URLError
from urllib.request import Request
import tempfile, os, hashlib
from pulse2percept.datasets.base import (
    get_data_dir, clear_data_dir, has_network, 
    osf_is_reachable, download_from_osf, _normalize_osf_download, 
    _sha256, _report_hook, fetch_url
)


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


def test_has_network_online_monkeypatch(monkeypatch):
    def _ok(addr, timeout=None):
        class _Sock:
            def close(self): pass
        return _Sock()
    monkeypatch.setattr("socket.create_connection", _ok)
    assert has_network() is True


def test_sha256_roundtrip():
    fd, p = tempfile.mkstemp()
    os.close(fd)
    try:
        data = b"abc123"
        with open(p, "wb") as f:
            f.write(data)
        h = hashlib.sha256(); h.update(data)
        assert _sha256(p) == h.hexdigest()
    finally:
        os.remove(p)


def test_report_hook_prints(monkeypatch, capsys):
    # total_size must be > 0 to avoid div by zero
    _report_hook(count=5, block_size=1024, total_size=10*1024)
    out = capsys.readouterr().out
    assert "Downloading" in out and "%" in out


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


def test_normalize_osf_download_variants():
    assert _normalize_osf_download("pf2ja").endswith("/pf2ja/download")
    assert _normalize_osf_download("https://osf.io/pf2ja").endswith("/pf2ja/download")
    assert _normalize_osf_download("https://osf.io/pf2ja/").endswith("/pf2ja/download")
    assert _normalize_osf_download("https://osf.io/pf2ja/download").endswith("/pf2ja/download")


def test_osf_is_reachable_head_success(monkeypatch):
    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): pass
    def _urlopen(req, timeout=None):
        assert isinstance(req, Request)
        return _Resp()
    monkeypatch.setattr("pulse2percept.datasets.base.urlopen", _urlopen)
    assert osf_is_reachable("https://osf.io/rduj4") is True


def test_osf_is_reachable_head_fail_get_success(monkeypatch):
    class _HeadErr(Exception): pass
    def _urlopen_head(req, timeout=None):
        # Simulate HEAD raising
        raise URLError("HEAD not supported")
    class _GetResp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self, n): return b"ok"
    calls = {"n": 0}
    def _urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:  # first call is HEAD
            raise URLError("HEAD not supported")
        return _GetResp()
    monkeypatch.setattr("pulse2percept.datasets.base.urlopen", _urlopen)
    assert osf_is_reachable("https://osf.io/rduj4") is True


def test_osf_is_reachable_total_fail(monkeypatch):
    def _urlopen(req, timeout=None):
        raise URLError("down")
    monkeypatch.setattr("pulse2percept.datasets.base.urlopen", _urlopen)
    assert osf_is_reachable("https://osf.io/rduj4") is False


def test_download_from_osf_returns_when_exists(tmp_path, monkeypatch):
    # Create existing file
    p = tmp_path / "han2021.zip"
    p.write_bytes(b"already here")
    # Make get_data_dir return tmp_path
    monkeypatch.setattr("pulse2percept.datasets.base.get_data_dir", lambda d=None: str(tmp_path))
    from pulse2percept.datasets.base import download_from_osf
    out = download_from_osf("pf2ja", "han2021.zip", checksum=None, data_path=str(tmp_path))
    assert out == str(p)


def test_download_from_osf_raises_when_not_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr("pulse2percept.datasets.base.get_data_dir", lambda d=None: str(tmp_path))
    with pytest.raises(IOError):
        download_from_osf("pf2ja", "han2021.zip", download_if_missing=False, data_path=str(tmp_path))


def test_download_from_osf_offline(tmp_path, monkeypatch):
    monkeypatch.setattr("pulse2percept.datasets.base.get_data_dir", lambda d=None: str(tmp_path))
    monkeypatch.setattr("pulse2percept.datasets.base.has_network", lambda: False)
    with pytest.raises(IOError):
        download_from_osf("pf2ja", "han2021.zip", data_path=str(tmp_path))


def test_download_from_osf_osf_down(tmp_path, monkeypatch):
    monkeypatch.setattr("pulse2percept.datasets.base.get_data_dir", lambda d=None: str(tmp_path))
    monkeypatch.setattr("pulse2percept.datasets.base.has_network", lambda: True)
    monkeypatch.setattr("pulse2percept.datasets.base.osf_is_reachable", lambda : False)
    with pytest.raises(IOError):
        download_from_osf("pf2ja", "han2021.zip", data_path=str(tmp_path))


def test_download_from_osf_calls_fetch_url(tmp_path, monkeypatch):
    monkeypatch.setattr("pulse2percept.datasets.base.get_data_dir", lambda d=None: str(tmp_path))
    monkeypatch.setattr("pulse2percept.datasets.base.has_network", lambda: True)
    monkeypatch.setattr("pulse2percept.datasets.base.osf_is_reachable", lambda : True)

    called = {}
    def _fake_fetch(url, file_path, progress_bar=None, remote_checksum=None):
        # Normalize expectation: ends with /download
        assert url.startswith("https://osf.io/28uqg") and url.endswith("/download")
        # Create the file to satisfy subsequent existence checks
        with open(file_path, "wb") as f:
            f.write(b"ok")
        called["args"] = (url, file_path, remote_checksum)
    monkeypatch.setattr("pulse2percept.datasets.base.fetch_url", _fake_fetch)

    out = download_from_osf("https://osf.io/28uqg", "beyeler2019.h5",
                            checksum="deadbeef", data_path=str(tmp_path))
    assert os.path.exists(out)
    assert called["args"][2] == "deadbeef"
