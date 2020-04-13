"""`fetch_url`"""
import hashlib
from urllib.request import urlretrieve
from os import path
import csv
import numpy as np
import pandas as pd

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

def fetch_url(url, file_path, remote_checksum=None):
    """Helper function to download a remote dataset into path
    Fetch a dataset pointed by url, save into path using the file_path
    and ensure its integrity based on the SHA256 Checksum of the
    downloaded file.

    Parameters
    ----------
    url : string
        url to the file.
    file_path: string
        Full path of the created file.
    """
    urlretrieve(url, file_path)
    checksum = _sha256(file_path)
    if remote_checksum != None and remote_checksum != checksum:
        raise IOError("{} has an SHA256 checksum ({}) "
                      "differing from expected ({}), "
                      "file may be corrupted.".format(file_path, checksum,
                                                      remote_checksum))