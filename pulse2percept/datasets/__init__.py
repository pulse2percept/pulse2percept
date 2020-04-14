"""Utilities to download and import datasets.

The ``datasets`` subpackage provides two kinds of helper functions
that can be used to load datasets from the bionic vision community:

*  **Dataset loaders** can be used to load small datasets that come
   pre-packaged with the pulse2percept software.
*  **Dataset fetchers** can be used to download larger datasets from a given
   URL and directly import them into pulse2percept.

.. autosummary::
    :toctree: _api

    base
    horsager2009
    beyeler2019
"""

from .base import clear_data_dir, get_data_dir, fetch_url
from .beyeler2019 import fetch_beyeler2019
from .horsager2009 import load_horsager2009


__all__ = [
    'clear_data_dir',
    'fetch_url',
    'fetch_beyeler2019',
    'get_data_dir',
    'load_horsager2009'
]
