"""Utilities to download and import datasets.

*  **Dataset loaders** can be used to load small datasets that come
   pre-packaged with the pulse2percept software.
*  **Dataset fetchers** can be used to download larger datasets from a given
   URL and directly import them into pulse2percept.

.. autosummary::
    :toctree: _api

    base
    horsager2009
    nanduri2012
    perezfornos2012
    beyeler2019

.. seealso::

    *  :ref:`Basic Concepts > Datasets <topics-datasets>`

"""

from .base import clear_data_dir, get_data_dir, fetch_url
from .beyeler2019 import fetch_beyeler2019
from .horsager2009 import load_horsager2009
from .nanduri2012 import load_nanduri2012
from .perezfornos2012 import load_perezfornos2012


__all__ = [
    'clear_data_dir',
    'fetch_url',
    'fetch_beyeler2019',
    'get_data_dir',
    'load_horsager2009',
    'load_nanduri2012',
    'load_perezfornos2012',
]
