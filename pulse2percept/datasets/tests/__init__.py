"""Utilities to test loaded datasets.

.. autosummary::
    :toctree: _api

    base
    test_base
    test_horsager2009
    test_beyeler2019
    test_nanduri2012

.. seealso::

    *  :ref:`Basic Concepts > Datasets <topics-datasets>`

"""

from .test_base import _remove_dir, tmp_data_dir, test_data_dir, test_fetch_url
from .test_beyeler2019 import test_fetch_beyeler2019, _is_beyeler2019_not_available
from .test_horsager2009 import test_load_horsager2009
from .test_nanduri2012 import test_load_nanduri2012


__all__ = [
    '_remove_dir',
    'tmp_data_dir',
    'test_data_dir',
    'test_fetch_url'
    'test_fetch_beyeler2019',
    '_is_beyeler2019_not_available',
    'test_load_horsager2009',
    'test_load_nanduri2012',
]
