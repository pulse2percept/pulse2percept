"""Datasets of beyeler2019 and Helper functions

.. autosummary::
    :toctree: _api

    base
"""

from .base import fetch_url
from ._beyeler2019 import fetch_beyeler2019

try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False

if not has_pandas:
    raise ImportError("You do not have pandas installed. "
                      "You can install it via $ pip install pandas.")

try:
    import h5py
    has_h5py = True
except ImportError:
    has_h5py = False

if not has_h5py:
    raise ImportError("You do not have h5py installed. "
                      "You can install it via $ pip install h5py.")

__all__ = [
    'fetch_url',
    'fetch_beyeler2019',
]