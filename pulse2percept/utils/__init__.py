"""Various utility and helper functions.

.. autosummary::
    :toctree: _api

    base
    convolution
    deprecation
    parallel

"""
from .base import (Frozen, FreezeError, PrettyPrint, GridXY, cart2pol,
                   find_files_like, gamma, pol2cart)
from .convolution import center_vector, conv
from .deprecation import deprecated
from .parallel import parfor

__all__ = [
    'cart2pol',
    'deprecated',
    'find_files_like',
    'FreezeError',
    'Frozen',
    'gamma',
    'GridXY',
    'parfor',
    'pol2cart',
    'PrettyPrint'
]
