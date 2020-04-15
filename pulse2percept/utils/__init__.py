"""Various utility and helper functions.

.. autosummary::
    :toctree: _api

    base
    geometry
    convolution
    deprecation
    parallel

"""
from .base import PrettyPrint, FreezeError, Frozen, gamma
from .geometry import (GridXY, RetinalCoordTransform, Curcio1990Transform,
                       Watson2014Transform, Watson2014DisplaceTransform,
                       cart2pol, pol2cart)
from .convolution import center_vector, conv
from .deprecation import deprecated
from .parallel import parfor

__all__ = [
    'cart2pol',
    'Curcio1990Transform',
    'deprecated',
    'FreezeError',
    'Frozen',
    'gamma',
    'GridXY',
    'parfor',
    'pol2cart',
    'PrettyPrint',
    'RetinalCoordTransform',
    'Watson2014DisplaceTransform',
    'Watson2014Transform'
]
