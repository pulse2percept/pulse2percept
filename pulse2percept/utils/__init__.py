"""Various utility and helper functions.

.. autosummary::
    :toctree: _api

    base
    geometry
    convolution
    deprecation
    optimize
    parallel

"""
from .base import PrettyPrint, FreezeError, Frozen, Data, gamma
from .geometry import (GridXY, RetinalCoordTransform, Curcio1990Transform,
                       Watson2014Transform, Watson2014DisplaceTransform,
                       cart2pol, pol2cart)
from .convolution import center_vector, conv
from .deprecation import deprecated
from .optimize import bisect
from .parallel import parfor

__all__ = [
    'bisect',
    'cart2pol',
    'Curcio1990Transform',
    'Data',
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
