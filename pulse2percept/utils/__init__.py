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
from .base import PrettyPrint, FreezeError, Frozen, Data, gamma, unique
from .geometry import (Grid2D, RetinalCoordTransform, Curcio1990Transform,
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
    'Grid2D',
    'parfor',
    'pol2cart',
    'PrettyPrint',
    'RetinalCoordTransform',
    'unique',
    'Watson2014DisplaceTransform',
    'Watson2014Transform'
]
