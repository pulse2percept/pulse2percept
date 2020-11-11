"""Various utility and helper functions.

.. autosummary::
    :toctree: _api

    base
    geometry
    array
    convolution
    optimize
    stats
    parallel
    deprecation

"""
from .base import PrettyPrint, FreezeError, Frozen, Data, cached, gamma
from .geometry import (Grid2D, RetinalCoordTransform, Curcio1990Transform,
                       Watson2014Transform, Watson2014DisplaceTransform,
                       cart2pol, pol2cart, delta_angle)
from .array import radial_mask, unique
from .convolution import center_vector, conv
from .optimize import bisect
from .stats import r2_score, circ_r2_score
from .parallel import parfor
from .deprecation import deprecated

__all__ = [
    'bisect',
    'cached',
    'cart2pol',
    'circ_r2_score',
    'Curcio1990Transform',
    'Data',
    'delta_angle',
    'deprecated',
    'FreezeError',
    'Frozen',
    'gamma',
    'Grid2D',
    'parfor',
    'pol2cart',
    'PrettyPrint',
    'r2_score',
    'radial_mask',
    'RetinalCoordTransform',
    'unique',
    'Watson2014DisplaceTransform',
    'Watson2014Transform'
]
