"""Various utility and helper functions.

.. autosummary::
    :toctree: _api

    base
    geometry
    array
    images
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
from .images import center_image, scale_image, shift_image, trim_image
from .convolution import center_vector, conv
from .optimize import bisect
from .stats import r2_score, circ_r2_score
from .parallel import parfor
from .deprecation import deprecated

__all__ = [
    'bisect',
    'cached',
    'cart2pol',
    'center_image',
    'center_vector',
    'circ_r2_score',
    'conv',
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
    'scale_image',
    'shift_image',
    'trim_image',
    'unique',
    'Watson2014DisplaceTransform',
    'Watson2014Transform'
]
