"""Various utility and helper functions.

.. autosummary::
    :toctree: _api
    :no-index:

    base
    constants
    geometry
    array
    images
    convolution
    optimize
    stats
    parallel
    deprecation
    three_dim

"""
from .base import (PrettyPrint, FreezeError, Frozen, Data, bijective26_name,
                   cached, gamma)
from .geometry import (cart2pol, pol2cart, delta_angle)
from .array import is_strictly_increasing, radial_mask, sample, unique
from .images import center_image, scale_image, shift_image, trim_image
from .convolution import center_vector, conv
from .optimize import bisect
from .stats import r2_score, circ_r2_score
from .parallel import parfor
from .deprecation import deprecated
from .three_dim import parse_3d_orient

__all__ = [
    'bijective26_name',
    'bisect',
    'cached',
    'cart2pol',
    'center_image',
    'center_vector',
    'circ_r2_score',
    'conv',
    'Data',
    'delta_angle',
    'deprecated',
    'FreezeError',
    'Frozen',
    'gamma',
    'is_strictly_increasing',
    'parfor',
    'parse_3d_orient',
    'pol2cart',
    'PrettyPrint',
    'r2_score',
    'radial_mask',
    'sample',
    'scale_image',
    'shift_image',
    'trim_image',
    'unique',
]
