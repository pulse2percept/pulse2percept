"""Various utility and helper functions.

.. autosummary::
    :toctree: _api

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

"""
from .base import (PrettyPrint, FreezeError, Frozen, Data, bijective26_name,
                   cached, gamma)
from .geometry import (Grid2D, VisualFieldMap, Curcio1990Map,
                       Watson2014Map, Watson2014DisplaceMap,
                       NoisyMap,
                       cart2pol, pol2cart, delta_angle)
from .array import is_strictly_increasing, radial_mask, sample, unique
from .images import center_image, scale_image, shift_image, trim_image
from .convolution import center_vector, conv
from .optimize import bisect
from .stats import r2_score, circ_r2_score
from .parallel import parfor
from .deprecation import deprecated

__all__ = [
    'bijective26_name',
    'bisect',
    'cached',
    'cart2pol',
    'center_image',
    'center_vector',
    'circ_r2_score',
    'conv',
    'Curcio1990Map',
    'Data',
    'delta_angle',
    'deprecated',
    'FreezeError',
    'Frozen',
    'gamma',
    'Grid2D',
    'is_strictly_increasing',
    'NoisyMap',
    'parfor',
    'pol2cart',
    'PrettyPrint',
    'r2_score',
    'radial_mask',
    'sample',
    'scale_image',
    'shift_image',
    'trim_image',
    'unique',
    'VisualFieldMap',
    'Watson2014DisplaceMap',
    'Watson2014Map'
]
