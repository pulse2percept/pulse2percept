"""Utility classes and functions used across pulse2percept.

Base Classes
------------

.. autosummary::
    :toctree:

    PrettyPrint
    Frozen
    FreezeError
    Parametrized
    Data
    cached
    base.freeze_class
    base.has_own_attr

Arrays
------

.. autosummary::
    :toctree:

    center_vector
    is_strictly_increasing
    radial_mask
    sample
    unique
    bijective26_name

Images
------

.. autosummary::
    :toctree:

    center_image
    scale_image
    shift_image
    trim_image

Geometry
--------

.. autosummary::
    :toctree:

    cart2pol
    pol2cart
    delta_angle
    parse_3d_orient

Numerics
--------

.. autosummary::
    :toctree:

    bisect
    conv
    gamma
    r2_score
    circ_r2_score

Animation
---------

.. autosummary::
    :toctree:

    HTMLAnimation
    frame_interval

Deprecation
-----------

.. autosummary::
    :toctree:

    deprecated
    deprecate_parameter
    deprecated_alias
    rename_parameter
    warn_deprecated_params
    rename_deprecated_params
    deprecation.is_deprecated

"""
from .base import (PrettyPrint, FreezeError, Frozen, Parametrized, Data,
                   bijective26_name, cached, gamma)
from .animation import HTMLAnimation, frame_interval
from .geometry import (cart2pol, pol2cart, delta_angle)
from .array import is_strictly_increasing, radial_mask, sample, unique
from .images import center_image, scale_image, shift_image, trim_image
from .convolution import center_vector, conv
from .optimize import bisect
from .stats import r2_score, circ_r2_score
from .deprecation import (deprecated, deprecate_parameter, deprecated_alias,
                          rename_parameter, warn_deprecated_params,
                          rename_deprecated_params)
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
    'deprecate_parameter',
    'deprecated',
    'deprecated_alias',
    'frame_interval',
    'FreezeError',
    'Frozen',
    'gamma',
    'HTMLAnimation',
    'is_strictly_increasing',
    'Parametrized',
    'parse_3d_orient',
    'pol2cart',
    'PrettyPrint',
    'r2_score',
    'radial_mask',
    'rename_deprecated_params',
    'rename_parameter',
    'sample',
    'scale_image',
    'shift_image',
    'trim_image',
    'unique',
    'warn_deprecated_params',
]
