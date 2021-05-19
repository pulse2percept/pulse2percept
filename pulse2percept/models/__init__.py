"""Computational models of the retina, such as phosphene and neural response models.

.. autosummary::
    :toctree: _api

    base
    temporal
    beyeler2019
    nanduri2012
    horsager2009

.. seealso::

    *  :ref:`Basic Concepts > Computational Models <topics-models>`

"""
from .base import (BaseModel, Model, NotBuiltError, SpatialModel,
                   TemporalModel)
from .temporal import FadingTemporal
from .beyeler2019 import (ScoreboardModel, ScoreboardSpatial, AxonMapSpatial,
                          AxonMapModel)
from .horsager2009 import Horsager2009Model, Horsager2009Temporal
from .nanduri2012 import (Nanduri2012Model, Nanduri2012Spatial,
                          Nanduri2012Temporal)

__all__ = [
    'AxonMapModel',
    'AxonMapSpatial',
    'BaseModel',
    'FadingTemporal',
    'Horsager2009Model',
    'Horsager2009Temporal',
    'Model',
    'Nanduri2012Model',
    'Nanduri2012Spatial',
    'Nanduri2012Temporal',
    'NotBuiltError',
    'ScoreboardModel',
    'ScoreboardSpatial',
    'SpatialModel',
    'TemporalModel',
]
