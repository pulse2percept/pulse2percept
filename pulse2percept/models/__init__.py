"""Computational models of the retina, such as phosphene and neural response models.

.. autosummary::
    :toctree: _api

    base
    beyeler2019
    nanduri2012

.. seealso::

    *  :ref:`Basic Concepts > Computational Models <topics-models>`

"""
from .base import (BaseModel, Model, NotBuiltError, SpatialModel,
                   TemporalModel)
from .beyeler2019 import (ScoreboardModel, ScoreboardSpatial, AxonMapSpatial,
                          AxonMapModel)
from .nanduri2012 import (Nanduri2012Model, Nanduri2012Spatial,
                          Nanduri2012Temporal)

__all__ = [
    'AxonMapModel',
    'AxonMapSpatial',
    'BaseModel',
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
