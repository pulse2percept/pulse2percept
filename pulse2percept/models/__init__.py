"""Computational models of the retina, such as phosphene and neural response models.

.. autosummary::
    :toctree: _api

    base
    scoreboard
    axon_map
    nanduri2012
    watson2014

"""
from .base import BaseModel, NotBuiltError
from .watson2014 import (Watson2014ConversionMixin, dva2ret, ret2dva,
                         Watson2014DisplacementMixin)
from .scoreboard import ScoreboardModel
from .axon_map import AxonMapModel
from .nanduri2012 import (Nanduri2012Model, Nanduri2012SpatialMixin,
                          Nanduri2012TemporalMixin)

__all__ = [
    'AxonMapModel',
    'BaseModel',
    'dva2ret',
    'Nanduri2012Model',
    'Nanduri2012SpatialMixin',
    'Nanduri2012TemporalMixin',
    'NotBuiltError',
    'ret2dva',
    'ScoreboardModel',
    'Watson2014ConversionMixin',
    'Watson2014DisplacementMixin'
]
