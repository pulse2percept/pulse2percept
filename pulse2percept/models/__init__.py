"""Computational models of the retina, such as phosphene and neural response models.

.. autosummary::
    :toctree: _api

    base
    scoreboard
    axon_map
    nanduri2012

.. seealso::

    *  :ref:`Basic Concepts > Computational Models <topics-models>`

"""
from .base import NotBuiltError, Model, SpatialModel, TemporalModel
from .scoreboard import ScoreboardModel, ScoreboardSpatial
from .axon_map import AxonMapModel, AxonMapSpatial
from .nanduri2012 import (Nanduri2012Model, Nanduri2012Spatial,
                          Nanduri2012Temporal)

__all__ = [
    'AxonMapModel',
    'AxonMapSpatial',
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
