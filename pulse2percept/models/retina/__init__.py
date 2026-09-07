"""Computational models of retinal stimulation

Phosphene and neural-response models for epiretinal, subretinal and
suprachoroidal devices, plus the retinal spatial base class they share.

.. autosummary::
    :toctree: _api

    base
    thompson2003
    horsager2009
    nanduri2012
    beyeler2019
    granley2021

.. seealso::

    *  :ref:`Basic Concepts > Computational Models <topics-models>`

"""
from .base import RetinalSpatial
from .beyeler2019 import (AxonMapModel, AxonMapSpatial, ScoreboardModel,
                          ScoreboardSpatial)
from .granley2021 import (BiphasicAxonMapModel, BiphasicAxonMapSpatial,
                          BiphasicScoreboardModel, BiphasicScoreboardSpatial)
from .horsager2009 import Horsager2009Model, Horsager2009Temporal
from .nanduri2012 import (Nanduri2012Model, Nanduri2012Spatial,
                          Nanduri2012Temporal)
from .thompson2003 import Thompson2003Model, Thompson2003Spatial

__all__ = [
    'AxonMapModel',
    'AxonMapSpatial',
    'BiphasicAxonMapModel',
    'BiphasicAxonMapSpatial',
    'BiphasicScoreboardModel',
    'BiphasicScoreboardSpatial',
    'Horsager2009Model',
    'Horsager2009Temporal',
    'Nanduri2012Model',
    'Nanduri2012Spatial',
    'Nanduri2012Temporal',
    'RetinalSpatial',
    'ScoreboardModel',
    'ScoreboardSpatial',
    'Thompson2003Model',
    'Thompson2003Spatial',
]
