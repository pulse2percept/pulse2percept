"""
Computational models of the prosthetic vision, such as phosphene and neural response models. Cortical models are in the cortex submodule.


.. autosummary::
    :toctree: _api

    cortex
    base
    temporal
    thompson2003
    horsager2009
    nanduri2012
    beyeler2019
    granley2021

.. seealso::

    *  :ref:`Basic Concepts > Computational Models <topics-models>`

"""
from .base import (BaseModel, Model, NotBuiltError, SpatialModel,
                   TemporalModel, TorchBaseModel)
from .temporal import FadingTemporal, TorchFadingTemporal
from .beyeler2019 import (ScoreboardModel, ScoreboardSpatial, AxonMapSpatial,
                          AxonMapModel)
from .horsager2009 import Horsager2009Model, Horsager2009Temporal
from .nanduri2012 import (Nanduri2012Model, Nanduri2012Spatial,
                          Nanduri2012Temporal)
from .granley2021 import BiphasicAxonMapModel, BiphasicAxonMapSpatial
from .thompson2003 import Thompson2003Model, Thompson2003Spatial

from . import cortex

__all__ = [
    'AxonMapModel',
    'AxonMapSpatial',
    'BaseModel',
    'cortex',
    'FadingTemporal',
    'Horsager2009Model',
    'Horsager2009Temporal',
    'Model',
    'Nanduri2012Model',
    'Nanduri2012Spatial',
    'Nanduri2012Temporal',
    'BiphasicAxonMapModel',
    'NotBuiltError',
    'ScoreboardModel',
    'ScoreboardSpatial',
    'SpatialModel',
    'TemporalModel',
    'TorchBaseModel',
    'TorchFadingTemporal',
    'Thompson2003Model',
    'Thompson2003Spatial',
]
