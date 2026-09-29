"""Models of retinal stimulation.

Base class
----------

.. autosummary::
    :toctree:

    RetinalSpatial

Scoreboard and Axon Map Models
------------------------------

.. autosummary::
    :toctree:

    ScoreboardModel
    ScoreboardSpatial
    AxonMapModel
    AxonMapSpatial
    Thompson2003Model
    Thompson2003Spatial

Pulse-Dependent Models
----------------------

.. autosummary::
    :toctree:

    BiphasicScoreboardModel
    BiphasicScoreboardSpatial
    BiphasicAxonMapModel
    BiphasicAxonMapSpatial
    Nanduri2012Model
    Nanduri2012Spatial
    Nanduri2012Temporal
    Horsager2009Model
    Horsager2009Temporal

Biphasic Effect Models
----------------------

Brightness, size, and streak scaling used by the biphasic
models.

.. autosummary::
    :toctree:

    granley2021.DefaultBrightModel
    granley2021.DefaultSizeModel
    granley2021.DefaultStreakModel

Photovoltaic Models
-------------------

.. autosummary::
    :toctree:

    Ho2018Model
    Ho2018Spatial
    Ho2018Temporal

.. seealso::

    *  :ref:`Core Concepts > Models and Percepts <topics-models>`

"""
from .base import RetinalSpatial
from .beyeler2019 import (AxonMapModel, AxonMapSpatial, ScoreboardModel,
                          ScoreboardSpatial)
from .granley2021 import (BiphasicAxonMapModel, BiphasicAxonMapSpatial,
                          BiphasicScoreboardModel, BiphasicScoreboardSpatial)
from .ho2018 import Ho2018Model, Ho2018Spatial, Ho2018Temporal
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
    'Ho2018Model',
    'Ho2018Spatial',
    'Ho2018Temporal',
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
