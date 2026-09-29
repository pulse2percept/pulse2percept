"""Models of cortical stimulation.

Base Class
----------

.. autosummary::
    :toctree:

    CortexSpatial

Models
------

.. autosummary::
    :toctree:

    ScoreboardModel
    ScoreboardSpatial
    DynaphosModel

.. seealso::

    *  :ref:`Core Concepts > Models and Percepts <topics-models>`

"""
from .base import CortexSpatial
from .scoreboard import ScoreboardModel, ScoreboardSpatial
from .dynaphos import DynaphosModel

__all__ = [
    'CortexSpatial',
    'DynaphosModel',
    'ScoreboardModel',
    'ScoreboardSpatial'
]
