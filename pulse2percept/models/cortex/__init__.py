"""Computational models of the cortex, such as phosphene and neural response models.

.. autosummary::
    :toctree: _api

    base

.. seealso::

    *  :ref:`Basic Concepts > Computational Models <topics-models>`

"""
from .base import ScoreboardModel, ScoreboardSpatial, CortexSpatial
from .dynaphos import DynaphosModel

__all__ = [
    'CortexSpatial',
    'DynaphosModel',
    'ScoreboardModel',
    'ScoreboardSpatial'
]
