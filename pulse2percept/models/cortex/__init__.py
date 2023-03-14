"""Computational models of the cortex, such as phosphene and neural response models.

.. autosummary::
    :toctree: _api

    base

.. seealso::

    *  :ref:`Basic Concepts > Computational Models <topics-models>`

"""
from .base import ScoreboardModel, ScoreboardSpatial


__all__ = [
    'ScoreboardModel',
    'ScoreboardSpatial'
]
