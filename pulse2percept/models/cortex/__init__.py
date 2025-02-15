"""Phosphene models for cortical implants

.. autosummary::
    :toctree: _api

    base
    dynaphos

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
