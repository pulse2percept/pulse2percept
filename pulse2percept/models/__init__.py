"""Computational models of prosthetic vision

The root namespace holds only anatomy-neutral model machinery: the abstract
base classes a model is assembled from, and temporal models that are not tied
to a particular stimulation site. Models of a specific target live in the
subpackage for the tissue they stimulate ---
:py:mod:`~pulse2percept.models.retina` and
:py:mod:`~pulse2percept.models.cortex`.

.. versionchanged:: 0.11.0

    Retinal models are no longer exported here; import them from
    ``models.retina``.

.. autosummary::
    :toctree: _api

    base
    temporal
    retina
    cortex

.. seealso::

    *  :ref:`Basic Concepts > Computational Models <topics-models>`

"""
from .base import BaseModel, Model, SpatialModel, TemporalModel
from .temporal import AlphaTemporal, FadingTemporal

from . import cortex
from . import retina

__all__ = [
    'AlphaTemporal',
    'BaseModel',
    'FadingTemporal',
    'Model',
    'SpatialModel',
    'TemporalModel',
    'cortex',
    'retina',
]
