"""Computational models of prosthetic vision.

Anatomy-neutral base classes and temporal models are available at the
top level. Models of a specific target tissue are in ``retina`` and
``cortex``.

Base Classes
------------

.. autosummary::
    :toctree:

    BaseModel
    Model
    SpatialModel
    TemporalModel

Temporal Models
---------------

Combine with any retinal or cortical spatial model.

.. autosummary::
    :toctree:

    AlphaTemporal
    FadingTemporal

Target-Specific Models
----------------------

.. autosummary::
    :toctree:

    retina
    cortex

.. seealso::

    *  :ref:`Core Concepts > Models and Percepts <topics-models>`

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
