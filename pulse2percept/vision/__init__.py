"""Residual vision: what a person still sees, and where it is lost.

.. autosummary::
    :toctree: _api

    composition
    scene
    scotoma

.. versionadded:: 0.11.0
"""
from .composition import compose_hybrid_vision
from .scene import Scene
from .scotoma import Scotoma

__all__ = [
    'compose_hybrid_vision',
    'Scene',
    'Scotoma'
]
