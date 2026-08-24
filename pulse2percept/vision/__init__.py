"""Residual vision: what a person still sees, and where it is lost.

.. autosummary::
    :toctree: _api

    composition
    scotoma

.. versionadded:: 0.11.0
"""
from .composition import compose_amd
from .scotoma import Scotoma

__all__ = [
    'compose_amd',
    'Scotoma'
]
