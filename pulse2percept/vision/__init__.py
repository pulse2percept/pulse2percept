"""Visual scenes, residual vision, and binocular views.

.. autosummary::
    :toctree: _api

    binocular
    scene
    scotoma

.. versionadded:: 0.11.0
"""
from .binocular import BinocularScene
from .scene import Scene
from .scotoma import Scotoma

__all__ = [
    'BinocularScene',
    'Scene',
    'Scotoma'
]
