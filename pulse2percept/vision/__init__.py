"""Visual scenes, residual vision, and binocular views.

.. autosummary::
    :toctree: _api

    binocular
    gaze
    scene
    scotoma

.. versionadded:: 0.11.0
"""
from .binocular import BinocularScene
from .gaze import Gaze
from .scene import Scene
from .scotoma import Scotoma

__all__ = [
    'BinocularScene',
    'Gaze',
    'Scene',
    'Scotoma'
]
