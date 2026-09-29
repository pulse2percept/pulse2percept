"""Visual scenes, residual vision, and binocular views.

.. versionadded:: 0.11.0

Scenes
------

.. autosummary::
    :toctree:

    Scene
    BinocularScene

Residual Vision
---------------

.. autosummary::
    :toctree:

    Scotoma

Eye Movements
-------------

.. autosummary::
    :toctree:

    Gaze

.. seealso::

    *  :ref:`Core Concepts > Scenes and Simulated Vision <topics-vision>`

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
