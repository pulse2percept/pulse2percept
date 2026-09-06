"""Visual field maps, retinotopy, and visuotopy

The root namespace holds only anatomy-independent machinery: the coordinate
grid a model simulates on, and the abstract map that turns visual field
coordinates into tissue coordinates. Concrete maps live in the subpackage for
the tissue they describe --- :py:mod:`~pulse2percept.topography.retina` and
:py:mod:`~pulse2percept.topography.cortex`.

.. versionchanged:: 0.11.0

    Anatomy-specific maps are no longer exported here; import them from
    ``topography.retina`` or ``topography.cortex``.

.. autosummary::
    :toctree: _api

    base
    retina
    cortex

"""
from . import cortex
from . import retina
from .base import Grid2D, VisualFieldMap

__all__ = [
    'Grid2D',
    'VisualFieldMap',
    'cortex',
    'retina',
]
