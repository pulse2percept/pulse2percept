"""Visual field maps and simulation grids.

Anatomy-neutral classes are available at the top level. Maps are
grouped by target tissue in ``retina`` and ``cortex``.

.. versionchanged:: 0.11.0

    Anatomy-specific maps are no longer exported here.

Grids
-----

.. autosummary::
    :toctree:

    Grid2D
    base.CoordinateGrid

Visual Field Maps
-----------------

.. autosummary::
    :toctree:

    VisualFieldMap
    retina
    cortex

.. seealso::

    *  :ref:`Core Concepts > Retinotopy and Coordinates <topics-coordinates>`

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
