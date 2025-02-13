"""Visual field maps, retinotopy, and visuotopy

.. autosummary::
    :toctree: _api
    :no-index:

    base
    retina
    cortex
    neuropythy

"""
from .base import (Grid2D, VisualFieldMap,)
from .retina import (RetinalMap, Curcio1990Map,
                       Watson2014Map, Watson2014DisplaceMap,)
from .cortex import (CorticalMap, Polimeni2006Map)
from .neuropythy import NeuropythyMap 


__all__ = [
    'CorticalMap',
    'Curcio1990Map',
    'Grid2D',
    'RetinalMap'
    'VisualFieldMap',
    'Watson2014DisplaceMap',
    'Watson2014Map',
    'Polimeni2006Map',
    'NeuropythyMap'
]
