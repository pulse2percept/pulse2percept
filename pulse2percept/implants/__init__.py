"""Visual prostheses, e.g. Argus II, Alpha-IMS, BVT-24, PRIMA, Cortivis

Generic device machinery lives at the root; devices live under the anatomical
target they stimulate.

.. autosummary::
    :toctree: _api

    retina
    cortex

.. autosummary::
    :toctree: _api

    base
    electrodes
    electrode_arrays
    rasters
    ensemble

.. seealso::

    *  :ref:`Basic Concepts > Visual Prostheses <topics-implants>`
"""
from .base import GridImplant, Implant
from .electrodes import (Electrode, PointSource, DiskElectrode,
                         SquareElectrode, HexElectrode)
from .electrode_arrays import ElectrodeArray, ElectrodeGrid
from .rasters import (Raster, SequentialRaster, CheckerboardRaster,
                      CustomRaster)
from .ensemble import EnsembleImplant
from . import cortex
from . import retina
from ..utils.deprecation import _deprecated_names

__all__ = [
    'CheckerboardRaster',
    'cortex',
    'CustomRaster',
    'DiskElectrode',
    'Electrode',
    'ElectrodeArray',
    'ElectrodeGrid',
    'EnsembleImplant',
    'GridImplant',
    'HexElectrode',
    'Implant',
    'PointSource',
    # Deprecated in 0.11.0, removed in 0.12.0:
    'ProsthesisSystem',
    'Raster',
    'retina',
    'SequentialRaster',
    'SquareElectrode',
]

# Deprecated in 0.11.0, removed in 0.12.0. Defined here as well as in
# ``base`` so that both import paths warn.
__getattr__ = _deprecated_names(__name__, {'ProsthesisSystem': Implant},
                                deprecated_version='0.11.0',
                                removed_version='0.12.0')
