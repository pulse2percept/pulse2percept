"""
Different prosthetic implants, such as Argus II, Alpha-IMS, BVT-24, PRIMA, Cortivis, etc.

.. autosummary::
    :toctree: _api

    cortex

.. autosummary::
    :toctree: _api

    base
    electrodes
    electrode_arrays
    rasters
    argus
    alpha
    bvt
    imie
    prima
    ensemble

.. seealso::

    *  :ref:`Basic Concepts > Visual Prostheses <topics-implants>`
"""
from .base import GridImplant, Implant, RectangleImplant
from .electrodes import (Electrode, PointSource, DiskElectrode,
                         SquareElectrode, HexElectrode)
from .electrode_arrays import ElectrodeArray, ElectrodeGrid
from .rasters import (Raster, SequentialRaster, CheckerboardRaster,
                      CustomRaster)
from .argus import ArgusI, ArgusII
from .alpha import AlphaIMS, AlphaAMS
from .bvt import BVT24, BVT44
from .prima import (PhotovoltaicPixel, PRIMAPivotal, Lorach2015Array,
                    Ho2019FlatArray, Huang2021Array, PRIMA, PRIMA75,
                    PRIMA55, PRIMA40)
from .imie import IMIE
from .ensemble import EnsembleImplant
from . import cortex
from ..utils.deprecation import _deprecated_names

__all__ = [
    'AlphaAMS',
    'AlphaIMS',
    'ArgusI',
    'ArgusII',
    'BVT24',
    'BVT44',
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
    'Ho2019FlatArray',
    'Huang2021Array',
    'Implant',
    'Lorach2015Array',
    'PhotovoltaicPixel',
    'PointSource',
    'PRIMAPivotal',
    # Deprecated in 0.11.0, removed in 0.12.0:
    'PRIMA',
    'PRIMA75',
    'PRIMA55',
    'PRIMA40',
    'ProsthesisSystem',
    'Raster',
    'RectangleImplant',
    'SequentialRaster',
    'SquareElectrode',
    'IMIE'
]

# Deprecated in 0.11.0, removed in 0.12.0. Defined here as well as in
# ``base`` so that both import paths warn.
__getattr__ = _deprecated_names(__name__, {'ProsthesisSystem': Implant},
                                deprecated_version='0.11.0',
                                removed_version='0.12.0')
