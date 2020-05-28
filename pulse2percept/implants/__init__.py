"""Different retinal prosthetic implants, such as Argus II, Alpha-IMS,
   BVA-24, and PRIMA.

.. autosummary::
    : toctree: _api

    base
    argus
    alpha
    bva
    prima

.. seealso::

    *  :ref:`Basic Concepts > Visual Prostheses <topics-implants>`
"""
from .base import (DiskElectrode,
                   Electrode,
                   ElectrodeArray,
                   ElectrodeGrid,
                   PointSource,
                   ProsthesisSystem,
                   SquareElectrode)
from .argus import ArgusI, ArgusII
from .alpha import AlphaIMS, AlphaAMS
from .bva import BVA24
from .prima import PRIMA

__all__ = [
    'AlphaAMS',
    'AlphaIMS',
    'ArgusI',
    'ArgusII',
    'BVA24',
    'DiskElectrode',
    'Electrode',
    'ElectrodeArray',
    'ElectrodeGrid',
    'PointSource',
    'PRIMA',
    'ProsthesisSystem',
    'SquareElectrode'
]
