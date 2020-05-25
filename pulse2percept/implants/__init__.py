"""Different retinal prosthetic implants, such as Argus II, Alpha-IMS,
   BVA-24, and Prima.

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
                   ProsthesisSystem)
from .argus import ArgusI, ArgusII
from .alpha import AlphaIMS, AlphaAMS
from .bva import BVA24
from .prima import Prima

__all__ = [
    'AlphaAMS',
    'AlphaIMS',
    'ArgusI',
    'ArgusII',
    'BVA24',
    'Prima',
    'DiskElectrode',
    'Electrode',
    'ElectrodeArray',
    'ElectrodeGrid',
    'PointSource',
    'ProsthesisSystem'
]
