"""This module provides a number of visual prostheses.

.. autosummary::
    :toctree: _api

    base
    argus
    alpha

"""
from .base import (DiskElectrode,
                   Electrode,
                   ElectrodeArray,
                   ElectrodeGrid,
                   PointSource,
                   ProsthesisSystem)
from .argus import ArgusI, ArgusII
from .alpha import AlphaIMS, AlphaAMS

__all__ = [
    'AlphaAMS',
    'AlphaIMS',
    'ArgusI',
    'ArgusII',
    'DiskElectrode',
    'Electrode',
    'ElectrodeArray',
    'ElectrodeGrid',
    'PointSource',
    'ProsthesisSystem'
]
