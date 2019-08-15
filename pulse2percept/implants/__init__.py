"""Implants

This module provides a number of visual prostheses.
"""
from .base import (DiskElectrode,
                   Electrode,
                   ElectrodeArray,
                   PointSource,
                   ProsthesisSystem)
from .argus import ArgusI, ArgusII
from .alpha import AlphaIMS

__all__ = [
    'AlphaIMS',
    'ArgusI',
    'ArgusII',
    'DiskElectrode',
    'Electrode',
    'ElectrodeArray',
    'PointSource',
    'ProsthesisSystem'
]
