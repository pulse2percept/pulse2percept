"""Different retinal prosthetic implants, such as Argus II and Alpha-IMS.

.. autosummary::
    : toctree: _api

    base
    argus
    alpha

"""
from .base import (DiskElectrode,
                   Electrode,
                   ElectrodeArray,
                   ElectrodeGrid,
                   ElectrodeGridHex,
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
    'ElectrodeGridHex',
    'PointSource',
    'ProsthesisSystem'
]
