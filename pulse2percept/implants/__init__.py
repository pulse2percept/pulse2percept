"""Visual prostheses, electrode arrays, and electrodes.

Anatomy-neutral classes are available at the top level. Devices are
grouped by target tissue in ``retina`` and ``cortex``.

Implants
--------

.. autosummary::
    :toctree:

    Implant
    GridImplant
    EnsembleImplant

Electrode Arrays
----------------

.. autosummary::
    :toctree:

    ElectrodeArray
    ElectrodeGrid

Electrodes
----------

.. autosummary::
    :toctree:

    Electrode
    PointSource
    DiskElectrode
    SquareElectrode
    HexElectrode

Rasters
-------

Split electrodes into groups that are stimulated at different
times.

.. autosummary::
    :toctree:

    Raster
    SequentialRaster
    CheckerboardRaster
    CustomRaster

Devices
-------

.. autosummary::
    :toctree:

    retina
    cortex

Deprecated in v0.11
-------------------

``ProsthesisSystem`` is an alias of :py:class:`Implant` and will be
removed in v0.12.

.. seealso::

    *  :ref:`Core Concepts > Implants <topics-implants>`
    *  :ref:`Core Concepts > Stimulation > Raster Scheduling <topics-rasters>`

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
