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
    'Raster',
    'retina',
    'SequentialRaster',
    'SquareElectrode',
]
