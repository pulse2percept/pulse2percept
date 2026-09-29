"""Retinal implants.

Base class
----------

.. autosummary::
    :toctree:

    RetinalImplant

Human implant systems
---------------------

.. autosummary::
    :toctree:

    ArgusI
    ArgusII
    IMIE
    AlphaIMS
    AlphaAMS
    PRIMAPivotal
    Suprachoroidal24
    Suprachoroidal44

Research array geometries
-------------------------

.. autosummary::
    :toctree:

    Lorach2015Array
    Ho2019FlatArray
    Huang2021Array
    PhotovoltaicPixel

Deprecated in v0.11
-------------------

These compatibility classes will be removed in v0.12.

.. autosummary::
    :toctree:

    PRIMA
    PRIMA75
    PRIMA55
    PRIMA40

.. seealso::

    *  :ref:`Core Concepts > Implants > Human Implant Systems
       <topics-implants-human>`
    *  :ref:`Core Concepts > Implants > Research Array Geometries
       <topics-implants-research>`

"""
from .base import RetinalImplant
from .argus import ArgusI, ArgusII
from .alpha import AlphaIMS, AlphaAMS
from .suprachoroidal import Suprachoroidal24, Suprachoroidal44
from .imie import IMIE
from .prima import (PhotovoltaicPixel, PRIMAPivotal, Lorach2015Array,
                    Ho2019FlatArray, Huang2021Array, PRIMA, PRIMA75,
                    PRIMA55, PRIMA40)

__all__ = [
    'AlphaAMS',
    'AlphaIMS',
    'ArgusI',
    'ArgusII',
    'Suprachoroidal24',
    'Suprachoroidal44',
    'Ho2019FlatArray',
    'Huang2021Array',
    'IMIE',
    'Lorach2015Array',
    'PhotovoltaicPixel',
    'PRIMAPivotal',
    'RetinalImplant',
    # Deprecated in 0.11.0, removed in 0.12.0:
    'PRIMA',
    'PRIMA75',
    'PRIMA55',
    'PRIMA40',
]
