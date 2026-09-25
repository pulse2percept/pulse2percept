"""Retinal implants: Argus, Alpha, suprachoroidal, IMIE, and PRIMA arrays

.. autosummary::
    :toctree: _api

    base
    argus
    alpha
    suprachoroidal
    imie
    prima

.. seealso::

    *  :ref:`Basic Concepts > Visual Prostheses <topics-implants>`
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
