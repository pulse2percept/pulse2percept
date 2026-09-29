"""Physical units.

Bare numbers use their documented units. Unitful values are checked for
dimensional compatibility and converted at API boundaries.

Core
----

.. autosummary::
    :toctree:

    Unit
    Quantity
    Dimension

Helpers
-------

.. autosummary::
    :toctree:

    as_value
    base.has_units
    DimensionMismatchError

Units
-----

===============  ======================================
time             ``s``, ``ms``, ``us``, ``ns``
frequency        ``Hz``, ``kHz``
distance         ``m``, ``cm``, ``mm``, ``um``, ``nm``
current          ``A``, ``mA``, ``uA``, ``nA``
voltage          ``V``, ``mV``, ``uV``
power            ``W``, ``mW``, ``uW``
charge           ``C``, ``mC``, ``uC``, ``nC``
angle            ``rad``, ``deg``
visual angle     ``dva``
threshold ratio  ``xTh``
none             ``dimensionless``
===============  ======================================

.. seealso::

    *  :ref:`Core Concepts > Physical Units <topics-units>`

"""
from .base import (Dimension, Unit, Quantity, DimensionMismatchError, as_value,
                   dimensionless,
                   # time
                   s, ms, us, ns,
                   # frequency
                   Hz, kHz,
                   # distance
                   m, cm, mm, um, nm,
                   # current
                   A, mA, uA, nA,
                   # voltage
                   V, mV, uV,
                   # power
                   W, mW, uW,
                   # charge
                   C, mC, uC, nC,
                   # angle
                   rad, deg,
                   # visual angle
                   dva,
                   # threshold ratio
                   xTh)

__all__ = [
    'as_value',
    'Dimension',
    'DimensionMismatchError',
    'dimensionless',
    'Quantity',
    'Unit',
    's', 'ms', 'us', 'ns',
    'Hz', 'kHz',
    'm', 'cm', 'mm', 'um', 'nm',
    'A', 'mA', 'uA', 'nA',
    'V', 'mV', 'uV',
    'W', 'mW', 'uW',
    'C', 'mC', 'uC', 'nC',
    'rad', 'deg',
    'dva',
    'xTh',
]
