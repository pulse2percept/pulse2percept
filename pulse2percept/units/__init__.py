"""Physical units used by pulse2percept.

Bare numbers retain their documented units. Unitful values are checked
for dimensional compatibility and converted at API boundaries.

.. autosummary::
    :toctree: _api

    base
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
                   # charge
                   C, mC, uC, nC,
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
    'C', 'mC', 'uC', 'nC',
    'dva',
    'xTh',
]
