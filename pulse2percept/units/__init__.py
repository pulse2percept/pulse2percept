"""Physical units.

A deliberately small unit system, inspired by Brian2, that lets p2p's public
API accept unitful values::

    from pulse2percept.units import uA, mA, ms, um, mm

    pulse = BiphasicPulse(50 * uA, 0.45 * ms)
    implant = Orion(x=15 * mm)

Three rules describe the whole feature:

1. Bare numbers keep working, and keep their documented meaning. p2p never
   warns about them.
2. Unitful values are dimension-checked and rescaled to the unit the code
   expects, so ``50 * uA`` and ``0.05 * mA`` are the same input (up to
   floating-point precision).
3. Units are stripped at the API boundary, before any numerical work. Cython
   kernels and NumPy arrays never see a
   :py:class:`~pulse2percept.units.Quantity`.

.. autosummary::
    :toctree: _api

    base

"""
from .base import (Dimension, Unit, Quantity, DimensionMismatchError, as_value,
                   has_units, dimensionless,
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
                   dva)

__all__ = [
    'as_value',
    'Dimension',
    'DimensionMismatchError',
    'dimensionless',
    'has_units',
    'Quantity',
    'Unit',
    's', 'ms', 'us', 'ns',
    'Hz', 'kHz',
    'm', 'cm', 'mm', 'um', 'nm',
    'A', 'mA', 'uA', 'nA',
    'V', 'mV', 'uV',
    'C', 'mC', 'uC', 'nC',
    'dva',
]
