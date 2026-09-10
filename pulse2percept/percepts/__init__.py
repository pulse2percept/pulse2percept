"""Visual percepts and phosphenes.

The ``metrics`` module is documented but deliberately not imported here:
:py:meth:`~pulse2percept.percepts.Percept.measure` loads it on demand, so
predicting a percept never pays for the measurement machinery. Import its
classes from ``pulse2percept.percepts.metrics`` if you need them directly.

.. autosummary::
    :toctree: _api

    base
    metrics

"""
from .base import Percept

__all__ = [
    'Percept'
]
