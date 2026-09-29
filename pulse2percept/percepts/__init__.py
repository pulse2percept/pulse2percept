"""Percepts predicted by a model, and phosphene measurements.

Core
----

.. autosummary::
    :toctree:

    Percept

Metrics
-------

Returned by :py:meth:`Percept.measure`. ``metrics`` is not
imported with ``percepts``; import it from
``pulse2percept.percepts.metrics``.

.. autosummary::
    :toctree:

    metrics.measure_percept
    metrics.PerceptMetrics
    metrics.FrameMetrics

.. seealso::

    *  :ref:`Core Concepts > Models and Percepts <topics-models>`

"""
from .base import Percept

__all__ = [
    'Percept'
]
