"""Deprecated. Use :py:mod:`pulse2percept.plotting`.

The cross-object figures moved to :py:mod:`pulse2percept.plotting`; the generic
statistical helpers in :py:mod:`~pulse2percept.viz.base` have no successor and
go away with this module.

.. deprecated:: 0.11.0

    Will be removed in version 0.12.0.

.. autosummary::
    :toctree: _api

    base
    argus

"""

from ..plotting import (play_stimulus_percept, plot_argus_phosphenes,
                        plot_argus_simulated_phosphenes,
                        plot_stimulus_percept)
from ..utils import deprecated
from .base import correlation_matrix, scatter_correlation


def _moved(func):
    """Re-export a function from ``plotting``, warning when it is called"""
    return deprecated(alt_func=f'pulse2percept.plotting.{func.__name__}',
                      deprecated_version='0.11.0',
                      removed_version='0.12.0')(func)


play_stimulus_percept = _moved(play_stimulus_percept)
plot_argus_phosphenes = _moved(plot_argus_phosphenes)
plot_argus_simulated_phosphenes = _moved(plot_argus_simulated_phosphenes)
plot_stimulus_percept = _moved(plot_stimulus_percept)

# Imported last: the shim re-exports the wrappers defined just above, and
# ``pulse2percept.viz.argus`` used to be an attribute of this package.
from . import argus  # noqa: E402,F401

__all__ = [
    'correlation_matrix',
    'play_stimulus_percept',
    'plot_argus_phosphenes',
    'plot_argus_simulated_phosphenes',
    'plot_stimulus_percept',
    'scatter_correlation'
]
