"""Deprecated. Use :py:mod:`pulse2percept.plotting`.

The cross-object figures moved to :py:mod:`pulse2percept.plotting`. The
generic statistical helpers in :py:mod:`~pulse2percept.viz.base` have no
replacement and are removed with this module.

.. deprecated:: 0.11.0

    Will be removed in version 0.12.0.

"""

from ..plotting import (play_stimulus_percept, plot_argus_phosphenes,
                        plot_argus_simulated_phosphenes,
                        plot_stimulus_percept)
from ..utils import deprecated
from .base import correlation_matrix, scatter_correlation


def _moved(func):
    """Returns ``func`` wrapped to emit a DeprecationWarning when called"""
    return deprecated(alt_func=f'pulse2percept.plotting.{func.__name__}',
                      deprecated_version='0.11.0',
                      removed_version='0.12.0')(func)


play_stimulus_percept = _moved(play_stimulus_percept)
plot_argus_phosphenes = _moved(plot_argus_phosphenes)
plot_argus_simulated_phosphenes = _moved(plot_argus_simulated_phosphenes)
plot_stimulus_percept = _moved(plot_stimulus_percept)

# Imported last, since it re-exports the wrappers above; also keeps
# ``pulse2percept.viz.argus`` available as an attribute.
from . import argus  # noqa: E402,F401

__all__ = [
    'correlation_matrix',
    'play_stimulus_percept',
    'plot_argus_phosphenes',
    'plot_argus_simulated_phosphenes',
    'plot_stimulus_percept',
    'scatter_correlation'
]
