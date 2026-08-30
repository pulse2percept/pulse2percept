"""Various visualization functions.

.. autosummary::
    :toctree: _api

    base
    argus
    comparison

"""

from .argus import plot_argus_phosphenes, plot_argus_simulated_phosphenes
from .base import scatter_correlation, correlation_matrix
from .comparison import play_stimulus_percept, plot_stimulus_percept

__all__ = [
    'correlation_matrix',
    'play_stimulus_percept',
    'plot_argus_phosphenes',
    'plot_argus_simulated_phosphenes',
    'plot_stimulus_percept',
    'scatter_correlation'
]
