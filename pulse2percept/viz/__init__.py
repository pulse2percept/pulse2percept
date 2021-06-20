"""Various visualization functions.

.. autosummary::
    :toctree: _api

    base
    axon_map

"""

from .axon_map import plot_axon_map, plot_implant_on_axon_map
from .argus import plot_argus_phosphenes, plot_argus_simulated_phosphenes
from .base import scatter_correlation, correlation_matrix


__all__ = [
    'correlation_matrix',
    'plot_argus_phosphenes',
    'plot_argus_simulated_phosphenes',
    'plot_axon_map',
    'plot_implant_on_axon_map',
    'scatter_correlation'
]
