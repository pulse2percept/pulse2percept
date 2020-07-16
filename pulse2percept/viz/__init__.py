"""Various visualization functions.

.. autosummary::
    :toctree: _api

	base
    axon_map

"""

from .base import scatter_correlation, correlation_matrix
from .argus import plot_argus_phosphenes
from .axon_map import plot_axon_map, plot_implant_on_axon_map

__all__ = [
    'correlation_matrix',
    'plot_argus_phosphenes',
    'plot_axon_map',
    'plot_implant_on_axon_map'
    'scatter_correlation'
]
