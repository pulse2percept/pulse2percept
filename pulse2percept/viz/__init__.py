"""Various visualization functions.

.. autosummary::
    :toctree: _api
    :no-index:

    base
    argus

"""

from .argus import plot_argus_phosphenes, plot_argus_simulated_phosphenes
from .base import scatter_correlation, correlation_matrix

__all__ = [
    'correlation_matrix',
    'plot_argus_phosphenes',
    'plot_argus_simulated_phosphenes',
    'scatter_correlation'
]
