"""Figures and animations that combine multiple pulse2percept objects.

Individual objects retain their own ``plot()`` and ``play()`` methods; this
package only holds views that no single object owns.

.. autosummary::
    :toctree: _api

    comparison
    argus

"""

from .argus import plot_argus_phosphenes, plot_argus_simulated_phosphenes
from .comparison import play_stimulus_percept, plot_stimulus_percept

__all__ = [
    'play_stimulus_percept',
    'plot_argus_phosphenes',
    'plot_argus_simulated_phosphenes',
    'plot_stimulus_percept'
]
