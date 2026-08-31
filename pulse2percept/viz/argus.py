"""Deprecated. Use :py:mod:`pulse2percept.plotting.argus`.

Keeps ``from pulse2percept.viz.argus import ...`` working. The names are the
deprecated wrappers; the implementation lives in
:py:mod:`pulse2percept.plotting.argus`.
"""
from . import plot_argus_phosphenes, plot_argus_simulated_phosphenes

__all__ = ['plot_argus_phosphenes', 'plot_argus_simulated_phosphenes']
