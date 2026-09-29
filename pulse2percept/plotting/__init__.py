"""Figures and animations that combine multiple pulse2percept objects.

Single objects have their own ``plot()`` and ``play()`` methods.

Stimulus and Percept
--------------------

.. autosummary::
    :toctree:

    plot_stimulus_percept
    play_stimulus_percept

Implant and Percept
-------------------

.. autosummary::
    :toctree:

    plot_implant_percept
    play_implant_percept

Argus Phosphene Drawings
------------------------

.. autosummary::
    :toctree:

    plot_argus_phosphenes
    plot_argus_simulated_phosphenes

"""

from .argus import plot_argus_phosphenes, plot_argus_simulated_phosphenes
from .comparison import (play_implant_percept, play_stimulus_percept,
                         plot_implant_percept, plot_stimulus_percept)

__all__ = [
    'play_implant_percept',
    'play_stimulus_percept',
    'plot_argus_phosphenes',
    'plot_argus_simulated_phosphenes',
    'plot_implant_percept',
    'plot_stimulus_percept'
]
