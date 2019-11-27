"""This module provides a number of stimuli.

.. autosummary::
    :toctree: _api

    base
    pulse_trains
"""

from .base import Stimulus
from .pulse_trains import (TimeSeries, MonophasicPulse, BiphasicPulse,
                           PulseTrain)

__all__ = [
    'BiphasicPulse',
    'MonophasicPulse',
    'PulseTrain',
    'Stimulus',
    'TimeSeries'
]
