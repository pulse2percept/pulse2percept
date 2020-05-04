"""Common electrical stimuli, such as charge-balanced square-wave pulse trains.

.. autosummary::
    :toctree: _api

    base
    pulses
    pulse_trains

.. seealso::

    *  :ref:`Basic Concepts > Electrical Stimuli <topics-stimuli>`

"""

# Pulses with net currents smaller than 10 picoamps are considered
# charge-balanced (here expressed in microamps):
MIN_AMP = 1e-5

from .base import Stimulus, TimeSeries
from .pulses import AsymmetricBiphasicPulse, BiphasicPulse, MonophasicPulse
from .pulse_trains import (BiphasicPulseTrain, AsymmetricBiphasicPulseTrain)

__all__ = [
    'AsymmetricBiphasicPulse',
    'AsymmetricBiphasicPulseTrain',
    'BiphasicPulse',
    'BiphasicPulseTrain',
    'MonophasicPulse',
    'Stimulus',
    'TimeSeries'
]
