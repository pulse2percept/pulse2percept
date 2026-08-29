"""Common electrical stimuli, such as charge-balanced square-wave pulse trains.

.. autosummary::
    :toctree: _api

    base
    names
    pulses
    pulse_trains
    images
    videos
    encoders
    psychophysics

.. seealso::

    *  :ref:`Basic Concepts > Electrical Stimuli <topics-stimuli>`

"""

from .base import Stimulus
from .names import ElectrodeNames
from .pulses import AsymmetricBiphasicPulse, BiphasicPulse, MonophasicPulse
from .pulse_trains import (PulseTrain, BiphasicPulseTrain,
                           BiphasicTripletTrain, AsymmetricBiphasicPulseTrain)
from .images import ImageStimulus, LogoBVL, LogoUCSB, SnellenChart
from .videos import VideoStimulus, BostonTrain, GirlPool
from .encoders import (Encoder, StimulusEncoder, AmplitudeEncoder,
                       FrequencyEncoder, PRIMAEncoder)
from .psychophysics import BarStimulus, GratingStimulus

__all__ = [
    'AmplitudeEncoder',
    'AsymmetricBiphasicPulse',
    'AsymmetricBiphasicPulseTrain',
    'BarStimulus',
    'BiphasicPulse',
    'BiphasicPulseTrain',
    'BiphasicTripletTrain',
    'BostonTrain',
    'ElectrodeNames',
    'Encoder',
    'FrequencyEncoder',
    'GirlPool',
    'GratingStimulus',
    'ImageStimulus',
    'LogoBVL',
    'LogoUCSB',
    'MonophasicPulse',
    'PRIMAEncoder',
    'PulseTrain',
    'SnellenChart',
    'Stimulus',
    'StimulusEncoder',
    'VideoStimulus'
]
