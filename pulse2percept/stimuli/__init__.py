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
from .encoders import StimulusEncoder, AmplitudeEncoder, FrequencyEncoder
from .psychophysics import BarStimulus, GratingStimulus
from ..utils import deprecated_names

#: ``StimulusEncoder`` was called ``Encoder`` in 0.10.0. The old name is
#: deliberately *not* imported above: a module-level ``__getattr__`` is only
#: consulted for names the module does not already have, and that hook is what
#: lets ``from pulse2percept.stimuli import Encoder`` warn while still handing
#: back ``StimulusEncoder`` itself.
__getattr__ = deprecated_names(globals(), {'Encoder': 'StimulusEncoder'},
                               deprecated_version='0.10.0',
                               removed_version='0.11.0')

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
    # Deprecated alias for `StimulusEncoder`, kept until 0.11.0 so that
    # ``from pulse2percept.stimuli import *`` does not break:
    'Encoder',
    'FrequencyEncoder',
    'GirlPool',
    'GratingStimulus',
    'ImageStimulus',
    'LogoBVL',
    'LogoUCSB',
    'MonophasicPulse',
    'PulseTrain',
    'SnellenChart',
    'Stimulus',
    'StimulusEncoder',
    'VideoStimulus'
]
