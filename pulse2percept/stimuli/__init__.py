"""Visual and electrical stimuli.

Core stimulus types are available at the top level. Generated psychophysics
stimuli and bundled sample assets live in the ``psychophysics`` and ``samples``
namespaces, respectively.

Core
----

.. autosummary::
    :toctree: _api

    Stimulus
    ImageStimulus
    VideoStimulus

Electrical stimuli
------------------

.. autosummary::
    :toctree: _api

    MonophasicPulse
    BiphasicPulse
    AsymmetricBiphasicPulse
    PulseTrain
    BiphasicPulseTrain
    BiphasicTripletTrain
    AsymmetricBiphasicPulseTrain

Encoders
--------

Convert visual stimuli to electrical stimulation.

.. autosummary::
    :toctree: _api

    Encoder
    StimulusEncoder
    AmplitudeEncoder
    FrequencyEncoder
    PRIMAEncoder

Psychophysics
-------------

Generated visual stimuli in degrees of visual angle and physical time.

.. autosummary::
    :toctree: _api

    psychophysics.bar
    psychophysics.grating
    psychophysics.landolt_c
    psychophysics.tumbling_e

Samples
-------

Bundled images and videos for examples, documentation, and tests.

.. autosummary::
    :toctree: _api

    samples.big_buck_bunny
    samples.bvl_cake
    samples.cajal_retina
    samples.logo_bvl
    samples.logo_ucsb
    samples.ucsb_bike
    samples.ucsb_flyover
    samples.ucsb_pedestrians
    samples.ucsb_surf
    samples.zebrafish_retina

Deprecated in v0.11
-------------------

These compatibility classes will be removed in v0.12.

.. autosummary::
    :toctree: _api

    BarStimulus
    GratingStimulus
    LogoBVL
    LogoUCSB

.. seealso::

    *  :ref:`Basic Concepts > Electrical Stimuli <topics-stimuli>`

"""

from .base import ImageStimulus, Stimulus, VideoStimulus
from .pulses import AsymmetricBiphasicPulse, BiphasicPulse, MonophasicPulse
from .pulse_trains import (PulseTrain, BiphasicPulseTrain,
                           BiphasicTripletTrain, AsymmetricBiphasicPulseTrain)
from .encoders import (Encoder, StimulusEncoder, AmplitudeEncoder,
                       FrequencyEncoder, PRIMAEncoder)
from .psychophysics import BarStimulus, GratingStimulus
from .samples import LogoBVL, LogoUCSB
from . import psychophysics, samples

__all__ = [
    'AmplitudeEncoder',
    'AsymmetricBiphasicPulse',
    'AsymmetricBiphasicPulseTrain',
    'BarStimulus',
    'BiphasicPulse',
    'BiphasicPulseTrain',
    'BiphasicTripletTrain',
    'Encoder',
    'FrequencyEncoder',
    'GratingStimulus',
    'ImageStimulus',
    'LogoBVL',
    'LogoUCSB',
    'MonophasicPulse',
    'PRIMAEncoder',
    'psychophysics',
    'PulseTrain',
    'samples',
    'Stimulus',
    'StimulusEncoder',
    'VideoStimulus'
]
