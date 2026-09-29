"""Visual and electrical stimuli.

Core stimulus types are available at the top level. Generated psychophysics
stimuli and bundled sample assets live in the ``psychophysics`` and ``samples``
namespaces, respectively.

Core
----

.. autosummary::
    :toctree:

    Stimulus
    ImageStimulus
    VideoStimulus

Electrical Stimuli
------------------

.. autosummary::
    :toctree:

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
    :toctree:

    Encoder
    ImplantEncoder
    PulseEncoder
    AmplitudeEncoder
    FrequencyEncoder
    TraceEncoder
    PhotovoltaicEncoder
    PRIMAEncoder

Psychophysics
-------------

Generated visual stimuli in degrees of visual angle and physical time.

.. autosummary::
    :toctree:

    psychophysics.bar
    psychophysics.grating
    psychophysics.landolt_c
    psychophysics.tumbling_e

Samples
-------

Bundled images and videos for examples, documentation, and tests.

.. autosummary::
    :toctree:

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
    :toctree:

    BarStimulus
    GratingStimulus
    LogoBVL
    LogoUCSB

.. seealso::

    *  :ref:`Core Concepts > Stimulation <topics-stimulation>`

"""

from .base import ImageStimulus, Stimulus, VideoStimulus
from .pulses import AsymmetricBiphasicPulse, BiphasicPulse, MonophasicPulse
from .pulse_trains import (PulseTrain, BiphasicPulseTrain,
                           BiphasicTripletTrain, AsymmetricBiphasicPulseTrain)
from .encoders import (Encoder, ImplantEncoder, PulseEncoder,
                       AmplitudeEncoder, FrequencyEncoder, PhotovoltaicEncoder,
                       PRIMAEncoder, TraceEncoder)
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
    'ImplantEncoder',
    'LogoBVL',
    'LogoUCSB',
    'MonophasicPulse',
    'PhotovoltaicEncoder',
    'PRIMAEncoder',
    'psychophysics',
    'PulseEncoder',
    'PulseTrain',
    'samples',
    'Stimulus',
    'TraceEncoder',
    'VideoStimulus'
]
