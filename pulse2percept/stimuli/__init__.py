"""Visual and electrical stimuli, and the containers that hold them.

Top-level names are reusable types; namespaced snake-case functions construct
particular content.

Core
----

What a stimulus is: the data container, its pixel-data specializations, and
the electrode naming they share.

.. autosummary::
    :toctree: _api

    Stimulus
    ImageStimulus
    VideoStimulus
    ElectrodeNames

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

Transformations of a visual stimulus into electrical stimulation.

.. autosummary::
    :toctree: _api

    Encoder
    StimulusEncoder
    AmplitudeEncoder
    FrequencyEncoder
    PRIMAEncoder

Psychophysics
-------------

Visual stimuli generated from their parameters, in degrees of visual angle
and physical time. Reached through
:py:mod:`pulse2percept.stimuli.psychophysics`, not the top-level namespace.

.. autosummary::
    :toctree: _api

    psychophysics.bar
    psychophysics.grating
    psychophysics.landolt_c
    psychophysics.tumbling_e

Samples
-------

Images and videos bundled with pulse2percept, for demos, docs, and tests.
Reached through :py:mod:`pulse2percept.stimuli.samples`.

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

Constructor subclasses replaced by the functions above. They are unchanged in
v0.11 and will be removed in v0.12.

.. autosummary::
    :toctree: _api

    BarStimulus
    GratingStimulus
    LogoBVL
    LogoUCSB

.. seealso::

    *  :ref:`Basic Concepts > Electrical Stimuli <topics-stimuli>`

"""

from .base import ElectrodeNames, ImageStimulus, Stimulus, VideoStimulus
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
    'ElectrodeNames',
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
