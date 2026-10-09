"""Pipelines exercised by the benchmark suite.

A :class:`Scenario` is one realistic pulse2percept workflow: build a stimulus,
bind a model to an implant, predict a percept. The benchmarks in
``test_predict.py`` are parametrized over :data:`SCENARIOS`.

Inputs and grids are reduced from the quickstart versions so that the suite
stays quick; the workflows are otherwise the same.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

import pulse2percept as p2p
from pulse2percept.units import Hz, dva, mm, ms, uA


def drifting_grating(n_frames=12, fps=6, shape=(48, 80)):
    """Return a drifting sinusoidal grating video (grayscale, 2 s).

    6 fps matches the Argus II encoder's 6 Hz pulse rate, so every frame is
    sampled.
    """
    rows, cols = shape
    x = np.linspace(0, 4 * np.pi, cols)[np.newaxis, :, np.newaxis]
    phase = 2 * np.pi * np.arange(n_frames) / n_frames
    return p2p.stimuli.VideoStimulus(
        np.tile(0.5 + 0.5 * np.sin(x - phase), (rows, 1, 1)),
        metadata={'fps': fps})


def imie():
    """Return an IMIE that frequency-encodes images at 2x threshold.

    The 80 uA threshold is uniform and illustrative, as in the
    ``BiphasicAxonMapModel`` example; ``FrequencyEncoder`` takes current, and
    the model converts it to threshold multiples.
    """
    implant = p2p.implants.retina.IMIE()
    implant.thresholds = 80 * uA
    implant.encoder = p2p.stimuli.FrequencyEncoder(freq_range=(0, 60) * Hz,
                                                   amp=160 * uA)
    return implant


def axonmap_fading(implant, verbose, **axon_cache):
    """Return an AxonMap + Fading composite; cache keywords go to AxonMap."""
    return p2p.models.Model(
        spatial=p2p.models.retina.AxonMapSpatial(
            implant, xrange=(-12, 12), yrange=(-8, 8), step=0.5,
            verbose=verbose, **axon_cache),
        temporal=p2p.models.FadingTemporal(verbose=verbose))


def scotoma_scene():
    """Return a 40 dva video scene with a central scotoma (quickstart)."""
    return p2p.vision.Scene(drifting_grating(), fov=40 * dva,
                            scotoma=p2p.vision.Scotoma.circle(5 * dva),
                            scotoma_fill=0, aperture='round')


#: Two saccades within the 2 s scene, as in the quickstart.
GAZE = p2p.vision.Gaze([(0, 0, 0),
                        (-8 * dva, -3 * dva, 600 * ms),
                        (6 * dva, -4 * dva, 1300 * ms)])


def letter_z():
    """Return the quickstart letter Z as ``(N, 2)`` dva, 8 samples per
    stroke."""
    corners = [(-4.5, 3.4), (-1.3, 3.4), (-3.8, 0), (-1.2, 0)]
    return np.vstack([np.linspace(start, end, 8)
                      for start, end in zip(corners[:-1], corners[1:])])


def trace(model, trajectory):
    """Trace ``trajectory`` through the nearest electrodes, then predict."""
    encoder = p2p.stimuli.TraceEncoder(
        model, amp=1000 * uA, freq=model.freq, phase_dur=model.p_dur,
        step_dur=100 * ms)
    return model.predict_percept(encoder.encode(trajectory))


@dataclass(frozen=True)
class Scenario:
    """One stimulus/implant/model workflow.

    Attributes
    ----------
    id : str
        Short identifier. Appears in the benchmark report, so keep it terse.
    stimulus : callable
        Takes no arguments, returns the input passed to ``predict``.
    implant : callable
        Takes no arguments, returns an ``Implant``.
    model : callable
        Takes ``implant`` and ``verbose`` (and ``axon_pickle``/
        ``ignore_pickle`` when ``caches_axons`` is True), returns an
        *unbuilt* model.
    predict : callable, optional
        Takes the model and the stimulus, returns a percept. Defaults to
        ``model.predict_percept(stimulus)``.
    implant_encodes : bool
        Whether ``implant.prepare_stim`` alone turns the stimulus into
        stimulation. False when that needs the model (``Scene`` registration,
        ``TraceEncoder``); the ``implant`` benchmark is then skipped.
    caches_axons : bool
        Whether the model caches its axon map to disk. ``AxonMapSpatial``
        pickles the axon bundles to ``axons.pickle`` on first build, which
        makes a warm build roughly twice as fast as a cold one. Other models
        reject ``axon_pickle``: ``Parametrized`` freezes attributes, so an
        unknown keyword raises ``FreezeError``.
    slow : bool
        Whether the scenario is excluded from the default run. Set when a
        single prediction takes more than a few seconds: timing calls it
        several times and peak memory once more, so the cost is roughly 10x.
        Slow scenarios run only with ``--runslow``.
    """

    id: str
    stimulus: Callable
    implant: Callable
    model: Callable
    predict: Callable = lambda model, stim: model.predict_percept(stim)
    implant_encodes: bool = True
    caches_axons: bool = False
    slow: bool = False


SCENARIOS = [
    # Image, FrequencyEncoder, and Granley's pulse-dependent axon map:
    Scenario(
        id='imie_biphasic_image',
        stimulus=p2p.stimuli.samples.logo_bvl,
        implant=imie,
        model=lambda **kwargs: p2p.models.retina.BiphasicAxonMapModel(
            xrange=(-14, 14), yrange=(-10, 10), step=0.5, **kwargs),
        caches_axons=True,
    ),
    # Video, AmplitudeEncoder, raster, and the Torch spatiotemporal composite:
    Scenario(
        id='argus2_axonmap_fading_video',
        stimulus=drifting_grating,
        implant=p2p.implants.retina.ArgusII,
        model=axonmap_fading,
        caches_axons=True,
    ),
    # Scene coordinates, gaze, and PRIMA's optical encoder:
    Scenario(
        id='prima_ho2018_scene_gaze',
        stimulus=scotoma_scene,
        implant=p2p.implants.retina.PRIMAPivotal,
        model=lambda **kwargs: p2p.models.retina.Ho2018Model(
            xrange=(-6, 6), yrange=(-6, 6), step=0.1, **kwargs),
        predict=lambda model, scene: model.predict_percept(scene, gaze=GAZE),
        implant_encodes=False,
    ),
    # Cortex: retinotopic map, placement, and sequential TraceEncoder input:
    Scenario(
        id='orion_dynaphos_trace',
        stimulus=letter_z,
        implant=p2p.implants.cortex.Orion,
        model=lambda **kwargs: p2p.models.cortex.DynaphosModel(
            visual_field_map=p2p.topography.cortex.Polimeni2006Map(
                regions=['v1']),
            implant_position=(20, -5) * mm, xrange=(-6, 0), yrange=(-1, 4.5),
            step=0.1, **kwargs),
        predict=trace,
        implant_encodes=False,
    ),
]
