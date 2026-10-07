"""Pipelines exercised by the benchmark suite.

A :class:`Scenario` is one end-to-end pipeline: build a stimulus, bind a model
to an implant, predict a percept. The benchmarks in ``test_predict.py`` are
parametrized over :data:`SCENARIOS`, so a new case needs only a new entry here.

The first two scenarios correspond to these one-liners::

    implant = p2p.implants.retina.ArgusII()
    p2p.models.retina.AxonMapModel(implant=implant, yrange=(-8, 8),
                                   xrange=(-12, 12)).predict_percept(
        as_current(implant, p2p.stimuli.samples.logo_bvl()))

    p2p.models.retina.ScoreboardModel(implant=p2p.implants.retina.PRIMAPivotal(),
                                      yrange=(-4, 4), xrange=(-4, 4), rho=50,
                                      step=0.1).predict_percept(
        p2p.stimuli.samples.logo_bvl().invert())

The PRIMA scenario runs its optical encoder directly. Electrical image
scenarios use :func:`as_current` to keep their historical benchmark workload.

Together the scenarios reach every compiled kernel used in percept prediction:
``_granley2021``, ``_nanduri2012``, ``_horsager2009`` and ``_thompson2003``.
A new scenario should reach a kernel no existing scenario reaches.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

import pulse2percept as p2p


def array_ptrain(implant_cls, amp=20):
    """Return a ``BiphasicPulseTrain`` on *every* electrode of ``implant_cls``.

    A bare ``BiphasicPulseTrain`` drives a single electrode:
    ``ArgusII().prepare_stim(BiphasicPulseTrain(...)).shape`` is ``(1, 29)``,
    not ``(60, 29)``, i.e. one sixtieth of the per-electrode work.

    A temporary implant supplies the electrode names, which adds about 2 ms to
    the ~15 ms ``stimulus`` benchmark.

    ``amp`` is in microamps for the current-based models. Granley uses
    multiples of threshold, so its scenario passes ``20 * xTh``; the workload
    is the same.
    """
    names = implant_cls().electrode_names
    return p2p.stimuli.Stimulus(
        {e: p2p.stimuli.BiphasicPulseTrain(20, amp, 0.45, stim_dur=200)
         for e in names})


#: Microamps per gray level of 1.0 in :func:`as_current`.
#:
#: 1.0 keeps amplitudes numerically identical to older benchmark runs. The
#: kernels do the same arithmetic on any amplitude, so the value can be
#: increased if a scenario needs a clinically plausible one.
GRAY_LEVEL_UA = 1.0


def drifting_grating(n_frames=94, fps=29.97, shape=(240, 426)):
    """Return a drifting sinusoidal grating video (grayscale, 94 frames)."""
    rows, cols = shape
    x = np.linspace(0, 4 * np.pi, cols)[np.newaxis, :, np.newaxis]
    phase = 2 * np.pi * np.arange(n_frames) / n_frames
    return p2p.stimuli.VideoStimulus(
        np.tile(0.5 + 0.5 * np.sin(x - phase), (rows, 1, 1)),
        metadata={'fps': fps})


def as_current(implant, picture, amp_max=GRAY_LEVEL_UA):
    """Sample an image onto an implant, treating gray levels as microamps.

    Implants and ``predict_percept`` reject images, which are dimensionless.
    User code converts an image to current with an encoder (see
    :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`). An encoder produces a
    pulse train per electrode, a much larger workload than the single static
    frame these scenarios have always measured. This helper instead scales the
    amplitudes that ``reshape_stim`` resampled onto the electrodes, so the
    kernels receive the same electrodes and values as in older runs.

    Returns the unprepared source passed to ``predict_percept``;
    ``prepare_stim`` is measured separately by the ``implant`` benchmark.
    """
    stim = implant.reshape_stim(picture)
    data = stim.data * amp_max
    if stim.time is None:
        # Flat array = N electrodes with no time axis, as for an image. An
        # (N, 1) array would get time [0] and a different `predict_percept`
        # path:
        return p2p.stimuli.Stimulus(data.ravel(), electrodes=stim.electrodes)
    return p2p.stimuli.Stimulus(data, electrodes=stim.electrodes,
                                time=stim.time)


def axonmap(n_threads, **kwargs):
    """Return an AxonMap model; ``n_threads`` is unused (Torch threads)."""
    return p2p.models.retina.AxonMapModel(xrange=(-12, 12), yrange=(-8, 8),
                                          **kwargs)


def axonmap_fading(implant, verbose, n_threads, **axon_cache):
    """Return an AxonMap + Fading composite; cache keywords go to AxonMap."""
    return p2p.models.Model(
        spatial=p2p.models.retina.AxonMapSpatial(
            implant, xrange=(-12, 12), yrange=(-8, 8), verbose=verbose,
            **axon_cache),
        temporal=p2p.models.FadingTemporal(verbose=verbose))


@dataclass(frozen=True)
class Scenario:
    """One stimulus/implant/model pipeline.

    Attributes
    ----------
    id : str
        Short identifier. Appears in the benchmark report, so keep it terse.
    stimulus : callable
        Takes no arguments, returns a stimulus.
    implant : callable
        Takes no arguments, returns an ``Implant``.
    source : callable, optional
        Takes the implant and the stimulus, returns what ``predict_percept``
        is given. Defaults to the stimulus unchanged; the image scenarios use
        :func:`as_current`.
    model : callable
        Takes keyword arguments, returns an *unbuilt* model. Always receives
        ``verbose`` and ``n_threads``; also receives ``implant`` unless
        ``binds_implant`` is False, and ``axon_pickle``/``ignore_pickle``
        when ``caches_axons`` is True.
    binds_implant : bool
        Whether the model takes an ``implant``. False for a temporal-only
        model, which receives the stimulus directly.
    caches_axons : bool
        Whether the model caches its axon map to disk. ``AxonMapSpatial``
        pickles the axon bundles to ``axons.pickle`` on first build, which
        makes a warm build roughly twice as fast as a cold one. Other models
        reject ``axon_pickle``: ``Parametrized`` freezes attributes, so an
        unknown keyword raises ``FreezeError``.
    slow : bool
        Whether the scenario is excluded from the default run. Set when a
        single ``predict_percept`` takes more than a few seconds: timing calls
        it several times and peak memory once more under ``tracemalloc``, so
        the cost is roughly 10x. Slow scenarios run only with ``--runslow``.
    plottable : bool
        Whether the percept can be drawn. A temporal-only model has no spatial
        grid (``xdva`` is None), and ``Percept.plot`` raises ``TypeError`` on
        it, so the plot benchmark is skipped.
    """

    id: str
    stimulus: Callable
    implant: Callable
    model: Callable
    source: Callable = lambda implant, stim: stim
    binds_implant: bool = True
    caches_axons: bool = False
    slow: bool = False
    plottable: bool = True


SCENARIOS = [
    Scenario(
        id='argus2_axonmap_logobvl',
        stimulus=lambda: p2p.stimuli.samples.logo_bvl(),
        implant=p2p.implants.retina.ArgusII,
        source=as_current,
        model=axonmap,
        caches_axons=True,
    ),
    Scenario(
        # Benchmark the image-to-optical encoding path for PRIMA.
        id='prima_scoreboard_logobvl',
        stimulus=lambda: p2p.stimuli.samples.logo_bvl().invert(),
        implant=p2p.implants.retina.PRIMAPivotal,
        # Scoreboard runs on Torch's thread pool, so `n_threads` is unused:
        model=lambda n_threads, **kwargs: p2p.models.retina.ScoreboardModel(
            xrange=(-4, 4), yrange=(-4, 4), rho=50, step=0.1, **kwargs),
    ),
    # Granley 2021 reads amplitude, frequency and pulse duration from each
    # electrode's BiphasicPulseTrain and rejects images. Amplitude is in xTh:
    Scenario(
        id='argus2_biphasic_ptrain',
        stimulus=lambda: array_ptrain(p2p.implants.retina.ArgusII,
                                      amp=20 * p2p.units.xTh),
        implant=p2p.implants.retina.ArgusII,
        model=lambda **kwargs: p2p.models.retina.BiphasicAxonMapModel(
            xrange=(-12, 12), yrange=(-8, 8), **kwargs),
        caches_axons=True,
    ),
    # Nanduri 2012: multi-frame output. Reaches both halves of _nanduri2012
    # (spatial_fast, temporal_fast) and the spatial -> temporal step in Model:
    Scenario(
        id='argus2_nanduri2012_ptrain',
        stimulus=lambda: array_ptrain(p2p.implants.retina.ArgusII),
        implant=p2p.implants.retina.ArgusII,
        model=lambda **kwargs: p2p.models.retina.Nanduri2012Model(
            xrange=(-4, 4), yrange=(-4, 4), step=0.5, **kwargs),
    ),
    # Horsager 2009: temporal-only, one trace per electrode, no spatial grid.
    # Only scenario that reaches _horsager2009:
    Scenario(
        id='argus2_horsager2009_ptrain',
        stimulus=lambda: array_ptrain(p2p.implants.retina.ArgusII),
        implant=p2p.implants.retina.ArgusII,
        model=lambda **kwargs: p2p.models.retina.Horsager2009Model(**kwargs),
        binds_implant=False,
        plottable=False,
    ),
    # Thompson 2003: spatial-only, image input. Only scenario that reaches
    # _thompson2003:
    Scenario(
        id='argus2_thompson2003_logobvl',
        stimulus=lambda: p2p.stimuli.samples.logo_bvl(),
        implant=p2p.implants.retina.ArgusII,
        source=as_current,
        model=lambda **kwargs: p2p.models.retina.Thompson2003Model(
            xrange=(-12, 12), yrange=(-8, 8), **kwargs),
    ),
    # Composed Model (separate spatial + temporal) on the Torch core. Both
    # components use Torch threads, so `n_threads` is unused:
    Scenario(
        id='argus2_scoreboard_fading_ptrain',
        stimulus=lambda: array_ptrain(p2p.implants.retina.ArgusII),
        implant=p2p.implants.retina.ArgusII,
        model=lambda implant, verbose, n_threads: p2p.models.Model(
            spatial=p2p.models.retina.ScoreboardSpatial(
                implant, xrange=(-4, 4), yrange=(-4, 4), step=0.5,
                verbose=verbose),
            temporal=p2p.models.FadingTemporal(verbose=verbose)),
    ),
    # AxonMap + Fading: the composite runs the Torch AxonMap core on every
    # electrode, the spatial-only scenarios on the compressed stimulus:
    Scenario(
        id='argus2_axonmap_fading_ptrain',
        stimulus=lambda: array_ptrain(p2p.implants.retina.ArgusII),
        implant=p2p.implants.retina.ArgusII,
        model=axonmap_fading,
        caches_axons=True,
    ),
    Scenario(
        id='argus2_axonmap_video',
        stimulus=drifting_grating,
        implant=p2p.implants.retina.ArgusII,
        source=as_current,
        model=axonmap,
        caches_axons=True,
    ),
]
