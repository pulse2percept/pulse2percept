"""Pipelines exercised by the benchmark suite.

A :class:`Scenario` is one end-to-end path through the library: build a
stimulus, hand it to an implant, build a model, predict a percept. The
benchmark functions in ``test_predict.py`` are written once and parametrized
over :data:`SCENARIOS`, so adding a case means adding an entry here and
nothing else.

The scenarios below are the reference workloads for the library's main purpose
-- predicting a percept from a stimulus, an implant and a phosphene model. The
first two correspond to these one-liners::

    p2p.models.AxonMapModel(yrange=(-8, 8), xrange=(-12, 12)).build(
        ).predict_percept(p2p.implants.ArgusII(stim=p2p.stimuli.LogoBVL()))

    p2p.models.ScoreboardModel(yrange=(-4, 4), xrange=(-4, 4), rho=50,
                               xystep=0.1).build().predict_percept(
        p2p.implants.PRIMA(stim=p2p.stimuli.LogoBVL().invert()))

Between them the scenarios cover every compiled kernel that a percept
prediction can go through -- ``_beyeler2019``, ``_granley2021``,
``_nanduri2012`` and the shared ``_temporal`` loop -- so that a change to any
one of them shows up here. That coverage is the selection criterion: a model
that shares a kernel with one already listed adds run time without adding
signal.
"""
from dataclasses import dataclass
from typing import Callable

import pulse2percept as p2p


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
        Takes a stimulus, returns a ``ProsthesisSystem``.
    model : callable
        Takes keyword arguments, returns an *unbuilt* model. Always receives
        ``verbose`` and ``n_threads``; also receives ``axon_pickle`` and
        ``ignore_pickle`` when ``caches_axons`` is True.
    caches_axons : bool
        Whether the model caches its axon map to disk. ``AxonMapSpatial``
        pickles the grown axon bundles to ``axons.pickle`` in the working
        directory on first build and reuses them afterwards, which makes a
        warm build roughly twice as fast as a cold one. Models without that
        cache reject ``axon_pickle`` outright -- ``Parametrized`` freezes
        attributes, so an unknown keyword raises ``FreezeError`` rather than
        being ignored -- which is why the benchmarks have to know which model
        is which.
    slow : bool
        Whether the scenario is too slow for the default run. Set this when a
        single ``predict_percept`` takes more than a few seconds: the timing
        loop calls it several times over, and measuring peak memory calls it
        once more under ``tracemalloc``, so the cost is multiplied by roughly
        an order of magnitude. Slow scenarios run only with ``--runslow``.
    """

    id: str
    stimulus: Callable
    implant: Callable
    model: Callable
    caches_axons: bool = False
    slow: bool = False


SCENARIOS = [
    Scenario(
        id='argus2_axonmap_logobvl',
        stimulus=lambda: p2p.stimuli.LogoBVL(),
        implant=lambda stim: p2p.implants.ArgusII(stim=stim),
        model=lambda **kwargs: p2p.models.AxonMapModel(xrange=(-12, 12),
                                                       yrange=(-8, 8),
                                                       **kwargs),
        caches_axons=True,
    ),
    Scenario(
        id='prima_scoreboard_logobvl',
        stimulus=lambda: p2p.stimuli.LogoBVL().invert(),
        implant=lambda stim: p2p.implants.PRIMA(stim=stim),
        model=lambda **kwargs: p2p.models.ScoreboardModel(xrange=(-4, 4),
                                                          yrange=(-4, 4),
                                                          rho=50, xystep=0.1,
                                                          **kwargs),
    ),
    # Granley 2021:
    Scenario(
        id='argus2_biphasic_ptrain',
        stimulus=lambda: p2p.stimuli.BiphasicPulseTrain(20, 1, 0.45),
        implant=lambda stim: p2p.implants.ArgusII(stim=stim),
        model=lambda **kwargs: p2p.models.BiphasicAxonMapModel(
            xrange=(-12, 12), yrange=(-8, 8), **kwargs),
        caches_axons=True,
    ),
    # Nanduri 2012: the first scenario with a temporal model
    Scenario(
        id='argus2_nanduri2012_ptrain',
        stimulus=lambda: p2p.stimuli.BiphasicPulseTrain(20, 20, 0.45,
                                                        stim_dur=200),
        implant=lambda stim: p2p.implants.ArgusII(stim=stim),
        model=lambda **kwargs: p2p.models.Nanduri2012Model(
            xrange=(-4, 4), yrange=(-4, 4), xystep=0.5, **kwargs),
    ),
    # A composed Model:
    Scenario(
        id='argus2_scoreboard_fading_ptrain',
        stimulus=lambda: p2p.stimuli.BiphasicPulseTrain(20, 20, 0.45,
                                                        stim_dur=200),
        implant=lambda stim: p2p.implants.ArgusII(stim=stim),
        model=lambda **kwargs: p2p.models.Model(
            spatial=p2p.models.ScoreboardSpatial(xrange=(-4, 4),
                                                 yrange=(-4, 4), xystep=0.5),
            temporal=p2p.models.FadingTemporal(), **kwargs),
    ),
    # A 94-frame video: the spatial model runs once per frame, so a single
    # predict_percept takes roughly a minute where the image scenarios above
    # take well under a second. Slow, so it stays out of the default run.
    Scenario(
        id='argus2_axonmap_bostontrain',
        stimulus=lambda: p2p.stimuli.BostonTrain().rgb2gray(),
        implant=lambda stim: p2p.implants.ArgusII(stim=stim),
        model=lambda **kwargs: p2p.models.AxonMapModel(xrange=(-12, 12),
                                                       yrange=(-8, 8),
                                                       **kwargs),
        caches_axons=True,
        slow=True,
    ),
]
