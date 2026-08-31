"""Pipelines exercised by the benchmark suite.

A :class:`Scenario` is one end-to-end path through the library: build a
stimulus, bind a model to an implant, predict a percept. The benchmark
functions in ``test_predict.py`` are written once and parametrized over
:data:`SCENARIOS`, so adding a case means adding an entry here and nothing
else.

The scenarios below are the reference workloads for the library's main purpose
-- predicting a percept from a stimulus, an implant and a phosphene model. The
first two correspond to these one-liners::

    implant = p2p.implants.ArgusII()
    p2p.models.AxonMapModel(implant=implant, yrange=(-8, 8),
                            xrange=(-12, 12)).predict_percept(
        as_current(implant, p2p.stimuli.LogoBVL()))

    p2p.models.ScoreboardModel(implant=p2p.implants.PRIMAPivotal(),
                               yrange=(-4, 4), xrange=(-4, 4), rho=50,
                               step=0.1).predict_percept(
        p2p.stimuli.LogoBVL().invert())

The PRIMA scenario runs its optical encoder directly. Electrical scenarios
use :func:`as_current` to preserve their historical benchmark workload.

Between them the scenarios reach every compiled kernel a percept prediction can
go through -- ``_beyeler2019``, ``_granley2021``, ``_nanduri2012``,
``_horsager2009``, ``_thompson2003`` and the shared ``_temporal`` loop -- so
that a change to any one of them shows up here. That coverage is the selection
criterion: a model that shares a kernel with one already listed adds run time
without adding signal, and a kernel no scenario reaches is a kernel the
regression check cannot see.
"""
from dataclasses import dataclass
from typing import Callable

import pulse2percept as p2p


def array_ptrain(implant_cls, amp=20):
    """A ``BiphasicPulseTrain`` on *every* electrode of ``implant_cls``.

    Handing a bare ``BiphasicPulseTrain`` to an implant drives a single
    electrode -- ``ArgusII().prepare_stim(BiphasicPulseTrain(...)).shape`` is
    ``(1, 29)``, not ``(60, 29)``. A benchmark built that way would exercise
    one sixtieth of the per-electrode work the kernels actually do, and would
    barely move if that work regressed. Every pulse-train scenario below goes
    through here instead.

    The throwaway implant is what supplies the electrode names, so the
    ``stimulus`` benchmark carries about 2 ms of implant construction on top of
    the 15 ms of building the pulse trains. That is small, and the alternative
    -- hard-coding the electrode names -- would duplicate the implant.

    ``amp`` defaults to microamps, which is what the current-based models
    below read. Granley reads multiples of threshold instead, so its scenario
    passes ``20 * xTh``; the workload is the same either way.
    """
    names = implant_cls().electrode_names
    return p2p.stimuli.Stimulus(
        {e: p2p.stimuli.BiphasicPulseTrain(20, amp, 0.45, stim_dur=200)
         for e in names})


#: Microamps that a gray level of 1.0 stands for in :func:`as_current`.
#:
#: One, so that the amplitudes are numerically what an image handed straight
#: to an implant used to produce, and these benchmarks stay comparable across
#: the release that made the reinterpretation explicit. Nothing here depends on
#: the value -- the kernels below do the same arithmetic on any amplitude --
#: so raise it if a scenario ever needs a clinically plausible one.
GRAY_LEVEL_UA = 1.0


def as_current(implant, picture, amp_max=GRAY_LEVEL_UA):
    """Sample a picture onto an implant, reading its gray levels as microamps.

    An image is dimensionless, and neither an implant nor ``predict_percept``
    accepts one: gray levels are not small currents. Turning one into current
    is an encoder's job (see
    :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`), and that is what a
    user should write.

    A benchmark wants something else. An encoder gives every electrode a pulse
    train, which is a far larger and quite differently shaped workload than the
    single static frame these scenarios have always measured -- and measuring
    something else is how a performance suite loses its history. So the
    reinterpretation the library used to perform silently is spelled out here
    instead, on the amplitudes ``reshape_stim`` has resampled onto the
    implant's electrodes. What the kernels see does not change: the same
    electrodes, the same number of columns, the same numbers in them.

    Returns the source a model is handed, not a prepared stimulus: the implant
    prepares it inside ``predict_percept``, and ``prepare_stim`` is measured
    on its own by the ``implant`` benchmark.
    """
    stim = implant.reshape_stim(picture)
    data = stim.data * amp_max
    if stim.time is None:
        # A *flat* sequence means N electrodes stimulated once each with no
        # time component, which is what an image handed to an implant
        # produces. Handing over the (N, 1) array instead would give it a time
        # axis of [0], and a spatial model takes a different path through
        # `predict_percept` for a stimulus that has one -- so the benchmark
        # would quietly start measuring something else.
        return p2p.stimuli.Stimulus(data.ravel(), electrodes=stim.electrodes)
    return p2p.stimuli.Stimulus(data, electrodes=stim.electrodes,
                                time=stim.time)


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
        model, which never sees an electrode and is handed the stimulus
        directly.
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
    plottable : bool
        Whether the resulting percept can be drawn. A temporal-only model has
        no spatial grid -- its percept comes back with ``xdva`` set to None --
        and ``Percept.plot`` raises ``TypeError`` on one. That is a limitation
        of the library rather than of the benchmark, so the plot benchmark is
        skipped for such scenarios instead of being worked around here.
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
        stimulus=lambda: p2p.stimuli.LogoBVL(),
        implant=p2p.implants.ArgusII,
        source=as_current,
        model=lambda **kwargs: p2p.models.AxonMapModel(xrange=(-12, 12),
                                                       yrange=(-8, 8),
                                                       **kwargs),
        caches_axons=True,
    ),
    Scenario(
        # Benchmark the image-to-optical encoding path for PRIMA.
        id='prima_scoreboard_logobvl',
        stimulus=lambda: p2p.stimuli.LogoBVL().invert(),
        implant=p2p.implants.PRIMAPivotal,
        model=lambda **kwargs: p2p.models.ScoreboardModel(xrange=(-4, 4),
                                                          yrange=(-4, 4),
                                                          rho=50, step=0.1,
                                                          **kwargs),
    ),
    # Granley 2021. Its stimulus is a pulse train rather than an image because
    # the model reads amplitude, frequency and pulse duration off each
    # electrode's BiphasicPulseTrain and rejects anything else. Its amplitude
    # is a multiple of threshold, unlike every other scenario here.
    Scenario(
        id='argus2_biphasic_ptrain',
        stimulus=lambda: array_ptrain(p2p.implants.ArgusII,
                                      amp=20 * p2p.units.xTh),
        implant=p2p.implants.ArgusII,
        model=lambda **kwargs: p2p.models.BiphasicAxonMapModel(
            xrange=(-12, 12), yrange=(-8, 8), **kwargs),
        caches_axons=True,
    ),
    # Nanduri 2012: the first scenario with a temporal model, so the first one
    # whose predict_percept returns more than a single frame. Reaches both
    # halves of _nanduri2012 (spatial_fast and temporal_fast) and the
    # spatial -> temporal handoff in Model.
    Scenario(
        id='argus2_nanduri2012_ptrain',
        stimulus=lambda: array_ptrain(p2p.implants.ArgusII),
        implant=p2p.implants.ArgusII,
        model=lambda **kwargs: p2p.models.Nanduri2012Model(
            xrange=(-4, 4), yrange=(-4, 4), step=0.5, **kwargs),
    ),
    # Horsager 2009: a temporal-only model, so predict_percept returns one
    # trace per electrode with no spatial grid at all. The only scenario that
    # reaches _horsager2009.
    Scenario(
        id='argus2_horsager2009_ptrain',
        stimulus=lambda: array_ptrain(p2p.implants.ArgusII),
        implant=p2p.implants.ArgusII,
        model=lambda **kwargs: p2p.models.Horsager2009Model(**kwargs),
        binds_implant=False,
        plottable=False,
    ),
    # Thompson 2003: a spatial-only model taking an image, and the only
    # scenario that reaches _thompson2003.
    Scenario(
        id='argus2_thompson2003_logobvl',
        stimulus=lambda: p2p.stimuli.LogoBVL(),
        implant=p2p.implants.ArgusII,
        source=as_current,
        model=lambda **kwargs: p2p.models.Thompson2003Model(
            xrange=(-12, 12), yrange=(-8, 8), **kwargs),
    ),
    # A composed Model, which is how a user combines a spatial and a temporal
    # model that were not written as a pair. Reaches the generic _temporal
    # kernel that FadingTemporal and friends share.
    Scenario(
        id='argus2_scoreboard_fading_ptrain',
        stimulus=lambda: array_ptrain(p2p.implants.ArgusII),
        implant=p2p.implants.ArgusII,
        # `Model` takes components, not parameters, so `verbose`/`n_threads`
        # go to the components that own them.
        model=lambda implant, **kwargs: p2p.models.Model(
            spatial=p2p.models.ScoreboardSpatial(implant, xrange=(-4, 4),
                                                 yrange=(-4, 4), step=0.5,
                                                 **kwargs),
            temporal=p2p.models.FadingTemporal(**kwargs)),
    ),
    # A 94-frame video: the spatial model runs once per frame, so a single
    # predict_percept takes roughly a minute where the image scenarios above
    # take well under a second. Slow, so it stays out of the default run.
    Scenario(
        id='argus2_axonmap_bostontrain',
        stimulus=lambda: p2p.stimuli.BostonTrain().rgb2gray(),
        implant=p2p.implants.ArgusII,
        source=as_current,
        model=lambda **kwargs: p2p.models.AxonMapModel(xrange=(-12, 12),
                                                       yrange=(-8, 8),
                                                       **kwargs),
        caches_axons=True,
        slow=True,
    ),
]
