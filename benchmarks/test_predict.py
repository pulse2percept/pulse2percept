"""Benchmarks for the core pipeline: stimulus -> implant -> model -> percept.

Every scenario in :mod:`scenarios` is measured at each stage of that pipeline
separately, and once end to end. The decomposition is the point: an end-to-end
number on its own cannot say whether a regression moved into stimulus
construction, the model build, or the percept computation.

Each benchmark reports wall-clock time through the ``benchmark`` fixture and
peak memory through ``benchmark.extra_info``, so both land in the same JSON
when the run is saved. The two are measured in separate runs on purpose; see
the ``peak_memory`` fixture for why.
"""
import matplotlib.pyplot as plt
import pytest


@pytest.mark.benchmark(group='stimulus')
def test_stimulus(benchmark, scenario, peak_memory):
    """Constructing the stimulus, before any implant is involved."""
    stim = benchmark(scenario.stimulus)
    benchmark.extra_info['peak_mem_mb'] = peak_memory(scenario.stimulus)
    benchmark.extra_info['stim_shape'] = str(stim.shape)


@pytest.mark.benchmark(group='implant')
def test_implant(benchmark, scenario, peak_memory):
    """Assigning the stimulus to the implant.

    This is not just bookkeeping: an image stimulus has far more pixels than
    the array has electrodes, so the setter downsamples it onto the electrode
    grid. Building the stimulus happens in ``setup`` and is not timed.
    """
    def setup():
        return (scenario.stimulus(),), {}

    implant = benchmark.pedantic(scenario.implant, setup=setup, rounds=20,
                                 iterations=1, warmup_rounds=1)
    stim = scenario.stimulus()
    benchmark.extra_info['peak_mem_mb'] = peak_memory(scenario.implant, stim)
    benchmark.extra_info['n_electrodes'] = implant.n_electrodes


@pytest.mark.benchmark(group='build')
def test_build(benchmark, scenario, make_model, peak_memory, n_threads):
    """Building the model, with any on-disk cache already warm.

    This is the path a user hits on every run after the first, so it is the
    number that describes their experience. For the axon-map models it is
    dominated by reading the pickled bundles and recomputing axon
    sensitivity; see ``test_build_cold`` for the underlying computation.
    """
    if scenario.caches_axons:
        # Populate the cache first so this benchmark measures the warm path
        # no matter which order the tests run in.
        make_model(ignore_pickle=False).build()

    def setup():
        return (make_model(ignore_pickle=False),), {}

    benchmark.pedantic(lambda model: model.build(), setup=setup, rounds=5,
                       iterations=1, warmup_rounds=1)
    benchmark.extra_info['peak_mem_mb'] = peak_memory(
        lambda: make_model(ignore_pickle=False).build())
    benchmark.extra_info['n_threads'] = n_threads


@pytest.mark.benchmark(group='build')
def test_build_cold(benchmark, scenario, make_model, peak_memory, n_threads):
    """Building the model from scratch, ignoring the on-disk cache.

    This is the actual Jansonius-model computation -- the part worth
    optimizing -- rather than the cost of unpickling last run's result.
    """
    if not scenario.caches_axons:
        pytest.skip(f'{scenario.id} has no on-disk cache, so a cold build is '
                    f'the same as a warm one (see test_build)')

    def setup():
        return (make_model(ignore_pickle=True),), {}

    benchmark.pedantic(lambda model: model.build(), setup=setup, rounds=5,
                       iterations=1, warmup_rounds=1)
    benchmark.extra_info['peak_mem_mb'] = peak_memory(
        lambda: make_model(ignore_pickle=True).build())
    benchmark.extra_info['n_threads'] = n_threads


@pytest.mark.benchmark(group='predict_percept')
def test_predict_percept(benchmark, built_model, implant, peak_memory,
                         n_threads):
    """Predicting the percept: the headline number for this library."""
    percept = benchmark(built_model.predict_percept, implant)
    benchmark.extra_info['peak_mem_mb'] = peak_memory(
        built_model.predict_percept, implant)
    benchmark.extra_info['n_threads'] = n_threads
    benchmark.extra_info['percept_shape'] = str(percept.shape)


@pytest.mark.benchmark(group='end_to_end')
def test_end_to_end(benchmark, scenario, make_model, peak_memory, n_threads):
    """The whole pipeline, as a user would write it in one line.

    The only departure from the one-liner in :mod:`scenarios` is that the
    model is constructed through ``make_model``, so that it writes its axon
    cache to a temporary directory instead of the current one.
    """
    def run():
        implant = scenario.implant(scenario.stimulus())
        return make_model().build().predict_percept(implant)

    percept = benchmark(run)
    benchmark.extra_info['peak_mem_mb'] = peak_memory(run)
    benchmark.extra_info['n_threads'] = n_threads
    benchmark.extra_info['percept_shape'] = str(percept.shape)


@pytest.mark.benchmark(group='plot')
def test_plot(benchmark, scenario, percept, peak_memory):
    """Drawing the percept.

    Mostly a matplotlib measurement rather than a pulse2percept one, kept in
    its own group so it never gets read as part of the model cost. The axes
    are cleared and reused between rounds: a fresh figure per round would
    accumulate hundreds of them, and clearing inside the timed section would
    charge the teardown to the plot.
    """
    if not scenario.plottable:
        pytest.skip(f'{scenario.id} has a temporal-only model, whose percept '
                    f'has no spatial grid for Percept.plot to draw')

    fig, ax = plt.subplots()
    try:
        def setup():
            ax.clear()
            return (), {'ax': ax}

        benchmark.pedantic(percept.plot, setup=setup, rounds=20, iterations=1,
                           warmup_rounds=1)
        ax.clear()
        benchmark.extra_info['peak_mem_mb'] = peak_memory(percept.plot, ax=ax)
    finally:
        plt.close(fig)
