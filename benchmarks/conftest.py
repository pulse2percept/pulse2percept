"""Fixtures and measurement helpers for the benchmark suite.

Benchmarks are skipped unless pytest is invoked with ``--benchmark-only``, so a
bare ``pytest`` at the repository root stays a test run. If ``pytest-benchmark``
is not installed, the benchmark modules are not collected at all, so
contributors without the ``benchmark`` extra see nothing break.
"""
import gc
import tracemalloc
from pathlib import Path

import pytest

try:
    import pytest_benchmark  # noqa: F401
except ImportError:  # pragma: no cover - depends on the local environment
    HAVE_BENCHMARK = False
    collect_ignore_glob = ['test_*.py']
else:
    HAVE_BENCHMARK = True

from scenarios import SCENARIOS

HERE = Path(__file__).parent


def pytest_addoption(parser):
    parser.addoption(
        '--n-threads', action='store', type=int, default=1,
        help='Number of OpenMP threads the models may use (default: 1). '
             'Timings are only comparable across machines when this is '
             'pinned, which is why it does not default to the library default '
             'of one thread per CPU.'
    )


def pytest_configure(config):
    # pytest-benchmark registers this marker itself, but not when the plugin is
    # disabled with ``-p no:benchmark``. Registering it here keeps that case
    # free of PytestUnknownMarkWarning noise.
    if not config.pluginmanager.hasplugin('benchmark'):
        config.addinivalue_line('markers',
                                'benchmark: mark a pulse2percept benchmark')


def pytest_collection_modifyitems(config, items):
    """Skip the benchmarks unless they were explicitly asked for."""
    if config.getoption('benchmark_only', default=False):
        return
    skip = pytest.mark.skip(reason='needs --benchmark-only to run')
    for item in items:
        if HERE in Path(str(item.fspath)).parents:
            item.add_marker(skip)


@pytest.fixture(scope='session')
def n_threads(pytestconfig):
    """Number of OpenMP threads to give the models."""
    return pytestconfig.getoption('n_threads')


@pytest.fixture(scope='session')
def axon_pickle(tmp_path_factory):
    """Path for the axon-map cache.

    Keeps ``AxonMapSpatial`` from writing ``axons.pickle`` into the directory
    the benchmarks happened to be run from, and keeps a stale cache from a
    previous run out of the measurement.
    """
    return str(tmp_path_factory.mktemp('axon_cache') / 'axons.pickle')


@pytest.fixture(scope='module', params=SCENARIOS, ids=lambda s: s.id)
def scenario(request):
    """The pipeline under test."""
    return request.param


@pytest.fixture(scope='module')
def make_model(scenario, n_threads, axon_pickle):
    """Factory for fresh, *unbuilt* models.

    Benchmarks that measure ``build`` need a new model for every round, and
    they need it built outside the timed section.
    """
    def _make(ignore_pickle=False):
        kwargs = {'verbose': False, 'n_threads': n_threads}
        if scenario.caches_axons:
            kwargs['axon_pickle'] = axon_pickle
            kwargs['ignore_pickle'] = ignore_pickle
        return scenario.model(**kwargs)
    return _make


@pytest.fixture(scope='module')
def implant(scenario):
    """An implant with the scenario's stimulus already assigned."""
    return scenario.implant(scenario.stimulus())


@pytest.fixture(scope='module')
def built_model(make_model):
    """A model that has been built once, shared by the benchmarks that need
    one. ``predict_percept`` does not mutate the model, so reuse is safe."""
    return make_model().build()


@pytest.fixture(scope='module')
def percept(built_model, implant):
    """A predicted percept, for the benchmarks that consume one."""
    return built_model.predict_percept(implant)


@pytest.fixture
def peak_memory():
    """Return a helper that measures peak memory of a single call, in MB.

    ``tracemalloc`` is used rather than RSS sampling because it is
    deterministic, needs no extra dependency, and works on Windows (which
    rules out ``pytest-memray``). It tracks NumPy data buffers, which is where
    essentially all of the memory in these workloads goes.

    It does *not* see raw ``malloc`` inside the Cython/OpenMP kernels, so
    treat the numbers as a floor for those code paths rather than a total.

    Always call this outside the timed section: tracing inflates run time
    several-fold, so a timing that included it would be measuring the
    profiler.
    """
    def _measure(fn, *args, **kwargs):
        gc.collect()
        tracemalloc.start()
        try:
            fn(*args, **kwargs)
            return round(tracemalloc.get_traced_memory()[1] / 1e6, 3)
        finally:
            tracemalloc.stop()
    return _measure
