"""Fixtures and measurement helpers for the benchmark suite.

Benchmarks are skipped unless pytest is invoked with ``--benchmark-only``. If
``pytest-benchmark`` is not installed, the benchmark modules are not collected.
"""
import gc
import tracemalloc
from pathlib import Path

import pytest

try:
    import pytest_benchmark  # noqa: F401
except ImportError:  # pragma: no cover - depends on the local environment
    HAVE_BENCHMARK = False
    # Modules that need the plugin; add new benchmark modules here.
    # test_compare.py does not need the plugin and must stay collectable.
    collect_ignore = ['test_predict.py']
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
    # Register the marker when pytest-benchmark is disabled (``-p no:benchmark``)
    # to avoid PytestUnknownMarkWarning:
    if not config.pluginmanager.hasplugin('benchmark'):
        config.addinivalue_line('markers',
                                'benchmark: mark a pulse2percept benchmark')


def pytest_collection_modifyitems(config, items):
    """Skip tests that use the ``benchmark`` fixture unless
    ``--benchmark-only`` is given.

    ``test_compare.py`` does not use the fixture, so it still runs in a plain
    ``pytest benchmarks/``.
    """
    if config.getoption('benchmark_only', default=False):
        return
    skip = pytest.mark.skip(reason='needs --benchmark-only to run')
    for item in items:
        if HERE not in Path(str(item.fspath)).parents:
            continue
        if 'benchmark' in getattr(item, 'fixturenames', ()):
            item.add_marker(skip)


@pytest.fixture(scope='session')
def n_threads(pytestconfig):
    """Number of OpenMP threads to give the models."""
    return pytestconfig.getoption('n_threads')


@pytest.fixture(scope='session')
def axon_pickle(tmp_path_factory):
    """Return a temporary path for the axon-map cache.

    Avoids writing ``axons.pickle`` into the working directory and reusing a
    stale cache from a previous run.
    """
    return str(tmp_path_factory.mktemp('axon_cache') / 'axons.pickle')


@pytest.fixture(scope='module', params=SCENARIOS, ids=lambda s: s.id)
def scenario(request):
    """Return the scenario under test.

    Scenarios flagged ``slow`` are skipped unless ``--runslow`` is given.
    ``--runslow`` is registered in the root ``conftest.py``; registering it
    here too is a conflicting-option error.
    """
    if request.param.slow and not request.config.getoption('runslow',
                                                           default=False):
        pytest.skip(f'{request.param.id} is slow; use --runslow to include it')
    return request.param


@pytest.fixture(scope='module')
def make_model(scenario, n_threads, axon_pickle):
    """Return a factory for fresh, *unbuilt* models.

    ``build`` benchmarks need a new model per round, created outside the timed
    section.
    """
    def _make(implant, ignore_pickle=False):
        kwargs = {'verbose': False, 'n_threads': n_threads}
        if scenario.binds_implant:
            kwargs['implant'] = implant
        if scenario.caches_axons:
            kwargs['axon_pickle'] = axon_pickle
            kwargs['ignore_pickle'] = ignore_pickle
        return scenario.model(**kwargs)
    return _make


@pytest.fixture(scope='module')
def implant(scenario):
    """The scenario's device."""
    return scenario.implant()


@pytest.fixture(scope='module')
def source(scenario, implant):
    """Return the input passed to ``predict_percept``."""
    return scenario.source(implant, scenario.stimulus())


@pytest.fixture(scope='module')
def built_model(make_model, implant):
    """Return a built model, shared across benchmarks.

    ``predict_percept`` does not mutate the model, so reuse is safe."""
    return make_model(implant).build()


@pytest.fixture(scope='module')
def percept(built_model, source):
    """Return a predicted percept."""
    return built_model.predict_percept(source)


@pytest.fixture
def peak_memory():
    """Return a helper that measures peak memory of a single call, in MB.

    Uses ``tracemalloc`` instead of RSS sampling: deterministic, no extra
    dependency, and works on Windows (which rules out ``pytest-memray``). It
    tracks NumPy data buffers, which hold nearly all memory in these workloads,
    but not raw ``malloc`` inside the Cython/OpenMP kernels, so the numbers are
    a floor for those code paths.

    Call outside the timed section: tracing inflates run time several-fold.
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
