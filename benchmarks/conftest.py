"""Fixtures and measurement helpers for the benchmark suite.

Benchmarks are skipped unless pytest is invoked with ``--benchmark-only``. If
``pytest-benchmark`` is not installed, the benchmark modules are not collected.
"""
import gc
import sys
import tempfile
import tracemalloc
from pathlib import Path

import pytest
# Loaded before any measurement, so its import-time allocations do not count
# toward whichever benchmark first imports it:
import torch

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
def make_model(scenario, axon_pickle):
    """Return a factory for fresh, *unbuilt* models.

    ``build`` benchmarks need a new model per round, created outside the timed
    section.
    """
    def _make(implant, ignore_pickle=False):
        kwargs = {'implant': implant, 'verbose': False}
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
def source(scenario):
    """Return the input passed to ``scenario.predict``."""
    return scenario.stimulus()


@pytest.fixture(scope='module')
def built_model(make_model, implant):
    """Return a built model, shared across benchmarks.

    Prediction does not mutate the model, so reuse is safe."""
    return make_model(implant).build()


@pytest.fixture(scope='module')
def percept(scenario, built_model, source):
    """Return a predicted percept."""
    return scenario.predict(built_model, source)


def _memray_peak(fn, *args, **kwargs):
    """Return the bytes live at the heap high-water mark during one call."""
    import memray
    with tempfile.TemporaryDirectory() as tmp:
        capture = Path(tmp) / 'capture.bin'
        with memray.Tracker(
                capture,
                file_format=memray.FileFormat.AGGREGATED_ALLOCATIONS):
            fn(*args, **kwargs)
        records = memray.FileReader(
            capture).get_high_watermark_allocation_records()
        return sum(record.size for record in records)


def _tracemalloc_peak(fn, *args, **kwargs):
    """Return the peak bytes tracemalloc traced during one call."""
    tracemalloc.start()
    try:
        fn(*args, **kwargs)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.fixture
def peak_memory():
    """Return a helper that measures the peak heap allocation of one call.

    The helper returns ``extra_info`` entries: ``peak_mem_mb`` (MB = 1e6
    bytes) and ``memory_metric``, which names the backend:

    - ``memray_heap_peak`` (Linux, macOS): bytes live at Memray's heap
      high-water mark. Counts native allocations (NumPy, Torch, Cython).
    - ``tracemalloc_peak`` (Windows, which Memray does not support): counts
      Python and NumPy allocations only, not Torch tensors or raw ``malloc``.

    Both count allocator calls made during the call, so a block reused from
    resident memory still counts. Call outside the timed section: tracking
    slows the call.
    """
    if sys.platform == 'win32':
        backend, metric = _tracemalloc_peak, 'tracemalloc_peak'
    else:
        backend, metric = _memray_peak, 'memray_heap_peak'

    def _measure(fn, *args, **kwargs):
        gc.collect()
        return {'peak_mem_mb': round(backend(fn, *args, **kwargs) / 1e6, 3),
                'memory_metric': metric}
    return _measure


@pytest.fixture(scope='session')
def cuda_device():
    """Return the CUDA device, or skip.

    Session-scoped, so the skip happens before module-scoped models build.
    """
    if not torch.cuda.is_available():
        pytest.skip('needs a CUDA device')
    return torch.device('cuda')


@pytest.fixture
def synchronized(cuda_device):
    """Return a wrapper that waits for a call's CUDA kernels to finish.

    Without it, timing covers only the asynchronous kernel launches.
    """
    def _wrap(fn):
        def _call(*args, **kwargs):
            result = fn(*args, **kwargs)
            torch.cuda.synchronize(cuda_device)
            return result
        return _call
    return _wrap


@pytest.fixture
def cuda_peak_memory(cuda_device, synchronized):
    """Return a helper that measures the CUDA memory one call allocates.

    The helper returns ``extra_info`` entries: ``peak_mem_mb``, the allocator's
    peak during the call minus its live allocation before it (MB = 1e6 bytes),
    and ``memory_metric='cuda_allocated_delta'``. Counts tensor memory only,
    not memory cached by the allocator or held by the CUDA context.
    """
    def _measure(fn, *args, **kwargs):
        gc.collect()
        torch.cuda.synchronize(cuda_device)
        baseline = torch.cuda.memory_allocated(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
        result = synchronized(fn)(*args, **kwargs)
        peak = torch.cuda.max_memory_allocated(cuda_device)
        del result
        return {'peak_mem_mb': round((peak - baseline) / 1e6, 3),
                'memory_metric': 'cuda_allocated_delta'}
    return _measure
