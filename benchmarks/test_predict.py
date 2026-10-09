"""Benchmarks for the core pipeline: stimulus -> implant -> model -> percept.

Every scenario in :mod:`scenarios` is measured at each stage separately, and
once end to end, so a regression can be located in stimulus construction, the
model build, or the percept computation.

Each benchmark reports wall-clock time through the ``benchmark`` fixture and
peak memory through ``benchmark.extra_info``, both saved in the same JSON.
Memory is measured in a separate, untimed call before timing (see the
``peak_memory`` fixture).
"""
import matplotlib.pyplot as plt
import pytest
import torch

from pulse2percept.models.base import (_delivered, _encoder_clock,
                                       _tensor_waveform)


@pytest.mark.benchmark(group='stimulus')
def test_stimulus(benchmark, scenario, peak_memory):
    """Construct the stimulus, before any implant is involved."""
    benchmark.extra_info.update(peak_memory(scenario.stimulus))
    stim = benchmark(scenario.stimulus)
    benchmark.extra_info['stim_shape'] = str(stim.shape)


@pytest.mark.benchmark(group='implant')
def test_implant(benchmark, scenario, implant, peak_memory):
    """Encode the stimulus into the stimulation the device delivers.

    Uses the implant's own encoder. Stimulus construction happens in
    ``setup`` and is not timed.
    """
    if not scenario.implant_encodes:
        pytest.skip(f'{scenario.id} needs the model to turn its stimulus '
                    f'into stimulation; see the predict_percept group')

    def setup():
        return (scenario.stimulus(),), {}

    benchmark.extra_info.update(peak_memory(implant.prepare_stim,
                                            scenario.stimulus()))
    benchmark.pedantic(implant.prepare_stim, setup=setup, rounds=20,
                       iterations=1, warmup_rounds=1)
    benchmark.extra_info['n_electrodes'] = implant.n_electrodes


@pytest.mark.benchmark(group='build')
def test_build(benchmark, scenario, implant, make_model, peak_memory):
    """Build the model with any on-disk cache already warm.

    This is every run after the first. For axon-map models it is dominated by
    reading the pickled bundles and recomputing axon sensitivity; see
    ``test_build_cold`` for the full computation.
    """
    if scenario.caches_axons:
        # Populate the cache first, independent of test order:
        make_model(implant, ignore_pickle=False).build()

    def setup():
        return (make_model(implant, ignore_pickle=False),), {}

    benchmark.extra_info.update(peak_memory(
        lambda: make_model(implant, ignore_pickle=False).build()))
    benchmark.pedantic(lambda model: model.build(), setup=setup, rounds=5,
                       iterations=1, warmup_rounds=1)


@pytest.mark.benchmark(group='build')
def test_build_cold(benchmark, scenario, implant, make_model, peak_memory):
    """Build the model from scratch, ignoring the on-disk cache.

    Measures the Jansonius axon-map computation instead of unpickling.
    """
    if not scenario.caches_axons:
        pytest.skip(f'{scenario.id} has no on-disk cache, so a cold build is '
                    f'the same as a warm one (see test_build)')

    def setup():
        return (make_model(implant, ignore_pickle=True),), {}

    benchmark.extra_info.update(peak_memory(
        lambda: make_model(implant, ignore_pickle=True).build()))
    benchmark.pedantic(lambda model: model.build(), setup=setup, rounds=5,
                       iterations=1, warmup_rounds=1)


@pytest.mark.benchmark(group='predict_percept')
def test_predict_percept(benchmark, scenario, built_model, source,
                         peak_memory):
    """Predict the percept (headline number).

    Includes encoding the stimulus. Where the implant can encode
    independently, the corresponding ``implant.prepare_stim`` path is also
    benchmarked in the ``implant`` group.
    """
    benchmark.extra_info.update(peak_memory(scenario.predict, built_model,
                                            source))
    percept = benchmark(scenario.predict, built_model, source)
    benchmark.extra_info['percept_shape'] = str(percept.shape)


@pytest.mark.benchmark(group='predict_tensor_cuda')
def test_predict_tensor_cuda(benchmark, scenario, built_model, source,
                             cuda_device, synchronized, cuda_peak_memory):
    """Predict on the Torch core with the waveform on a CUDA device.

    Times ``_predict_tensor`` only: stimulus preparation and the copy to the
    device happen beforehand.
    """
    # Scene and TraceEncoder input need the model, not just the implant:
    if (not scenario.implant_encodes or
            not getattr(built_model, '_has_tensor_core', False)):
        pytest.skip(f'{scenario.id} does not run on the Torch core')
    stim = built_model._prepared(source)
    if not built_model._uses_tensor_core(stim):
        pytest.skip(f'{scenario.id} does not run on the Torch core')
    # As in `Model._predict_tensor_core`:
    delivered = _delivered(stim)
    if not delivered.is_compressed:
        delivered.compress()
    waveform, time = _tensor_waveform(built_model.spatial, delivered)
    waveform = waveform.to(cuda_device)
    clock = _encoder_clock(stim)

    def predict():
        with torch.inference_mode():
            return built_model._predict_tensor(waveform, time,
                                               frame_clock=clock)

    benchmark.extra_info.update(cuda_peak_memory(predict))
    resp = benchmark(synchronized(predict))
    benchmark.extra_info['resp_shape'] = str(tuple(resp.data.shape))


@pytest.mark.benchmark(group='end_to_end')
def test_end_to_end(benchmark, scenario, make_model, peak_memory):
    """Run the whole workflow: implant, stimulus, model build, prediction.

    The model is built through ``make_model``, which writes the axon cache to
    a temporary directory.
    """
    def run():
        implant = scenario.implant()
        return scenario.predict(make_model(implant), scenario.stimulus())

    benchmark.extra_info.update(peak_memory(run))
    percept = benchmark(run)
    benchmark.extra_info['percept_shape'] = str(percept.shape)


@pytest.mark.benchmark(group='plot')
def test_plot(benchmark, scenario, percept, peak_memory):
    """Draw the percept.

    Mostly matplotlib time, kept in its own group. The axes are cleared in
    ``setup`` and reused, to avoid hundreds of figures and to keep teardown out
    of the timed section.
    """
    fig, ax = plt.subplots()
    try:
        def setup():
            ax.clear()
            return (), {'ax': ax}

        benchmark.extra_info.update(peak_memory(percept.plot, ax=ax))
        benchmark.pedantic(percept.plot, setup=setup, rounds=20, iterations=1,
                           warmup_rounds=1)
    finally:
        plt.close(fig)
