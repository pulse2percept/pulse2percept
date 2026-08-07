# pulse2percept benchmarks

A small suite that measures the library's main job: predicting a percept from a
stimulus, an implant and a phosphene model. It tracks **execution time** and
**peak memory** for two reference pipelines, broken down by pipeline stage.

These are benchmarks, not tests. They assert nothing and cannot fail on a
regression; they produce numbers you compare against numbers from another run.

## Running

Install the extra once:

```bash
pip install -e ".[benchmark]"
```

Then, from the repository root:

```bash
pytest benchmarks/ --benchmark-only
```

Without `--benchmark-only` every benchmark is skipped, so a bare `pytest` at the
repository root stays a test run and never pays for them. They are also outside
the `pulse2percept` package, so `pytest --pyargs pulse2percept` -- what CI and
`make tests` run -- does not collect them at all.

Useful invocations:

```bash
# one stage, or one scenario
pytest benchmarks/ --benchmark-only -k predict_percept
pytest benchmarks/ --benchmark-only -k argus2_axonmap

# see how the models scale with threads (see the caveat below)
pytest benchmarks/ --benchmark-only --n-threads=8

# save a run, then compare a later one against it
pytest benchmarks/ --benchmark-only --benchmark-save=baseline
pytest benchmarks/ --benchmark-only --benchmark-compare=0001
```

Saved runs land in `.benchmarks/` as JSON, including the memory numbers.

## What is measured

Each of the two scenarios in `scenarios.py` is measured at every stage, rather
than only end to end. That decomposition is the whole point: an end-to-end
number cannot tell you whether a regression moved into stimulus construction,
the model build, or the percept computation.

| group | what it covers |
| --- | --- |
| `stimulus` | building the stimulus, before any implant exists |
| `implant` | assigning the stimulus to the implant, including the downsampling of an image onto the electrode grid |
| `build` | `model.build()`, both warm (`test_build`, what a user hits on every run after the first) and cold (`test_build_cold`, ignoring the on-disk axon cache -- the actual computation, and the thing worth optimizing). Same group so the two sit side by side in the report |
| `predict_percept` | the headline number |
| `end_to_end` | the whole one-liner |
| `plot` | drawing the percept; mostly a matplotlib measurement, kept separate so it is never read as model cost |

## Reading the numbers

**Compare `min`, not `mean`.** Noise only ever makes a run slower, so the
minimum is the most stable estimate. Watch `stddev` as a signal of how quiet the
machine was, not as a property of the code.

**Threads are pinned to 1 by default.** The library defaults `n_threads` to one
per CPU, which makes results incomparable between machines and between runs on a
loaded machine. Pinning gives a number that means something in isolation. Use
`--n-threads` deliberately, and never compare a run to a baseline taken at a
different thread count.

**Memory is measured separately from time.** `tracemalloc` inflates run time
several-fold, so a timing that included it would be measuring the profiler. Each
benchmark therefore runs its payload one extra time under `tracemalloc` and
records `peak_mem_mb` in `extra_info`.

**Memory numbers are a floor, not a total.** `tracemalloc` tracks NumPy data
buffers, which is where essentially all of the memory in these workloads goes,
but it does not see raw `malloc` inside the Cython/OpenMP kernels. It was chosen
over RSS sampling because it is deterministic, needs no extra dependency, and
works on Windows -- which rules out `pytest-memray`.

**Run on a quiet machine.** Shared CI runners are too noisy for these numbers to
mean much, which is why nothing here gates a pull request.

## Adding a scenario

Add a `Scenario` to `scenarios.py`. The benchmark functions are parametrized
over that list, so a new entry is picked up by every stage automatically and no
other file changes. For example, a temporal model:

```python
Scenario(
    id='argus2_axonmap_fading',
    stimulus=lambda: p2p.stimuli.BostonTrain(),
    implant=lambda stim: p2p.implants.ArgusII(stim=stim),
    model=lambda **kwargs: p2p.models.Model(
        spatial=p2p.models.AxonMapSpatial,
        temporal=p2p.models.FadingTemporal, **kwargs),
)
```

Two things to know before adding a video scenario. Predicting a percept for
`BostonTrain()` takes on the order of a minute per call, which is far too slow
for a suite that should stay runnable in about a minute total -- give it a
`@pytest.mark.slow` and an opt-in flag rather than putting it in the default
run. And `Model(...)` routes unknown keywords to its sub-models, so check that
whatever you pass is accepted: `Parametrized` freezes attributes, so a keyword
the model does not know raises `FreezeError` instead of being ignored.

## Scope

Deliberately not included: no CI job, no regression gate, no historical
tracking. If per-commit history over time becomes the goal, that is the point to
consider [asv](https://asv.readthedocs.io/), which is built for it. It was not
chosen here because it builds an isolated environment per commit, which is heavy
for a Cython/OpenMP project and awkward on Windows.
