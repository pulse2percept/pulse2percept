.. _dev-benchmarks:

========================
pulse2percept benchmarks
========================

A small suite that measures the library's main job: predicting a percept from a
stimulus, an implant and a phosphene model. It tracks **execution time** and
**peak memory** for the reference pipelines in ``scenarios.py``, broken down by
pipeline stage.

A benchmark on its own asserts nothing -- it produces a number, and a number
means something only next to another number. ``compare.py`` supplies the other
number, and the ``Benchmarks`` workflow runs the pair on every pull request:
the base branch and the branch, measured on the same runner minutes apart, with
a regression past the thresholds failing the job. See `Comparing two runs`_.


Running
=======

Install the extra once:

.. code-block:: bash

    pip install -e ".[benchmark]"

Then, from the repository root:

.. code-block:: bash

    pytest benchmarks/ --benchmark-only

Without ``--benchmark-only`` every benchmark is skipped, so a bare ``pytest`` at
the repository root stays a test run and never pays for them. They also live
outside the ``pulse2percept`` package, so ``pytest --pyargs pulse2percept`` --
what CI and ``make tests`` run -- does not collect them at all. ``make bench``
wraps the whole thing.

Useful invocations:

.. code-block:: bash

    # one stage, or one scenario
    pytest benchmarks/ --benchmark-only -k predict_percept
    pytest benchmarks/ --benchmark-only -k argus2_axonmap_logobvl

    # include the scenarios that are too slow for the default run
    pytest benchmarks/ --benchmark-only --runslow

    # see how the models scale with threads (see the caveat below)
    pytest benchmarks/ --benchmark-only --n-threads=8

    # save a run, then compare a later one against it
    pytest benchmarks/ --benchmark-only --benchmark-save=baseline
    pytest benchmarks/ --benchmark-only --benchmark-compare=0001

Saved runs land in ``.benchmarks/`` as JSON, including the memory numbers.


Comparing two runs
==================

``pytest-benchmark``'s own ``--benchmark-compare`` reports on time only, and
knows nothing about the memory recorded in ``extra_info``. ``compare.py`` reads
two ``--benchmark-json`` files and reports on both:

.. code-block:: bash

    git checkout master
    pytest benchmarks/ --benchmark-only --benchmark-json=base.json
    git checkout my-branch
    pytest benchmarks/ --benchmark-only --benchmark-json=head.json
    python benchmarks/compare.py base.json head.json

It prints a Markdown table and exits non-zero if anything regressed. Time and
memory are held to different standards, because they are not equally
trustworthy:

**Memory is highly repeatable.** ``tracemalloc`` counts allocations rather than
sampling the process, so repeated runs of unchanged code report the same peak
to the byte.

**Time depends on runner load.** The minimum over many rounds may drift between
runs of unchanged code, so we set a generous threshold to catch 20% regressions
instead of every tiny performance change.

Both checks also require an absolute change, not just a ratio, since some
benchmarks are small enough (a 0.2 ms build, a 0.08 MB prediction) that a large
ratio is noise on a number nobody notices. Ratio-only breaches are still shown,
marked ``(under floor)``; they just do not fail the run. All four limits are
options -- see ``python benchmarks/compare.py --help``.

Because this is the code that decides whether a pull request passes, its
decision logic is tested in ``test_compare.py`` on synthetic data.


On a pull request
=================

``.github/workflows/benchmarks.yml`` runs the above automatically: it builds
and benchmarks the base commit, then the pull request, then compares. The table
lands in the run's job summary, and both JSON files are uploaded as artifacts.

It builds the package twice, and that is deliberate. Storing numbers from an
earlier run and comparing against them later is the obvious design and does not
work here: GitHub hands out whatever CPU it has, so a cross-runner comparison
carries a spread far larger than the regressions worth catching. Measuring both
sides in one job on one runner is what makes the timings comparable at all.

**When the job fails**, read which metric tripped. 
If the regression turns out to be real and you decide it is worth paying for
(e.g., because of a more accurate model), say so in the pull request and merge
over the failure. **Do not raise the thresholds to turn the check green.** 
Once the change is merged, the new cost *is* the baseline that every later
comparison runs against.

The job also fails if the two runs share **no** benchmarks at all.


What is measured
================

Every scenario is measured at each stage, rather than only end to end. That
decomposition is the whole point: an end-to-end number cannot tell you whether a
regression moved into stimulus construction, the model build, or the percept
computation.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Group
     - What it covers
   * - ``stimulus``
     - Building the stimulus, before any implant exists.
   * - ``implant``
     - Assigning the stimulus to the implant, including the downsampling of an
       image or video onto the electrode grid.
   * - ``build``
     - ``model.build()``, both warm (``test_build``, what a user hits on every
       run after the first) and cold (``test_build_cold``, ignoring the on-disk
       axon cache -- the actual computation, and the thing worth optimizing).
       Same group, so the two sit side by side in the report.
   * - ``predict_percept``
     - The headline number.
   * - ``end_to_end``
     - The whole one-liner.
   * - ``plot``
     - Drawing the percept. Mostly a matplotlib measurement, kept in its own
       group so it is never read as model cost.


Reading the numbers
===================

**Compare** ``min``, **not** ``mean``. Noise only ever makes a run slower, so
the minimum is the most stable estimate. Watch ``stddev`` as a signal of how
quiet the machine was, not as a property of the code.

**Threads are pinned to 1 by default.** The library defaults ``n_threads`` to
one per CPU, which makes results incomparable between machines, and between runs
on a loaded machine. Pinning gives a number that means something in isolation.
Use ``--n-threads`` deliberately, and never compare a run against a baseline
taken at a different thread count.

**Memory is measured separately from time.** ``tracemalloc`` inflates run time
several-fold, so a timing that included it would be measuring the profiler. Each
benchmark therefore runs its payload one extra time under ``tracemalloc`` and
records ``peak_mem_mb`` in ``extra_info``.

**Memory numbers are a floor, not a total.** ``tracemalloc`` tracks NumPy data
buffers, which is where essentially all of the memory in these workloads goes,
but it does not see raw ``malloc`` inside the Cython/OpenMP kernels. It was
chosen over RSS sampling because it is deterministic, needs no extra dependency,
and works on Windows -- which rules out ``pytest-memray``.

**Run on a quiet machine.** An absolute timing off a shared CI runner means
very little. The pull request check works around that by measuring both sides
on the same runner and gating only on large ratios, but any number you intend
to quote, or any regression you intend to act on, should come from a quiet
machine.


Adding a scenario
=================

Add a ``Scenario`` to ``scenarios.py``. The benchmark functions are
parametrized over that list, so a new entry is picked up by every stage
automatically and no other file changes. For example, a temporal model:

.. code-block:: python

    Scenario(
        id='argus2_axonmap_fading',
        stimulus=lambda: array_ptrain(p2p.implants.ArgusII),
        implant=lambda stim: p2p.implants.ArgusII(stim=stim),
        model=lambda **kwargs: p2p.models.Model(
            spatial=p2p.models.AxonMapSpatial(xrange=(-12, 12),
                                              yrange=(-8, 8)),
            temporal=p2p.models.FadingTemporal(), **kwargs),
    )

The criterion for a new entry is a **compiled kernel no existing scenario
reaches**. Every scenario costs run time in every pull request, and a model
that shares its kernel with one already listed buys none of the regression
coverage that cost is for. A kernel that *no* scenario reaches is the opposite
problem: it is a kernel this check cannot see regress at all.

Four things to know.

**Stimulate the whole array.** Handing a bare ``BiphasicPulseTrain`` to an
implant assigns it to one electrode, not all of them -- ``ArgusII(stim=...)``
then has a stimulus of shape ``(1, 29)`` rather than ``(60, 29)``. A benchmark
built that way exercises a sixtieth of the per-electrode work and barely moves
when that work regresses. Use the ``array_ptrain`` helper, as above.

**Match the stimulus to the model.** ``BiphasicAxonMapModel`` reads pulse
parameters off each electrode and rejects an image; a temporal model given a
single-frame stimulus measures nothing temporal.

**Sub-model parameters go on the sub-model instance**, as above. Keywords
handed to ``Model(...)`` itself reach *both* sub-models, and ``Parametrized``
freezes attributes, so anything the temporal model does not recognize raises
rather than being quietly ignored.

**Set the capability flags.** A temporal-only model's percept has no spatial
grid, and ``Percept.plot`` raises on one, so such a scenario needs
``plottable=False``. And if the scenario takes more than a few seconds per
``predict_percept`` call, mark it ``slow=True`` so it stays out of the default
run:

.. code-block:: python

    Scenario(
        id='argus2_axonmap_bostontrain',
        ...
        slow=True,
    )

Slow scenarios only run with ``--runslow``, mirroring the convention the test
suite already uses. Keeping the default run to roughly a minute is what makes it
something people actually run before opening a pull request.


Scope
=====

Deliberately not included: **no historical tracking**. Each pull request is
compared against its own base and nothing is stored, so the suite can say "this
branch is slower than master" but not "the library got slower over the last six
months". Answering the second question well means a per-commit series, and that
is the point to consider `asv <https://asv.readthedocs.io/>`_, which is built
for it. It was not chosen here because it builds an isolated environment per
commit, which is heavy for a Cython/OpenMP project and awkward on Windows.

The pull request check also posts no comment. Commenting needs a token with
write access, which the ``pull_request`` event does not give a fork, so a
comment step would fail on exactly the pull requests that most need review. The
report goes to the job summary instead, which every run has.
