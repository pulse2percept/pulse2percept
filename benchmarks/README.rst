.. _dev-benchmarks:

======================
Performance Benchmarks
======================

A small suite that measures percept prediction from a stimulus, an implant and
a phosphene model. It tracks **execution time** and **peak memory** (resident
CPU memory, or CUDA allocations on a GPU) for the reference pipelines in
``scenarios.py``, broken down by pipeline stage.

``compare.py`` compares two runs. The ``Benchmarks`` workflow runs the base
branch and the pull request on the same runner minutes apart, and fails the job
on a regression past the thresholds. See `Comparing two runs`_.


Running
=======

Install the extra once:

.. code-block:: bash

    pip install -e ".[benchmark]"

Then, from the repository root:

.. code-block:: bash

    pytest benchmarks/ --benchmark-only

Without ``--benchmark-only`` every benchmark is skipped, so a bare ``pytest`` at
the repository root does not run them. They live outside the ``pulse2percept``
package, so ``pytest --pyargs pulse2percept`` (used by CI and ``make tests``)
does not collect them. ``make bench`` runs the suite.

Useful invocations:

.. code-block:: bash

    # one stage, or one scenario
    pytest benchmarks/ --benchmark-only -k predict_percept
    pytest benchmarks/ --benchmark-only -k argus2_axonmap_logobvl

    # include the scenarios that are too slow for the default run
    pytest benchmarks/ --benchmark-only --runslow

    # save a run, then compare a later one against it
    pytest benchmarks/ --benchmark-only --benchmark-save=baseline
    pytest benchmarks/ --benchmark-only --benchmark-compare=0001

Saved runs are stored in ``.benchmarks/`` as JSON, including the memory numbers.


Comparing two runs
==================

``pytest-benchmark``'s ``--benchmark-compare`` reports time only and ignores
the memory recorded in ``extra_info``. ``compare.py`` reads two
``--benchmark-json`` files and reports both:

.. code-block:: bash

    git checkout master
    pytest benchmarks/ --benchmark-only --benchmark-json=base.json
    git checkout my-branch
    pytest benchmarks/ --benchmark-only --benchmark-json=head.json
    python benchmarks/compare.py base.json head.json

It prints a Markdown table and exits non-zero if anything regressed. Time and
memory use different thresholds:

**Memory is compared only between like metrics.** Each benchmark tags
``peak_mem_mb`` with ``memory_metric`` (see `Reading the numbers`_). If the two
runs differ in tag, or a run predates the tag, that benchmark is compared on
time only.

**Time depends on runner load.** The minimum over many rounds may drift between
runs of unchanged code, so the time threshold is a generous 2x and catches only
major regressions.

Both checks also require an absolute change in addition to the ratio, because
some benchmarks are tiny (a 0.2 ms build, a 0.08 MB prediction). Ratio-only
breaches are shown as ``(under floor)`` and do not fail the run. All four limits
are options; see ``python benchmarks/compare.py --help``.

The pass/fail logic is tested on synthetic data in ``test_compare.py``.


On a pull request
=================

``.github/workflows/benchmarks.yml`` builds and benchmarks the base commit, then
the pull request, then compares. The table goes to the run's job summary, and
both JSON files are uploaded as artifacts.

The package is built twice on purpose. GitHub runners vary in CPU, so numbers
stored from an earlier run are not comparable; only both sides measured in one
job on one runner are.

**When the job fails**, check which metric tripped. If the regression is real
and acceptable (e.g., a more accurate model), say so in the pull request and
merge over the failure. **Do not increase the thresholds to make the check
pass.** Once merged, the new cost is the baseline for later comparisons.

The job also fails if the two runs share **no** benchmarks.


What is measured
================

Every scenario is measured at each stage, so a regression can be located in
stimulus construction, the model build, or the percept computation.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Group
     - What it covers
   * - ``stimulus``
     - Building the stimulus, before any implant exists.
   * - ``implant``
     - Source to device-ready stimulation: the downsampling of an image or
       video onto the electrode grid, then ``implant.prepare_stim``.
   * - ``build``
     - ``model.build()``, both warm (``test_build``, every run after the
       first) and cold (``test_build_cold``, ignoring the on-disk axon cache,
       i.e. the actual computation). Both are in one group so they appear side
       by side in the report.
   * - ``predict_percept``
     - The headline number. Includes the preparation also timed separately
       under ``implant``.
   * - ``predict_tensor_cuda``
     - The Torch core (``_predict_tensor``) with the waveform on a CUDA
       device, for scenarios that run on it. Excludes stimulus preparation and
       the copy to the device. Skipped without CUDA, including on the pull
       request check.
   * - ``end_to_end``
     - The whole one-liner.
   * - ``plot``
     - Drawing the percept. Mostly matplotlib time, kept in its own group so
       it is not read as model cost.


Reading the numbers
===================

**Compare** ``min``, **not** ``mean``. Noise only makes a run slower, so the
minimum is the most stable estimate. ``stddev`` indicates machine load, not a
property of the code.

**Pin threads for comparable timings.** Torch defaults to one thread per
core, which makes results incomparable between machines and between runs on a
loaded machine. Set ``OMP_NUM_THREADS=1``, as the pull request check does, and
never compare a run against a baseline taken at a different thread count.

**Memory is measured before time.** Each benchmark runs its payload once,
untimed, and records ``peak_mem_mb`` and ``memory_metric`` in ``extra_info``.
Timing runs afterwards, so memory the allocator retained from earlier calls
does not hide the working set.

**Two memory metrics.** ``memory_metric`` says what ``peak_mem_mb`` measured
(MB = 1e6 bytes):

- ``rss_delta``: peak process RSS during the call minus RSS before it, sampled
  about every 1 ms with ``psutil``. Covers NumPy, Torch, and Cython
  allocations alike. Allocations shorter than a sampling interval can be
  missed, and the number is noisier than an allocation count.
- ``cuda_allocated_delta``: peak of Torch's CUDA allocator during the call
  minus its live allocation before it. Counts tensors only, not the allocator
  cache or the CUDA context, and is repeatable.

**Run on a quiet machine.** Absolute timings from a shared CI runner are
unreliable. The pull request check measures both sides on the same runner and
gates only on large ratios, but numbers you quote or act on should come from a
quiet machine.


Adding a scenario
=================

Add a ``Scenario`` to ``scenarios.py``. The benchmark functions are
parametrized over that list, so every stage picks up the new entry with no
other file changes. For example, a temporal model:

.. code-block:: python

    Scenario(
        id='argus2_axonmap_fading',
        stimulus=lambda: array_ptrain(p2p.implants.retina.ArgusII),
        implant=p2p.implants.retina.ArgusII,
        model=lambda implant, **kwargs: p2p.models.Model(
            spatial=p2p.models.retina.AxonMapSpatial(implant, xrange=(-12, 12),
                                              yrange=(-8, 8)),
            temporal=p2p.models.FadingTemporal(), **kwargs),
    )

Add a scenario only if it reaches a **compiled kernel no existing scenario
reaches**. Every scenario adds run time to every pull request, and a model that
shares its kernel with an existing scenario adds no regression coverage.

**Stimulate the whole array.** A bare ``BiphasicPulseTrain`` passed to an
implant drives one electrode: ``ArgusII().prepare_stim(...)`` then has shape
``(1, 29)`` instead of ``(60, 29)``, so the benchmark covers a sixtieth of the
per-electrode work. Use the ``array_ptrain`` helper, as above.

**Match the stimulus to the model.** ``BiphasicAxonMapModel`` reads pulse
parameters from each electrode, rejects an image, and takes amplitude as a
multiple of threshold (``array_ptrain(..., amp=20 * p2p.units.xTh)``), not a
current. A temporal model given a single-frame stimulus measures nothing
temporal.

**An image is not a stimulus.** Gray levels are dimensionless, and both
``prepare_stim`` and ``predict_percept`` reject them; user code converts an
image to current with a ``PulseEncoder``. That would benchmark a pulse train
per electrode instead of a single static frame, so the image scenarios use the
``as_current`` helper, which samples the image onto the electrodes as current
explicitly. See its docstring.

**Sub-model parameters go on the sub-model instance**, as above. Keywords
passed to ``Model(...)`` reach *both* sub-models, and ``Parametrized`` freezes
attributes, so a keyword the temporal model does not recognize causes an error.

**Set the capability flags.** A temporal-only model takes no ``implant``, so
the scenario needs ``binds_implant=False``. Its percept has no spatial grid and
``Percept.plot`` fails on it, so it also needs ``plottable=False``. If the
scenario takes more than a few seconds per ``predict_percept`` call, set
``slow=True`` to exclude it from the default run:

.. code-block:: python

    Scenario(
        id='my_slow_scenario',
        ...
        slow=True,
    )

Slow scenarios run only with ``--runslow``, as in the test suite. The default
run takes about a minute so that it is practical before opening a pull request.


Scope
=====

**No historical tracking.** Each pull request is compared against its own base
and nothing is stored, so the suite detects "this branch is slower than master"
but not "the library got slower over six months". That requires a per-commit
series, e.g. with `asv <https://asv.readthedocs.io/>`_. asv was not used because
it builds an isolated environment per commit, which is heavy for a
Cython project and awkward on Windows.

The pull request check posts no comment. Commenting requires a token with write
access, which the ``pull_request`` event does not give to forks. The report goes
to the job summary instead.
