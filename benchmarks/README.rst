.. _dev-benchmarks:

======================
Performance Benchmarks
======================

A small suite that measures percept prediction from a stimulus, an implant and
a phosphene model. It tracks **execution time** and **peak memory** (CPU heap
allocations, or CUDA allocations on a GPU) for the reference pipelines in
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
    pytest benchmarks/ --benchmark-only -k argus2_axonmap_fading_video

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

**Memory is more repeatable than time.** Each memory metric counts
allocations (see `Reading the numbers`_). Memory is compared only when both runs report the same
``memory_metric``; if the tags differ, or a run predates the tag, that
benchmark is compared on time only.

**Time depends on runner load.** The minimum over many rounds may drift between
runs of unchanged code, so the time threshold is a generous 2x and catches only
major regressions.

Both checks also require an absolute change in addition to the ratio, because
some benchmarks are tiny (a 0.2 ms build, a 0.08 MB prediction). Ratio-only
breaches are shown as ``(under floor)`` and do not fail the run. All four limits
are options; see ``python benchmarks/compare.py --help``.

The pass/fail logic is tested on synthetic data in ``test_compare.py``, and
the memory helpers in ``test_memory.py``.


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

Four workflows, reduced from the quickstart to benchmark size:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Scenario
     - Workflow
   * - ``imie_biphasic_image``
     - IMIE encodes an ``ImageStimulus`` with ``FrequencyEncoder`` (2x a
       uniform 80 uA threshold, 0-60 Hz) for ``BiphasicAxonMapModel``.
   * - ``argus2_axonmap_fading_video``
     - Argus II's default ``AmplitudeEncoder`` (6 Hz) and sequential raster
       encode a 2 s, 6 fps ``VideoStimulus`` for an ``AxonMapSpatial`` +
       ``FadingTemporal`` composite on the Torch core.
   * - ``prima_ho2018_scene_gaze``
     - The same video in a 40 dva ``Scene`` with a scotoma, viewed through
       PRIMA Pivotal's optical encoder by ``Ho2018Model``, with two saccades
       in ``Gaze``.
   * - ``orion_dynaphos_trace``
     - ``TraceEncoder`` maps a letter Z in dva onto Orion electrodes on V1
       (``Polimeni2006Map``) and stimulates them in sequence for
       ``DynaphosModel``.

Every scenario is measured at each stage, so a regression can be located in
stimulus construction, encoding, the model build, or the percept computation.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Group
     - What it covers
   * - ``stimulus``
     - Building the stimulus, before any implant exists.
   * - ``implant``
     - ``implant.prepare_stim``: the implant's encoder turns an image or video
       into stimulation. Skipped for scene and trace input, which need the
       model.
   * - ``build``
     - ``model.build()``, both warm (``test_build``, every run after the
       first) and cold (``test_build_cold``, ignoring the on-disk axon cache,
       i.e. the actual computation). Both are in one group so they appear side
       by side in the report.
   * - ``predict_percept``
     - The headline number: the scenario's ``predict``. Includes encoding the
       stimulus. Where the implant can encode independently, the
       corresponding ``implant.prepare_stim`` path is also benchmarked under
       ``implant``.
   * - ``predict_tensor_cuda``
     - The Torch core (``_predict_tensor``) with the waveform on a CUDA
       device, for scenarios that run on it (the Argus II video). Excludes
       stimulus preparation and the copy to the device. Skipped without CUDA,
       including on the pull request check.
   * - ``end_to_end``
     - The whole workflow, including implant and model construction.
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

**Memory is measured separately from time.** Tracking slows the call, so each
benchmark first runs its payload once, untimed, and records ``peak_mem_mb`` and
``memory_metric`` in ``extra_info``.

**Memory metrics.** ``memory_metric`` says what ``peak_mem_mb`` measured
(MB = 1e6 bytes). All three count allocations made during the call. They are
substantially more repeatable than process RSS and do not depend on whether an
allocation needs additional resident pages:

- ``memray_heap_peak`` (Linux, macOS): bytes live at the heap high-water mark,
  from `Memray <https://bloomberg.github.io/memray/>`_. Counts native
  allocations, including NumPy, Torch and the Cython kernels.
- ``tracemalloc_peak`` (Windows, which Memray does not support): Python and
  NumPy allocations only. Torch tensors and raw ``malloc`` are invisible, so
  Torch paths report far less than their actual peak.
- ``cuda_allocated_delta``: peak of Torch's CUDA allocator during the call
  minus its live allocation before it. Counts tensors only, not the allocator
  cache or the CUDA context.

Process RSS was not used: allocator reuse made later benchmarks report no
growth, so the result depended on test order.

**Run on a quiet machine.** Absolute timings from a shared CI runner are
unreliable. The pull request check measures both sides on the same runner and
gates only on large ratios, but numbers you quote or act on should come from a
quiet machine.


Adding a scenario
=================

Scenarios represent distinct, realistic pulse2percept workflows. Add one only
when it exercises a materially different user-facing pipeline not already
represented. Every scenario adds run time to every pull request.

Add a ``Scenario`` to ``scenarios.py``. The benchmark functions are
parametrized over that list, so every stage picks up the new entry with no
other file changes.

**Keep it benchmark-sized.** Shorten videos, coarsen model grids, and shorten
trajectories until a prediction takes well under a second, while keeping the
workflow the same. Set ``slow=True`` only if no such version exists; slow
scenarios run only with ``--runslow``.

**Use the implant's encoder.** Image and video input goes through
``implant.encoder``, as in user code. Match the video frame rate to the
encoder's pulse rate, or frames go unsampled.

**Override** ``predict`` when the workflow is not
``model.predict_percept(stimulus)``, e.g. a ``Scene`` with ``gaze`` or a
``TraceEncoder``. If ``implant.prepare_stim`` alone cannot turn the stimulus
into stimulation, set ``implant_encodes=False`` to skip the ``implant`` stage.

**Sub-model parameters go on the sub-model instance** (see ``axonmap_fading``).
Keywords passed to ``Model(...)`` reach *both* sub-models, and
``Parametrized`` freezes attributes, so a keyword the temporal model does not
recognize causes an error.


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
