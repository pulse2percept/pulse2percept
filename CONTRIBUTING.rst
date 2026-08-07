=============================
Contributing to pulse2percept
=============================

Thank you for contributing to pulse2percept.

If you found a bug or want to request a feature, please open an issue in the
`Issue Tracker`_. Small, self-contained fixes may be submitted directly as a
pull request, but please discuss substantial changes with the maintainers before
investing significant time in an implementation.

.. warning::

   **Every contribution must have an accountable human contributor.**

   AI-assisted tools may be used, but fully autonomous submissions are not
   accepted. The submitting human must review and understand the changes, take
   responsibility for them, and participate in the review process. See
   :ref:`dev-contributing-ai` for the complete policy.

This guide covers:

* `Recommended workflow`_
* `Human responsibility and AI-assisted tools`_
* `Setting up a development environment`_
* `Submitting a pull request`_
* `Code style`_
* `Documenting your code`_
* `Documenting API changes`_
* `Testing your code`_

.. _Issue Tracker: https://github.com/pulse2percept/pulse2percept/issues


.. _dev-contributing-workflow:

Recommended workflow
====================

1. **Choose or open an issue.**

   Check the `Issue Tracker`_ for existing work. For nontrivial bug fixes,
   enhancements, or API changes, comment on the issue before starting so that
   the scope can be agreed upon and duplicate work can be avoided.

2. **Fork and install the repository.**

   Follow the :ref:`Installation Guide <install-source>` and install the
   development dependencies as described below.

3. **Create a focused branch.**

   Use a short, descriptive branch name and keep unrelated changes out of the
   same pull request.

4. **Implement, document, and test the change.**

   Bug fixes should include a regression test. New features should include
   tests and user-facing documentation where appropriate.

5. **Open a pull request.**

   Explain what changed, why it changed, and how it was tested. Link the
   relevant issue and disclose substantial use of AI-assisted tools as
   described below.

6. **Participate in review.**

   Respond to questions, revise the contribution when needed, and keep the
   branch current enough for the test suite to run against the proposed
   change.


.. _dev-contributing-ai:

Human responsibility and AI-assisted tools
==========================================

AI-assisted tools may be used to help write code, tests, documentation, issue
reports, or pull-request text. However, every contribution must have a clearly
identified human contributor who takes responsibility for the submitted work.

The human contributor must:

* review and understand every submitted change;
* verify the behavior and tests to a reasonable extent before submission;
* describe the change and its limitations accurately;
* be able to explain and revise the contribution during review; and
* disclose substantial use of AI-assisted tools in the pull-request
  description.

Substantial use includes tools that generated or materially rewrote code,
tests, or documentation, as well as agents that selected an issue, implemented
a solution, reviewed their own output, or opened the pull request. Routine
autocomplete and spelling or grammar assistance do not need to be disclosed.

Fully autonomous submissions are not accepted. Pull requests submitted by an
agent or bot account without an accountable human contributor may be closed
even when the proposed change appears useful. Automated analysis or self-review
by the same system does not count as independent review.

This policy does not apply to project automation explicitly installed or
configured by the maintainers, such as dependency-update bots.


.. _dev-contributing-setup:

Setting up a development environment
====================================

pulse2percept requires Python 3.11 or newer.

After forking the repository, clone your fork and add the main repository as
``upstream``:

.. code-block:: bash

    git clone https://github.com/<username>/pulse2percept.git
    cd pulse2percept
    git remote add upstream https://github.com/pulse2percept/pulse2percept.git

Create and activate a virtual environment using your preferred environment
manager. Then install pulse2percept in editable mode with the development
dependencies:

.. code-block:: bash

    python -m pip install --upgrade pip
    python -m pip install -e ".[dev]"

Before starting new work, update your local ``master`` branch:

.. code-block:: bash

    git fetch upstream
    git switch master
    git merge --ff-only upstream/master

Create a branch for the change:

.. code-block:: bash

    git switch -c fix-short-description

If ``git switch`` is unavailable in your Git version, the equivalent command
is:

.. code-block:: bash

    git checkout -b fix-short-description


.. _dev-contributing-pr:

Submitting a pull request
=========================

Commit the change with a concise, descriptive message and push the branch to
your fork:

.. code-block:: bash

    git add <changed-files>
    git commit -m "Fix short description of the issue"
    git push -u origin fix-short-description

Open a pull request against the ``master`` branch of
``pulse2percept/pulse2percept``.

Use one of the following prefixes in the pull-request title:

* ``[ENH]`` for enhancements
* ``[FIX]`` for bug fixes
* ``[TST]`` for new or updated tests
* ``[DOC]`` for new or updated documentation
* ``[STY]`` for stylistic changes
* ``[REF]`` for refactoring existing code

A good pull request:

* addresses one coherent issue;
* links the relevant issue, for example ``Fixes #123``;
* explains the problem, the proposed solution, and important tradeoffs;
* lists the tests that were run;
* includes tests and documentation required by the change;
* avoids unrelated formatting or refactoring; and
* passes the automated checks, or clearly identifies any failure believed to
  be unrelated.

Do not assume that a failing check is unrelated to the pull request.
Investigate the failure and document what you found. The maintainers will
decide whether a failure can safely be treated as independent of the proposed
change.

Open a draft pull request when the implementation is incomplete or when early
feedback would be useful. Mark it ready for review only when you believe the
change is complete.


Code style
==========

Follow the style of the surrounding code and keep changes easy to review.
Prefer clear, maintainable code over clever or compressed implementations.

The continuous-integration workflow runs ``flake8``. To run the corresponding
check locally:

.. code-block:: bash

    flake8 pulse2percept --ignore N802,N806,W504 --select W503 \
        --count --show-source --statistics

Avoid reformatting code that is unrelated to the contribution.


.. _dev-contributing-doc:

Documenting your code
=====================

Public functions, methods, classes, and modules should use `NumPy docstrings`_.

Document:

* the purpose and behavior of the object;
* parameters and their accepted types or shapes;
* return values;
* relevant exceptions, warnings, units, and side effects; and
* examples or notes when they materially help users understand the API.

Documentation should describe the actual behavior of the code, including edge
cases introduced or fixed by the contribution.

To build the documentation locally:

.. code-block:: bash

    python -m pip install -r doc/requirements.txt
    make doc

Alternatively, invoke Sphinx directly:

.. code-block:: bash

    python -m sphinx -b html doc doc/_build/html

The generated documentation is available at
``doc/_build/html/index.html``.

.. _NumPy docstrings:
   https://numpydoc.readthedocs.io/en/latest/format.html


.. _dev-contributing-changes:

Documenting API changes
=======================

Discuss substantial changes to the public API with the maintainers before
implementation. Preserve backward compatibility unless an incompatible change
has been explicitly agreed upon.

User-facing API changes should be annotated in the relevant docstring:

* Use ``.. versionadded::`` when adding a new public API.
* Use ``.. versionchanged::`` when changing documented behavior, accepted
  inputs, return values, shapes, units, or other user-visible semantics.
* Use ``.. deprecated::`` when deprecating a public API.

Include the pulse2percept version in which the change will appear. Ask a
maintainer which version to use if it is not yet clear.

.. _reST directives:
   https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html


.. _dev-contributing-test:

Testing your code
=================

Tests are written with `pytest`_ and `NumPy testing`_ utilities.

* Bug fixes must include a regression test that fails without the fix and
  passes with it.
* New features must include tests for their core behavior and important edge
  cases.
* Tests should be deterministic, focused, and independent of local machine
  state whenever possible.
* Place tests in the relevant ``tests`` directory and follow the naming and
  organization of nearby tests.

Run a focused test while developing:

.. code-block:: bash

    pytest path/to/test_file.py -q

Run the full default test suite from the repository root:

.. code-block:: bash

    pytest --pyargs pulse2percept --doctest-modules

Pull-request checks also run tests marked as slow. Run them locally when the
change affects those code paths:

.. code-block:: bash

    pytest --pyargs pulse2percept --doctest-modules --runslow

GitHub Actions runs the test suite on the supported Python versions and
operating systems. A pull request is normally merged only after the required
checks pass.


.. _dev-contributing-bench:

Benchmarking your code
======================

The ``benchmarks`` directory holds a small suite that measures execution time
and peak memory for the library's main job: predicting a percept from a
stimulus, an implant and a phosphene model. Run it when a change touches a
hot path:

.. code-block:: bash

    pip install -e ".[benchmark]"
    pytest benchmarks/ --benchmark-only

Benchmarks are skipped unless ``--benchmark-only`` is given, and they live
outside the ``pulse2percept`` package, so a normal test run never pays for
them. Nothing gates a pull request on them: shared CI runners are too noisy
for the numbers to mean much, so a run on a quiet local machine is the source
of truth.

To check a change for a regression, save a baseline before it and compare
after:

.. code-block:: bash

    pytest benchmarks/ --benchmark-only --benchmark-save=baseline
    pytest benchmarks/ --benchmark-only --benchmark-compare=0001

See ``benchmarks/README.md`` for what each number measures, how to read it,
and how to add a scenario.

.. _pytest: https://docs.pytest.org/
.. _NumPy testing:
   https://numpy.org/doc/stable/reference/routines.testing.html
.. _GitHub Actions:
   https://github.com/pulse2percept/pulse2percept/actions


Thank you
=========

Thank you for helping improve pulse2percept.
