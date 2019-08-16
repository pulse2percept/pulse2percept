=============================
Contributing to pulse2percept
=============================

We're excited you're here and want to contribute.

These guidelines are designed to make it as easy as possible to get involved.
If you have any questions that aren't discussed below, please let us know by
opening an `issue on GitHub`_!

Before you start you'll need to set up a free `GitHub`_ account and sign in.
Here are some `instructions`_.

Already know what you're looking for in this guide? Jump to the following
sections:

*   `Understanding issue labels`_
*   `Making a change`_
*   `Testing your code`_
*   `Recognizing contributions`_

.. _issue on GitHub: https://github.com/uwescience/pulse2percept/issues
.. _GitHub: https://github.com/
.. _instructions: https://help.github.com/articles/signing-up-for-a-new-github-account

Understanding issue labels
==========================

Make sure to check out the current list of `issue labels`_:

*   .. image:: https://img.shields.io/badge/-bugs-fc2929.svg
       :target: https://github.com/uwescience/pulse2percept/labels/bug
       :alt: Bug
       :align: left

    *These issues point to problems in the project.*

    If you find new a bug, please provide as much information as possible to
    recreate the error.
    The issue template will automatically populate any new issue you open, and
    contains information we've found to be helpful in addressing bug reports.
    Please fill it out to the best of your ability!

    .. note::

        If you experience the same bug as one already listed in an open issue,
        please add any additional information that you have as a comment.

*   .. image:: https://img.shields.io/badge/-help%20wanted-c2e0c6.svg
       :target: https://github.com/uwescience/pulse2percept/labels/help-wanted
       :alt: Help wanted
       :align: left

    *These issues contain a task that a member of the team has determined we
    need additional help with.*

    If you feel that you can contribute to one of these issues, we especially
    encourage you to do so!

    .. note::

        Issues that are also labelled as `good first issue`_ are a great place
        to start if you're looking to make your first contribution.

*   .. image:: https://img.shields.io/badge/-enhancement-00FF09.svg
       :target: https://github.com/uwescience/pulse2percept/labels/enhancement
       :alt: Enhancement
       :align: left

    *These issues are asking for new features to be added to the project.*

    Please try to make sure that your requested enhancement is distinct from
    any others that have already been requested or implemented.

    .. note::

        If you find one that's similar but there are subtle differences, please
        reference the other request in your issue.

.. _issue labels: https://github.com/uwescience/pulse2percept/labels
.. _good first issue: https://github.com/uwescience/pulse2percept/issues?q=is%3Aopen+is%3Aissue+label%3Agood-first-issue

Making a change
===============

We appreciate all contributions to pulse2percept, but those accepted fastest
will follow a workflow similar to the following:

1. Comment on an existing issue or open a new issue referencing your addition
-----------------------------------------------------------------------------

This allows other members of the pulse2percept development team to confirm that
you aren't overlapping with work that's currently underway and that everyone is
on the same page with the goal of the work you're going to carry out.

`This blog`_ is a nice explanation of why putting this work in up front is so
useful to everyone involved.

.. _This blog: https://www.igvita.com/2011/12/19/dont-push-your-pull-requests/

2. Fork the pulse2percept repository to your profile
-----------------------------------------------------

This is now your own unique `fork`_ (or "copy") of the
`pulse2percept repository`_.
Changes here won't affect anyone else's work, so it's a safe space to explore
edits to the code!

You can `clone`_ your pulse2percept repository in order to create a local copy
of  the code on your machine:

.. code-block:: bash

    git clone https://github.com/<username>/pulse2percept.git

Make sure to replace ``<username>`` with your GitHub user name
(**not** "uwescience").

.. note::

    A "fork" is basically a "remote copy" of a GitHub repository. For example,
    forking pulse2percept creates
    "https://github.com/<username>/pulse2percept.git" from
    "https://github.com/uwescience/pulse2percept.git".

    A "clone" is basically a "local copy" of your GitHub repository. For
    example, cloning pulse2percept creates a local "pulse2percept" directory
    (including all the git machinery and history) from
    "https://github.com/<username>/pulse2percept.git".

Then open a terminal, navigate to your local pulse2percept clone, and install
all dependencies required for development:

.. code-block:: bash

    pip3 install -r requirements-dev.txt

Now you are ready to build pulse2percept:

    pip3 install -e .

Make sure to keep your fork up to date with the original pulse2percept
repository. One way to do this is to `configure a new remote`_ named "upstream"
and `sync your fork`_ with the upstream repository:

.. code-block:: bash

    git remote add upstream https://github.com/uwescience/pulse2percept.git
    git pull upstream master

where "master" is the branch you want to sync.

.. _fork: https://help.github.com/articles/fork-a-repo/
.. _pulse2percept repository: https://github.com/uwescience/pulse2percept
.. _clone: https://help.github.com/articles/cloning-a-repository
.. _configure a new remote: https://help.github.com/articles/configuring-a-remote-for-a-fork
.. _sync your fork: https://help.github.com/articles/syncing-a-fork/

3. Make the changes you've discussed
------------------------------------

General guidelines:

*   Perform all your work on a `new branch`_ of the repository. For example,
    say you want to add "feature1" to the latest version of pulse2percept.
    First make sure you have the latest code:

    .. code-block:: bash

        git checkout master
        git pull upstream master

    Then create a new branch (aptly named "feature1"):

    .. code-block:: bash

        git checkout -b feature1

    Add and commit your changes to this branch. Then push it to your remote
    repository on GitHub:

    .. code-block:: bash

        git push origin feature1

    Now you can go to GitHub and submit a pull request (see
    `4. Submit a pull request`_).

*   New code should be tested, whenever feasible (see `Testing your code`_).

    *   Bug fixes should include an example that exposes the issue.

    *   Any new features should have tests that show at least a minimal
        example.

.. note::

    If you're not sure what this means for your code, please ask in your pull
    request.

.. _new branch: https://help.github.com/articles/about-branches

4. Submit a pull request
------------------------

A new `pull request`_ (PR) for your changes should be created from your fork of
the repository.

.. important::

    Pull requests should be submitted early and often (please don't mix too
    many unrelated changes within one PR)!

When opening a pull request, please use one of the following prefixes:

* **ENH:** for enhancements
* **FIX:** for bug fixes
* **TST:** for new or updated tests
* **DOC:** for new or updated documentation
* **STY:** for stylistic changes
* **REF:** for refactoring existing code

.. note::

    If your pull request is not yet ready to be merged, please also include the
    **WIP** prefix (you can remove the prefix once your PR is ready to be
    merged).

    This tells the development team that your pull request is a
    "work-in-progress", and that you plan to continue working on it.

Once your PR is ready, a member of the development team will review your
changes  to confirm that they can be merged into the main codebase.

.. note::

    Review and discussion on new code can begin well before the work is
    complete, and the more discussion the better!

    The development team may prefer a different path than you've outlined, so
    it's better to discuss it and get approval at the early stage of your work.

.. _pull request: https://help.github.com/articles/creating-a-pull-request-from-a-fork/

Testing your code
=================

pulse2percept uses `pytest`_ and `numpy-testing`_ for testing.

Every subpackage of pulse2percept (e.g., :py:mod:`~pulse2percept.stimuli`)
has a subdirectory called "tests".
Within the test directory, there is a "test_<subsubpackage>.py" file for every
subsubpackage of pulse2percept (e.g.,
"pulse2percept/stimuli/tests/test_pulse_trains.py" for
:py:mod:`~pulse2percept.stimuli.pulse_trains`).
When you contribute new code, you are expected to test your code in the
corresponding test file.

You can run the test suite with:

.. code-block:: bash

    pytest --doctest-modules --showlocals -v pulse2percept

Successful tasks will be marked with "PASSED", unsuccessful ones with "FAILED".
We will usually not accept pull requests that don't pass all tests.

.. _pytest: https://pytest.org
.. _numpy-testing: https://docs.scipy.org/doc/numpy/reference/routines.testing.html

Writing your own tests
----------------------

If you work on code from an existing subpackage (e.g.,
:py:mod:`pulse2percept.stimuli.pulse_trains`), open the corresponding test file
(e.g., "pulse2percept/stimuli/tests/test_pulse_trains.py").

You can add a new test by adding a function whose name starts with "test\_",
followed by the name of the class or function you want to test.
For example:

*   ``def test_TimeSeries`` for testing the
    :py:class:`~pulse2percept.stimuli.TimeSeries` object (note that this
    function already exists).
*   ``def test_TimeSeries_resample`` for testing the
    :py:meth:`~pulse2percept.stimuli.TimeSeries.resample` method of the
    :py:class:`~pulse2percept.stimuli.TimeSeries` object.
*   ``def test_newfunc`` for a new function called ``newfunc``.

Within this function, you want to make sure your code works as expected.
Useful `numpy-testing`_ routines for achieving this include:

*   ``assert_equal(actual, desired)`` returns an ``AssertionError`` if two
    objects are not equal.
*   ``assert_almost_equal(actual, desired, decimal=7)`` returns an
    ``AssertionError`` if two items are not equal up to desired precision
    (good for testing doubles).
*   ``assert_raises(exception_class)`` fails unless an ``Exception`` of class
    ``exception_class`` is thrown.

In addition, we provide
:py:meth:`~pulse2percept.utils.testing.assert_warns_msg` to ensure that a
specific warning message is thrown.

.. seealso:: :ref:`Tutorial: Writing your own test case <tutorial-writing-tests>`

.. important::

    When you are working on your changes, test frequently to ensure you are not
    breaking the existing code.

Recognizing contributions
=========================

We welcome and recognize all contributions from documentation to testing to
code development.

Thank you!
==========

You are awesome.

*This guide is based on contributing guidelines from the `Nipype`_ project.*

.. _Nipype: https://github.com/nipy/nipype
