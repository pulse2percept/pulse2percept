=============================
Contributing to pulse2percept
=============================

.. note::

    If you found a bug or want to request a feature, please open an issue in our
    `Issue Tracker`_ on GitHub. Make sure to
    :ref:`label your issue appropriately <dev-contributing-issue-labels>`.

We are excited that you are here and want to contribute!
If you have any questions that aren't discussed below, please let us know
by opening an issue in our `Issue Tracker`_ on GitHub.

.. _Issue Tracker: https://github.com/pulse2percept/pulse2percept/issues

Already know what you're looking for in this guide? Jump to the following
sections:

*   `Recommended workflow`_
*   `Understanding issue labels`_
*   `Contributing code`_
*   `Testing your code`_

.. _dev-contributing-workflow:

Recommended workflow
====================

We appreciate all contributions, but those accepted fastest will follow a
workflow similar to the following:

1.  **GitHub account**:
    Before getting started, you will need to set up a free `GitHub account`_.

2.  **Claim an issue on GitHub**:
    Check the `Issue Tracker`_ to find an issue you want to work on (look for
    issues labeled `good-first-issue`_ or `help-wanted`_), and add a comment
    announcing your intention to work on it.
    If no such issue exists, create a new one (see
    `Understanding issue labels`_).
    This allows other members of the development team to confirm that you
    aren't overlapping with existing work and that everyone is on the same page
    with the goal of your proposed work.

3.  **Fork the repo**:
    Follow the :ref:`Installation Guide <install-source>` to fork the repo and
    install all developer dependencies. Make sure to keep your code up-to-date
    with the :ref:`upstream repository <install-upgrade>`.
    This is now your own unique pulse2percept copy - changes here won't affect
    anyone else's work.

4.  **Create a new branch**:
    You should always work on a `new branch`_. "git add" and "git commit" the
    work you proposed to do, and "git push" it to your remote repository on
    GitHub.

    .. important::

        All code additions must be :ref:`documented <dev-contributing-doc>` and
        :ref:`tested <dev-contributing-test>`.
        See `Contributing code`_ below for more detailed instructions.

5.  **Submit a pull request**:
    When opening a `pull request`_ (PR) against the pulse2percept master branch,
    please use one of the following prefixes:

    * **[ENH]** for enhancements
    * **[FIX]** for bug fixes
    * **[TST]** for new or updated tests
    * **[DOC]** for new or updated documentation
    * **[STY]** for stylistic changes
    * **[REF]** for refactoring existing code

    Once your PR is ready, request a review from `@arokem`_ and/or
    `@mbeyeler`_, who will review your changes before merging them into the
    main codebase.

    .. note:: 
 
        If your PR is not yet ready to be merged, click on the dropdown arrow next to
        the "Create pull request" button and choose "Create draft pull request" instead.

        This will put your PR in `draft state`_ and block merging until you change the status
        of the PR to "Ready for review".

.. _GitHub account: https://help.github.com/articles/signing-up-for-a-new-github-account
.. _good-first-issue: https://github.com/pulse2percept/pulse2percept/labels/good-first-issue
.. _help-wanted: https://github.com/pulse2percept/pulse2percept/labels/help-wanted
.. _new branch: https://help.github.com/articles/about-branches
.. _pull request: https://help.github.com/articles/creating-a-pull-request-from-a-fork/
.. _@arokem: https://github.com/arokem
.. _@mbeyeler: https://github.com/mbeyeler
.. _draft state: https://github.blog/2019-02-14-introducing-draft-pull-requests

.. _dev-contributing-issue-labels:

Understanding issue labels
==========================

Make sure to check out the current list of `issue labels`_:

*   .. image:: https://img.shields.io/badge/-bug-fc2929.svg
       :target: https://github.com/pulse2percept/pulse2percept/labels/bug
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

*   .. image:: https://img.shields.io/badge/-enhancement-00FF09.svg
       :target: https://github.com/pulse2percept/pulse2percept/labels/enhancement
       :alt: Enhancement
       :align: left

    *These issues are asking for new features to be added to the project.*

    Please try to make sure that your requested enhancement is distinct from
    any others that have already been requested or implemented.

    .. note::

        If you find one that's similar but there are subtle differences, please
        reference the other request in your issue.

*   .. image:: https://img.shields.io/badge/-help%20wanted-c2e0c6.svg
       :target: https://github.com/pulse2percept/pulse2percept/labels/help-wanted
       :alt: Help wanted
       :align: left

    *These issues contain a task that a member of the team has determined we
    need additional help with.*

    If you feel that you can contribute to one of these issues, we especially
    encourage you to do so!

    .. note::

        Issues that are also labelled as `good first issue`_ are a great place
        to start if you're looking to make your first contribution.

.. _issue labels: https://github.com/pulse2percept/pulse2percept/labels
.. _good first issue: https://github.com/pulse2percept/pulse2percept/issues?q=is%3Aopen+is%3Aissue+label%3Agood-first-issue

Contributing code
=================

Perform all your work on a `new branch`_ of the repository. For example,
say you want to add "feature1" to the latest version of pulse2percept:

1.  Make sure you have the latest code:

    .. code-block:: bash

        git checkout master
        git pull upstream master

    .. note::

        If you get an error saying "upstream does not appear to be a git
        repository", you need to run the following command first:
        ``git remote add upstream https://github.com/pulse2percept/pulse2percept.git``

2.  Create a new branch (aptly named "feature1" or similar):

    .. code-block:: bash

        git checkout -b feature1

3.  Add and commit your changes to this branch:
    
    .. code-block:: bash

        git add newfile.py
        git commit -m "add new feature1 file"
    
4.  Then push it to your remote repository on GitHub:

    .. code-block:: bash

        git push origin feature1

    .. important::

        All code additions must be :ref:`documented <dev-contributing-doc>` and
        :ref:`tested <dev-contributing-test>`.

5.  Go to GitHub and `submit a pull request`_:

    1.  Click on "compare across forks" at the top of the page.

    2.  Choose "pulse2percept/pulse2percept" as the base repository and "master"
        as the base branch.

    3.  Choose "<username>/pulse2percept" as the head repository and "feature1"
        as the compare branch, where "<username>" is your GitHub user name.

    4.  Click on "Create pull request" (or "Create draft pull request" if your work
        is not ready to be merged) and describe the work you have done.
        Make sure to mention the issue number you are addressing (use # as
        prefix).

        An easy way to list all the things you changed is to use a list of
        checkboxes (type ``- [X]``; or ``- [ ]`` for an item that has yet to be
        implemented).

.. _submit a pull request: https://github.com/pulse2percept/pulse2percept/compare

.. _dev-contributing-doc:

Documenting your code
=====================

You are expected to document your code using `NumPy docstrings`_.
Make sure to:

*  supply short and long descriptions,
*  describe all input arguments to a function/method,
*  describe the output of a function/method,
*  provide examples of how to use your code.

For example, consider an appropriate docstring for a hypothetical function
``rad2deg``:

.. code-block:: python

    def rad2deg(angle_rad):
        """Converts radians to degrees

        This function converts an angle in radians to degrees.

        Parameters
        ----------
        angle_rad : int, float
            The input angle in radians in (between 0 and 2pi)

        Returns
        -------
        angle_deg : float
            The corresponding angle in degrees (between 0 and 360 deg)

        Examples
        --------
        Converting pi to degrees:
        >>> import numpy as np
        >>> rad2deg(np.pi)
        180.0

        .. seealso:: `deg2rad`
        """
        ...

You can generate the documentation yourself using Sphinx.
If you installed ``make``, type the following from your root directory:

.. code-block:: bash

    make doc

Otherwise, type the following from your root directory:

.. code-block:: bash

    cd doc
    pip3 install -r requirements.txt
    make html

The generated documentation can then be found in ``doc/_build/html``.
To see the documentation, "doc/_build/html/index.html" in your browser of
choice, e.g.:

.. code-block:: bash

    google-chrome doc/_build/html/index.html

.. _NumPy docstrings: https://numpydoc.readthedocs.io/en/latest/format.html 

.. _dev-contributing-changes:

Documenting API changes
=======================

API changes that affect the user should be documented in order to help the user
sort out version differences (see `reST directives`_):

*  Whenever a new API call is added, include a ``.. versionadded::`` statement
   right before listing the function parameters that mentions the pulse2percept
   version where the feature first appeared.
*  Whenever the API of a function/class is changed, include a
   ``.. versionchanged::`` statement right before listing the function 
   parameters that explains what/how functionality changed in a particular
   pulse2percept version.

.. _reST directives: https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html


.. _dev-contributing-test:

Testing your code
=================

You are expected to test your code using `pytest`_:

*   Bug fixes should include an example that exposes the issue.

*   New features should have tests that show at least a minimal example.

Running the test suite
----------------------

pulse2percept uses `pytest`_ and `numpy-testing`_ for testing.

Every subpackage of pulse2percept (e.g., :py:mod:`~pulse2percept.stimuli`)
has a subdirectory called "tests".
Within the test directory, there is a "test_<subsubpackage>.py" file for every
subsubpackage of pulse2percept (e.g.,
"pulse2percept/stimuli/tests/test_pulse_trains.py" for the
:py:mod:`~pulse2percept.stimuli.pulse_trains` module).

When you contribute new code, you are expected to test your code in the
corresponding test file.

You can run the test suite from your root directory with:

.. code-block:: bash

    pip3 install -r requirements-dev.txt
    pytest --doctest-modules --showlocals -v pulse2percept

Successful tasks will be marked with "PASSED", unsuccessful ones with "FAILED".
We will usually not accept pull requests that don't pass all tests.

.. note::

    Whenever you submit a pull request, the test suite is automatically run in the
    background using `GitHub Actions`_. This will make sure that all tests pass on
    all supported platforms whenever changes are made to the code.

.. _pytest: https://pytest.org
.. _numpy-testing: https://docs.scipy.org/doc/numpy/reference/routines.testing.html
.. _GitHub Actions: https://github.com/pulse2percept/pulse2percept/actions

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

Thank you
=========

You are awesome!

*This guide is based on contributing guidelines from the `Nipype`_ project.*

.. _Nipype: https://github.com/nipy/nipype

