# -*- coding: utf-8 -*-
"""
.. _tutorial-writing-tests:

==========================
Writing your own test case
==========================

In this tutorial, you will learn how to write a test case for your contributed
code, and make sure the test passes.

Outline
=======

* `Prerequisites`_
* `Scenario`_
* `Implementation`_

.. note::

    This tutorial is intended for people interested in contributing code
    :ref:`contributing code <dev-contributing>` to pulse2percept.

Prerequisites
=============

This tutorial assumes the following:

* pulse2percept is already installed on your machine.
  If you are having trouble with installation, see
  :ref:`Installation <install>`.

* You have read :ref:`Contributing to pulse2percept <dev-contributing>`
  and have followed the steps to fork and clone a development version of
  pulse2percept to your local computer.

Scenario
========

The goal of this tutorial is to add a new function to the
:py:mod:`~pulse2percept.stimuli.pulse_trains` subpackage that returns the
largest element in a :py:class:`~pulse2percept.stimuli.TimeSeries` object.

The function, which we will call ``largest_timeseries_element``, should:

* accept a :py:class:`~pulse2percept.stimuli.TimeSeries` object, or throw a
  ``TypeError`` if a different object is passed
* throw a warning if the largest value is 0
* return the largest value in the
  :py:class:`~pulse2percept.stimuli.TimeSeries`' ``data`` container

.. note::

    If this were a real problem, there would already be an issue for it in
    pulse2percept's `issue tracker`_.

    In that case, you would open the specific issue on GitHub and assign
    yourself to it. You can also post a comment to let the community know that
    you are going to submit a pull request on this issue.

.. _issue tracker: https://github.com/uwescience/pulse2percept/issues

Implementation
==============

In this tutorial, we will follow what is known as `test-driven development`_
(TDD). TDD turns the common work flow (write code, then test it interactively)
on its head with the goal of producing better code faster. Instead, we will
follow the following recipe:

* Write the test function, ``test_largest_timeseries_element`` **first**.
* Then write a ``largest_timeseries_element`` function that should pass those
  tests.
* If the function produces any wrong answers, fix it and re-run the test
  function.

.. note::

    In Python, function and variable names are generally lowercase, with words
    separated by underscores. Class names, on the other hand, typically use
    CapWords convention.

    See `PEP8`_ for a full Python style guide.

.. _test-driven development: https://www.freecodecamp.org/news/learning-to-test-with-python-997ace2d8abe
.. _PEP8: https://www.python.org/dev/peps/pep-0008

Creating a new branch
---------------------

Following the general guidelines outlined in
:ref:`Contributing to pulse2percept <dev-contributing>`, we
need to perform all our work on a new branch.

First, we need to make sure we are working off the latest code:

.. code-block: bash

    git checkout master
    git pull upstream master

Then we will create a new branch (aptly named "largest-timeseries-element"
or similar):

.. code-block: bash

    git checkout -b largest-timeseries-element

Writing the test function
-------------------------

Because our code is related to :py:class:`~pulse2percept.stimuli.TimeSeries`,
which lives in the :py:mod:`pulse2percept.stimuli.pulse_trains` subpackage, our
new function should go in the same subpackage.

The corresponding test file is
"pulse2percept/stimuli/tests/test_pulse_trains.py".

In this file, we will create a new test function.
For consistency, it is important that our function be named
"test\_<name of function to test>", where "<name of function to test>" is
identical to the function added to the
:py:mod:`~pulse2percept.stimuli.pulse_trains` subpackage.
For example:

* ``def test_TimeSeries`` for testing the
  :py:class:`~pulse2percept.stimuli.TimeSeries` object (note that this function
  already exists).
* ``def test_TimeSeries_resample`` for testing the
  :py:meth:`~pulse2percept.stimuli.TimeSeries.resample` method of the
  :py:class:`~pulse2percept.stimuli.TimeSeries` object.
* ``def test_newfunc`` for a new function called ``newfunc``.

Our test function should therefore be called
``test_largest_timeseries_element``.

.. important::

    For `pytest`_ to run your test function, its name must start with "test\_".

Within our function, we have access to a number of `numpy-testing`_ routines
that can compare desired to actual output, such as:

* ``assert_equal(actual, desired)`` returns an ``AssertionError`` if two
  objects are not equal.
* ``assert_almost_equal(actual, desired, decimal=7)`` returns an
  ``AssertionError`` if two items are not equal up to desired precision
  (good for testing doubles).
* ``assert_raises(exception_class)`` fails unless an ``Exception`` of class
  ``exception_class`` is thrown.

Typically, we want to make sure the function works for a few simple cases.
Our first draft for a test function might thus look like this:

.. code-block:: python

    import numpy as np
    import numpy.testing as npt
    from pulse2percept.stimuli import (largest_timeseries_element, TimeSeries,
                                       PulseTrain)


    def test_largest_timeseries_element():
        # Create a simple TimeSeries object:
        ts = TimeSeries(1, np.array([0, 1.5, 2]))
        # Use almost_equal because we are comparing doubles:
        npt.assert_almost_equal(largest_timeseries_element(ts), 2.0)


We can now run the entire test suite from the pulse2percept root directory:

.. code-block:: bash

    make tests

Alternatively, we can run a single test file by specifying its path:

.. code-block:: bash

    pytest pulse2percept/stimuli/tests/test_pulse_trains.py

Even better yet, we can run just a single test from a single file:

.. code-block:: bash

    pytest pulse2percept/stimuli/tests/test_pulse_train.py::test_largest_timeseries_element

What we expect to see is an ``ImportError``, because we have not actually
written the ``largest_timeseries_element`` function yet! So let's get going.

.. _pytest: https://pytest.org
.. _numpy-testing: https://docs.scipy.org/doc/numpy/reference/routines.testing.html

Writing the actual function
---------------------------

The next step is to add the actual function to
"pulse2percept/stimuli/pulse_trains.py":

.. code-block:: python

    import numpy as np


    def largest_timeseries_element(ts):
        \"\"\"Return the largest element of a TimeSeries object

        Parameters
        ----------
        ts:
            TimeSeries
            A TimeSeries object

        Returns
        -------
        max:
            double
            The largest value in the TimeSeries data
        \"\"\"
        return np.max(ts.data)

Note how we make use of `docstring`_ notation here to document the functions
purpose, input arguments, and return values.

Now we can run the test suite again, and find... still an ``ImportError``.
What's going on here?

.. _docstring: https: // numpydoc.readthedocs.io / en / latest / format.html

Adding the function to the subpackage __init__
----------------------------------------------

To make sure the function gets imported, we have to edit the subpackage's
"__init__.py" file. It might look something like this:

.. code-block:: python

    \"\"\"Stimuli

    This module provides a number of stimuli.
    \"\"\"

    from .base import Stimulus
    from .pulse_trains import (TimeSeries, MonophasicPulse, BiphasicPulse,
                               PulseTrain)

    __all__ = [
        'BiphasicPulse',
        'MonophasicPulse',
        'PulseTrain',
        'Stimulus',
        'TimeSeries'
    ]

.. note::

    One of the purposes of this file is to enumerate all the functions and
    objects that should be imported in this subpackage.

    The ``__all__`` variable lists all functions and objects to be imported
    when somebody types ``from pulse2percept import *``.

To make sure our new function gets imported, we need to modify the file as
follows:

.. code-block:: python

    \"\"\"Stimuli

    This module provides a number of stimuli.
    \"\"\"

    from .base import Stimulus
    from .pulse_trains import (TimeSeries, MonophasicPulse, BiphasicPulse,
                               PulseTrain, largest_timeseries_element)

    __all__ = [
        'BiphasicPulse',
        'largest_timeseries_element'
        'MonophasicPulse',
        'PulseTrain',
        'Stimulus',
        'TimeSeries'
    ]

.. note::

    In agreement with `PEP8`_, our function name is lowercase and uses
    underscores to separate words, whereas the other variables listed in this
    file are all class names (and thus should use CapWords convention).

Now we are able to run the test suite without ``ImportError``!

.. note::

    You might have to run ``make`` first to install the code changes.
    ``make tests`` does that automatically.

Updating the test function
--------------------------

Now the test passes, but have not yet implemented all the functionality
outlined under `Scenario`_. Specifically, we need to throw a ``TypeError`` if
the input argument is not of type
:py:class:`~pulse2percept.stimuli.TimeSeries`, and throw a warning if the
largest value is 0.

We should therefore update ``test_largest_timeseries_element`` as follows:

.. code-block:: python

    import pytest
    import numpy as np
    import numpy.testing as npt
    from pulse2percept.stimuli import (largest_timeseries_element, TimeSeries,
                                       PulseTrain)
    from pulse2percept.utils.testing import assert_warns_msg

    def test_largest_timeseries_element():
        # Create a simple TimeSeries object:
        ts = TimeSeries(1, np.array([0, 1.5, 2]))
        # Use almost_equal because we are comparing doubles:
        npt.assert_almost_equal(largest_timeseries_element(ts), 2.0)

        # Make sure an error is thrown here:
        with pytest.raises(TypeError):
            largest_timeseries_element(3.0)
        with pytest.raises(TypeError):
            largest_timeseries_element([0, 1.5, 2])

        # Make sure a warning is thrown here:
        ts = TimeSeries(1, np.array([0, 0, 0]))
        assert_warns_msg(UserWarning, largest_timeseries_element, [ts],
                         "is zero")

The :py:func:`~pulse2percept.utils.assert_warns_msg` takes as input the
`warning category`_ to expect (``UserWarning``), the function to test
(``largest_timeseries_element``), a list of objects to pass (``[ts]``),
and a warning message (or substring thereof) to expect (``"is zero"``).

Other things to add:

* You might want to check whether your function works on different NumPy
  arrays: ``np.array([1])``? 2-D? 3-D? etc.
* You might want to check whether the warning is raised when the maximum
  values is *really close but not exactly* 0.
* You might want to check if your function works for subclasses of
  :py:class:`~pulse2percept.stimuli.TimeSeries`, such as
  :py:class:`~pulse2percept.stimuli.MonophasicPulse` and
  :py:class:`~pulse2percept.stimuli.PulseTrain`.

.. _warning category: https://docs.python.org/3/library/warnings.html

Updating the actual function
----------------------------

To make this test pass, we have to make a few changes to our function.

For one, we can check a variable's type with ``isinstance``.

For another, we can produce a warning with the ``logging`` package.

.. code-block:: python

    import logging
    import numpy as np

    def largest_timeseries_element(ts):
        \"\"\"Return the largest element of a TimeSeries object

        Parameters
        ----------
        ts:
            TimeSeries
            A TimeSeries object

        Returns
        -------
        max:
            double
            The largest value in the TimeSeries data
        \"\"\"
        if not isinstance(ts, TimeSeries):
            raise TypeError("Input argument 'ts' is not a TimeSeries object.")
        max_val = np.max(ts.data)
        if np.isclose(max_val, 0):
            # Use `isclose` because we are comparing doubles:
            logging.getLogger(__name__).warn("Max val is zero.")
        return max_val

A few things to observe here:

* We raise ``TypeError``, which is the same type of error that we test
  against in our test function. We could have also chosen a different
  `exception class`_.

* We don't need to import :py:class:`~pulse2percept.stimuli.TimeSeries` here,
  because it is defined in the same file.

* We use `isinstance`_ to check if the input argument is an instance or
  subclass of :py:class:`~pulse2percept.stimuli.TimeSeries`. Note that a
  :py:class:`~pulse2percept.stimuli.PulseTrain` would also return True,
  because it is a subclass of :py:class:`~pulse2percept.stimuli.TimeSeries`.

* We use `np.isclose`_ to check whether two values are equal within a
  tolerance. We could also pass an absolute/relative tolerance level.

* We use ``logging.getLogger(__name__)`` to access the `logger`_ that was
  already created by pulse2percept for this very file ("__name__").

.. note::

    You would import :py:class:`~pulse2percept.stimuli.Stimulus` by writing
    ``from .base import Stimulus``, because it lives in the same directory
    (".") in the file "base.py".
    Similarly, you could import :py:class:`~pulse2percept.implants.ArgusII` via
    ``from ..implants import ArgusII``.

.. _exception class: https://docs.python.org/3/library/exceptions.html
.. _isinstance: https://www.programiz.com/python-programming/methods/built-in/isinstance
.. _np.isclose: https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.isclose.html
.. _logger: https://docs.python.org/3/library/logging.html

Verifying the result
--------------------

The final test is to make sure that ``make tests`` marks all tests with
"PASSED". It is important to run all tests, because sometimes our code change
(e.g., a bug fix) inadvertently breaks other pieces of the software.
Running the full test suite makes sure this doesn't happen.

Once all tests pass, you are ready to submit your pull request
(see :ref:`Contributing to pulse2percept <dev-contributing>`).

Good luck!

"""
