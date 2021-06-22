"""`assert_warns_msg`"""

import pytest
import numpy.testing as npt
import warnings


def assert_warns_msg(expected_warning, func, msg, *args, **kwargs):
    """Assert a call leads to a warning with a specific message

    Test whether a function call leads to a warning of type
    ``expected_warning`` with a message that contains the string ``msg``.

    Parameters
    ----------
    expected_warning : warning class
        The class of warning to be checked; e.g., DeprecationWarning
    func : object
        The class, method, property, or function to be called as\
        func(\*args, \*\*kwargs)
    msg : str
        The message or a substring of the message to test for.
    \*args : positional arguments to ``func``
    \*\*kwargs: keyword arguments to ``func``

    """
    # Make sure we are not leaking warnings:
    warnings.resetwarnings()
    # Run the function that is supposed to produce a warning:
    with pytest.warns(expected_warning) as record:
        func(*args, **kwargs)
    # Check the number of generated warnings:
    if len(record) != 1:
        print('Generated warnings:')
        for r in record:
            print('-', r)
    npt.assert_equal(len(record), 1)
    # Check the message:
    if msg is not None:
        npt.assert_equal(msg in ''.join(record[0].message.args[0]), True)
    # Remove generated warnings from registry:
    warnings.resetwarnings()
