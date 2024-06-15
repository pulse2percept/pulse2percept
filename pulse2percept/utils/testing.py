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



def generate_standard_benchmark(ModelClass, grid=True, elecs=True, time=True, **kwargs):
    """Generates a standard benchmark for a model class
    
    Note that this does not do any checks for correctness, but simply
    runs the model with different parameters and records the time it takes.

    Parameters
    ----------
    ModelClass : p2p.model.Model
        The model class to be tested.
    grid : bool, optional
        Whether to parametrize the test with different pixel grid sizes
    elecs : bool, optional
        Whether to parametrize the test with different electrode grid sizes
    time : bool, optional
        Whether to parametrize the test with different numbers of time points.
    **kwargs : dict
        Additional keyword arguments to be passed to the model class.
    
    Returns
    -------
    benchmark : function
        A function that runs the benchmark test, returning
    """

    def benchmark(benchmark, grid=None, elecs=None, time=None):
        """
        
        """
        if grid is None or int(grid) == 100:
            gridspec = ((-4, 5), (-4, 5), 1)
        elif int(grid) == 500:
            gridspec = ((-6, 6), (-4.5, 5), 0.5)
        elif int(grid) == 1000:
            gridspec = ((-3, 3), (-4.75, 5), 0.25)
        elif int(grid) == 5000:
            gridspec = ((-2.9, 3), (-4.9, 5), 0.1)
        elif int(grid) == 10000:
            gridspec = ((-4.9, 5), (-4.9, 5), 0.1)

        model_kwargs = {
            'xrange': gridspec[0],
            'yrange': gridspec[1],
            'xystep': gridspec[2],
        }
        # replace defaults with user-provided values
        model_kwargs = {**model_kwargs, **kwargs} 
        model = ModelClass(**model_kwargs)
        model.build()

        

        
    
