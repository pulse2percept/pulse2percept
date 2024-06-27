"""`assert_warns_msg`"""

import pytest
import numpy.testing as npt
import warnings
import numpy as np


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



def standard_model_benchmark(model, grid=None, elecs=None, times=None, 
                               implant=None, **kwargs):
    """
    Generates a benchmark function that can be used to test the performance
    of a model for a standard benchmark test.

    Example usage:
    ```
    def test_standard_benchmark(benchmark):
        model = AxonMapModel()
        percept = benchmark(standard_benchmark_factory(model)())
    ```

    It is useful when combined with get_runspec() to run a standard benchmark
    test with multiple parameters.

    Parameters
    ----------
    model : object
        The model to be tested.
    grids : int or None
        The grid size. If None, the default grid size of 100 pixels will be used.
        Options: 100, 500, 1000, 5000, 10000.
    elecs : int or None
        The number of electrodes. If None, 20 electrodes will be used.
        Options: 1, 20, 100, 225, 1000, 5000.
    times : int, None or 'biphasic'
        The time. If None, 1 time point will be used. If 'biphasic', a biphasic
        pulse train will be used. Any other integer value can be used.
    implant : object or None
        The implant to be used. If None, a RectangleImplant will be used.

    Returns
    -------
    function
        A benchmark function just containing predict_percept for the specified setup.
    """
    from ..models import BiphasicAxonMapModel, BiphasicAxonMapSpatial
    from ..models.cortex import DynaphosModel
    from ..implants import RectangleImplant
    from ..stimuli import BiphasicPulseTrain
    if grid is None:
        grid = 100
    if isinstance(grid, tuple):
        if len(grid) == 2:
            raise ValueError('grids must be a single integer or a tuple of 3 integers')
        gridspec = grid
    elif int(grid) == 100:
        gridspec = ((-4, 5), (-4, 5), 1)
    elif int(grid) == 500:
        gridspec = ((-6, 6), (-4.5, 5), 0.5)
    elif int(grid) == 1000:
        gridspec = ((-6, 6), (-9.5, 10), 0.5)
    elif int(grid) == 5000:
        gridspec = ((-4.8, 5), (-9.8, 10), 0.2)
    elif int(grid) == 10000:
        gridspec = ((-4.9, 5), (-4.9, 5), 0.1)

    model_kwargs = {
        'xrange': gridspec[0],
        'yrange': gridspec[1],
        'xystep': gridspec[2],
    } if hasattr(model, 'xrange') else {}
    # replace defaults with user-provided values
    model_kwargs = {**model_kwargs, **kwargs} 
    model.build(**model_kwargs)
    if not isinstance(grid, tuple) and hasattr(model, 'grid'):
        npt.assert_equal(model.grid.shape[0] * model.grid.shape[1], int(grid))

    if elecs is None:
        elecs = 100
    if int(elecs) == 20:
        # (shape1, shape2, spacing)
        shapespec = (4, 5, 400)
    elif int(elecs) == 1:
        shapespec = (1, 1, 400)
    elif int(elecs) == 100:
        shapespec = (10, 10, 250)
    elif int(elecs) == 225:
        shapespec = (15, 15, 200)
    elif int(elecs) == 1000:
        shapespec = (25, 40, 50)
    elif int(elecs) == 5000:
        shapespec = (50, 100, 20)

    if implant is None:
        implant = RectangleImplant(shape=shapespec[0:2], spacing=shapespec[2])
    
    npt.assert_equal(len(implant.electrodes), int(elecs))
    if times is None:
        times = 1
        biphasic_classes = [BiphasicAxonMapModel, BiphasicAxonMapSpatial, DynaphosModel]
        if np.any([isinstance(model, b) for b in biphasic_classes]):
            times = 'biphasic'
    if times == 'biphasic':
        implant.stim = {e : BiphasicPulseTrain(20., np.random.rand(1).item(), .45) for e in implant.electrode_names}
    else:
        implant.stim = np.random.rand(len(implant.electrodes), int(times)).astype(np.float32)
    def bench_fn():
        return model.predict_percept(implant)
    # print(model.grid.shape, len(implant.electrodes), implant.stim.shape)
    return bench_fn
        
    
def get_bench_runspec(grids=True, elecs=True, times=True, biphasic=False):
    """
    Generates a list of tuples that can be used in combination with
    pytest.mark.parameterize to run a standard benchmark test.

    Example usage:
    ```
    @pytest.mark.parametrize('grid, elecs, time', get_runspec())
    def test_standard_benchmark(benchmark, grid, elecs, time):
        model = AxonMapModel()
        percept = benchmark(standard_benchmark_factory(model, grid, elecs, time)())
    ```

    Parameters
    ----------
    grids : bool or list
        If True, the grid size will be varied. If a list is provided, the
        grid sizes will be taken from the list. They must be one of the valid
        sizes for the standard benchmark: 100, 500, 1000, 5000, 10000.
    elecs : bool
        If True, the number of electrodes will be varied. If a list is provided,
        the number of electrodes will be taken from the list. They must be one
        of the valid sizes for the standard benchmark: 1, 20, 100, 225, 1000, 5000.
    times : bool
        If True, the time will be varied. If a list is provided, the time will
        be taken from the list. Any time can be used.

    Returns
    -------
    list
        A list of tuples that can be used in combination with
        pytest.mark.parameterize to run a standard benchmark test.
    
    """
    if biphasic == True:
        times = 'biphasic'
    default_times = 'biphasic' if times == 'biphasic' else 1
    default_elecs = 20
    default_grids = 100
    runspecs = []
    if grids == True:
        for g in [100, 500, 1000, 5000, 10000]:
            runspecs.append((g, default_elecs, default_times))
    elif isinstance(grids, list):
        for g in grids:
            runspecs.append((g, default_elecs, default_times))
    if elecs == True:
        for e in [1, 20, 100, 225, 1000, 5000]:
            runspecs.append((default_grids, e, default_times))
    elif isinstance(elecs, list):
        for e in elecs:
            runspecs.append((default_grids, e, default_times))
    if times == True:
        for times in [1, 10, 100, 1000]:
            runspecs.append((default_grids, default_elecs, times))
    elif isinstance(times, list):
        for t in times:
            runspecs.append((default_grids, default_elecs, t))
    if grids == False and elecs == False and times == False:
        runspecs.append((default_grids, default_elecs, default_times))
    return list(set(runspecs))

        
    
