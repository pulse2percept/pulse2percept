""":py:class:`~pulse2percept.utils.parfor`"""
import numpy as np
import multiprocessing
import joblib


# JobLib is optional. Rather than trying to import them all over, try
# once and then remember by setting a flag.
try:
    import joblib
    has_joblib = True
except (ImportError, AttributeError):
    has_joblib = False


def parfor(func, in_list, out_shape=None, n_jobs=-1, engine=None,
           scheduler='threading', func_args=[], func_kwargs={}):
    """
    Parallel for loop for NumPy arrays

    Parameters
    ----------
    func : callable
        The function to apply to each item in the array. Must have the form:
        func(arr, idx, \*args, \*kwargs) where arr is an ndarray and idx is an
        index into that array (a tuple). The Return of ``func`` needs to be one
        item (e.g. float, int) per input item.
    in_list : list
       All legitimate inputs to the function to operate over.
    out_shape : int or tuple of ints, optional
        If set, output will be reshaped accordingly. The new shape should be
        compatible with the original shape. If an integer, then the result will
        be a 1-D array of that length. One shape dimension can be -1. In this
        case, the value is inferred from the length of the array and remaining
        dimensions.
    n_jobs : integer, optional, default: 1
        The number of jobs to perform in parallel. -1 to use all cpus
    engine : str, optional, default: JobLib (if available), else serial
        {'joblib', 'serial'}
        The last one is useful for debugging -- runs the code without any
        parallelization.
    scheduler : str, optional, default: 'threading'
        Which scheduler to use (irrelevant for 'serial' engine):
        - 'threading': a scheduler backed by a thread pool
        - 'multiprocessing': a scheduler backed by a process pool
    \*func_args : list, optional
        Positional arguments to ``func``
    \*\*func_kwargs : dict, optional
        Keyword arguments to ``func``

    Returns
    -------
    ndarray
        NumPy array of identical shape to ``arr``

    Note
    ----
        Equivalent to pyAFQ version (blob e20eaa0 from June 3, 2016):
        https://github.com/arokem/pyAFQ/blob/master/AFQ/utils/parallel.py
    """
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
        n_jobs = n_jobs - 1

    if engine is None:
        if has_joblib:
            engine = 'joblib'
        else:
            engine = 'serial'
    if engine.lower() == 'joblib':
        p = joblib.Parallel(n_jobs=n_jobs, backend=scheduler)
        d = joblib.delayed(func)
        d_l = []
        for in_element in in_list:
            if isinstance(in_element, list):
                d_l.append(d(*in_element, *func_args, **func_kwargs))
            else:
                d_l.append(d(in_element, *func_args, **func_kwargs))
        results = p(d_l)
    elif engine.lower() == 'serial':
        results = []
        for in_element in in_list:
            if isinstance(in_element, list):
                results.append(func(*in_element, *func_args, **func_kwargs))
            else:
                results.append(func(in_element, *func_args, **func_kwargs))
    else:
        raise ValueError(f'Acceptable values for `engine` are: "serial" '
                         f'or "joblib", not "{engine}".')
    if out_shape is not None:
        return np.array(results).reshape(out_shape)
    else:
        return results
