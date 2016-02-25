"""
Utility functions for pulse2percept
"""
import numpy as np
from joblib import Parallel, delayed
from itertools import product


try:
    from numba import jit
    has_jit = True
except ImportError:
    has_jit = False


class Parameters(object):
    def __init__(self, **params):
        for k, v in params.items():
            self.__dict__[k] = v

    def __repr__(self):
        my_list = []
        for k, v in self.__dict__.items():
            my_list.append("%s : %s" % (k, v))
        my_list.sort()
        my_str = "\n".join(my_list)
        return my_str

    def __setattr(self, name, value):
        self.__dict__[name] = values


class TimeSeries(object):
    def __init__(self, tsample, data):
        """
        Represent a time-series
        """
        self.data = data
        self.tsample = tsample
        self.sampling_rate = 1 / tsample
        self.duration = self.data.shape[-1] * tsample
        self.time = np.linspace(tsample, self.duration, data.shape[-1])
        self.shape = data.shape

    def __getitem__(self, y):
        return TimeSeries(self.tsample, self.data[y])

    def resample(self, factor):
        TimeSeries.__init__(self, self.tsample * factor,
                            self.data[..., ::factor])


def _sparseconv(v, a):
    """
    Returns the discrete, linear convolution of two one-dimensional sequences.
    output is of length len(v) + len(a) -1 (same as the default for numpy.convolve)

    v is typically the kernel, a is the input to the system

    Can run faster than numpy.convolve if:
    (1) a is much longer than v
    (2) a is sparse (has lots of zeros)
    """
    v_len = v.shape[-1]
    a_len = a.shape[-1]
    out = np.zeros(a_len +  v_len - 1)

    pos = np.where(a != 0)[0]
    # add shifted and scaled copies of v only where a is nonzero
    for p in pos:
        out[p:p + v_len] = out[p:p + v_len] + v * a[p]
    return out

if has_jit:
    _sparseconvj = jit(_sparseconv)


def sparseconv(v, a, dojit=True):
    """
    Returns the discrete, linear convolution of two one-dimensional sequences.
    output is of length len(v) + len(a) -1 (same as the default for numpy.convolve)

    v is typically the kernel, a is the input to the system

    Can run faster than numpy.convolve if:
    (1) a is much longer than v
    (2) a is sparse (has lots of zeros)
    """
    if dojit:
        if not has_jit:
            e_s = ("You do not have numba ",
                   "please run sparsconv with dojit=False")
            raise ValueError(e_s)
        return _sparseconvj(v, a)
    else:
        return _sparseconv(v, a)


def parfor(arr, func, n_jobs=1, *args, **kwargs):
    """
    Parallel for loop for numpy arrays

    Parameters
    ----------
    arr : ndarray
        Input array to operate on

    func : callable
        The function to apply to each item in the array. Must have the form:
        func(arr, idx, *args, *kwargs) where arr is an ndarray and idx is an
        index into that array (a tuple). The Return of `func` needs to be one
        item (e.g. float, int) per input item.

    n_jobs : integer, optional
        The number of jobs to perform in parallel
        Default: 1

    args : list, optional
        Positional arguments to `func`

    kwargs : list, optional
        Keyword arguments to `func`

    Returns
    -------
    ndarray of identical shape to `arr`

    Examples
    --------
    >>> def power_it(arr, idx, n=2):
    ...     return arr[idx] ** n
    >>> my_array = np.arange(100).reshape(10, 10)
    >>> parfor(my_array, power_it, n=3, n_jobs=2)
    """
    idx = product(*(range(s) for s in arr.shape))
    results = Parallel(n_jobs=n_jobs)(delayed(func)(arr, i, *args, **kwargs)
                                      for i in idx)
    return np.array(results).reshape(arr.shape)
