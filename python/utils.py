"""
Utility functions for pulse2percept
"""
import numpy as np
from joblib import Parallel, delayed
from itertools import product
import multiprocessing
from joblib.pool import has_shareable_memory


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


def _sparseconv(v, a, mode):
    """
    Returns the discrete, linear convolution of two one-dimensional sequences.
    output is of length len(v) + len(a) -1 (same as the default for
    numpy.convolve).

    v is typically the kernel, a is the input to the system

    Can run faster than numpy.convolve if:
    (1) a is much longer than v
    (2) a is sparse (has lots of zeros)
    """
    # v = asarray(v)
    # a = asarray(a)
    v_len = v.shape[-1]
    a_len = a.shape[-1]
    out = np.zeros(a_len +  v_len - 1)

    pos = np.where(a != 0)[0]
    # add shifted and scaled copies of v only where a is nonzero
    for p in pos:
        out[p:p + v_len] = out[p:p + v_len] + v * a[p]

    if mode == 'full':
        return out
    elif mode == 'valid':
        return _centered(out, a_len - v_len + 1)
    elif mode == 'same':
        return _centered(out, a_len)
    else:
        raise ValueError("Acceptable mode flags are 'valid',"
                         " 'same', or 'full'.")


if has_jit:
    _sparseconvj = jit(_sparseconv)


def sparseconv(kernel, data, mode='full', dojit=True):
    """
    Returns the discrete, linear convolution of two one-dimensional sequences.

    Can run faster than numpy.convolve if:
    (1) `data` is much longer than `kernel`
    (2) `data` is sparse (has lots of zeros)

    Parameters
    ----------
    kernel : array_like
        First input, typically the kernel.
    data : array_like
        Second input, typically the data array.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
	``full``
        The output is the full discrete linear convolution of the inputs.
        (Default)
	``valid``
        The output consists only of those elements that do not rely on
        zero-padding.
	``same``
        The output is the same size as `data`, centered with respect to the
        'full' output.
    dojit : boolean
        A flag indicating whether to use numba's just-in-time compilation
        option.

    """
    if dojit:
        if not has_jit:
            e_s = ("You do not have numba ",
                   "please run sparsconv with dojit=False")
            raise ValueError(e_s)
        return _sparseconvj(kernel, data, mode)
    else:
        return _sparseconv(kernel, data, mode)


def _centered(vec, newlen):
    """
    Returns the center `newlen` portion of a vector.

    Adapted from scipy.signal.signaltools._centered:
    https://github.com/scipy/scipy/blob/v0.18.0/scipy/signal/signaltools.py#L236-L243

    """
    currlen = vec.shape[-1]
    startind = (currlen - newlen) // 2
    endind = startind + newlen
    return vec[startind:endind]


def parfor(func, in_list, out_shape=None, n_jobs=-1, func_args=[],
           func_kwargs={}):
    """
    Parallel for loop for numpy arrays

    Parameters
    ----------
    func : callable
        The function to apply to each item in the array. Must have the form:
        func(arr, idx, *args, *kwargs) where arr is an ndarray and idx is an
        index into that array (a tuple). The Return of `func` needs to be one
        item (e.g. float, int) per input item.

    in_list : list
       All legitimate inputs to the function to operate over.

    n_jobs : integer, optional
        The number of jobs to perform in parallel. -1 to use all cpus
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
    """
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
        n_jobs = n_jobs - 1

    p = Parallel(n_jobs=n_jobs, backend="multiprocessing", max_nbytes=1e6)
    d = delayed(func)
    d_l = []
    for in_element in in_list:
        d_l.append(d(in_element, *func_args, **func_kwargs))
    results = p(d_l)

    if out_shape is not None:
        return np.array(results).reshape(out_shape)
    else:
        return results


def mov2npy(movie_file, out_file):

    # Don't import cv at module level. Instead we'll use this on python 2 sometimes...
    try:
        import cv
    except ImportError:
        e_s = "You do not have opencv installed. "
        e_s += "You probably want to run this in Python 2"
        raise ImportError(e_s)

    capture = cv.CaptureFromFile(movie_file)
    frames = []
    img = cv.QueryFrame(capture)
    while img is not None:
        tmp = cv.CreateImage(cv.GetSize(img), 8, 3)
        cv.CvtColor(img, tmp, cv.CV_BGR2RGB)
        frames.append(np.asarray(cv.GetMat(tmp)))
        img = cv.QueryFrame(capture)
    frames = np.fliplr(np.rot90(np.mean(frames, -1).T, -1))
    np.save(out_file, frames)
