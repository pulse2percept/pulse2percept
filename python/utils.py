"""
Utility functions for pulse2percept
"""
import numpy as np

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


def sparseconv(v,a):
    """
    Returns the discrete, linear convolution of two one-dimensional sequences.
    output is of length len(v) + len(a) -1 (same as the default for numpy.convolve)

    v is typically the kernel, a is the input to the system

    Can run faster than numpy.convolve if:
    (1) a is much longer than v
    (2) a is sparse (has lots of zeros)
    """
    na = len(a)
    nv = len(v)

    # find the indices into a (a should be sparse for it to run fast)
    pos = np.where(a != 0)[0]
    npos = len(pos)

    # zero the output - same length as a
    out = np.zeros(na+nv-1)

    # add shifted and scaled copies of v only where a is nonzero
    for i in range(0,npos):
        out[pos[i]:(pos[i]+nv)] = out[pos[i]:(pos[i]+nv)]+v*a[pos[i]]

    return(out)


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
>>>>>>> master
