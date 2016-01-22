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
    """Returns the discrete, linear convolution of two one-dimensional sequences.
       output is of length len(v) + len(a) -1 (same as the default for numpy.convolve)
       
       Runs faster than numpy.convolve if:
       (1) a is longer than v (v is typically the kernel, a is the input to the system)
       (2) a is sparse (has lots of zeros)
    """       
    na = len(a)
    nv = len(v)
    
    # find the indices into a (a should be sparse)
    pos = np.where(a != 0)[0]
    
    # zero the output - same length as a
    out = np.zeros(na+nv-1)

    # loop through the indices in the output
    for i in range(0,na):
        # find indices into pos in the inverval back in time from i the length of v
        iid = np.logical_and(pos>i-nv,pos<=i)
        if any(iid): # if there is any nonzero simulus during this time   
            id = pos[iid]  #find the indices into a where a is nonzero
            out[i] = np.dot(v[i-id],a[id])  #dot product of flipped kernel and a   
    
    return(out)

