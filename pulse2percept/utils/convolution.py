"""
convolution
===========
"""

import numpy as np
from scipy import signal as sps

try:
    from numba import jit
    has_jit = True
except ImportError:
    has_jit = False


def center_vector(vec, newlen):
    """Returns the center ``newlen`` portion of a vector.

    Adapted from ``scipy.signal.signaltools._centered``:
    github.com/scipy/scipy/blob/v0.18.0/scipy/signal/signaltools.py#L236-L243

    """
    currlen = vec.shape[-1]
    startind = (currlen - newlen) // 2
    endind = startind + newlen
    return vec[startind:endind]


def _sparseconv(data, kernel, mode):
    """Returns the discrete, linear convolution of two 1D sequences

    This function returns the discrete, linear convolution of two
    one-dimensional sequences, where the length of the output is determined
    by ``mode``.
    Can run faster than ``np.convolve`` if:
    (1) ``data`` is much longer than ``kernel``
    (2) ``data`` is sparse (has lots of zeros)
    """
    # NOTE Numba 0.44 has trouble with jitting nested functions when they
    # raise exceptions, so we don't raise ValueError here.
    kernel_len = kernel.size
    data_len = data.size
    out = np.zeros(data_len + kernel_len - 1)

    pos = np.where(data.ravel() != 0)[0]
    # Add shifted and scaled copies of `kernel` only where `data` is nonzero
    for p in pos:
        out[p:p + kernel_len] = (out[p:p + kernel_len] +
                                 kernel.ravel() * data.ravel()[p])
    if mode.lower() == 'full':
        return out
    elif mode.lower() == 'valid':
        return center_vector(out, data_len - kernel_len + 1)
    elif mode.lower() == 'same':
        return center_vector(out, data_len)


def conv(data, kernel, mode='full', method='fft', use_jit=True):
    """Convoles data with a kernel using either FFT or sparse convolution

    This function convolves data with a kernel, relying either on the
    fast Fourier transform (FFT) or a sparse convolution function.

    Parameters
    ----------
    data : array_like
        First input, typically the data array
    kernel : array_like
        Second input, typically the kernel
    mode : str {'full', 'valid', 'same'}, optional, default: 'full'
        A string indicating the size of the output:

        - ``full``:
            The output is the full discrete linear convolution of the inputs.
        - ``valid``:
            The output consists only of those elements that do not rely on
            zero-padding.
        - ``same``:
            The output is the same size as ``data``, centered with respect to
            the 'full' output.

    method : str {'fft', 'sparse'}, optional, default: 'fft'
        A string indicating the convolution method:

        - ``fft``:
            Use the fast Fourier transform (FFT).
        - ``sparse``:
            Use the sparse convolution.

    use_jit : bool, optional, default: True
        A flag indicating whether to use Numba's just-in-time compilation
        option (only relevant for ``method=='sparse'``).
    """
    if mode not in ['full', 'valid', 'same']:
        raise ValueError("Acceptable mode flags are 'full', 'valid', or "
                         "'same'.")
    if method not in ['fft', 'sparse']:
        raise ValueError("Acceptable methods are: 'fft', 'sparse'.")
    if method.lower() == 'fft':
        # Use FFT: faster on non-sparse data
        conved = sps.fftconvolve(data, kernel, mode)
    elif method.lower() == 'sparse':
        # Use sparseconv: faster on sparse data
        if use_jit:
            if not has_jit:
                e_s = "You do not have numba, please run sparseconv with "
                e_s += "`use_jit`=False."
                raise ImportError(e_s)
            sparseconv = jit(_sparseconv)
        else:
            sparseconv = _sparseconv
        conved = sparseconv(data, kernel, mode)
    return conved
