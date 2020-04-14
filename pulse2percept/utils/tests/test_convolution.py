import numpy as np
import numpy.testing as npt
import pytest

# Import whole module so we can reload it to test ImportErrors:
from pulse2percept.utils import convolution, gamma
from unittest import mock
from imp import reload


@pytest.mark.parametrize('mode', ('full', 'valid', 'same'))
@pytest.mark.parametrize('method', ('sparse', 'fft'))
def test_conv(mode, method):
    reload(convolution)
    # time vector for stimulus (long)
    stim_dur = 0.5  # seconds
    tsample = 0.001 / 1000
    t = np.arange(0, stim_dur, tsample)

    # stimulus (10 Hz anondic and cathodic pulse train)
    stim = np.zeros_like(t)
    stim[::1000] = 1
    stim[100::1000] = -1

    # kernel
    _, gg = gamma(1, 0.005, tsample)

    # make sure conv returns the same result as np.convolve for all modes:
    npconv = np.convolve(stim, gg, mode=mode)
    conv = convolution.conv(stim, gg, mode=mode, method=method)
    npt.assert_equal(conv.shape, npconv.shape)
    npt.assert_almost_equal(conv, npconv)

    with pytest.raises(ValueError):
        convolution.conv(gg, stim, mode="invalid")
    with pytest.raises(ValueError):
        convolution.conv(gg, stim, method="invalid")
