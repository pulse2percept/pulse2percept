import numpy as np
import numpy.testing as npt
import pytest

# Import whole module so we can reload it:
from pulse2percept.utils import convolution
from pulse2percept import utils
from unittest import mock
from imp import reload


@pytest.mark.parametrize('mode', ('full', 'valid', 'same'))
def test_sparseconv(mode):
    # time vector for stimulus (long)
    maxT = .5  # seconds
    nt = 100000
    t = np.linspace(0, maxT, nt)

    # stimulus (10 Hz anondic and cathodic pulse train)
    stim = np.zeros(nt)
    stim[0:nt:10000] = 1
    stim[100:nt:1000] = -1

    # time vector for impulse response (shorter)
    tt = t[t < .1]

    # impulse reponse (kernel)
    G = np.exp(-tt / .005)

    # make sure sparseconv returns the same result as np.convolve
    # for all modes:
    conv = np.convolve(stim, G, mode=mode)
    sparse_conv = convolution.sparseconv(stim, G, mode=mode, use_jit=False)
    npt.assert_equal(conv.shape, sparse_conv.shape)
    npt.assert_almost_equal(conv, sparse_conv)

    with pytest.raises(ValueError):
        convolution.sparseconv(G, stim, mode='invalid')


@pytest.mark.parametrize('mode', ('full', 'valid', 'same'))
@pytest.mark.parametrize('method', ('fft', 'sparse'))
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
    _, gg = utils.gamma(1, 0.005, tsample)

    # make sure conv returns the same result as
    # make sure sparseconv returns the same result as np.convolve
    # for all modes:
    npconv = np.convolve(stim, gg, mode=mode)
    conv = convolution.conv(stim, gg, mode=mode, method=method)
    npt.assert_equal(conv.shape, npconv.shape)
    npt.assert_almost_equal(conv, npconv)

    with pytest.raises(ValueError):
        convolution.conv(gg, stim, mode="invalid")
    with pytest.raises(ValueError):
        convolution.conv(gg, stim, method="invalid")

    with mock.patch.dict("sys.modules", {"numba": {}}):
        with pytest.raises(ImportError):
            reload(convolution)
            convolution.conv(stim, gg, mode='full', method='sparse',
                             use_jit=True)
