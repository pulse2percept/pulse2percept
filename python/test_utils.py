import utils
import numpy as np
import numpy.testing as npt

def test_Parameters():
    my_params = utils.Parameters(foo='bar', list=[1, 2, 3])
    assert my_params.foo == 'bar'
    assert my_params.list == [1, 2, 3]
    assert str(my_params) == 'foo : bar\nlist : [1, 2, 3]'
    my_params.tuple = (1, 2, 3)
    assert my_params.tuple == (1, 2, 3)

def test_sparseconv():
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
    ntt = len(tt)

    # impulse reponse (kernel)
    G = np.exp(-tt/.005)

    # np.convolve
    outConv = np.convolve(stim, G)
    outConv = outConv[0:len(t)]

    # utils.sparseconv
    outSparseconv = utils.sparseconv(G, stim, dojit=False)
    outSparseconv = outSparseconv[0:len(t)]
    npt.assert_almost_equal(outConv, outSparseconv)
