import numpy as np
import numpy.testing as npt
import pytest

import pulse2percept as p2p


def test_Parameters():
    my_params = p2p.utils.Parameters(foo='bar', list=[1, 2, 3])
    assert my_params.foo == 'bar'
    assert my_params.list == [1, 2, 3]
    assert str(my_params) == 'foo : bar\nlist : [1, 2, 3]'
    my_params.tuple = (1, 2, 3)
    assert my_params.tuple == (1, 2, 3)


def test_TimeSeries():
    max_val = 2.0
    max_idx = 156
    data_orig = np.random.rand(10, 10, 1000)
    data_orig[4, 4, max_idx] = max_val
    ts = p2p.utils.TimeSeries(1.0, data_orig)

    # Make sure function returns largest element
    tmax, vmax = ts.max()
    npt.assert_equal(tmax, max_idx)
    npt.assert_equal(vmax, max_val)

    # Make sure function returns largest frame
    tmax, fmax = ts.max_frame()
    npt.assert_equal(tmax, max_idx)
    npt.assert_equal(fmax.data, data_orig[:, :, max_idx])

    # Make sure resampling works
    tsample_new = 2.0
    ts_new = ts.resample(tsample_new)
    npt.assert_equal(ts_new.tsample, tsample_new)
    npt.assert_equal(ts_new.data.shape[-1], ts.data.shape[-1] / tsample_new)
    tmax_new, vmax_new = ts_new.max()
    npt.assert_equal(tmax_new, tmax)
    npt.assert_equal(vmax_new, vmax)

    # Make sure resampling leaves old data unaffected (deep copy)
    ts_new.data[0, 0, 0] = max_val * 2.0
    tmax_new, vmax_new = ts_new.max()
    npt.assert_equal(tmax_new, 0)
    npt.assert_equal(vmax_new, max_val * 2.0)
    tmax, vmax = ts.max()
    npt.assert_equal(tmax, max_idx)
    npt.assert_equal(vmax, max_val)

    # Make sure adding two TimeSeries objects works:
    # Must have the right type and size
    with pytest.raises(TypeError):
        ts + 4.0
    with pytest.raises(ValueError):
        ts_wrong_size = p2p.utils.TimeSeries(1.0, np.ones((2, 2)))
        ts + ts_wrong_size

    # Adding messes only with the last dimension of the array
    ts_add = ts + ts
    npt.assert_equal(ts_add.shape[:-1], ts.shape[:-1])
    npt.assert_equal(ts_add.shape[-1], ts.shape[-1] * 2)

    # If necessary, the second pulse train is resampled to the first
    ts_add = ts + ts_new
    npt.assert_equal(ts_add.shape[:-1], ts.shape[:-1])
    npt.assert_equal(ts_add.shape[-1], ts.shape[-1] * 2)
    ts_add = ts_new + ts
    npt.assert_equal(ts_add.shape[:-1], ts_new.shape[:-1])
    npt.assert_equal(ts_add.shape[-1], ts_new.shape[-1] * 2)

    # New one is a deep copy: Old data is unaffected
    tmax, vmax = ts.max()
    _, vmax_add = ts_new.max()
    ts_new.data[0, 0, 0] = vmax_add * 2.0
    tmax2, vmax2 = ts.max()
    npt.assert_equal(tmax, tmax2)
    npt.assert_equal(vmax, vmax2)


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

    # impulse reponse (kernel)
    G = np.exp(-tt / .005)

    # make sure sparseconv returns the same result as np.convolve
    # for all modes
    modes = ["full", "valid", "same"]
    for mode in modes:
        # np.convolve
        conv = np.convolve(stim, G, mode=mode)

        # p2p.utils.sparseconv
        sparse_conv = p2p.utils.sparseconv(stim, G, mode=mode, dojit=False)

        npt.assert_equal(conv.shape, sparse_conv.shape)
        npt.assert_almost_equal(conv, sparse_conv)

    with pytest.raises(ValueError):
        p2p.utils.sparseconv(G, stim, mode='invalid', dojit=False)


def test_conv():
    # time vector for stimulus (long)
    stim_dur = 0.5  # seconds
    tsample = 0.001 / 1000
    t = np.arange(0, stim_dur, tsample)

    # stimulus (10 Hz anondic and cathodic pulse train)
    stim = np.zeros_like(t)
    stim[::1000] = 1
    stim[100::1000] = -1

    # kernel
    tt, gg = p2p.utils.gamma(1, 0.005, tsample)

    # make sure conv returns the same result as
    # make sure sparseconv returns the same result as np.convolve
    # for all modes
    methods = ["fft", "sparse"]
    modes = ["full", "valid", "same"]
    for mode in modes:
        # np.convolve
        npconv = tsample * np.convolve(stim, gg, mode=mode)

        for method in methods:
            conv = p2p.utils.conv(stim, gg, tsample, mode=mode, method=method)

            npt.assert_equal(conv.shape, npconv.shape)
            npt.assert_almost_equal(conv, npconv)

    with pytest.raises(ValueError):
        p2p.utils.conv(gg, stim, tsample, mode="invalid")
    with pytest.raises(ValueError):
        p2p.utils.conv(gg, stim, tsample, method="invalid")


def power_it(num, n=2):
    return num ** n


def test_parfor():
    my_array = np.arange(100).reshape(10, 10)
    i, j = np.random.randint(0, 9, 2)
    my_list = list(my_array.ravel())
    npt.assert_equal(p2p.utils.parfor(power_it, my_list,
                                      out_shape=my_array.shape)[i, j],
                     power_it(my_array[i, j]))

    # If it's not reshaped, the first item should be the item 0, 0:
    npt.assert_equal(p2p.utils.parfor(power_it, my_list)[0],
                     power_it(my_array[0, 0]))


def test_gamma():
    tsample = 0.005 / 1000
    for tau in [0.001, 0.01, 0.1]:
        for n in [1, 2, 5]:
            t, g = p2p.utils.gamma(n, tau, tsample)
            npt.assert_equal(np.arange(0, t[-1] + tsample / 2.0, tsample), t)
            if n > 1:
                npt.assert_equal(g[0], 0.0)

            # Make sure area under the curve is normalized
            npt.assert_almost_equal(np.trapz(np.abs(g), dx=tsample), 1.0,
                                    decimal=2)

            # Make sure peak sits correctly
            npt.assert_almost_equal(g.argmax() * tsample, tau * (n - 1))


def test_traverse_randomly():
    # Generate a list of integers
    sequence = np.arange(100).tolist()

    # Iterate the sequence in random order
    shuffled_idx = []
    shuffled_val = []
    for idx, value in enumerate(p2p.utils.traverse_randomly(sequence)):
        shuffled_idx.append(idx)
        shuffled_val.append(value)

    # Make sure every element was visited once
    npt.assert_equal(shuffled_idx, np.arange(len(sequence)).tolist())
    npt.assert_equal(shuffled_val.sort(), sequence.sort())
