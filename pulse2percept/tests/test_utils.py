import numpy as np
import numpy.testing as npt
import pytest
import copy

from .. import utils

try:
    # Python 3
    from unittest import mock
except ImportError:
    # Python 2
    import mock

try:
    # Python 3
    from imp import reload
except ImportError:
    pass


@utils.deprecated(behavior='raise')
def raise_deprecated():
    pass


def test_deprecated():
    with pytest.raises(RuntimeError):
        raise_deprecated()


def test_TimeSeries():
    max_val = 2.0
    max_idx = 156
    data_orig = np.random.rand(10, 10, 1000)
    data_orig[4, 4, max_idx] = max_val
    ts = utils.TimeSeries(1.0, data_orig)

    # Make sure function returns largest element
    tmax, vmax = ts.max()
    npt.assert_equal(tmax, max_idx)
    npt.assert_equal(vmax, max_val)

    # Make sure function returns largest frame
    tmax, fmax = ts.max_frame()
    npt.assert_equal(tmax, max_idx)
    npt.assert_equal(fmax.data, data_orig[:, :, max_idx])

    # Make sure getitem works
    npt.assert_equal(isinstance(ts[3], utils.TimeSeries), True)
    npt.assert_equal(ts[3].data, ts.data[3])


def test_TimeSeries_resample():
    max_val = 2.0
    max_idx = 156
    data_orig = np.random.rand(10, 10, 1000)
    data_orig[4, 4, max_idx] = max_val
    ts = utils.TimeSeries(1.0, data_orig)
    tmax, vmax = ts.max()

    # Resampling with same sampling step shouldn't change anything
    ts_new = ts.resample(ts.tsample)
    npt.assert_equal(ts_new.shape, ts.shape)
    npt.assert_equal(ts_new.duration, ts.duration)

    # Make sure resampling works
    tsample_new = 4
    ts_new = ts.resample(tsample_new)
    npt.assert_equal(ts_new.tsample, tsample_new)
    npt.assert_equal(ts_new.data.shape[-1], ts.data.shape[-1] / tsample_new)
    npt.assert_equal(ts_new.duration, ts.duration)
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


def test_TimeSeries_append():
    max_val = 2.0
    max_idx = 156
    data_orig = np.random.rand(10, 10, 1000)
    data_orig[4, 4, max_idx] = max_val
    ts_orig = utils.TimeSeries(1.0, data_orig)

    # Make sure adding two TimeSeries objects works:
    # Must have the right type and size
    ts = copy.deepcopy(ts_orig)
    with pytest.raises(TypeError):
        ts.append(4.0)
    with pytest.raises(ValueError):
        ts_wrong_size = utils.TimeSeries(1.0, np.ones((2, 2)))
        ts.append(ts_wrong_size)

    # Adding messes only with the last dimension of the array
    ts = copy.deepcopy(ts_orig)
    ts.append(ts)
    npt.assert_equal(ts.shape[:-1], ts_orig.shape[:-1])
    npt.assert_equal(ts.shape[-1], ts_orig.shape[-1] * 2)

    # If necessary, the second pulse train is resampled to the first
    ts = copy.deepcopy(ts_orig)
    tsample_new = 2.0
    ts_new = ts.resample(tsample_new)
    ts.append(ts_new)
    npt.assert_equal(ts.shape[:-1], ts_orig.shape[:-1])
    npt.assert_equal(ts.shape[-1], ts_orig.shape[-1] * 2)
    ts_add = copy.deepcopy(ts_new)
    ts_add.append(ts_orig)
    npt.assert_equal(ts_add.shape[:-1], ts_new.shape[:-1])
    npt.assert_equal(ts_add.shape[-1], ts_new.shape[-1] * 2)


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

        # utils.sparseconv
        sparse_conv = utils.sparseconv(stim, G, mode=mode, use_jit=False)

        npt.assert_equal(conv.shape, sparse_conv.shape)
        npt.assert_almost_equal(conv, sparse_conv)

    with pytest.raises(ValueError):
        utils.sparseconv(G, stim, mode='invalid')


def test_conv():
    reload(utils)
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
    # for all modes
    methods = ["fft", "sparse"]
    modes = ["full", "valid", "same"]
    for mode in modes:
        # np.convolve
        npconv = np.convolve(stim, gg, mode=mode)

        for method in methods:
            conv = utils.conv(stim, gg, mode=mode, method=method)

            npt.assert_equal(conv.shape, npconv.shape)
            npt.assert_almost_equal(conv, npconv)

    with pytest.raises(ValueError):
        utils.conv(gg, stim, mode="invalid")
    with pytest.raises(ValueError):
        utils.conv(gg, stim, method="invalid")

    with mock.patch.dict("sys.modules", {"numba": {}}):
        with pytest.raises(ImportError):
            reload(utils)
            utils.conv(stim, gg, mode='full', method='sparse', use_jit=True)


def power_it(num, n=2):
    return num ** n


@pytest.mark.xfail
def test_parfor():
    my_array = np.arange(100).reshape(10, 10)
    i, j = np.random.randint(0, 9, 2)
    my_list = list(my_array.ravel())

    expected_00 = power_it(my_array[0, 0])
    expected_ij = power_it(my_array[i, j])

    with pytest.raises(ValueError):
        utils.parfor(power_it, my_list, engine='unknown')
    with pytest.raises(ValueError):
        utils.parfor(power_it, my_list, engine='dask', scheduler='unknown')

    for engine in ['serial', 'joblib', 'dask']:
        for scheduler in ['threading', 'multiprocessing']:
            # `backend` only relevant for dask, will be ignored for others
            # and should thus still give the right result
            calculated_00 = utils.parfor(power_it, my_list, engine=engine,
                                         scheduler=scheduler,
                                         out_shape=my_array.shape)[0, 0]
            calculated_ij = utils.parfor(power_it, my_list, engine=engine,
                                         scheduler=scheduler,
                                         out_shape=my_array.shape)[i, j]

        npt.assert_equal(expected_00, calculated_00)
        npt.assert_equal(expected_ij, calculated_ij)

        if engine == 'serial':
            continue

        with mock.patch.dict("sys.modules", {engine: {}}):
            reload(utils)
            with pytest.raises(ImportError):
                utils.parfor(power_it, my_list, engine=engine,
                             out_shape=my_array.shape)[0, 0]
        reload(utils)


def test_gamma():
    tsample = 0.005 / 1000

    with pytest.raises(ValueError):
        t, g = utils.gamma(0, 0.1, tsample)
    with pytest.raises(ValueError):
        t, g = utils.gamma(2, -0.1, tsample)
    with pytest.raises(ValueError):
        t, g = utils.gamma(2, 0.1, -tsample)

    for tau in [0.001, 0.01, 0.1]:
        for n in [1, 2, 5]:
            t, g = utils.gamma(n, tau, tsample)
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
    for idx, value in enumerate(utils.traverse_randomly(sequence)):
        shuffled_idx.append(idx)
        shuffled_val.append(value)

    # Make sure every element was visited once
    npt.assert_equal(shuffled_idx, np.arange(len(sequence)).tolist())
    npt.assert_equal(shuffled_val.sort(), sequence.sort())
