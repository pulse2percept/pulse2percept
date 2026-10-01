import numpy as np
import copy
import warnings
import numpy.testing as npt
import pytest

from pulse2percept.models import AlphaTemporal, FadingTemporal
from pulse2percept.models.retina import Nanduri2012Temporal
from pulse2percept.models._temporal import alpha_fast, fading_fast
from pulse2percept.models.base import _FrameClock, _ModelResponse
from pulse2percept.stimuli import (Stimulus, MonophasicPulse, BiphasicPulse,
                                   BiphasicPulseTrain)
from pulse2percept.percepts import Percept
from pulse2percept.units import ms
from pulse2percept.utils import FreezeError


def test_FadingTemporal():
    model = FadingTemporal()
    # User can set their own params:
    model.dt = 0.1
    npt.assert_equal(model.dt, 0.1)
    model.build(dt=1e-4)
    npt.assert_equal(model.dt, 1e-4)
    # User cannot add more model parameters:
    with pytest.raises(FreezeError):
        model.rho = 100

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(None), None)

    # Zero in = zero out:
    stim = BiphasicPulse(0, 1)
    percept = model.predict_percept(stim, t_percept=[0, 1, 2])
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, (1, 1, 3))
    npt.assert_almost_equal(percept.data, 0)

    # Can't request the same time twice (the Cython loop increments
    # `idx_frame` after each write):
    with pytest.raises(ValueError):
        stim = Stimulus(np.ones((1, 100)))
        model.predict_percept(stim, t_percept=[0.2, 0.2])

    # Simple decay for single cathodic pulse. Current flows from t=DT to t=1,
    # so sample-and-hold at dt=5e-3 ms integrates for 0.995 ms, giving slightly
    # less than the ideal 1-exp(-1) = 0.632:
    model = FadingTemporal(tau=1).build()
    stim = MonophasicPulse(-1, 1, stim_dur=10)
    percept = model.predict_percept(stim, np.arange(stim.duration))
    npt.assert_almost_equal(percept.data.ravel()[:3], [0, 0.628, 0.230],
                            decimal=3)
    npt.assert_almost_equal(percept.data.ravel()[-1], 0, decimal=3)

    # But all zeros for anodic pulse:
    stim = MonophasicPulse(1, 1, stim_dur=10)
    percept = model.predict_percept(stim, np.arange(stim.duration))
    npt.assert_almost_equal(percept.data, 0)


@pytest.mark.parametrize('model_cls', (FadingTemporal, AlphaTemporal))
def test_generic_temporal_tau_at_least_one_step(model_cls):
    # tau must be >= dt. tau <= 0 divides by zero; tau < dt overshoots the
    # drive by dt/tau and oscillates (at tau=dt/2, between 2x drive and 0):
    for tau in (-1, 0, 0.005 / 2, 0.004):
        with pytest.raises(ValueError):
            model_cls(tau=tau, dt=0.005).build()
    # tau == dt is the fastest stable setting:
    model_cls(tau=0.005, dt=0.005).build()


@pytest.mark.parametrize('model_cls', (FadingTemporal, AlphaTemporal))
def test_deepcopy_generic_temporal(model_cls):
    original = model_cls()
    copied = copy.deepcopy(original)

    # Assert they are different objects
    npt.assert_equal(id(original) != id(copied), True)

    # Assert the objects are equivalent to each other
    npt.assert_equal(original == copied, True)

    # Assert building one object does not affect the copied
    original.build()
    npt.assert_equal(copied.is_built, False)
    npt.assert_equal(original != copied, True)

    # Changing the copy does not affect the original:
    copied = copy.deepcopy(original)
    copied.verbose = False
    npt.assert_equal(original.verbose, True)
    npt.assert_equal(original != copied, True)

    # Assert "destroying" the original doesn't affect the copied
    original = None
    npt.assert_equal(copied is not None, True)


def test_FadingTemporal_matches_reference_integrator():
    """The leaky integrator matches a plain Python reference

    ``fading_fast`` loops over time outside and space inside (to vectorize).
    The reference steps one location at a time. The stimulus straddles zero,
    so anodic samples must contribute nothing (half-wave rectification).
    """
    model = FadingTemporal(dt=0.01, tau=50, thresh_percept=0).build()
    n_space, n_stim = 3, 5
    rng = np.random.default_rng(0)
    data = (rng.random((n_space, n_stim)) - 0.5).astype(np.float32)
    npt.assert_equal(np.any(data > 0) and np.any(data < 0), True)
    t_stim = np.arange(n_stim, dtype=np.float32) * 2.0
    stim = Stimulus(data, time=t_stim)
    t_percept = np.array([0.0, 2.0, 4.0, 8.0])
    got = model.predict_percept(stim, t_percept=t_percept).data.reshape(
        n_space, -1)

    dt, tau = np.float32(model.dt), np.float32(model.tau)
    # Compute `dt / tau` once, as in the kernel:
    dt_tau = np.float32(dt / tau)
    idx_p = np.uint32(np.round(t_percept / model.dt))
    for s in range(n_space):
        bright = np.float32(0.0)
        idx_stim, frame = 0, 0
        for i in range(int(idx_p[-1]) + 1):
            # Several stimulus frames can fall inside one `dt`. See
            # `test_FadingTemporal_frames_closer_together_than_dt`:
            while (idx_stim + 1 < n_stim and
                   np.float32(i) * dt >= t_stim[idx_stim + 1]):
                idx_stim += 1
            amp = data[s, idx_stim]
            drive = np.float32(max(-amp, 0.0))
            bright = np.float32(bright + dt_tau * (drive - bright))
            if bright < 0:
                bright = np.float32(0.0)
            if i == idx_p[frame]:
                # Not exact: compilers may fuse `bright + dt_tau * x` into one
                # FMA (Clang on Apple Silicon does, MSVC on x86-64 does not),
                # while NumPy never does:
                npt.assert_allclose(got[s, frame], bright, rtol=1e-6)
                frame += 1


def test_FadingTemporal_rectifies_the_drive():
    """A charge-balanced pulse train produces a persistent percept

    Without rectification, the anodic phase of a biphasic pulse cancels the
    cathodic phase in the (linear) leaky integrator, so brightness is nonzero
    for only one phase (1.8% duty cycle).
    """
    train = BiphasicPulseTrain(20, -50, 0.46, stim_dur=1000)
    model = FadingTemporal(tau=100).build()
    t = np.round(np.arange(0, 1000, 0.05), 5)
    bright = model.predict_percept(train, t_percept=t).data.ravel()
    late = bright[t >= 500]
    # Brightness persists between pulses (without rectification, it returns
    # to within 0.6% of zero after every pulse):
    npt.assert_array_less(0.5, late.min() / late.max())
    # Brightness accumulates: the steady-state minimum exceeds the single-pulse
    # peak:
    one_pulse = bright[t < 50].max()
    npt.assert_array_less(one_pulse, late.min())
    npt.assert_array_less(2 * one_pulse, late.max())

    # A slower train is dimmer, because it decays longer between pulses
    # (required for frequency modulation):
    def steady(freq):
        stim = BiphasicPulseTrain(freq, -50, 0.46, stim_dur=1000)
        return model.predict_percept(stim, t_percept=t).data.ravel()[-1]

    rates = [10, 20, 50, 100]
    npt.assert_equal(np.all(np.diff([steady(f) for f in rates]) > 0), True)

    # Purely cathodic stimuli are unaffected by rectification:
    model = FadingTemporal(tau=1).build()
    percept = model.predict_percept(MonophasicPulse(-1, 1, stim_dur=10),
                                    np.arange(10))
    npt.assert_almost_equal(percept.data.ravel()[:3], [0, 0.628, 0.230],
                            decimal=3)


@pytest.mark.parametrize('n_space', (1, 63, 64, 65, 130))
def test_FadingTemporal_block_boundaries(n_space):
    """Locations are integrated in fixed-size blocks, the last one partial

    Sizes on either side of the block width test the partial last block.
    """
    model = FadingTemporal(dt=0.05, tau=30).build()
    rng = np.random.default_rng(n_space)
    data = (rng.random((n_space, 4)) - 0.7).astype(np.float32)
    stim = Stimulus(data, time=np.arange(4, dtype=float) * 5)
    percept = model.predict_percept(stim, t_percept=[0, 5, 10, 15])
    npt.assert_equal(percept.data.shape, (n_space, 1, 4))
    # Every location is integrated, including the partial block:
    single = np.stack([
        model.predict_percept(Stimulus(data[i:i + 1], time=stim.time),
                              t_percept=[0, 5, 10, 15]).data.ravel()
        for i in range(n_space)])
    npt.assert_array_equal(percept.data.reshape(n_space, -1), single)


def test_FadingTemporal_thread_count_invariant():
    """The result does not depend on the number of threads"""
    rng = np.random.default_rng(7)
    data = (rng.random((200, 6)) - 0.6).astype(np.float32)
    stim = Stimulus(data, time=np.arange(6, dtype=float) * 3)
    serial = FadingTemporal(dt=0.05, tau=40, n_threads=1).build(
        ).predict_percept(stim, t_percept=[0, 5, 10, 15]).data
    for n_threads in (2, 3, 8):
        parallel = FadingTemporal(dt=0.05, tau=40, n_threads=n_threads).build(
            ).predict_percept(stim, t_percept=[0, 5, 10, 15]).data
        npt.assert_array_equal(parallel, serial)


def test_FadingTemporal_long_run_matches_closed_form():
    """A long constant drive matches the closed-form recurrence

    Stepping one `dt` at a time in float32 loses accuracy near the fixed point
    (increments of ~`dt / tau` round away). At `dt=0.005`, `tau=100`, a 1 s
    constant drive stepped that way is ~9000 ulps low, so the reference is the
    recurrence in float64.
    """
    dt, tau, amp = 0.005, 100.0, 50.0
    # One frame longer than all output points, so the drive is constant:
    stim = Stimulus(np.array([[-amp, 0.0]]), time=[0.0, 1e6])
    t_percept = np.array([200.0, 700.0, 1500.0])
    got = FadingTemporal(dt=dt, tau=tau, thresh_percept=0,
                         reduce='last').build().predict_percept(
        stim, t_percept=t_percept).data.ravel()

    # `q` uses the kernel's float32 `dt / tau`, composed in float64. Each run
    # covers the steps after the previous output point up to this one:
    q = 1.0 - float(np.float32(dt / tau))
    idx = np.round(t_percept / dt).astype(np.int64)
    want, bright, prev = [], 0.0, -1
    for i in idx:
        bright = amp + (bright - amp) * q ** (i - prev)
        want.append(bright)
        prev = i
    npt.assert_allclose(got, want, rtol=1e-6)
    # Brightness is still rising at every point, so a wrong `q**n` would show:
    npt.assert_array_less(want[0], want[1])
    npt.assert_array_less(want[1], want[2])
    npt.assert_array_less(want[2], amp)


def test_FadingTemporal_peak_is_exact():
    """The in-kernel peak equals the max over every simulation step

    Not bit-exact: `fading_fast` composes runs of steps that share a stimulus
    frame into one affine map, so dense and sparse output have different
    rounding. See `test_FadingTemporal_long_run_matches_closed_form`.
    """
    rng = np.random.default_rng(3)
    data = (rng.random((5, 12)) - 0.5).astype(np.float32) * 40
    t_stim = (np.arange(12) * 4.0).astype(np.float32)
    dt, tau = 0.05, 20.0
    # Brightness at every simulation step:
    n_sim = int(round(44 / dt)) + 1
    dense = fading_fast(data, t_stim, np.arange(n_sim, dtype=np.uint32), dt,
                        tau, 0.0, 1, 0)
    out = np.array([37, 210, 400, 601, 880], dtype=np.uint32)
    peak = fading_fast(data, t_stim, out, dt, tau, 0.0, 1, 1)
    last = fading_fast(data, t_stim, out, dt, tau, 0.0, 1, 0)
    # Each interval runs from the previous output point up to and including
    # this one:
    lo = np.r_[0, out[:-1]]
    brute = np.stack([dense[:, a:b + 1].max(axis=1)
                      for a, b in zip(lo, out)], axis=1)
    npt.assert_allclose(peak, brute, rtol=1e-5)
    # `reduce='last'` returns the value at the interval end:
    npt.assert_allclose(last, dense[:, out], rtol=1e-5)
    # The interval includes its end point, so peak >= last exactly:
    npt.assert_equal(np.all(peak >= last), True)
    npt.assert_equal(np.any(peak > last), True)
    # The peak does not depend on the number of threads:
    for n_threads in (2, 4, 8):
        npt.assert_array_equal(
            fading_fast(np.tile(data, (40, 1)), t_stim, out, dt, tau, 0.0,
                        n_threads, 1),
            np.tile(peak, (40, 1)))


def test_FadingTemporal_reduce():
    """`reduce` applies only to automatic output times, not to `t_percept`"""
    stim = BiphasicPulseTrain(20, -50, 0.46, stim_dur=200)
    peak_model = FadingTemporal(tau=100).build()
    npt.assert_equal(peak_model.reduce, 'peak')
    last_model = FadingTemporal(tau=100, reduce='last').build()

    # Explicit `t_percept` returns those instants, whatever `reduce` is:
    t = [0, 50, 100, 150]
    npt.assert_array_equal(peak_model.predict_percept(stim, t_percept=t).data,
                           last_model.predict_percept(stim, t_percept=t).data)

    # With automatic output times, `reduce='last'` samples the interval ends:
    got = last_model.predict_percept(stim)
    npt.assert_array_equal(
        got.data, last_model.predict_percept(stim, t_percept=got.time).data)
    # 'peak' is never below 'last' (the interval includes its end):
    peaked = peak_model.predict_percept(stim)
    npt.assert_almost_equal(peaked.time, got.time)
    npt.assert_equal(np.all(peaked.data >= got.data), True)
    npt.assert_equal(np.any(peaked.data > got.data), True)

    with pytest.raises(ValueError):
        FadingTemporal(reduce='mean').build().predict_percept(stim)


def test_FadingTemporal_frames_closer_together_than_dt():
    """Several stimulus frames can fall inside one simulation step

    Encoded pulse edges are on the DT=1e-3 ms grid while `dt` defaults to
    5e-3 ms, so this is the normal case.
    """
    # A 0.1 ms cathodic blip strictly between two simulation steps. At t=0.5
    # sample-and-hold reads amplitude 0, so the blip is never seen:
    t_stim = np.array([0.0, 0.1, 0.2, 0.3, 10.0], dtype=np.float32)
    data = np.array([[0.0, -100.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    dt = 0.5
    idx = np.arange(0, 21, dtype=np.uint32)
    got = fading_fast(data, t_stim, idx, dt, 100.0, 0.0, 1, 0).ravel()
    npt.assert_array_equal(got, np.zeros_like(got))

    # In general, the frame at each step is given by `searchsorted`:
    rng = np.random.default_rng(11)
    n_stim = 40
    # Frame times far closer together than `dt`, plus a couple of long gaps:
    gaps = rng.choice([0.001, 0.002, 0.05, 1.3], size=n_stim - 1)
    t_stim = np.concatenate(([0.0], np.cumsum(gaps))).astype(np.float32)
    data = ((rng.random((3, n_stim)) - 0.5) * 60).astype(np.float32)
    tau = 25.0
    idx = np.arange(0, int(t_stim[-1] / dt) + 1, dtype=np.uint32)
    got = fading_fast(data, t_stim, idx, dt, tau, 0.0, 1, 0)

    frame = np.searchsorted(t_stim, (idx * dt).astype(np.float32),
                            side='right') - 1
    npt.assert_equal(np.any(np.diff(frame) > 1), True)  # frames skipped
    want = np.zeros_like(got)
    for s in range(data.shape[0]):
        bright = np.float32(0.0)
        for i, f in enumerate(frame):
            drive = np.float32(max(-data[s, f], 0.0))
            bright = np.float32(bright + np.float32(dt) *
                                (drive - bright) / np.float32(tau))
            bright = max(bright, np.float32(0.0))
            want[s, i] = bright
    npt.assert_allclose(got, want, rtol=1e-6, atol=1e-7)


def test_TemporalModel_reduce_fallback():
    """`reduce='peak'` works for models without an in-kernel peak

    `FadingTemporal` tracks the peak in its integrator. For other models,
    `predict_percept` samples each interval several times and keeps the max.
    """
    # No encoder metadata, so this uses the default 20 ms output grid:
    stim = BiphasicPulseTrain(20, 50, 0.46, stim_dur=200)
    npt.assert_equal(Nanduri2012Temporal()._reduces_intervals, False)
    peak = Nanduri2012Temporal(reduce='peak').build().predict_percept(stim)
    last = Nanduri2012Temporal(reduce='last').build().predict_percept(stim)
    npt.assert_almost_equal(peak.time, np.arange(0, 201, 20))
    npt.assert_almost_equal(peak.time, last.time)
    npt.assert_equal(np.any(peak.data != last.data), True)
    # The samples include the output point, so peak >= last:
    npt.assert_array_less(last.data - 1e-7, peak.data)

    # Sampling approximates the peak but never exceeds the true (dense) peak:
    model = Nanduri2012Temporal(reduce='peak').build()
    dense = model.predict_percept(
        stim, t_percept=np.round(np.arange(0, 181, model.dt), 5)).data.ravel()
    idx = np.round(peak.time / model.dt).astype(int)
    true = np.array([dense[max(0, a):b + 1].max()
                     for a, b in zip(np.r_[0, idx[:-1]], idx)])
    npt.assert_array_less(peak.data.ravel() - 1e-7, true)
    npt.assert_allclose(peak.data.ravel(), true, atol=0.01)

    # Published models default to `reduce='last'`:
    npt.assert_equal(Nanduri2012Temporal().reduce, 'last')
    npt.assert_array_equal(
        Nanduri2012Temporal().build().predict_percept(stim).data, last.data)
    # The generic model defaults to 'peak':
    npt.assert_equal(FadingTemporal().reduce, 'peak')


@pytest.mark.parametrize('model_cls', (FadingTemporal, AlphaTemporal,
                                       Nanduri2012Temporal))
def test_TemporalModel_keeps_silent_rows(model_cls):
    # Compression drops all-zero rows internally; the output keeps them:
    data = np.zeros((4, 4))
    data[1] = [-20, 20, 0, 0]
    data[3] = [0, -40, 40, 0]
    time = [0, 1, 2, 5]
    t_percept = [1, 2, 3, 5]
    model = model_cls().build()
    percept = model.predict_percept(Stimulus(data, time=time),
                                    t_percept=t_percept)
    npt.assert_equal(percept.data.shape, (4, 1, 4))
    npt.assert_equal(percept.data[[0, 2]], 0)
    active = model.predict_percept(Stimulus(data[[1, 3]], time=time),
                                   t_percept=t_percept)
    npt.assert_equal(np.all(np.any(active.data != 0, axis=-1)), True)
    npt.assert_array_equal(percept.data[[1, 3]], active.data)


def test_TemporalModel_blank_percept_warning():
    # FadingTemporal is driven by cathodic (negative) current, so an
    # all-positive stimulus (e.g., an unencoded grayscale image) gives zero:
    anodic = Stimulus(np.ones((4, 10)), time=np.arange(10) * 10.0)
    model = FadingTemporal().build()
    with pytest.warns(UserWarning, match='all-zero percept'):
        percept = model.predict_percept(anodic, t_percept=[0, 20, 40])
    npt.assert_almost_equal(percept.data, 0)

    # Flipping the polarity (as the warning suggests) works:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        cathodic = model.predict_percept(Stimulus(-anodic.data,
                                                  time=anodic.time),
                                         t_percept=[0, 20, 40])
    npt.assert_equal(cathodic.data.max() > 0, True)

    # No polarity warning for a cathodic stimulus that is too weak:
    weak = Stimulus(np.full((4, 10), -1e-12), time=np.arange(10) * 10.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        model.predict_percept(weak, t_percept=[0, 20, 40])

    # Nanduri2012Temporal uses the opposite sign convention:
    nanduri = Nanduri2012Temporal().build()
    npt.assert_equal(nanduri._drive_sign, 1)
    npt.assert_equal(FadingTemporal()._drive_sign, -1)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        nanduri.predict_percept(anodic, t_percept=[0, 20, 40])


def test_FadingTemporal_tau_limits():
    """Limiting cases of `tau`, which sets both rise and decay

    *  `tau == dt`: no dynamics. Brightness reaches the drive in one step, so
       the model is a half-wave rectifier (percept = cathodic part of the
       stimulus).
    *  `tau -> inf`: brightness never fades but never charges either, so the
       percept vanishes as `1/tau`.
    """
    dt = 0.005
    # Edges on the `dt` grid, so sample-and-hold is exact:
    stim = Stimulus(np.array([[0.0, -50.0, 0.0, 30.0, 0.0]]),
                    time=[0.0, 1.0, 2.0, 3.0, 4.0])
    t = np.round(np.arange(0, 4, dt), 5)
    rectified = np.maximum(-np.asarray(stim.data).ravel(), 0)[
        np.searchsorted(np.asarray(stim.time), t, side='right') - 1]

    got = FadingTemporal(tau=dt, dt=dt).build().predict_percept(
        stim, t_percept=t).data.ravel()
    npt.assert_array_equal(got, rectified)
    # The anodic phase is removed, not inverted:
    npt.assert_equal(np.any(np.asarray(stim.data) > 0), True)
    npt.assert_equal(got.max(), 50.0)

    # Coarse output gives the same values. At tau == dt the decay per step is
    # total, so a composed run of 100 steps must land exactly on the drive:
    coarse = np.round(np.arange(0, 4, 0.5), 5)
    npt.assert_array_equal(
        FadingTemporal(tau=dt, dt=dt, reduce='last').build().predict_percept(
            stim, t_percept=coarse).data.ravel(),
        rectified[np.searchsorted(t, coarse)])

    # With larger tau, brightness lags the drive: it stays below the drive
    # while on, and is still nonzero after the drive ends:
    lagged = FadingTemporal(tau=10 * dt, dt=dt).build().predict_percept(
        stim, t_percept=t).data.ravel()
    npt.assert_array_less(lagged.max(), rectified.max())
    npt.assert_equal(np.any(lagged[rectified == 0] > 0), True)

    # For large tau, the peak scales as 1/tau:
    peaks = []
    for tau in (1e4, 1e5, 1e6):
        peaks.append(FadingTemporal(tau=tau).build().predict_percept(
            stim, t_percept=t).data.max())
    npt.assert_equal(np.all(np.diff(peaks) < 0), True)
    npt.assert_allclose(np.multiply(peaks, [1e4, 1e5, 1e6]), 50.0 * 1.0,
                        rtol=0.05)


def clocked(stim):
    """Return ``stim`` as a response on ten 50 ms encoder frames"""
    return _ModelResponse(stim.data, stim.time, ms, (len(stim.electrodes), 1),
                          frame_clock=_FrameClock(np.arange(10) * 50.0, 50.0))


def test_FadingTemporal_reduce_limits():
    """`reduce` only matters when brightness rises and falls"""
    # Constant cathodic current: brightness rises monotonically, so the peak
    # of every interval is at its end:
    rising = clocked(Stimulus(np.full((4, 2), -20.0), time=[0.0, 500.0]))
    peak = FadingTemporal(tau=100).build()._predict_response(rising)
    last = FadingTemporal(tau=100, reduce='last').build()._predict_response(
        rising)
    npt.assert_equal(peak.time.size, 10)
    npt.assert_array_equal(peak.data, last.data)
    npt.assert_array_less(-1e-9, np.diff(peak.data, axis=-1))

    # A pulse train rises and falls, so 'peak' and 'last' differ:
    train = clocked(BiphasicPulseTrain(20, -50, 0.46, stim_dur=500))
    peak = FadingTemporal(tau=100).build()._predict_response(train)
    last = FadingTemporal(tau=100, reduce='last').build()._predict_response(
        train)
    npt.assert_equal(np.any(peak.data != last.data), True)
    npt.assert_array_less(last.data - 1e-9, peak.data)


def test_AlphaTemporal():
    model = AlphaTemporal()
    npt.assert_equal(model.tau, 100)
    npt.assert_equal(model.reduce, 'peak')
    model.dt = 0.1
    npt.assert_equal(model.dt, 0.1)
    model.build(dt=1e-3)
    npt.assert_equal(model.dt, 1e-3)
    with pytest.raises(FreezeError):
        model.rho = 100

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(None), None)

    # Zero in = zero out (both stages start at zero):
    percept = model.predict_percept(BiphasicPulse(0, 1), t_percept=[0, 1, 2])
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, (1, 1, 3))
    npt.assert_almost_equal(percept.data, 0)

    with pytest.raises(ValueError):
        model.predict_percept(Stimulus(np.ones((1, 100))),
                              t_percept=[0.2, 0.2])


def test_AlphaTemporal_rectifies_the_drive():
    """Only the cathodic half of the stimulus drives the cascade."""
    model = AlphaTemporal(tau=20, thresh_percept=0).build()
    t = np.round(np.arange(0, 100, 0.5), 5)

    anodic = model.predict_percept(MonophasicPulse(1, 1, stim_dur=100),
                                   t_percept=t)
    npt.assert_array_equal(anodic.data, 0)

    cathodic = model.predict_percept(MonophasicPulse(-1, 1, stim_dur=100),
                                     t_percept=t)
    npt.assert_equal(cathodic.data.max() > 0, True)


def _alpha_reference(data, t_stim, idx_percept, dt, tau, reduce_peak=False):
    """Return the two-state explicit Euler recurrence per location, in float64

    Stage 2 reads stage 1 from the *start* of the step (this gives the rise
    delay). float64 because `alpha_fast` composes constant-drive runs into
    one update and cannot match a float32 step-by-step replay.
    """
    a = float(np.float32(np.float32(dt) / np.float32(tau)))
    n_stim = len(t_stim)
    out = np.zeros((data.shape[0], len(idx_percept)))
    for s in range(data.shape[0]):
        x = y = 0.0
        running = 0.0
        idx_stim, frame = 0, 0
        for i in range(int(idx_percept[-1]) + 1):
            while (idx_stim + 1 < n_stim and
                   np.float32(i) * np.float32(dt) >= t_stim[idx_stim + 1]):
                idx_stim += 1
            drive = max(-float(data[s, idx_stim]), 0.0)
            x_old = x
            x = x + a * (drive - x)
            y = y + a * (x_old - y)
            running = max(running, y)
            if frame < len(idx_percept) and i == idx_percept[frame]:
                out[s, frame] = running if reduce_peak else y
                running = y
                frame += 1
    return out


def test_AlphaTemporal_matches_reference_recurrence():
    """The two-state cascade matches a plain Python reference

    The stimulus straddles zero, to test half-wave rectification. Stage 2 uses
    the previous stage-1 value (otherwise the drive would pass through in one
    step, with no rise).
    """
    model = AlphaTemporal(dt=0.01, tau=50, thresh_percept=0).build()
    n_space, n_stim = 3, 5
    rng = np.random.default_rng(0)
    data = (rng.random((n_space, n_stim)) - 0.5).astype(np.float32)
    npt.assert_equal(np.any(data > 0) and np.any(data < 0), True)
    t_stim = np.arange(n_stim, dtype=np.float32) * 2.0
    t_percept = np.array([0.0, 2.0, 4.0, 8.0])
    got = model.predict_percept(Stimulus(data, time=t_stim),
                                t_percept=t_percept).data.reshape(n_space, -1)

    idx_p = np.uint32(np.round(t_percept / model.dt))
    want = _alpha_reference(data, t_stim, idx_p, model.dt, model.tau)
    # Not exact: the kernel composes each constant-drive run into one float32
    # update (within a few parts in 1e6 of float64):
    npt.assert_allclose(got, want, rtol=1e-5)

    # Stage 2 must use the previous stage-1 value:
    dt, tau = 0.01, 50.0
    step = alpha_fast(np.array([[-1.0]], dtype=np.float32),
                      np.array([0.0], dtype=np.float32),
                      np.arange(3, dtype=np.uint32), dt, tau, 0.0, 1, 0)
    npt.assert_array_equal(step[0, 0], 0)
    npt.assert_allclose(step[0, 1], (dt / tau) ** 2, rtol=1e-6)


def test_AlphaTemporal_impulse_is_alpha_shaped():
    """A brief pulse produces an alpha-shaped response

    `dt` is 200x shorter than `tau`, so the pulse is effectively an impulse
    with response `t/tau**2 exp(-t/tau)`: zero at onset, a single maximum at
    `t = tau`, monotonic on either side.
    """
    tau, dt = 20.0, 0.1
    model = AlphaTemporal(tau=tau, dt=dt, thresh_percept=0).build()
    # One `dt` step of cathodic current, on the simulation grid:
    stim = Stimulus(np.array([[0.0, -1.0, 0.0]]), time=[0.0, dt, 2 * dt])
    t = np.round(np.arange(0, 6 * tau, dt), 5)
    y = model.predict_percept(stim, t_percept=t).data.ravel()

    # Zero at onset and one step later (stage 2 uses the previous stage 1):
    npt.assert_array_equal(y[:2], 0)
    npt.assert_array_less(0, y[2:].min())
    peak = int(np.argmax(y))
    # The impulse response peaks near tau:
    npt.assert_equal(0 < peak < len(y) - 1, True)
    npt.assert_allclose(t[peak], tau, rtol=0.05)
    npt.assert_equal(np.all(np.diff(y[1:peak + 1]) > 0), True)
    # Non-strict on the way down: float32 can leave the peak flat for a step
    # or two, and `argmax` returns the first:
    npt.assert_equal(np.all(np.diff(y[peak:]) <= 0), True)
    npt.assert_array_less(y[-1], y[peak])
    # With unit DC gain, the peak is the alpha function peak scaled by the
    # impulse area (`dt`):
    npt.assert_allclose(y[peak], dt / (np.e * tau), rtol=0.02)


def test_AlphaTemporal_dc_gain_is_unity():
    """Sustained drive approaches the drive amplitude

    Both stages are unit-gain leaky integrators. (Normalizing the impulse
    response to peak at 1 would scale this by `e * tau`.)
    """
    for tau in (10.0, 100.0):
        model = AlphaTemporal(tau=tau, thresh_percept=0).build()
        drive = 3.0
        stim = Stimulus(np.array([[-drive, -drive]]), time=[0.0, 40 * tau])
        t = np.round(np.arange(0, 30 * tau, tau / 10), 5)
        # Tolerance allows float32 error over 3000 steps, but not a gain of
        # `e * tau`:
        npt.assert_allclose(
            model.predict_percept(stim, t_percept=t).data.ravel()[-1], drive,
            rtol=5e-3)


#: Constant-drive runs for the composed update, as
#: (data, t_stim, tau, t_percept). ``dt`` is 0.05 throughout.
_ALPHA_RUNS = {
    'rising': ([[-40.0, -40.0]], [0.0, 1e6], 20.0, [1.0, 10.0, 40.0]),
    'falling': ([[-40.0, 0.0]], [0.0, 3.0], 20.0, [3.0, 20.0, 60.0]),
    # Drive ends at 2 ms but stage 1 > stage 2, so brightness keeps rising
    # and peaks mid-interval:
    'interior_peak': ([[-60.0, 0.0]], [0.0, 2.0], 8.0, [2.0, 20.0, 60.0]),
    # Output points one step on either side of a drive change:
    'peak_at_transition': ([[-50.0, -1.0, 0.0]], [0.0, 4.9, 5.0], 6.0,
                           [4.85, 4.9, 4.95, 5.0, 5.05, 30.0]),
    'tau_eq_dt': ([[-30.0, 0.0, -10.0]], [0.0, 1.0, 2.0], 0.05,
                  [1.0, 2.0, 3.0]),
    # Frame times 50x closer together than one simulation step:
    'frames_inside_one_dt': ([[0.0, -90.0, 0.0, -20.0, 0.0]],
                             [0.0, 0.001, 0.002, 0.02, 9.0], 15.0,
                             [0.05, 1.0, 9.0]),
}


@pytest.mark.parametrize('case', sorted(_ALPHA_RUNS))
def test_AlphaTemporal_run_composition_matches_recurrence(case):
    """A composed constant-drive run matches the float64 recurrence

    `alpha_fast` advances both stages across a whole constant-drive run at
    once.
    """
    data, t_stim, tau, t_percept = _ALPHA_RUNS[case]
    dt = 0.05
    data = np.array(data, dtype=np.float32)
    t_stim = np.array(t_stim, dtype=np.float32)
    idx = np.uint32(np.round(np.array(t_percept) / dt))
    got = {}
    for reduce_peak in (0, 1):
        got[reduce_peak] = np.asarray(
            alpha_fast(data, t_stim, idx, dt, tau, 0.0, 1, reduce_peak))
        npt.assert_allclose(
            got[reduce_peak],
            _alpha_reference(data, t_stim, idx, dt, tau, bool(reduce_peak)),
            rtol=1e-5, atol=1e-6)
    # The interval includes its end point, so peak >= last exactly:
    npt.assert_equal(np.all(got[1] >= got[0]), True)


def test_AlphaTemporal_peak_is_exact():
    """The in-kernel peak equals the max over every simulation step

    Not bit-exact: `alpha_fast` composes steps between output points into one
    update, so dense and sparse output have different rounding.
    """
    rng = np.random.default_rng(3)
    data = (rng.random((5, 12)) - 0.5).astype(np.float32) * 40
    t_stim = (np.arange(12) * 4.0).astype(np.float32)
    dt, tau = 0.05, 20.0
    n_sim = int(round(44 / dt)) + 1
    dense = alpha_fast(data, t_stim, np.arange(n_sim, dtype=np.uint32), dt,
                       tau, 0.0, 1, 0)
    out = np.array([37, 210, 400, 601, 880], dtype=np.uint32)
    peak = alpha_fast(data, t_stim, out, dt, tau, 0.0, 1, 1)
    last = alpha_fast(data, t_stim, out, dt, tau, 0.0, 1, 0)
    # Each interval runs from the previous output point up to and including
    # this one:
    lo = np.r_[0, out[:-1]]
    brute = np.stack([dense[:, a:b + 1].max(axis=1)
                      for a, b in zip(lo, out)], axis=1)
    npt.assert_allclose(peak, brute, rtol=1e-5)
    npt.assert_allclose(last, dense[:, out], rtol=1e-5)
    npt.assert_equal(np.all(peak >= last), True)
    npt.assert_equal(np.any(peak > last), True)

    # An interval whose max is at neither end (tests the turning-point search):
    single = np.array([[-60.0, 0.0]], dtype=np.float32)
    edges = np.array([0.0, 2.0], dtype=np.float32)
    fine = np.asarray(alpha_fast(single, edges,
                                 np.arange(1201, dtype=np.uint32), dt, 8.0,
                                 0.0, 1, 0)).ravel()
    npt.assert_equal(0 < int(fine.argmax()) < 1200, True)
    span = np.array([40, 400, 1200], dtype=np.uint32)
    got = np.asarray(alpha_fast(single, edges, span, dt, 8.0, 0.0, 1,
                                1)).ravel()
    lo = np.r_[0, span[:-1]]
    npt.assert_allclose(
        got, [fine[a:b + 1].max() for a, b in zip(lo, span)], rtol=1e-5)


def test_AlphaTemporal_reduce():
    """`reduce` applies only to automatic output times, not to `t_percept`"""
    stim = BiphasicPulseTrain(20, -50, 0.46, stim_dur=200)
    # Short tau so brightness still ripples between pulses of a 20 Hz train
    # (at `tau=100` the response is a smooth ramp):
    peak_model = AlphaTemporal(tau=10).build()
    last_model = AlphaTemporal(tau=10, reduce='last').build()

    t = [0, 50, 100, 150]
    npt.assert_array_equal(peak_model.predict_percept(stim, t_percept=t).data,
                           last_model.predict_percept(stim, t_percept=t).data)

    got = last_model.predict_percept(stim)
    peaked = peak_model.predict_percept(stim)
    npt.assert_almost_equal(peaked.time, got.time)
    npt.assert_equal(np.all(peaked.data >= got.data), True)
    npt.assert_equal(np.any(peaked.data > got.data), True)


@pytest.mark.parametrize('n_space', (1, 64, 65))
def test_AlphaTemporal_block_boundaries(n_space):
    """Locations are integrated in fixed-size blocks, the last one partial

    `alpha_fast` has its own block handling (two states per location).
    """
    model = AlphaTemporal(dt=0.05, tau=30).build()
    rng = np.random.default_rng(n_space)
    data = (rng.random((n_space, 4)) - 0.7).astype(np.float32)
    stim = Stimulus(data, time=np.arange(4, dtype=float) * 5)
    t = [0, 5, 10, 15]
    percept = model.predict_percept(stim, t_percept=t)
    npt.assert_equal(percept.data.shape, (n_space, 1, 4))
    single = np.stack([
        model.predict_percept(Stimulus(data[i:i + 1], time=stim.time),
                              t_percept=t).data.ravel()
        for i in range(n_space)])
    npt.assert_array_equal(percept.data.reshape(n_space, -1), single)
    # The result does not depend on the number of threads:

    for n_threads in (2, 3, 8):
        parallel = AlphaTemporal(dt=0.05, tau=30, n_threads=n_threads).build(
        ).predict_percept(stim, t_percept=t).data
        npt.assert_array_equal(parallel, percept.data)
