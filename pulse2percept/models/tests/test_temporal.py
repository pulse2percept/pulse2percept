import numpy as np
import copy
import warnings
import numpy.testing as npt
import pytest

from pulse2percept.models import FadingTemporal, Nanduri2012Temporal
from pulse2percept.models._temporal import fading_fast
from pulse2percept.stimuli import (Stimulus, MonophasicPulse, BiphasicPulse,
                                   BiphasicPulseTrain)
from pulse2percept.percepts import Percept
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

    # Can't request the same time more than once (this would break the Cython
    # loop, because `idx_frame` is incremented after a write; also doesn't
    # make much sense):
    with pytest.raises(ValueError):
        stim = Stimulus(np.ones((1, 100)))
        model.predict_percept(stim, t_percept=[0.2, 0.2])

    # Simple decay for single cathodic pulse. The pulse carries current from
    # t=DT to t=1, so sample-and-hold on the dt=5e-3 ms grid drives the
    # integrator for 0.995 ms rather than a round 1 ms -- which is why this
    # sits just under the 1-exp(-1) = 0.632 an ideal rectangle would give:
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

    # tau has to be at least one simulation step. Zero and negatives divide the
    # integrator by zero; anything under `dt` makes it overshoot its drive by
    # dt/tau and then oscillate, so at tau=dt/2 brightness alternates between
    # twice the drive and nothing. All of these used to build happily:
    for tau in (-1, 0, 0.005 / 2, 0.004):
        with pytest.raises(ValueError):
            FadingTemporal(tau=tau, dt=0.005).build()
    # ... and exactly one step is fine, being the rectifier limit:
    FadingTemporal(tau=0.005, dt=0.005).build()


def test_deepcopy_FadingTemporal():
    original = FadingTemporal()
    copied = copy.deepcopy(original)

    # Assert they are different objects
    npt.assert_equal(id(original) != id(copied), True)

    # Assert the objects are equivalent to each other
    npt.assert_equal(original == copied, True)

    # Assert building one object does not affect the copied
    original.build()
    npt.assert_equal(copied.is_built, False)
    npt.assert_equal(original != copied, True)

    # which should be unique to each SpatialModel object
    copied = copy.deepcopy(original)
    copied.verbose = False
    npt.assert_equal(original.verbose, True)
    npt.assert_equal(original != copied, True)

    # Assert "destroying" the original doesn't affect the copied
    original = None
    npt.assert_equal(copied is not None, True)

def test_FadingTemporal_matches_reference_integrator():
    """Pin the leaky integrator against a plain Python reference.

    ``fading_fast`` runs time in the outer loop and space in the inner one so
    that the inner loop vectorizes. This walks a couple of locations the
    obvious way -- one at a time, stepping forward -- and checks the kernel
    agrees exactly. The stimulus straddles zero, so it also pins the half-wave
    rectification: anodic samples must contribute nothing at all rather than
    driving brightness down.
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
    idx_p = np.uint32(np.round(t_percept / model.dt))
    for s in range(n_space):
        bright = np.float32(0.0)
        idx_stim, frame = 0, 0
        for i in range(int(idx_p[-1]) + 1):
            # Advance until caught up, not by one frame per step: several
            # stimulus frames can fall inside one `dt`. See
            # `test_FadingTemporal_frames_closer_together_than_dt`.
            while (idx_stim + 1 < n_stim and
                   np.float32(i) * dt >= t_stim[idx_stim + 1]):
                idx_stim += 1
            amp = data[s, idx_stim]
            drive = np.float32(max(-amp, 0.0))
            bright = np.float32(bright + dt * (drive - bright) / tau)
            if bright < 0:
                bright = np.float32(0.0)
            if i == idx_p[frame]:
                npt.assert_array_equal(got[s, frame], bright)
                frame += 1


def test_FadingTemporal_rectifies_the_drive():
    """A charge-balanced pulse train has to produce a percept that persists.

    A leaky integrator is linear, so driving it with -A lets the anodic phase
    of a biphasic pulse undo exactly what the cathodic phase did: brightness
    spikes for the length of one phase and returns to zero, leaving a 1.8%
    duty cycle rather than a phosphene. The pulse still delivers zero net
    charge either way; rectifying is what stops the model's *drive* from
    cancelling along with it.
    """
    train = BiphasicPulseTrain(20, -50, 0.46, stim_dur=1000)
    model = FadingTemporal(tau=100).build()
    t = np.round(np.arange(0, 1000, 0.05), 5)
    bright = model.predict_percept(train, t_percept=t).data.ravel()
    late = bright[t >= 500]
    # Brightness holds up between pulses instead of collapsing to zero. The
    # unrectified model came back to within 0.6% of zero after every pulse:
    npt.assert_array_less(0.5, late.min() / late.max())
    # ... and it accumulates over the train rather than repeating one transient:
    # even at its dimmest the steady state is above what one pulse produces:
    one_pulse = bright[t < 50].max()
    npt.assert_array_less(one_pulse, late.min())
    npt.assert_array_less(2 * one_pulse, late.max())

    # A slower train is dimmer, because it spends longer fading between pulses.
    # This is the whole basis of frequency modulation, and the unrectified
    # model could not express it:
    def steady(freq):
        stim = BiphasicPulseTrain(freq, -50, 0.46, stim_dur=1000)
        return model.predict_percept(stim, t_percept=t).data.ravel()[-1]

    rates = [10, 20, 50, 100]
    npt.assert_equal(np.all(np.diff([steady(f) for f in rates]) > 0), True)

    # Rectification only removes the anodic half, so a stimulus that never goes
    # anodic is unaffected by it:
    model = FadingTemporal(tau=1).build()
    percept = model.predict_percept(MonophasicPulse(-1, 1, stim_dur=10),
                                    np.arange(10))
    npt.assert_almost_equal(percept.data.ravel()[:3], [0, 0.628, 0.230],
                            decimal=3)


@pytest.mark.parametrize('n_space', (1, 63, 64, 65, 130))
def test_FadingTemporal_block_boundaries(n_space):
    """Locations are integrated in fixed-size blocks; the last one is partial.

    Sizes either side of the block width catch an off-by-one in how the tail
    of the last block is handled.
    """
    model = FadingTemporal(dt=0.05, tau=30).build()
    rng = np.random.default_rng(n_space)
    data = (rng.random((n_space, 4)) - 0.7).astype(np.float32)
    stim = Stimulus(data, time=np.arange(4, dtype=float) * 5)
    percept = model.predict_percept(stim, t_percept=[0, 5, 10, 15])
    npt.assert_equal(percept.data.shape, (n_space, 1, 4))
    # Every location must be integrated, not just the ones in whole blocks:
    single = np.stack([
        model.predict_percept(Stimulus(data[i:i + 1], time=stim.time),
                              t_percept=[0, 5, 10, 15]).data.ravel()
        for i in range(n_space)])
    npt.assert_array_equal(percept.data.reshape(n_space, -1), single)


def test_FadingTemporal_thread_count_invariant():
    """Threads take a block of locations each; the result must not depend on
    how many of them there are."""
    rng = np.random.default_rng(7)
    data = (rng.random((200, 6)) - 0.6).astype(np.float32)
    stim = Stimulus(data, time=np.arange(6, dtype=float) * 3)
    serial = FadingTemporal(dt=0.05, tau=40, n_threads=1).build(
        ).predict_percept(stim, t_percept=[0, 5, 10, 15]).data
    for n_threads in (2, 3, 8):
        parallel = FadingTemporal(dt=0.05, tau=40, n_threads=n_threads).build(
            ).predict_percept(stim, t_percept=[0, 5, 10, 15]).data
        npt.assert_array_equal(parallel, serial)


def test_FadingTemporal_peak_is_exact():
    """The in-kernel peak must equal a dense scan of the same interval.

    Sampling an interval a fixed number of times cannot summarize a transient
    shorter than its own step, which is why the peak is tracked across every
    `dt` step instead. That makes it exact however coarse the output rate is,
    and this pins it against brightness computed at every single step.
    """
    rng = np.random.default_rng(3)
    data = (rng.random((5, 12)) - 0.5).astype(np.float32) * 40
    t_stim = (np.arange(12) * 4.0).astype(np.float32)
    dt, tau = 0.05, 20.0
    # Brightness at every single simulation step:
    n_sim = int(round(44 / dt)) + 1
    dense = fading_fast(data, t_stim, np.arange(n_sim, dtype=np.uint32), dt,
                        tau, 0.0, 1, 0)
    out = np.array([37, 210, 400, 601, 880], dtype=np.uint32)
    peak = fading_fast(data, t_stim, out, dt, tau, 0.0, 1, 1)
    last = fading_fast(data, t_stim, out, dt, tau, 0.0, 1, 0)
    # The interval a percept point summarizes runs from the previous one up to
    # and including it -- brightness is continuous, so the value carried across
    # the boundary is a floor on what the next interval reaches:
    lo = np.r_[0, out[:-1]]
    brute = np.stack([dense[:, a:b + 1].max(axis=1)
                      for a, b in zip(lo, out)], axis=1)
    npt.assert_array_equal(peak, brute)
    # Reducing to the closing instant is what the model always did:
    npt.assert_array_equal(last, dense[:, out])
    npt.assert_equal(np.all(peak >= last), True)
    npt.assert_equal(np.any(peak > last), True)
    # Threads take a block of locations each; the peak must not depend on how
    # many of them there are:
    for n_threads in (2, 4, 8):
        npt.assert_array_equal(
            fading_fast(np.tile(data, (40, 1)), t_stim, out, dt, tau, 0.0,
                        n_threads, 1),
            np.tile(peak, (40, 1)))


def test_FadingTemporal_reduce():
    """`reduce` governs the times the model picks, not the ones you name."""
    stim = BiphasicPulseTrain(20, -50, 0.46, stim_dur=200)
    peak_model = FadingTemporal(tau=100).build()
    npt.assert_equal(peak_model.reduce, 'peak')
    last_model = FadingTemporal(tau=100, reduce='last').build()

    # Naming `t_percept` asks for those instants, whatever `reduce` says:
    t = [0, 50, 100, 150]
    npt.assert_array_equal(peak_model.predict_percept(stim, t_percept=t).data,
                           last_model.predict_percept(stim, t_percept=t).data)

    # Letting the model pick, `reduce='last'` is the old behaviour exactly:
    got = last_model.predict_percept(stim)
    npt.assert_array_equal(
        got.data, last_model.predict_percept(stim, t_percept=got.time).data)
    # ... and 'peak' is never below it, because the interval contains its end:
    peaked = peak_model.predict_percept(stim)
    npt.assert_almost_equal(peaked.time, got.time)
    npt.assert_equal(np.all(peaked.data >= got.data), True)
    npt.assert_equal(np.any(peaked.data > got.data), True)

    with pytest.raises(ValueError):
        FadingTemporal(reduce='mean').build().predict_percept(stim)


def test_FadingTemporal_frames_closer_together_than_dt():
    """Several stimulus frames can fall inside one simulation step.

    Encoded pulses put their edges on the DT=1e-3 ms grid while `dt` defaults
    to 5e-3 ms, so this is the normal case rather than a corner case. A kernel
    that advances one stimulus frame per simulation step falls behind and
    integrates current the stimulus no longer carries.
    """
    # A 0.1 ms cathodic blip that begins and ends strictly between two
    # simulation steps. Sample-and-hold at t=0.5 has to read the frame that is
    # current *then* -- amplitude 0 -- so the blip is never seen at all:
    t_stim = np.array([0.0, 0.1, 0.2, 0.3, 10.0], dtype=np.float32)
    data = np.array([[0.0, -100.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    dt = 0.5
    idx = np.arange(0, 21, dtype=np.uint32)
    got = fading_fast(data, t_stim, idx, dt, 100.0, 0.0, 1, 0).ravel()
    npt.assert_array_equal(got, np.zeros_like(got))

    # More generally, the frame in force at each step is what `searchsorted`
    # says it is, however many frames went by since the last one:
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
    npt.assert_equal(np.any(np.diff(frame) > 1), True)  # the case under test
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
    """`reduce='peak'` has to mean something for a model that cannot reduce.

    `FadingTemporal` tracks the peak inside its own integrator, but the
    published models cannot, so `predict_percept` samples each interval
    several times over and keeps the largest. That fallback used to be wired
    only into the encoder frame clock: for an ordinary stimulus the output
    grid was built by a different branch, no subsampling happened, and
    `reduce='peak'` silently meant `'last'`.
    """
    # No encoder metadata, so this takes the plain 20 ms output grid:
    stim = BiphasicPulseTrain(20, 50, 0.46, stim_dur=200)
    npt.assert_equal(Nanduri2012Temporal()._reduces_intervals, False)
    peak = Nanduri2012Temporal(reduce='peak').build().predict_percept(stim)
    last = Nanduri2012Temporal(reduce='last').build().predict_percept(stim)
    npt.assert_almost_equal(peak.time, np.arange(0, 201, 20))
    npt.assert_almost_equal(peak.time, last.time)
    npt.assert_equal(np.any(peak.data != last.data), True)
    # The samples land on the output point too, so the peak of an interval is
    # never below the instant it ends on:
    npt.assert_array_less(last.data - 1e-7, peak.data)

    # Sampling is only an approximation of the peak, but it may not overshoot
    # the true one, which brightness at every single `dt` step gives:
    model = Nanduri2012Temporal(reduce='peak').build()
    dense = model.predict_percept(
        stim, t_percept=np.round(np.arange(0, 181, model.dt), 5)).data.ravel()
    idx = np.round(peak.time / model.dt).astype(int)
    true = np.array([dense[max(0, a):b + 1].max()
                     for a, b in zip(np.r_[0, idx[:-1]], idx)])
    npt.assert_array_less(peak.data.ravel() - 1e-7, true)
    npt.assert_allclose(peak.data.ravel(), true, atol=0.01)

    # A published model keeps reporting what it always did unless asked:
    npt.assert_equal(Nanduri2012Temporal().reduce, 'last')
    npt.assert_array_equal(
        Nanduri2012Temporal().build().predict_percept(stim).data, last.data)
    # The generic model is the one that opts in:
    npt.assert_equal(FadingTemporal().reduce, 'peak')


def test_TemporalModel_blank_percept_warning():
    # Brightness in FadingTemporal is driven by cathodic (negative) current,
    # so an all-positive stimulus -- which is what assigning a grayscale image
    # or video directly to `implant.stim` produces -- integrates away to
    # nothing. That used to be silent:
    anodic = Stimulus(np.ones((4, 10)), time=np.arange(10) * 10.0)
    model = FadingTemporal().build()
    with pytest.warns(UserWarning, match='all-zero percept'):
        percept = model.predict_percept(anodic, t_percept=[0, 20, 40])
    npt.assert_almost_equal(percept.data, 0)

    # Flipping the polarity is what the warning suggests, and it works:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        cathodic = model.predict_percept(Stimulus(-anodic.data,
                                                  time=anodic.time),
                                         t_percept=[0, 20, 40])
    npt.assert_equal(cathodic.data.max() > 0, True)

    # A stimulus that does contain cathodic current but is simply too weak to
    # see does not get the polarity warning, because that is not its problem:
    weak = Stimulus(np.full((4, 10), -1e-12), time=np.arange(10) * 10.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        model.predict_percept(weak, t_percept=[0, 20, 40])

    # Nanduri2012Temporal runs on the opposite convention, so the same anodic
    # stimulus is the *right* polarity for it:
    nanduri = Nanduri2012Temporal().build()
    npt.assert_equal(nanduri._drive_sign, 1)
    npt.assert_equal(FadingTemporal()._drive_sign, -1)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        nanduri.predict_percept(anodic, t_percept=[0, 20, 40])


def test_FadingTemporal_tau_limits():
    """The two limiting cases of the single time constant.

    `tau` sets how fast brightness chases its drive, and it sets the rise and
    the decay together -- there is only the one constant. So the limits are not
    the ones intuition suggests:

    *  `tau` at the simulation step is the *no dynamics* limit. One step is
       then a full time constant, brightness reaches its drive within a single
       step, and the model degenerates into a plain half-wave rectifier: the
       percept is the cathodic part of the stimulus and nothing else.
    *  `tau` to infinity is *not* the same thing. Brightness never fades, but
       it never charges either, and the percept vanishes as `1/tau`. A model
       that holds a percept forever would need the decay decoupled from the
       rise, which this one cannot express.
    """
    dt = 0.005
    # Edges on the `dt` grid, so sample-and-hold is unambiguous:
    stim = Stimulus(np.array([[0.0, -50.0, 0.0, 30.0, 0.0]]),
                    time=[0.0, 1.0, 2.0, 3.0, 4.0])
    t = np.round(np.arange(0, 4, dt), 5)
    rectified = np.maximum(-np.asarray(stim.data).ravel(), 0)[
        np.searchsorted(np.asarray(stim.time), t, side='right') - 1]

    got = FadingTemporal(tau=dt, dt=dt).build().predict_percept(
        stim, t_percept=t).data.ravel()
    npt.assert_array_equal(got, rectified)
    # The anodic half is gone rather than inverted, so this really is a
    # rectifier and not a pass-through:
    npt.assert_equal(np.any(np.asarray(stim.data) > 0), True)
    npt.assert_equal(got.max(), 50.0)

    # Slower than that and brightness lags its drive, so the two part company.
    # Lagging cuts both ways: brightness never catches the drive while it is on,
    # and it is still on its way down once the drive has gone:
    lagged = FadingTemporal(tau=10 * dt, dt=dt).build().predict_percept(
        stim, t_percept=t).data.ravel()
    npt.assert_array_less(lagged.max(), rectified.max())
    npt.assert_equal(np.any(lagged[rectified == 0] > 0), True)

    # The other end: the percept dies away as 1/tau, so `tau` cannot be used to
    # make a percept persist -- it only makes it dimmer.
    peaks = []
    for tau in (1e4, 1e5, 1e6):
        peaks.append(FadingTemporal(tau=tau).build().predict_percept(
            stim, t_percept=t).data.max())
    npt.assert_equal(np.all(np.diff(peaks) < 0), True)
    npt.assert_allclose(np.multiply(peaks, [1e4, 1e5, 1e6]), 50.0 * 1.0,
                        rtol=0.05)


def test_FadingTemporal_reduce_limits():
    """`reduce` can only matter where brightness rises and falls."""
    # Constant cathodic current: brightness climbs toward it and never turns
    # around, so the peak of every interval is the instant it ends on and the
    # two settings have nothing to disagree about.
    rising = Stimulus(np.full((4, 2), -20.0), time=[0.0, 500.0])
    rising.metadata['encoder'] = {'frame_time': np.arange(10) * 50.0,
                                  'frame_dur': 50.0, 'n_frames': 10}
    peak = FadingTemporal(tau=100).build().predict_percept(rising)
    last = FadingTemporal(tau=100, reduce='last').build().predict_percept(
        rising)
    npt.assert_array_equal(peak.data, last.data)
    npt.assert_array_less(-1e-9, np.diff(peak.data, axis=-1))

    # A pulse train is the case they were built to disagree about:
    train = BiphasicPulseTrain(20, -50, 0.46, stim_dur=500)
    train.metadata['encoder'] = {'frame_time': np.arange(10) * 50.0,
                                 'frame_dur': 50.0, 'n_frames': 10}
    peak = FadingTemporal(tau=100).build().predict_percept(train)
    last = FadingTemporal(tau=100, reduce='last').build().predict_percept(
        train)
    npt.assert_equal(np.any(peak.data != last.data), True)
    npt.assert_array_less(last.data - 1e-9, peak.data)
