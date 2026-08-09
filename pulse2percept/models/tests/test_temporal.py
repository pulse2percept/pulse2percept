import numpy as np
import copy
import warnings
import numpy.testing as npt
import pytest

from pulse2percept.models import FadingTemporal, Nanduri2012Temporal
from pulse2percept.stimuli import Stimulus, MonophasicPulse, BiphasicPulse
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

    # Simple decay for single cathodic pulse:
    model = FadingTemporal(tau=1).build()
    stim = MonophasicPulse(-1, 1, stim_dur=10)
    percept = model.predict_percept(stim, np.arange(stim.duration))
    npt.assert_almost_equal(percept.data.ravel()[:3], [0, 0.633, 0.232],
                            decimal=3)
    npt.assert_almost_equal(percept.data.ravel()[-1], 0, decimal=3)

    # But all zeros for anodic pulse:
    stim = MonophasicPulse(1, 1, stim_dur=10)
    percept = model.predict_percept(stim, np.arange(stim.duration))
    npt.assert_almost_equal(percept.data, 0)

    # tau cannot be negative:
    with pytest.raises(ValueError):
        FadingTemporal(tau=-1).build()


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
    agrees exactly.
    """
    model = FadingTemporal(dt=0.01, tau=50, thresh_percept=0).build()
    n_space, n_stim = 3, 5
    rng = np.random.default_rng(0)
    data = (rng.random((n_space, n_stim)) - 0.5).astype(np.float32)
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
            if idx_stim + 1 < n_stim and np.float32(i) * dt >= t_stim[idx_stim + 1]:
                idx_stim += 1
            amp = data[s, idx_stim]
            bright = np.float32(bright + dt * (-amp - bright) / tau)
            if bright < 0:
                bright = np.float32(0.0)
            if i == idx_p[frame]:
                npt.assert_array_equal(got[s, frame], bright)
                frame += 1


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
