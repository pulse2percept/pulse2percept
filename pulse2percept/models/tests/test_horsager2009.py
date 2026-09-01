import copy

import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.implants import Implant, PointSource
from pulse2percept.stimuli import (BiphasicPulse, BiphasicPulseTrain,
                                   Stimulus)
from pulse2percept.percepts import Percept
from pulse2percept.models import Horsager2009Model, Horsager2009Temporal
from pulse2percept.utils import FreezeError


def test_Horsager2009Temporal():
    model = Horsager2009Temporal()
    # User can set their own params:
    model.dt = 0.1
    npt.assert_equal(model.dt, 0.1)
    model.build(dt=1e-4)
    npt.assert_equal(model.dt, 1e-4)
    # User cannot add more model parameters:
    with pytest.raises(FreezeError):
        model.rho = 100

    # Nothing in, None out:
    implant = Implant(PointSource(0, 0, 0))
    npt.assert_equal(model.predict_percept(implant.prepare_stim(None)),
                     None)

    # Zero in = zero out:
    percept = model.predict_percept(implant.prepare_stim(np.zeros((1, 6))),
                                    t_percept=[0, 1, 2])
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, (1, 1, 3))
    npt.assert_almost_equal(percept.data, 0)

    # Can't request the same time more than once (this would break the Cython
    # loop, because `idx_frame` is incremented after a write; also doesn't
    # make much sense):
    with pytest.raises(ValueError):
        model.predict_percept(implant.prepare_stim(np.ones((1, 100))),
                              t_percept=[0.2, 0.2])

    # Single-pulse brightness from Fig.3. These three (amp, phase_dur) pairs sit
    # on one threshold curve in the paper, so the model should call them equally
    # bright. It does so only approximately -- they come out within about 3% of
    # each other at this `dt`, and the shortest pulse is the one that strays.
    #
    # They used to agree to five figures, but that was a numerical accident:
    # until 0.10.0 the kernel advanced only one stimulus frame per simulation
    # step, and at dt=5e-3 ms a pulse edge and the sample after it share a step.
    # That stretched every phase by about one step, which is 7% of a 0.075 ms
    # phase and 0.1% of a 4 ms one -- inflating exactly the short conditions
    # that fall short here. Refining `dt` does not bring the agreement back; the
    # three converge to 101.0, 105.6 and 110.1, so the spread is a property of
    # the model as implemented, not of the time step.
    model = Horsager2009Temporal().build()
    bright = []
    for amp, pdur in zip([188.077, 89.74, 10.55], [0.075, 0.15, 4.0]):
        stim = BiphasicPulse(amp, pdur, interphase_dur=pdur, stim_dur=200,
                             cathodic_first=True)
        t_percept = np.arange(0, stim.time[-1] + model.dt / 2, model.dt)
        percept = model.predict_percept(stim, t_percept=t_percept)
        bright.append(percept.data.max())
    npt.assert_allclose(bright, np.mean(bright), rtol=0.03)
    npt.assert_almost_equal(bright, [107.20, 109.31, 110.13], decimal=2)

    # Fixed-duration brightness from Fig.4, which again puts these three
    # conditions at one threshold. As with Fig.3 above, the model reproduces
    # that to a few percent rather than exactly, and the agreement was closer
    # before 0.10.0 only because the kernel was stretching every phase by about
    # one simulation step. These phases are 0.075 ms, so a 5e-3 ms step is 7% of
    # one, and the correction moves the three by different amounts.
    model = Horsager2009Temporal().build()
    bright = []
    for amp, freq in zip([136.01, 120.34, 57.73], [5, 15, 225]):
        stim = BiphasicPulseTrain(freq, amp, 0.075, interphase_dur=0.075,
                                  stim_dur=200, cathodic_first=True)
        t_percept = np.arange(0, stim.time[-1] + model.dt / 2, model.dt)
        percept = model.predict_percept(stim, t_percept=t_percept)
        bright.append(percept.data.max())
    npt.assert_allclose(bright, np.mean(bright), rtol=0.03)
    npt.assert_almost_equal(bright, [35.27, 36.21, 35.49], decimal=2)


def test_deepcopy_Horsager2009Temporal():
    original = Horsager2009Temporal()
    copied = copy.deepcopy(original)

    # Assert these are two different objects
    npt.assert_equal(id(original) != id(copied), True)

    # Assert the objects are equivalent
    npt.assert_equal(original.__dict__, copied.__dict__)
    npt.assert_equal(original == copied, True)

    # Assert changing the original doesn't affect the copied
    original.verbose = False
    npt.assert_equal(original != copied, True)


def test_Horsager2009Model():
    model = Horsager2009Model()
    npt.assert_equal(hasattr(model, 'has_space'), True)
    npt.assert_equal(model.has_space, False)
    npt.assert_equal(hasattr(model, 'has_time'), True)
    npt.assert_equal(model.has_time, True)

    # User can set `dt`:
    model.temporal.dt = 1e-5
    npt.assert_almost_equal(model.temporal.dt, 1e-5)
    model.temporal.build(dt=3e-4)
    npt.assert_almost_equal(model.temporal.dt, 3e-4)

    # User cannot add more model parameters:
    with pytest.raises(FreezeError):
        model.temporal.rho = 100

    # Model and TemporalModel give the same result
    for amp, freq in zip([136.02, 120.35, 57.71], [5, 15, 225]):
        stim = BiphasicPulseTrain(freq, amp, 0.075, interphase_dur=0.075,
                                  stim_dur=200, cathodic_first=True)
        model1 = Horsager2009Model().build()
        model2 = Horsager2009Temporal().build()
        npt.assert_almost_equal(model1.predict_percept(stim).data,
                                model2.predict_percept(stim).data)


def test_deepcopy_Horsager2009Model():
    original = Horsager2009Model()
    copied = copy.deepcopy(original)

    # Assert these are two different objects
    npt.assert_equal(id(original) != id(copied), True)

    # Assert the objects are equivalent
    npt.assert_equal(original == copied, True)

    # Assert changing the original doesn't affect the copied
    original.temporal.verbose = False
    npt.assert_equal(original != copied, True)

def _horsager_reference(data, t_stim, t_percept, dt, tau1, tau2, tau3, eps,
                        beta, thresh_percept):
    """The cascade, written out one location and one step at a time."""
    f = np.float32
    dt, tau1, tau2, tau3 = f(dt), f(tau1), f(tau2), f(tau3)
    beta, thresh = f(beta), f(thresh_percept)
    eps = f(f(eps) / f(1000.0))
    idx_p = np.uint32(np.round(np.asarray(t_percept) / dt))
    out = np.zeros((data.shape[0], len(idx_p)), dtype=np.float32)
    # A negative `beta` makes the nonlinearity infinite wherever the rectifier
    # zeroed its input, and the cascade below then subtracts inf from inf. The
    # kernel produces the same NaN, which is what this reference is for:
    with np.errstate(invalid='ignore', divide='ignore'):
        for s in range(data.shape[0]):
            ca = r1 = r2 = r4a = r4b = r4c = f(0.0)
            idx_stim, frame = 0, 0
            for i in range(int(idx_p[-1]) + 1):
                if (idx_stim + 1 < len(t_stim) and
                        f(i) * dt >= t_stim[idx_stim + 1]):
                    idx_stim += 1
                amp = data[s, idx_stim]
                r1 = f(r1 + dt * (-amp - r1) / tau1)
                ca = f(ca + dt * (amp if amp > 0 else f(0.0)))
                r2 = f(r2 + dt * (ca - r2) / tau2)
                # `pow(0, beta)` is 1 at beta == 0 and inf below it, not 0:
                r3 = f(np.float_power(f(max(r1 - eps * r2, f(0.0))), beta))
                r4a = f(r4a + dt * (r3 - r4a) / tau3)
                r4b = f(r4b + dt * (r4a - r4b) / tau3)
                r4c = f(r4c + dt * (r4b - r4c) / tau3)
                if i == idx_p[frame]:
                    out[s, frame] = r4c if abs(r4c) >= thresh else f(0.0)
                    frame += 1
    return out


@pytest.mark.parametrize('beta', (3.43, 1.0, 0.0, -0.5))
def test_Horsager2009Temporal_power_nonlinearity(beta):
    """The power nonlinearity must hold up for any exponent.

    The kernel skips ``powf`` wherever the half-wave rectifier in front of it
    has already zeroed the argument, which it does on the overwhelming
    majority of steps. What it substitutes has to be ``pow(0, beta)``, which
    is 1 at ``beta == 0`` and infinite for negative ``beta`` -- not zero.
    """
    rng = np.random.default_rng(0)
    data = ((rng.random((3, 6)) - 0.5) * 100).astype(np.float32)
    t_stim = (np.arange(6) * 4.0).astype(np.float32)
    t_percept = np.arange(0, 20, 2.0)
    model = Horsager2009Temporal(dt=0.01, beta=beta).build()
    got = model.predict_percept(Stimulus(data, time=t_stim),
                                t_percept=t_percept).data.reshape(3, -1)
    want = _horsager_reference(data, t_stim, t_percept, model.dt, model.tau1,
                               model.tau2, model.tau3, model.eps, beta,
                               model.thresh_percept)
    npt.assert_allclose(got, want, rtol=1e-5, equal_nan=True)


def test_Horsager2009Temporal_beta_zero_is_not_zero():
    """At beta == 0 the nonlinearity returns 1 even where its input is zero.

    ``x ** 0`` is 1 for every ``x``, including the zero the half-wave
    rectifier produces, so the cascade charges towards 1 no matter how small
    the stimulus. A shortcut that assumed ``pow(0, beta) == 0`` would silence
    the model here; this pins that it does not. The stimulus is tiny but
    nonzero, because an all-zero one is dropped before the kernel sees it.
    """
    stim = Stimulus(np.full((1, 2), 1e-6, dtype=np.float32),
                    time=np.array([0.0, 400.0]))
    percept = Horsager2009Temporal(dt=0.01, beta=0.0).build().predict_percept(
        stim, t_percept=[0, 100, 200, 400])
    # Driven by r3 == 1 throughout, the three-stage integrator climbs towards
    # 1 rather than tracking the (negligible) stimulus:
    npt.assert_array_less(0.5, percept.data[0, 0, -1])
    npt.assert_array_less(percept.data[0, 0, -1], 1.0)
    # ...and it is monotonically charging, not decaying:
    npt.assert_array_less(percept.data[0, 0, :-1], percept.data[0, 0, 1:])
