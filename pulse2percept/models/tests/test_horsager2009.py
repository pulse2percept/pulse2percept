import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.implants import ProsthesisSystem, PointSource
from pulse2percept.stimuli import Stimulus, BiphasicPulse, BiphasicPulseTrain
from pulse2percept.percepts import Percept
from pulse2percept.models import Horsager2009Model, Horsager2009Temporal
from pulse2percept.utils import FreezeError


def test_Nanduri2012Temporal():
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
    implant = ProsthesisSystem(PointSource(0, 0, 0))
    npt.assert_equal(model.predict_percept(implant.stim), None)

    # Zero in = zero out:
    implant.stim = np.zeros((1, 6))
    percept = model.predict_percept(implant.stim, t_percept=[0, 1, 2])
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, (1, 1, 3))
    npt.assert_almost_equal(percept.data, 0)

    # Can't request the same time more than once (this would break the Cython
    # loop, because `idx_frame` is incremented after a write; also doesn't
    # make much sense):
    with pytest.raises(ValueError):
        implant.stim = np.ones((1, 100))
        model.predict_percept(implant.stim, t_percept=[0.2, 0.2])

    # Single-pulse brightness from Fig.3:
    model = Horsager2009Temporal().build()
    for amp, pdur in zip([188.077, 89.74, 10.55], [0.075, 0.15, 4.0]):
        stim = BiphasicPulse(amp, pdur, interphase_dur=pdur, stim_dur=200,
                             cathodic_first=True)
        npt.assert_almost_equal(model.predict_percept(stim).data.max(), 110.3,
                                decimal=2)

    # Fixed-duration brightness from Fig.4:
    model = Horsager2009Temporal().build()
    for amp, freq in zip([136.02, 120.35, 57.71], [5, 15, 225]):
        stim = BiphasicPulseTrain(freq, amp, 0.075, interphase_dur=0.075,
                                  stim_dur=200, cathodic_first=True)
        npt.assert_almost_equal(model.predict_percept(stim).data.max(), 36.3,
                                decimal=2)


def test_Horsager2009Model():
    model = Horsager2009Model()
    npt.assert_equal(hasattr(model, 'has_space'), True)
    npt.assert_equal(model.has_space, False)
    npt.assert_equal(hasattr(model, 'has_time'), True)
    npt.assert_equal(model.has_time, True)

    # User can set `dt`:
    model.temporal.dt = 1e-5
    npt.assert_almost_equal(model.dt, 1e-5)
    npt.assert_almost_equal(model.temporal.dt, 1e-5)
    model.build(dt=3e-4)
    npt.assert_almost_equal(model.dt, 3e-4)
    npt.assert_almost_equal(model.temporal.dt, 3e-4)

    # User cannot add more model parameters:
    with pytest.raises(FreezeError):
        model.rho = 100


def test_Horsager2009Model():
    # Model and TemporalModel give the same result
    for amp, freq in zip([136.02, 120.35, 57.71], [5, 15, 225]):
        stim = BiphasicPulseTrain(freq, amp, 0.075, interphase_dur=0.075,
                                  stim_dur=200, cathodic_first=True)
        model1 = Horsager2009Model().build()
        model2 = Horsager2009Temporal().build()
        implant = ProsthesisSystem(PointSource(0, 0, 0), stim=stim)
        npt.assert_almost_equal(model1.predict_percept(implant).data,
                                model2.predict_percept(stim).data)
