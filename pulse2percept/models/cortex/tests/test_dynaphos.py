import numpy.testing as npt
import pytest
import numpy as np
import copy
import matplotlib.pyplot as plt

from pulse2percept.models.cortex import DynaphosModel
from pulse2percept.implants import ProsthesisSystem, ElectrodeArray, DiskElectrode
from pulse2percept.implants.cortex import Cortivis, Orion
from pulse2percept.topography import Polimeni2006Map
from pulse2percept.percepts import Percept
from pulse2percept.stimuli import BiphasicPulseTrain
from pulse2percept.units import (DimensionMismatchError, Quantity, mA,
                                 ms, s, uA, um)

def test_DynaphosModel():
    model = DynaphosModel(xrange=(-3, 3), yrange=(-3, 3), xystep=0.1).build()

    npt.assert_equal(model.regions, ['v1'])
    npt.assert_equal(model.vfmap.regions, ['v1'])

    # can't set frequency/pulse dur that don't match up
    with pytest.raises(ValueError):
        model.build(freq=300,p_dur=10)

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(Cortivis()), None)

    implant = Cortivis(x=1000, stim={e:BiphasicPulseTrain(freq=300,amp=0,phase_dur=1) for e in Cortivis().electrode_names})
    # Zero in = zero out:
    percept = model.predict_percept(implant)
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, list(model.grid.x.shape)+[51]) # 51 time points
    npt.assert_almost_equal(percept.data, 0)

    # Can't pass stimulus with no time component
    with pytest.raises(ValueError):
        model.predict_percept(Cortivis(stim=[300 for e in Cortivis().electrode_names]))

def test_predict_spatial():
    # test that no current can spread between hemispheres
    model = DynaphosModel(xrange=(-3, 3), yrange=(-3, 3), xystep=0.5).build()
    implant = Orion(x = 15000)
    implant.stim = {e:BiphasicPulseTrain(freq=300,amp=2000,phase_dur=0.17) for e in implant.electrode_names}
    # Check brightest frame of percept
    percept = model.predict_percept(implant).max(axis='frames')
    half = percept.shape[1] // 2
    npt.assert_equal(np.all(percept[:, half+1:] == 0), True)
    npt.assert_equal(np.all(percept[:, :half] != 0), True)

def test_temporal_predict():
    model = DynaphosModel(xystep=0.1).build()
    # User can set params
    model.dt = 40
    npt.assert_equal(model.dt, 40)

    implant = Cortivis(stim=np.zeros((96, 100)))

    # Can't request the same time more than once (this would break the Cython
    # loop, because `idx_frame` is incremented after a write; also doesn't
    # make much sense):
    with pytest.raises(ValueError):
        implant.stim = np.ones((96, 100))
        model.predict_percept(implant, t_percept=[0.2, 0.2])

    # Brightness scales with amplitude:
    model.dt = 20
    sdur = 1000.0  # stimulus duration (ms)
    pdur = 0.45  # (ms)
    t_percept = np.arange(0, sdur, 20)
    implant = ProsthesisSystem(ElectrodeArray(DiskElectrode(0, 0, 0, 260)))
    bright_amp = []
    for amp in np.linspace(20, 70, 5):
        implant.stim = BiphasicPulseTrain(20, amp, pdur, interphase_dur=pdur,
                                          stim_dur=sdur)
        percept = model.predict_percept(implant, t_percept=t_percept)
        bright_amp.append(percept.data.max())
    bright_amp_ref = np.array([0.0, 0.0, 0.0, 0.66, 0.841])
    npt.assert_almost_equal(bright_amp, bright_amp_ref, decimal=3)

    # Test that default models give expected values
    implant = Orion(x=15000, stim={'55': BiphasicPulseTrain(freq=300, amp=100, phase_dur=0.17)})
    percept = model.predict_percept(implant)
    npt.assert_equal(np.sum(percept.data > 0.0122), 147)
    npt.assert_equal(np.sum(percept.data > 0.0375), 96)
    npt.assert_equal(np.sum(percept.data > 0.3305), 49)
    npt.assert_equal(np.sum(percept.data > 0.8451), 39)
    npt.assert_equal(np.sum(percept.data > 0.8883), 9)

def test_deepcopy_Dynaphos():
    original = DynaphosModel()
    copied = copy.deepcopy(original)

    # Assert these are two different objects
    npt.assert_equal(id(original) != id(copied), True)

    # Assert these objects are equivalent
    npt.assert_equal(original.__dict__, copied.__dict__)

    # Assert building one object does not affect the copied
    original.build()
    npt.assert_equal(copied.is_built, False)
    # Array-aware: a plain dict comparison raises once the model is
    # built, because `array == array` cannot be coerced to a bool.
    npt.assert_raises(AssertionError, npt.assert_equal,
                      original.__dict__, copied.__dict__)

    # Assert destroying the original doesn't affect the copied
    original = None
    npt.assert_equal(copied is not None, True)

def test_dynaphos_plot():
    # make sure that plotting works before and after building
    m = DynaphosModel()
    m.plot()
    plt.close()
    m.build()
    m.plot()
    plt.close()


def test_DynaphosModel_units():
    """A unitful parameter lands on the same percept as the bare one

    `rheobase` is a current in microamps, and 0.0239 mA is 23.9 uA -- but
    23.900000000000002 after the multiplication, which is why this compares
    with a tolerance rather than for equality.
    """
    kwargs = dict(xrange=(-3, 3), yrange=(-3, 3), xystep=0.5)
    bare = DynaphosModel(rheobase=23.9, **kwargs).build()
    unitful = DynaphosModel(rheobase=0.0239 * mA, **kwargs).build()
    npt.assert_allclose(unitful.rheobase, 23.9, rtol=1e-12)
    npt.assert_equal(isinstance(unitful.rheobase, Quantity), False)
    implant = Cortivis(stim={'11': BiphasicPulseTrain(20, 50, 0.45,
                                                      stim_dur=100)})
    npt.assert_allclose(unitful.predict_percept(implant).data,
                        bare.predict_percept(implant).data, rtol=1e-6)
    # The model states what its numbers mean:
    npt.assert_equal((bare.stimulus_unit, bare.space_unit, bare.time_unit),
                     (uA, um, ms))
    with pytest.raises(DimensionMismatchError):
        DynaphosModel(rheobase=5 * ms)


def test_DynaphosModel_t_percept_units():
    """This model overrides `predict_percept`, so it normalizes for itself"""
    model = DynaphosModel(xrange=(-3, 3), yrange=(-3, 3), xystep=1).build()
    implant = Cortivis(stim={'11': BiphasicPulseTrain(20, 50, 0.45,
                                                      stim_dur=100)})
    bare = model.predict_percept(implant, t_percept=[0, 20, 40])
    for spelling in ([0, 20, 40] * ms, np.array([0, .02, .04]) * s):
        unitful = model.predict_percept(implant, t_percept=spelling)
        npt.assert_allclose(unitful.data, bare.data, rtol=1e-12)
        npt.assert_allclose(unitful.time, [0, 20, 40], rtol=1e-12)
    with pytest.raises(DimensionMismatchError):
        model.predict_percept(implant, t_percept=[0, 20] * uA)


def test_DynaphosModel_default_frame_clock_stops_at_the_stimulus():
    """The default output clock does not run past the end of the stimulus

    `arange`'s half-open end used to be nudged by the literal 1, which meant
    one *millisecond*: with a `dt` finer than that it emitted frames after the
    stimulus was over, and for a model counting in anything but milliseconds
    it would have been meaningless.
    """
    stim = BiphasicPulseTrain(20, 50, 0.1, stim_dur=10)
    implant = Cortivis(stim={'11': stim})
    kwargs = dict(xrange=(-2, 2), yrange=(-2, 2), xystep=1)

    # Coarser than a millisecond, which is the case the literal was written
    # for, and still the same clock it always produced:
    model = DynaphosModel(dt=2, **kwargs).build()
    npt.assert_allclose(model.predict_percept(implant).time,
                        np.arange(0, 11, 2), rtol=1e-12)

    # Finer than a millisecond, which is where it overshot:
    model = DynaphosModel(dt=0.5, **kwargs).build()
    percept = model.predict_percept(implant)
    npt.assert_allclose(percept.time, np.arange(0, 10.25, 0.5), rtol=1e-12)
    npt.assert_equal(percept.time[-1] <= implant.stim.time[-1], True)
    # The endpoint is included, not dropped:
    npt.assert_allclose(percept.time[-1], implant.stim.time[-1], rtol=1e-12)
