import numpy.testing as npt
import pytest
import numpy as np
import copy
from pulse2percept.models.cortex import DynaphosModel
from pulse2percept.implants import ProsthesisSystem, ElectrodeArray, DiskElectrode
from pulse2percept.implants.cortex import Cortivis, Orion
from pulse2percept.topography import Polimeni2006Map
from pulse2percept.percepts import Percept
from pulse2percept.stimuli import BiphasicPulseTrain

def test_DynaphosModel():
    model = DynaphosModel(xrange=(-3, 3), yrange=(-3, 3), xystep=0.1).build()

    npt.assert_equal(model.regions, ['v1'])
    npt.assert_equal(model.retinotopy.regions, ['v1'])

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
    bright_amp_ref = np.array([0.0, 0.208, 0.416, 0.66, 0.841])
    npt.assert_almost_equal(bright_amp, bright_amp_ref, decimal=3)

    # Test that default models give expected values
    implant = Orion(x=15000, stim={'55': BiphasicPulseTrain(freq=300, amp=100, phase_dur=0.17)})
    percept = model.predict_percept(implant)
    npt.assert_equal(np.sum(percept.data > 0.0122), 149)
    npt.assert_equal(np.sum(percept.data > 0.0375), 97)
    npt.assert_equal(np.sum(percept.data > 0.3305), 50)
    npt.assert_equal(np.sum(percept.data > 0.8451), 39)
    npt.assert_equal(np.sum(percept.data > 0.8883), 9)

def test_deepcopy_Dynaphos():
    original = DynaphosModel()
    copied = copy.deepcopy(original)

    # Assert these are two different objects
    npt.assert_equal(id(original) != id(copied), True)

    # Assert these objects are equivalent
    npt.assert_equal(original.__dict__ == copied.__dict__, True)

    # Assert building one object does not affect the copied
    original.build()
    npt.assert_equal(copied.is_built, False)
    npt.assert_equal(original.__dict__ != copied.__dict__, True)

    # Assert destroying the original doesn't affect the copied
    original = None
    npt.assert_equal(copied is not None, True)