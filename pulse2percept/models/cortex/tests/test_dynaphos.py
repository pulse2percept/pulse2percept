import numpy.testing as npt
import pytest
import numpy as np
import copy
from pulse2percept.models.cortex import DynaphosModel
from pulse2percept.implants.cortex import Cortivis, Orion
from pulse2percept.topography import Polimeni2006Map
from pulse2percept.percepts import Percept
from pulse2percept.stimuli import BiphasicPulseTrain

def test_DynaphosModel():
    model = DynaphosModel(xrange=(-3, 3), yrange=(-3, 3), xystep=0.1).build()

    npt.assert_equal(model.regions, ['v1'])
    npt.assert_equal(model.retinotopy.regions, ['v1'])

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(Cortivis()), None)

    implant = Cortivis(x=1000, stim={e:BiphasicPulseTrain(freq=300,amp=0,phase_dur=1) for e in Cortivis().electrode_names})
    # Zero in = zero out:
    percept = model.predict_percept(implant)
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, list(model.grid.x.shape)+[51]) # 51 time points
    npt.assert_almost_equal(percept.data, 0)

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