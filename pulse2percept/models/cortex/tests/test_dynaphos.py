import numpy.testing as npt
import pytest
import numpy as np
import copy
from pulse2percept.models.cortex import DynaphosSpatial
from pulse2percept.implants.cortex import Cortivis, Orion
from pulse2percept.topography import Polimeni2006Map
from pulse2percept.percepts import Percept

def test_DynaphosSpatial():
    model = DynaphosSpatial(xrange=(-3, 3), yrange=(-3, 3), xystep=0.1).build()

    npt.assert_equal(model.regions, ['v1'])
    npt.assert_equal(model.retinotopy.regions, ['v1'])

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(Cortivis()), None)

    implant = Cortivis(x=1000, stim=np.zeros(96))
    # Zero in = zero out:
    percept = model.predict_percept(implant)
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [1])
    npt.assert_almost_equal(percept.data, 0)

def test_predict_spatial():
    # test that no current can spread between hemispheres
    model = DynaphosSpatial(xrange=(-3, 3), yrange=(-3, 3), xystep=0.5).build()
    implant = Orion(x = 15000)
    implant.stim = {e:500 for e in implant.electrode_names}
    percept = model.predict_percept(implant)
    half = percept.shape[1] // 2
    npt.assert_equal(np.all(percept.data[:, half+1:] == 0), True)
    npt.assert_equal(np.all(percept.data[:, :half] != 0), True)

@pytest.mark.parametrize('ModelClass', [DynaphosSpatial])
def test_deepcopy_Dynaphos(ModelClass):
    original = ModelClass()
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