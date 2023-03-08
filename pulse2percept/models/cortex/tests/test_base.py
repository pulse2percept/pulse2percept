import numpy.testing as npt
import numpy as np
import copy
from pulse2percept.models.cortex import ScoreboardModel, ScoreboardSpatial
from pulse2percept.implants.cortex import Cortivis
from pulse2percept.topography import Polimeni2006Map
from pulse2percept.percepts import Percept

def test_ScoreboardSpatial():
    # ScoreboardSpatial automatically sets `rho`:
    model = ScoreboardSpatial(xystep=1)

    # User can set `rho`:
    model.rho = 123
    npt.assert_equal(model.rho, 123)
    model.build(rho=987)
    npt.assert_equal(model.rho, 987)

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(Cortivis()), None)

    # Converting ret <=> dva
    npt.assert_equal(isinstance(model.retinotopy, Polimeni2006Map), True)
    npt.assert_equal(np.isnan(model.retinotopy.dva_to_v1([0], [0])), True)

    implant = Cortivis(x=1000, stim=np.zeros(96))
    # Zero in = zero out:
    percept = model.predict_percept(implant)
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [1])
    npt.assert_almost_equal(percept.data, 0)

    
    model = ScoreboardSpatial(xrange=(-5, 0), yrange=(-3, 3), xystep=0.1, rho=400).build()
    implant = Cortivis(x=30000, stim={str(i) : [1, 0] for i in range(1, 96, 3)})
    percept = model.predict_percept(implant)
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [2])
    npt.assert_equal(np.all(percept.data[:, :, 1] == 0), True)
    pmax = percept.data.max()
    npt.assert_almost_equal(percept.data[27, 18, 0], pmax)
    npt.assert_almost_equal(percept.data[30, 13, 0], 1.96066)
    npt.assert_almost_equal(percept.data[32, 8, 0], 0.013483)
    npt.assert_equal(np.sum(percept.data > 0.75), 122)
    npt.assert_equal(np.sum(percept.data > 1), 105)
    npt.assert_almost_equal(percept.time, [0, 1])


def test_deepcopy_ScoreboardSpatial():
    original = ScoreboardSpatial()
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

