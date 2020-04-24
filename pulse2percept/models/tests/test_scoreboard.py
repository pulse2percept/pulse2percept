import numpy as np
import numpy.testing as npt

from pulse2percept import stimuli
from pulse2percept.implants import ArgusI, ArgusII
from pulse2percept.models import ScoreboardSpatial, ScoreboardModel, Percept


def test_ScoreboardSpatial():
    # ScoreboardSpatial automatically sets `rho`:
    model = ScoreboardSpatial(engine='serial', xystep=5)

    # User can set `rho`:
    model.rho = 123
    npt.assert_equal(model.rho, 123)
    model.build(rho=987)
    npt.assert_equal(model.rho, 987)

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(ArgusI()), None)

    # Zero in = zero out:
    implant = ArgusI(stim=np.zeros(16))
    percept = model.predict_percept(implant)
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [1])
    npt.assert_almost_equal(percept.data, 0)

    # Multiple frames are processed independently:
    model = ScoreboardSpatial(engine='serial', rho=200, xystep=5)
    model.build()
    percept = model.predict_percept(ArgusI(stim={'A1': [1, 2]}))
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [2])
    pmax = percept.data.max(axis=(0, 1))
    npt.assert_almost_equal(percept.data[2, 3, :], pmax)
    npt.assert_almost_equal(pmax[1] / pmax[0], 2.0)


def test_ScoreboardModel():
    # ScoreboardModel automatically sets `rho`:
    model = ScoreboardModel(engine='serial', xystep=5)
    npt.assert_equal(model.has_space, True)
    npt.assert_equal(model.has_time, False)
    npt.assert_equal(hasattr(model.spatial, 'rho'), True)

    # User can set `rho`:
    model.rho = 123
    npt.assert_equal(model.rho, 123)
    npt.assert_equal(model.spatial.rho, 123)
    model.build(rho=987)
    npt.assert_equal(model.rho, 987)
    npt.assert_equal(model.spatial.rho, 987)

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(ArgusI()), None)

    # Zero in = zero out:
    implant = ArgusI(stim=np.zeros(16))
    npt.assert_almost_equal(model.predict_percept(implant).data, 0)

    # Multiple frames are processed independently:
    model = ScoreboardModel(engine='serial', rho=200, xystep=5)
    model.build()
    percept = model.predict_percept(ArgusI(stim={'A1': [1, 2]}))
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [2])
    pmax = percept.data.max(axis=(0, 1))
    npt.assert_almost_equal(percept.data[2, 3, :], pmax)
    npt.assert_almost_equal(pmax[1] / pmax[0], 2.0)


def test_ScoreboardModel_predict_percept():
    model = ScoreboardModel(xystep=1, rho=100, thresh_percept=0)
    model.build()
    # Single-electrode stim:
    img_stim = np.zeros(60)
    img_stim[47] = 1
    percept = model.predict_percept(ArgusII(stim=img_stim))
    # Single bright pixel, very small Gaussian kernel:
    npt.assert_equal(np.sum(percept.data > 0.9), 1)
    npt.assert_equal(np.sum(percept.data > 0.5), 1)
    npt.assert_equal(np.sum(percept.data > 0.1), 1)
    npt.assert_equal(np.sum(percept.data > 0.00001), 9)
    # Brightest pixel is in lower right:
    npt.assert_almost_equal(percept.data[18, 25, 0], np.max(percept.data))

    # Full Argus II: 60 bright spots
    model = ScoreboardModel(engine='serial', xystep=1, rho=100)
    model.build()
    percept = model.predict_percept(ArgusII(stim=np.ones(60)))
    npt.assert_equal(np.sum(np.isclose(percept.data, 0.9, rtol=0.1, atol=0.1)),
                     60)
