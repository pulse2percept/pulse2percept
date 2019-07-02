import numpy as np
import numpy.testing as npt

from pulse2percept import implants
from pulse2percept import stimuli
from pulse2percept import models


def test_ScoreboardModel():
    # ScoreboardModel automatically sets `rho`:
    model = models.ScoreboardModel(engine='serial', xystep=5)
    npt.assert_equal(hasattr(model, 'rho'), True)

    # User can set `rho`:
    model.rho = 123
    npt.assert_equal(model.rho, 123)
    model.build(rho=987)
    npt.assert_equal(model.rho, 987)

    # Zero in = zero out:
    implant = implants.ArgusI(stim=np.zeros((4, 4)))
    npt.assert_almost_equal(model.predict_percept(implant), 0)


def test_ScoreboardModel_predict_percept():
    model = models.ScoreboardModel(xystep=1, rho=100, thresh_percept=0)
    model.build()
    # Single-electrode stim:
    img_stim = np.zeros((6, 10))
    img_stim[4, 7] = 1
    percept = model.predict_percept(implants.ArgusII(stim=img_stim))
    # Single bright pixel, very small Gaussian kernel:
    npt.assert_equal(np.sum(percept > 0.9), 1)
    npt.assert_equal(np.sum(percept > 0.5), 1)
    npt.assert_equal(np.sum(percept > 0.1), 1)
    npt.assert_equal(np.sum(percept > 0.00001), 9)
    # Brightest pixel is in lower right:
    npt.assert_almost_equal(percept[18, 25], np.max(percept))

    # Full Argus II: 60 bright spots
    model = models.ScoreboardModel(engine='serial', xystep=1, rho=100)
    model.build()
    percept = model.predict_percept(implants.ArgusII(stim=np.ones((6, 10))))
    npt.assert_equal(np.sum(np.isclose(percept, 0.9, rtol=0.1, atol=0.1)), 60)
