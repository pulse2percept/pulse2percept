from types import SimpleNamespace
import numpy as np
import pytest
import numpy.testing as npt

from matplotlib.axes import Subplot
import matplotlib.pyplot as plt


from pulse2percept.implants import ArgusI, ArgusII
from pulse2percept.percepts import Percept
from pulse2percept.models import (Thompson2003Spatial, Thompson2003Model)
from pulse2percept.utils import Curcio1990Map, Watson2014DisplaceMap
from pulse2percept.utils.testing import assert_warns_msg


def test_Thompson2003Spatial():
    # Thompson2003Spatial automatically sets `radius`:
    model = Thompson2003Spatial(engine='serial', xystep=5)
    # User can set `radius`:
    model.radius = 123
    npt.assert_equal(model.radius, 123)
    model.build(radius=987)
    npt.assert_equal(model.radius, 987)

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(ArgusI()), None)

    # Converting ret <=> dva
    model2 = Thompson2003Spatial(retinotopy=Watson2014DisplaceMap())
    npt.assert_equal(isinstance(model2.retinotopy, Watson2014DisplaceMap),
                     True)

    implant = ArgusI(stim=np.zeros(16))
    # Zero in = zero out:
    percept = model.predict_percept(implant)
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [1])
    npt.assert_almost_equal(percept.data, 0)

    # Multiple frames are processed independently:
    model = Thompson2003Spatial(engine='serial', radius=200, xystep=5,
                                xrange=(-20, 20), yrange=(-15, 15))
    model.build()
    percept = model.predict_percept(ArgusI(stim={'A1': [1, 0], 'B3': [0, 2]}))
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [2])
    pmax = percept.data.max(axis=(0, 1))
    npt.assert_almost_equal(percept.data[2, 3, 0], pmax[0])
    npt.assert_almost_equal(percept.data[2, 3, 1], 0)
    npt.assert_almost_equal(percept.data[3, 4, 0], 0)
    npt.assert_almost_equal(percept.data[3, 4, 1], pmax[1])
    npt.assert_almost_equal(percept.time, [0, 1])


def test_Thompson2003Model():
    model = Thompson2003Model(engine='serial', xystep=5)
    npt.assert_equal(model.has_space, True)
    npt.assert_equal(model.has_time, False)
    npt.assert_equal(hasattr(model.spatial, 'radius'), True)

    # User can set `radius`:
    model.radius = 123
    npt.assert_equal(model.radius, 123)
    npt.assert_equal(model.spatial.radius, 123)
    model.build(radius=987)
    npt.assert_equal(model.radius, 987)
    npt.assert_equal(model.spatial.radius, 987)

    # Converting ret <=> dva
    npt.assert_equal(isinstance(model.retinotopy, Curcio1990Map), True)
    npt.assert_almost_equal(model.retinotopy.ret2dva(0, 0), (0, 0))
    npt.assert_almost_equal(model.retinotopy.dva2ret(0, 0), (0, 0))
    model2 = Thompson2003Model(retinotopy=Watson2014DisplaceMap())
    npt.assert_equal(isinstance(model2.retinotopy, Watson2014DisplaceMap),
                     True)
    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(ArgusI()), None)

    # Zero in = zero out:
    implant = ArgusI(stim=np.zeros(16))
    npt.assert_almost_equal(model.predict_percept(implant).data, 0)

    # Multiple frames are processed independently:
    model = Thompson2003Model(engine='serial', radius=1000, xystep=5,
                              xrange=(-20, 20), yrange=(-15, 15))
    model.build()
    percept = model.predict_percept(ArgusI(stim={'A1': [1, 2]}))
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [2])
    pmax = percept.data.max(axis=(0, 1))
    npt.assert_almost_equal(percept.data[2, 3, :], pmax)
    print(pmax, percept.data)
    npt.assert_almost_equal(pmax[1] / pmax[0], 2.0)
    npt.assert_almost_equal(percept.time, [0, 1])


def test_Thompson2003Model_predict_percept():
    model = Thompson2003Model(xystep=0.55, radius=100, thresh_percept=0,
                              xrange=(-20, 20), yrange=(-15, 15))
    model.build()
    # Single-electrode stim:
    img_stim = np.zeros(60)
    img_stim[47] = 1
    percept = model.predict_percept(ArgusII(stim=img_stim))
    # Single bright pixel, very small Gaussian kernel:
    npt.assert_equal(np.sum(percept.data > 0.5), 1)
    npt.assert_equal(np.sum(percept.data > 0.00001), 1)
    # Brightest pixel is in lower right:
    npt.assert_almost_equal(percept.data[33, 46, 0], np.max(percept.data))

    # Full Argus II: 60 bright spots
    model = Thompson2003Model(engine='serial', xystep=0.55, radius=100)
    model.build()
    percept = model.predict_percept(ArgusII(stim=np.ones(60)))
    npt.assert_equal(np.sum(np.isclose(percept.data, 1.0, rtol=0.1, atol=0.1)),
                     84)

    # Model gives same outcome as Spatial:
    spatial = Thompson2003Spatial(engine='serial', xystep=1, radius=100)
    spatial.build()
    spatial_percept = model.predict_percept(ArgusII(stim=np.ones(60)))
    npt.assert_almost_equal(percept.data, spatial_percept.data)
    npt.assert_equal(percept.time, None)

    # Warning for nonzero electrode-retina distances
    implant = ArgusI(stim=np.ones(16), z=10)
    msg = ("Nonzero electrode-retina distances do not have any effect on the "
           "model output.")
    assert_warns_msg(UserWarning, model.predict_percept, msg, implant)
