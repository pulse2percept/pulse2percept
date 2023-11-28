import numpy.testing as npt
import pytest
import numpy as np
import copy
from pulse2percept.models.cortex import ScoreboardModel, ScoreboardSpatial
from pulse2percept.models import ScoreboardSpatial as BeyelerScoreboard
from pulse2percept.implants.cortex import Cortivis, Orion
from pulse2percept.implants import ArgusII
from pulse2percept.topography import Polimeni2006Map
from pulse2percept.percepts import Percept
from pulse2percept.topography import Watson2014Map


@pytest.mark.parametrize('ModelClass', [ScoreboardModel, ScoreboardSpatial])
@pytest.mark.parametrize('jitter_boundary', [True, False])
@pytest.mark.parametrize('regions', 
    [['v1'], ['v2'], ['v3'], ['v1', 'v2'], ['v2', 'v3'], ['v1', 'v3'], ['v1', 'v2', 'v3']])
def test_ScoreboardSpatial(ModelClass, jitter_boundary, regions):
    # ScoreboardSpatial automatically sets `regions`
    retinotopy = Polimeni2006Map(k=15, a=.5, b=90, jitter_boundary=jitter_boundary, regions=regions)
    model = ModelClass(xrange=(-3, 3), yrange=(-3, 3), xystep=0.1, retinotopy=retinotopy).build()
    npt.assert_equal(model.regions, regions)
    npt.assert_equal(model.retinotopy.regions, regions)

    # User can set `rho`:
    model.rho = 123
    npt.assert_equal(model.rho, 123)
    model.build(rho=987)
    npt.assert_equal(model.rho, 987)

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(Cortivis()), None)

    # Converting ret <=> dva
    retinotopy = Polimeni2006Map(k=15, a=0.5, b=90, jitter_boundary=jitter_boundary, regions=regions)
    model = ModelClass(xrange=(-3, 3), yrange=(-3, 3), xystep=1, retinotopy=retinotopy).build()
    npt.assert_equal(isinstance(model.retinotopy, Polimeni2006Map), True)
    if jitter_boundary:
        npt.assert_equal(np.isnan(model.retinotopy.dva_to_v1([0], [0])), False)
        if 'v1' in regions:
            npt.assert_equal(model.grid.v1.x[~np.isnan(model.grid.v1.x)].size, 49)
        if 'v2' in regions:
            npt.assert_equal(model.grid.v2.x[~np.isnan(model.grid.v2.x)].size, 49)
        if 'v3' in regions:
            npt.assert_equal(model.grid.v3.x[~np.isnan(model.grid.v3.x)].size, 49)
    else:
        npt.assert_equal(np.isnan(model.retinotopy.dva_to_v1([0], [0])), True)
        if 'v1' in regions:
            npt.assert_equal(model.grid.v1.x[~np.isnan(model.grid.v1.x)].size, 42)
        if 'v2' in regions:
            npt.assert_equal(model.grid.v2.x[~np.isnan(model.grid.v2.x)].size, 36)
        if 'v3' in regions:
            npt.assert_equal(model.grid.v3.x[~np.isnan(model.grid.v3.x)].size, 36)

    implant = Cortivis(x=1000, stim=np.zeros(96))
    # Zero in = zero out:
    percept = model.predict_percept(implant)
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [1])
    npt.assert_almost_equal(percept.data, 0)


@pytest.mark.parametrize('ModelClass', [ScoreboardModel, ScoreboardSpatial])
@pytest.mark.parametrize('regions', 
    [['v1'], ['v2'], ['v3'], ['v1', 'v2'], ['v2', 'v3'], ['v1', 'v3'], ['v1', 'v2', 'v3']])
def test_predict_spatial(ModelClass, regions):
    # test that no current can spread between hemispheres
    model = ModelClass(xrange=(-3, 3), yrange=(-3, 3), xystep=0.5, rho=100000, regions=regions).build()
    implant = Orion(x = 15000)
    implant.stim = {e:5 for e in implant.electrode_names}
    percept = model.predict_percept(implant)
    half = percept.shape[1] // 2
    npt.assert_equal(np.all(percept.data[:, half+1:] == 0), True)
    npt.assert_equal(np.all(percept.data[:, :half] != 0), True)

    # implant only in v1, shouldnt change with v2/v3
    vfmap = Polimeni2006Map(k=15, a=0.5, b=90)
    model = ModelClass(xrange=(-5, 0), yrange=(-3, 3), xystep=0.1, rho=400, retinotopy=vfmap).build()
    elecs = [79, 49, 19, 80, 50, 20, 90, 61, 31, 2, 72, 42, 12, 83, 53, 23, 93, 64, 34, 5, 75, 45, 15, 86, 56, 26, 96, 67, 37, 8, 68, 38]
    implant = Cortivis(x=30000, y=0, rot=0, stim={str(i) : [1, 0] for i in elecs})
    percept = model.predict_percept(implant)
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [2])
    npt.assert_equal(np.all(percept.data[:, :, 1] == 0), True)
    pmax = percept.data.max()
    npt.assert_almost_equal(percept.data[33, 18, 0], pmax)
    npt.assert_almost_equal(percept.data[30, 13, 0], 1.96066, 5)
    npt.assert_almost_equal(percept.data[32, 8, 0], 0.013312, 5)
    npt.assert_equal(np.sum(percept.data > 0.75), 122)
    npt.assert_equal(np.sum(percept.data > 1), 105)
    npt.assert_almost_equal(percept.time, [0, 1])

    if 'v1' in regions:
        # make sure cortical representation is flipped
        vfmap = Polimeni2006Map(k=15, a=0.5, b=90)
        model = ModelClass(xrange=(-5, 0), yrange=(-3, 3), xystep=0.1, rho=400, retinotopy=vfmap).build()
        implant = Orion(x=30000, y=0, rot=0, stim={'40' : 1,  '94' :5})
        percept = model.predict_percept(implant)
        half = model.grid.shape[0] // 2
        npt.assert_equal(np.sum(percept.data[:half, :, :]) >  np.sum(percept.data[half:, :, :]), True)


@pytest.mark.parametrize('ModelClass', [ScoreboardModel, ScoreboardSpatial])
@pytest.mark.parametrize('regions', [['v1', 'v2'], ['v1', 'v3'], ['v2', 'v3']])
def test_predict_spatial_regionsum(ModelClass,regions):
    print(regions)
    model1 = ModelClass(xrange=(-3, 3), yrange=(-3, 3), xystep=0.1, rho=10000, regions=regions[0]).build()
    model2 = ModelClass(xrange=(-3, 3), yrange=(-3, 3), xystep=0.1, rho=10000, regions=regions[1]).build()
    model_both = ModelClass(xrange=(-3, 3), yrange=(-3, 3), xystep=0.1, rho=10000, regions=regions).build()

    implant = Orion(x = 10000, y=10000)
    implant.stim = {e : 1 for e in implant.electrode_names}

    percept1 = model1.predict_percept(implant)
    percept2 = model2.predict_percept(implant)
    percept_both = model_both.predict_percept(implant)

    npt.assert_almost_equal(percept1.data + percept2.data, percept_both.data)


@pytest.mark.parametrize('ModelClass', [ScoreboardModel, ScoreboardSpatial])
@pytest.mark.parametrize('stimval', np.arange(0, 5, 1))
def test_eq_beyeler(ModelClass, stimval):
    

    retinotopy = Watson2014Map()
    cortex = ModelClass(xrange=(-3, 3), yrange=(-3, 3), xystep=0.1, rho=200 * stimval, regions=['ret'], retinotopy=retinotopy).build()
    retina = BeyelerScoreboard(xrange=(-3, 3), yrange=(-3, 3), xystep=0.1, rho=200 * stimval).build()

    implant = ArgusII()
    implant.stim = {e : 3 for e in implant.electrode_names[::stimval+1]}

    p1 = cortex.predict_percept(implant)
    p2 = retina.predict_percept(implant)

    npt.assert_equal(p1.data, p2.data)



@pytest.mark.parametrize('ModelClass', [ScoreboardModel, ScoreboardSpatial])
def test_deepcopy_Scoreboard(ModelClass):
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


@pytest.mark.parametrize('ModelClass', [ScoreboardModel, ScoreboardSpatial])
def test_plot(ModelClass):
    # make sure that plotting works before and after building
    m = ModelClass()
    m.plot()
    m.build()
    m.plot()
