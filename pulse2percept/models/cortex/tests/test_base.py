import numpy.testing as npt
import pytest
import numpy as np
import copy
import matplotlib.pyplot as plt

from pulse2percept.models.cortex import ScoreboardModel, ScoreboardSpatial
from pulse2percept.models import ScoreboardSpatial as BeyelerScoreboard
from pulse2percept.implants.cortex import Cortivis, Orion, LinearEdgeThread
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
    vfmap = Polimeni2006Map(k=15, a=.5, b=90, jitter_boundary=jitter_boundary, regions=regions)
    model = ModelClass(xrange=(-3, 3), yrange=(-3, 3), step=0.1, vfmap=vfmap).build()
    npt.assert_equal(model.regions, regions)
    npt.assert_equal(model.vfmap.regions, regions)

    # User can set `rho`:
    model.rho = 123
    npt.assert_equal(model.rho, 123)
    model.build(rho=987)
    npt.assert_equal(model.rho, 987)

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(Cortivis()), None)

    # Converting ret <=> dva
    vfmap = Polimeni2006Map(k=15, a=0.5, b=90, jitter_boundary=jitter_boundary, regions=regions)
    model = ModelClass(xrange=(-3, 3), yrange=(-3, 3), step=1, vfmap=vfmap).build()
    npt.assert_equal(isinstance(model.vfmap, Polimeni2006Map), True)
    if jitter_boundary:
        npt.assert_equal(np.isnan(model.vfmap.dva_to_v1([0], [0])), False)
        if 'v1' in regions:
            npt.assert_equal(model.grid.v1.x[~np.isnan(model.grid.v1.x)].size, 49)
        if 'v2' in regions:
            npt.assert_equal(model.grid.v2.x[~np.isnan(model.grid.v2.x)].size, 49)
        if 'v3' in regions:
            npt.assert_equal(model.grid.v3.x[~np.isnan(model.grid.v3.x)].size, 49)
    else:
        npt.assert_equal(np.isnan(model.vfmap.dva_to_v1([0], [0])), True)
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
    model = ModelClass(xrange=(-3, 3), yrange=(-3, 3), step=0.5, rho=100000, regions=regions).build()
    implant = Orion(x = 15000)
    implant.stim = {e:5 for e in implant.electrode_names}
    percept = model.predict_percept(implant)
    half = percept.shape[1] // 2
    npt.assert_equal(np.all(percept.data[:, half+1:] == 0), True)
    npt.assert_equal(np.all(percept.data[:, :half] != 0), True)

    # implant only in v1, shouldnt change with v2/v3
    vfmap = Polimeni2006Map(k=15, a=0.5, b=90)
    model = ModelClass(xrange=(-5, 0), yrange=(-3, 3), step=0.1, rho=400, vfmap=vfmap).build()
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
        model = ModelClass(xrange=(-5, 0), yrange=(-3, 3), step=0.1, rho=400, vfmap=vfmap).build()
        implant = Orion(x=30000, y=0, rot=0, stim={'40' : 1,  '94' :5})
        percept = model.predict_percept(implant)
        half = model.grid.shape[0] // 2
        npt.assert_equal(np.sum(percept.data[:half, :, :]) >  np.sum(percept.data[half:, :, :]), True)


@pytest.mark.parametrize('ModelClass', [ScoreboardModel, ScoreboardSpatial])
@pytest.mark.parametrize('regions', [['v1', 'v2'], ['v1', 'v3'], ['v2', 'v3']])
def test_predict_spatial_regionsum(ModelClass,regions):
    print(regions)
    model1 = ModelClass(xrange=(-3, 3), yrange=(-3, 3), step=0.1, rho=10000, regions=regions[0]).build()
    model2 = ModelClass(xrange=(-3, 3), yrange=(-3, 3), step=0.1, rho=10000, regions=regions[1]).build()
    model_both = ModelClass(xrange=(-3, 3), yrange=(-3, 3), step=0.1, rho=10000, regions=regions).build()

    implant = Orion(x = 10000, y=10000)
    implant.stim = {e : 1 for e in implant.electrode_names}

    percept1 = model1.predict_percept(implant)
    percept2 = model2.predict_percept(implant)
    percept_both = model_both.predict_percept(implant)

    npt.assert_almost_equal(percept1.data + percept2.data, percept_both.data)


@pytest.mark.parametrize('ModelClass', [ScoreboardModel, ScoreboardSpatial])
@pytest.mark.parametrize('stimval', np.arange(0, 5, 1))
def test_eq_beyeler(ModelClass, stimval):
    

    vfmap = Watson2014Map()
    cortex = ModelClass(xrange=(-3, 3), yrange=(-3, 3), step=0.1, rho=200 * stimval, regions=['ret'], vfmap=vfmap).build()
    retina = BeyelerScoreboard(xrange=(-3, 3), yrange=(-3, 3), step=0.1, rho=200 * stimval).build()

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


@pytest.mark.parametrize('ModelClass', [ScoreboardModel, ScoreboardSpatial])
def test_plot(ModelClass):
    # make sure that plotting works before and after building
    m = ModelClass()
    m.plot()
    plt.close()
    m.build()
    m.plot()
    plt.close()


def test_poli_nlink():
    # make sure that the polimeni map and neuralink work togther with scoreboard
    # since this is an odd combo of 2d map and 3d implant
    model = ScoreboardModel(rho=800, step=.5).build()
    npt.assert_equal(model.grid.v1.z is None, True)
    npt.assert_equal(model.grid.v1.x is None, False)
    implant = LinearEdgeThread(x=20000,)
    implant.stim = {e : 1 for e in implant.electrode_names}
    percept = model.predict_percept(implant)
    npt.assert_almost_equal(np.sum(percept.data), 32.494125, decimal=3)
    npt.assert_equal(np.sum(percept.data > .05), 4)


@pytest.mark.parametrize('ModelClass', [ScoreboardModel, ScoreboardSpatial])
def test_CortexSpatial_meridian_blend(ModelClass):
    # The hemifields are mapped onto opposite hemispheres, so a cortical model
    # blends across x=0 -- and only there, and only along x.
    def make(**params):
        return ModelClass(xrange=(-5, 5), yrange=(-5, 5), step=0.2, rho=800,
                          **params).build()

    # Close to the midline, so the phosphenes land on the vertical meridian
    # and are cut off by it -- which is the seam this blends across. An array
    # further out produces a percept that never reaches x=0 and so has no seam
    # to show:
    implant = Cortivis(x=5000)
    implant.stim = {e: 1 for e in implant.electrode_names}
    plain = make()
    unblended = plain.predict_percept(implant).data
    npt.assert_array_less(0, unblended.max())

    # The default is 0, and 0 has to be the model exactly as it was:
    npt.assert_equal(plain.meridian_blend, 0)
    npt.assert_array_equal(
        make(meridian_blend=0).predict_percept(implant).data, unblended)

    width = 0.5
    blended = make(meridian_blend=width).predict_percept(implant).data
    npt.assert_equal(blended.shape, unblended.shape)
    npt.assert_equal(blended.dtype, unblended.dtype)

    x = plain.grid.x[0, :]
    # The vertical meridian is where the two half-field models meet, and the
    # step across it is what the blend is for:
    seam = np.argsort(np.abs(x))[:2]

    def jump(data):
        return np.abs(data[:, seam[0], 0] - data[:, seam[1], 0]).max()

    npt.assert_array_less(0, jump(unblended))
    npt.assert_array_less(jump(blended), jump(unblended) / 10)

    # The change stays within a few widths of the meridian; the far field is
    # untouched. A column counts as having moved if it moved by at least a
    # thousandth of the largest change anywhere, so this is a bound on where
    # the blend acts rather than on float noise:
    delta = np.abs(blended - unblended)
    cols = delta.max(axis=(0, 2)) > delta.max() * 1e-3
    npt.assert_equal(np.any(cols), True)
    npt.assert_array_less(np.abs(x[cols]).max(), 4 * width)

    # It is the *vertical* meridian, so the blur runs along x and not along y:
    # a row that was dark stays dark, because nothing is carried into it from
    # the rows above and below...
    dark_rows = unblended.max(axis=(1, 2)) == 0
    npt.assert_equal(np.any(dark_rows), True)
    npt.assert_array_equal(blended[dark_rows], 0)
    # ...while a column that was dark does light up, from its neighbors along
    # x, which is the smoothing this is supposed to do:
    dark_cols = unblended.max(axis=(0, 2)) == 0
    npt.assert_equal(np.any(blended[:, dark_cols] > 0), True)


def test_CortexSpatial_meridian_blend_reapplies_threshold():
    # Blending pulls brightness across the meridian, which could otherwise
    # lift a point that `thresh_percept` had zeroed back off zero.
    implant = Cortivis(x=5000)
    implant.stim = {e: 1 for e in implant.electrode_names}
    model = ScoreboardModel(xrange=(-5, 5), yrange=(-5, 5), step=0.2, rho=800,
                            meridian_blend=0.5, thresh_percept=0.1).build()
    data = model.predict_percept(implant).data
    npt.assert_equal(np.any(data > 0), True)
    # Nothing survives strictly between zero and the threshold:
    npt.assert_equal(np.any((np.abs(data) > 0) & (np.abs(data) < 0.1)), False)
