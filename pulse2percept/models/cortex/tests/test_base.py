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


def _spatial(model):
    """The spatial model itself, or the one a composite wraps."""
    return getattr(model, 'spatial', model)


@pytest.mark.parametrize('ModelClass', [ScoreboardModel, ScoreboardSpatial])
@pytest.mark.parametrize('jitter_boundary', [True, False])
@pytest.mark.parametrize('regions', 
    [['v1'], ['v2'], ['v3'], ['v1', 'v2'], ['v2', 'v3'], ['v1', 'v3'], ['v1', 'v2', 'v3']])
def test_ScoreboardSpatial(ModelClass, jitter_boundary, regions):
    # ScoreboardSpatial automatically sets `regions`
    vfmap = Polimeni2006Map(k=15, a=.5, b=90, jitter_boundary=jitter_boundary, regions=regions)
    model = ModelClass(implant=Cortivis(), xrange=(-3, 3), yrange=(-3, 3), step=0.1, vfmap=vfmap).build()
    spatial = _spatial(model)
    npt.assert_equal(spatial.regions, regions)
    npt.assert_equal(spatial.vfmap.regions, regions)

    # User can set `rho`:
    spatial.rho = 123
    npt.assert_equal(spatial.rho, 123)
    spatial.build(rho=987)
    npt.assert_equal(spatial.rho, 987)

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(None), None)

    # Converting ret <=> dva
    vfmap = Polimeni2006Map(k=15, a=0.5, b=90, jitter_boundary=jitter_boundary, regions=regions)
    model = ModelClass(implant=Cortivis(), xrange=(-3, 3), yrange=(-3, 3), step=1, vfmap=vfmap).build()
    spatial = _spatial(model)
    npt.assert_equal(isinstance(spatial.vfmap, Polimeni2006Map), True)
    if jitter_boundary:
        npt.assert_equal(np.isnan(spatial.vfmap.dva_to_v1([0], [0])), False)
        if 'v1' in regions:
            npt.assert_equal(
                spatial.grid.v1.x[~np.isnan(spatial.grid.v1.x)].size,
                49)
        if 'v2' in regions:
            npt.assert_equal(
                spatial.grid.v2.x[~np.isnan(spatial.grid.v2.x)].size,
                49)
        if 'v3' in regions:
            npt.assert_equal(
                spatial.grid.v3.x[~np.isnan(spatial.grid.v3.x)].size,
                49)
    else:
        npt.assert_equal(np.isnan(spatial.vfmap.dva_to_v1([0], [0])), True)
        if 'v1' in regions:
            npt.assert_equal(
                spatial.grid.v1.x[~np.isnan(spatial.grid.v1.x)].size,
                42)
        if 'v2' in regions:
            npt.assert_equal(
                spatial.grid.v2.x[~np.isnan(spatial.grid.v2.x)].size,
                36)
        if 'v3' in regions:
            npt.assert_equal(
                spatial.grid.v3.x[~np.isnan(spatial.grid.v3.x)].size,
                36)

    # Zero in = zero out:
    percept = model.predict_percept(np.zeros(96))
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, list(spatial.grid.x.shape) + [1])
    npt.assert_almost_equal(percept.data, 0)


@pytest.mark.parametrize('ModelClass', [ScoreboardModel, ScoreboardSpatial])
@pytest.mark.parametrize('regions', 
    [['v1'], ['v2'], ['v3'], ['v1', 'v2'], ['v2', 'v3'], ['v1', 'v3'], ['v1', 'v2', 'v3']])
def test_predict_spatial(ModelClass, regions):
    # test that no current can spread between hemispheres
    implant = Orion(x=15000)
    model = ModelClass(implant=implant, xrange=(-3, 3), yrange=(-3, 3),
                       step=0.5, rho=100000, regions=regions).build()
    percept = model.predict_percept({e: 5 for e in implant.electrode_names})
    half = percept.shape[1] // 2
    npt.assert_equal(np.all(percept.data[:, half+1:] == 0), True)
    npt.assert_equal(np.all(percept.data[:, :half] != 0), True)

    # implant only in v1, shouldnt change with v2/v3
    vfmap = Polimeni2006Map(k=15, a=0.5, b=90)
    model = ModelClass(implant=Cortivis(x=30000, y=0, rot=0), xrange=(-5, 0),
                       yrange=(-3, 3), step=0.1, rho=400, vfmap=vfmap).build()
    elecs = [79, 49, 19, 80, 50, 20, 90, 61, 31, 2, 72, 42, 12, 83, 53, 23, 93, 64, 34, 5, 75, 45, 15, 86, 56, 26, 96, 67, 37, 8, 68, 38]
    percept = model.predict_percept({str(i): [1, 0] for i in elecs})
    npt.assert_equal(percept.shape, list(_spatial(model).grid.x.shape) + [2])
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
        model = ModelClass(implant=Orion(x=30000, y=0, rot=0),
                           xrange=(-5, 0), yrange=(-3, 3), step=0.1, rho=400,
                           vfmap=vfmap).build()
        percept = model.predict_percept({'40': 1, '94': 5})
        half = _spatial(model).grid.shape[0] // 2
        npt.assert_equal(np.sum(percept.data[:half, :, :]) >  np.sum(percept.data[half:, :, :]), True)


@pytest.mark.parametrize('ModelClass', [ScoreboardModel, ScoreboardSpatial])
@pytest.mark.parametrize('regions', [['v1', 'v2'], ['v1', 'v3'], ['v2', 'v3']])
def test_predict_spatial_regionsum(ModelClass,regions):
    print(regions)
    implant = Orion(x=10000, y=10000)
    grid = dict(implant=implant, xrange=(-3, 3), yrange=(-3, 3), step=0.1,
                rho=10000)
    model1 = ModelClass(regions=regions[0], **grid).build()
    model2 = ModelClass(regions=regions[1], **grid).build()
    model_both = ModelClass(regions=regions, **grid).build()

    source = {e: 1 for e in implant.electrode_names}

    percept1 = model1.predict_percept(source)
    percept2 = model2.predict_percept(source)
    percept_both = model_both.predict_percept(source)

    # Separate filtering introduces float32 round-off.
    npt.assert_almost_equal(percept1.data + percept2.data, percept_both.data,
                            decimal=4)


@pytest.mark.parametrize('ModelClass', [ScoreboardModel, ScoreboardSpatial])
@pytest.mark.parametrize('stimval', np.arange(0, 5, 1))
def test_eq_beyeler(ModelClass, stimval):
    

    vfmap = Watson2014Map()
    implant = ArgusII()
    cortex = ModelClass(implant=implant, xrange=(-3, 3), yrange=(-3, 3),
                        step=0.1, rho=200 * stimval, regions=['ret'],
                        vfmap=vfmap, meridian_blend=0).build()
    retina = BeyelerScoreboard(implant=implant, xrange=(-3, 3),
                               yrange=(-3, 3), step=0.1,
                               rho=200 * stimval).build()

    source = {e: 3 for e in implant.electrode_names[::stimval + 1]}

    p1 = cortex.predict_percept(source)
    p2 = retina.predict_percept(source)

    npt.assert_equal(p1.data, p2.data)



@pytest.mark.parametrize('ModelClass', [ScoreboardModel, ScoreboardSpatial])
def test_deepcopy_Scoreboard(ModelClass):
    original = ModelClass(implant=Cortivis())
    copied = copy.deepcopy(original)

    # Assert these are two different objects
    npt.assert_equal(id(original) != id(copied), True)

    # Assert these objects are equivalent
    npt.assert_equal(original.__dict__, copied.__dict__)

    # Assert building one object does not affect the copied
    original.build()
    npt.assert_equal(copied.is_built, False)
    npt.assert_raises(AssertionError, npt.assert_equal,
                      original.__dict__, copied.__dict__)

    # Assert destroying the original doesn't affect the copied
    original = None
    npt.assert_equal(copied is not None, True)


@pytest.mark.parametrize('ModelClass', [ScoreboardModel, ScoreboardSpatial])
def test_plot(ModelClass):
    # make sure that plotting works before and after building
    m = ModelClass(implant=Cortivis())
    m.plot()
    plt.close()
    m.build()
    m.plot()
    plt.close()


def test_poli_nlink():
    # make sure that the polimeni map and neuralink work togther with scoreboard
    # since this is an odd combo of 2d map and 3d implant
    implant = LinearEdgeThread(x=20000)
    model = ScoreboardModel(implant=implant, rho=800, step=.5).build()
    npt.assert_equal(_spatial(model).grid.v1.z is None, True)
    npt.assert_equal(_spatial(model).grid.v1.x is None, False)
    percept = model.predict_percept({e: 1 for e in implant.electrode_names})
    npt.assert_almost_equal(np.sum(percept.data), 32.494125, decimal=3)
    npt.assert_equal(np.sum(percept.data > .05), 4)


def _straddling_pair(coord):
    """Indices nearest zero from below and above."""
    below = np.flatnonzero(coord < 0)
    above = np.flatnonzero(coord > 0)
    return below[np.argmax(coord[below])], above[np.argmin(coord[above])]


@pytest.mark.parametrize('ModelClass', [ScoreboardModel, ScoreboardSpatial])
def test_CortexSpatial_meridian_blend(ModelClass):
    def make(**params):
        # Offset by half a step so no sample sits exactly on the
        # meridian:
        return ModelClass(xrange=(-5.1, 4.9), yrange=(-5, 5), step=0.2,
                          rho=800, **params).build()

    # Close to the midline, so the phosphenes land on the vertical meridian
    implant = Cortivis(x=5000)
    source = {e: 1 for e in implant.electrode_names}
    plain = make(implant=implant, meridian_blend=0)
    unblended = plain.predict_percept(source).data
    npt.assert_array_less(0, unblended.max())

    default_model = make(implant=implant)
    npt.assert_equal(_spatial(default_model).meridian_blend, 0.1)
    default_data = default_model.predict_percept(source).data
    npt.assert_equal(np.array_equal(default_data, unblended), False)

    width = 0.5
    blended = make(implant=implant,
                   meridian_blend=width).predict_percept(source).data
    npt.assert_equal(blended.shape, unblended.shape)
    npt.assert_equal(blended.dtype, unblended.dtype)

    x = _spatial(plain).grid.x[0, :]
    seam = _straddling_pair(x)

    def jump(data):
        return np.abs(data[:, seam[0], 0] - data[:, seam[1], 0]).max()

    npt.assert_array_less(0, jump(unblended))
    npt.assert_array_less(jump(blended), jump(unblended))

    # The change stays within a few widths of the meridian:
    delta = np.abs(blended - unblended)
    cols = delta.max(axis=(0, 2)) > delta.max() * 1e-3
    npt.assert_equal(np.any(cols), True)
    npt.assert_array_less(np.abs(x[cols]).max(), 4 * width)

    # *vertical* meridian:
    dark_rows = unblended.max(axis=(1, 2)) == 0
    npt.assert_equal(np.any(dark_rows), True)
    npt.assert_array_equal(blended[dark_rows], 0)
    dark_cols = unblended.max(axis=(0, 2)) == 0
    npt.assert_equal(np.any(blended[:, dark_cols] > 0), True)


def test_CortexSpatial_meridian_blend_reapplies_threshold():
    # Blending pulls brightness across the meridian, which could otherwise
    # lift a point that `thresh_percept` had zeroed back off zero.
    implant = Cortivis(x=5000)
    model = ScoreboardModel(implant=implant, xrange=(-5, 5), yrange=(-5, 5),
                            step=0.2, rho=800, meridian_blend=0.5,
                            thresh_percept=0.1).build()
    data = model.predict_percept(
        {e: 1 for e in implant.electrode_names}).data
    npt.assert_equal(np.any(data > 0), True)
    # Nothing survives strictly between zero and the threshold:
    npt.assert_equal(np.any((np.abs(data) > 0) & (np.abs(data) < 0.1)), False)


def _user_warnings(build):
    """The UserWarning messages a build emits, and nothing else"""
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        build()
    return [str(w.message) for w in caught
            if issubclass(w.category, UserWarning)]


def _cortex_grid(ndim):
    return dict(xrange=(-3, 3), yrange=(-3, 3), step=1,
                vfmap=Polimeni2006Map(regions=['v1'], jitter_boundary=True,
                                      ndim=ndim))


@pytest.mark.parametrize('ndim', [2, 3])
def test_cortical_scoreboard_warns_when_rho_is_wider_than_the_pitch(ndim):
    """The same Gaussian spread as the retinal model, so the same warning"""
    grid = _cortex_grid(ndim)
    # Cortivis' 400 um pitch, against a current spread three times as wide:
    said = _user_warnings(
        ScoreboardModel(implant=Cortivis(), rho=1200, **grid).build)
    npt.assert_equal(any('pitch (400 um)' in w for w in said), True)
    npt.assert_equal(any('ratio of 3.00' in w for w in said), True)
    npt.assert_equal(
        _user_warnings(ScoreboardModel(implant=Cortivis(), rho=400,
                                       **grid).build), [])


def test_a_three_dimensional_map_counts_depth_as_spacing():
    """Pitch is measured in whichever dimensions the model reads

    A Neuralink thread stacks its electrodes along z at one (x, y). A 3-D map
    reads that depth and sees 50 um neighbours; a 2-D one projects them onto
    the same point, where there is no spacing left to compare rho against.
    """
    thread = LinearEdgeThread()
    said = _user_warnings(
        ScoreboardModel(implant=thread, rho=200, **_cortex_grid(3)).build)
    npt.assert_equal(any('pitch (50 um)' in w for w in said), True)
    npt.assert_equal(
        _user_warnings(ScoreboardModel(implant=thread, rho=200,
                                       **_cortex_grid(2)).build), [])
