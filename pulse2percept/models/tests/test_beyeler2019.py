import numpy as np
import pytest
import numpy.testing as npt

from matplotlib.axes import Subplot
import matplotlib.pyplot as plt


from pulse2percept.implants import ArgusI, ArgusII
from pulse2percept.percepts import Percept
from pulse2percept.models import (AxonMapSpatial, AxonMapModel,
                                  ScoreboardSpatial, ScoreboardModel)
from pulse2percept.utils.testing import assert_warns_msg


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

    # Converting ret <=> dva
    npt.assert_almost_equal(model.ret2dva(0), 0)
    npt.assert_almost_equal(model.dva2ret(0), 0)

    implant = ArgusI(stim=np.zeros(16))
    # Zero in = zero out:
    percept = model.predict_percept(implant)
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [1])
    npt.assert_almost_equal(percept.data, 0)

    # Multiple frames are processed independently:
    model = ScoreboardSpatial(engine='serial', rho=200, xystep=5,
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
    model = ScoreboardModel(engine='serial', rho=200, xystep=5,
                            xrange=(-20, 20), yrange=(-15, 15))
    model.build()
    percept = model.predict_percept(ArgusI(stim={'A1': [1, 2]}))
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [2])
    pmax = percept.data.max(axis=(0, 1))
    npt.assert_almost_equal(percept.data[2, 3, :], pmax)
    npt.assert_almost_equal(pmax[1] / pmax[0], 2.0)
    npt.assert_almost_equal(percept.time, [0, 1])


def test_ScoreboardModel_predict_percept():
    model = ScoreboardModel(xystep=0.55, rho=100, thresh_percept=0,
                            xrange=(-20, 20), yrange=(-15, 15))
    model.build()
    # Single-electrode stim:
    img_stim = np.zeros(60)
    img_stim[47] = 1
    percept = model.predict_percept(ArgusII(stim=img_stim))
    # Single bright pixel, very small Gaussian kernel:
    npt.assert_equal(np.sum(percept.data > 0.8), 1)
    npt.assert_equal(np.sum(percept.data > 0.5), 2)
    npt.assert_equal(np.sum(percept.data > 0.1), 7)
    npt.assert_equal(np.sum(percept.data > 0.00001), 32)
    # Brightest pixel is in lower right:
    npt.assert_almost_equal(percept.data[33, 46, 0], np.max(percept.data))

    # Full Argus II: 60 bright spots
    model = ScoreboardModel(engine='serial', xystep=0.55, rho=100)
    model.build()
    percept = model.predict_percept(ArgusII(stim=np.ones(60)))
    npt.assert_equal(np.sum(np.isclose(percept.data, 0.8, rtol=0.1, atol=0.1)),
                     92)

    # Model gives same outcome as Spatial:
    spatial = ScoreboardSpatial(engine='serial', xystep=1, rho=100)
    spatial.build()
    spatial_percept = model.predict_percept(ArgusII(stim=np.ones(60)))
    npt.assert_almost_equal(percept.data, spatial_percept.data)
    npt.assert_equal(percept.time, None)

    # Warning for nonzero electrode-retina distances
    implant = ArgusI(stim=np.ones(16), z=10)
    msg = ("Nonzero electrode-retina distances do not have any effect on the "
           "model output.")
    assert_warns_msg(UserWarning, model.predict_percept, msg, implant)


@pytest.mark.parametrize('engine', ('serial', 'cython'))
def test_AxonMapSpatial(engine):
    # AxonMapSpatial automatically sets `rho`, `axlambda`:
    model = AxonMapSpatial(engine=engine, xystep=5)

    # User can set `rho`:
    model.rho = 123
    npt.assert_equal(model.rho, 123)
    model.build(rho=987)
    npt.assert_equal(model.rho, 987)

    # Converting ret <=> dva
    npt.assert_almost_equal(model.ret2dva(0), 0)
    npt.assert_almost_equal(model.dva2ret(0), 0)

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(ArgusI()), None)

    # Zero in = zero out:
    implant = ArgusI(stim=np.zeros(16))
    percept = model.predict_percept(implant)
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [1])
    npt.assert_almost_equal(percept.data, 0)
    npt.assert_equal(percept.time, None)

    # Multiple frames are processed independently:
    model = AxonMapSpatial(engine=engine, rho=200, axlambda=100, xystep=5,
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

    # Lambda cannot be too small:
    with pytest.raises(ValueError):
        AxonMapSpatial(axlambda=9).build()


def test_AxonMapSpatial_plot():
    model = AxonMapSpatial()
    for use_dva, xlim in zip([True, False], [(-18, 18), (-5000, 5000)]):
        ax = model.plot(use_dva=use_dva)
        npt.assert_equal(isinstance(ax, Subplot), True)
        npt.assert_equal(ax.get_xlim(), xlim)

    # Quadrants can be annotated:
    for ann_q, n_q in [(True, 6), (False, 0)]:
        fig, ax = plt.subplots()
        model.plot(annotate=ann_q, ax=ax)
        npt.assert_equal(len(ax.child_axes), int(n_q > 0))
        if len(ax.child_axes) > 0:
            npt.assert_equal(len(ax.child_axes[0].texts), n_q)
        plt.close(fig)


@pytest.mark.parametrize('engine', ('serial', 'cython'))
def test_AxonMapModel(engine):
    set_params = {'xystep': 2, 'engine': engine, 'rho': 432, 'axlambda': 20,
                  'n_axons': 9, 'n_ax_segments': 50,
                  'xrange': (-30, 30), 'yrange': (-20, 20),
                  'loc_od': (5, 6)}
    model = AxonMapModel()
    for param in set_params:
        npt.assert_equal(hasattr(model.spatial, param), True)

    # User can override default values
    for key, value in set_params.items():
        setattr(model.spatial, key, value)
        npt.assert_equal(getattr(model.spatial, key), value)
    model = AxonMapModel(**set_params)
    model.build(**set_params)
    for key, value in set_params.items():
        npt.assert_equal(getattr(model.spatial, key), value)

    # Zeros in, zeros out:
    implant = ArgusII(stim=np.zeros(60))
    npt.assert_almost_equal(model.predict_percept(implant).data, 0)
    implant.stim = np.zeros(60)
    npt.assert_almost_equal(model.predict_percept(implant).data, 0)

    # Implant and model must be built for same eye:
    with pytest.raises(ValueError):
        implant = ArgusII(eye='LE', stim=np.zeros(60))
        model.predict_percept(implant)
    with pytest.raises(ValueError):
        AxonMapModel(eye='invalid').build()
    with pytest.raises(ValueError):
        AxonMapModel(xystep=5).build(eye='invalid')

    # Lambda cannot be too small:
    with pytest.raises(ValueError):
        AxonMapModel(axlambda=9).build()


@pytest.mark.parametrize('eye', ('LE', 'RE'))
@pytest.mark.parametrize('loc_od', ((15.5, 1.5), (7.0, 3.0), (-2.0, -2.0)))
@pytest.mark.parametrize('sign', (-1.0, 1.0))
@pytest.mark.parametrize('engine', ('serial', 'cython'))
def test_AxonMapModel__jansonius2009(eye, loc_od, sign, engine):
    # With `rho` starting at 0, all axons should originate in the optic disc
    # center
    model = AxonMapModel(loc_od=loc_od, xystep=2, engine=engine,
                         ax_segments_range=(0, 45),
                         n_ax_segments=100)
    for phi0 in [-135.0, 66.0, 128.0]:
        ax_pos = model.spatial._jansonius2009(phi0)
        npt.assert_almost_equal(ax_pos[0, 0], loc_od[0])
        npt.assert_almost_equal(ax_pos[0, 1], loc_od[1])

    # These axons should all end at the meridian
    for phi0 in [110.0, 135.0, 160.0]:
        model = AxonMapModel(loc_od=(15, 2), xystep=2, engine=engine,
                             n_ax_segments=801,
                             ax_segments_range=(0, 45))
        ax_pos = model.spatial._jansonius2009(sign * phi0)
        npt.assert_almost_equal(ax_pos[-1, 1], 0.0, decimal=1)

    # `phi0` must be within [-180, 180]
    for phi0 in [-200.0, 181.0]:
        with pytest.raises(ValueError):
            failed = AxonMapModel(xystep=2, engine=engine)
            failed.spatial._jansonius2009(phi0)

    # `n_rho` must be >= 1
    for n_rho in [-1, 0]:
        with pytest.raises(ValueError):
            model = AxonMapModel(n_ax_segments=n_rho, xystep=2,
                                 engine=engine)
            model.spatial._jansonius2009(0.0)

    # `ax_segments_range` must have min <= max
    for lorho in [-200.0, 90.0]:
        with pytest.raises(ValueError):
            model = AxonMapModel(ax_segments_range=(lorho, 45), xystep=2,
                                 engine=engine)
            model.spatial._jansonius2009(0)
    for hirho in [-200.0, 40.0]:
        with pytest.raises(ValueError):
            model = AxonMapModel(ax_segments_range=(45, hirho), xystep=2,
                                 engine=engine)
            model.spatial._jansonius2009(0)

    # A single axon fiber with `phi0`=0 should return a single pixel location
    # that corresponds to the optic disc
        model = AxonMapModel(loc_od=loc_od, xystep=2, engine=engine, eye=eye,
                             ax_segments_range=(0, 0),
                             n_ax_segments=1)
        single_fiber = model.spatial._jansonius2009(0)
        npt.assert_equal(len(single_fiber), 1)
        npt.assert_almost_equal(single_fiber[0], loc_od)


@pytest.mark.parametrize('engine', ('serial', 'cython'))
def test_AxonMapModel_grow_axon_bundles(engine):
    for n_axons in [1, 2, 3, 5, 10]:
        model = AxonMapModel(xystep=2, engine=engine, n_axons=n_axons,
                             axons_range=(-20, 20), xrange=(-20, 20),
                             yrange=(-15, 15))
        bundles = model.spatial.grow_axon_bundles()
        npt.assert_equal(len(bundles), n_axons)


@pytest.mark.parametrize('engine', ('serial', 'cython'))
def test_AxonMapModel_find_closest_axon(engine):
    model = AxonMapModel(xystep=1, engine=engine, n_axons=5,
                         xrange=(-20, 20), yrange=(-15, 15),
                         axons_range=(-45, 45))
    model.build()

    # Pretend there is an axon close to each point on the grid:
    bundles = [np.array([x + 0.001, y - 0.001],
                        dtype=np.float32).reshape((1, 2))
               for x, y in zip(model.spatial.grid.xret.ravel(),
                               model.spatial.grid.yret.ravel())]
    closest = model.spatial.find_closest_axon(bundles)
    for ax1, ax2 in zip(bundles, closest):
        npt.assert_almost_equal(ax1[0, 0], ax2[0, 0])
        npt.assert_almost_equal(ax1[0, 1], ax2[0, 1])

    # Looking up just one point does not return a list of axons:
    axon = bundles[0]
    closest = model.spatial.find_closest_axon(bundles, xret=axon[0, 0],
                                              yret=axon[0, 1])
    npt.assert_almost_equal(closest, axon)

    # Return the index as well:
    closest, closest_idx = model.spatial.find_closest_axon(bundles,
                                                           xret=axon[0, 0],
                                                           yret=axon[0, 1],
                                                           return_index=True)
    npt.assert_almost_equal(closest, axon)
    npt.assert_equal(closest_idx, 0)


@pytest.mark.parametrize('engine', ('serial', 'cython'))
def test_AxonMapModel_calc_axon_sensitivity(engine):
    model = AxonMapModel(xystep=2, engine=engine, n_axons=10,
                         xrange=(-20, 20), yrange=(-15, 15),
                         axons_range=(-30, 30))
    model.build()
    xyret = np.column_stack((model.spatial.grid.xret.ravel(),
                             model.spatial.grid.yret.ravel()))
    bundles = model.spatial.grow_axon_bundles()
    axons = model.spatial.find_closest_axon(bundles)
    contrib = model.spatial.calc_axon_sensitivity(axons)

    # Check lambda math:
    for ax, xy in zip(contrib, xyret):
        axon = np.insert(ax, 0, list(xy) + [0], axis=0)
        d2 = np.cumsum(np.sqrt(np.diff(axon[:, 0], axis=0) ** 2 +
                               np.diff(axon[:, 1], axis=0) ** 2))**2
        sensitivity = np.exp(-d2 / (2.0 * model.spatial.axlambda ** 2))
        npt.assert_almost_equal(sensitivity, ax[:, 2])


@pytest.mark.parametrize('engine', ('serial', 'cython'))
def test_AxonMapModel_calc_bundle_tangent(engine):
    model = AxonMapModel(xystep=5, engine=engine, n_axons=500,
                         xrange=(-20, 20), yrange=(-15, 15),
                         n_ax_segments=500, axons_range=(-180, 180),
                         ax_segments_range=(3, 50))
    npt.assert_almost_equal(model.spatial.calc_bundle_tangent(0, 0), 0.4819,
                            decimal=3)
    npt.assert_almost_equal(model.spatial.calc_bundle_tangent(0, 1000),
                            -0.5532, decimal=3)
    with pytest.raises(TypeError):
        model.spatial.calc_bundle_tangent([0], 1000)
    with pytest.raises(TypeError):
        model.spatial.calc_bundle_tangent(0, [1000])


@pytest.mark.parametrize('engine', ('serial', 'cython'))
def test_AxonMapModel_predict_percept(engine):
    model = AxonMapModel(xystep=0.55, axlambda=100, rho=100,
                         thresh_percept=0, engine=engine,
                         xrange=(-20, 20), yrange=(-15, 15),
                         n_axons=500)
    model.build()
    # Single-electrode stim:
    img_stim = np.zeros(60)
    img_stim[47] = 1
    percept = model.predict_percept(ArgusII(stim=img_stim))
    # Single bright pixel, rest of arc is less bright:
    npt.assert_equal(np.sum(percept.data > 0.8), 1)
    npt.assert_equal(np.sum(percept.data > 0.6), 2)
    npt.assert_equal(np.sum(percept.data > 0.1), 7)
    npt.assert_equal(np.sum(percept.data > 0.0001), 32)
    # Overall only a few bright pixels:
    npt.assert_almost_equal(np.sum(percept.data), 3.31321, decimal=3)
    # Brightest pixel is in lower right:
    npt.assert_almost_equal(percept.data[33, 46, 0], np.max(percept.data))
    # Top half is empty:
    npt.assert_almost_equal(np.sum(percept.data[:27, :, 0]), 0)
    # Same for lower band:
    npt.assert_almost_equal(np.sum(percept.data[39:, :, 0]), 0)

    # Full Argus II with small lambda: 60 bright spots
    model = AxonMapModel(engine='serial', xystep=1, rho=100, axlambda=40,
                         xrange=(-20, 20), yrange=(-15, 15), n_axons=500)
    model.build()
    percept = model.predict_percept(ArgusII(stim=np.ones(60)))
    # Most spots are pretty bright, but there are 2 dimmer ones (due to their
    # location on the retina):
    npt.assert_equal(np.sum(percept.data > 0.5), 28)
    npt.assert_equal(np.sum(percept.data > 0.275), 56)

    # Model gives same outcome as Spatial:
    spatial = AxonMapSpatial(engine='serial', xystep=1, rho=100, axlambda=40,
                             xrange=(-20, 20), yrange=(-15, 15), n_axons=500)
    spatial.build()
    spatial_percept = spatial.predict_percept(ArgusII(stim=np.ones(60)))
    npt.assert_almost_equal(percept.data, spatial_percept.data)
    npt.assert_equal(percept.time, None)

    # Warning for nonzero electrode-retina distances
    implant = ArgusI(stim=np.ones(16), z=10)
    msg = ("Nonzero electrode-retina distances do not have any effect on the "
           "model output.")
    assert_warns_msg(UserWarning, model.predict_percept, msg, implant)
