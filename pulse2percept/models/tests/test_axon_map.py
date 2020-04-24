import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept import models
from pulse2percept import stimuli
from pulse2percept import implants


def test_AxonMapModel():
    set_params = {'xystep': 2, 'engine': 'serial', 'rho': 432, 'axlambda': 2,
                  'n_axons': 9, 'n_ax_segments': 50,
                  'xrange': (-30, 30), 'yrange': (-20, 20),
                  'loc_od_x': 5, 'loc_od_y': 6}
    model = models.AxonMapModel()
    for param in set_params:
        npt.assert_equal(hasattr(model.spatial, param), True)

    # User can override default values
    for key, value in set_params.items():
        setattr(model.spatial, key, value)
        npt.assert_equal(getattr(model.spatial, key), value)
    model = models.AxonMapModel(**set_params)
    model.build(**set_params)
    for key, value in set_params.items():
        npt.assert_equal(getattr(model.spatial, key), value)

    # Zeros in, zeros out:
    implant = implants.ArgusII(stim=np.zeros(60))
    npt.assert_almost_equal(model.predict_percept(implant).data, 0)
    implant.stim = np.zeros(60)
    npt.assert_almost_equal(model.predict_percept(implant).data, 0)

    # Implant and model must be built for same eye:
    with pytest.raises(ValueError):
        implant = implants.ArgusII(eye='LE', stim=np.zeros(60))
        model.predict_percept(implant)


@pytest.mark.parametrize('eye', ('LE', 'RE'))
@pytest.mark.parametrize('loc_od', ((15.5, 1.5), (7.0, 3.0), (-2.0, -2.0)))
@pytest.mark.parametrize('sign', (-1.0, 1.0))
def test_AxonMapModel__jansonius2009(eye, loc_od, sign):
    # With `rho` starting at 0, all axons should originate in the optic disc
    # center
    model = models.AxonMapModel(loc_od_x=loc_od[0], loc_od_y=loc_od[1],
                                xystep=2, engine='serial',
                                ax_segments_range=(0, 45),
                                n_ax_segments=100)
    for phi0 in [-135.0, 66.0, 128.0]:
        ax_pos = model.spatial._jansonius2009(phi0)
        npt.assert_almost_equal(ax_pos[0, 0], loc_od[0])
        npt.assert_almost_equal(ax_pos[0, 1], loc_od[1])

    # These axons should all end at the meridian
    for phi0 in [110.0, 135.0, 160.0]:
        model = models.AxonMapModel(loc_od_x=15, loc_od_y=2,
                                    xystep=2, engine='serial',
                                    n_ax_segments=801,
                                    ax_segments_range=(0, 45))
        ax_pos = model.spatial._jansonius2009(sign * phi0)
        print(ax_pos[-1, :])
        npt.assert_almost_equal(ax_pos[-1, 1], 0.0, decimal=1)

    # `phi0` must be within [-180, 180]
    for phi0 in [-200.0, 181.0]:
        with pytest.raises(ValueError):
            failed = models.AxonMapModel(xystep=2, engine='serial')
            failed.spatial._jansonius2009(phi0)

    # `n_rho` must be >= 1
    for n_rho in [-1, 0]:
        with pytest.raises(ValueError):
            model = models.AxonMapModel(n_ax_segments=n_rho, xystep=2,
                                        engine='serial')
            model.spatial._jansonius2009(0.0)

    # `ax_segments_range` must have min <= max
    for lorho in [-200.0, 90.0]:
        with pytest.raises(ValueError):
            model = models.AxonMapModel(ax_segments_range=(lorho, 45), xystep=2,
                                        engine='serial')
            model.spatial._jansonius2009(0)
    for hirho in [-200.0, 40.0]:
        with pytest.raises(ValueError):
            model = models.AxonMapModel(ax_segments_range=(45, hirho), xystep=2,
                                        engine='serial')
            model.spatial._jansonius2009(0)

    # A single axon fiber with `phi0`=0 should return a single pixel location
    # that corresponds to the optic disc
        model = models.AxonMapModel(loc_od_x=loc_od[0], loc_od_y=loc_od[1],
                                    xystep=2, engine='serial', eye=eye,
                                    ax_segments_range=(0, 0),
                                    n_ax_segments=1)
        single_fiber = model.spatial._jansonius2009(0)
        npt.assert_equal(len(single_fiber), 1)
        npt.assert_almost_equal(single_fiber[0], loc_od)


def test_AxonMapModel_grow_axon_bundles():
    for n_axons in [1, 2, 3, 5, 10]:
        model = models.AxonMapModel(xystep=2, engine='serial', n_axons=n_axons,
                                    axons_range=(-20, 20))
        model.build()
        bundles = model.spatial.grow_axon_bundles()
        npt.assert_equal(len(bundles), n_axons)


def test_AxonMapModel_find_closest_axon():
    model = models.AxonMapModel(xystep=1, engine='serial', n_axons=5,
                                axons_range=(-45, 45))
    model.build()
    # Pretend there is an axon close to each point on the grid:
    bundles = [np.array([x + 0.001, y - 0.001]).reshape((1, 2))
               for x, y in zip(model.spatial.grid.xret.ravel(),
                               model.spatial.grid.yret.ravel())]
    closest = model.spatial.find_closest_axon(bundles)
    for ax1, ax2 in zip(bundles, closest):
        npt.assert_almost_equal(ax1[0, 0], ax2[0, 0])
        npt.assert_almost_equal(ax1[0, 1], ax2[0, 1])


def test_AxonMapModel_calc_axon_contribution():
    model = models.AxonMapModel(xystep=2, engine='serial', n_axons=10,
                                axons_range=(-30, 30))
    model.build()
    xyret = np.column_stack((model.spatial.grid.xret.ravel(),
                             model.spatial.grid.yret.ravel()))
    bundles = model.spatial.grow_axon_bundles()
    axons = model.spatial.find_closest_axon(bundles)
    contrib = model.spatial.calc_axon_contribution(axons)

    # Check lambda math:
    for ax, xy in zip(contrib, xyret):
        axon = np.insert(ax, 0, list(xy) + [0], axis=0)
        d2 = np.cumsum(np.diff(axon[:, 0], axis=0) ** 2 +
                       np.diff(axon[:, 1], axis=0) ** 2)
        sensitivity = np.exp(-d2 / (2.0 * model.spatial.axlambda ** 2))
        npt.assert_almost_equal(sensitivity, ax[:, 2])


def test_AxonMapModel__calc_bundle_tangent():
    model = models.AxonMapModel(xystep=5, engine='serial', n_axons=500,
                                n_ax_segments=500, axons_range=(-180, 180),
                                ax_segments_range=(3, 50))
    npt.assert_almost_equal(model.spatial.calc_bundle_tangent(0, 0), 0.4819,
                            decimal=3)
    npt.assert_almost_equal(model.spatial.calc_bundle_tangent(0, 1000),
                            -0.5532, decimal=3)


def test_AxonMapModel_predict_percept():
    model = models.AxonMapModel(xystep=1, axlambda=100, thresh_percept=0)
    model.build()
    # Single-electrode stim:
    img_stim = np.zeros(60)
    img_stim[47] = 1
    percept = model.predict_percept(implants.ArgusII(stim=img_stim))
    # Single bright pixel, rest of arc is less bright:
    npt.assert_equal(np.sum(percept.data > 0.9), 1)
    npt.assert_equal(np.sum(percept.data > 0.5), 2)
    npt.assert_equal(np.sum(percept.data > 0.1), 8)
    npt.assert_equal(np.sum(percept.data > 0.0001), 28)
    # Overall only a few bright pixels:
    npt.assert_almost_equal(np.sum(percept.data), 3.207, decimal=3)
    # Brightest pixel is in lower right:
    npt.assert_almost_equal(percept.data[18, 25, 0], np.max(percept.data))
    # Top half is empty:
    npt.assert_almost_equal(np.sum(percept.data[:15, :, 0]), 0)
    # Same for lower band:
    npt.assert_almost_equal(np.sum(percept.data[21:, :, 0]), 0)

    # Full Argus II with small lambda: 60 bright spots
    model = models.AxonMapModel(engine='serial', xystep=1, rho=100,
                                axlambda=40)
    model.build()
    percept = model.predict_percept(implants.ArgusII(stim=np.ones(60)))
    # Most spots are pretty bright, but there are 2 dimmer ones (due to their
    # location on the retina):
    npt.assert_equal(np.sum(percept.data > 0.5), 58)
    npt.assert_equal(np.sum(percept.data > 0.275), 60)
