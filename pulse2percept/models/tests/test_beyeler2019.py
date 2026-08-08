from types import SimpleNamespace
import numpy as np
import pytest
import numpy.testing as npt
import copy
import os
import pickle
import warnings

from matplotlib.axes import Subplot
import matplotlib.pyplot as plt


from pulse2percept.implants import ArgusI, ArgusII
from pulse2percept.percepts import Percept
from pulse2percept.stimuli import Stimulus
from pulse2percept.models import (AxonMapSpatial, AxonMapModel,
                                  ScoreboardSpatial, ScoreboardModel)
from pulse2percept.models.beyeler2019 import _AXON_CACHE_VERSION
from pulse2percept.topography import Watson2014Map, Watson2014DisplaceMap
from pulse2percept.utils.testing import assert_warns_msg

# Building an axon map writes a cache to a relative path; keep it in a
# temporary directory instead of wherever pytest was started from:
pytestmark = pytest.mark.usefixtures('axon_cache_in_tmp')


def test_ScoreboardSpatial():
    # ScoreboardSpatial automatically sets `rho`:
    model = ScoreboardSpatial(xystep=5)

    # User can set `rho`:
    model.rho = 123
    npt.assert_equal(model.rho, 123)
    model.build(rho=987)
    npt.assert_equal(model.rho, 987)

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(ArgusI()), None)

    # Converting ret <=> dva
    npt.assert_equal(isinstance(model.vfmap, Watson2014Map), True)
    npt.assert_almost_equal(model.vfmap.ret_to_dva(0, 0), (0, 0))
    npt.assert_almost_equal(model.vfmap.dva_to_ret(0, 0), (0, 0))
    model2 = ScoreboardSpatial(vfmap=Watson2014DisplaceMap())
    npt.assert_equal(isinstance(model2.vfmap, Watson2014DisplaceMap),
                     True)

    implant = ArgusI(stim=np.zeros(16))
    # Zero in = zero out:
    percept = model.predict_percept(implant)
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [1])
    npt.assert_almost_equal(percept.data, 0)

    # Multiple frames are processed independently:
    model = ScoreboardSpatial(rho=200, xystep=5,
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


def test_deepcopy_ScoreboardSpatial():
    original = ScoreboardSpatial()
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

def test_ScoreboardModel():
    # ScoreboardModel automatically sets `rho`:
    model = ScoreboardModel(xystep=5)
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

    # Converting ret <=> dva
    npt.assert_equal(isinstance(model.vfmap, Watson2014Map), True)
    npt.assert_almost_equal(model.vfmap.ret_to_dva(0, 0), (0, 0))
    npt.assert_almost_equal(model.vfmap.dva_to_ret(0, 0), (0, 0))
    model2 = ScoreboardModel(vfmap=Watson2014DisplaceMap())
    npt.assert_equal(isinstance(model2.vfmap, Watson2014DisplaceMap),
                     True)
    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(ArgusI()), None)

    # Zero in = zero out:
    implant = ArgusI(stim=np.zeros(16))
    npt.assert_almost_equal(model.predict_percept(implant).data, 0)

    # Multiple frames are processed independently:
    model = ScoreboardModel(rho=200, xystep=5,
                            xrange=(-20, 20), yrange=(-15, 15))
    model.build()
    percept = model.predict_percept(ArgusI(stim={'A1': [1, 2]}))
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [2])
    pmax = percept.data.max(axis=(0, 1))
    npt.assert_almost_equal(percept.data[2, 3, :], pmax)
    npt.assert_almost_equal(pmax[1] / pmax[0], 2.0)
    npt.assert_almost_equal(percept.time, [0, 1])


def test_deepcopy_ScoreboardModel():
    original = ScoreboardModel()
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
    model = ScoreboardModel(xystep=0.55, rho=100)
    model.build()
    percept = model.predict_percept(ArgusII(stim=np.ones(60)))
    npt.assert_equal(np.sum(np.isclose(percept.data, 0.8, rtol=0.1, atol=0.1)),
                     88)

    # Model gives same outcome as Spatial:
    spatial = ScoreboardSpatial(xystep=1, rho=100)
    spatial.build()
    spatial_percept = model.predict_percept(ArgusII(stim=np.ones(60)))
    npt.assert_almost_equal(percept.data, spatial_percept.data)
    npt.assert_equal(percept.time, None)

    # Warning for nonzero electrode-retina distances
    implant = ArgusI(stim=np.ones(16), z=10)
    msg = ("Nonzero electrode-retina distances do not have any effect on the "
           "model output.")
    assert_warns_msg(UserWarning, model.predict_percept, msg, implant)


def test_AxonMapSpatial():
    # AxonMapSpatial automatically sets `rho`, `axlambda`:
    model = AxonMapSpatial(xystep=5)

    # User can set `rho`:
    model.rho = 123
    npt.assert_equal(model.rho, 123)
    model.build(rho=987)
    npt.assert_equal(model.rho, 987)

    # Converting ret <=> dva
    npt.assert_equal(isinstance(model.vfmap, Watson2014Map), True)
    npt.assert_almost_equal(model.vfmap.ret_to_dva(0, 0), (0, 0))
    npt.assert_almost_equal(model.vfmap.dva_to_ret(0, 0), (0, 0))
    model2 = AxonMapSpatial(vfmap=Watson2014DisplaceMap())
    npt.assert_equal(isinstance(model2.vfmap, Watson2014DisplaceMap),
                     True)

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(ArgusI()), None)

    # Zero in = zero out:
    implant = ArgusI(stim=np.zeros(16))
    percept = model.predict_percept(implant)
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [1])
    npt.assert_almost_equal(percept.data, 0)
    npt.assert_equal(percept.time, None)

    # Lambda cannot be too small:
    with pytest.raises(ValueError):
        AxonMapSpatial(axlambda=9).build()

    # Multiple frames are processed independently:
    model = AxonMapSpatial(rho=200, axlambda=100, xystep=5,
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


def test_deepcopy_AxonMapSpatial():
    original = AxonMapSpatial()
    copied = copy.deepcopy(original)

    # Assert these are two different objects
    npt.assert_equal(id(original) != id(copied), True)

    # Assert these objects are equivalent
    npt.assert_equal(original.__dict__, copied.__dict__)
    npt.assert_equal(original == copied, True)

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

def test_AxonMapSpatial_plot():
    model = AxonMapSpatial()
    for use_dva, xlim in zip([True, False], [(-18, 18), (-5000, 5000)]):
        ax = model.plot(use_dva=use_dva)
        npt.assert_equal(isinstance(ax, Subplot), True)
        npt.assert_equal(ax.get_xlim(), xlim)
    # Simulated area might be larger than that:
    model = AxonMapSpatial(xrange=(-20.5, 20.5), yrange=(-16.1, 16.1))
    ax = model.plot(use_dva=True)
    npt.assert_almost_equal(ax.get_xlim(), (-21, 21))
    npt.assert_almost_equal(ax.get_ylim(), (-18, 18))
    ax = model.plot(use_dva=False)
    npt.assert_almost_equal(ax.get_xlim(), (-6000, 6000))
    npt.assert_almost_equal(ax.get_ylim(), (-5000, 5000))

    # Figure size can be changed:
    ax = model.plot(figsize=(8, 7))
    npt.assert_almost_equal(ax.figure.get_size_inches(), (8, 7))

    # Quadrants can be annotated:
    for ann_q, n_q in [(True, 6), (False, 0)]:
        fig, ax = plt.subplots()
        model.plot(annotate=ann_q, ax=ax)
        npt.assert_equal(len(ax.child_axes), int(n_q > 0))
        if len(ax.child_axes) > 0:
            npt.assert_equal(len(ax.child_axes[0].texts), n_q)
        plt.close(fig)


def test_AxonMapModel():
    set_params = {'xystep': 2, 'rho': 432, 'axlambda': 20,
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

    # Converting ret <=> dva
    npt.assert_equal(isinstance(model.vfmap, Watson2014Map), True)
    npt.assert_almost_equal(model.vfmap.ret_to_dva(0, 0), (0, 0))
    npt.assert_almost_equal(model.vfmap.dva_to_ret(0, 0), (0, 0))
    model2 = AxonMapModel(vfmap=Watson2014DisplaceMap())
    npt.assert_equal(isinstance(model2.vfmap, Watson2014DisplaceMap),
                     True)

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

# Build the model inside the test, not in the decorator: arguments to
# `parametrize` are evaluated at import time, so building here would run on
# every pytest invocation (even `--collect-only`, even when this test is
# deselected) and would surface any failure as a collection error.
@pytest.mark.parametrize('build', (False, True))
def test_deepcopy_AxonMapModel(build):
    original = AxonMapModel()
    if build:
        original.build()
    copied = copy.deepcopy(original)

    # Assert these are two different objects
    npt.assert_equal(id(original) != id(copied), True)

    # Assert the objects are equivalent
    npt.assert_equal(original.__dict__, copied.__dict__)
    
    # Assert that __eq__ works
    npt.assert_equal(original == copied, True)

    # Assert they do not share the same AxonMapSpatial Object
    npt.assert_equal(original.spatial == copied.spatial, True)
    npt.assert_equal(id(original.spatial) != id(copied.spatial), True)

    # Assert changing copied doesn't change original
    copied.spatial.xrange = (-10, 10)
    npt.assert_equal(original.spatial != copied.spatial, True)


@ pytest.mark.parametrize('eye', ('LE', 'RE'))
@ pytest.mark.parametrize('loc_od', ((15.5, 1.5), (7.0, 3.0), (-2.0, -2.0)))
@ pytest.mark.parametrize('sign', (-1.0, 1.0))
def test_AxonMapModel__jansonius2009(eye, loc_od, sign):
    # With `rho` starting at 0, all axons should originate in the optic disc
    # center
    model = AxonMapModel(loc_od=loc_od, xystep=2,
                         ax_segments_range=(0, 45),
                         n_ax_segments=100)
    for phi0 in [-135.0, 66.0, 128.0]:
        ax_pos = model.spatial._jansonius2009(phi0)
        npt.assert_almost_equal(ax_pos[0, 0], loc_od[0])
        npt.assert_almost_equal(ax_pos[0, 1], loc_od[1])

    # These axons should all end at the meridian
    for phi0 in [110.0, 135.0, 160.0]:
        model = AxonMapModel(loc_od=(15, 2), xystep=2,
                             n_ax_segments=801,
                             ax_segments_range=(0, 45))
        ax_pos = model.spatial._jansonius2009(sign * phi0)
        npt.assert_almost_equal(ax_pos[-1, 1], 0.0, decimal=1)

    # `phi0` must be within [-180, 180]
    for phi0 in [-200.0, 181.0]:
        with pytest.raises(ValueError):
            failed = AxonMapModel(xystep=2)
            failed.spatial._jansonius2009(phi0)

    # `n_rho` must be >= 1
    for n_rho in [-1, 0]:
        with pytest.raises(ValueError):
            model = AxonMapModel(n_ax_segments=n_rho, xystep=2)
            model.spatial._jansonius2009(0.0)

    # `ax_segments_range` must have min <= max
    for lorho in [-200.0, 90.0]:
        with pytest.raises(ValueError):
            model = AxonMapModel(ax_segments_range=(lorho, 45), xystep=2)
            model.spatial._jansonius2009(0)
    for hirho in [-200.0, 40.0]:
        with pytest.raises(ValueError):
            model = AxonMapModel(ax_segments_range=(45, hirho), xystep=2)
            model.spatial._jansonius2009(0)

    # A single axon fiber with `phi0`=0 should return a single pixel location
    # that corresponds to the optic disc
        model = AxonMapModel(loc_od=loc_od, xystep=2, eye=eye,
                             ax_segments_range=(0, 0),
                             n_ax_segments=1)
        single_fiber = model.spatial._jansonius2009(0)
        npt.assert_equal(len(single_fiber), 1)
        npt.assert_almost_equal(single_fiber[0], loc_od)


def test_AxonMapModel_grow_axon_bundles():
    for n_axons in [1, 2, 3, 5, 10]:
        model = AxonMapModel(xystep=2, n_axons=n_axons,
                             axons_range=(-20, 20), xrange=(-20, 20),
                             yrange=(-15, 15))
        bundles = model.spatial.grow_axon_bundles()
        npt.assert_equal(len(bundles), n_axons)


def test_AxonMapModel_find_closest_axon():
    model = AxonMapModel(xystep=1, n_axons=5,
                         xrange=(-20, 20), yrange=(-15, 15),
                         axons_range=(-45, 45))
    model.build()

    # Pretend there is an axon close to each point on the grid:
    bundles = [np.array([x + 0.001, y - 0.001],
                        dtype=np.float32).reshape((1, 2))
               for x, y in zip(model.spatial.grid.ret.x.ravel(),
                               model.spatial.grid.ret.y.ravel())]
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


def test_AxonMapModel_calc_axon_sensitivity():
    model = AxonMapModel(xystep=2, n_axons=10,
                         xrange=(-20, 20), yrange=(-15, 15),
                         axons_range=(-30, 30))
    model.build()
    xyret = np.column_stack((model.spatial.grid.ret.x.ravel(),
                             model.spatial.grid.ret.y.ravel()))
    bundles = model.spatial.grow_axon_bundles()
    axons = model.spatial.find_closest_axon(bundles)
    axon_contrib = model.spatial.calc_axon_sensitivity(axons)

    # Check lambda math. `calc_axon_sensitivity` walks the axon in float64
    # and rounds once at the end, so the reference has to be built the same
    # way: `model_ax` is float32, and accumulating the arc length at that
    # precision costs about as much accuracy as the whole comparison has to
    # spare. Building it here in float32 left roughly a 1.2x margin against
    # the tolerance, which held on some platforms and not on others.
    max_d2 = -2.0 * model.axlambda ** 2 * np.log(model.min_ax_sensitivity)
    for model_ax, xy in zip(axon_contrib, xyret):
        axon = np.insert(model_ax, 0, list(xy) + [0],
                         axis=0).astype(np.float64)
        d2 = np.cumsum(np.sqrt(np.diff(axon[:, 0], axis=0) ** 2 +
                               np.diff(axon[:, 1], axis=0) ** 2))**2
        idx_d2 = d2 < max_d2
        sensitivity = np.exp(-d2[idx_d2] / (2.0 * model.spatial.axlambda ** 2))
        # A relative bound, unlike `assert_almost_equal`'s absolute one: the
        # sensitivities span [min_ax_sensitivity, 1], and float32 resolves
        # them to ~1.2e-7 relative wherever they sit in that range.
        npt.assert_allclose(model_ax[:, 2], sensitivity, rtol=1e-6)


@ pytest.mark.parametrize('pad', (True, False))
def test_AxonMapModel_calc_axon_sensitivity_deprecated_pad(pad):
    # 'pad' used to pad all axons to the length of the longest one for the
    # (now removed) jax backend. It is still accepted, but ignored:
    model = AxonMapModel(xystep=2, n_axons=10, xrange=(-20, 20),
                         yrange=(-15, 15), axons_range=(-30, 30))
    model.build()
    axons = model.spatial.find_closest_axon(model.spatial.grow_axon_bundles())
    with pytest.deprecated_call():
        deprecated = model.spatial.calc_axon_sensitivity(axons, pad=pad)
    # Always the unpadded list, even for pad=True:
    expected = model.spatial.calc_axon_sensitivity(axons)
    npt.assert_equal(isinstance(deprecated, list), True)
    npt.assert_equal(len(deprecated), len(expected))
    for ax_dep, ax_exp in zip(deprecated, expected):
        npt.assert_almost_equal(ax_dep, ax_exp)
    # Not passing it does not warn:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        model.spatial.calc_axon_sensitivity(axons)


def test_AxonMapModel_calc_bundle_tangent():
    model = AxonMapModel(xystep=5, n_axons=500,
                         xrange=(-20, 20), yrange=(-15, 15),
                         n_ax_segments=500, axons_range=(-180, 180),
                         ax_segments_range=(3, 50))
    npt.assert_almost_equal(model.spatial.calc_bundle_tangent(0, 0), -0.4819,
                            decimal=3)
    npt.assert_almost_equal(model.spatial.calc_bundle_tangent(0, 1000),
                            -0.268, decimal=3)
    with pytest.raises(TypeError):
        model.spatial.calc_bundle_tangent([0], 1000)
    with pytest.raises(TypeError):
        model.spatial.calc_bundle_tangent(0, [1000])


def test_AxonMapModel_calc_bundle_tangent_fast():
    model = AxonMapModel(xystep=5, n_axons=500,
                         xrange=(-20, 20), yrange=(-15, 15),
                         n_ax_segments=500, axons_range=(-180, 180),
                         ax_segments_range=(3, 50))
    npt.assert_almost_equal(model.spatial.calc_bundle_tangent_fast(0, 0), -0.4819,
                            decimal=3)
    npt.assert_almost_equal(model.spatial.calc_bundle_tangent_fast(0, 1000),
                            -0.268, decimal=3)
    
    npt.assert_almost_equal(model.spatial.calc_bundle_tangent_fast(np.array([0, 0.]), np.array([0, 1000.])),
                            np.array([-0.4819, -0.268]), decimal=3)



def test_AxonMapModel_predict_percept():
    model = AxonMapModel(xystep=0.55, axlambda=100, rho=100,
                         thresh_percept=0,
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
    npt.assert_almost_equal(np.sum(percept.data), 3.4062, decimal=3)
    # Brightest pixel is in lower right:
    npt.assert_almost_equal(percept.data[33, 46, 0], np.max(percept.data))
    # Top half is empty:
    npt.assert_almost_equal(np.sum(percept.data[:27, :, 0]), 0)
    # Same for lower band:
    npt.assert_almost_equal(np.sum(percept.data[39:, :, 0]), 0)

    # Full Argus II with small lambda: 60 bright spots
    model = AxonMapModel(xystep=1, rho=100, axlambda=40,
                         xrange=(-20, 20), yrange=(-15, 15), n_axons=500)
    model.build()
    percept = model.predict_percept(ArgusII(stim=np.ones(60)))
    # Most spots are pretty bright, but there are 2 dimmer ones (due to their
    # location on the retina):
    npt.assert_equal(np.sum(percept.data > 0.5), 28)
    npt.assert_equal(np.sum(percept.data > 0.275), 56)

    # Model gives same outcome as Spatial:
    spatial = AxonMapSpatial(xystep=1, rho=100, axlambda=40,
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


@pytest.mark.parametrize('ModelClass', (ScoreboardModel, AxonMapModel))
def test_min_current_spread(ModelClass):
    """The default current-spread cutoff must not change the percept.

    ``min_current_spread`` drops an electrode's contribution once its
    Gaussian has decayed past the given fraction of its peak. At the default
    of 1e-8 the dropped term is below what a float32 sum can resolve, so it
    is meant to buy speed at no cost to the result.
    """
    stim = np.zeros(60)
    stim[[10, 33, 47]] = [1.0, -0.5, 0.75]
    implant = ArgusII(stim=stim)
    kwargs = {'xystep': 0.75, 'xrange': (-12, 12), 'yrange': (-8, 8),
              'rho': 200}

    exact = ModelClass(min_current_spread=0,
                       **kwargs).build().predict_percept(implant).data
    default = ModelClass(**kwargs).build().predict_percept(implant).data
    npt.assert_allclose(default, exact, rtol=1e-5,
                        atol=1e-6 * np.abs(exact).max())

    # A coarse cutoff *does* change the result, which is how we know the
    # parameter reaches the kernel at all:
    coarse = ModelClass(min_current_spread=0.5,
                        **kwargs).build().predict_percept(implant).data
    assert np.abs(coarse - exact).max() > 1e-3

    # A cutoff of 1 or more would drop every electrode:
    model = ModelClass(min_current_spread=1, **kwargs).build()
    with pytest.raises(ValueError):
        model.predict_percept(implant)


@pytest.mark.parametrize('ModelClass', (ScoreboardModel, AxonMapModel))
def test_predict_percept_frames_are_independent(ModelClass):
    """Each frame of a multi-frame stimulus is predicted on its own.

    The spatial kernels evaluate the electrode-to-point Gaussian once and
    reuse it across every time point, so this guards against one frame
    leaking into another.
    """
    rng = np.random.default_rng(42)
    data = rng.normal(size=(60, 4)).astype(np.float32)
    model = ModelClass(xystep=1, xrange=(-10, 10), yrange=(-8, 8),
                       rho=200).build()

    joint = model.predict_percept(
        ArgusII(stim=Stimulus(data, time=[0, 1, 2, 3]))).data
    npt.assert_equal(joint.shape[-1], 4)
    for i in range(data.shape[1]):
        frame = model.predict_percept(
            ArgusII(stim=Stimulus(data[:, i:i + 1]))).data
        npt.assert_allclose(joint[..., i], frame[..., 0], rtol=1e-5,
                            atol=1e-6 * np.abs(frame).max())


@pytest.mark.parametrize('ModelClass', (ScoreboardModel, AxonMapModel))
def test_predict_percept_all_zero_stim(ModelClass):
    """An all-zero stimulus produces an all-zero percept.

    The kernels skip electrodes that are zero for the whole stimulus, so the
    case where *every* electrode is skipped is worth pinning down.
    """
    model = ModelClass(xystep=1, xrange=(-10, 10), yrange=(-8, 8)).build()
    percept = model.predict_percept(ArgusII(stim=np.zeros(60)))
    npt.assert_equal(np.all(percept.data == 0), True)


@pytest.mark.parametrize('ModelClass', (ScoreboardModel, AxonMapModel))
def test_predict_percept_thread_count_invariant(ModelClass):
    """The percept must not depend on how many threads computed it.

    ``fast_axon_map`` hands each thread its own row of a scratch buffer, so
    this covers both the indexing of that buffer and the case where the
    stimulus has a single frame (the padding that keeps two threads off the
    same cache line).
    """
    stim = np.zeros(60)
    stim[[5, 22, 51]] = [1.0, 0.6, -0.3]
    implant = ArgusII(stim=stim)
    kwargs = {'xystep': 1, 'xrange': (-10, 10), 'yrange': (-8, 8),
              'rho': 200}

    serial = ModelClass(n_threads=1,
                        **kwargs).build().predict_percept(implant).data
    for n_threads in (2, 3, 8):
        parallel = ModelClass(
            n_threads=n_threads, **kwargs).build().predict_percept(implant)
        npt.assert_array_equal(parallel.data, serial)


def test_AxonMapModel_find_closest_axon_return_segment():
    """``return_segment`` reports where in the axon the closest point is."""
    model = AxonMapModel(xystep=2, n_axons=20, xrange=(-12, 12),
                         yrange=(-12, 12), axons_range=(-45, 45))
    model.build()
    spatial = model.spatial
    bundles = spatial.grow_axon_bundles()
    xyret = np.column_stack((spatial.grid.ret.x.ravel(),
                             spatial.grid.ret.y.ravel()))

    axons, idx_seg = spatial.find_closest_axon(bundles, return_segment=True)
    npt.assert_equal(len(idx_seg), len(xyret))
    # The reported segment is the one `argmin` would have picked:
    for axon, seg, xy in zip(axons, idx_seg, xyret):
        expected = np.argmin((axon[:, 0] - xy[0]) ** 2 +
                             (axon[:, 1] - xy[1]) ** 2)
        npt.assert_equal(seg, expected)

    # Both flags together, in the documented order:
    axons2, idx_ax, idx_seg2 = spatial.find_closest_axon(
        bundles, return_index=True, return_segment=True)
    npt.assert_array_equal(idx_seg2, idx_seg)
    for axon, idx in zip(axons2, idx_ax):
        npt.assert_array_equal(axon, bundles[idx])

    # A single query point still returns scalars, not arrays:
    single, idx_ax1, idx_seg1 = spatial.find_closest_axon(
        bundles, xret=xyret[0, 0], yret=xyret[0, 1], return_index=True,
        return_segment=True)
    npt.assert_equal(np.ndim(idx_ax1), 0)
    npt.assert_equal(np.ndim(idx_seg1), 0)
    npt.assert_array_equal(single, bundles[idx_ax1])


def test_AxonMapModel_calc_axon_sensitivity_empty_bundle():
    """A bundle with no segments is rejected rather than silently skipped."""
    model = AxonMapModel(xystep=4, n_axons=5, xrange=(-8, 8), yrange=(-8, 8))
    model.build()
    n_points = model.spatial.grid.ret.x.size
    bundles = [np.zeros((0, 2), dtype=np.float32)] * n_points
    with pytest.raises(ValueError):
        model.spatial.calc_axon_sensitivity(bundles)


def test_AxonMapModel_build_cache_roundtrip(tmp_path):
    """A warm build off the cache reproduces the cold build exactly."""
    pickle_file = str(tmp_path / 'axons.pickle')

    def build(ignore_pickle):
        return AxonMapModel(xystep=1, xrange=(-8, 8), yrange=(-8, 8),
                            n_axons=200, axon_pickle=pickle_file,
                            ignore_pickle=ignore_pickle).build().spatial

    cold = build(True)
    npt.assert_equal(os.path.isfile(pickle_file), True)
    warm = build(False)
    npt.assert_array_equal(warm.axon_contrib, cold.axon_contrib)
    npt.assert_array_equal(warm.axon_idx_start, cold.axon_idx_start)
    npt.assert_array_equal(warm.axon_idx_end, cold.axon_idx_end)

    # A cache written by an older version is regrown, not misread:
    with open(pickle_file, 'rb') as f:
        params, _ = pickle.load(f)
    with open(pickle_file, 'wb') as f:
        pickle.dump((params, [np.zeros((3, 2), dtype=np.float32)]), f)
    stale = build(False)
    npt.assert_array_equal(stale.axon_contrib, cold.axon_contrib)
    # ...and the file is left in the current format:
    with open(pickle_file, 'rb') as f:
        _, payload = pickle.load(f)
    npt.assert_equal(payload[0], _AXON_CACHE_VERSION)
