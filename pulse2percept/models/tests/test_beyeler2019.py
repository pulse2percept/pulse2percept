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
    model = ScoreboardSpatial(step=5)

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
    model = ScoreboardSpatial(rho=200, step=5,
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
    model = ScoreboardModel(step=5)
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
    model = ScoreboardModel(rho=200, step=5,
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
    model = ScoreboardModel(step=0.55, rho=100, thresh_percept=0,
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
    model = ScoreboardModel(step=0.55, rho=100)
    model.build()
    percept = model.predict_percept(ArgusII(stim=np.ones(60)))
    npt.assert_equal(np.sum(np.isclose(percept.data, 0.8, rtol=0.1, atol=0.1)),
                     88)

    # Model gives same outcome as Spatial:
    spatial = ScoreboardSpatial(step=1, rho=100)
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
    # AxonMapSpatial automatically sets `rho`, `lam`:
    model = AxonMapSpatial(step=5)

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
        AxonMapSpatial(lam=9).build()

    # Multiple frames are processed independently:
    model = AxonMapSpatial(rho=200, lam=100, step=5,
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
    set_params = {'step': 2, 'rho': 432, 'lam': 20,
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
        AxonMapModel(step=5).build(eye='invalid')

    # Lambda cannot be too small:
    with pytest.raises(ValueError):
        AxonMapModel(lam=9).build()


@pytest.mark.parametrize('cls', [AxonMapSpatial, AxonMapModel])
def test_AxonMap_deprecated_axlambda(cls):
    # `lam` was called `axlambda` until 0.10.0. The old name still works,
    # everywhere the new one does, but warns:
    msg = "The 'axlambda' parameter of"
    assert_warns_msg(DeprecationWarning, cls, msg, axlambda=400)
    with pytest.warns(DeprecationWarning):
        model = cls(axlambda=400)
    npt.assert_equal(model.lam, 400)

    # Setting and getting the attribute:
    assert_warns_msg(DeprecationWarning, setattr, msg, model, 'axlambda', 500)
    npt.assert_equal(model.lam, 500)
    with pytest.warns(DeprecationWarning):
        npt.assert_equal(model.axlambda, 500)

    # And `set_params` and `build`. `Model.set_params` takes a dict, whereas
    # `SpatialModel.set_params` takes keyword arguments:
    if cls is AxonMapModel:
        set_params = lambda: model.set_params({'axlambda': 600})
    else:
        set_params = lambda: model.set_params(axlambda=600)
    assert_warns_msg(DeprecationWarning, set_params, msg)
    npt.assert_equal(model.lam, 600)
    # `build` reads the axon cache, which raises a ResourceWarning of its own,
    # so this one cannot insist on a single warning:
    with pytest.warns(DeprecationWarning, match="'axlambda' parameter"):
        model.build(axlambda=700)
    npt.assert_equal(model.lam, 700)

    # The new name stays silent:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        model = cls(lam=400)
        model.lam = 500
        npt.assert_equal(model.lam, 500)


@pytest.mark.parametrize('cls', [AxonMapSpatial, AxonMapModel])
def test_AxonMap_axlambda_and_lam_collide(cls):
    # `axlambda` and `lam` are the same parameter, so supplying both must
    # raise rather than let the order they were passed in decide the value.
    # `**kwargs` preserves insertion order, so check both spellings:
    for params in ({'axlambda': 400, 'lam': 500},
                   {'lam': 500, 'axlambda': 400}):
        with pytest.raises(TypeError, match="same parameter"):
            cls(**params)
        model = cls(step=5)
        with pytest.raises(TypeError, match="same parameter"):
            model.build(**params)
        with pytest.raises(TypeError, match="same parameter"):
            if cls is AxonMapModel:
                model.set_params(params)
            else:
                model.set_params(**params)


@pytest.mark.parametrize('cls', [AxonMapSpatial, AxonMapModel])
def test_AxonMap_axlambda_warning_blames_caller(cls):
    # A deprecation warning is only actionable if it points at the line that
    # used the old name. The alias is reached directly on a spatial model, but
    # through `Model.__getattr__`/`__setattr__` on a composite one, so the
    # attribution has to hold for both:
    model = cls(step=5)
    with pytest.warns(DeprecationWarning) as record:
        model.axlambda
    npt.assert_equal(record[0].filename, __file__)
    with pytest.warns(DeprecationWarning) as record:
        model.axlambda = 400
    npt.assert_equal(record[0].filename, __file__)
    # The constructor reaches it through a chain of `super().__init__` calls
    # instead, whose depth differs between the two classes:
    with pytest.warns(DeprecationWarning) as record:
        cls(axlambda=400)
    npt.assert_equal(record[0].filename, __file__)


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
    model = AxonMapModel(loc_od=loc_od, step=2,
                         ax_segments_range=(0, 45),
                         n_ax_segments=100)
    for phi0 in [-135.0, 66.0, 128.0]:
        ax_pos = model.spatial._jansonius2009(phi0)
        npt.assert_almost_equal(ax_pos[0, 0], loc_od[0])
        npt.assert_almost_equal(ax_pos[0, 1], loc_od[1])

    # These axons should all end at the meridian
    for phi0 in [110.0, 135.0, 160.0]:
        model = AxonMapModel(loc_od=(15, 2), step=2,
                             n_ax_segments=801,
                             ax_segments_range=(0, 45))
        ax_pos = model.spatial._jansonius2009(sign * phi0)
        npt.assert_almost_equal(ax_pos[-1, 1], 0.0, decimal=1)

    # `phi0` must be within [-180, 180]
    for phi0 in [-200.0, 181.0]:
        with pytest.raises(ValueError):
            failed = AxonMapModel(step=2)
            failed.spatial._jansonius2009(phi0)

    # `n_rho` must be >= 1
    for n_rho in [-1, 0]:
        with pytest.raises(ValueError):
            model = AxonMapModel(n_ax_segments=n_rho, step=2)
            model.spatial._jansonius2009(0.0)

    # `ax_segments_range` must have min <= max
    for lorho in [-200.0, 90.0]:
        with pytest.raises(ValueError):
            model = AxonMapModel(ax_segments_range=(lorho, 45), step=2)
            model.spatial._jansonius2009(0)
    for hirho in [-200.0, 40.0]:
        with pytest.raises(ValueError):
            model = AxonMapModel(ax_segments_range=(45, hirho), step=2)
            model.spatial._jansonius2009(0)

    # A single axon fiber with `phi0`=0 should return a single pixel location
    # that corresponds to the optic disc
        model = AxonMapModel(loc_od=loc_od, step=2, eye=eye,
                             ax_segments_range=(0, 0),
                             n_ax_segments=1)
        single_fiber = model.spatial._jansonius2009(0)
        npt.assert_equal(len(single_fiber), 1)
        npt.assert_almost_equal(single_fiber[0], loc_od)


def test_AxonMapModel_grow_axon_bundles():
    for n_axons in [1, 2, 3, 5, 10]:
        model = AxonMapModel(step=2, n_axons=n_axons,
                             axons_range=(-20, 20), xrange=(-20, 20),
                             yrange=(-15, 15))
        bundles = model.spatial.grow_axon_bundles()
        npt.assert_equal(len(bundles), n_axons)


def test_AxonMapModel_find_closest_axon():
    model = AxonMapModel(step=1, n_axons=5,
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


@pytest.mark.parametrize('n_threads', (1, 3))
def test_AxonMapModel_find_closest_axon_respects_n_threads(monkeypatch,
                                                           n_threads):
    """The KD-tree query stays inside the model's thread budget.

    ``n_threads``/``n_jobs`` is the one knob this package gives for capping
    CPU use, and the tree query is part of ``build``. Passing ``workers=-1``
    here would let ``AxonMapModel(n_threads=1).build()`` fan out over every
    core anyway.
    """
    from pulse2percept.models import beyeler2019

    seen = []

    class RecordingKDTree(beyeler2019.cKDTree):
        def query(self, *args, **kwargs):
            seen.append(kwargs.get('workers'))
            return super().query(*args, **kwargs)

    monkeypatch.setattr(beyeler2019, 'cKDTree', RecordingKDTree)
    model = AxonMapSpatial(n_threads=n_threads)
    bundles = [np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
               np.array([[10.0, 10.0], [11.0, 11.0]], dtype=np.float32)]
    closest, idx = model.find_closest_axon(bundles, xret=[0.5, 10.5],
                                           yret=[0.5, 10.5],
                                           return_index=True)
    npt.assert_equal(idx, [0, 1])
    npt.assert_equal(seen, [n_threads])


def test_AxonMapModel_calc_axon_sensitivity():
    model = AxonMapModel(step=2, n_axons=10,
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
    max_d2 = -2.0 * model.lam ** 2 * np.log(model.min_ax_sensitivity)
    for model_ax, xy in zip(axon_contrib, xyret):
        axon = np.insert(model_ax, 0, list(xy) + [0],
                         axis=0).astype(np.float64)
        d2 = np.cumsum(np.sqrt(np.diff(axon[:, 0], axis=0) ** 2 +
                               np.diff(axon[:, 1], axis=0) ** 2))**2
        idx_d2 = d2 < max_d2
        sensitivity = np.exp(-d2[idx_d2] / (2.0 * model.spatial.lam ** 2))
        # A relative bound, unlike `assert_almost_equal`'s absolute one: the
        # sensitivities span [min_ax_sensitivity, 1], and float32 resolves
        # them to ~1.2e-7 relative wherever they sit in that range.
        npt.assert_allclose(model_ax[:, 2], sensitivity, rtol=1e-6)


def test_AxonMapModel_calc_axon_sensitivity_removed_pad():
    # 'pad' used to pad all axons to the length of the longest one for the
    # (now removed) jax backend. Deprecated in 0.9.1, removed in 0.10.0:
    model = AxonMapModel(step=2, n_axons=10, xrange=(-20, 20),
                         yrange=(-15, 15), axons_range=(-30, 30))
    model.build()
    axons = model.spatial.find_closest_axon(model.spatial.grow_axon_bundles())
    with pytest.raises(TypeError):
        model.spatial.calc_axon_sensitivity(axons, pad=True)


def test_AxonMapModel_calc_bundle_tangent():
    model = AxonMapModel(step=5, n_axons=500,
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
    model = AxonMapModel(step=5, n_axons=500,
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
    # `meridian_blend=0` throughout: the expectations below pin the axon-map
    # computation, which the default postprocessing does not change. The blend
    # itself is covered by `test_AxonMapSpatial_meridian_blend`.
    model = AxonMapModel(step=0.55, lam=100, rho=100,
                         thresh_percept=0, meridian_blend=0,
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
    model = AxonMapModel(step=1, rho=100, lam=40, meridian_blend=0,
                         xrange=(-20, 20), yrange=(-15, 15), n_axons=500)
    model.build()
    percept = model.predict_percept(ArgusII(stim=np.ones(60)))
    # Most spots are pretty bright, but there are 2 dimmer ones (due to their
    # location on the retina):
    npt.assert_equal(np.sum(percept.data > 0.5), 28)
    npt.assert_equal(np.sum(percept.data > 0.275), 56)

    # Model gives same outcome as Spatial:
    spatial = AxonMapSpatial(step=1, rho=100, lam=40, meridian_blend=0,
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
    """The default current-spread cutoff barely moves a sparse percept.

    ``min_current_spread`` drops an electrode's contribution once its
    Gaussian has decayed past the given fraction of its peak. This pins the
    everyday case -- a handful of electrodes at unit amplitude, where the
    default cutoff is not worth thinking about. See
    ``test_min_current_spread_error_bound`` for the case where it is.
    """
    stim = np.zeros(60)
    stim[[10, 33, 47]] = [1.0, -0.5, 0.75]
    implant = ArgusII(stim=stim)
    kwargs = {'step': 0.75, 'xrange': (-12, 12), 'yrange': (-8, 8),
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
@pytest.mark.parametrize('amp', (1.0, 1000.0))
def test_min_current_spread_error_bound(ModelClass, amp):
    """The cutoff is an approximation, and stays inside its documented bound.

    The kernels compare the Gaussian against the cutoff *before* scaling it
    by the stimulus and summing over electrodes, so the quantity dropped at a
    point is ``sum_i gauss_i * amp_i``, not ``gauss`` alone. Every electrode
    of the array is driven here so that all 60 individually sub-cutoff terms
    accumulate -- the adversarial case for a per-electrode cutoff -- and the
    amplitude is swept because the bound scales with it.
    """
    min_spread = 1e-8
    stim = np.full(60, amp)
    implant = ArgusII(stim=stim)
    kwargs = {'step': 0.75, 'xrange': (-14, 14), 'yrange': (-10, 10),
              'rho': 200}

    exact = ModelClass(min_current_spread=0,
                       **kwargs).build().predict_percept(implant).data
    default = ModelClass(min_current_spread=min_spread,
                         **kwargs).build().predict_percept(implant).data
    # What the docs promise: `min_current_spread` times the summed amplitude,
    # plus whatever the float32 accumulation itself costs:
    dropped = min_spread * np.abs(stim).sum()
    assert np.abs(default - exact).max() <= dropped + 1e-6 * np.abs(exact).max()

    # It is not, however, a no-op. Points that every electrode is far from
    # come back as exactly zero rather than merely small -- a relative error
    # of 100% at those points, however small they are in absolute terms:
    zeroed = (np.abs(exact) > 0) & (default == 0)
    assert zeroed.any()
    assert np.abs(exact[zeroed]).max() <= dropped


@pytest.mark.parametrize('ModelClass', (ScoreboardModel, AxonMapModel))
def test_predict_percept_frames_are_independent(ModelClass):
    """Each frame of a multi-frame stimulus is predicted on its own.

    The spatial kernels evaluate the electrode-to-point Gaussian once and
    reuse it across every time point, so this guards against one frame
    leaking into another.
    """
    rng = np.random.default_rng(42)
    data = rng.normal(size=(60, 4)).astype(np.float32)
    model = ModelClass(step=1, xrange=(-10, 10), yrange=(-8, 8),
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
    model = ModelClass(step=1, xrange=(-10, 10), yrange=(-8, 8)).build()
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
    kwargs = {'step': 1, 'xrange': (-10, 10), 'yrange': (-8, 8),
              'rho': 200}

    serial = ModelClass(n_threads=1,
                        **kwargs).build().predict_percept(implant).data
    for n_threads in (2, 3, 8):
        parallel = ModelClass(
            n_threads=n_threads, **kwargs).build().predict_percept(implant)
        npt.assert_array_equal(parallel.data, serial)


def test_AxonMapModel_find_closest_axon_return_segment():
    """``return_segment`` reports where in the axon the closest point is."""
    model = AxonMapModel(step=2, n_axons=20, xrange=(-12, 12),
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
    model = AxonMapModel(step=4, n_axons=5, xrange=(-8, 8), yrange=(-8, 8))
    model.build()
    n_points = model.spatial.grid.ret.x.size
    bundles = [np.zeros((0, 2), dtype=np.float32)] * n_points
    with pytest.raises(ValueError):
        model.spatial.calc_axon_sensitivity(bundles)


def test_AxonMapModel_build_cache_roundtrip(tmp_path):
    """A warm build off the cache reproduces the cold build exactly."""
    pickle_file = str(tmp_path / 'axons.pickle')

    def build(ignore_pickle):
        return AxonMapModel(step=1, xrange=(-8, 8), yrange=(-8, 8),
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


def test_AxonMapModel_build_rejects_pre_step_cache(tmp_path):
    """A cache naming the grid step `xystep` is regrown, and stays quiet

    The parameter dict is versioned along with the payload, so a cache written
    before the rename is discarded outright. Reading it instead would probe
    `self.xystep` -- a deprecated name the caller never used -- and warn about
    it on every build, since the stale entry validates and the file is
    therefore never rewritten.
    """
    pickle_file = str(tmp_path / 'axons.pickle')

    def build(ignore_pickle=False):
        return AxonMapModel(step=1, xrange=(-8, 8), yrange=(-8, 8),
                            n_axons=200, axon_pickle=pickle_file,
                            ignore_pickle=ignore_pickle).build().spatial

    cold = build(ignore_pickle=True)
    with open(pickle_file, 'rb') as f:
        params, payload = pickle.load(f)
    # Rewrite it the way v0.9.1 would have:
    params['xystep'] = params.pop('step')
    with open(pickle_file, 'wb') as f:
        pickle.dump((params, (2, *payload[1:])), f)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        warm = build()
    npt.assert_array_equal(warm.axon_contrib, cold.axon_contrib)
    # The stale file was replaced, so the next build is a cache hit:
    with open(pickle_file, 'rb') as f:
        params, payload = pickle.load(f)
    npt.assert_equal('xystep' in params, False)
    npt.assert_equal(payload[0], _AXON_CACHE_VERSION)


def _straddling_pair(coord):
    """Indices nearest zero from below and above."""
    below = np.flatnonzero(coord < 0)
    above = np.flatnonzero(coord > 0)
    return below[np.argmax(coord[below])], above[np.argmin(coord[above])]


@pytest.mark.parametrize('ModelClass', [AxonMapSpatial, AxonMapModel])
def test_AxonMapSpatial_meridian_blend(ModelClass):
    # The axon map is cut along the horizontal raphe, so this blends across
    # y=0 -- and only there, and only along y.
    def make(**params):
        # Offset by half a step so the nearest rows straddle the raphe.
        return ModelClass(xrange=(-6, 6), yrange=(-6.125, 5.875), step=0.25,
                          rho=200, lam=400, n_axons=250, n_ax_segments=200,
                          ignore_pickle=True, **params).build()

    implant = ArgusII(stim={'C4': 1, 'C8': 1})
    plain = make(meridian_blend=0)
    unblended = plain.predict_percept(implant).data

    # The model blends out of the box, so this exercises the default rather
    # than passing a width of its own:
    width = 1
    blended_model = make()
    npt.assert_equal(blended_model.meridian_blend, width)
    blended = blended_model.predict_percept(implant).data
    npt.assert_equal(blended.shape, unblended.shape)
    npt.assert_equal(blended.dtype, unblended.dtype)

    y, x = plain.grid.y[:, 0], plain.grid.x[0, :]
    # The raphe is where the two halves of the axon map meet, and the step
    # across it is what the blend is for. Percept rows are ordered by the
    # grid's y, so the seam is the pair of rows straddling y=0:
    seam = _straddling_pair(y)

    def jump(data):
        return np.abs(data[seam[0], :, 0] - data[seam[1], :, 0]).max()

    npt.assert_array_less(0, jump(unblended))
    npt.assert_array_less(jump(blended), jump(unblended))

    # It is the *horizontal* meridian this model blends across, so the change
    # is a band in y and not one in x. Which rows and columns moved:
    # Count a row or column as having moved if it moved by at least a
    # thousandth of the largest change anywhere, so the bound below is about
    # where the blend acts rather than about float noise:
    delta = np.abs(blended - unblended)
    moved = delta.max() * 1e-3
    rows = delta.max(axis=(1, 2)) > moved
    cols = delta.max(axis=(0, 2)) > moved
    npt.assert_equal(np.any(rows), True)
    # Every row that moved is within a few widths of the raphe, so the far
    # field is untouched...
    npt.assert_array_less(np.abs(y[rows]).max(), 4 * width)
    # ...while columns moved right across the grid, including far from x=0,
    # which a blend across the vertical meridian would have left alone:
    npt.assert_array_less(4 * width, np.abs(x[cols]).max())


def test_AxonMapSpatial_meridian_blend_reapplies_threshold():
    # Blending pulls brightness across the raphe, which could otherwise lift a
    # point that `thresh_percept` had zeroed back off zero.
    implant = ArgusII(stim={'C4': 1})
    model = AxonMapSpatial(xrange=(-6, 6), yrange=(-6, 6), step=0.25, rho=200,
                           lam=400, n_axons=250, n_ax_segments=200,
                           ignore_pickle=True, meridian_blend=1,
                           thresh_percept=0.1).build()
    data = model.predict_percept(implant).data
    npt.assert_equal(np.any(data > 0), True)
    # Nothing survives strictly between zero and the threshold:
    npt.assert_equal(np.any((np.abs(data) > 0) & (np.abs(data) < 0.1)), False)


def test_AxonMapSpatial_meridian_blend_over_time():
    # Every frame is blended, and each one on its own.
    implant = ArgusII(stim=Stimulus({'C4': [0, 1, 2], 'C8': [2, 1, 0]}))
    model = AxonMapSpatial(xrange=(-6, 6), yrange=(-6, 6), step=0.5, rho=200,
                           lam=400, n_axons=250, n_ax_segments=200,
                           ignore_pickle=True, meridian_blend=1).build()
    percept = model.predict_percept(implant)
    npt.assert_equal(percept.data.shape[-1], 3)
    for t in range(3):
        frame = ArgusII(encoder=None, stim=Stimulus(
            {'C4': [0, 1, 2][t], 'C8': [2, 1, 0][t]}))
        npt.assert_allclose(percept.data[..., t],
                            model.predict_percept(frame).data[..., 0],
                            atol=1e-6)
