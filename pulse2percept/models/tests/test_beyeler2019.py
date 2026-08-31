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


from pulse2percept.implants import ArgusI, ArgusII, PRIMAPivotal
from pulse2percept.percepts import Percept
from pulse2percept.stimuli import (ImageStimulus, LogoBVL, Stimulus,
                                   VideoStimulus)
from pulse2percept.models import (AxonMapSpatial, AxonMapModel,
                                  ScoreboardSpatial, ScoreboardModel)
from pulse2percept.models.beyeler2019 import _AXON_CACHE_VERSION
from pulse2percept.models._beyeler2019 import fast_axon_map
from pulse2percept.topography import Watson2014Map, Watson2014DisplaceMap
from pulse2percept.units import (DimensionMismatchError, deg,
                                 dimensionless, dva, mW, mm, rad)
from pulse2percept.utils.testing import assert_warns_msg

# Building an axon map writes a cache to a relative path; keep it in a
# temporary directory instead of wherever pytest was started from:
pytestmark = pytest.mark.usefixtures('axon_cache_in_tmp')


def test_ScoreboardSpatial():
    # ScoreboardSpatial automatically sets `rho`:
    model = ScoreboardSpatial(implant=ArgusII(), step=5)

    # User can set `rho`:
    model.rho = 123
    npt.assert_equal(model.rho, 123)
    model.build(rho=987)
    npt.assert_equal(model.rho, 987)

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(None), None)

    # Converting ret <=> dva
    npt.assert_equal(isinstance(model.vfmap, Watson2014Map), True)
    npt.assert_almost_equal(model.vfmap.ret_to_dva(0, 0), (0, 0))
    npt.assert_almost_equal(model.vfmap.dva_to_ret(0, 0), (0, 0))
    model2 = ScoreboardSpatial(implant=ArgusII(), vfmap=Watson2014DisplaceMap())
    npt.assert_equal(isinstance(model2.vfmap, Watson2014DisplaceMap),
                     True)

    # Zero in = zero out:
    percept = model.predict_percept(np.zeros(60))
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [1])
    npt.assert_almost_equal(percept.data, 0)

    # Multiple frames are processed independently:
    model = ScoreboardSpatial(implant=ArgusI(), rho=200, step=5,
                              xrange=(-20, 20), yrange=(-15, 15))
    model.build()
    percept = model.predict_percept({'A1': [1, 0], 'B3': [0, 2]})
    npt.assert_equal(percept.shape,
                     list(_spatial(model).grid.x.shape) + [2])
    pmax = percept.data.max(axis=(0, 1))
    npt.assert_almost_equal(percept.data[2, 3, 0], pmax[0])
    npt.assert_almost_equal(percept.data[2, 3, 1], 0)
    npt.assert_almost_equal(percept.data[3, 4, 0], 0)
    npt.assert_almost_equal(percept.data[3, 4, 1], pmax[1])
    npt.assert_almost_equal(percept.time, [0, 1])


def test_deepcopy_ScoreboardSpatial():
    original = ScoreboardSpatial(implant=ArgusII())
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
    model = ScoreboardModel(implant=ArgusII(), step=5)
    npt.assert_equal(model.has_space, True)
    npt.assert_equal(model.has_time, False)
    npt.assert_equal(hasattr(model.spatial, 'rho'), True)

    # User can set `rho`:
    model.spatial.rho = 123
    npt.assert_equal(model.spatial.rho, 123)
    model.spatial.build(rho=987)
    npt.assert_equal(model.spatial.rho, 987)

    # Converting ret <=> dva
    npt.assert_equal(isinstance(model.spatial.vfmap, Watson2014Map), True)
    npt.assert_almost_equal(model.spatial.vfmap.ret_to_dva(0, 0), (0, 0))
    npt.assert_almost_equal(model.spatial.vfmap.dva_to_ret(0, 0), (0, 0))
    model2 = ScoreboardModel(implant=ArgusII(), vfmap=Watson2014DisplaceMap())
    npt.assert_equal(isinstance(model2.spatial.vfmap, Watson2014DisplaceMap),
                     True)
    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(None), None)

    # Zero in = zero out:
    npt.assert_almost_equal(model.predict_percept(np.zeros(60)).data, 0)

    # Multiple frames are processed independently:
    model = ScoreboardModel(implant=ArgusI(), rho=200, step=5,
                            xrange=(-20, 20), yrange=(-15, 15))
    model.build()
    percept = model.predict_percept({'A1': [1, 2]})
    npt.assert_equal(percept.shape,
                     list(_spatial(model).grid.x.shape) + [2])
    pmax = percept.data.max(axis=(0, 1))
    npt.assert_almost_equal(percept.data[2, 3, :], pmax)
    npt.assert_almost_equal(pmax[1] / pmax[0], 2.0)
    npt.assert_almost_equal(percept.time, [0, 1])


def test_deepcopy_ScoreboardModel():
    original = ScoreboardModel(implant=ArgusII())
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
    model = ScoreboardModel(implant=ArgusII(), step=0.55, rho=100, thresh_percept=0,
                            xrange=(-20, 20), yrange=(-15, 15))
    model.build()
    # Single-electrode stim:
    img_stim = np.zeros(60)
    img_stim[47] = 1
    percept = model.predict_percept(img_stim)
    # Single bright pixel, very small Gaussian kernel:
    npt.assert_equal(np.sum(percept.data > 0.8), 1)
    npt.assert_equal(np.sum(percept.data > 0.5), 2)
    npt.assert_equal(np.sum(percept.data > 0.1), 7)
    npt.assert_equal(np.sum(percept.data > 0.00001), 32)
    # Brightest pixel is in lower right:
    npt.assert_almost_equal(percept.data[33, 46, 0], np.max(percept.data))

    # Full Argus II: 60 bright spots
    model = ScoreboardModel(implant=ArgusII(), step=0.55, rho=100)
    model.build()
    percept = model.predict_percept(np.ones(60))
    npt.assert_equal(np.sum(np.isclose(percept.data, 0.8, rtol=0.1, atol=0.1)),
                     88)

    # Model gives same outcome as Spatial:
    spatial = ScoreboardSpatial(implant=ArgusII(), step=1, rho=100)
    spatial.build()
    spatial_percept = model.predict_percept(np.ones(60))
    npt.assert_almost_equal(percept.data, spatial_percept.data)
    npt.assert_equal(percept.time, None)

    # Warning for nonzero electrode-retina distances
    raised = ScoreboardModel(implant=ArgusII(z=10), step=0.55, rho=100)
    raised.build()
    # Framed as a limitation of the model, not as a claim that distance is
    # irrelevant, and named so the reader knows which model is silent about it:
    assert_warns_msg(UserWarning, raised.predict_percept,
                     "ScoreboardSpatial does not model electrode-retina distance",
                     np.ones(60))
    assert_warns_msg(UserWarning, raised.predict_percept,
                     "not parameterized by this model", np.ones(60))


def test_AxonMapSpatial():
    # AxonMapSpatial automatically sets `rho`, `lam`:
    model = AxonMapSpatial(implant=ArgusII(), step=5)

    # User can set `rho`:
    model.rho = 123
    npt.assert_equal(model.rho, 123)
    model.build(rho=987)
    npt.assert_equal(model.rho, 987)

    # Converting ret <=> dva
    npt.assert_equal(isinstance(model.vfmap, Watson2014Map), True)
    npt.assert_almost_equal(model.vfmap.ret_to_dva(0, 0), (0, 0))
    npt.assert_almost_equal(model.vfmap.dva_to_ret(0, 0), (0, 0))
    model2 = AxonMapSpatial(implant=ArgusII(), vfmap=Watson2014DisplaceMap())
    npt.assert_equal(isinstance(model2.vfmap, Watson2014DisplaceMap),
                     True)

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(None), None)

    # Zero in = zero out:
    percept = model.predict_percept(np.zeros(60))
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [1])
    npt.assert_almost_equal(percept.data, 0)
    npt.assert_equal(percept.time, None)

    # Lambda cannot be too small:
    with pytest.raises(ValueError):
        AxonMapSpatial(implant=ArgusII(), lam=9).build()

    # Multiple frames are processed independently:
    model = AxonMapSpatial(implant=ArgusI(), rho=200, lam=100, step=5,
                           xrange=(-20, 20), yrange=(-15, 15))
    model.build()
    percept = model.predict_percept({'A1': [1, 0], 'B3': [0, 2]})
    npt.assert_equal(percept.shape,
                     list(_spatial(model).grid.x.shape) + [2])
    pmax = percept.data.max(axis=(0, 1))
    npt.assert_almost_equal(percept.data[2, 3, 0], pmax[0])
    npt.assert_almost_equal(percept.data[2, 3, 1], 0)
    npt.assert_almost_equal(percept.data[3, 4, 0], 0)
    npt.assert_almost_equal(percept.data[3, 4, 1], pmax[1])
    npt.assert_almost_equal(percept.time, [0, 1])


def test_deepcopy_AxonMapSpatial():
    original = AxonMapSpatial(implant=ArgusII())
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
    model = AxonMapSpatial(implant=ArgusII())
    for use_dva, xlim in zip([True, False], [(-18, 18), (-5000, 5000)]):
        ax = model.plot(use_dva=use_dva)
        npt.assert_equal(isinstance(ax, Subplot), True)
        npt.assert_equal(ax.get_xlim(), xlim)
    # Simulated area might be larger than that:
    model = AxonMapSpatial(implant=ArgusII(), xrange=(-20.5, 20.5), yrange=(-16.1, 16.1))
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
    model = AxonMapModel(implant=ArgusII())
    for param in set_params:
        npt.assert_equal(hasattr(model.spatial, param), True)

    # User can override default values
    for key, value in set_params.items():
        setattr(model.spatial, key, value)
        npt.assert_equal(getattr(model.spatial, key), value)
    model = AxonMapModel(implant=ArgusII(), **set_params)
    model.spatial.build(**set_params)
    for key, value in set_params.items():
        npt.assert_equal(getattr(model.spatial, key), value)

    # Converting ret <=> dva
    npt.assert_equal(isinstance(model.spatial.vfmap, Watson2014Map), True)
    npt.assert_almost_equal(model.spatial.vfmap.ret_to_dva(0, 0), (0, 0))
    npt.assert_almost_equal(model.spatial.vfmap.dva_to_ret(0, 0), (0, 0))
    model2 = AxonMapModel(implant=ArgusII(), vfmap=Watson2014DisplaceMap())
    npt.assert_equal(isinstance(model2.spatial.vfmap, Watson2014DisplaceMap),
                     True)

    # Zeros in, zeros out:
    npt.assert_almost_equal(model.predict_percept(np.zeros(60)).data, 0)

    # The eye is the implanted one, and is not settable on its own:
    npt.assert_equal(
        AxonMapModel(implant=ArgusII(eye='LE'), step=5).spatial.eye, 'LE')
    with pytest.raises(TypeError):
        AxonMapModel(implant=ArgusII(), eye='LE')

    # Lambda cannot be too small:
    with pytest.raises(ValueError):
        AxonMapModel(implant=ArgusII(), lam=9).build()


@pytest.mark.parametrize('cls', [AxonMapSpatial, AxonMapModel])
def test_AxonMap_removed_axlambda(cls):
    # `lam` was called `axlambda` until 0.10.0; the old name was removed
    # in 0.11.0, so it is now an unknown parameter:
    with pytest.raises(TypeError):
        cls(ArgusII(), axlambda=400)
    with pytest.raises(AttributeError):
        model = cls(ArgusII(), step=5)
        if cls is AxonMapModel:
            model.set_params({'axlambda': 400})
        else:
            model.set_params(axlambda=400)


@pytest.mark.parametrize('build', (False, True))
def test_deepcopy_AxonMapModel(build):
    original = AxonMapModel(implant=ArgusII())
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
    model = AxonMapModel(implant=ArgusII(), loc_od=loc_od, step=2,
                         ax_segments_range=(0, 45),
                         n_ax_segments=100)
    for phi0 in [-135.0, 66.0, 128.0]:
        ax_pos = model.spatial._jansonius2009(phi0)
        npt.assert_almost_equal(ax_pos[0, 0], loc_od[0])
        npt.assert_almost_equal(ax_pos[0, 1], loc_od[1])

    # These axons should all end at the meridian
    for phi0 in [110.0, 135.0, 160.0]:
        model = AxonMapModel(implant=ArgusII(), loc_od=(15, 2), step=2,
                             n_ax_segments=801,
                             ax_segments_range=(0, 45))
        ax_pos = model.spatial._jansonius2009(sign * phi0)
        npt.assert_almost_equal(ax_pos[-1, 1], 0.0, decimal=1)

    # `phi0` must be within [-180, 180]
    for phi0 in [-200.0, 181.0]:
        with pytest.raises(ValueError):
            failed = AxonMapModel(implant=ArgusII(), step=2)
            failed.spatial._jansonius2009(phi0)

    # `n_rho` must be >= 1
    for n_rho in [-1, 0]:
        with pytest.raises(ValueError):
            model = AxonMapModel(implant=ArgusII(), n_ax_segments=n_rho, step=2)
            model.spatial._jansonius2009(0.0)

    # `ax_segments_range` must have min <= max
    for lorho in [-200.0, 90.0]:
        with pytest.raises(ValueError):
            model = AxonMapModel(implant=ArgusII(), ax_segments_range=(lorho, 45), step=2)
            model.spatial._jansonius2009(0)
    for hirho in [-200.0, 40.0]:
        with pytest.raises(ValueError):
            model = AxonMapModel(implant=ArgusII(), ax_segments_range=(45, hirho), step=2)
            model.spatial._jansonius2009(0)

    # A single axon fiber with `phi0`=0 should return a single pixel location
    # that corresponds to the optic disc
        model = AxonMapModel(implant=ArgusII(eye=eye), loc_od=loc_od, step=2,
                             ax_segments_range=(0, 0),
                             n_ax_segments=1)
        single_fiber = model.spatial._jansonius2009(0)
        npt.assert_equal(len(single_fiber), 1)
        npt.assert_almost_equal(single_fiber[0], loc_od)


def test_AxonMapModel_grow_axon_bundles():
    for n_axons in [1, 2, 3, 5, 10]:
        model = AxonMapModel(implant=ArgusII(), step=2, n_axons=n_axons,
                             axons_range=(-20, 20), xrange=(-20, 20),
                             yrange=(-15, 15))
        bundles = model.spatial.grow_axon_bundles()
        npt.assert_equal(len(bundles), n_axons)


def test_AxonMapModel_find_closest_axon():
    model = AxonMapModel(implant=ArgusII(), step=1, n_axons=5,
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
    here would let ``AxonMapModel(implant=ArgusII(), n_threads=1).build()`` fan out over every
    core anyway.
    """
    from pulse2percept.models import beyeler2019

    seen = []

    class RecordingKDTree(beyeler2019.cKDTree):
        def query(self, *args, **kwargs):
            seen.append(kwargs.get('workers'))
            return super().query(*args, **kwargs)

    monkeypatch.setattr(beyeler2019, 'cKDTree', RecordingKDTree)
    model = AxonMapSpatial(implant=ArgusII(), n_threads=n_threads)
    bundles = [np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
               np.array([[10.0, 10.0], [11.0, 11.0]], dtype=np.float32)]
    closest, idx = model.find_closest_axon(bundles, xret=[0.5, 10.5],
                                           yret=[0.5, 10.5],
                                           return_index=True)
    npt.assert_equal(idx, [0, 1])
    npt.assert_equal(seen, [n_threads])


def test_AxonMapModel_calc_axon_sensitivity():
    model = AxonMapModel(implant=ArgusII(), step=2, n_axons=10,
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
    max_d2 = -2.0 * model.spatial.lam ** 2 * np.log(
        model.spatial.min_ax_sensitivity)
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
    model = AxonMapModel(implant=ArgusII(), step=2, n_axons=10, xrange=(-20, 20),
                         yrange=(-15, 15), axons_range=(-30, 30))
    model.build()
    axons = model.spatial.find_closest_axon(model.spatial.grow_axon_bundles())
    with pytest.raises(TypeError):
        model.spatial.calc_axon_sensitivity(axons, pad=True)


def test_AxonMapModel_calc_bundle_tangent():
    model = AxonMapModel(implant=ArgusII(), step=5, n_axons=500,
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
    model = AxonMapModel(implant=ArgusII(), step=5, n_axons=500,
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
    model = AxonMapModel(implant=ArgusII(), step=0.55, lam=100, rho=100,
                         thresh_percept=0, meridian_blend=0,
                         xrange=(-20, 20), yrange=(-15, 15),
                         n_axons=500)
    model.build()
    # Single-electrode stim:
    img_stim = np.zeros(60)
    img_stim[47] = 1
    percept = model.predict_percept(img_stim)
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
    model = AxonMapModel(implant=ArgusII(), step=1, rho=100, lam=40, meridian_blend=0,
                         xrange=(-20, 20), yrange=(-15, 15), n_axons=500)
    model.build()
    percept = model.predict_percept(np.ones(60))
    # Most spots are pretty bright, but there are 2 dimmer ones (due to their
    # location on the retina):
    npt.assert_equal(np.sum(percept.data > 0.5), 28)
    npt.assert_equal(np.sum(percept.data > 0.275), 56)

    # Model gives same outcome as Spatial:
    spatial = AxonMapSpatial(implant=ArgusII(), step=1, rho=100, lam=40, meridian_blend=0,
                             xrange=(-20, 20), yrange=(-15, 15), n_axons=500)
    spatial.build()
    spatial_percept = spatial.predict_percept(np.ones(60))
    npt.assert_almost_equal(percept.data, spatial_percept.data)
    npt.assert_equal(percept.time, None)

    # Warning for nonzero electrode-retina distances
    raised = AxonMapModel(implant=ArgusII(z=10), step=1, rho=100, lam=40,
                          meridian_blend=0, n_axons=250, n_ax_segments=200,
                          ignore_pickle=True).build()
    # Framed as a limitation of the model, not as a claim that distance is
    # irrelevant, and named so the reader knows which model is silent about it:
    assert_warns_msg(UserWarning, raised.predict_percept,
                     "AxonMapSpatial does not model electrode-retina distance",
                     np.ones(60))
    assert_warns_msg(UserWarning, raised.predict_percept,
                     "not parameterized by this model", np.ones(60))


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
    kwargs = {'implant': ArgusII(), 'step': 0.75, 'xrange': (-12, 12),
              'yrange': (-8, 8), 'rho': 200}

    exact = ModelClass(min_current_spread=0,
                       **kwargs).build().predict_percept(stim).data
    default = ModelClass(**kwargs).build().predict_percept(stim).data
    npt.assert_allclose(default, exact, rtol=1e-5,
                        atol=1e-6 * np.abs(exact).max())

    # A coarse cutoff *does* change the result, which is how we know the
    # parameter reaches the kernel at all:
    coarse = ModelClass(min_current_spread=0.5,
                        **kwargs).build().predict_percept(stim).data
    assert np.abs(coarse - exact).max() > 1e-3

    # A cutoff of 1 or more would drop every electrode:
    model = ModelClass(min_current_spread=1, **kwargs).build()
    with pytest.raises(ValueError):
        model.predict_percept(stim)


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
    kwargs = {'implant': ArgusII(), 'step': 0.75, 'xrange': (-14, 14),
              'yrange': (-10, 10), 'rho': 200}

    exact = ModelClass(min_current_spread=0,
                       **kwargs).build().predict_percept(stim).data
    default = ModelClass(min_current_spread=min_spread,
                         **kwargs).build().predict_percept(stim).data
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
    model = ModelClass(implant=ArgusII(), step=1, xrange=(-10, 10),
                       yrange=(-8, 8), rho=200).build()

    joint = model.predict_percept(
        Stimulus(data, time=[0, 1, 2, 3])).data
    npt.assert_equal(joint.shape[-1], 4)
    for i in range(data.shape[1]):
        frame = model.predict_percept(
            Stimulus(data[:, i:i + 1])).data
        npt.assert_allclose(joint[..., i], frame[..., 0], rtol=1e-5,
                            atol=1e-6 * np.abs(frame).max())


@pytest.mark.parametrize('ModelClass', (ScoreboardModel, AxonMapModel))
def test_predict_percept_all_zero_stim(ModelClass):
    """An all-zero stimulus produces an all-zero percept.

    The kernels skip electrodes that are zero for the whole stimulus, so the
    case where *every* electrode is skipped is worth pinning down.
    """
    model = ModelClass(implant=ArgusII(), step=1, xrange=(-10, 10),
                       yrange=(-8, 8)).build()
    percept = model.predict_percept(np.zeros(60))
    npt.assert_equal(np.all(percept.data == 0), True)


def test_fast_axon_map_cutoff_band_boundaries():
    """An electrode exactly on either edge of the cutoff still contributes.

    ``fast_axon_map`` binary-searches the x band ``[ax_x - r, ax_x + r]`` and
    walks it until x leaves, so both ends are places an off-by-one can hide.
    ``cutoff_r2`` is an exact float32 square here, so that what is being
    pinned is the band's boundary rather than the rounding of its ``sqrt``.
    """
    rho = np.float32(200.0)
    cutoff_r2 = np.float32(360000.0)  # r = 600 um, exactly
    # One pixel whose axon is a single segment at the origin, sensitivity 1:
    segments = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    start = np.array([0], dtype=np.uint32)
    end = np.array([1], dtype=np.uint32)

    def bright(x_el):
        x_el = np.ascontiguousarray(x_el, dtype=np.float32)
        stim = np.full((len(x_el), 1), -1.0, dtype=np.float32)
        return fast_axon_map(stim, x_el, np.zeros_like(x_el), segments,
                             start, end, rho, np.float32(0.0), cutoff_r2,
                             1).ravel()[0]

    for x in (-600.0, 600.0):
        npt.assert_array_less(0.0, abs(bright([x])))
    for x in (-600.5, 600.5):
        npt.assert_equal(bright([x]), 0.0)

    # ... and the walk in between drops exactly the electrodes outside it:
    x_el = np.array([-900., -600.5, -600., -300., 0., 300., 600., 600.5,
                     900.], dtype=np.float32)
    # Summed in increasing x, which is the order the kernel visits them in:
    want = np.float32(0.0)
    two_rho2 = 2.0 * rho * rho
    for x in x_el[np.abs(x_el) <= 600.0]:
        want = np.float32(want - np.float32(np.exp(-x * x / two_rho2)))
    npt.assert_allclose(bright(x_el), want, rtol=1e-6)


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
    kwargs = {'implant': ArgusII(), 'step': 1, 'xrange': (-10, 10),
              'yrange': (-8, 8), 'rho': 200}

    serial = ModelClass(n_threads=1,
                        **kwargs).build().predict_percept(stim).data
    for n_threads in (2, 3, 8):
        parallel = ModelClass(
            n_threads=n_threads, **kwargs).build().predict_percept(stim)
        npt.assert_array_equal(parallel.data, serial)


def test_AxonMapModel_find_closest_axon_return_segment():
    """``return_segment`` reports where in the axon the closest point is."""
    model = AxonMapModel(implant=ArgusII(), step=2, n_axons=20, xrange=(-12, 12),
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
    model = AxonMapModel(implant=ArgusII(), step=4, n_axons=5, xrange=(-8, 8), yrange=(-8, 8))
    model.build()
    n_points = model.spatial.grid.ret.x.size
    bundles = [np.zeros((0, 2), dtype=np.float32)] * n_points
    with pytest.raises(ValueError):
        model.spatial.calc_axon_sensitivity(bundles)


def test_AxonMapModel_build_cache_roundtrip(tmp_path):
    """A warm build off the cache reproduces the cold build exactly."""
    pickle_file = str(tmp_path / 'axons.pickle')

    def build(ignore_pickle):
        return AxonMapModel(implant=ArgusII(), step=1, xrange=(-8, 8), yrange=(-8, 8),
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
    before the 0.10.0 rename is discarded outright rather than validated
    against a model that no longer has a `xystep` parameter.
    """
    pickle_file = str(tmp_path / 'axons.pickle')

    def build(ignore_pickle=False):
        return AxonMapModel(implant=ArgusII(), step=1, xrange=(-8, 8), yrange=(-8, 8),
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


def _spatial(model):
    """The spatial model itself, or the one a composite wraps."""
    return getattr(model, 'spatial', model)


def _straddling_pair(coord):
    """Indices nearest zero from below and above."""
    below = np.flatnonzero(coord < 0)
    above = np.flatnonzero(coord > 0)
    return below[np.argmax(coord[below])], above[np.argmin(coord[above])]


@pytest.mark.parametrize('ModelClass', [AxonMapSpatial, AxonMapModel])
def test_AxonMapSpatial_meridian_blend(ModelClass):
    def make(**params):
        # Offset by half a step so the nearest rows straddle the raphe.
        return ModelClass(implant=ArgusII(), xrange=(-6, 6),
                          yrange=(-6.125, 5.875), step=0.25, rho=200, lam=400,
                          n_axons=250, n_ax_segments=200, ignore_pickle=True,
                          **params).build()

    source = {'C4': 1, 'C8': 1}
    plain = make(meridian_blend=0)
    unblended = plain.predict_percept(source).data

    width = 1
    blended_model = make()
    npt.assert_equal(_spatial(blended_model).meridian_blend, width)
    blended = blended_model.predict_percept(source).data
    npt.assert_equal(blended.shape, unblended.shape)
    npt.assert_equal(blended.dtype, unblended.dtype)

    y, x = _spatial(plain).grid.y[:, 0], _spatial(plain).grid.x[0, :]
    # The raphe is where the two halves of the axon map meet:
    seam = _straddling_pair(y)

    def jump(data):
        return np.abs(data[seam[0], :, 0] - data[seam[1], :, 0]).max()

    npt.assert_array_less(0, jump(unblended))
    npt.assert_array_less(jump(blended), jump(unblended))

    # Blend across horizontal meridian:
    delta = np.abs(blended - unblended)
    moved = delta.max() * 1e-3
    rows = delta.max(axis=(1, 2)) > moved
    cols = delta.max(axis=(0, 2)) > moved
    npt.assert_equal(np.any(rows), True)
    # Every row that moved is within a few widths of the raphe, so the far
    # field is untouched...
    npt.assert_array_less(np.abs(y[rows]).max(), 4 * width)
    # ...while columns moved right across the grid:
    npt.assert_array_less(4 * width, np.abs(x[cols]).max())


def test_AxonMapSpatial_meridian_blend_reapplies_threshold():
    # Blending pulls brightness across the raphe, which could otherwise lift a
    # point that `thresh_percept` had zeroed back off zero.
    model = AxonMapSpatial(implant=ArgusII(), xrange=(-6, 6), yrange=(-6, 6),
                           step=0.25, rho=200, lam=400, n_axons=250,
                           n_ax_segments=200, ignore_pickle=True,
                           meridian_blend=1, thresh_percept=0.1).build()
    data = model.predict_percept({'C4': 1}).data
    npt.assert_equal(np.any(data > 0), True)
    # Nothing survives strictly between zero and the threshold:
    npt.assert_equal(np.any((np.abs(data) > 0) & (np.abs(data) < 0.1)), False)


def test_AxonMapSpatial_meridian_blend_over_time():
    # Every frame is blended, and each one on its own.
    model = AxonMapSpatial(implant=ArgusII(), xrange=(-6, 6), yrange=(-6, 6),
                           step=0.5, rho=200, lam=400, n_axons=250,
                           n_ax_segments=200, ignore_pickle=True,
                           meridian_blend=1).build()
    percept = model.predict_percept(
        Stimulus({'C4': [0, 1, 2], 'C8': [2, 1, 0]}))
    npt.assert_equal(percept.data.shape[-1], 3)
    for t in range(3):
        frame = Stimulus({'C4': [0, 1, 2][t], 'C8': [2, 1, 0][t]})
        npt.assert_allclose(percept.data[..., t],
                            model.predict_percept(frame).data[..., 0],
                            atol=1e-6)


def test_AxonMapSpatial_axons_range_units():
    """`axons_range` is a range of ordinary polar angles, stored in degrees"""
    npt.assert_equal(AxonMapSpatial(implant=ArgusII()).get_param_units()['axons_range'], deg)
    bare = AxonMapSpatial(implant=ArgusII(), axons_range=(-30, 30))
    npt.assert_equal(AxonMapSpatial(implant=ArgusII(), axons_range=(-30 * deg, 30 * deg)).
                     axons_range, bare.axons_range)
    npt.assert_allclose(
        AxonMapSpatial(implant=ArgusII(), axons_range=np.array([-np.pi, np.pi]) / 6 * rad).
        axons_range, bare.axons_range, rtol=1e-12)
    with pytest.raises(DimensionMismatchError):
        AxonMapSpatial(implant=ArgusII(), axons_range=(-30 * dva, 30 * dva))


def _user_warnings(build):
    """The UserWarning messages a build emits, and nothing else

    Building an axon map also emits ResourceWarnings from the pickle cache,
    which have nothing to do with what these tests are about.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        build()
    return [str(w.message) for w in caught
            if issubclass(w.category, UserWarning)]


def test_axon_map_eye_follows_the_implant():
    """The eye is the implanted one, and cannot drift out of step with it"""
    implant = ArgusII(eye='RE')
    model = AxonMapModel(implant=implant, step=2, n_axons=50,
                         n_ax_segments=30).build()
    npt.assert_equal(model.spatial.eye, 'RE')
    # The optic disc is on the nasal side, which is a different side per eye:
    npt.assert_equal(model.spatial.loc_od[0] > 0, True)

    # Turning the *bound implant* around is the one build-invalidating change
    # the parameter machinery cannot see, so the model checks it itself:
    implant.eye = 'LE'
    npt.assert_equal(model.spatial.eye, 'LE')
    npt.assert_equal(model.is_built, False)
    model.predict_percept({'A1': 20})
    npt.assert_equal(model.is_built, True)
    npt.assert_equal(model.spatial.loc_od[0] < 0, True)


def test_axon_map_warns_when_the_implant_is_not_epiretinal():
    from pulse2percept.implants import GridImplant, Lorach2015Array
    grid = dict(step=1, xrange=(-2, 2), yrange=(-2, 2), n_axons=50,
                n_ax_segments=30)
    said = _user_warnings(AxonMapModel(implant=Lorach2015Array(), **grid).build)
    npt.assert_equal(any('subretinal' in w for w in said), True)
    npt.assert_equal(any('scoreboard model' in w for w in said), True)
    # An implant whose placement nobody wrote down says nothing either way.
    # Its pitch is wide enough not to trip the other warning:
    quiet = GridImplant(shape=(3, 3), spacing=2000)
    npt.assert_equal(_user_warnings(AxonMapModel(implant=quiet, **grid).build),
                     [])


@pytest.mark.parametrize('ModelClass', [ScoreboardModel, AxonMapModel])
def test_rho_wider_than_the_electrode_pitch_warns(ModelClass):
    from pulse2percept.implants import GridImplant
    extra = {'n_axons': 50, 'n_ax_segments': 30} if ModelClass is AxonMapModel         else {}
    grid = dict(step=1, xrange=(-2, 2), yrange=(-2, 2), **extra)
    dense = ModelClass(implant=GridImplant(shape=(3, 3), spacing=100),
                       rho=400, **grid)
    said = _user_warnings(dense.build)
    # The numbers a reader needs to judge it, not a verdict:
    npt.assert_equal(any('pitch (100 um)' in w for w in said), True)
    npt.assert_equal(any('ratio of 4.00' in w for w in said), True)
    # rho at the pitch is the boundary, and is not warned about:
    matched = ModelClass(implant=GridImplant(shape=(3, 3), spacing=400),
                         rho=400, **grid)
    npt.assert_equal(_user_warnings(matched.build), [])


@pytest.mark.parametrize('ModelClass', [ScoreboardModel, AxonMapModel])
def test_electrode_pitch_ignores_a_dimension_the_model_drops(ModelClass):
    """A retinal model reads x and y, so z cannot pull neighbours apart"""
    from pulse2percept.implants import (DiskElectrode, ElectrodeArray,
                                        Implant)
    extra = {'n_axons': 50, 'n_ax_segments': 30} if ModelClass is AxonMapModel         else {}
    # Three electrodes 100 um apart in x, but 1000 um apart in z. Reading all
    # three coordinates would call that a ~1005 um pitch and stay quiet:
    stacked = Implant(ElectrodeArray(
        [DiskElectrode(100 * i, 0, 1000 * i, 50) for i in range(3)]))
    model = ModelClass(implant=stacked, rho=400, step=1, xrange=(-2, 2),
                       yrange=(-2, 2), **extra)
    said = _user_warnings(model.build)
    npt.assert_equal(any('pitch (100 um)' in w for w in said), True)


def test_scoreboard_visualizes_a_photovoltaic_implant():
    """Scoreboard accepts normalized optical drive from PRIMA."""
    implant = PRIMAPivotal()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        model = ScoreboardModel(implant=implant, rho=200, step=0.05,
                                xrange=(-2, 2), yrange=(-2, 2))
        percept = model.predict_percept(LogoBVL())
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, tuple(model.spatial.grid.x.shape) + (1,))
    npt.assert_equal(np.all(np.isfinite(percept.data)), True)
    npt.assert_equal(percept.data.max() > 0, True)
    # Delivered stimulation is optical; the spatial view is normalized.
    delivered = implant.prepare_stim(LogoBVL())
    npt.assert_equal(delivered.unit, mW / mm ** 2)
    npt.assert_equal(delivered._spatial_view().unit, dimensionless)
    # Dark input produces zero drive.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        dark = model.predict_percept(ImageStimulus(np.zeros((32, 32))))
    npt.assert_almost_equal(dark.data.max(), 0)


def test_scoreboard_visualizes_a_photovoltaic_video():
    """Normalized-drive semantics survive video resampling."""
    implant = PRIMAPivotal()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        model = ScoreboardModel(implant=implant, rho=200, step=0.5,
                                xrange=(-2, 2), yrange=(-2, 2))
        # Only the middle source frame is lit.
        frames = np.zeros((8, 8, 3))
        frames[..., 1] = 1.0
        video = VideoStimulus(frames, time=[0.0, 40.0, 80.0])
        percept = model.predict_percept(video)
    npt.assert_equal(percept.shape[:2], tuple(model.spatial.grid.x.shape))
    npt.assert_equal(np.all(np.isfinite(percept.data)), True)
    npt.assert_equal(percept.data.max() > 0, True)
    # One percept frame per projector frame.
    npt.assert_equal(percept.shape[-1], percept.time.size)
    npt.assert_almost_equal(np.diff(percept.time), 1000 / 30, decimal=3)
    lit = percept.data.max(axis=(0, 1)) > 0
    npt.assert_equal(lit.any() and not lit.all(), True)


def test_scoreboard_refuses_a_bare_optical_waveform():
    # Bare irradiance is not a valid Scoreboard input.
    implant = PRIMAPivotal()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        model = ScoreboardModel(implant=implant, rho=200, step=0.5,
                                xrange=(-2, 2), yrange=(-2, 2))
    bare = Stimulus(implant.prepare_stim(LogoBVL()))
    npt.assert_equal(bare._has_spatial_view, False)
    with pytest.raises(DimensionMismatchError) as excinfo:
        model.predict_percept(bare)
    npt.assert_equal('irradiance' in str(excinfo.value), True)
