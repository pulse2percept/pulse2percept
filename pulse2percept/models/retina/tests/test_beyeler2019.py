from types import SimpleNamespace
import numpy as np
import pytest
import numpy.testing as npt
import copy
import os
import pickle
import warnings

import torch
from matplotlib.axes import Subplot
import matplotlib.pyplot as plt


from pulse2percept.implants.retina import ArgusI, ArgusII, PRIMAPivotal
from pulse2percept.percepts import Percept
from pulse2percept.stimuli import (ImageStimulus, Stimulus, samples,
                                   VideoStimulus)
from pulse2percept.models.retina import (AxonMapSpatial, AxonMapModel,
                                         ScoreboardSpatial, ScoreboardModel)
from pulse2percept.models.base import _GAUSSIAN_CUTOFF, SpatialModel
from pulse2percept.models.retina.beyeler2019 import _AXON_CACHE_VERSION
from pulse2percept.topography.retina import (Montesano2020Map,
                                             Watson2014Map)
from pulse2percept.units import (DimensionMismatchError, deg,
                                 dimensionless, dva, mW, mm, rad, um)
from pulse2percept.utils.testing import assert_warns_msg

# Axon map caches use a relative path; write them to a temp directory:
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
    npt.assert_equal(isinstance(model.visual_field_map, Watson2014Map), True)
    npt.assert_almost_equal(model.visual_field_map.ret_to_dva(0, 0), (0, 0))
    npt.assert_almost_equal(model.visual_field_map.dva_to_ret(0, 0), (0, 0))
    model2 = ScoreboardSpatial(implant=ArgusII(),
                               visual_field_map=Montesano2020Map())
    npt.assert_equal(isinstance(model2.visual_field_map,
                                Montesano2020Map),
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
    # States now differ (NumPy check because the dicts contain arrays):
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
    npt.assert_equal(isinstance(model.spatial.visual_field_map, Watson2014Map),
                     True)
    npt.assert_almost_equal(model.spatial.visual_field_map.ret_to_dva(0, 0),
                            (0, 0))
    npt.assert_almost_equal(model.spatial.visual_field_map.dva_to_ret(0, 0),
                            (0, 0))
    model2 = ScoreboardModel(implant=ArgusII(),
                             visual_field_map=Montesano2020Map())
    npt.assert_equal(isinstance(model2.spatial.visual_field_map,
                                Montesano2020Map),
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
    # States now differ (NumPy check because the dicts contain arrays):
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
    # Warning names the model:
    assert_warns_msg(UserWarning, raised.predict_percept,
                     "ScoreboardSpatial does not model electrode-retina distance",
                     np.ones(60))
    assert_warns_msg(UserWarning, raised.predict_percept,
                     "not parameterized by this model", np.ones(60))

    # `implant_depth` also counts as electrode-retina distance:
    placed = ScoreboardModel(implant=ArgusII(), implant_depth=500 * um,
                             step=0.55, rho=100)
    placed.build()
    assert_warns_msg(UserWarning, placed.predict_percept,
                     "ScoreboardSpatial does not model "
                     "electrode-retina distance",
                     np.ones(60))
    # No warning for a flat implant at the retinal surface:
    flat = ScoreboardModel(implant=ArgusII(), step=0.55, rho=100)
    flat.build()
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        flat.predict_percept(np.ones(60))


def test_AxonMapSpatial():
    # AxonMapSpatial automatically sets `rho`, `lam`:
    model = AxonMapSpatial(implant=ArgusII(), step=5)

    # User can set `rho`:
    model.rho = 123
    npt.assert_equal(model.rho, 123)
    model.build(rho=987)
    npt.assert_equal(model.rho, 987)

    # Converting ret <=> dva
    npt.assert_equal(isinstance(model.visual_field_map, Watson2014Map), True)
    npt.assert_almost_equal(model.visual_field_map.ret_to_dva(0, 0), (0, 0))
    npt.assert_almost_equal(model.visual_field_map.dva_to_ret(0, 0), (0, 0))
    model2 = AxonMapSpatial(implant=ArgusII(),
                            visual_field_map=Montesano2020Map())
    npt.assert_equal(isinstance(model2.visual_field_map,
                                Montesano2020Map),
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
    # States now differ (NumPy check because the dicts contain arrays):
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
    npt.assert_equal(isinstance(model.spatial.visual_field_map, Watson2014Map),
                     True)
    npt.assert_almost_equal(model.spatial.visual_field_map.ret_to_dva(0, 0),
                            (0, 0))
    npt.assert_almost_equal(model.spatial.visual_field_map.dva_to_ret(0, 0),
                            (0, 0))
    model2 = AxonMapModel(implant=ArgusII(),
                          visual_field_map=Montesano2020Map())
    npt.assert_equal(isinstance(model2.spatial.visual_field_map,
                                Montesano2020Map),
                     True)

    # Zeros in, zeros out:
    npt.assert_almost_equal(model.predict_percept(np.zeros(60)).data, 0)

    # `eye` comes from the implant and cannot be passed to the model:
    npt.assert_equal(
        AxonMapModel(implant=ArgusII(eye='left'), step=5).spatial.eye, 'left')
    with pytest.raises(TypeError):
        AxonMapModel(implant=ArgusII(), eye='left')

    # Lambda cannot be too small:
    with pytest.raises(ValueError):
        AxonMapModel(implant=ArgusII(), lam=9).build()


@pytest.mark.parametrize('cls', [AxonMapSpatial, AxonMapModel])
def test_AxonMap_removed_axlambda(cls):
    # `axlambda` (renamed to `lam` in 0.10.0) was removed in 0.11.0:
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


@ pytest.mark.parametrize('eye', ('left', 'right'))
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


#: Right-eye bundles (dva) from the float32 Cython kernel of v0.11, default
#: geometry: phi0 -> (length, {segment: (x, y)}). Left eyes mirror x.
JANSONIUS_FROZEN = {
    -135.0: (247, {1: (15.429131, 1.415479), 10: (14.786551, 0.661526),
                   100: (7.107345, -5.158602), 246: (-9.149109, -0.096284)}),
    -90.0: (500, {1: (15.499711, 1.399744), 10: (15.471113, 0.492827),
                  100: (12.650759, -8.607186), 499: (-34.064930, -6.581625)}),
    0.0: (500, {1: (15.600118, 1.515367), 10: (16.493704, 1.569789),
                100: (24.697498, -0.167505), 499: (46.087349, -26.291218)}),
    66.0: (500, {1: (15.537006, 1.600287), 10: (15.786694, 2.516116),
                 100: (15.608869, 11.540594), 499: (-8.213854, 44.018780)}),
    128.0: (202, {1: (15.438311, 1.567043), 10: (14.882763, 2.172236),
                  100: (8.203381, 7.287514), 201: (-4.640015, 0.103467)}),
}


@pytest.mark.parametrize('eye', ('right', 'left'))
def test_AxonMapSpatial_jansonius_frozen(eye):
    spatial = AxonMapSpatial(ArgusII(eye=eye))
    spatial._correct_loc_od()
    sign = 1 if eye == 'right' else -1
    bundles = spatial._jansonius_bundles(list(JANSONIUS_FROZEN), eye=eye)
    for bundle, (n_seg, segments) in zip(bundles, JANSONIUS_FROZEN.values()):
        npt.assert_equal(len(bundle), n_seg)
        npt.assert_equal(bundle[0], (sign * 15.5, 1.5))
        for idx, (x, y) in segments.items():
            # A few float32 ulps of the largest coordinate:
            npt.assert_allclose(bundle[idx], (sign * x, y), rtol=0, atol=2e-5)
    # The scalar entry point matches the vectorized one:
    npt.assert_equal(spatial._jansonius2009(66.0, eye=eye), bundles[3])


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

    # Check lambda math. `calc_axon_sensitivity` accumulates arc length in
    # float64 and rounds once, so build the reference in float64 too
    # (float32 accumulation uses up most of the tolerance):
    max_d2 = -2.0 * model.spatial.lam ** 2 * np.log(
        model.spatial.min_ax_sensitivity)
    for model_ax, xy in zip(axon_contrib, xyret):
        axon = np.insert(model_ax, 0, list(xy) + [0],
                         axis=0).astype(np.float64)
        d2 = np.cumsum(np.sqrt(np.diff(axon[:, 0], axis=0) ** 2 +
                               np.diff(axon[:, 1], axis=0) ** 2))**2
        idx_d2 = d2 < max_d2
        sensitivity = np.exp(-d2[idx_d2] / (2.0 * model.spatial.lam ** 2))
        # Relative tolerance: sensitivities span [min_ax_sensitivity, 1] and
        # float32 resolves them to ~1.2e-7 relative:
        npt.assert_allclose(model_ax[:, 2], sensitivity, rtol=1e-6)


def test_AxonMapModel_calc_axon_sensitivity_removed_pad():
    # 'pad' (for the removed jax backend) was deprecated in 0.9.1, removed in
    # 0.10.0:
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
    # `meridian_blend=0` throughout; see `test_AxonMapSpatial_meridian_blend`:
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
    # Most spots are bright; 2 are dimmer due to their retinal location:
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
    # Warning names the model:
    assert_warns_msg(UserWarning, raised.predict_percept,
                     "AxonMapSpatial does not model electrode-retina distance",
                     np.ones(60))
    assert_warns_msg(UserWarning, raised.predict_percept,
                     "not parameterized by this model", np.ones(60))


@pytest.mark.parametrize('ModelClass', (ScoreboardModel, AxonMapModel))
@pytest.mark.parametrize('amp', (1.0, 1000.0))
def test_cutoff_error_bound(ModelClass, amp, monkeypatch):
    """The fixed cutoff error stays within its documented bound

    The cutoff is applied to the Gaussian *before* scaling by amplitude and
    summing, so the error at a point is ``sum_i gauss_i * amp_i``. All 60
    electrodes are driven (worst case for a per-electrode cutoff), and the
    amplitude is varied because the bound scales with it.
    """
    stim = np.full(60, amp)
    model = ModelClass(implant=ArgusII(), step=0.75, xrange=(-14, 14),
                       yrange=(-10, 10), rho=200).build()
    default = model.predict_percept(stim).data
    monkeypatch.setattr(SpatialModel, '_cutoff_r2',
                        lambda self, rho: np.float32(np.inf))
    exact = model.predict_percept(stim).data
    # Documented bound: the cutoff fraction times the summed amplitude, plus
    # float32 accumulation error:
    dropped = _GAUSSIAN_CUTOFF * np.abs(stim).sum()
    assert np.abs(default - exact).max() <= dropped + 1e-6 * np.abs(exact).max()

    # Points far from all electrodes become exactly zero (100% relative
    # error, but within the absolute bound):
    zeroed = (np.abs(exact) > 0) & (default == 0)
    assert zeroed.any()
    assert np.abs(exact[zeroed]).max() <= dropped


@pytest.mark.parametrize('ModelClass', (ScoreboardModel, AxonMapModel))
def test_predict_percept_frames_are_independent(ModelClass):
    """Each frame of a multi-frame stimulus is predicted independently

    The kernels compute the electrode-to-point Gaussian once and reuse it
    across time points.
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
    """An all-zero stimulus produces an all-zero percept

    The kernels skip electrodes that are zero for the whole stimulus; here
    every electrode is skipped.
    """
    model = ModelClass(implant=ArgusII(), step=1, xrange=(-10, 10),
                       yrange=(-8, 8)).build()
    percept = model.predict_percept(np.zeros(60))
    npt.assert_equal(np.all(percept.data == 0), True)


def test_AxonMapSpatial_cutoff_band_boundaries(monkeypatch):
    """An electrode exactly on either edge of the cutoff still contributes

    Candidate electrodes form the sorted x band ``[ax_x - r, ax_x + r]``.
    ``cutoff_r2`` is an exact float32 square, so ``sqrt`` rounding does not
    affect the band edges.
    """
    monkeypatch.setattr(AxonMapSpatial, '_cutoff_r2',
                        lambda self, rho: np.float32(360000.0))  # r = 600 um
    spatial = AxonMapSpatial(ArgusII(), rho=200, step=1, xrange=(-2, 2),
                             yrange=(-2, 2), n_axons=50, n_ax_segments=50,
                             meridian_blend=0).build()
    names = list(spatial.implant.electrode_names)
    # Pixel 0 gets one segment at the origin, sensitivity 1; no other axons:
    n_px = spatial.grid.x.size
    spatial.axon_contrib = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    spatial.axon_idx_start = np.array([0] + [1] * (n_px - 1))
    spatial.axon_idx_end = np.ones(n_px, dtype=int)

    def bright(x_el):
        # Place the first electrodes at `x_el` on the x axis:
        x_el = np.asarray(x_el, dtype=np.float32)
        coords = (x_el, np.zeros_like(x_el), np.zeros_like(x_el))
        with monkeypatch.context() as m:
            m.setattr(AxonMapSpatial, '_electrode_coords',
                      lambda *args, **kwargs: coords)
            stim = Stimulus(-np.ones(len(x_el)),
                            electrodes=names[:len(x_el)])
            percept = spatial.predict_percept(stim)
        return percept.data.ravel()[0]

    for x in (-600.0, 600.0):
        npt.assert_array_less(0.0, abs(bright([x])))
    for x in (-600.5, 600.5):
        npt.assert_equal(bright([x]), 0.0)

    # Electrodes outside the band are dropped:
    x_el = np.array([-900., -600.5, -600., -300., 0., 300., 600., 600.5,
                     900.], dtype=np.float32)
    two_rho2 = np.float32(2 * 200 ** 2)
    want = -np.sum(np.exp(-x_el[np.abs(x_el) <= 600] ** 2 / two_rho2))
    npt.assert_allclose(bright(x_el), want, rtol=1e-6)


def test_AxonMapSpatial_matches_frozen_cython():
    """The Torch core reproduces the v0.11 Cython ``fast_axon_map``

    Expected values were recorded from ``fast_axon_map`` (``bf72e31``) with
    the same ``rho``, threshold and cutoff. The winning segment changes over
    time, negative responses win, one segment lies beyond the cutoff, and
    pixel 2 has an empty axon.
    """
    spatial = AxonMapSpatial(ArgusII(), rho=100, thresh_percept=0)
    # Electrodes deliberately out of x order:
    x_el = np.array([300, -200, 50, 450], dtype=np.float32)
    y_el = np.array([0, 100, -150, 50], dtype=np.float32)
    stim = torch.tensor([[1.0, -0.5, 0.0],
                         [-0.8, 0.3, 1.2],
                         [0.6, 0.9, -0.4],
                         [0.0, -1.1, 0.7]])
    # (x, y, sensitivity) per segment:
    spatial.axon_contrib = np.array([
        [-150, 80, 0.9], [0, 0, 0.6], [200, -50, 1.0], [380, 30, 0.4],
        [100, -100, 0.5], [-100, 50, 0.8], [900, 0, 1.0],
        [420, 60, 0.7], [250, -20, 0.3]], dtype=np.float32)
    spatial.axon_idx_start = np.array([0, 4, 7, 7])
    spatial.axon_idx_end = np.array([4, 7, 7, 9])
    want = np.array([[0.6533213, -0.47610235, 0.9307646],
                     [-0.32124075, 0.32977402, 0.49979118],
                     [0, 0, 0],
                     [0.28464806, -0.8746721, 0.46606955]], dtype=np.float32)
    with torch.inference_mode():
        got = spatial._predict_axon_map_tensor(stim, x_el, y_el).numpy()
    npt.assert_allclose(got, want, rtol=1e-6)
    # Empty axon is +0.0, as in Cython:
    npt.assert_equal(got[2], np.zeros(3, dtype=np.float32))


def test_predict_percept_thread_count_invariant():
    """The percept does not depend on the number of threads"""
    stim = np.zeros(60)
    stim[[5, 22, 51]] = [1.0, 0.6, -0.3]
    kwargs = {'implant': ArgusII(), 'step': 1, 'xrange': (-10, 10),
              'yrange': (-8, 8), 'rho': 200}

    serial = ScoreboardModel(n_threads=1,
                             **kwargs).build().predict_percept(stim).data
    for n_threads in (2, 3, 8):
        parallel = ScoreboardModel(
            n_threads=n_threads, **kwargs).build().predict_percept(stim)
        npt.assert_array_equal(parallel.data, serial)


@pytest.mark.parametrize('cls', (AxonMapSpatial, AxonMapModel))
def test_AxonMap_has_no_thread_or_cutoff_params(cls):
    # Removed in 0.12: prediction runs on Torch, and the cutoff is fixed:
    for param in ('n_threads', 'n_jobs', 'min_current_spread'):
        with pytest.raises(TypeError):
            cls(ArgusII(), **{param: 2})


def test_AxonMapModel_find_closest_axon_return_segment():
    """``return_segment`` returns the index of the closest axon segment"""
    model = AxonMapModel(implant=ArgusII(), step=2, n_axons=20, xrange=(-12, 12),
                         yrange=(-12, 12), axons_range=(-45, 45))
    model.build()
    spatial = model.spatial
    bundles = spatial.grow_axon_bundles()
    xyret = np.column_stack((spatial.grid.ret.x.ravel(),
                             spatial.grid.ret.y.ravel()))

    axons, idx_seg = spatial.find_closest_axon(bundles, return_segment=True)
    npt.assert_equal(len(idx_seg), len(xyret))
    # Same as `argmin` of the squared distance:
    for axon, seg, xy in zip(axons, idx_seg, xyret):
        expected = np.argmin((axon[:, 0] - xy[0]) ** 2 +
                             (axon[:, 1] - xy[1]) ** 2)
        npt.assert_equal(seg, expected)

    # Both flags, in the documented order:
    axons2, idx_ax, idx_seg2 = spatial.find_closest_axon(
        bundles, return_index=True, return_segment=True)
    npt.assert_array_equal(idx_seg2, idx_seg)
    for axon, idx in zip(axons2, idx_ax):
        npt.assert_array_equal(axon, bundles[idx])

    # A single query point returns scalars:
    single, idx_ax1, idx_seg1 = spatial.find_closest_axon(
        bundles, xret=xyret[0, 0], yret=xyret[0, 1], return_index=True,
        return_segment=True)
    npt.assert_equal(np.ndim(idx_ax1), 0)
    npt.assert_equal(np.ndim(idx_seg1), 0)
    npt.assert_array_equal(single, bundles[idx_ax1])


def test_AxonMapModel_calc_axon_sensitivity_empty_bundle():
    """A bundle with no segments raises ValueError"""
    model = AxonMapModel(implant=ArgusII(), step=4, n_axons=5, xrange=(-8, 8), yrange=(-8, 8))
    model.build()
    n_points = model.spatial.grid.ret.x.size
    bundles = [np.zeros((0, 2), dtype=np.float32)] * n_points
    with pytest.raises(ValueError):
        model.spatial.calc_axon_sensitivity(bundles)


def test_AxonMapModel_build_cache_roundtrip(tmp_path):
    """A build from the cache matches the uncached build exactly"""
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

    # A cache from an older version is regrown:
    with open(pickle_file, 'rb') as f:
        params, _ = pickle.load(f)
    with open(pickle_file, 'wb') as f:
        pickle.dump((params, [np.zeros((3, 2), dtype=np.float32)]), f)
    stale = build(False)
    npt.assert_array_equal(stale.axon_contrib, cold.axon_contrib)
    # File is rewritten in the current format:
    with open(pickle_file, 'rb') as f:
        _, payload = pickle.load(f)
    npt.assert_equal(payload[0], _AXON_CACHE_VERSION)

    # v3 bundles came from the Cython Jansonius kernel and are regrown, even
    # though the layout is unchanged. Shifted bundles expose a reuse:
    npt.assert_equal(_AXON_CACHE_VERSION, 4)
    _, bundles, bundle_id, idx_segment = payload
    with open(pickle_file, 'wb') as f:
        pickle.dump((params, (3, [b + 1 for b in bundles], bundle_id,
                              idx_segment)), f)
    npt.assert_array_equal(build(False).axon_contrib, cold.axon_contrib)


def test_AxonMapModel_build_rejects_pre_step_cache(tmp_path):
    """A pre-0.10.0 cache with `xystep` is regrown without a warning

    The parameter dict is versioned with the payload, so the old cache is
    discarded.
    """
    pickle_file = str(tmp_path / 'axons.pickle')

    def build(ignore_pickle=False):
        return AxonMapModel(implant=ArgusII(), step=1, xrange=(-8, 8), yrange=(-8, 8),
                            n_axons=200, axon_pickle=pickle_file,
                            ignore_pickle=ignore_pickle).build().spatial

    cold = build(ignore_pickle=True)
    with open(pickle_file, 'rb') as f:
        params, payload = pickle.load(f)
    # Rewrite it in the v0.9.1 format:
    params['xystep'] = params.pop('step')
    with open(pickle_file, 'wb') as f:
        pickle.dump((params, (2, *payload[1:])), f)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        warm = build()
    npt.assert_array_equal(warm.axon_contrib, cold.axon_contrib)
    # Stale file was replaced:
    with open(pickle_file, 'rb') as f:
        params, payload = pickle.load(f)
    npt.assert_equal('xystep' in params, False)
    npt.assert_equal(payload[0], _AXON_CACHE_VERSION)


def _spatial(model):
    """Return the spatial model, or the one wrapped by a composite model"""
    return getattr(model, 'spatial', model)


def _straddling_pair(coord):
    """Return the indices nearest zero from below and above"""
    below = np.flatnonzero(coord < 0)
    above = np.flatnonzero(coord > 0)
    return below[np.argmax(coord[below])], above[np.argmin(coord[above])]


@pytest.mark.parametrize('ModelClass', [AxonMapSpatial, AxonMapModel])
def test_AxonMapSpatial_meridian_blend(ModelClass):
    def make(**params):
        # Half-step offset so the nearest rows straddle the raphe:
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
    # The raphe separates the two halves of the axon map:
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
    # Changed rows stay within 4 blend widths of the raphe:
    npt.assert_array_less(np.abs(y[rows]).max(), 4 * width)
    # Changed columns span the grid:
    npt.assert_array_less(4 * width, np.abs(x[cols]).max())


def test_AxonMapSpatial_meridian_blend_reapplies_threshold():
    # Blending can lift sub-threshold points above zero, so `thresh_percept`
    # is applied again afterward:
    model = AxonMapSpatial(implant=ArgusII(), xrange=(-6, 6), yrange=(-6, 6),
                           step=0.25, rho=200, lam=400, n_axons=250,
                           n_ax_segments=200, ignore_pickle=True,
                           meridian_blend=1, thresh_percept=0.1).build()
    data = model.predict_percept({'C4': 1}).data
    npt.assert_equal(np.any(data > 0), True)
    # No values strictly between zero and the threshold:
    npt.assert_equal(np.any((np.abs(data) > 0) & (np.abs(data) < 0.1)), False)


def test_AxonMapSpatial_meridian_blend_over_time():
    # Each frame is blended independently:
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
    """`axons_range` accepts angle units and is stored in degrees"""
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
    """Return the UserWarning messages emitted by `build`

    Ignores ResourceWarnings from the pickle cache.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        build()
    return [str(w.message) for w in caught
            if issubclass(w.category, UserWarning)]


def test_axon_map_eye_follows_the_implant():
    """`eye` follows the implant's eye"""
    implant = ArgusII(eye='right')
    model = AxonMapModel(implant=implant, step=2, n_axons=50,
                         n_ax_segments=30).build()
    npt.assert_equal(model.spatial.eye, 'right')
    # The optic disc is on the nasal side, which differs per eye:
    npt.assert_equal(model.spatial.loc_od[0] > 0, True)

    # Changing the bound implant's eye resets `is_built`:
    implant.eye = 'left'
    npt.assert_equal(model.spatial.eye, 'left')
    npt.assert_equal(model.is_built, False)
    model.predict_percept({'A1': 20})
    npt.assert_equal(model.is_built, True)
    npt.assert_equal(model.spatial.loc_od[0] < 0, True)


def test_axon_map_needs_an_implant_with_an_eye():
    """A generic implant without an eye raises TypeError

    ``eye`` was removed from :py:class:`~pulse2percept.implants.Implant` in
    0.11; the optic disc location requires it.
    """
    from pulse2percept.implants import ElectrodeGrid, GridImplant
    from pulse2percept.implants.retina import RetinalImplant
    grid = dict(rho=200, xrange=(-2, 2), yrange=(-2, 2), step=1, n_axons=50,
                n_ax_segments=30, ignore_pickle=True)
    model = AxonMapModel(implant=GridImplant((3, 3), 2000), **grid)
    with pytest.raises(TypeError) as excinfo:
        model.build()
    npt.assert_equal('RetinalImplant' in str(excinfo.value), True)
    # Same for the property:
    with pytest.raises(TypeError):
        model.spatial.eye
    # Wrapping the array in a RetinalImplant works:
    fixed = AxonMapModel(
        implant=RetinalImplant(ElectrodeGrid((3, 3), 2000), eye='left'),
        **grid).build()
    npt.assert_equal(fixed.spatial.eye, 'left')
    npt.assert_equal(fixed.is_built, True)


def test_axon_map_warns_when_the_implant_is_not_epiretinal():
    from pulse2percept.implants import ElectrodeGrid
    from pulse2percept.implants.retina import (Lorach2015Array,
                                               RetinalImplant)
    grid = dict(step=1, xrange=(-2, 2), yrange=(-2, 2), n_axons=50,
                n_ax_segments=30)
    said = _user_warnings(AxonMapModel(implant=Lorach2015Array(), **grid).build)
    npt.assert_equal(any('subretinal' in w for w in said), True)
    npt.assert_equal(any('scoreboard model' in w for w in said), True)
    # No warning for an implant without a placement. Its pitch is wide enough
    # to avoid the rho warning:
    quiet = RetinalImplant(ElectrodeGrid((3, 3), 2000))
    npt.assert_equal(_user_warnings(AxonMapModel(implant=quiet, **grid).build),
                     [])


@pytest.mark.parametrize('ModelClass', [ScoreboardModel, AxonMapModel])
def test_rho_wider_than_the_electrode_pitch_warns(ModelClass):
    from pulse2percept.implants import ElectrodeGrid
    from pulse2percept.implants.retina import RetinalImplant
    extra = {'n_axons': 50, 'n_ax_segments': 30} if ModelClass is AxonMapModel         else {}
    grid = dict(step=1, xrange=(-2, 2), yrange=(-2, 2), **extra)
    dense = ModelClass(implant=RetinalImplant(ElectrodeGrid((3, 3), 100)),
                       rho=400, **grid)
    said = _user_warnings(dense.build)
    # Warning reports pitch and ratio:
    npt.assert_equal(any('pitch (100 um)' in w for w in said), True)
    npt.assert_equal(any('ratio of 4.00' in w for w in said), True)
    # No warning for rho equal to the pitch:
    matched = ModelClass(implant=RetinalImplant(ElectrodeGrid((3, 3), 400)),
                         rho=400, **grid)
    npt.assert_equal(_user_warnings(matched.build), [])


@pytest.mark.parametrize('ModelClass', [ScoreboardModel, AxonMapModel])
def test_electrode_pitch_ignores_a_dimension_the_model_drops(ModelClass):
    """Retinal pitch uses x and y only, ignoring z"""
    from pulse2percept.implants import DiskElectrode, ElectrodeArray
    from pulse2percept.implants.retina import RetinalImplant
    extra = {'n_axons': 50, 'n_ax_segments': 30} if ModelClass is AxonMapModel         else {}
    # 100 um apart in x, 1000 um in z (a 3D pitch would be ~1005 um):
    stacked = RetinalImplant(ElectrodeArray(
        [DiskElectrode(100 * i, 0, 1000 * i, 50) for i in range(3)]))
    model = ModelClass(implant=stacked, rho=400, step=1, xrange=(-2, 2),
                       yrange=(-2, 2), **extra)
    said = _user_warnings(model.build)
    npt.assert_equal(any('pitch (100 um)' in w for w in said), True)


def test_scoreboard_visualizes_a_photovoltaic_implant():
    """Scoreboard accepts normalized optical drive from PRIMA"""
    implant = PRIMAPivotal()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        model = ScoreboardModel(implant=implant, rho=200, step=0.05,
                                xrange=(-2, 2), yrange=(-2, 2))
        percept = model.predict_percept(samples.logo_bvl())
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, tuple(model.spatial.grid.x.shape) + (1,))
    npt.assert_equal(np.all(np.isfinite(percept.data)), True)
    npt.assert_equal(percept.data.max() > 0, True)
    # Delivered stimulation is optical; the spatial view is normalized.
    delivered = implant.prepare_stim(samples.logo_bvl())
    npt.assert_equal(delivered.unit, mW / mm ** 2)
    npt.assert_equal(delivered._spatial_view().unit, dimensionless)
    # Dark input produces zero drive.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        dark = model.predict_percept(ImageStimulus(np.zeros((32, 32))))
    npt.assert_almost_equal(dark.data.max(), 0)


def test_scoreboard_visualizes_a_photovoltaic_video():
    """Scoreboard accepts normalized drive after video resampling"""

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
    bare = Stimulus(implant.prepare_stim(samples.logo_bvl()))
    npt.assert_equal(bare._has_spatial_view, False)
    with pytest.raises(DimensionMismatchError) as excinfo:
        model.predict_percept(bare)
    npt.assert_equal('irradiance' in str(excinfo.value), True)
