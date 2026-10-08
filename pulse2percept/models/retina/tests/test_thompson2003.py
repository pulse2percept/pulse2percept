from types import SimpleNamespace
import numpy as np
import copy
import pytest
import numpy.testing as npt

from matplotlib.axes import Subplot
import matplotlib.pyplot as plt


from pulse2percept.implants import ElectrodeArray, Implant, PointSource
from pulse2percept.implants.retina import ArgusI, ArgusII
from pulse2percept.percepts import Percept
from pulse2percept.models import FadingTemporal, Model
from pulse2percept.models.retina import Thompson2003Spatial, Thompson2003Model
from pulse2percept.models.retina import thompson2003
from pulse2percept.stimuli import BiphasicPulseTrain, Stimulus
from pulse2percept.topography.retina import (Curcio1990Map,
                                             Montesano2020Map)
from pulse2percept.utils.testing import assert_warns_msg


def test_Thompson2003Spatial():
    # Thompson2003Spatial automatically sets `radius`:
    model = Thompson2003Spatial(implant=ArgusI(), step=5)
    # User can set `radius`:
    model.radius = 123
    npt.assert_equal(model.radius, 123)
    model.build(radius=987)
    npt.assert_equal(model.radius, 987)

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(None), None)

    # Converting ret <=> dva
    model2 = Thompson2003Spatial(implant=ArgusI(),
                                 visual_field_map=Montesano2020Map())
    npt.assert_equal(isinstance(model2.visual_field_map,
                                Montesano2020Map),
                     True)

    # Zero in = zero out:
    percept = model.predict_percept(np.zeros(16))
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [1])
    npt.assert_almost_equal(percept.data, 0)

    # Multiple frames are processed independently:
    model = Thompson2003Spatial(implant=ArgusI(), radius=200, step=5,
                                xrange=(-20, 20), yrange=(-15, 15))
    model.build()
    percept = model.predict_percept({'A1': [1, 0], 'B3': [0, 2]})
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [2])
    pmax = percept.data.max(axis=(0, 1))
    npt.assert_almost_equal(percept.data[2, 3, 0], pmax[0])
    npt.assert_almost_equal(percept.data[2, 3, 1], 0)
    npt.assert_almost_equal(percept.data[3, 4, 0], 0)
    npt.assert_almost_equal(percept.data[3, 4, 1], pmax[1])
    npt.assert_almost_equal(percept.time, [0, 1])


def test_deepcopy_Thompson2003Spatial():
    original = Thompson2003Spatial(implant=ArgusII())
    copied = copy.deepcopy(original)

    # Assert they are different objects
    npt.assert_equal(id(original) != id(copied), True)

    # Assert the objects are equivalent to each other
    npt.assert_equal(original == copied, True)

    # Assert building one object does not affect the copied
    original.build()
    npt.assert_equal(copied.is_built, False)
    npt.assert_equal(original != copied, True)

    # Change the copied attribute by "destroying" the visual_field_map
    # attribute which should be unique to each SpatialModel object
    copied = copy.deepcopy(original)
    copied.visual_field_map = None
    npt.assert_equal(original.visual_field_map is not None, True)
    npt.assert_equal(original != copied, True)

    # Assert "destroying" the original doesn't affect the copied
    original = None
    npt.assert_equal(copied is not None, True)


def test_Thompson2003Model():
    model = Thompson2003Model(implant=ArgusI(), step=5)
    npt.assert_equal(model.has_space, True)
    npt.assert_equal(model.has_time, False)
    npt.assert_equal(hasattr(model.spatial, 'radius'), True)

    # User can set `radius`:
    model.spatial.radius = 123
    npt.assert_equal(model.spatial.radius, 123)
    model.spatial.build(radius=987)
    npt.assert_equal(model.spatial.radius, 987)

    # Converting ret <=> dva
    npt.assert_equal(isinstance(model.spatial.visual_field_map, Curcio1990Map),
                     True)
    npt.assert_almost_equal(model.spatial.visual_field_map.ret_to_dva(0, 0),
                            (0, 0))
    npt.assert_almost_equal(model.spatial.visual_field_map.dva_to_ret(0, 0),
                            (0, 0))
    model2 = Thompson2003Model(implant=ArgusI(),
                               visual_field_map=Montesano2020Map())
    npt.assert_equal(isinstance(model2.spatial.visual_field_map,
                                Montesano2020Map),
                     True)
    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(None), None)

    # Zero in = zero out:
    npt.assert_almost_equal(model.predict_percept(np.zeros(16)).data, 0)

    # Multiple frames are processed independently:
    model = Thompson2003Model(implant=ArgusI(), radius=1000, step=5,
                              xrange=(-20, 20), yrange=(-15, 15))
    model.build()
    percept = model.predict_percept({'A1': [1, 2]})
    npt.assert_equal(percept.shape,
                     list(model.spatial.grid.x.shape) + [2])
    pmax = percept.data.max(axis=(0, 1))
    npt.assert_almost_equal(percept.data[2, 3, :], pmax)
    print(pmax, percept.data)
    npt.assert_almost_equal(pmax[1] / pmax[0], 2.0)
    npt.assert_almost_equal(percept.time, [0, 1])


def test_Thompson2003Model_predict_percept():
    model = Thompson2003Model(implant=ArgusII(), step=0.55, radius=100, thresh_percept=0,
                              xrange=(-20, 20), yrange=(-15, 15))
    model.build()
    # Single-electrode stim:
    img_stim = np.zeros(60)
    img_stim[47] = 1
    percept = model.predict_percept(img_stim)
    # Single bright pixel, very small Gaussian kernel:
    npt.assert_equal(np.sum(percept.data > 0.5), 1)
    npt.assert_equal(np.sum(percept.data > 0.00001), 1)
    # Brightest pixel is in lower right:
    npt.assert_almost_equal(percept.data[33, 46, 0], np.max(percept.data))

    # Full Argus II: 60 bright spots
    model = Thompson2003Model(implant=ArgusII(), step=0.55, radius=100)
    model.build()
    percept = model.predict_percept(np.ones(60))
    npt.assert_equal(np.sum(np.isclose(percept.data, 1.0, rtol=0.1, atol=0.1)),
                     84)

    # Model gives same outcome as Spatial:
    spatial = Thompson2003Spatial(implant=ArgusII(), step=1, radius=100)
    spatial.build()
    spatial_percept = model.predict_percept(np.ones(60))
    npt.assert_almost_equal(percept.data, spatial_percept.data)
    npt.assert_equal(percept.time, None)

    # Warning for nonzero electrode-retina distances
    raised = Thompson2003Model(implant=ArgusII(z=10), step=0.55, radius=100)
    raised.build()
    # Warning names the model:
    assert_warns_msg(UserWarning, raised.predict_percept,
                     "Thompson2003Spatial does not model electrode-retina distance",
                     np.ones(60))
    assert_warns_msg(UserWarning, raised.predict_percept,
                     "not parameterized by this model", np.ones(60))


def test_deepcopy_Thompson2003Model():
    original = Thompson2003Model(implant=ArgusII())
    copied = copy.deepcopy(original)

    # Assert they are different objects
    npt.assert_equal(id(original) != id(copied), True)

    # Assert the objects are equivalent to each other
    npt.assert_equal(original == copied, True)

    # Assert building one object does not affect the copied
    original.build()
    npt.assert_equal(copied.is_built, False)
    npt.assert_equal(original != copied, True)

    # Change the copied attribute by "destroying" the visual_field_map
    # attribute which should be unique to each SpatialModel object
    copied = copy.deepcopy(original)
    copied.spatial.visual_field_map = None
    npt.assert_equal(original.spatial.visual_field_map is not None, True)
    npt.assert_equal(original != copied, True)

    # Assert "destroying" the original doesn't affect the copied
    original = None
    npt.assert_equal(copied is not None, True)


def _row_spatial(x_el, radius, **params):
    """Return a built Thompson2003Spatial with point electrodes at ``x_el``
    and grid points at x = 0, 140, 280, 420, 560 um."""
    electrodes = [PointSource(x, 0, 0) for x in x_el]
    return Thompson2003Spatial(
        Implant(ElectrodeArray(electrodes)), radius=radius, xrange=(0, 2),
        yrange=(0, 0), step=0.5, **params).build()


def _spatial_paths(model, amps):
    """Return the public and tensor responses to static amplitudes."""
    import torch
    names = model.implant.electrode_names
    public = model.predict_percept(
        Stimulus(np.array(amps, dtype=float), electrodes=names))
    tensor = model._predict_tensor(
        torch.tensor(amps, dtype=torch.float32)[:, None], None)
    return public.data.ravel(), tensor.data.numpy().ravel()


def test_Thompson2003Spatial_disk_is_open():
    # 0 and 140 um are inside; 280 um is exactly on the radius and excluded:
    model = _row_spatial([0], 280)
    npt.assert_equal(model.grid.ret.x.ravel(), [0, 140, 280, 420, 560])
    for got in _spatial_paths(model, [3]):
        npt.assert_equal(got, [3, 3, 0, 0, 0])


@pytest.mark.parametrize('amps, want', [
    ([10, 5], [10, 15, 15, 5, 0]),
    ([10, -25], [10, -15, -15, -25, 0]),
])
def test_Thompson2003Spatial_sums_signed_amplitudes(amps, want):
    # The disks around 140 and 280 um overlap at 140 and 280 um:
    model = _row_spatial([140, 280], 200)
    for got in _spatial_paths(model, amps):
        npt.assert_equal(got, want)


@pytest.mark.parametrize('thresh, want', [
    (2, [10, -2, -2, -12, 0]),
    (2.5, [10, 0, 0, -12, 0]),
])
def test_Thompson2003Spatial_threshold(thresh, want):
    # |-2| == 2 is kept; |-2| < 2.5 is zeroed:
    model = _row_spatial([140, 280], 200, thresh_percept=thresh)
    for got in _spatial_paths(model, [10, -12]):
        npt.assert_equal(got, want)


def test_Thompson2003Spatial_nan_grid_point_is_zero():
    model = _row_spatial([0], 300)
    model.grid.ret.x[0, 1] = np.nan
    for got in _spatial_paths(model, [3]):
        npt.assert_equal(got, [3, 0, 3, 0, 0])


def test_Thompson2003Spatial_dropout(monkeypatch):
    # Drops electrode 0 in frame 0 and electrode 1 in frame 1:
    calls = []

    def sample(electrodes, k):
        calls.append((electrodes.copy(), k))
        return [len(calls) - 1]

    monkeypatch.setattr(thompson2003, 'sample', sample)
    model = _row_spatial([140, 280], 200, dropout=1)
    percept = model.predict_percept(
        Stimulus([[10, 20], [5, 7]], electrodes=[0, 1], time=[0, 1]))
    npt.assert_equal(percept.data[0, :, 0], [0, 5, 5, 5, 0])
    npt.assert_equal(percept.data[0, :, 1], [20, 20, 20, 0, 0])
    # One draw per frame, over all stimulated electrodes:
    npt.assert_equal(len(calls), 2)
    for electrodes, k in calls:
        npt.assert_equal(electrodes, [0, 1])
        npt.assert_equal(k, 1)


def test_Thompson2003Spatial_dropout_matches_sampled_mask(monkeypatch):
    # Real sampling: the response omits exactly the sampled electrodes.
    drawn = []

    def sample(*args, **kwargs):
        drawn.append(thompson2003_sample(*args, **kwargs))
        return drawn[-1]

    thompson2003_sample = thompson2003.sample
    monkeypatch.setattr(thompson2003, 'sample', sample)
    model = Thompson2003Spatial(ArgusI(), radius=500, dropout=0.25, step=1)
    amps = np.random.default_rng(1).uniform(1, 2, (16, 3))
    percept = model.predict_percept(Stimulus(amps, time=[0, 1, 2]))
    npt.assert_equal(len(drawn), 3)
    model.dropout = None
    for t, dropped in enumerate(drawn):
        frame = amps[:, t].copy()
        frame[dropped] = 0
        want = model.predict_percept(Stimulus(frame)).data[..., 0]
        npt.assert_allclose(percept.data[..., t], want, rtol=1e-6)


@pytest.mark.parametrize('dropout, tensor', [
    (None, True), (0, True), (0.0, True), (2, False), (0.25, False),
])
def test_Thompson2003Spatial_tensor_core_requires_no_dropout(dropout, tensor):
    import torch
    model = Model(Thompson2003Spatial(ArgusI(), dropout=dropout, step=1),
                  FadingTemporal()).build()
    npt.assert_equal(model._has_tensor_core, tensor)
    waveform = torch.zeros((16, 2))
    if tensor:
        model.spatial._predict_tensor(waveform, [0, 1])
    else:
        with pytest.raises(NotImplementedError, match='dropout'):
            model.spatial._predict_tensor(waveform, [0, 1])


def test_Thompson2003_composite_with_dropout_applies_it(monkeypatch):
    # Drops every electrode, so the staged route must return zero:
    monkeypatch.setattr(thompson2003, 'sample',
                        lambda electrodes, k: electrodes)
    model = Model(Thompson2003Spatial(ArgusI(), radius=400, step=1,
                                      dropout=16),
                  FadingTemporal()).build()
    stim = {'A1': BiphasicPulseTrain(20, 30, 0.45, stim_dur=50)}
    npt.assert_equal(model.predict_percept(stim).data, 0)
    model.spatial.dropout = None
    npt.assert_equal(model.predict_percept(stim).data.max() > 0, True)


def test_Thompson2003Spatial_tensor_matches_predict_percept():
    import torch
    model = Thompson2003Spatial(ArgusII(), radius=400, thresh_percept=0.5,
                                xrange=(-12, 12), yrange=(-8, 8), step=0.5)
    model.build()
    wf = np.random.default_rng(42).normal(0, 3, (60, 5))
    wf[::4] = 0
    time = np.arange(5.0)
    public = model.predict_percept(
        Stimulus(wf, electrodes=model.implant.electrode_names, time=time))
    resp = model._predict_tensor(torch.tensor(wf, dtype=torch.float32), time)
    assert isinstance(resp.data, torch.Tensor)
    want = public.data.reshape(resp.data.shape)
    # The threshold must zero some, but not all, of the response:
    assert 0 < np.mean(want == 0) < 1
    npt.assert_allclose(resp.data.numpy(), want, rtol=1e-6, atol=1e-5)


def test_Thompson2003Spatial_tensor_gradcheck():
    import torch
    # thresh_percept=0 keeps every response off the threshold boundary:
    model = _row_spatial([140, 280], 200)
    waveform = torch.tensor([[10.0, -3.0], [5.0, 2.0]], dtype=torch.float64,
                            requires_grad=True)
    torch.autograd.gradcheck(
        lambda w: model._predict_tensor(w, [0, 1]).data, (waveform,))
    # The Jacobian is the disk-incidence matrix:
    resp = model._predict_tensor(waveform, [0, 1]).data
    resp[:, 0].sum().backward()
    npt.assert_equal(waveform.grad.numpy(), [[3, 0], [3, 0]])


def test_Thompson2003Model_with_temporal_runs_on_torch():
    model = Model(Thompson2003Spatial(ArgusI(), radius=400, step=0.5,
                                      thresh_percept=0.5),
                  FadingTemporal()).build()
    stim = {'A1': BiphasicPulseTrain(20, 30, 0.45, stim_dur=50)}
    npt.assert_equal(model._uses_tensor_core(model._prepared(stim)), True)
    percept = model.predict_percept(stim)
    npt.assert_equal(isinstance(percept.data, np.ndarray), True)
    npt.assert_equal(percept.data.max() > 0, True)
