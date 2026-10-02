"""Torch execution of {Scoreboard,AxonMap}Spatial -> FadingTemporal."""
import numpy as np
import numpy.testing as npt
import pytest
import torch

from pulse2percept.implants import GridImplant
from pulse2percept.implants.retina import ArgusI, ArgusII, PRIMAPivotal
from pulse2percept.models import AlphaTemporal, FadingTemporal, Model
from pulse2percept.models.base import _blend_meridian, _ModelResponse
from pulse2percept.models.retina import (AxonMapSpatial, ScoreboardSpatial,
                                         Thompson2003Spatial)
from pulse2percept.models.retina import beyeler2019
from pulse2percept.stimuli import AmplitudeEncoder, ImageStimulus, Stimulus
from pulse2percept.topography import Grid2D
from pulse2percept.topography.retina import Curcio1990Map
from pulse2percept.units import ms

# float32 tolerances: accumulation order differs from the Cython loops. ATOL
# covers cancellation in mixed-polarity sums of ~30 uA terms.
RTOL, ATOL = 1e-6, 1e-5

# Sample times (ms). Steps of 1-2 us put several transitions within one
# dt=5 us simulation step.
TIME = np.array([0, 0.3, 0.301, 0.302, 0.75, 1.2, 1.2013, 2.0, 9.0, 23.0,
                 31.0, 47.5, 52.0])


def _waveform(n_el, seed=42):
    """Return mixed-polarity amplitudes (uA) with silent electrodes/samples."""
    rng = np.random.default_rng(seed)
    wf = rng.normal(0, 30, (n_el, TIME.size))
    wf[:, [2, 7, -1]] = 0
    wf[::4] = 0
    return wf


def _spatial(**params):
    params = {'xrange': (-6, 6), 'yrange': (-5, 5), 'step': 0.5,
              'thresh_percept': 0.5, **params}
    return ScoreboardSpatial(ArgusI(), **params)


def _model(reduce='peak', **params):
    return Model(_spatial(**params), FadingTemporal(tau=2, reduce=reduce))


@pytest.mark.parametrize('params', [
    {},
    {'min_current_spread': 0.05, 'thresh_percept': 2},
    {'implant_position': (300, -200), 'implant_rotation': 20},
    {'location_noise': 0.5},
])
def test_ScoreboardSpatial_tensor_parity(params):
    spatial = _spatial(**params).build()
    wf = _waveform(spatial.implant.n_electrodes)
    # Every column differs, so compression keeps all time points:
    expected = spatial.predict_percept(
        Stimulus(wf, electrodes=spatial.implant.electrode_names, time=TIME))
    resp = spatial._predict_tensor(torch.tensor(wf, dtype=torch.float32),
                                   TIME)
    assert isinstance(resp.data, torch.Tensor)
    assert resp.data.shape == (spatial.grid.x.size, TIME.size)
    assert resp.shape == spatial.grid.x.shape
    assert resp.space is spatial.grid
    assert resp.frame_clock is None
    npt.assert_allclose(resp.data.numpy(),
                        expected.data.reshape(resp.data.shape),
                        rtol=RTOL, atol=ATOL)


def test_ScoreboardSpatial_tensor_parity_large_grid():
    # ~100 of 1600 electrodes contribute to each grid point:
    spatial = ScoreboardSpatial(GridImplant((40, 40), 70), rho=65,
                                xrange=(-5, 5), yrange=(-5, 5), step=0.25,
                                thresh_percept=5).build()
    wf = _waveform(spatial.implant.n_electrodes)
    expected = spatial.predict_percept(
        Stimulus(wf, electrodes=spatial.implant.electrode_names, time=TIME))
    resp = spatial._predict_tensor(torch.tensor(wf, dtype=torch.float32),
                                   TIME)
    expected = expected.data.reshape(resp.data.shape)
    # The threshold must zero some, but not all, of the response:
    assert 0 < np.mean(expected == 0) < 1
    # float32 rounding grows with the summed magnitude, so bound the error
    # relative to the peak response:
    npt.assert_allclose(resp.data.numpy(), expected, rtol=RTOL,
                        atol=1e-6 * np.abs(expected).max())


@pytest.mark.parametrize('meridian', ['vertical', 'horizontal'])
@pytest.mark.parametrize('width', [0.1, 0.5, 3.0])
def test_blend_meridian_tensor_parity(meridian, width):
    grid = Grid2D((-4.1, 3.9), (-3, 3.4), step=0.2)
    grid.build(Curcio1990Map())
    resp = np.random.default_rng(1).normal(0, 10, (grid.x.size, 3))
    for dtype, atol in ((np.float32, ATOL), (np.float64, 1e-12)):
        expected = _blend_meridian(resp.astype(dtype), grid, meridian, width)
        blended = _blend_meridian(torch.tensor(resp.astype(dtype)), grid,
                                  meridian, width)
        assert blended.dtype == torch.from_numpy(expected).dtype
        npt.assert_allclose(blended.numpy(), expected, rtol=RTOL, atol=atol)


@pytest.mark.parametrize('meridian', ['vertical', 'horizontal'])
def test_blend_meridian_tensor_gradcheck(meridian):
    grid = Grid2D((-1.1, 0.9), (-0.8, 1.2), step=0.2)
    grid.build(Curcio1990Map())
    resp = np.random.default_rng(2).normal(0, 1, (grid.x.size, 2))
    resp = torch.tensor(resp, dtype=torch.float64, requires_grad=True)
    # Radius 20 > 11 samples, exercises a kernel wider than the axis:
    assert torch.autograd.gradcheck(
        lambda r: _blend_meridian(r, grid, meridian, 1.0), (resp,))


@pytest.mark.parametrize('reduce', ['last', 'peak'])
@pytest.mark.parametrize('t_percept', [None, [0.3, 0.305, 1.0, 2.0, 50.0]])
def test_FadingTemporal_tensor_parity(reduce, t_percept):
    temporal = FadingTemporal(tau=2, reduce=reduce, thresh_percept=0.1)
    wf = _waveform(8)
    expected = temporal.predict_percept(Stimulus(wf, time=TIME),
                                        t_percept=t_percept)
    resp = temporal._predict_response(
        _ModelResponse(torch.tensor(wf, dtype=torch.float32), TIME, ms,
                       (wf.shape[0],)), t_percept=t_percept)
    assert isinstance(resp.data, torch.Tensor)
    npt.assert_allclose(resp.time, expected.time)
    assert np.any(expected.data > 0)
    assert torch.all(resp.data[::4] == 0)
    npt.assert_allclose(resp.data.numpy(),
                        expected.data.reshape(resp.data.shape),
                        rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('reduce', ['last', 'peak'])
@pytest.mark.parametrize('t_percept', [None, [0.5, 1.0, 2.0, 25.0, 60.0]])
def test_Model_tensor_parity(reduce, t_percept):
    model = _model(reduce=reduce)
    wf = _waveform(model.implant.n_electrodes)
    expected = model.predict_percept(
        Stimulus(wf, electrodes=model.implant.electrode_names, time=TIME),
        t_percept=t_percept)
    resp = model._predict_tensor(torch.tensor(wf, dtype=torch.float32), TIME,
                                 t_percept=t_percept)
    assert resp.data.shape == (model.spatial.grid.x.size, expected.time.size)
    npt.assert_allclose(resp.time, expected.time)
    npt.assert_allclose(resp.data.numpy(),
                        expected.data.reshape(resp.data.shape),
                        rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('reduce', ['last', 'peak'])
def test_Model_tensor_autograd(reduce):
    model = _model(reduce=reduce)
    waveform = torch.tensor(_waveform(model.implant.n_electrodes),
                            dtype=torch.float32, requires_grad=True)
    resp = model._predict_tensor(waveform, TIME,
                                 t_percept=[0.5, 1.0, 2.0, 25.0, 60.0])
    # A NumPy round trip would drop the graph:
    assert isinstance(resp.data, torch.Tensor)
    assert resp.data.requires_grad and resp.data.grad_fn is not None
    resp.data.square().mean().backward()
    assert waveform.grad is not None
    assert torch.all(torch.isfinite(waveform.grad))
    assert waveform.grad.abs().sum() > 0
    # Silent electrodes still receive gradient through the Gaussian spread:
    assert waveform.grad[::4].abs().sum() > 0


def test_Model_tensor_gradcheck():
    # Exact autograd gradient of the piecewise-smooth forward model:
    model = Model(_spatial(xrange=(-3, 3), yrange=(-2, 2), step=1,
                           thresh_percept=0),
                  FadingTemporal(tau=0.5, reduce='peak'))
    waveform = torch.tensor(_waveform(model.implant.n_electrodes),
                            dtype=torch.float64, requires_grad=True)
    torch.autograd.gradcheck(
        lambda w: model._predict_tensor(w, TIME, t_percept=[1.0, 25.0]).data,
        (waveform,))


def test_Model_tensor_float64():
    model = _model()
    wf = _waveform(model.implant.n_electrodes)
    resp = model._predict_tensor(torch.tensor(wf), TIME)
    assert resp.data.dtype == torch.float64
    expected = model.predict_percept(
        Stimulus(wf, electrodes=model.implant.electrode_names, time=TIME))
    npt.assert_allclose(resp.data.numpy(),
                        expected.data.reshape(resp.data.shape),
                        rtol=RTOL, atol=1e-4)


def test_Model_tensor_errors():
    model = _model()
    n_el = model.implant.n_electrodes
    waveform = torch.zeros((n_el, TIME.size))
    with pytest.raises(TypeError, match='torch.Tensor'):
        model._predict_tensor(waveform.numpy(), TIME)
    for dtype in (torch.int32, torch.float16, torch.bfloat16):
        with pytest.raises(TypeError, match='float32 or float64'):
            model._predict_tensor(waveform.to(dtype), TIME)
    with pytest.raises(ValueError, match='shape'):
        model._predict_tensor(waveform[0], TIME)
    with pytest.raises(ValueError, match='shape'):
        model._predict_tensor(waveform[1:], TIME)
    with pytest.raises(ValueError, match="'time' must have shape"):
        model._predict_tensor(waveform, TIME[1:])
    with pytest.raises(ValueError, match='strictly increasing'):
        model._predict_tensor(waveform, TIME[::-1])
    for bad in (np.nan, np.inf):
        time = TIME.copy()
        time[5] = bad
        with pytest.raises(ValueError, match='finite'):
            model._predict_tensor(waveform, time)
    model.spatial.n_gray = 8
    with pytest.raises(NotImplementedError, match='n_gray'):
        model._predict_tensor(waveform, TIME)


@pytest.mark.parametrize('model', [
    Model(Thompson2003Spatial(ArgusI()), FadingTemporal()),
    Model(_spatial(), AlphaTemporal()),
    Model(_spatial()),
    Model(temporal=FadingTemporal()),
])
def test_Model_tensor_unsupported(model):
    waveform = torch.zeros((16, TIME.size))
    with pytest.raises(NotImplementedError):
        model._predict_tensor(waveform, TIME)


def test_Model_tensor_requires_electrical_implant():
    # A photovoltaic implant is driven by irradiance, not cathodic current:
    implant = PRIMAPivotal()
    model = Model(ScoreboardSpatial(implant, rho=50), FadingTemporal())
    waveform = torch.zeros((implant.n_electrodes, TIME.size))
    with pytest.raises(NotImplementedError, match='electrical current'):
        model._predict_tensor(waveform, TIME)


def _image_model(implant=None, reduce='peak', **params):
    """Scoreboard + Fading model whose implant encodes images."""
    implant = ArgusII() if implant is None else implant
    names = implant.electrode_names
    implant.deactivate([names[0], names[-1]])
    implant.encoder = AmplitudeEncoder(
        amp_range=(10, 50), freq=60, phase_dur=0.3, interphase_dur=0.1,
        cathodic_first=False, clock=0.1, frame_dur=100)
    params = {'xrange': (-6, 6), 'yrange': (-5, 5), 'step': 0.5,
              'thresh_percept': 0, **params}
    return Model(ScoreboardSpatial(implant, **params),
                 FadingTemporal(tau=2, reduce=reduce)).build()


# An encoded image has no frame clock in `_predict_tensor`, so pass t_percept:
IMAGE_T = [5.0, 20.0, 50.0, 99.0]


@pytest.mark.parametrize('reduce', ['last', 'peak'])
def test_Model_tensor_image_parity(reduce):
    model = _image_model(reduce=reduce)
    img = np.random.default_rng(7).uniform(-0.2, 1.2, (13, 17))
    expected = model.predict_percept(ImageStimulus(img), t_percept=IMAGE_T)
    waveform, time = model.implant.encoder._encode_tensor(
        torch.tensor(img, dtype=torch.float32))
    resp = model._predict_tensor(waveform, time, t_percept=IMAGE_T)
    npt.assert_allclose(resp.time, expected.time)
    expected = expected.data.reshape(resp.data.shape)
    assert np.abs(expected).max() > 1
    npt.assert_allclose(resp.data.numpy(), expected, rtol=RTOL, atol=ATOL)


def test_Model_tensor_image_autograd():
    model = _image_model()
    img = np.random.default_rng(7).uniform(0, 1, (13, 17))
    image = torch.tensor(img, dtype=torch.float32, requires_grad=True)
    waveform, time = model.implant.encoder._encode_tensor(image)
    resp = model._predict_tensor(waveform, time, t_percept=IMAGE_T)
    # A NumPy round trip would drop the graph:
    assert waveform.requires_grad and resp.data.requires_grad
    resp.data.square().mean().backward()
    assert image.grad is not None
    assert torch.all(torch.isfinite(image.grad))
    assert image.grad.abs().sum() > 0


def test_Model_tensor_image_gradcheck():
    # Exact gradient of image -> percept; gray levels stay inside (0, 1) so
    # clipping is smooth, and amp_lo > 0 keeps every pulse in the schedule:
    model = _image_model(ArgusI(), reduce='last', step=1)
    img = np.random.default_rng(8).uniform(0.1, 0.9, (3, 4))

    def percept(image):
        waveform, time = model.implant.encoder._encode_tensor(image)
        return model._predict_tensor(waveform, time, t_percept=[20.0]).data

    torch.autograd.gradcheck(percept, (torch.tensor(img, requires_grad=True),))


def _axon_spatial(implant=None, **params):
    # Grid straddles the horizontal meridian, so `meridian_blend` applies:
    params = {'xrange': (-6, 6), 'yrange': (-4, 4), 'step': 0.5,
              'n_axons': 200, 'n_ax_segments': 200, 'thresh_percept': 0.5,
              'ignore_pickle': True, **params}
    return AxonMapSpatial(ArgusII() if implant is None else implant, **params)


def _axon_parity(spatial, wf):
    """Return the Cython response after checking tensor parity."""
    expected = spatial.predict_percept(
        Stimulus(wf, electrodes=spatial.implant.electrode_names, time=TIME))
    resp = spatial._predict_tensor(torch.tensor(wf, dtype=torch.float32),
                                   TIME)
    expected = expected.data.reshape(resp.data.shape)
    assert resp.shape == spatial.grid.x.shape
    # float32 rounding grows with the summed magnitude:
    npt.assert_allclose(resp.data.numpy(), expected, rtol=RTOL,
                        atol=1e-6 * np.abs(expected).max())
    return expected


@pytest.mark.parametrize('params', [
    {},
    {'meridian_blend': 0},
    {'min_current_spread': 0.05, 'thresh_percept': 5},
    {'implant_position': (300, -200), 'implant_rotation': 20},
    {'location_noise': 0.5},
])
def test_AxonMapSpatial_tensor_parity(params):
    spatial = _axon_spatial(**params).build()
    expected = _axon_parity(spatial, _waveform(spatial.implant.n_electrodes))
    # Thresholding zeros some, but not all, of a signed response:
    assert 0 < np.mean(expected == 0) < 1
    assert expected.min() < 0 < expected.max()


def test_axon_blocks():
    # 4 electrodes, 10 time points, float64: 208 bytes per segment plus 80
    # per padded slot, so pixel 4 (40 segments) exceeds the budget alone:
    counts = np.array([3, 0, 5, 1, 40, 2, 0, 2])
    blocks = beyeler2019._axon_blocks(counts, 4, 10, 8, 3000)
    assert blocks == [(0, 3), (3, 4), (4, 5), (5, 8)]
    assert beyeler2019._axon_blocks(counts, 4, 10, 8, 10 ** 9) == [(0, 8)]
    # Longer waveforms give smaller blocks:
    assert len(beyeler2019._axon_blocks(counts, 4, 1000, 8, 3000)) == 8


def test_AxonMapSpatial_tensor_blocks(monkeypatch):
    # Mix of multi-pixel blocks and single axons above budget:
    monkeypatch.setattr(beyeler2019, '_AXON_BLOCK_BYTES', 20000)
    spatial = _axon_spatial().build()
    blocks = beyeler2019._axon_blocks(
        spatial.axon_idx_end - spatial.axon_idx_start, 60, TIME.size, 4,
        20000)
    sizes = {p1 - p0 for p0, p1 in blocks}
    assert 1 in sizes and max(sizes) > 1
    _axon_parity(spatial, _waveform(spatial.implant.n_electrodes))


def test_AxonMapSpatial_tensor_cathodic():
    # Selects the segment with largest |response|, not the largest value:
    spatial = _axon_spatial(meridian_blend=0).build()
    expected = _axon_parity(spatial,
                            -np.abs(_waveform(spatial.implant.n_electrodes)))
    assert expected.max() == 0 and expected.min() < 0


def test_AxonMapSpatial_tensor_thresh_inclusive():
    spatial = _axon_spatial(meridian_blend=0, thresh_percept=0).build()
    waveform = torch.tensor(_waveform(spatial.implant.n_electrodes),
                            dtype=torch.float32)
    resp = spatial._predict_tensor(waveform, TIME).data
    thresh = resp.abs().max() / 2
    spatial.thresh_percept = float(resp.abs()[resp.abs() >= thresh].min())
    thresholded = spatial._predict_tensor(waveform, TIME).data
    keep = resp.abs() >= spatial.thresh_percept
    npt.assert_equal(thresholded.numpy(), torch.where(keep, resp, 0).numpy())


def test_AxonMapSpatial_tensor_meridian_blend():
    wf = _waveform(60)
    unblended = _axon_parity(_axon_spatial(meridian_blend=0,
                                           thresh_percept=5).build(), wf)
    blended = _axon_parity(_axon_spatial(meridian_blend=2,
                                         thresh_percept=5).build(), wf)
    assert not np.allclose(blended, unblended)
    # Threshold reapplied after blending:
    assert np.all((blended == 0) | (np.abs(blended) >= 5))


def test_AxonMapModel_tensor_autograd():
    model = Model(_axon_spatial(), FadingTemporal(tau=2, reduce='peak'))
    waveform = torch.tensor(_waveform(model.implant.n_electrodes),
                            dtype=torch.float32, requires_grad=True)
    resp = model._predict_tensor(waveform, TIME,
                                 t_percept=[0.5, 1.0, 2.0, 25.0, 60.0])
    resp.data.square().mean().backward()
    assert torch.all(torch.isfinite(waveform.grad))
    assert waveform.grad.abs().sum() > 0
    # Silent electrodes still receive gradient through the Gaussian spread:
    assert waveform.grad[::4].abs().sum() > 0


def test_AxonMapModel_tensor_gradcheck(monkeypatch):
    # Several blocks, so gradients pass each block's index_put and gather:
    monkeypatch.setattr(beyeler2019, '_AXON_BLOCK_BYTES', 50000)
    model = Model(_axon_spatial(ArgusI(), xrange=(-3, 3), yrange=(-2, 2),
                                step=1, thresh_percept=0),
                  FadingTemporal(tau=0.5, reduce='peak'))
    # No silent samples: an all-zero response ties every segment:
    wf = np.random.default_rng(7).normal(0, 30, (16, TIME.size))
    waveform = torch.tensor(wf, dtype=torch.float64, requires_grad=True)
    torch.autograd.gradcheck(
        lambda w: model._predict_tensor(w, TIME, t_percept=[1.0, 25.0]).data,
        (waveform,))


def test_AxonMapModel_tensor_float64():
    model = Model(_axon_spatial(), FadingTemporal(tau=2, reduce='peak'))
    wf = _waveform(model.implant.n_electrodes)
    resp = model._predict_tensor(torch.tensor(wf), TIME)
    assert resp.data.dtype == torch.float64
    expected = model.predict_percept(
        Stimulus(wf, electrodes=model.implant.electrode_names, time=TIME))
    npt.assert_allclose(resp.data.numpy(),
                        expected.data.reshape(resp.data.shape),
                        rtol=RTOL, atol=1e-4)
    model.spatial.n_gray = 8
    with pytest.raises(NotImplementedError, match='n_gray'):
        model._predict_tensor(torch.tensor(wf), TIME)
