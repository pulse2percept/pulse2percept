"""Torch execution of ScoreboardSpatial -> FadingTemporal."""
import numpy as np
import numpy.testing as npt
import pytest
import torch

from pulse2percept.implants.retina import ArgusI
from pulse2percept.models import AlphaTemporal, FadingTemporal, Model
from pulse2percept.models.base import _ModelResponse
from pulse2percept.models.retina import ScoreboardSpatial, Thompson2003Spatial
from pulse2percept.stimuli import Stimulus
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
    with pytest.raises(TypeError, match='floating-point'):
        model._predict_tensor(waveform.int(), TIME)
    with pytest.raises(ValueError, match='shape'):
        model._predict_tensor(waveform[0], TIME)
    with pytest.raises(ValueError, match='shape'):
        model._predict_tensor(waveform[1:], TIME)
    with pytest.raises(ValueError, match="'time' must have shape"):
        model._predict_tensor(waveform, TIME[1:])
    with pytest.raises(ValueError, match='strictly increasing'):
        model._predict_tensor(waveform, TIME[::-1])
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
