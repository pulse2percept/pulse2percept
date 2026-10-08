"""Torch execution of {Scoreboard,AxonMap,Nanduri2012,Thompson2003}Spatial ->
{Fading,Alpha,Nanduri2012,Horsager2009}Temporal."""
from dataclasses import replace

import numpy as np
import numpy.testing as npt
import pytest
import torch

from pulse2percept.implants import GridImplant, Implant
from pulse2percept.implants.cortex import Orion
from pulse2percept.implants.retina import ArgusI, ArgusII, PRIMAPivotal
from pulse2percept.models import AlphaTemporal, FadingTemporal, Model
from pulse2percept.models import cortex
from pulse2percept.models.base import (_blend_meridian, _delivered,
                                       _encoder_clock, _ModelResponse,
                                       _scoreboard_response, _to_percept)
from pulse2percept.models.retina import (AxonMapSpatial, Horsager2009Temporal,
                                         Nanduri2012Model, Nanduri2012Spatial,
                                         Nanduri2012Temporal,
                                         ScoreboardSpatial,
                                         Thompson2003Spatial)
from pulse2percept.models.retina import beyeler2019
from pulse2percept.percepts import Percept
from pulse2percept.stimuli import (AmplitudeEncoder, BiphasicPulseTrain,
                                   ImageStimulus, PulseEncoder, Stimulus,
                                   VideoStimulus)
from pulse2percept.stimuli import encoders
from pulse2percept.topography import Grid2D
from pulse2percept.topography.retina import Curcio1990Map
from pulse2percept.units import mA, ms, xTh

# float32 tolerances: accumulation order differs from the Cython loops. ATOL
# covers cancellation in mixed-polarity sums of ~30 uA terms.
RTOL, ATOL = 1e-6, 1e-5


def _staged_percept(model, source, t_percept=None):
    """Return the percept of the staged route: NumPy encoding, the spatial
    response, then the temporal model's NumPy route.

    Reference for the Torch composite, which ``Model.predict_percept`` uses.
    """
    with pytest.MonkeyPatch.context() as mp:
        # The NumPy encoder that `AmplitudeEncoder.encode` replaces for gray
        # images and videos:
        mp.setattr(AmplitudeEncoder, 'encode', PulseEncoder.encode)
        stim = model.implant._prepare_stim(source)
    resp = model.spatial._predict_response(_delivered(stim))
    resp = replace(resp, frame_clock=_encoder_clock(stim))
    return _to_percept(model.temporal._predict_response(resp,
                                                        t_percept=t_percept))

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


def _model(reduce='peak', temporal=FadingTemporal, **params):
    return Model(_spatial(**params), temporal(tau=2, reduce=reduce))


def _scoreboard_reference(spatial, wf):
    """Return the flat Scoreboard response from its definition, in float64.

    Each region sums ``exp(-r**2 / (2 rho**2))``-weighted amplitudes over the
    electrodes within the cutoff (and, on a split map, in the same
    hemisphere), then is thresholded. Regions are summed, then blended across
    the meridian.
    """
    el = spatial._electrode_coords(spatial.implant.electrode_array, None,
                                   electrodes=spatial.implant.electrode_names)
    el = [np.asarray(c, dtype=float) for c in el]
    vfm = spatial.visual_field_map
    if isinstance(spatial, cortex.ScoreboardSpatial):
        grids = [spatial.grid[region] for region in spatial.regions]
    else:
        grids = [spatial.grid.ret]
    resp = 0
    for grid in grids:
        coords = [np.ravel(getattr(grid, axis)).astype(float)[:, None]
                  for axis in 'xyz'[:vfm.ndim]]
        r2 = sum((g - e) ** 2 for g, e in zip(coords, el))
        with np.errstate(invalid='ignore'):
            # NaN (unmapped) grid points fail the comparison:
            w = np.where(r2 <= spatial._cutoff_r2(spatial.rho),
                         np.exp(-r2 / (2 * spatial.rho ** 2)), 0)
        if getattr(vfm, 'split_map', False):
            boundary = vfm.left_offset / 2
            w[(coords[0] < boundary) != (el[0] < boundary)] = 0
        region = w @ wf
        resp = resp + np.where(np.abs(region) >= spatial.thresh_percept,
                               region, 0)
    return spatial._postprocess_spatial(resp)


def _assert_peak_close(actual, expected):
    """Assert parity; float32 rounding grows with the summed magnitude, so
    bound the error relative to the peak response."""
    npt.assert_allclose(actual, expected, rtol=RTOL,
                        atol=1e-6 * np.abs(expected).max())


def _assert_matches_reference(spatial, wf):
    """Assert that both prediction paths match ``_scoreboard_reference``."""
    expected = _scoreboard_reference(spatial, wf)
    # Every column differs, so compression keeps all time points:
    percept = spatial.predict_percept(
        Stimulus(wf, electrodes=spatial.implant.electrode_names, time=TIME))
    resp = spatial._predict_tensor(torch.tensor(wf, dtype=torch.float32),
                                   TIME)
    assert isinstance(resp.data, torch.Tensor)
    assert resp.data.shape == (spatial.grid.x.size, TIME.size)
    assert resp.shape == spatial.grid.x.shape
    assert resp.space is spatial.grid
    assert resp.frame_clock is None
    npt.assert_allclose(resp.time, TIME)
    # The threshold must zero some, but not all, of the response:
    assert 0 < np.mean(expected == 0) < 1
    for actual in (resp.data.numpy(), percept.data.reshape(expected.shape)):
        npt.assert_array_equal(actual == 0, expected == 0)
        _assert_peak_close(actual, expected)
    return expected


@pytest.mark.parametrize('params', [
    {},
    {'thresh_percept': 2},
    {'implant_position': (300, -200), 'implant_rotation': 20},
    {'location_noise': 0.5},
])
def test_ScoreboardSpatial_matches_reference(params):
    _assert_matches_reference(_spatial(**params).build(),
                              _waveform(ArgusI().n_electrodes))


def test_ScoreboardSpatial_matches_reference_large_grid():
    # ~100 of 1600 electrodes contribute to each grid point:
    spatial = ScoreboardSpatial(GridImplant((40, 40), 70), rho=65,
                                xrange=(-5, 5), yrange=(-5, 5), step=0.25,
                                thresh_percept=5).build()
    _assert_matches_reference(spatial,
                              _waveform(spatial.implant.n_electrodes))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_scoreboard_response_cutoff_boundary(dtype):
    # Electrodes at r2 = 0, just inside, exactly at, just beyond, and far
    # beyond the cutoff (100 um^2), where `exp` would see -1.25e5 unclamped.
    # The second grid point is unmapped (NaN):
    x_el = np.array([0, np.nextafter(np.float32(10), 0), 10,
                     np.nextafter(np.float32(10), 11), 1000],
                    dtype=np.float32)
    grid = (np.array([0, np.nan], dtype=np.float32),
            np.zeros(2, dtype=np.float32))
    # The identity waveform returns the weights themselves:
    weights = _scoreboard_response(torch.eye(x_el.size, dtype=dtype), grid,
                                   (x_el, np.zeros_like(x_el)), rho=2,
                                   cutoff_r2=100, thresh=0)
    assert weights.dtype == dtype
    r2 = torch.tensor(x_el) ** 2
    assert r2[1] < 100 and r2[2] == 100 and r2[3] > 100
    # Kept weights are the unclamped float32 Gaussian; the rest are zero:
    expected = torch.where(r2 <= 100, torch.exp(-r2 / 8), 0.0).to(dtype)
    assert torch.equal(weights[0], expected)
    assert torch.all(weights[0, :3] > 0)
    assert torch.equal(weights[1], torch.zeros(x_el.size, dtype=dtype))


def test_ScoreboardSpatial_prunes_silent_electrodes_only_in_predict_percept(
        monkeypatch):
    spatial = _spatial(rho=200).build()
    n_el = spatial.implant.n_electrodes
    wf = _waveform(n_el)
    seen = []
    core = ScoreboardSpatial._predict_scoreboard_tensor

    def spy(self, waveform, *coords):
        seen.append(waveform.shape[0])
        return core(self, waveform, *coords)

    monkeypatch.setattr(ScoreboardSpatial, '_predict_scoreboard_tensor', spy)
    names = spatial.implant.electrode_names
    # `predict_percept` skips electrodes that are zero throughout:
    percept = spatial.predict_percept(Stimulus(wf, electrodes=names,
                                               time=TIME))
    assert seen == [n_el - wf[::4].shape[0]]
    assert np.any(percept.data != 0)
    silent = spatial._predict_spatial(
        spatial.implant.electrode_array,
        Stimulus(np.zeros_like(wf), electrodes=names, time=TIME))
    assert seen[-1] == 0
    assert silent.shape == (spatial.grid.x.size, TIME.size)
    assert np.all(silent == 0)
    # `_predict_tensor` keeps them, so they receive gradient:
    waveform = torch.tensor(wf, dtype=torch.float32, requires_grad=True)
    spatial._predict_tensor(waveform, TIME).data.square().sum().backward()
    assert seen[-1] == n_el
    assert waveform.grad[::4].abs().sum() > 0


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


@pytest.mark.parametrize('temporal', [FadingTemporal, AlphaTemporal])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('reduce', ['last', 'peak'])
@pytest.mark.parametrize('t_percept', [None, [0.3, 0.305, 1.0, 2.0, 50.0]])
def test_generic_temporal_tensor_parity(temporal, dtype, reduce, t_percept):
    temporal = temporal(tau=2, reduce=reduce, thresh_percept=0.1)
    wf = _waveform(8)
    expected = temporal.predict_percept(Stimulus(wf, time=TIME),
                                        t_percept=t_percept)
    resp = temporal._predict_response(
        _ModelResponse(torch.tensor(wf, dtype=dtype), TIME, ms,
                       (wf.shape[0],)), t_percept=t_percept)
    assert isinstance(resp.data, torch.Tensor)
    assert resp.data.dtype == dtype
    assert expected.data.dtype == np.float32
    npt.assert_allclose(resp.time, expected.time)
    assert np.any(expected.data > 0)
    assert torch.all(resp.data[::4] == 0)
    # float64 differs from the float32 public route by float32 rounding:
    npt.assert_allclose(resp.data.numpy(),
                        expected.data.reshape(resp.data.shape),
                        rtol=RTOL if dtype == torch.float32 else 1e-5,
                        atol=ATOL)


def _temporal_gradcheck(temporal, data, time, t_percept, reduce):
    """Gradcheck the temporal response to float64 ``data`` at fixed timing."""
    temporal.build()
    t_percept = np.asarray(t_percept, dtype=float)

    def predict(d):
        resp = _ModelResponse(d, np.asarray(time, dtype=float), ms,
                              (d.shape[0],))
        return temporal._predict_temporal_tensor(resp, t_percept, reduce)

    assert torch.autograd.gradcheck(
        predict, (torch.tensor(data, dtype=torch.float64,
                               requires_grad=True),))


@pytest.mark.parametrize('temporal', [FadingTemporal, AlphaTemporal])
def test_generic_temporal_gradcheck_last(temporal):
    # Cathodic throughout, so rectification is smooth:
    data = -np.random.default_rng(4).uniform(1, 5, (3, 6))
    _temporal_gradcheck(temporal(tau=1, dt=0.05), data,
                        [0, 0.5, 1.1, 2, 3.3, 4], [0.5, 1.5, 2.5, 5.0],
                        'last')


@pytest.mark.parametrize('temporal, data, time, t_percept', [
    # Alternating cathodic and anodic frames; each interval's peak is a
    # unique run end:
    (FadingTemporal(tau=1, dt=0.05), [[-3, 2, -5, 2, -4, 2],
                                      [-1, 1, -2, 3, -6, 1]],
     [0, 1, 2, 3, 4, 5], [2.5, 5.5]),
    # Stage 2 peaks inside the (2, 20] ms interval:
    (AlphaTemporal(tau=8, dt=0.05), [[-60, 2], [-40, 3]], [0, 2],
     [2, 20, 60]),
])
def test_generic_temporal_gradcheck_peak(temporal, data, time, t_percept):
    _temporal_gradcheck(temporal, data, time, t_percept, 'peak')


@pytest.mark.parametrize('temporal', [FadingTemporal, AlphaTemporal])
@pytest.mark.parametrize('reduce', ['last', 'peak'])
@pytest.mark.parametrize('t_percept', [None, [0.5, 1.0, 2.0, 25.0, 60.0]])
def test_Model_tensor_parity(reduce, t_percept, temporal):
    model = _model(reduce=reduce, temporal=temporal)
    wf = _waveform(model.implant.n_electrodes)
    expected = _staged_percept(
        model,
        Stimulus(wf, electrodes=model.implant.electrode_names, time=TIME),
        t_percept=t_percept)
    resp = model._predict_tensor(torch.tensor(wf, dtype=torch.float32), TIME,
                                 t_percept=t_percept)
    assert resp.data.shape == (model.spatial.grid.x.size, expected.time.size)
    npt.assert_allclose(resp.time, expected.time)
    npt.assert_allclose(resp.data.numpy(),
                        expected.data.reshape(resp.data.shape),
                        rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('temporal', [FadingTemporal, AlphaTemporal])
@pytest.mark.parametrize('reduce', ['last', 'peak'])
def test_Model_tensor_autograd(reduce, temporal):
    # rho=200 puts bright points within the cutoff (about 5.3 rho) of the
    # silent electrodes, 800 um from their neighbors:
    model = _model(reduce=reduce, temporal=temporal, rho=200)
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
    expected = _staged_percept(
        model,
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
    Model(Thompson2003Spatial(ArgusI(), dropout=2), FadingTemporal()),
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


def _encoding_implant(implant=None, amp_range=(10, 50)):
    """Return an image-encoding implant with two deactivated electrodes."""
    implant = ArgusII() if implant is None else implant
    names = implant.electrode_names
    implant.deactivate([names[0], names[-1]])
    implant.encoder = AmplitudeEncoder(
        amp_range=amp_range, freq=60, phase_dur=0.3, interphase_dur=0.1,
        cathodic_first=False, clock=0.1, frame_dur=100)
    return implant


def _image_model(implant=None, reduce='peak', amp_range=(10, 50),
                 temporal=FadingTemporal, **params):
    """Scoreboard + generic temporal model whose implant encodes images."""
    implant = _encoding_implant(implant, amp_range)
    params = {'xrange': (-6, 6), 'yrange': (-5, 5), 'step': 0.5,
              'thresh_percept': 0, **params}
    return Model(ScoreboardSpatial(implant, **params),
                 temporal(tau=2, reduce=reduce)).build()


# Output times within the 100 ms image frame:
IMAGE_T = [5.0, 20.0, 50.0, 99.0]


@pytest.mark.parametrize('reduce', ['last', 'peak'])
def test_Model_tensor_image_parity(reduce):
    model = _image_model(reduce=reduce)
    img = np.random.default_rng(7).uniform(-0.2, 1.2, (13, 17))
    expected = _staged_percept(model, ImageStimulus(img), t_percept=IMAGE_T)
    resp = model._predict_visual_tensor(
        torch.tensor(img, dtype=torch.float32), t_percept=IMAGE_T)
    npt.assert_allclose(resp.time, expected.time)
    expected = expected.data.reshape(resp.data.shape)
    assert np.abs(expected).max() > 1
    npt.assert_allclose(resp.data.numpy(), expected, rtol=RTOL, atol=ATOL)


def _assert_pixel_grad(source, resp):
    """Backpropagate a squared response loss and check the pixel gradient."""
    # A NumPy round trip would drop the graph:
    assert isinstance(resp.data, torch.Tensor)
    assert resp.data.requires_grad and resp.data.grad_fn is not None
    resp.data.square().mean().backward()
    assert source.grad is not None
    assert torch.all(torch.isfinite(source.grad))
    return source.grad


def test_Model_tensor_image_autograd():
    model = _image_model()
    img = np.random.default_rng(7).uniform(0, 1, (13, 17))
    image = torch.tensor(img, dtype=torch.float32, requires_grad=True)
    resp = model._predict_visual_tensor(image, t_percept=IMAGE_T)
    assert _assert_pixel_grad(image, resp).abs().sum() > 0


def test_Model_tensor_image_black_autograd():
    # With amp_range[0] == 0, black keeps the pulse schedule but has zero
    # amplitude. The squared response loss still has zero gradient at zero.
    model = _image_model(amp_range=(0, 50))
    image = torch.zeros((13, 17), requires_grad=True)
    resp = model._predict_visual_tensor(image, t_percept=IMAGE_T)
    assert torch.all(resp.data == 0)
    assert torch.all(_assert_pixel_grad(image, resp) == 0)


def test_Model_tensor_image_gradcheck():
    # Exact gradient of image -> percept; gray levels stay inside (0, 1) so
    # clipping is smooth, and amp_lo > 0 keeps every pulse in the schedule:
    model = _image_model(ArgusI(), reduce='last', step=1)
    img = np.random.default_rng(8).uniform(0.1, 0.9, (3, 4))
    torch.autograd.gradcheck(
        lambda image: model._predict_visual_tensor(image,
                                                   t_percept=[20.0]).data,
        (torch.tensor(img, requires_grad=True),))


@pytest.mark.parametrize('reduce', ['last', 'peak'])
def test_Model_tensor_image_frame_clock(reduce):
    # With the encoder's frame clock, automatic output times match the
    # public image path:
    model = _image_model(reduce=reduce)
    img = np.random.default_rng(7).uniform(0, 1, (13, 17))
    expected = _staged_percept(model, ImageStimulus(img))
    resp = model._predict_visual_tensor(
        torch.tensor(img, dtype=torch.float32))
    npt.assert_equal(resp.time, expected.time)
    npt.assert_allclose(resp.data.numpy(),
                        expected.data.reshape(resp.data.shape),
                        rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('frame_dur', [None, 40])
@pytest.mark.parametrize('reduce', ['last', 'peak'])
def test_Model_tensor_video_parity(reduce, frame_dur):
    # Non-round 29.97 fps; explicit `frame_dur` retimes the output frames:
    model = _image_model(reduce=reduce)
    model.implant.encoder.frame_dur = frame_dur
    vid = np.random.default_rng(10).uniform(-0.2, 1.2, (13, 17, 6))
    expected = _staged_percept(
        model, VideoStimulus(vid, metadata={'fps': 29.97}))
    resp = model._predict_visual_tensor(
        torch.tensor(vid, dtype=torch.float32), fps=29.97)
    npt.assert_equal(resp.time, expected.time)
    assert resp.time.size == 6
    npt.assert_equal(resp.frame_clock.source_time,
                     expected._frame_clock.source_time)
    expected = expected.data.reshape(resp.data.shape)
    assert np.abs(expected).max() > 0.1
    npt.assert_allclose(resp.data.numpy(), expected, rtol=RTOL, atol=ATOL)


def test_Model_tensor_video_autograd():
    # Without a raster, every electrode pulses at 0 and 100 ms, in frames 0
    # and 3 of six 33.3 ms frames:
    model = _image_model(ArgusII(raster=None))
    encoder = model.implant.encoder
    encoder.freq, encoder.frame_dur = 10, None
    vid = np.random.default_rng(11).uniform(0, 1, (13, 17, 6))
    video = torch.tensor(vid, dtype=torch.float32, requires_grad=True)
    with pytest.warns(UserWarning, match='deliver no pulse'):
        resp = model._predict_visual_tensor(video, fps=30)
    driven = _assert_pixel_grad(video, resp).abs().sum(dim=(0, 1)) > 0
    npt.assert_equal(driven.numpy(), [True, False, False, True, False, False])


def test_Model_tensor_video_gradcheck():
    # Exact gradient of video -> percept through the frame indexing; gray
    # levels stay inside (0, 1) so clipping is smooth:
    model = _image_model(ArgusI(), reduce='last', step=1)
    model.implant.encoder.frame_dur = None
    vid = np.random.default_rng(12).uniform(0.1, 0.9, (3, 4, 2))
    torch.autograd.gradcheck(
        lambda video: model._predict_visual_tensor(video, fps=20).data,
        (torch.tensor(vid, requires_grad=True),))


def test_Model_tensor_visual_requires_amplitude_encoder():
    model = _image_model()
    model.implant.encoder = None
    with pytest.raises(NotImplementedError, match='AmplitudeEncoder'):
        model._predict_visual_tensor(torch.ones((13, 17)))
    with pytest.raises(NotImplementedError, match='AmplitudeEncoder'):
        Model(temporal=FadingTemporal())._predict_visual_tensor(
            torch.ones((13, 17)))


def _axon_spatial(implant=None, **params):
    # Grid straddles the horizontal meridian, so `meridian_blend` applies:
    params = {'xrange': (-6, 6), 'yrange': (-4, 4), 'step': 0.5,
              'n_axons': 200, 'n_ax_segments': 200, 'thresh_percept': 0.5,
              'ignore_pickle': True, **params}
    return AxonMapSpatial(ArgusII() if implant is None else implant, **params)


def _axon_paths(spatial, wf):
    """Return the ``predict_percept`` response after checking the tensor path.

    Both paths share one Torch kernel, so this checks API-path consistency
    (compressed stimulus vs. all electrodes), not the model itself.
    """
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
    {'thresh_percept': 5},
    {'implant_position': (300, -200), 'implant_rotation': 20},
    {'location_noise': 0.5},
])
def test_AxonMapSpatial_tensor_matches_predict_percept(params):
    spatial = _axon_spatial(**params).build()
    expected = _axon_paths(spatial, _waveform(spatial.implant.n_electrodes))
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
    _axon_paths(spatial, _waveform(spatial.implant.n_electrodes))


def test_AxonMapSpatial_tensor_cathodic():
    # Selects the segment with largest |response|, not the largest value:
    spatial = _axon_spatial(meridian_blend=0).build()
    expected = _axon_paths(spatial,
                            -np.abs(_waveform(spatial.implant.n_electrodes)))
    assert expected.max() == 0 and expected.min() < 0


def _one_axon(spatial, segments):
    """Give pixel 0 an axon of ``(x, y, sensitivity)`` segments; no others."""
    n_px, n_seg = spatial.grid.x.size, len(segments)
    spatial.axon_contrib = np.array(segments, dtype=np.float32)
    spatial.axon_idx_start = np.array([0] + [n_seg] * (n_px - 1))
    spatial.axon_idx_end = np.full(n_px, n_seg)


def _drive(spatial, name, amp):
    """Return a waveform driving only electrode ``name``."""
    wf = np.zeros((spatial.implant.n_electrodes, TIME.size))
    wf[list(spatial.implant.electrode_names).index(name)] = amp
    return wf


@pytest.mark.parametrize('sign', [1, -1])
def test_AxonMapSpatial_tensor_first_tie(sign):
    # Two segments of one axon at the electrode with opposite sensitivity:
    # exactly tied |response|, so the first segment's sign wins.
    spatial = _axon_spatial(meridian_blend=0, thresh_percept=0).build()
    x_el, y_el, _ = spatial._electrode_coords(
        spatial.implant.electrode_array, None, electrodes=['C3'])
    _one_axon(spatial, [[x_el[0], y_el[0], sign], [x_el[0], y_el[0], -sign]])
    amp = np.random.default_rng(5).normal(0, 30, TIME.size)
    expected = _axon_paths(spatial, _drive(spatial, 'C3', amp))
    npt.assert_equal(expected[0], sign * amp.astype(np.float32))
    assert np.all(expected[1:] == 0)


def test_AxonMapSpatial_tensor_first_tie_outside_cutoff():
    # The first segment is beyond the cutoff of every electrode. At samples
    # where the waveform is zero, both tie at 0 and the first wins, so no
    # gradient reaches the waveform there:
    spatial = _axon_spatial(meridian_blend=0, thresh_percept=0).build()
    x_el, y_el, _ = spatial._electrode_coords(
        spatial.implant.electrode_array, None, electrodes=['C3'])
    _one_axon(spatial, [[x_el[0] + 1e5, y_el[0], 1], [x_el[0], y_el[0], 1]])
    amp = np.random.default_rng(6).normal(0, 30, TIME.size)
    silent = [2, 7, 12]
    amp[silent] = 0
    wf = _drive(spatial, 'C3', amp)
    expected = _axon_paths(spatial, wf)
    npt.assert_equal(expected[0], amp.astype(np.float32))
    waveform = torch.tensor(wf, dtype=torch.float64, requires_grad=True)
    spatial._predict_tensor(waveform, TIME).data[0].sum().backward()
    row = list(spatial.implant.electrode_names).index('C3')
    npt.assert_equal(waveform.grad[row].numpy(),
                     np.where(np.isin(np.arange(TIME.size), silent), 0, 1))


@pytest.mark.parametrize('below', [False, True])
def test_AxonMapSpatial_tensor_cutoff_inclusive(below, monkeypatch):
    # One segment 100 um right of the rightmost electrode, level with it:
    # r2 == cutoff_r2 exactly, which is retained.
    cutoff_r2 = np.float32(100 * 100)
    if below:
        cutoff_r2 = np.nextafter(cutoff_r2, np.float32(0))
    monkeypatch.setattr(AxonMapSpatial, '_cutoff_r2',
                        lambda self, rho: cutoff_r2)
    spatial = _axon_spatial(meridian_blend=0, thresh_percept=0).build()
    names = spatial.implant.electrode_names
    x_el, y_el, _ = spatial._electrode_coords(
        spatial.implant.electrode_array, None, electrodes=names)
    edge = int(np.argmax(x_el))
    _one_axon(spatial, [[x_el[edge] + 100, y_el[edge], 1]])
    amp = np.random.default_rng(7).normal(0, 30, TIME.size)
    expected = _axon_paths(spatial, _drive(spatial, names[edge], amp))
    if below:
        assert np.all(expected == 0)
    else:
        gauss = np.exp(np.float32(-1e4) / (np.float32(2) *
                                           np.float32(spatial.rho) ** 2))
        npt.assert_allclose(expected[0], gauss * amp, rtol=1e-6)


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
    unblended = _axon_paths(_axon_spatial(meridian_blend=0,
                                           thresh_percept=5).build(), wf)
    blended = _axon_paths(_axon_spatial(meridian_blend=2,
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


def test_AxonMapModel_tensor_image_autograd():
    model = Model(_axon_spatial(_encoding_implant()),
                  FadingTemporal(tau=2, reduce='peak')).build()
    img = np.random.default_rng(7).uniform(0, 1, (13, 17))
    image = torch.tensor(img, dtype=torch.float32, requires_grad=True)
    resp = model._predict_visual_tensor(image, t_percept=IMAGE_T)
    assert resp.data.abs().max() > 0
    assert _assert_pixel_grad(image, resp).abs().sum() > 0


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
    expected = _staged_percept(
        model,
        Stimulus(wf, electrodes=model.implant.electrode_names, time=TIME))
    npt.assert_allclose(resp.data.numpy(),
                        expected.data.reshape(resp.data.shape),
                        rtol=RTOL, atol=1e-4)
    model.spatial.n_gray = 8
    with pytest.raises(NotImplementedError, match='n_gray'):
        model._predict_tensor(torch.tensor(wf), TIME)


# Public `Model.predict_percept` runs supported composites on the Torch core;
# `_staged_percept` is the reference:

COMPOSITES = ['retina', 'cortex', 'axonmap', 'thompson']


def _composite(kind, reduce='peak', temporal=FadingTemporal):
    """Return a built composite that predicts on the Torch core."""
    temporal = temporal(tau=2, reduce=reduce)
    if kind == 'retina':
        return Model(_spatial(), temporal).build()
    if kind == 'cortex':
        # Reaches V1-V3 near the vertical meridian:
        return Model(cortex.ScoreboardSpatial(
            Orion(), implant_position=(10000, 10000), xrange=(-4.1, 3.9),
            yrange=(-3, 3), step=0.2, rho=1000, thresh_percept=0.5),
            temporal).build()
    if kind == 'thompson':
        return Model(Thompson2003Spatial(
            ArgusI(), xrange=(-6, 6), yrange=(-5, 5), step=0.5,
            thresh_percept=0.5), temporal).build()
    return Model(_axon_spatial(), temporal).build()


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('kind', COMPOSITES)
def test_predict_rejects_nonfinite_stimulus(kind, bad):
    model = _composite(kind)
    wf = _waveform(model.implant.n_electrodes)
    wf[1, 4] = bad
    source = Stimulus(wf, electrodes=model.implant.electrode_names, time=TIME)
    # Torch composite, spatial-only model, and spatial stage:
    for predict in (model.predict_percept,
                    Model(model.spatial).predict_percept,
                    model.spatial.predict_percept):
        with pytest.raises(ValueError, match='must be finite'):
            predict(source)
    with pytest.raises(ValueError, match='must be finite'):
        model._predict_tensor(torch.tensor(wf, dtype=torch.float32), TIME)


def _assert_same_percept(percept, expected):
    # No tensor leaks through the public API:
    assert isinstance(percept, Percept)
    assert isinstance(percept.data, np.ndarray)
    assert isinstance(percept.metadata['stim'], Stimulus)
    assert percept.data.dtype == expected.data.dtype
    assert percept.data.shape == expected.data.shape
    assert np.abs(expected.data).max() > 0
    npt.assert_equal(percept.time, expected.time)
    assert percept.time_unit == expected.time_unit
    npt.assert_equal(percept.xdva, expected.xdva)
    npt.assert_equal(percept.ydva, expected.ydva)
    # float32 rounding grows with the summed magnitude:
    npt.assert_allclose(percept.data, expected.data, rtol=RTOL,
                        atol=max(ATOL, 1e-6 * np.abs(expected.data).max()))
    assert list(percept.metadata) == list(expected.metadata)
    npt.assert_equal(percept.metadata.get('source_frame_time'),
                     expected.metadata.get('source_frame_time'))
    stim, ref = percept.metadata['stim'], expected.metadata['stim']
    assert type(stim) is type(ref) and stim.unit == ref.unit
    # Torch and SciPy sampling round differently in float32:
    npt.assert_allclose(stim.data, ref.data, rtol=RTOL, atol=ATOL)
    npt.assert_equal(stim.electrodes, ref.electrodes)
    npt.assert_equal(stim.time, ref.time)


@pytest.mark.parametrize('temporal', [FadingTemporal, AlphaTemporal])
@pytest.mark.parametrize('kind', COMPOSITES)
@pytest.mark.parametrize('reduce', ['last', 'peak'])
@pytest.mark.parametrize('t_percept', [None, [0.5, 1.0, 2.0, 25.0, 60.0]])
def test_Model_predict_percept_torch_parity(kind, reduce, t_percept,
                                            temporal):
    model = _composite(kind, reduce=reduce, temporal=temporal)
    source = Stimulus(_waveform(model.implant.n_electrodes),
                      electrodes=model.implant.electrode_names, time=TIME)
    _assert_same_percept(model.predict_percept(source, t_percept=t_percept),
                         _staged_percept(model, source, t_percept=t_percept))


@pytest.mark.parametrize('kind', COMPOSITES)
def test_Model_predict_percept_torch_sparse(kind):
    # A few electrodes, out of implant order, one deactivated, in mA:
    model = _composite(kind)
    names = model.implant.electrode_names
    model.implant.deactivate(names[3])
    picked = [names[9], names[3], names[1]]
    amp = np.random.default_rng(3).normal(0, 0.03, (3, TIME.size))
    source = Stimulus(amp * mA, electrodes=picked, time=TIME)
    percept = model.predict_percept(source)
    _assert_same_percept(percept, _staged_percept(model, source))
    npt.assert_equal(list(percept.metadata['stim'].electrodes),
                     [names[9], names[1]])


def test_Model_predict_percept_torch_pulse_train():
    model = _composite('retina')
    source = {'C3': BiphasicPulseTrain(20, 30, 0.45, stim_dur=200),
              'A1': BiphasicPulseTrain(35, 20, 0.2, stim_dur=150)}
    _assert_same_percept(model.predict_percept(source),
                         _staged_percept(model, source))


def _black_left(shape, seed=7):
    """Return gray levels, black in the left half, so amp_range=(0, x)
    leaves some electrodes without pulses."""
    pixels = np.random.default_rng(seed).uniform(-0.2, 1.2, shape)
    pixels[:, :shape[1] // 2] = 0
    return pixels


@pytest.mark.parametrize('amp_range', [(10, 50), (0, 50)])
@pytest.mark.parametrize('reduce', ['last', 'peak'])
@pytest.mark.parametrize('t_percept', [None, IMAGE_T])
def test_Model_predict_percept_torch_image(reduce, t_percept, amp_range):
    model = _image_model(reduce=reduce, amp_range=amp_range)
    source = ImageStimulus(_black_left((13, 17)))
    percept = model.predict_percept(source, t_percept=t_percept)
    _assert_same_percept(percept,
                         _staged_percept(model, source, t_percept=t_percept))
    # Deactivated electrodes are absent from the prepared stimulus:
    names = model.implant.electrode_names
    delivered = percept.metadata['stim'].electrodes
    assert {names[0], names[-1]}.isdisjoint(delivered)


def test_Model_predict_percept_torch_image_axonmap():
    model = Model(_axon_spatial(_encoding_implant(amp_range=(0, 50))),
                  FadingTemporal(tau=2)).build()
    source = ImageStimulus(_black_left((13, 17)))
    _assert_same_percept(model.predict_percept(source),
                         _staged_percept(model, source))


@pytest.mark.parametrize('video', [False, True])
def test_Model_predict_percept_torch_visual_alpha(video, monkeypatch):
    model = _image_model(amp_range=(0, 50), temporal=AlphaTemporal)
    source = (VideoStimulus(_black_left((13, 17, 6)), metadata={'fps': 29.97})
              if video else ImageStimulus(_black_left((13, 17))))
    expected = _staged_percept(model, source)

    def staged(*args, **kwargs):
        raise AssertionError("Staged path called")

    monkeypatch.setattr(AlphaTemporal, '_predict_temporal', staged)
    _assert_same_percept(model.predict_percept(source), expected)


@pytest.mark.parametrize('frame_dur', [None, 40])
@pytest.mark.parametrize('reduce', ['last', 'peak'])
def test_Model_predict_percept_torch_video(reduce, frame_dur):
    # Output frames follow the 29.97 fps source clock unless `frame_dur`
    # retimes them:
    model = _image_model(reduce=reduce, amp_range=(0, 50))
    model.implant.encoder.frame_dur = frame_dur
    source = VideoStimulus(_black_left((13, 17, 6)), metadata={'fps': 29.97})
    percept = model.predict_percept(source)
    expected = _staged_percept(model, source)
    _assert_same_percept(percept, expected)
    assert percept.time.size == 6
    assert ('source_frame_time' in percept.metadata) == (frame_dur is None)
    for field in ('time', 'dur', 'source_time', 'source_dur'):
        npt.assert_equal(getattr(percept._frame_clock, field),
                         getattr(expected._frame_clock, field))


@pytest.mark.parametrize('video', [False, True])
def test_Model_predict_percept_visual_skips_numpy_encoder(video,
                                                          monkeypatch):
    model = _image_model(amp_range=(0, 50))
    model.implant.encoder.frame_dur = None
    shape = (13, 17, 6) if video else (13, 17)
    source = (VideoStimulus(_black_left(shape), metadata={'fps': 29.97})
              if video else ImageStimulus(_black_left(shape)))
    expected = _staged_percept(model, source)

    def numpy_route(*args, **kwargs):
        raise AssertionError("NumPy preparation or encoding called")

    for owner, name in ((Implant, '_prepare_stim'),
                        (Implant, 'reshape_stim'),
                        (PulseEncoder, 'encode'),
                        (encoders, '_sampled_frames'),
                        (Model, '_predict_tensor_core')):
        monkeypatch.setattr(owner, name, numpy_route)
    sampled, predicted = [], []
    sample, predict = Implant._sample_image_tensor, Model._predict_tensor

    def sample_spy(self, image):
        sampled.append(image.dtype)
        return sample(self, image)

    def predict_spy(self, *args, **kwargs):
        predicted.append(torch.is_inference_mode_enabled())
        return predict(self, *args, **kwargs)

    monkeypatch.setattr(Implant, '_sample_image_tensor', sample_spy)
    monkeypatch.setattr(Model, '_predict_tensor', predict_spy)
    percept = model.predict_percept(source)
    # Torch from electrode sampling onward, once, under inference mode:
    assert sampled == [torch.float32] and predicted == [True]
    _assert_same_percept(percept, expected)


class _CheckedArgusII(ArgusII):
    def check_stim(self, stim):
        return super().check_stim(stim)


class _SubclassedEncoder(AmplitudeEncoder):
    pass


def _unsupported(model, case):
    """Make ``case`` unsupported by Torch encoding; return the source."""
    implant, encoder = model.implant, model.implant.encoder
    pixels = np.random.default_rng(7).uniform(0, 1, (13, 17))
    timing = {'freq': 60, 'phase_dur': 0.3, 'clock': 0.1, 'frame_dur': 100}
    if case in ('rgb', 'rgba'):
        return ImageStimulus(np.random.default_rng(7).uniform(
            0, 1, (13, 17, len(case))))
    if case == 'n_levels':
        encoder.n_levels = 4
    elif case == 'stretch':
        encoder.stretch = True
    elif case == 'preprocess':
        implant.preprocess = lambda stim: stim
    elif case == 'safe_mode':
        implant.safe_mode = True
    elif case == 'max_current':
        implant.max_current = 1e5
    elif case == 'xTh':
        implant.encoder = AmplitudeEncoder(amp_range=(0.5 * xTh, 2 * xTh),
                                           **timing)
        implant.thresholds = {name: 20 for name in implant.electrode_names}
    elif case == 'encoder_subclass':
        implant.encoder = _SubclassedEncoder(amp_range=(10, 50), **timing)
    elif case == 'spatial':
        model.spatial = _DoubledScoreboard(implant, xrange=(-6, 6),
                                           yrange=(-5, 5), step=0.5,
                                           thresh_percept=0)
    elif case == 'temporal':
        model.temporal = _HalvedFading(tau=2)
    return ImageStimulus(pixels)


@pytest.mark.parametrize('case', [
    'rgb', 'rgba', 'n_levels', 'stretch', 'preprocess', 'check_stim',
    'safe_mode', 'max_current', 'xTh', 'encoder_subclass', 'spatial',
    'temporal'])
def test_Model_predict_percept_visual_fallback(case, monkeypatch):
    model = _image_model(_CheckedArgusII() if case == 'check_stim' else None)
    source = _unsupported(model, case)
    expected = _staged_percept(model, source)

    def visual(*args, **kwargs):
        raise AssertionError("Torch encoding route called")

    monkeypatch.setattr(Model, '_predict_visual_percept', visual)
    # The prepared route still runs, with the same result as before:
    _assert_same_percept(model.predict_percept(source), expected)


@pytest.mark.parametrize('temporal', [FadingTemporal, AlphaTemporal])
@pytest.mark.parametrize('kind', COMPOSITES)
def test_Model_predict_percept_skips_staged_path(kind, temporal, monkeypatch):
    model = _composite(kind, temporal=temporal)

    def staged(*args, **kwargs):
        raise AssertionError("Staged path called")

    monkeypatch.setattr(type(model.spatial), '_predict_spatial', staged)
    monkeypatch.setattr(temporal, '_predict_temporal', staged)
    source = Stimulus(_waveform(model.implant.n_electrodes),
                      electrodes=model.implant.electrode_names, time=TIME)
    assert np.abs(model.predict_percept(source).data).max() > 0


# Subclasses that inherit a tensor core keep the legacy path, whichever
# legacy hook they override:

class _DoubledScoreboard(ScoreboardSpatial):
    def _predict_spatial(self, electrode_array, stim):
        return 2 * super()._predict_spatial(electrode_array, stim)


class _DoubledResponse(ScoreboardSpatial):
    def _predict_response(self, stim, t_percept=None):
        resp = super()._predict_response(stim, t_percept=t_percept)
        return replace(resp, data=2 * resp.data)


class _HalvedFading(FadingTemporal):
    def _stim_values(self, stim):
        return 0.5 * super()._stim_values(stim)


def _subclassed(spatial=ScoreboardSpatial, temporal=FadingTemporal):
    return Model(spatial(ArgusI(), xrange=(-6, 6), yrange=(-5, 5), step=0.5),
                 temporal(tau=2))


@pytest.mark.parametrize('model, reference', [
    (lambda: Model(_spatial(n_gray=8), FadingTemporal(tau=2)),
     _staged_percept),
    (lambda: _subclassed(spatial=_DoubledScoreboard), _staged_percept),
    (lambda: _subclassed(spatial=_DoubledResponse), _staged_percept),
    (lambda: _subclassed(temporal=_HalvedFading), _staged_percept),
    (lambda: Model(_spatial()),
     lambda model, source: model.spatial.predict_percept(source)),
    (lambda: Model(temporal=FadingTemporal(tau=2)),
     lambda model, source: model.temporal.predict_percept(source)),
])
def test_Model_predict_percept_staged_fallback(model, reference, monkeypatch):
    model = model().build()

    def torch_core(*args, **kwargs):
        raise AssertionError("Torch core called")

    monkeypatch.setattr(Model, '_predict_tensor', torch_core)
    source = Stimulus(_waveform(16), electrodes=ArgusI().electrode_names,
                      time=TIME)
    # `n_gray` quantizes with randomly initialized k-means:
    np.random.seed(0)
    percept = model.predict_percept(source)
    np.random.seed(0)
    npt.assert_array_equal(percept.data, reference(model, source).data)


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
@pytest.mark.parametrize('reduce', ['last', 'peak'])
@pytest.mark.parametrize('t_percept', [None, [5.0, 40.0, 150.0]])
def test_TemporalModel_ignores_response_metadata(backend, reduce, t_percept):
    # Metadata is provenance only; timing comes from `frame_clock`:
    model = _image_model()
    vid = np.random.default_rng(13).uniform(0, 1, (13, 17, 6))
    waveform, time, clock = model.implant.encoder._encode_tensor(
        torch.tensor(vid, dtype=torch.float32), fps=29.97)
    data = waveform if backend == 'torch' else waveform.numpy()
    # Wildly different provenance, including keys that imply another clock:
    stim_a = model.implant.prepare_stim(VideoStimulus(
        np.random.default_rng(14).uniform(0, 1, (5, 5, 40)),
        metadata={'fps': 10}))
    stim_b = Stimulus({'A1': BiphasicPulseTrain(1000, -500, 0.1,
                                                stim_dur=5000)})
    resp_a = _ModelResponse(data, time, ms, (data.shape[0],),
                            frame_clock=clock,
                            metadata={'stim': stim_a, 'garbage': 123})
    resp_b = replace(resp_a, metadata={
        'stim': stim_b, 'garbage': 'completely different',
        'source_frame_time': np.zeros(1),
        'encoder': {'frame_time': np.zeros(1), 'frame_dur': 500.0}})
    temporal = FadingTemporal(tau=2, reduce=reduce)
    out_a = temporal._predict_response(resp_a, t_percept=t_percept)
    out_b = temporal._predict_response(resp_b, t_percept=t_percept)
    data_a, data_b = (np.asarray(out.data) for out in (out_a, out_b))
    assert np.abs(data_a).max() > 0
    npt.assert_array_equal(data_a, data_b)
    npt.assert_array_equal(out_a.time, out_b.time)
    npt.assert_equal(out_a.metadata.get('source_frame_time'),
                     out_b.metadata.get('source_frame_time'))
    # Provenance is passed through, not interpreted:
    assert out_a.metadata['stim'] is stim_a
    assert out_b.metadata['stim'] is stim_b


def test_Model_predict_percept_torch_polarity_warning():
    model = _composite('retina')
    # Anodic only, which FadingTemporal rectifies away:
    source = Stimulus(np.abs(_waveform(model.implant.n_electrodes)),
                      electrodes=model.implant.electrode_names, time=TIME)
    with pytest.warns(UserWarning, match='all-zero percept'):
        _staged_percept(model, source)
    with pytest.warns(UserWarning, match='all-zero percept'):
        percept = model.predict_percept(source)
    assert not np.any(percept.data)


def test_Model_tensor_polarity_warning():
    model = _composite('retina')
    waveform = torch.tensor(np.abs(_waveform(model.implant.n_electrodes)),
                            dtype=torch.float32, requires_grad=True)
    with pytest.warns(UserWarning, match='all-zero percept'):
        resp = model._predict_tensor(waveform, TIME)
    # The warning does not detach the response:
    assert resp.data.requires_grad and resp.data.grad_fn is not None


# Nanduri 2012 and Horsager 2009 on Torch:

def _nanduri_spatial(**params):
    params = {'xrange': (-6, 6), 'yrange': (-5, 5), 'step': 0.5,
              'thresh_percept': 0.5, **params}
    return Nanduri2012Spatial(ArgusI(), **params).build()


def _nanduri_model(reduce='last', **params):
    return Nanduri2012Model(ArgusI(), xrange=(-6, 6), yrange=(-5, 5),
                            step=0.5, reduce=reduce, **params).build()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_Nanduri2012Spatial_tensor_parity(dtype):
    spatial = _nanduri_spatial()
    wf = _waveform(spatial.implant.n_electrodes)
    percept = spatial.predict_percept(
        Stimulus(wf, electrodes=spatial.implant.electrode_names, time=TIME))
    resp = spatial._predict_tensor(torch.tensor(wf, dtype=dtype), TIME)
    assert resp.data.dtype == dtype
    assert resp.space is spatial.grid
    expected = percept.data.reshape(resp.data.shape)
    # The threshold must zero some, but not all, of the response:
    assert 0 < np.mean(expected == 0) < 1
    npt.assert_array_equal(resp.data.numpy() == 0, expected == 0)
    _assert_peak_close(resp.data.numpy(), expected)


def test_Nanduri2012Spatial_tensor_gradcheck():
    spatial = _nanduri_spatial(xrange=(-3, 3), yrange=(-2, 2), step=1,
                               thresh_percept=0)
    waveform = torch.tensor(_waveform(spatial.implant.n_electrodes),
                            dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda w: spatial._predict_tensor(w, TIME).data, (waveform,))


@pytest.mark.parametrize('temporal, params', [
    (Horsager2009Temporal, {}),
    (Horsager2009Temporal, {'thresh_percept': 1}),
    (Nanduri2012Temporal, {}),
    # Thresholding also resets the state:
    (Nanduri2012Temporal, {'thresh_percept': 0.01}),
])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('t_percept', [None, [0.3, 0.305, 1.0, 2.0, 50.0]])
def test_retina_temporal_tensor_parity(temporal, params, dtype, t_percept):
    # `reduce='peak'` subsamples automatic output intervals:
    temporal = temporal(reduce='peak', **params)
    wf = _waveform(8)
    expected = temporal.predict_percept(Stimulus(wf, time=TIME),
                                        t_percept=t_percept)
    resp = temporal._predict_response(
        _ModelResponse(torch.tensor(wf, dtype=dtype), TIME, ms,
                       (wf.shape[0],)), t_percept=t_percept)
    assert isinstance(resp.data, torch.Tensor)
    assert resp.data.dtype == dtype
    assert expected.data.dtype == np.float32
    npt.assert_allclose(resp.time, expected.time)
    expected = expected.data.reshape(resp.data.shape)
    assert np.any(expected > 0)
    if params:
        assert np.any(expected[1:4] == 0)
    assert torch.all(resp.data[::4] == 0)
    # float64 differs from the float32 public route by float32 rounding,
    # which Horsager's power nonlinearity amplifies by about beta:
    npt.assert_allclose(resp.data.numpy(), expected,
                        rtol=RTOL if dtype == torch.float32 else 1e-5,
                        atol=1e-6 * np.abs(expected).max())


@pytest.mark.parametrize('temporal, sign', [(Horsager2009Temporal, -1),
                                            (Nanduri2012Temporal, 1)])
def test_retina_temporal_gradcheck(temporal, sign):
    # Mostly driving polarity, away from rectifier kinks, plus one opposite
    # frame. Both polarities make anodic charge, so the final zero frame
    # decays into steps that are skipped as rectified:
    data = sign * np.random.default_rng(4).uniform(5, 20, (2, 4))
    data[:, 2] = -sign * 3
    data[:, -1] = 0
    _temporal_gradcheck(temporal(dt=0.05), data, [0, 0.5, 1.0, 3.0],
                        [0.5, 1.5, 4.0, 10.0, 60.0], 'last')


@pytest.mark.parametrize('reduce', ['last', 'peak'])
@pytest.mark.parametrize('t_percept', [None, [0.5, 1.0, 2.0, 25.0, 60.0]])
def test_Nanduri2012Model_torch_parity(reduce, t_percept):
    model = _nanduri_model(reduce=reduce)
    wf = _waveform(model.implant.n_electrodes)
    source = Stimulus(wf, electrodes=model.implant.electrode_names, time=TIME)
    expected = _staged_percept(model, source, t_percept=t_percept)
    _assert_same_percept(model.predict_percept(source, t_percept=t_percept),
                         expected)
    resp = model._predict_tensor(torch.tensor(wf, dtype=torch.float32), TIME,
                                 t_percept=t_percept)
    npt.assert_allclose(resp.time, expected.time)
    _assert_peak_close(resp.data.numpy(),
                       expected.data.reshape(resp.data.shape))


@pytest.mark.parametrize('reduce', ['last', 'peak'])
def test_Nanduri2012Model_tensor_autograd(reduce):
    model = _nanduri_model(reduce=reduce)
    waveform = torch.tensor(_waveform(model.implant.n_electrodes),
                            dtype=torch.float32, requires_grad=True)
    # Automatic output times, which `reduce` summarizes:
    resp = model._predict_tensor(waveform, TIME)
    assert resp.data.requires_grad and resp.data.grad_fn is not None
    resp.data.square().mean().backward()
    assert torch.all(torch.isfinite(waveform.grad))
    # Current spread reaches every grid point, even from silent electrodes:
    assert waveform.grad[::4].abs().sum() > 0


def test_Nanduri2012Model_predict_percept_skips_staged_path(monkeypatch):
    model = _nanduri_model()

    def staged(*args, **kwargs):
        raise AssertionError("Staged path called")

    monkeypatch.setattr(Nanduri2012Spatial, '_predict_spatial', staged)
    monkeypatch.setattr(Nanduri2012Temporal, '_predict_temporal', staged)
    source = Stimulus(_waveform(model.implant.n_electrodes),
                      electrodes=model.implant.electrode_names, time=TIME)
    assert np.abs(model.predict_percept(source).data).max() > 0
