"""p2p.to_torch: batched Torch workflows against predict_percept."""
import numpy as np
import numpy.testing as npt
import pytest
import torch

import pulse2percept as p2p
from pulse2percept.implants.cortex import Orion
from pulse2percept.implants.retina import ArgusII, PRIMAPivotal
from pulse2percept.models import AlphaTemporal, FadingTemporal, Model, cortex
from pulse2percept.models.retina import (AxonMapModel, AxonMapSpatial,
                                         BiphasicAxonMapModel,
                                         Granley2023Model,
                                         Horsager2009Temporal,
                                         Nanduri2012Model, ScoreboardModel,
                                         ScoreboardSpatial,
                                         Thompson2003Model)
from pulse2percept.stimuli import (AmplitudeEncoder, BiphasicPulseTrain,
                                   FrequencyEncoder, ImageStimulus, Stimulus,
                                   VideoStimulus)
from pulse2percept.units import DimensionMismatchError, mm, xTh

SHAPE = (12, 15)
GRID = {'xrange': (-6, 6), 'yrange': (-4, 4), 'step': 0.5}
AXONS = {'n_axons': 200, 'n_ax_segments': 200, 'ignore_pickle': True}


def _images(n=3, shape=SHAPE, seed=0):
    """Return black, an asymmetric ramp, and random gray images."""
    rows, cols = np.mgrid[:shape[0], :shape[1]]
    ramp = (0.7 * (rows + 0.5) / shape[0] + 0.3 * (cols + 0.5) / shape[1])
    rand = np.random.default_rng(seed).random((n,) + shape)
    images = np.concatenate([np.zeros((1,) + shape), ramp[None] ** 2, rand])
    return images[:n].astype(np.float32)


def _encoded(implant, amp_range=(0, 50)):
    implant.encoder = AmplitudeEncoder(amp_range=amp_range)
    return implant


def _argus(amp_range=(0, 50), **params):
    return _encoded(ArgusII(**params), amp_range)


def _percepts(model, sources):
    """Return ``predict_percept`` data, time first: ``(n, [T,] Hp, Wp)``."""
    data = [np.moveaxis(model.predict_percept(s).data, -1, 0)
            for s in sources]
    return np.array(data)


def _run(module, x):
    with torch.no_grad():
        return module(torch.as_tensor(x)).numpy()


def _assert_close(actual, expected):
    # float32 matrix products sum in a backend-dependent order:
    npt.assert_allclose(actual, expected, rtol=0,
                        atol=1e-6 * max(np.abs(expected).max(), 1))


SPATIAL = {
    'scoreboard': lambda: ScoreboardModel(_argus(), rho=150, **GRID),
    'axonmap': lambda: AxonMapModel(_argus(), rho=150, **AXONS, **GRID),
    'left-eye': lambda: AxonMapModel(_argus(eye='left'), rho=150, **AXONS,
                                     **GRID),
    'placed': lambda: AxonMapModel(
        _argus(), rho=150, implant_position=(300, -200), implant_rotation=20,
        location_noise=0.5, **AXONS, **GRID),
    'unencoded': lambda: ScoreboardModel(ArgusII(encoder=None), rho=150,
                                         **GRID),
    'thompson': lambda: Thompson2003Model(_argus(), **GRID),
    'cortex': lambda: cortex.ScoreboardModel(
        _encoded(Orion()), implant_position=(10, 10) * mm,
        xrange=(-4.1, 3.9), yrange=(-3, 3), step=0.2, rho=1000),
}


@pytest.mark.parametrize('make', SPATIAL.values(), ids=SPATIAL.keys())
def test_image_parity_spatial(make):
    model = make()
    images = _images()
    expected = _percepts(model, [ImageStimulus(i) for i in images])[:, 0]
    module = p2p.to_torch(model, SHAPE)
    assert module.t_percept is None
    # Batched, and one image at a time:
    actual = _run(module, images[:, None])
    assert actual.shape == (3, 1) + model.spatial.grid.x.shape
    _assert_close(actual[:, 0], expected)
    _assert_close(_run(module, images[1:2, None])[0, 0], expected[1])
    assert np.all(expected[0] == 0) == np.all(actual[0] == 0)
    # An asymmetric image gives an asymmetric percept:
    assert not np.allclose(expected[1], expected[1][::-1])


TEMPORAL = {
    'scoreboard-fading': lambda: Model(
        ScoreboardSpatial(_argus(), rho=150, **GRID),
        FadingTemporal(reduce='peak')),
    'axonmap-alpha': lambda: Model(
        AxonMapSpatial(_argus(), rho=150, **AXONS, **GRID), AlphaTemporal()),
    'scoreboard-horsager': lambda: Model(
        ScoreboardSpatial(_argus(), rho=150, **GRID), Horsager2009Temporal()),
    'nanduri': lambda: Nanduri2012Model(_argus(), xrange=(-3, 3),
                                        yrange=(-2, 2), step=1),
}


@pytest.mark.parametrize('make', TEMPORAL.values(), ids=TEMPORAL.keys())
def test_image_parity_temporal(make):
    model = make()
    images = _images(2)
    expected = [model.predict_percept(ImageStimulus(i)) for i in images]
    module = p2p.to_torch(model, SHAPE)
    npt.assert_allclose(module.t_percept, expected[0].time)
    actual = _run(module, images[:, None])
    assert actual.shape == (2, 1, module.t_percept.size) + \
        model.spatial.grid.x.shape
    _assert_close(actual[:, 0],
                  np.array([np.moveaxis(p.data, -1, 0) for p in expected]))
    assert np.any(actual[1] > 0)


def test_image_parity_t_percept():
    model = TEMPORAL['scoreboard-fading']()
    t_percept = [10, 50, 125, 400]
    image = _images(2)[1]
    expected = model.predict_percept(ImageStimulus(image), t_percept=t_percept)
    module = p2p.to_torch(model, SHAPE, t_percept=t_percept)
    npt.assert_allclose(module.t_percept, t_percept)
    _assert_close(_run(module, image[None, None])[0, 0],
                  np.moveaxis(expected.data, -1, 0))


@pytest.mark.parametrize('model', ['scoreboard', 'unencoded'])
def test_video_parity_spatial(model):
    model = SPATIAL[model]()
    # Distinct frames: `predict_percept` drops repeated ones.
    video = np.random.default_rng(1).random(SHAPE + (4,)).astype(np.float32)
    expected = model.predict_percept(VideoStimulus(video,
                                                   metadata={'fps': 10}))
    module = p2p.to_torch(model, (4,) + SHAPE, fps=10)
    npt.assert_allclose(module.t_percept, expected.time)
    actual = _run(module, np.moveaxis(video, -1, 0)[None, None])
    _assert_close(actual[0, 0], np.moveaxis(expected.data, -1, 0))


@pytest.mark.parametrize('model', ['scoreboard', 'unencoded'])
def test_video_one_frame(model):
    # A one-frame video keeps its time axis; values are those of the frame:
    model = SPATIAL[model]()
    image = _images(2)[1]
    module = p2p.to_torch(model, (1,) + SHAPE, fps=10)
    npt.assert_allclose(module.t_percept, [0])
    actual = _run(module, image[None, None, None])
    assert actual.shape == (1, 1, 1) + model.spatial.grid.x.shape
    expected = model.predict_percept(ImageStimulus(image)).data[..., 0]
    _assert_close(actual[0, 0, 0], expected)


def test_video_parity_temporal():
    model = TEMPORAL['scoreboard-fading']()
    video = np.random.default_rng(2).random(SHAPE + (3,)).astype(np.float32)
    expected = model.predict_percept(VideoStimulus(video,
                                                   metadata={'fps': 20}))
    module = p2p.to_torch(model, (3,) + SHAPE, fps=20)
    npt.assert_allclose(module.t_percept, expected.time)
    actual = _run(module, np.moveaxis(video, -1, 0)[None, None])
    _assert_close(actual[0, 0], np.moveaxis(expected.data, -1, 0))


def test_rgb_parity():
    model = SPATIAL['scoreboard']()
    rgb = np.random.default_rng(3).random(SHAPE + (3,)).astype(np.float32)
    expected = model.predict_percept(ImageStimulus(rgb)).data[..., 0]
    module = p2p.to_torch(model, SHAPE)
    _assert_close(_run(module, np.moveaxis(rgb, -1, 0)[None])[0, 0],
                  expected)
    with pytest.raises(ValueError, match='C=3'):
        module(torch.zeros((1, 4) + SHAPE))


def test_image_orientation_and_electrode_order():
    # On a (6, 10) image every pixel lands on one Argus II electrode: row 0 at
    # the smallest y (row A), column 0 at the smallest x (column 1).
    model = SPATIAL['scoreboard']()
    image = np.zeros((1, 1, 6, 10), dtype=np.float32)
    image[0, 0, 0, 0] = 1
    module = p2p.to_torch(model, (6, 10))
    drive = module.pulses.drive(module.pulses(
        module.sampler(torch.as_tensor(image))))[0, :, 0]
    names = module.electrode_names
    assert names == list(model.implant.electrode_names)
    assert drive[names.index('A1')] == 50 and drive.count_nonzero() == 1
    # Electrode input, same electrode:
    direct = p2p.to_torch(model, input_kind='electrodes')
    single = np.zeros((1, len(names)), dtype=np.float32)
    single[0, names.index('A1')] = 50
    expected = model.predict_percept({'A1': 50}).data[..., 0]
    _assert_close(_run(direct, single)[0, 0], expected)
    _assert_close(_run(module, image)[0, 0], expected)


def test_electrode_parity_spatial():
    model = SPATIAL['axonmap']()
    names = model.implant.electrode_names
    drive = np.random.default_rng(4).normal(0, 20, (3, len(names)))
    drive[0] = 0
    drive = drive.astype(np.float32)
    expected = _percepts(model, [Stimulus(d, electrodes=names)
                                 for d in drive])[:, 0]
    module = p2p.to_torch(model, input_kind='electrodes')
    _assert_close(_run(module, drive)[:, 0], expected)


def test_electrode_parity_temporal():
    model = TEMPORAL['scoreboard-fading']()
    names = model.implant.electrode_names
    time = np.array([0, 0.3, 0.75, 1.2, 9, 23, 47.5])
    wave = np.random.default_rng(5).normal(0, 30, (2, len(names), time.size))
    wave = wave.astype(np.float32)
    expected = [model.predict_percept(Stimulus(w, electrodes=names,
                                               time=time)) for w in wave]
    module = p2p.to_torch(model, input_kind='electrodes', time=time)
    npt.assert_allclose(module.time, time)
    npt.assert_allclose(module.t_percept, expected[0].time)
    _assert_close(_run(module, wave)[:, 0],
                  np.array([np.moveaxis(p.data, -1, 0) for p in expected]))
    with pytest.raises(ValueError, match=r'\(B, E, T\)'):
        module(torch.zeros((1, len(names))))


# Granley2023


def _granley(amp_range=(0 * xTh, 3 * xTh), **params):
    return Granley2023Model(_argus(amp_range, **params), xrange=(-6, 6),
                            yrange=(-4, 4), step=0.5)


@pytest.mark.parametrize('calibrated', [False, True])
def test_granley_image_parity(calibrated):
    model = _granley()
    if calibrated:
        # Current, converted back to threshold multiples:
        model = _granley(amp_range=(0, 90))
        model.implant.thresholds = {
            n: 20 + i for i, n in enumerate(model.implant.electrode_names)}
    images = _images()
    expected = _percepts(model, [ImageStimulus(i) for i in images])[:, 0]
    module = p2p.to_torch(model, SHAPE)
    actual = _run(module, images[:, None])
    _assert_close(actual[:, 0], expected)
    assert np.any(expected[1] > 0)


def test_granley_electrode_parity():
    model = _granley()
    names = list(model.implant.electrode_names)
    rng = np.random.default_rng(6)
    amp = np.where(rng.random((2, len(names))) < 0.3,
                   rng.uniform(0, 4, (2, len(names))), 0).astype(np.float32)
    freq = rng.uniform(5, 60, len(names)).astype(np.float32)
    pdur = rng.uniform(0.2, 1, len(names)).astype(np.float32)
    expected = _percepts(model, [
        {n: BiphasicPulseTrain(f, a * xTh, p)
         for n, a, f, p in zip(names, row, freq, pdur) if a}
        for row in amp])[:, 0]
    module = p2p.to_torch(model, input_kind='electrodes', freq=freq,
                          phase_dur=pdur)
    _assert_close(_run(module, amp)[:, 0], expected)
    # The same parameters passed to the call, one row per batch element:
    with torch.no_grad():
        actual = p2p.to_torch(model, input_kind='electrodes')(
            torch.as_tensor(amp), torch.as_tensor(freq).expand(2, -1),
            torch.as_tensor(pdur)).numpy()
    _assert_close(actual[:, 0], expected)


def test_granley_drops_zero_frequency():
    # A 0 Hz train delivers no pulse, even if amp_cutoff < 0:
    model = Granley2023Model(ArgusII(), amp_cutoff=-1, **GRID)
    module = p2p.to_torch(model, input_kind='electrodes', freq=0,
                          phase_dur=0.45)
    out = _run(module, np.ones((1, 60), dtype=np.float32))
    assert np.all(out == 0)
    out = _run(module, np.zeros((1, 60), dtype=np.float32))
    assert np.all(out == 0)


def test_granley_gradcheck():
    model = Granley2023Model(ArgusII(), xrange=(-3, 3), yrange=(-2, 2),
                             step=1)
    module = p2p.to_torch(model, input_kind='electrodes').double()
    rng = np.random.default_rng(7)
    # Away from the amplitude cutoff, area floor and eccentricity clip:
    amp, freq, pdur = (torch.tensor(rng.uniform(lo, hi, (2, 60)),
                                    requires_grad=True)
                       for lo, hi in ((1, 3), (10, 50), (0.3, 0.6)))
    assert torch.autograd.gradcheck(module, (amp, freq, pdur))


# Gradients


def test_gradcheck_image_temporal():
    model = Model(ScoreboardSpatial(_argus(), rho=300, xrange=(-3, 3),
                                    yrange=(-2, 2), step=1),
                  FadingTemporal(tau=2, reduce='last'))
    module = p2p.to_torch(model, (6, 10)).double()
    # Gray levels inside (0, 1), away from the clipping kinks:
    x = torch.tensor(np.random.default_rng(8).uniform(0.1, 0.9,
                                                      (2, 1, 6, 10)),
                     requires_grad=True)
    assert torch.autograd.gradcheck(module, (x,))


def test_gradcheck_axonmap():
    model = AxonMapModel(_argus(), rho=200, xrange=(-3, 3), yrange=(-2, 2),
                         step=1, n_axons=100, n_ax_segments=100,
                         ignore_pickle=True)
    module = p2p.to_torch(model, (6, 10)).double()
    x = torch.tensor(np.random.default_rng(9).uniform(0.1, 0.9,
                                                      (2, 1, 6, 10)),
                     requires_grad=True)
    assert torch.autograd.gradcheck(module, (x,))


@pytest.mark.parametrize('make, black', [
    (SPATIAL['axonmap'], True),
    # Temporal models half-wave rectify current; at exactly 0 uA (black)
    # autograd takes the flat side:
    (TEMPORAL['scoreboard-fading'], False)])
def test_gradient_reaches_pixels(make, black):
    module = p2p.to_torch(make(), SHAPE)
    images = torch.tensor(_images(), requires_grad=True)
    (module(images[:, None]) - 1).square().mean().backward()
    grad = images.grad.abs().sum(dim=(1, 2))
    assert torch.all(torch.isfinite(images.grad))
    assert torch.all(grad[1:] > 0) and (grad[0] > 0) == black


def test_batch_independence():
    module = p2p.to_torch(TEMPORAL['scoreboard-fading'](), SHAPE)
    images = torch.tensor(_images()[:, None], requires_grad=True)
    out = module(images)
    out[1].sum().backward()
    assert images.grad[1].abs().sum() > 0
    assert torch.all(images.grad[[0, 2]] == 0)
    with torch.no_grad():
        assert torch.equal(module(images[1:2])[0], out[1])


# Stimulation semantics


def test_deactivated_electrodes():
    model = SPATIAL['scoreboard']()
    model.implant.deactivate(['A1', 'C5', 'F10'])
    images = _images()
    expected = _percepts(model, [ImageStimulus(i) for i in images])[:, 0]
    module = p2p.to_torch(model, SHAPE)
    _assert_close(_run(module, images[:, None])[:, 0], expected)
    # Electrode input, too:
    direct = p2p.to_torch(model, input_kind='electrodes')
    drive = torch.ones((1, 60), requires_grad=True)
    direct(drive).sum().backward()
    off = [direct.electrode_names.index(n) for n in ('A1', 'C5', 'F10')]
    assert torch.all(drive.grad[0, off] == 0)
    assert drive.grad[0].abs().sum() > 0


def test_threshold_calibration():
    model = Model(ScoreboardSpatial(_argus((0 * xTh, 3 * xTh)), rho=150,
                                    **GRID), FadingTemporal())
    thresholds = {n: 10 + i for i, n in enumerate(
        model.implant.electrode_names)}
    model.implant.thresholds = thresholds
    image = _images(2)[1]
    expected = model.predict_percept(ImageStimulus(image))
    module = p2p.to_torch(model, SHAPE)
    _assert_close(_run(module, image[None, None])[0, 0],
                  np.moveaxis(expected.data, -1, 0))
    # Every activated electrode needs a threshold, or none:
    del thresholds['B3']
    model.implant.thresholds = thresholds
    with pytest.raises(NotImplementedError, match='B3'):
        p2p.to_torch(model, SHAPE)
    # Scoreboard needs current:
    model.implant.thresholds = None
    with pytest.raises(DimensionMismatchError, match='threshold'):
        p2p.to_torch(model, SHAPE)


def test_max_current():
    # 60 electrodes at 50 uA; Argus II's raster fires 10 at a time:
    model = TEMPORAL['scoreboard-fading']()
    white = np.ones((1, 1) + SHAPE, dtype=np.float32)
    model.implant.max_current = 450
    with pytest.raises(ValueError, match='max_current'):
        model.predict_percept(ImageStimulus(white[0, 0]))
    module = p2p.to_torch(model, SHAPE)
    with pytest.raises(ValueError, match='max_current=450.0'):
        module(torch.as_tensor(white))
    # The total, not each electrode, is limited:
    model.implant.max_current = 500
    expected = model.predict_percept(ImageStimulus(white[0, 0]))
    module = p2p.to_torch(model, SHAPE)
    _assert_close(_run(module, white)[0, 0],
                  np.moveaxis(expected.data, -1, 0))
    # Without a raster, all 60 fire at once:
    model.implant.raster = None
    with pytest.raises(ValueError, match='max_current'):
        p2p.to_torch(model, SHAPE)(torch.as_tensor(white))


def test_safe_mode():
    model = TEMPORAL['scoreboard-fading']()
    model.implant.safe_mode = True
    # Biphasic pulses are charge-balanced:
    image = _images(2)[1]
    expected = model.predict_percept(ImageStimulus(image))
    _assert_close(_run(p2p.to_torch(model, SHAPE), image[None, None])[0, 0],
                  np.moveaxis(expected.data, -1, 0))
    # A static drive is not:
    spatial = ScoreboardModel(ArgusII(safe_mode=True), rho=150, **GRID)
    with pytest.raises(ValueError, match='charge-balanced'):
        spatial.predict_percept({'A1': 10})
    direct = p2p.to_torch(spatial, input_kind='electrodes')
    with pytest.raises(ValueError, match='charge-balanced'):
        direct(torch.full((1, 60), 10.0))
    assert np.all(_run(direct, np.zeros((1, 60), dtype=np.float32)) == 0)


def test_zero_frequency_encoder():
    implant = ArgusII()
    implant.encoder = AmplitudeEncoder(amp_range=(10, 50), freq=0)
    module = p2p.to_torch(ScoreboardModel(implant, **GRID), SHAPE)
    assert np.all(_run(module, np.ones((1, 1) + SHAPE,
                                       dtype=np.float32)) == 0)


# Module behavior


def test_snapshot():
    model = SPATIAL['axonmap']()
    module = p2p.to_torch(model, SHAPE)
    image = _images(2)[1:, None]
    before = _run(module, image)
    model.spatial.rho = 400
    model.implant.deactivate('C5')
    npt.assert_array_equal(_run(module, image), before)


def test_dtype():
    module = p2p.to_torch(SPATIAL['scoreboard'](), SHAPE)
    images = _images()[:, None]
    single = _run(module, images)
    double = _run(module.double(), images.astype(np.float64))
    assert double.dtype == np.float64
    _assert_close(double, single)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('make', [SPATIAL['axonmap'], SPATIAL['thompson'],
                                  TEMPORAL['scoreboard-fading'],
                                  lambda: _granley()])
def test_cuda(make):
    module = p2p.to_torch(make(), SHAPE)
    images = torch.as_tensor(_images())[:, None]
    expected = _run(module, images)
    module = module.to('cuda')
    assert all(b.device.type == 'cuda' for b in module.buffers())
    images = images.cuda().requires_grad_()
    out = module(images)
    assert out.device.type == 'cuda'
    out.square().mean().backward()
    assert images.grad.device.type == 'cuda'
    assert torch.all(torch.isfinite(images.grad))
    _assert_close(out.detach().cpu().numpy(), expected)


@pytest.mark.parametrize('kind', ['axonmap-image', 'granley-electrodes'])
def test_training_smoke(kind):
    torch.manual_seed(0)
    if kind == 'axonmap-image':
        forward = p2p.to_torch(SPATIAL['axonmap'](), SHAPE)
        n_out, scale = np.prod(SHAPE), 1
    else:
        forward = p2p.to_torch(_granley(), input_kind='electrodes', freq=20,
                               phase_dur=0.45)
        # 0-3 xTh, mostly above the amplitude cutoff:
        n_out, scale = 60, 3
    net = torch.nn.Sequential(torch.nn.Flatten(),
                              torch.nn.Linear(np.prod(SHAPE), 32),
                              torch.nn.ReLU(), torch.nn.Linear(32, n_out),
                              torch.nn.Sigmoid())
    before = [p.detach().clone() for p in net.parameters()]
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    images = torch.as_tensor(_images(8))[:, None]
    for _ in range(3):
        stim = scale * net(images)
        percepts = forward(stim.reshape(images.shape) if n_out > 60 else stim)
        targets = torch.nn.functional.interpolate(
            images, size=percepts.shape[-2:], mode='bilinear')
        loss = torch.nn.functional.mse_loss(
            percepts / percepts.detach().max(), targets)
        optimizer.zero_grad()
        loss.backward()
        assert torch.isfinite(loss)
        for p in net.parameters():
            assert torch.all(torch.isfinite(p.grad)) and p.grad.any()
        optimizer.step()
    assert all(not torch.equal(a, b) for a, b in zip(before,
                                                     net.parameters()))


# Unsupported configurations


def _custom_preprocess():
    implant = _argus(preprocess=lambda stim: stim)
    return ScoreboardModel(implant, **GRID)


def _limited_granley():
    model = _granley()
    model.implant.max_current = 500
    return model


def _frequency_encoded():
    implant = ArgusII()
    implant.encoder = FrequencyEncoder()
    return ScoreboardModel(implant, **GRID)


@pytest.mark.parametrize('make, kwargs, error, match', [
    (_frequency_encoded, {}, NotImplementedError, 'pulse timing'),
    (lambda: ScoreboardModel(_argus(), n_gray=8, **GRID), {},
     NotImplementedError, 'n_gray'),
    (lambda: Granley2023Model(_argus((0 * xTh, 3 * xTh)), n_gray=8,
                              **GRID), {}, NotImplementedError, 'n_gray'),
    (lambda: Thompson2003Model(_argus(), dropout=2, **GRID), {},
     NotImplementedError, 'current parameters'),
    (lambda: BiphasicAxonMapModel(_argus(), **AXONS, **GRID), {},
     NotImplementedError, 'pulse-train parameters'),
    (lambda: Model(ScoreboardSpatial(ArgusII(encoder=None), **GRID),
                   FadingTemporal()), {}, NotImplementedError, 'encoder'),
    (lambda: Model(temporal=FadingTemporal()), {}, NotImplementedError,
     'spatial model'),
    (lambda: ScoreboardModel(PRIMAPivotal(), **GRID), {}, NotImplementedError,
     'electrical current'),
    (_custom_preprocess, {}, NotImplementedError, 'preprocess'),
    (_granley, {'input_shape': (2,) + SHAPE}, NotImplementedError,
     'still images'),
    (_limited_granley, {'input_shape': None, 'input_kind': 'electrodes'},
     NotImplementedError, 'max_current'),
    (SPATIAL['scoreboard'], {'input_shape': None}, ValueError, 'input_shape'),
    (SPATIAL['scoreboard'], {'input_shape': SHAPE, 'input_kind': 'video'},
     ValueError, 'input_shape'),
    (SPATIAL['scoreboard'], {'t_percept': 10}, ValueError, 'temporal'),
    (SPATIAL['scoreboard'], {'fps': 10}, ValueError, 'video'),
    (SPATIAL['scoreboard'], {'freq': 10}, ValueError, 'Granley2023'),
    (TEMPORAL['scoreboard-fading'],
     {'input_shape': None, 'input_kind': 'electrodes'}, ValueError, 'time'),
    (lambda: 'model', {}, TypeError, 'Model or SpatialModel'),
])
def test_unsupported(make, kwargs, error, match):
    kwargs = {'input_shape': SHAPE, **kwargs}
    with pytest.raises(error, match=match):
        p2p.to_torch(make(), **kwargs)


def test_input_validation():
    module = p2p.to_torch(SPATIAL['scoreboard'](), SHAPE)
    for bad in ((1,) + SHAPE, (1, 1, 5, 5), (1, 2) + SHAPE):
        with pytest.raises(ValueError, match=r'\(B, C, H, W\)'):
            module(torch.zeros(bad))
    with pytest.raises(TypeError, match='uint8'):
        module(torch.zeros((1, 1) + SHAPE, dtype=torch.uint8))
    for bad in (np.nan, np.inf):
        image = torch.zeros((2, 1) + SHAPE)
        image[1, 0, 3, 4] = bad
        with pytest.raises(ValueError, match='finite'):
            module(image)
    direct = p2p.to_torch(SPATIAL['scoreboard'](), input_kind='electrodes')
    with pytest.raises(ValueError, match='finite'):
        direct(torch.full((1, 60), np.nan))
    with pytest.raises(TypeError, match='Granley2023'):
        direct(torch.zeros((1, 60)), freq=torch.ones(60))
    granley = p2p.to_torch(_granley(), input_kind='electrodes')
    with pytest.raises(ValueError, match="'freq' is required"):
        granley(torch.zeros((1, 60)), phase_dur=torch.ones(60))
    with pytest.raises(ValueError, match='positive'):
        granley(torch.zeros((1, 60)), freq=torch.ones(60),
                phase_dur=torch.zeros(60))
