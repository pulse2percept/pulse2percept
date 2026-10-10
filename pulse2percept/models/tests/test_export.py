"""Exact ONNX export: predict_percept -> Torch module -> ONNX Runtime."""
import json

import numpy as np
import numpy.testing as npt
import pytest
import torch

import pulse2percept as p2p
from pulse2percept.implants.cortex import Orion
from pulse2percept.implants.retina import ArgusII
from pulse2percept.models import FadingTemporal, Model
from pulse2percept.models import cortex
from pulse2percept.models.cortex.tests.test_tensor import _CurvedPolimeni
from pulse2percept.models.export import _deploy_module, _save_onnx
from pulse2percept.models.retina import (AxonMapModel, AxonMapSpatial,
                                         Nanduri2012Spatial, ScoreboardModel,
                                         ScoreboardSpatial)
from pulse2percept.models.tests.test_tensor import (_axon_spatial, _one_axon,
                                                    _waveform)
from pulse2percept.stimuli import (AmplitudeEncoder, FrequencyEncoder,
                                   ImageStimulus, Stimulus)
from pulse2percept.topography.cortex import Polimeni2006Map
from pulse2percept.units import mm, xTh

# float32 parity relative to the peak |response|: matrix products sum in a
# runtime-dependent order (BLAS, ONNX Runtime).
RTOL = 1e-6


def _assert_close(actual, expected):
    npt.assert_allclose(actual, expected, rtol=0,
                        atol=RTOL * np.abs(expected).max())


def _run_onnx(path, data):
    ort = pytest.importorskip('onnxruntime')
    session = ort.InferenceSession(str(path),
                                   providers=['CPUExecutionProvider'])
    return session.run(None, {'image': np.asarray(data, dtype=np.float32)})[0]


def _images(shape):
    """Return diagnostic gray images; 'marker' is asymmetric under flips."""
    h, w = shape
    rows, cols = np.mgrid[:h, :w].astype(np.float32)
    marker = np.zeros(shape, dtype=np.float32)
    marker[0, 0], marker[0, 1], marker[1, 0] = 1, 0.5, 0.25
    return {'black': np.zeros(shape, dtype=np.float32),
            'gray': np.full(shape, 0.5, dtype=np.float32),
            'white': np.ones(shape, dtype=np.float32),
            'hramp': cols / max(w - 1, 1), 'vramp': rows / max(h - 1, 1),
            'marker': marker}


def _chain(model, images, tmp_path):
    """Export ``model``; return predict_percept, Torch module and ONNX
    Runtime percepts of ``images``, each ``(n_images, Hp, Wp)``."""
    pytest.importorskip('onnxscript')
    path = tmp_path / 'model.onnx'
    shape = images[0].shape
    p2p.export_onnx(model, path, shape)
    spatial = model.spatial if isinstance(model, Model) else model
    module = _deploy_module(spatial, shape)
    expected, eager, onnx = [], [], []
    for image in images:
        expected.append(model.predict_percept(ImageStimulus(image)).data[
            ..., 0])
        with torch.inference_mode():
            eager.append(module(torch.as_tensor(image)[None, None]).numpy())
        onnx.append(_run_onnx(path, image[None, None]))
    return (np.array(expected), np.array(eager)[:, 0, 0],
            np.array(onnx)[:, 0, 0])


def _assert_chain(model, images, tmp_path):
    expected, eager, onnx = _chain(model, images, tmp_path)
    _assert_close(eager, expected)
    _assert_close(onnx, eager)
    return expected


def _encoded(implant, amp_range=(0, 50)):
    implant.encoder = AmplitudeEncoder(amp_range=amp_range)
    return implant


def _retina(**params):
    params = {'xrange': (-8, 8), 'yrange': (-6, 6), 'step': 0.25, **params}
    return ScoreboardModel(ArgusII(), **params)


def _cortex(implant=None, **params):
    # Reaches V1-V3 near the vertical meridian; no grid column sits on it:
    params = {'implant_position': (10, 10) * mm, 'xrange': (-4.1, 3.9),
              'yrange': (-3, 3), 'step': 0.2, 'rho': 1000,
              'thresh_percept': 0.5, **params}
    return cortex.ScoreboardModel(
        _encoded(Orion() if implant is None else implant), **params)


# Image sampling


@pytest.mark.parametrize('shape', [(6, 10), (13, 7), (40, 63)])
def test_drive_matches_prepare_stim(shape):
    implant = ArgusII()
    module = _deploy_module(_retina().spatial.build(), shape)
    for image in _images(shape).values():
        view = implant.prepare_stim(ImageStimulus(image))._spatial_view()
        expected = dict(zip(view.electrodes, view.data.ravel()))
        with torch.inference_mode():
            drive = module.drive(torch.as_tensor(image)[None, None])
        npt.assert_allclose(drive.numpy().ravel(),
                            [expected[e] for e in implant.electrode_names],
                            rtol=1e-6, atol=1e-5)


def test_drive_orientation():
    # On a (6, 10) image every pixel lands on one Argus II electrode: row 0
    # at the smallest y (row A), column 0 at the smallest x (column 1):
    image = np.zeros((6, 10), dtype=np.float32)
    image[0, 0], image[1, 3] = 1, 0.5
    module = _deploy_module(_retina().spatial.build(), image.shape)
    with torch.inference_mode():
        drive = module.drive(torch.as_tensor(image)[None, None]).numpy()
    drive = dict(zip(ArgusII().electrode_names, drive.ravel()))
    assert drive.pop('A1') == 50 and drive.pop('B4') == 25
    assert all(v == 0 for v in drive.values())


@pytest.mark.parametrize('shape', [(6, 10), (13, 7)])
def test_export_onnx_images(shape, tmp_path):
    expected = _assert_chain(_retina(), list(_images(shape).values()),
                             tmp_path)
    # Raw response: not normalized, and black is not white:
    assert np.all(expected[0] == 0) and expected[2].max() > 1
    # A flipped marker would light another corner:
    marker = expected[-1]
    assert not np.allclose(marker, marker[::-1]) and \
        not np.allclose(marker, marker[:, ::-1])


def test_export_onnx_deactivated(tmp_path):
    implant = ArgusII()
    implant.deactivate(['A1', 'C5'])
    model = ScoreboardModel(implant, xrange=(-8, 8), yrange=(-6, 6))
    _assert_chain(model, [np.ones((6, 10), dtype=np.float32)], tmp_path)


def test_export_onnx_frequency_zero(tmp_path):
    # A 0 Hz pulse train delivers nothing:
    implant = ArgusII()
    implant.encoder = AmplitudeEncoder(amp_range=(10, 50), freq=0)
    model = ScoreboardModel(implant, xrange=(-8, 8), yrange=(-6, 6))
    expected = _assert_chain(model, [np.ones((6, 10), dtype=np.float32)],
                             tmp_path)
    assert np.all(expected == 0)


# Scoreboard


def _pixels(shape, *where):
    image = np.zeros(shape, dtype=np.float32)
    for r, c in where:
        image[r, c] = 1
    return image


@pytest.mark.parametrize('params, image', [
    # One electrode, and two adjacent ones:
    ({}, _pixels((6, 10), (2, 4))),
    ({'rho': 400}, _pixels((6, 10), (2, 4), (2, 5))),
    ({'implant_position': (300, -200), 'implant_rotation': 20},
     _images((40, 63))['hramp']),
    ({'thresh_percept': 5, 'rho': 300}, _images((40, 63))['vramp']),
    ({'location_noise': 0.5}, _images((13, 7))['marker']),
])
def test_export_onnx_retina_scoreboard(params, image, tmp_path):
    expected = _assert_chain(_retina(**params), [image], tmp_path)
    assert 0 < np.mean(expected == 0) < 1


def test_export_onnx_retina_scoreboard_thresh_boundary(tmp_path):
    # One electrode at 50 uA: each response is a single exact product, so a
    # response equal to the threshold must survive (`|r| < thresh` is 0):
    image = _pixels((6, 10), (2, 4))
    model = _retina()
    resp = model.predict_percept(ImageStimulus(image)).data
    thresh = float(np.sort(resp[resp > 0])[resp[resp > 0].size // 2])
    model.spatial.thresh_percept = thresh
    expected, eager, onnx = _chain(model, [image], tmp_path)
    assert np.any(expected == thresh)
    assert np.all((expected == 0) | (expected >= thresh))
    npt.assert_array_equal(eager, expected)
    npt.assert_array_equal(onnx, expected)


class _HoledPolimeni(Polimeni2006Map):
    """Polimeni2006Map that maps no location with |y| < 0.6 dva."""

    def from_dva(self):
        def holed(to_cortex):
            def fn(x, y):
                hole = np.abs(y) < 0.6
                return tuple(np.where(hole, np.nan, c)
                             for c in to_cortex(x, y))
            return fn
        return {region: holed(fn) for region, fn in super().from_dva().items()}


@pytest.mark.parametrize('params', [
    # Default map, with meridian blending:
    {},
    # Each region is thresholded before the sum:
    {'visual_field_map': Polimeni2006Map(regions=['v1', 'v2', 'v3']),
     'meridian_blend': 0, 'thresh_percept': 2},
    {'visual_field_map': Polimeni2006Map(regions=['v1', 'v2', 'v3']),
     'meridian_blend': 0.5, 'thresh_percept': 2},
    # No current crosses the fissure:
    {'visual_field_map': Polimeni2006Map(), 'implant_position': (-10, 0) * mm,
     'rho': 3000, 'thresh_percept': 0},
    {'visual_field_map': _HoledPolimeni(), 'meridian_blend': 0},
])
def test_export_onnx_cortex_scoreboard(params, tmp_path):
    expected = _assert_chain(_cortex(**params), [_images((30, 30))['hramp']],
                             tmp_path)
    assert np.any(expected != 0)


def test_export_onnx_cortex_scoreboard_unmapped(tmp_path):
    model = _cortex(visual_field_map=_HoledPolimeni(), meridian_blend=0,
                    thresh_percept=0)
    expected = _assert_chain(model, [np.ones((30, 30), dtype=np.float32)],
                             tmp_path)[0]
    hole = np.abs(model.spatial.grid.y) < 0.6
    assert np.all(expected[hole] == 0) and np.any(expected[~hole] != 0)


def test_export_onnx_cortex_scoreboard_3d(tmp_path):
    model = _cortex(implant_position=(0, 0), rho=2000,
                    visual_field_map=_CurvedPolimeni(
                        regions=['v1', 'v2', 'v3']))
    assert np.ptp(model.spatial.build().grid.v1.z) > 0
    expected = _assert_chain(model, [_images((30, 30))['vramp']], tmp_path)
    assert 0 < np.mean(expected == 0) < 1


# AxonMap: the spatial module alone, with signed drive


def _adapter_chain(spatial, drive, tmp_path):
    """Return predict_percept, Torch module and ONNX Runtime responses
    ``(P,)`` to drive ``(E,)``."""
    names = spatial.implant.electrode_names
    expected = spatial.predict_percept(Stimulus(drive, electrodes=names))
    drive = torch.as_tensor(drive, dtype=torch.float32)[:, None]
    adapter = spatial._onnx_adapter()
    with torch.inference_mode():
        eager = adapter(drive).numpy().ravel()
    pytest.importorskip('onnxscript')
    path = tmp_path / 'adapter.onnx'
    _save_onnx(adapter, drive, path)
    return (expected.data.ravel(), eager,
            _run_onnx(path, drive.numpy()).ravel())


def _assert_adapter_chain(spatial, drive, tmp_path):
    expected, eager, onnx = _adapter_chain(spatial, drive, tmp_path)
    _assert_close(eager, expected)
    _assert_close(onnx, eager)
    return expected


def _single(spatial, name, amp):
    """Return drive ``(E,)`` of only electrode ``name``."""
    drive = np.zeros(spatial.implant.n_electrodes)
    drive[list(spatial.implant.electrode_names).index(name)] = amp
    return drive


def _site(spatial, name):
    x, y, _ = spatial._electrode_coords(spatial.implant.electrode_array,
                                        None, electrodes=[name])
    return x[0], y[0]


@pytest.mark.parametrize('params', [
    {},
    {'meridian_blend': 0},
    {'thresh_percept': 5},
    {'implant_position': (300, -200), 'implant_rotation': 20},
    {'location_noise': 0.5},
])
def test_axon_map_adapter_signed(params, tmp_path):
    spatial = _axon_spatial(**params).build()
    # Mixed polarity on all electrodes at once:
    expected = _assert_adapter_chain(
        spatial, _waveform(spatial.implant.n_electrodes)[:, 0], tmp_path)
    assert expected.min() < 0 < expected.max()


def test_axon_map_adapter_cathodic(tmp_path):
    # The winner has the largest |response|, not the largest value:
    spatial = _axon_spatial(meridian_blend=0).build()
    drive = -np.abs(_waveform(spatial.implant.n_electrodes)[:, 1])
    expected = _assert_adapter_chain(spatial, drive, tmp_path)
    assert np.all(expected <= 0) and expected.min() < 0


def test_axon_map_adapter_simultaneous(tmp_path):
    # Segment 1 sits on C3, segment 2 (less sensitive) on C4. Each electrode
    # alone picks its own segment; together, segment 1 wins for both:
    spatial = _axon_spatial(meridian_blend=0, thresh_percept=0).build()
    (x3, y3), (x4, y4) = _site(spatial, 'C3'), _site(spatial, 'C4')
    _one_axon(spatial, [[x3, y3, 1], [x4, y4, 0.8]])
    alone = [_assert_adapter_chain(spatial, _single(spatial, e, a),
                                   tmp_path)[0]
             for e, a in (('C3', 10), ('C4', -10))]
    both = _single(spatial, 'C3', 10) + _single(spatial, 'C4', -10)
    together = _assert_adapter_chain(spatial, both, tmp_path)[0]
    assert alone == [10, -8]
    assert together > 0 and not np.isclose(together, sum(alone))


@pytest.mark.parametrize('sign', [1, -1])
def test_axon_map_adapter_first_tie(sign, tmp_path):
    # Opposite sensitivities tie |response| exactly; the first segment wins:
    spatial = _axon_spatial(meridian_blend=0, thresh_percept=0).build()
    x, y = _site(spatial, 'C3')
    _one_axon(spatial, [[x, y, sign], [x, y, -sign]])
    expected, eager, onnx = _adapter_chain(
        spatial, _single(spatial, 'C3', 17.5), tmp_path)
    for actual in (expected, eager, onnx):
        assert actual[0] == sign * 17.5 and np.all(actual[1:] == 0)


@pytest.mark.parametrize('below', [False, True])
def test_axon_map_adapter_cutoff_inclusive(below, tmp_path, monkeypatch):
    # One segment 100 um right of C10: r2 == cutoff_r2 is retained.
    cutoff_r2 = np.float32(100 * 100)
    if below:
        cutoff_r2 = np.nextafter(cutoff_r2, np.float32(0))
    monkeypatch.setattr(AxonMapSpatial, '_cutoff_r2',
                        lambda self, rho: cutoff_r2)
    spatial = _axon_spatial(meridian_blend=0, thresh_percept=0).build()
    x, y = _site(spatial, 'C10')
    _one_axon(spatial, [[x + 100, y, 1]])
    expected, eager, onnx = _adapter_chain(
        spatial, _single(spatial, 'C10', 20), tmp_path)
    for actual in (eager, onnx):
        npt.assert_array_equal(actual, expected)
    assert (expected[0] == 0) == below


def test_axon_map_adapter_thresh_inclusive(tmp_path):
    # One electrode: every response is exact, so a response equal to the
    # threshold must survive (`|r| >= thresh` is kept):
    spatial = _axon_spatial(meridian_blend=0, thresh_percept=0).build()
    drive = _single(spatial, 'C3', 30)
    resp = spatial.predict_percept(
        Stimulus(drive, electrodes=spatial.implant.electrode_names)).data
    spatial.thresh_percept = float(np.sort(resp[resp > 0])[
        resp[resp > 0].size // 2])
    expected, eager, onnx = _adapter_chain(spatial, drive, tmp_path)
    assert np.any(expected == spatial.thresh_percept)
    for actual in (eager, onnx):
        npt.assert_array_equal(actual, expected)


def test_axon_map_adapter_meridian_blend(tmp_path):
    drive = _waveform(60)[:, 0]
    unblended = _assert_adapter_chain(
        _axon_spatial(meridian_blend=0, thresh_percept=5).build(), drive,
        tmp_path)
    blended = _assert_adapter_chain(
        _axon_spatial(meridian_blend=2, thresh_percept=5).build(), drive,
        tmp_path)
    assert not np.allclose(blended, unblended)
    # Threshold reapplied after blending:
    assert np.all((blended == 0) | (np.abs(blended) >= 5))


def test_axon_map_adapter_short_and_empty_axons(tmp_path):
    # Pixel 0 has a one-segment axon; every other pixel has none:
    spatial = _axon_spatial(meridian_blend=0, thresh_percept=0).build()
    x, y = _site(spatial, 'D2')
    _one_axon(spatial, [[x + 50, y, 0.5]])
    expected = _assert_adapter_chain(spatial, _single(spatial, 'D2', 40),
                                     tmp_path)
    assert expected[0] > 0 and np.all(expected[1:] == 0)


@pytest.mark.parametrize('params', [{}, {'thresh_percept': 5}])
def test_export_onnx_axon_map(params, tmp_path):
    model = AxonMapModel(ArgusII(), **{
        'xrange': (-6, 6), 'yrange': (-4, 4), 'step': 0.5, 'n_axons': 200,
        'n_ax_segments': 200, 'ignore_pickle': True, **params})
    images = _images((30, 50))
    expected = _assert_chain(model, [images['marker'], images['hramp']],
                             tmp_path)
    assert np.any(expected != 0)


# Public API


def test_export_onnx_sidecar(tmp_path):
    pytest.importorskip('onnxscript')
    model = _cortex(meridian_blend=0)
    assert not model.is_built
    p2p.export_onnx(model, tmp_path / 'cortex.onnx', [17, 23])
    # Builds as prediction does:
    assert model.is_built
    with open(tmp_path / 'cortex.json') as f:
        meta = json.load(f)
    grid = model.spatial.grid
    assert meta['schema_version'] == 1
    assert meta['model'] == 'ScoreboardModel'
    assert meta['spatial_model'] == 'ScoreboardSpatial'
    assert meta['implant'] == 'Orion'
    assert meta['input']['shape'] == [1, 1, 17, 23]
    assert meta['output']['shape'] == [1, 1, *grid.x.shape]
    npt.assert_allclose(meta['percept_grid']['x'], grid.x[0])
    npt.assert_allclose(meta['percept_grid']['y'], grid.y[:, 0])
    assert meta['electrodes'] == [str(e) for e in
                                  model.implant.electrode_names]
    assert meta['encoder']['amp_range'] == [0, 50]
    assert meta['spatial_params']['rho'] == 1000
    assert meta['spatial_param_units']['rho'] == 'um'
    assert meta['approximations'] == []
    session_shape = _run_onnx(tmp_path / 'cortex.onnx',
                              np.zeros((1, 1, 17, 23))).shape
    assert list(session_shape) == meta['output']['shape']


class _Subclassed(ScoreboardSpatial):
    pass


class _CustomEncoder(AmplitudeEncoder):
    pass


def _with_encoder(encoder):
    implant = ArgusII()
    implant.encoder = encoder
    return ScoreboardModel(implant)


def _with_implant(**params):
    implant = ArgusII(**params)
    return ScoreboardModel(implant)


@pytest.mark.parametrize('model, error, match', [
    (Model(ScoreboardSpatial(ArgusII()), FadingTemporal()),
     NotImplementedError, 'spatial-only'),
    (FadingTemporal(), TypeError, 'Model or SpatialModel'),
    (ScoreboardModel(ArgusII(), n_gray=8), NotImplementedError, 'n_gray'),
    (_Subclassed(ArgusII()), NotImplementedError, '_Subclassed'),
    (Model(Nanduri2012Spatial(ArgusII())), NotImplementedError,
     'Nanduri2012Spatial'),
    (_with_encoder(None), NotImplementedError, 'AmplitudeEncoder'),
    (_with_encoder(FrequencyEncoder()), NotImplementedError,
     'AmplitudeEncoder'),
    (_with_encoder(_CustomEncoder()), NotImplementedError,
     'AmplitudeEncoder'),
    (_with_encoder(AmplitudeEncoder(stretch=True)), NotImplementedError,
     'stretch'),
    (_with_encoder(AmplitudeEncoder(n_levels=8)), NotImplementedError,
     'n_levels'),
    (_with_encoder(AmplitudeEncoder(amp_range=(0 * xTh, 2 * xTh))),
     NotImplementedError, 'uA'),
    (_with_implant(safe_mode=True), NotImplementedError, 'safe_mode'),
    (_with_implant(preprocess=lambda stim: stim), NotImplementedError,
     'preprocess'),
])
def test_export_onnx_unsupported(model, error, match, tmp_path):
    path = tmp_path / 'model.onnx'
    with pytest.raises(error, match=match):
        p2p.export_onnx(model, path, (8, 8))
    assert not path.exists()


@pytest.mark.parametrize('shape', [(8,), (8, 0), (8, 2.5), (1, 2, 3)])
def test_export_onnx_input_shape(shape, tmp_path):
    with pytest.raises(ValueError, match='input_shape'):
        p2p.export_onnx(_retina(), tmp_path / 'model.onnx', shape)


def test_export_onnx_rejects_mismatch(tmp_path, monkeypatch):
    # A graph that disagrees with predict_percept is not left behind:
    pytest.importorskip('onnxscript')
    from pulse2percept.models import _deploy
    forward = _deploy._Scoreboard.forward
    monkeypatch.setattr(_deploy._Scoreboard, 'forward',
                        lambda self, drive: 2 * forward(self, drive))
    path = tmp_path / 'model.onnx'
    with pytest.raises(RuntimeError, match='predict_percept'):
        p2p.export_onnx(_retina(), path, (6, 10))
    assert not path.exists()
