"""Torch execution of cortical ScoreboardSpatial -> FadingTemporal."""
import numpy as np
import numpy.testing as npt
import pytest
import torch

from pulse2percept.implants.cortex import LinearEdgeThread, Neuralink, Orion
from pulse2percept.models import FadingTemporal, Model
from pulse2percept.models.cortex import ScoreboardSpatial
from pulse2percept.models.tests.test_tensor import (
    ATOL, RTOL, TIME, _assert_matches_reference, _assert_peak_close,
    _cython_percept, _scoreboard_reference, _waveform)
from pulse2percept.stimuli import Stimulus
from pulse2percept.topography.cortex import Polimeni2006Map
from pulse2percept.units import mm


def _spatial(implant=None, **params):
    # Reaches V1-V3 near the vertical meridian; no grid column sits on it:
    params = {'implant_position': (10, 10) * mm, 'xrange': (-4.1, 3.9),
              'yrange': (-3, 3), 'step': 0.2, 'rho': 1000,
              'thresh_percept': 0.5, **params}
    return ScoreboardSpatial(Orion() if implant is None else implant,
                             **params)


def _tensor(spatial, wf, dtype=torch.float32):
    return spatial._predict_tensor(torch.tensor(wf, dtype=dtype), TIME)


@pytest.mark.parametrize('params', [
    {},
    {'meridian_blend': 0},
    {'meridian_blend': 0.5, 'thresh_percept': 2},
    {'regions': ['v1', 'v2', 'v3']},
    {'thresh_percept': 2},
    {'implant_position': (12, -4) * mm, 'implant_rotation': 30},
    {'location_noise': 0.5, 'implant_position': (20, 0) * mm},
])
def test_ScoreboardSpatial_matches_reference(params):
    spatial = _spatial(**params).build()
    _assert_matches_reference(spatial,
                              _waveform(spatial.implant.n_electrodes))


def test_ScoreboardSpatial_tensor_meridian_blend():
    # The default blend changes the response near the vertical meridian:
    wf = _waveform(Orion().n_electrodes)
    plain = _scoreboard_reference(_spatial(meridian_blend=0).build(), wf)
    spatial = _spatial().build()
    assert spatial.meridian_blend == 0.1
    expected = _scoreboard_reference(spatial, wf)
    assert np.abs(expected - plain).max() > 1
    npt.assert_allclose(_tensor(spatial, wf).data.numpy(), expected,
                        rtol=RTOL, atol=ATOL)


def test_ScoreboardSpatial_tensor_regions():
    # Each region is thresholded before the regional sum:
    spatial = _spatial(regions=['v1', 'v2', 'v3'], meridian_blend=0,
                       thresh_percept=2).build()
    wf = _waveform(spatial.implant.n_electrodes)
    for region in spatial.regions:
        alone = _spatial(regions=[region], meridian_blend=0).build()
        assert np.any(_scoreboard_reference(alone, wf) != 0)
    expected = _scoreboard_reference(spatial, wf)
    resp = _tensor(spatial, wf).data
    npt.assert_allclose(resp.numpy(), expected, rtol=RTOL, atol=ATOL)
    summed = _tensor(spatial.build(thresh_percept=0), wf).data
    summed = torch.where(summed.abs() >= 2, summed, 0.0)
    assert not np.allclose(summed.numpy(), expected, atol=ATOL)


def test_ScoreboardSpatial_tensor_hemispheres():
    # Orion straddles the fissure; a wide spread would otherwise cross it:
    spatial = _spatial(implant_position=(-10, 0) * mm, rho=3000,
                       meridian_blend=0, thresh_percept=0).build()
    x_el = spatial._electrode_coords(
        spatial.implant.electrode_array, None,
        electrodes=spatial.implant.electrode_names)[0]
    left = x_el < spatial.visual_field_map.left_offset / 2
    assert 0 < left.sum() < left.size
    wf = _waveform(spatial.implant.n_electrodes)
    npt.assert_allclose(_tensor(spatial, wf).data.numpy(),
                        _scoreboard_reference(spatial, wf), rtol=RTOL,
                        atol=ATOL)
    # The left hemisphere alone lights only the right visual field:
    wf[~left] = 0
    resp = _tensor(spatial, wf).data.numpy()
    x = spatial.grid.x.ravel()
    assert np.all(resp[x < 0] == 0)
    assert np.any(resp[x > 0] != 0)
    npt.assert_allclose(resp, _scoreboard_reference(spatial, wf),
                        rtol=RTOL, atol=ATOL)


def test_ScoreboardSpatial_tensor_ignores_z():
    # On a 2D map, electrode z of a 3D implant has no effect:
    implant = LinearEdgeThread(x=20000)
    wf = _waveform(implant.n_electrodes)
    resps = []
    for depth in (0, 2000):
        spatial = _spatial(implant, implant_position=(0, 0), rho=800,
                           step=0.5, implant_depth=depth).build()
        z_el = spatial._electrode_coords(
            implant.electrode_array, None,
            electrodes=implant.electrode_names)[2]
        assert np.ptp(z_el) > 0
        resp = _tensor(spatial, wf).data.numpy()
        npt.assert_allclose(resp, _scoreboard_reference(spatial, wf),
                            rtol=RTOL, atol=ATOL)
        resps.append(resp)
    assert np.any(resps[0] != 0)
    npt.assert_array_equal(resps[0], resps[1])


def test_ScoreboardSpatial_tensor_float64():
    spatial = _spatial().build()
    wf = _waveform(spatial.implant.n_electrodes)
    resp = _tensor(spatial, wf, dtype=torch.float64)
    assert resp.data.dtype == torch.float64
    npt.assert_allclose(resp.data.numpy(),
                        _scoreboard_reference(spatial, wf), rtol=RTOL,
                        atol=1e-4)


def test_ScoreboardSpatial_tensor_unsupported():
    spatial = _spatial(n_gray=8).build()
    waveform = torch.zeros((spatial.implant.n_electrodes, TIME.size))
    with pytest.raises(NotImplementedError, match='n_gray'):
        spatial._predict_tensor(waveform, TIME)


class _CurvedPolimeni(Polimeni2006Map):
    """Split-map Polimeni2006Map on a curved surface, with depth z (um)."""

    def __init__(self, **params):
        super().__init__(ndim=3, **params)

    def from_dva(self):
        def lift(to_cortex):
            def to_3d(x, y):
                xc, yc = to_cortex(x, y)
                return xc, yc, 3000 * np.cos(yc / 10000)
            return to_3d
        return {region: lift(fn) for region, fn in super().from_dva().items()}


# dva locations on both sides of the vertical meridian:
LOCS_3D = [(-1, 0.5), (0.4, -0.5), (1.5, 1), (-2, -1)]


def _spatial_3d(**params):
    """Return a 3D-map model with one thread 300 um above each of LOCS_3D."""
    visual_field_map = _CurvedPolimeni(regions=['v1', 'v2', 'v3'])
    to_v1 = visual_field_map.from_dva()['v1']
    threads = {}
    for i, (x, y) in enumerate(LOCS_3D):
        xc, yc, zc = (c.item() for c in to_v1(np.array([x]), np.array([y])))
        threads[str(i)] = LinearEdgeThread(x=xc, y=yc, z=zc + 300)
    params = {'xrange': (-4.1, 3.9), 'yrange': (-3, 3), 'step': 0.2,
              'rho': 2000, 'thresh_percept': 0.5, **params}
    return ScoreboardSpatial(Neuralink(threads),
                             visual_field_map=visual_field_map, **params)


@pytest.mark.parametrize('params', [
    {},
    {'meridian_blend': 0},
    {'thresh_percept': 2},
])
def test_ScoreboardSpatial_3d_matches_reference(params):
    spatial = _spatial_3d(**params).build()
    z_grid = spatial.grid.v1.z
    z_el = spatial._electrode_coords(
        spatial.implant.electrode_array, None,
        electrodes=spatial.implant.electrode_names)[2]
    assert np.ptp(z_grid) > 0 and np.ptp(z_el) > 0
    _assert_matches_reference(spatial,
                              _waveform(spatial.implant.n_electrodes))


def test_ScoreboardSpatial_tensor_3d_hemispheres():
    spatial = _spatial_3d(meridian_blend=0, thresh_percept=0).build()
    x_el = spatial._electrode_coords(
        spatial.implant.electrode_array, None,
        electrodes=spatial.implant.electrode_names)[0]
    left = x_el < spatial.visual_field_map.left_offset / 2
    assert 0 < left.sum() < left.size
    # The left hemisphere alone lights only the right visual field:
    wf = _waveform(spatial.implant.n_electrodes)
    wf[~left] = 0
    resp = _tensor(spatial, wf).data.numpy()
    x = spatial.grid.x.ravel()
    assert np.all(resp[x < 0] == 0)
    assert np.any(resp[x > 0] != 0)
    _assert_peak_close(resp, _scoreboard_reference(spatial, wf))


def test_ScoreboardSpatial_tensor_neuropythy():
    # Toy NeuropythyMap: 3D, three regions, unmapped grid points:
    pytest.importorskip('neuropythy')
    from pulse2percept.topography.cortex.tests.test_neuropythy import \
        ToyNeuropythyMap
    visual_field_map = ToyNeuropythyMap()
    implant = Neuralink.from_neuropythy(
        visual_field_map, locs=np.array([[0, 0], [1, 1], [2, 1.5]]))
    spatial = _spatial(implant, implant_position=(0, 0), rho=300,
                       visual_field_map=visual_field_map).build()
    unmapped = np.isnan(spatial.grid.v1.x.ravel())
    assert np.any(unmapped)
    wf = _waveform(implant.n_electrodes)
    expected = _scoreboard_reference(spatial, wf)
    resp = _tensor(spatial, wf).data.numpy()
    assert np.all(np.isfinite(resp))
    assert np.all(resp[unmapped] == 0)
    _assert_peak_close(resp, expected)


def _model(reduce='peak', **params):
    return Model(_spatial(**params), FadingTemporal(tau=2, reduce=reduce))


@pytest.mark.parametrize('reduce', ['last', 'peak'])
@pytest.mark.parametrize('t_percept', [None, [0.5, 1.0, 2.0, 25.0, 60.0]])
def test_Model_tensor_parity(reduce, t_percept):
    model = _model(reduce=reduce)
    wf = _waveform(model.implant.n_electrodes)
    expected = _cython_percept(
        model,
        Stimulus(wf, electrodes=model.implant.electrode_names, time=TIME),
        t_percept=t_percept)
    resp = model._predict_tensor(torch.tensor(wf, dtype=torch.float32), TIME,
                                 t_percept=t_percept)
    assert resp.data.shape == (model.spatial.grid.x.size, expected.time.size)
    npt.assert_allclose(resp.time, expected.time)
    assert np.any(expected.data > 0)
    npt.assert_allclose(resp.data.numpy(),
                        expected.data.reshape(resp.data.shape),
                        rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('reduce', ['last', 'peak'])
def test_Model_tensor_autograd(reduce):
    model = _model(reduce=reduce, regions=['v1', 'v2', 'v3'])
    waveform = torch.tensor(_waveform(model.implant.n_electrodes),
                            dtype=torch.float32, requires_grad=True)
    resp = model._predict_tensor(waveform, TIME,
                                 t_percept=[0.5, 1.0, 2.0, 25.0, 60.0])
    assert resp.data.requires_grad and resp.data.grad_fn is not None
    resp.data.square().mean().backward()
    assert torch.all(torch.isfinite(waveform.grad))
    # Silent electrodes still receive gradient through the Gaussian spread:
    assert waveform.grad[::4].abs().sum() > 0


def test_Model_tensor_gradcheck():
    # Thresholds off; a wide blend puts every grid column in the taper:
    model = Model(_spatial(xrange=(-2.5, 2.5), yrange=(-2, 2), step=1,
                           thresh_percept=0, meridian_blend=1,
                           regions=['v1', 'v2']),
                  FadingTemporal(tau=0.5, reduce='peak'))
    waveform = torch.tensor(_waveform(model.implant.n_electrodes),
                            dtype=torch.float64, requires_grad=True)
    torch.autograd.gradcheck(
        lambda w: model._predict_tensor(w, TIME, t_percept=[1.0, 25.0]).data,
        (waveform,))


@pytest.mark.parametrize('reduce', ['last', 'peak'])
def test_Model_tensor_3d(reduce):
    model = Model(_spatial_3d(), FadingTemporal(tau=2, reduce=reduce))
    wf = _waveform(model.implant.n_electrodes)
    t_percept = [0.5, 1.0, 2.0, 25.0, 60.0]
    expected = _cython_percept(
        model,
        Stimulus(wf, electrodes=model.implant.electrode_names, time=TIME),
        t_percept=t_percept)
    waveform = torch.tensor(wf, dtype=torch.float32, requires_grad=True)
    resp = model._predict_tensor(waveform, TIME, t_percept=t_percept)
    assert np.any(expected.data > 0)
    npt.assert_allclose(resp.data.detach().numpy(),
                        expected.data.reshape(resp.data.shape),
                        rtol=RTOL, atol=ATOL)
    resp.data.square().mean().backward()
    assert torch.all(torch.isfinite(waveform.grad))
    # Silent electrodes still receive gradient through the Gaussian spread:
    assert waveform.grad[::4].abs().sum() > 0
