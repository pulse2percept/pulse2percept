import numpy as np
import numpy.testing as npt
import pytest
import torch

from pulse2percept.implants import ElectrodeArray, GridImplant, PointSource
from pulse2percept.implants.retina import ArgusII, RetinalImplant
from pulse2percept.models import FadingTemporal
from pulse2percept.models.retina import AxonMapModel, Granley2023Model
from pulse2percept.models.retina.granley2023 import _mvg_response, _mvg_shape
from pulse2percept.stimuli import (AmplitudeEncoder, BiphasicPulseTrain,
                                   MonophasicPulse, VideoStimulus, samples)
from pulse2percept.units import Hz, uA, xTh

# AxonMapModel caches axons to a relative path; write them to a temp dir:
pytestmark = pytest.mark.usefixtures('axon_cache_in_tmp')

#: Default ``_mvg_shape`` parameters of [Granley2023]_.
_SHAPE = dict(rho=75, lam=0.9, a0=0.4733, a1=0.5211, a2=0.016, a3=0.5,
              a4=-0.2122, amp_cutoff=0.25)

_GRID = dict(xrange=(-5, 5), yrange=(-4, 4), step=1)


def _implant(electrodes, eye='right'):
    """Return a retinal implant of point sources at ``{name: (x, y)}`` (um)"""
    return RetinalImplant(ElectrodeArray(
        {name: PointSource(x, y, 0) for name, (x, y) in electrodes.items()}),
        eye=eye)


def _shape(freq, amp, pdur, **params):
    """Return ``_mvg_shape`` of scalar pulse parameters as floats"""
    out = _mvg_shape(torch.tensor([float(freq)]), torch.tensor([float(amp)]),
                     torch.tensor([float(pdur)]), **{**_SHAPE, **params})
    return [float(v) for v in out]


def _principal_axis(percept):
    """Return the intensity-weighted major-axis angle (deg, mod 180) in dva"""
    w = percept.data[..., 0].ravel()
    # Rows run top-down:
    x, y = np.meshgrid(percept.xdva, percept.ydva[::-1])
    pts = np.stack((x.ravel(), y.ravel()))
    d = pts - (pts * w).sum(axis=1, keepdims=True) / w.sum()
    _, vecs = np.linalg.eigh((d * w) @ d.T)
    return np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1])) % 180


# -----------------------------------------------------------------------------
# Reference parity
# -----------------------------------------------------------------------------

#: Electrodes ``(name, x, y, freq, amp, pdur)`` of the reference cases. ``x``,
#: ``y`` are retinal um at p2p 410ece, whose Watson2014Map does not flip y.
_REF_CASES = {
    'single': [('A', 600, 450, 20, 1.5, 0.45)],
    'multi': [('A', -900, -300, 20, 1.5, 0.45),
              ('B', 0, 600, 50, 3.0, 0.2),
              ('C', 900, 0, 10, 0.8, 1.0)],
}

#: ``MVGModel(xrange=(-5, 5), yrange=(-4, 4), xystep=1)`` percepts from
#: bionicvisionlab/2023-NeurIPS-HILO@975dc0f (code/src/phosphene_model.py) on
#: pulse2percept@410ece, Python 3.10. Rows run from y=-4 to y=4 (410ece).
_REF = {
    'single_RE': [
        [0.0253179, 0.0503404, 0.078738, 0.0968793, 0.0937685, 0.071394,
         0.0427607, 0.0201469, 0.00746704, 0.00217705, 0.000499307],
        [0.0283297, 0.0686039, 0.130687, 0.195838, 0.230855, 0.214073,
         0.156157, 0.0896068, 0.0404482, 0.0143627, 0.00401192],
        [0.021898, 0.0645842, 0.14984, 0.273469, 0.392616, 0.443411,
         0.393934, 0.275309, 0.151354, 0.0654559, 0.022268],
        [0.0116926, 0.042, 0.118677, 0.263794, 0.461256, 0.63445,
         0.686485, 0.584311, 0.391234, 0.206066, 0.0853801],
        [0.00431283, 0.0188677, 0.0649313, 0.175779, 0.374336, 0.627095,
         0.826389, 0.856672, 0.698591, 0.448137, 0.22614],
        [0.0010989, 0.00585508, 0.0245406, 0.0809126, 0.209858, 0.428169,
         0.6872, 0.867621, 0.8617, 0.673225, 0.413756],
        [0.000193421, 0.00125514, 0.00640711, 0.0257282, 0.0812713,
         0.20195, 0.394755, 0.607004, 0.734233, 0.698643, 0.522945],
        [2.35175e-05, 0.000185865, 0.00115554, 0.00565132, 0.0217417,
         0.0657985, 0.156645, 0.293358, 0.432173, 0.500837, 0.456576],
        [1.97527e-06, 1.9013e-05, 0.000143964, 0.000857501, 0.00401786,
         0.0148093, 0.0429392, 0.0979379, 0.175722, 0.248018, 0.27537],
    ],
    'multi_RE': [
        [1.22622, 1.39256, 1.18651, 0.888036, 0.65782, 0.462086, 0.282289,
         0.152079, 0.0861878, 0.0617858, 0.048076],
        [1.23108, 1.87432, 1.98568, 1.5305, 0.930798, 0.494546, 0.263287,
         0.185212, 0.189187, 0.196946, 0.163619],
        [1.10861, 1.85688, 2.29805, 2.00666, 1.23115, 0.580132, 0.319183,
         0.335882, 0.42418, 0.431757, 0.326059],
        [0.908046, 1.35516, 1.8408, 2.00415, 1.58519, 0.926152, 0.546947,
         0.521019, 0.586007, 0.532465, 0.358098],
        [0.612158, 0.723353, 1.02182, 1.5007, 1.74027, 1.43959, 0.923238,
         0.609729, 0.484766, 0.365979, 0.216004],
        [0.307285, 0.278887, 0.393568, 0.822624, 1.43632, 1.70993,
         1.36701, 0.77933, 0.367793, 0.170133, 0.0756987],
        [0.109032, 0.076597, 0.105265, 0.323099, 0.845091, 1.46326,
         1.63633, 1.18396, 0.560532, 0.178842, 0.0412833],
        [0.0267864, 0.0148028, 0.0195495, 0.0896567, 0.349057, 0.886868,
         1.44728, 1.51527, 1.01781, 0.438757, 0.121493],
        [0.00452229, 0.00199371, 0.00252006, 0.0174522, 0.100772,
         0.376924, 0.904794, 1.39292, 1.37504, 0.870345, 0.353218],
    ],
    'single_LE': [
        [7.6352e-09, 2.55047e-07, 5.71999e-06, 8.61277e-05, 0.000870692,
         0.00590962, 0.0269295, 0.0823893, 0.169234, 0.233387, 0.216093],
        [6.87526e-08, 1.90808e-06, 3.55532e-05, 0.000444769, 0.00373563,
         0.0210653, 0.0797523, 0.202719, 0.345953, 0.396383, 0.30492],
        [5.01082e-07, 1.15538e-05, 0.00017886, 0.00185899, 0.0129722,
         0.060775, 0.191165, 0.403708, 0.572399, 0.544884, 0.348244],
        [2.95582e-06, 5.66242e-05, 0.000728282, 0.00628884, 0.0364599,
         0.141917, 0.370874, 0.650716, 0.766534, 0.60624, 0.321907],
        [1.41124e-05, 0.000224611, 0.00240014, 0.0172193, 0.0829406,
         0.268221, 0.582363, 0.84892, 0.830834, 0.545928, 0.240841],
        [5.45345e-05, 0.000721125, 0.00640211, 0.0381601, 0.152711,
         0.410302, 0.740136, 0.896382, 0.728867, 0.397903, 0.145841],
        [0.000170566, 0.00187387, 0.0138217, 0.0684472, 0.227575,
         0.508002, 0.761344, 0.766073, 0.517527, 0.23473, 0.0714791],
        [0.000431784, 0.00394114, 0.0241518, 0.0993693, 0.274491, 0.50907,
         0.633871, 0.529905, 0.297419, 0.112076, 0.028355],
        [0.00088469, 0.00670893, 0.0341578, 0.116761, 0.267968, 0.412896,
         0.427141, 0.296672, 0.138342, 0.0433118, 0.009104],
    ],
}


@pytest.mark.parametrize('case, eye', [('single', 'right'),
                                       ('multi', 'right'),
                                       ('single', 'left')])
def test_Granley2023_matches_the_reference(case, eye):
    electrodes = _REF_CASES[case]
    # Same visual-field location: current maps flip y from retina to dva.
    implant = _implant({n: (x, -y) for n, x, y, *_ in electrodes}, eye=eye)
    model = Granley2023Model(implant, **_GRID)
    data = model.predict_percept(
        {n: BiphasicPulseTrain(f, a * xTh, p)
         for n, _, _, f, a, p in electrodes}).data[..., 0]
    ref = np.flipud(_REF[f"{case}_{eye[0].upper()}E"])
    # The reference centers each phosphene one pixel below its electrode
    # (off by one in its pixel conversion):
    npt.assert_allclose(data[:-1], ref[1:], rtol=1e-4, atol=1e-6)


# -----------------------------------------------------------------------------
# Equations
# -----------------------------------------------------------------------------

def test_Granley2023_shape_at_reference_pulse():
    # 2 xTh at default a3 gives area rho; 0.45 ms gives eccentricity lam:
    bright, area, ecc = _shape(20, 2, 0.45)
    npt.assert_allclose(bright, 0.4733 * 2 ** 0.5211 + 0.016 * 20, rtol=1e-6)
    npt.assert_allclose(area, 75, rtol=1e-6)
    npt.assert_allclose(ecc, 0.9, rtol=1e-6)


def test_Granley2023_amplitude_cutoff_is_strict():
    npt.assert_equal(_shape(20, 0.25, 0.45)[0], 0)
    npt.assert_equal(_shape(20, 0.2501, 0.45)[0] > 0, True)


def test_Granley2023_area_floor_and_eccentricity_clip():
    npt.assert_equal(_shape(20, 0.01, 0.45)[1], 1)
    # Short pulses raise eccentricity (a4 < 0) up to the clip:
    npt.assert_allclose(_shape(20, 2, 0.01)[2], 0.99, rtol=1e-6)
    npt.assert_allclose(_shape(20, 2, 0.45, lam=-1)[2], 0)


def test_Granley2023_peak_is_the_brightness():
    # An electrode at the fovea sits on a grid point:
    model = Granley2023Model(_implant({'A': (0, 0)}), **_GRID)
    data = model.predict_percept(
        {'A': BiphasicPulseTrain(30, 3 * xTh, 0.45)}).data
    npt.assert_allclose(data.max(), _shape(30, 3, 0.45)[0], rtol=1e-6)
    npt.assert_allclose(data[4, 5, 0], data.max())


def test_Granley2023_sums_electrodes():
    model = Granley2023Model(_implant({'A': (-600, 300), 'B': (700, -200)}),
                             **_GRID)
    a = {'A': BiphasicPulseTrain(20, 2 * xTh, 0.45)}
    b = {'B': BiphasicPulseTrain(60, 3 * xTh, 0.2)}
    npt.assert_allclose(model.predict_percept({**a, **b}).data,
                        model.predict_percept(a).data +
                        model.predict_percept(b).data, rtol=1e-5)


def test_Granley2023_does_not_threshold_the_percept():
    # `thresh_percept` sets the contour, not a cutoff:
    model = Granley2023Model(_implant({'A': (0, 0)}), **_GRID)
    data = model.predict_percept(
        {'A': BiphasicPulseTrain(20, 2 * xTh, 0.45)}).data
    faint = (data > 0) & (data < np.exp(-2))
    npt.assert_equal(np.count_nonzero(faint) > 10, True)


@pytest.mark.parametrize('step', [0.1, 0.2])
def test_Granley2023_rho_is_an_area_in_pixels(step):
    # The `thresh_percept` contour covers `rho` pixels at any `step`, so its
    # angular area scales with step ** 2:
    model = Granley2023Model(_implant({'A': (0, 0)}), xrange=(-8, 8),
                             yrange=(-8, 8), step=step, rho=200)
    data = model.predict_percept(
        {'A': BiphasicPulseTrain(20, 2 * xTh, 0.45)}).data
    inside = np.count_nonzero(data >= np.exp(-2) * data.max())
    npt.assert_allclose(inside, 200, rtol=0.05)


# -----------------------------------------------------------------------------
# Orientation
# -----------------------------------------------------------------------------

def test_Granley2023_orientation_from_known_tangent():
    # Same bundles as test_AxonMapModel_calc_bundle_tangent_fast, where the
    # retinal tangent at the fovea is -0.4819 rad. Watson2014Map flips y, so
    # the visual-field tangent is +0.4819:
    model = Granley2023Model(_implant({'A': (0, 0)}), step=5, n_axons=500,
                             xrange=(-20, 20), yrange=(-15, 15),
                             ax_segments_range=(3, 50))
    model.build()
    npt.assert_almost_equal(model.spatial._el_theta[0], 0.4819 - np.pi / 2,
                            decimal=3)


@pytest.mark.parametrize('eye', ['right', 'left'])
@pytest.mark.parametrize('xy', [(-1200, -1500), (800, -2000)])
def test_Granley2023_major_axis_follows_the_axon_map(eye, xy):
    # The ellipse aligns with the AxonMap streak through the same electrode,
    # in current p2p coordinates. A sign error would be off by ~100 deg here:
    grid = dict(xrange=(-12, 12), yrange=(-12, 12), step=0.4,
                n_ax_segments=300)
    implant = _implant({'A': xy}, eye=eye)
    ellipse = Granley2023Model(implant, rho=50, **grid).predict_percept(
        {'A': BiphasicPulseTrain(20, 2 * xTh, 0.45)})
    streak = AxonMapModel(implant, rho=50, lam=200, meridian_blend=0,
                          **grid).predict_percept({'A': 10})
    diff = _principal_axis(ellipse) - _principal_axis(streak)
    npt.assert_array_less(abs((diff + 90) % 180 - 90), 10)


def test_Granley2023_orient_scale():
    model = Granley2023Model(_implant({'A': (800, -2000)}), xrange=(-4, 10),
                             yrange=(0, 14), step=0.2, rho=200)
    train = {'A': BiphasicPulseTrain(20, 2 * xTh, 0.45)}
    npt.assert_equal(
        abs(_principal_axis(model.predict_percept(train)) - 90) > 45, True)
    # A zero angle puts the major axis along the vertical meridian:
    model.spatial.orient_scale = 0
    npt.assert_allclose(_principal_axis(model.predict_percept(train)), 90,
                        atol=0.5)


# -----------------------------------------------------------------------------
# Stimulus contract
# -----------------------------------------------------------------------------

def _argus(**kwargs):
    return Granley2023Model(ArgusII(**kwargs), **_GRID)


def test_Granley2023_reads_threshold_multiples_or_calibrated_current():
    relative = _argus().predict_percept(
        {'C5': BiphasicPulseTrain(20, 2 * xTh, 0.45)}).data
    npt.assert_equal(np.any(relative), True)
    calibrated = _argus(thresholds=80 * uA).predict_percept(
        {'C5': BiphasicPulseTrain(20, 160 * uA, 0.45)}).data
    npt.assert_allclose(calibrated, relative)


def test_Granley2023_uncalibrated_current_raises():
    with pytest.raises(ValueError, match='threshold'):
        _argus().predict_percept({'C5': BiphasicPulseTrain(20, 160, 0.45)})


@pytest.mark.parametrize('train', [
    MonophasicPulse(-1, 0.45, stim_dur=100),
    BiphasicPulseTrain(20, 1 * xTh, 0.45, delay_dur=1, stim_dur=100),
    BiphasicPulseTrain(20, 1 * xTh, 0.45, cathodic_first=False),
])
def test_Granley2023_rejects_other_pulses(train):
    with pytest.raises(TypeError):
        _argus().predict_percept({'C5': train})


def test_Granley2023_zero_stimulus_is_dark():
    data = _argus().predict_percept(
        {'C5': BiphasicPulseTrain(20, 0 * xTh, 0.45)}).data
    npt.assert_equal(data, 0)


def test_Granley2023_reads_an_encoded_image():
    encoder = AmplitudeEncoder(amp_range=(0 * xTh, 2 * xTh), freq=20 * Hz)
    model = _argus(encoder=encoder)
    npt.assert_equal(np.any(model.predict_percept(samples.logo_bvl()).data),
                     True)


def test_Granley2023_rejects_an_encoded_video():
    model = _argus(thresholds=80)
    video = VideoStimulus(np.random.rand(4, 4, 3), time=[0, 200, 400])
    with pytest.raises(NotImplementedError, match='frames'):
        model.predict_percept(video)


def test_Granley2023_one_representative_frame():
    model = _argus()
    train = {'C5': BiphasicPulseTrain(20, 2 * xTh, 0.45)}
    single = model.predict_percept(train).data
    npt.assert_equal(single.shape[-1], 1)
    frames = model.predict_percept(train, t_percept=[0, 10, 20]).data
    npt.assert_equal(frames.shape[-1], 3)
    npt.assert_allclose(frames[..., :1], single)
    npt.assert_equal(frames[..., 1:], 0)


# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------

def test_Granley2023_rejects_a_temporal_model():
    with pytest.raises(TypeError, match='temporal'):
        _argus().temporal = FadingTemporal()


@pytest.mark.parametrize('params, match', [
    (dict(step=(1, 0.5)), 'square'),
    (dict(thresh_percept=0), 'thresh_percept'),
    (dict(thresh_percept=1), 'thresh_percept'),
])
def test_Granley2023_rejects_invalid_geometry(params, match):
    model = _argus()
    model.spatial.set_params(**params)
    with pytest.raises(ValueError, match=match):
        model.build()


def test_Granley2023_needs_an_eye():
    model = Granley2023Model(GridImplant(shape=(2, 2), spacing=400), **_GRID)
    with pytest.raises(TypeError, match='laterality'):
        model.build()


# -----------------------------------------------------------------------------
# Torch kernel
# -----------------------------------------------------------------------------

def test_Granley2023_kernel_is_differentiable():
    freq = torch.tensor([20.0, 40.0], requires_grad=True)
    amp = torch.tensor([2.0, 1.5], requires_grad=True)
    pdur = torch.tensor([0.3, 0.6], requires_grad=True)
    x, y = torch.meshgrid(torch.arange(-8.0, 9), torch.arange(-8.0, 9),
                          indexing='xy')
    resp = _mvg_response(freq, amp, pdur, x.ravel(), y.ravel(),
                         torch.tensor([-2.0, 3.0]), torch.tensor([1.0, -1.0]),
                         torch.tensor([0.4, -1.1]), thresh_percept=np.exp(-2),
                         **_SHAPE)
    resp.sum().backward()
    for param in (freq, amp, pdur):
        npt.assert_equal(bool(torch.isfinite(param.grad).all()), True)
        npt.assert_equal(bool((param.grad != 0).all()), True)
