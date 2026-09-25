"""Tests for the [Ho2018]_ photovoltaic model"""
import inspect
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.implants import ElectrodeGrid, PointSource
from pulse2percept.implants.retina import (Lorach2015Array, PRIMAPivotal,
                                           RetinalImplant)
from pulse2percept.implants.retina.prima import _PhotovoltaicRetinalImplant
from pulse2percept.models import Model
from pulse2percept.models.base import _electrode_pitch
from pulse2percept.models.retina import (Ho2018Model, Ho2018Spatial,
                                         Ho2018Temporal)
from pulse2percept.models.retina.ho2018 import _radiant_exposure
from pulse2percept.stimuli import (ImageStimulus, PhotovoltaicEncoder,
                                   PRIMAEncoder, Stimulus, VideoStimulus,
                                   samples)
from pulse2percept.stimuli.encoders import (_NormalizedStimulus,
                                            _OpticalStimulus)
from pulse2percept.topography.retina import Watson2014Map
from pulse2percept.units import DimensionMismatchError, dva
from pulse2percept.vision import Scene, Scotoma

#: The stimulation condition [Ho2018]_ normalizes against.
REF = {'irradiance': 9, 'pulse_dur': 4, 'freq': 20, 'wavelength': 880}


class TinyArray(_PhotovoltaicRetinalImplant):
    """A 2x2 photovoltaic array whose 200 um pitch exceeds the default rho"""
    __slots__ = ()

    placement = 'subretinal'

    def __init__(self, shape=(2, 2), spacing=200, encoder=None):
        self.eye = 'right'
        self.preprocess = False
        self.safe_mode = False
        self.encoder = encoder
        self.electrode_array = ElectrodeGrid(shape, spacing,
                                             electrode_type=PointSource)


def tiny_implant(**optics):
    """A small photovoltaic array driven at the [Ho2018]_ settings."""
    return TinyArray(encoder=PhotovoltaicEncoder(**{**REF, **optics}))


def spot(n=4):
    """A picture with one fully lit quadrant."""
    img = np.zeros((n, n))
    img[:n // 2, :n // 2] = 1
    return ImageStimulus(img)


def tiny_model(implant=None, **params):
    """A Ho model on a small grid."""
    kwargs = {'xrange': (-2, 2), 'yrange': (-2, 2), 'step': 0.25,
              'verbose': False, **params}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return Ho2018Model(tiny_implant() if implant is None else implant,
                           **kwargs)


# -- Input contract ---------------------------------------------------------

def test_accepts_an_optical_schedule():
    implant = tiny_implant()
    stim = implant.prepare_stim(spot())
    npt.assert_equal(isinstance(stim, _OpticalStimulus), True)
    percept = tiny_model(implant).predict_percept(spot())
    npt.assert_equal(percept.data.ndim, 3)
    npt.assert_equal(np.any(percept.data > 0), True)


def test_rejects_current():
    array = ElectrodeGrid((2, 2), 200, electrode_type=PointSource)
    model = tiny_model(RetinalImplant(array))
    with pytest.raises(DimensionMismatchError) as excinfo:
        model.predict_percept({'A1': 20})
    npt.assert_equal('irradiance' in str(excinfo.value), True)


def test_rejects_gray_levels():
    # Checked on the prepared stimulus: as a *source*, the encoder would
    # read the drive as a picture and re-encode it.
    implant = tiny_implant()
    drive = implant.prepare_stim(spot())._spatial_view()
    npt.assert_equal(drive._is_normalized_drive, True)
    with pytest.raises(DimensionMismatchError) as excinfo:
        tiny_model(implant)._predict_percept(drive)
    npt.assert_equal('irradiance' in str(excinfo.value), True)


def test_rejects_bare_irradiance():
    # Irradiance samples without a schedule carry no ON duration.
    implant = tiny_implant()
    bare = Stimulus(implant.prepare_stim(spot()))
    npt.assert_equal(bare._has_spatial_view, False)
    with pytest.raises(TypeError) as excinfo:
        tiny_model(implant).predict_percept(bare)
    npt.assert_equal('schedule' in str(excinfo.value), True)


def test_temporal_stage_rejects_current():
    temporal = Ho2018Temporal(verbose=False)
    npt.assert_equal(temporal.stimulus_unit.dimension.is_dimensionless, True)
    current = Stimulus(np.array([[20.0, 0.0]]), electrodes=['A1'],
                       time=[0, 50])
    with pytest.raises(DimensionMismatchError) as excinfo:
        temporal.predict_percept(current)
    npt.assert_equal('dimensionless' in str(excinfo.value), True)


def test_temporal_stage_rejects_gray_levels():
    # Dimensionless, so the unit alone does not disqualify it. Time-varying
    # on purpose: a still image would be refused for lacking a time axis.
    temporal = Ho2018Temporal(verbose=False)
    video = VideoStimulus(np.zeros((2, 2, 4)), time=np.arange(4) * 50.0)
    npt.assert_equal(video.unit.dimension.is_dimensionless, True)
    npt.assert_equal(video._is_normalized_drive, False)
    with pytest.raises(DimensionMismatchError) as excinfo:
        temporal.predict_percept(video, t_percept=np.arange(4) * 50.0)
    npt.assert_equal('gray levels' in str(excinfo.value), True)


def test_temporal_stage_takes_a_normalized_drive():
    drive = _NormalizedStimulus(np.ones((2, 4)), electrodes=['a', 'b'],
                                time=np.arange(4) * 50.0)
    percept = Ho2018Temporal(verbose=False).predict_percept(
        drive, t_percept=np.arange(4) * 50.0)
    npt.assert_equal(np.any(percept.data > 0), True)


def test_temporal_stage_takes_the_spatial_percept():
    implant = tiny_implant()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        spatial = Ho2018Spatial(implant, xrange=(-2, 2), yrange=(-2, 2),
                                step=0.25, verbose=False)
    drive = spatial.predict_percept(spot())
    percept = Ho2018Temporal(verbose=False).predict_percept(drive)
    npt.assert_equal(np.any(percept.data > 0), True)


# -- Structured stimulus routing --------------------------------------------

def test_spatial_stage_receives_the_schedule(monkeypatch):
    seen = []
    model = tiny_model()
    original = Ho2018Spatial._predict_prepared

    def spy(self, stim, t_percept=None):
        seen.append(stim)
        return original(self, stim, t_percept=t_percept)

    monkeypatch.setattr(Ho2018Spatial, '_predict_prepared', spy)
    model.predict_percept(spot())
    npt.assert_equal(len(seen), 1)
    npt.assert_equal(isinstance(seen[0], _OpticalStimulus), True)


def test_does_not_render_the_waveform(monkeypatch):
    rendered = []
    monkeypatch.setattr(_OpticalStimulus, '_render',
                        lambda self: rendered.append(1))
    tiny_model().predict_percept(spot())
    npt.assert_equal(rendered, [])


def test_granley_still_gets_the_structured_stimulus():
    from pulse2percept.models.retina.granley2021 import _BiphasicSpatialMixin
    npt.assert_equal(_BiphasicSpatialMixin._needs_structured_stim, True)
    npt.assert_equal(Ho2018Spatial._needs_structured_stim, True)
    from pulse2percept.models.retina import ScoreboardSpatial
    npt.assert_equal(ScoreboardSpatial._needs_structured_stim, False)


# -- Optical parameter preservation -----------------------------------------

@pytest.mark.parametrize('optics, factor', [
    ({'irradiance': 18}, 2.0),
    ({'pulse_dur': 8}, 2.0),
    ({'irradiance': 4.5, 'pulse_dur': 8}, 1.0),
])
def test_activation_follows_radiant_exposure(optics, factor):
    ref = tiny_model(tiny_implant()).predict_percept(spot()).data.max()
    got = tiny_model(tiny_implant(**optics)).predict_percept(spot()).data.max()
    npt.assert_almost_equal(got / ref, factor, decimal=4)


def test_uses_the_schedule_not_the_spatial_view():
    # `_spatial_view` normalizes irradiance x duty cycle away, so doubling
    # both leaves it unchanged.
    implant = tiny_implant()
    brighter = tiny_implant(irradiance=18, pulse_dur=8)
    npt.assert_almost_equal(
        implant.prepare_stim(spot())._spatial_view().data.max(),
        brighter.prepare_stim(spot())._spatial_view().data.max())
    npt.assert_array_less(
        tiny_model(implant).predict_percept(spot()).data.max(),
        tiny_model(brighter).predict_percept(spot()).data.max())


def test_reference_condition_gives_unit_drive():
    stim = tiny_implant().prepare_stim(spot())
    npt.assert_almost_equal(_radiant_exposure(stim).max(), 1.0)
    npt.assert_almost_equal(_radiant_exposure(stim).min(), 0.0)
    half = tiny_implant(pulse_dur=2).prepare_stim(spot())
    npt.assert_almost_equal(_radiant_exposure(half).max(), 0.5)


# -- Spatial response -------------------------------------------------------

def test_default_rho_is_half_the_electrode_pitch():
    # A pulse2percept convention, resolved against the bound implant when the
    # model is built -- not the receptive-field size [Ho2018]_ reports.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        spatial = Ho2018Spatial(PRIMAPivotal(), xrange=(-1, 1),
                                yrange=(-1, 1), step=0.5, verbose=False)
        npt.assert_equal(spatial.rho, None)
        spatial.build()
        model = Ho2018Model(PRIMAPivotal(), xrange=(-1, 1), yrange=(-1, 1),
                            step=0.5, verbose=False).build()
    pitch = _electrode_pitch(spatial)
    npt.assert_almost_equal(pitch, 100, decimal=6)
    npt.assert_almost_equal(spatial.rho, pitch / 2)
    npt.assert_almost_equal(model.spatial.rho, pitch / 2)


def test_explicit_rho_is_left_alone():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        spatial = Ho2018Spatial(PRIMAPivotal(), rho=250, xrange=(-1, 1),
                                yrange=(-1, 1), step=0.5,
                                verbose=False).build()
        model = Ho2018Model(PRIMAPivotal(), rho=250, xrange=(-1, 1),
                            yrange=(-1, 1), step=0.5, verbose=False).build()
    npt.assert_almost_equal(spatial.rho, 250)
    npt.assert_almost_equal(model.spatial.rho, 250)


def test_rho_re_resolves_when_the_implant_changes():
    spatial = Ho2018Spatial(tiny_implant(), xrange=(-1, 1), yrange=(-1, 1),
                            step=0.5, verbose=False).build()
    npt.assert_almost_equal(spatial.rho, 100)
    spatial.implant = TinyArray(spacing=400,
                                encoder=PhotovoltaicEncoder(**REF))
    spatial.build()
    npt.assert_almost_equal(spatial.rho, 200)


def test_rho_needs_a_pitch_or_a_value():
    # One electrode has no nearest neighbor to measure a pitch against.
    single = TinyArray(shape=(1, 1), encoder=PhotovoltaicEncoder(**REF))
    with pytest.raises(ValueError) as excinfo:
        Ho2018Spatial(single, xrange=(-1, 1), yrange=(-1, 1), step=0.5,
                      verbose=False).build()
    npt.assert_equal('pitch' in str(excinfo.value), True)
    # An explicit rho is all it takes:
    spatial = Ho2018Spatial(single, rho=97.5, xrange=(-1, 1), yrange=(-1, 1),
                            step=0.5, verbose=False).build()
    npt.assert_almost_equal(spatial.rho, 97.5)


def test_spatial_profile_is_a_gaussian_of_rho():
    # Read the width off the kernel: `Percept.measure` reports an FWHM-like
    # extent, which is not how [Ho2018]_ defines a receptive field.
    implant = TinyArray(shape=(1, 1),
                        encoder=PhotovoltaicEncoder(**REF))
    rho = 97.5
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        model = Ho2018Spatial(implant, rho=rho, xrange=(0, 2), yrange=(0, 0),
                              step=0.05, verbose=False).build()
    percept = model.predict_percept(ImageStimulus(np.ones((1, 1))))
    profile = percept.data[0, :, 0]
    xret, _ = Watson2014Map().dva_to_ret(model.grid.x[0, :],
                                         model.grid.y[0, :])
    expected = profile[0] * np.exp(-xret ** 2 / (2 * rho ** 2))
    npt.assert_allclose(profile, expected, rtol=1e-4)


def test_spatial_drive_is_zero_outside_the_schedule():
    # Zero-order hold holds *within* the schedule; before the first pulse and
    # after the stimulus ends nothing is delivered.
    implant = tiny_implant()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        spatial = Ho2018Spatial(implant, xrange=(-2, 2), yrange=(-2, 2),
                                step=0.25, verbose=False)
    stim = implant.prepare_stim(spot())
    last = stim.pulse_time[-1]
    t = np.array([-20.0, stim.pulse_time[0], last, stim.duration,
                  stim.duration + 200.0])
    drive = spatial.predict_percept(spot(), t_percept=t).data.max(axis=(0, 1))
    npt.assert_almost_equal(drive[0], 0)
    npt.assert_array_less(0, drive[1])
    npt.assert_array_less(0, drive[2])
    npt.assert_almost_equal(drive[3], 0)
    npt.assert_almost_equal(drive[4], 0)


def test_spatial_drive_is_zero_before_a_delayed_first_pulse():
    # A source whose time axis starts late puts the first pulse after t=0, so
    # "before the schedule" is not the same as "negative time".
    implant = tiny_implant()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        spatial = Ho2018Spatial(implant, xrange=(-2, 2), yrange=(-2, 2),
                                step=0.25, verbose=False)
    video = VideoStimulus(np.ones((2, 2, 3)),
                          time=np.array([100.0, 150.0, 200.0]))
    stim = implant.prepare_stim(video)
    npt.assert_almost_equal(stim.pulse_time[0], 100.0)
    t = np.array([0.0, 99.0, 100.0, stim.duration])
    drive = spatial.predict_percept(video, t_percept=t).data.max(axis=(0, 1))
    npt.assert_almost_equal(drive[0], 0)
    npt.assert_almost_equal(drive[1], 0)
    npt.assert_array_less(0, drive[2])
    npt.assert_almost_equal(drive[3], 0)


def test_warns_about_ignored_electrode_distance():
    implant = tiny_implant()
    for electrode in implant.electrode_array.electrode_objects:
        electrode.z = 100
    model = tiny_model(implant)
    with pytest.warns(UserWarning, match='electrode-retina distance'):
        model.predict_percept(spot())


# -- Temporal response ------------------------------------------------------

def kernel(temporal, dt=0.05, stop=400):
    """Return sample times (ms) and the normalized impulse response."""
    t = np.arange(0, stop, dt)
    return t, temporal.impulse_response(t) * temporal._gain


def test_impulse_response_matches_the_reported_landmarks():
    # RCS pON: 50 +/- 3 ms to first peak, 94 +/- 5 ms to zero crossing.
    temporal = Ho2018Temporal(verbose=False).build()
    t, h = kernel(temporal)
    npt.assert_allclose(t[np.argmax(h)], 50.0, atol=1.0)
    npt.assert_almost_equal(h.max(), 1.0, decimal=3)
    after = np.flatnonzero((h[:-1] > 0) & (h[1:] <= 0) & (t[:-1] > 40))
    npt.assert_allclose(t[after[0]], 94.0, atol=1.0)


def test_impulse_response_is_biphasic_and_dc_free():
    temporal = Ho2018Temporal(verbose=False).build()
    t, h = kernel(temporal, stop=2000)
    npt.assert_array_less(h.min(), -0.1)
    # Zero DC gain makes the filter purely transient; the residual is
    # coefficient rounding plus the truncated tail.
    npt.assert_array_less(abs(np.trapezoid(h, t)),
                          0.01 * np.trapezoid(abs(h), t))


def test_response_adapts_to_a_sustained_pulse_train():
    model = tiny_model()
    percept = model.predict_percept(spot())
    peaks = percept.data.max(axis=(0, 1))
    npt.assert_equal(peaks.size > 4, True)
    # Every pulse period delivers the same drive, yet the response decays.
    npt.assert_array_less(peaks[-1], 0.2 * peaks.max())
    npt.assert_array_less(0, peaks.max())


def test_automatic_output_times_are_exact():
    # `reduce='peak'` would subsample each interval eight times; at 20 Hz that
    # underestimates the true peak by tens of percent, so the default reports
    # the instant it actually computes.
    model = tiny_model()
    npt.assert_equal(model.temporal.reduce, 'last')
    npt.assert_equal(Ho2018Temporal().reduce, 'last')
    auto = model.predict_percept(spot())
    asked = model.predict_percept(spot(), t_percept=auto.time)
    npt.assert_allclose(auto.data, asked.data, rtol=1e-6, atol=1e-7)


def test_response_is_never_negative():
    percept = tiny_model().predict_percept(spot())
    npt.assert_equal(np.all(percept.data >= 0), True)


# -- Temporal input semantics -----------------------------------------------

def test_static_image_produces_a_pulse_train_response():
    # A static image is still delivered as repeated optical pulses.
    model = tiny_model()
    percept = model.predict_percept(spot())
    period = 1e3 / REF['freq']
    npt.assert_equal(percept.time.size > 1, True)
    npt.assert_allclose(np.diff(percept.time), period, rtol=1e-3)
    frames = np.round(percept.data.max(axis=(0, 1)), 6)
    npt.assert_equal(np.unique(frames).size > 1, True)
    # A still has no source-video frames to report against.
    npt.assert_equal('source_frame_time' in percept.metadata, False)


def test_video_produces_a_changing_response():
    frames = np.zeros((4, 4, 6))
    frames[:2, :2, :3] = 1
    frames[2:, 2:, 3:] = 1
    video = VideoStimulus(frames, time=np.arange(6) * 50.0)
    percept = tiny_model().predict_percept(video)
    top_left = percept.data[:8, :8, :].max(axis=(0, 1))
    bottom_right = percept.data[-8:, -8:, :].max(axis=(0, 1))
    npt.assert_array_less(bottom_right[2], top_left[2])
    npt.assert_array_less(top_left[-1], bottom_right[-1])


def prima_model():
    """A Ho model of PRIMA (30 Hz projector) on a coarse grid."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return Ho2018Model(PRIMAPivotal(), xrange=(-2, 2), yrange=(-2, 2),
                           step=0.5, verbose=False)


@pytest.mark.parametrize('fps', [15, 24.52, 29.97, 30, 60])
def test_video_reports_on_the_source_clock(fps):
    # The projector clock sets stimulation; the source clock sets reporting.
    n = 12
    frames = np.zeros((8, 8, n))
    frames[..., ::2] = 1
    video = VideoStimulus(frames, time=np.arange(n) * (1e3 / fps))
    model = prima_model()
    stim = model.implant.prepare_stim(video)
    npt.assert_allclose(np.diff(stim.pulse_time), 1e3 / 30, atol=0.01)
    # Drive stays on the pulse clock:
    drive = model.spatial.predict_percept(video)
    npt.assert_allclose(drive.time, stim.pulse_time)
    percept = model.predict_percept(video)
    npt.assert_equal(percept.time.size, n)
    npt.assert_allclose(percept.time, video.time + 1e3 / fps,
                        atol=n * model.temporal.dt)
    npt.assert_allclose(percept.metadata['source_frame_time'], video.time)
    # Automatic output equals explicit evaluation at the same times:
    asked = model.predict_percept(video, t_percept=percept.time)
    npt.assert_allclose(asked.data, percept.data, rtol=1e-6, atol=1e-7)
    npt.assert_equal('source_frame_time' in asked.metadata, False)
    # Explicit `t_percept` always wins:
    npt.assert_almost_equal(
        model.predict_percept(video, t_percept=[0, 10, 20]).time, [0, 10, 20])


def test_pedestrian_scene_reports_on_the_video_clock():
    # 24.52 Hz source through the 30 Hz PRIMA projector.
    video = samples.ucsb_pedestrians(resize=(43, 80))
    scene = Scene(video, fov=40 * dva, scotoma=Scotoma.circle(5 * dva),
                  scotoma_fill=0)
    percept = prima_model().predict_percept(scene, gaze=(0, 0) * dva)
    rendered = scene.render(percept=percept, gaze=(0, 0) * dva,
                            vmax=percept.data.max())
    npt.assert_equal(rendered.data.shape[-1], video.time.size)
    npt.assert_allclose(rendered.time, percept.time)
    npt.assert_equal(percept.time.size, video.time.size)
    npt.assert_allclose(percept.metadata['source_frame_time'], video.time)
    # Last output closes the last source frame, not the last pulse period:
    npt.assert_allclose(percept.time[-1],
                        video.time[-1] + np.diff(video.time).mean(), atol=0.5)


# -- What the wrapper forwards ----------------------------------------------

def test_model_does_not_quantize_the_retinal_drive():
    # `n_gray` would quantize drive before temporal filtering.
    npt.assert_equal('n_gray' in inspect.signature(Ho2018Model).parameters,
                     False)
    npt.assert_equal('n_gray' in inspect.signature(Ho2018Spatial).parameters,
                     True)
    npt.assert_equal(tiny_model().spatial.n_gray, None)


def test_threshold_applies_to_brightness_only():
    # Thresholding drive first would delete what repeated pulses sum to.
    model = tiny_model(thresh_percept=0.5)
    npt.assert_almost_equal(model.spatial.thresh_percept, 0)
    npt.assert_almost_equal(model.temporal.thresh_percept, 0.5)
    data = model.predict_percept(spot()).data
    weak = data[(data > 0)]
    npt.assert_equal(np.all(weak >= 0.5), True)


# -- Devices and scenes -----------------------------------------------------

def test_prima_end_to_end():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        model = Ho2018Model(PRIMAPivotal(), xrange=(-2, 2), yrange=(-2, 2),
                            step=0.25, verbose=False)
    npt.assert_equal(isinstance(model.implant.encoder, PRIMAEncoder), True)
    percept = model.predict_percept(spot(8))
    npt.assert_equal(percept.data.shape[:2], model.spatial.grid.x.shape)
    npt.assert_equal(np.any(percept.data > 0), True)


def test_works_on_a_non_prima_photovoltaic_array():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        model = Ho2018Model(Lorach2015Array(), xrange=(-2, 2),
                            yrange=(-2, 2), step=0.25, verbose=False).build()
    percept = model.predict_percept(spot(8))
    npt.assert_equal(np.any(percept.data > 0), True)


def test_scene_prediction():
    scene = Scene(spot(16), fov=(8, 8))
    percept = tiny_model().predict_percept(scene)
    npt.assert_equal(percept.data.ndim, 3)
    npt.assert_equal(np.any(percept.data > 0), True)


# -- Composition ------------------------------------------------------------

def test_components_compose_by_hand():
    implant = tiny_implant()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        spatial = Ho2018Spatial(implant, xrange=(-2, 2), yrange=(-2, 2),
                                step=0.25, verbose=False)
    model = Model(spatial=spatial, temporal=Ho2018Temporal(verbose=False))
    npt.assert_allclose(model.predict_percept(spot()).data,
                        tiny_model(implant).predict_percept(spot()).data)


def test_spatial_only_reports_on_the_pulse_clock():
    implant = tiny_implant()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        spatial = Ho2018Spatial(implant, xrange=(-2, 2), yrange=(-2, 2),
                                step=0.25, verbose=False)
    percept = spatial.predict_percept(spot())
    npt.assert_allclose(percept.time,
                        implant.prepare_stim(spot()).pulse_time)
