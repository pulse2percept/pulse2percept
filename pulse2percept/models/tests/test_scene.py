"""The model as the glue between a scene and an implant (#668)

Registration lives here, not in the implant and not in the encoder: the model
is the only object that holds a retinotopy *and* is handed an implant.

Scenes are laid out one degree per pixel with an odd pixel count, so that
pixel centers land on whole degrees and the center pixel sits on the origin.
Expected values are then plain arithmetic.
"""
import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.implants import (ElectrodeGrid, PointSource,
                                    ProsthesisSystem)
from pulse2percept.models import (Model, NotBuiltError, ScoreboardModel,
                                  ScoreboardSpatial)
from pulse2percept.models.base import _scene_driven_implant
from pulse2percept.models.cortex import ScoreboardModel as CortexScoreboard
from pulse2percept.percepts import Percept
from pulse2percept.stimuli import (AmplitudeEncoder, BiphasicPulse,
                                   ImageStimulus, VideoStimulus)
from pulse2percept.topography import Curcio1990Map, RetinalMap, Watson2014Map
from pulse2percept.units import dva, ms, s
from pulse2percept.vision import Scene, Scotoma

#: A square scene laid out so that one pixel is exactly one degree and the
#: center pixel sits on the origin.
SCENE_PX = 41
HALF = (SCENE_PX - 1) // 2

#: Gray level 1 maps onto this many microamps, so an electrode's amplitude
#: reads back as the gray level it was given.
AMP_MAX = 100.0


class SquareMap(RetinalMap):
    """A retinal map that is neither 280 um/dva nor linear

    Retinal x grows as the square of eccentricity, so a registration that
    assumes any fixed micron-per-degree ratio lands in the wrong place.
    """

    def dva_to_ret(self, xdva, ydva):
        return np.sign(xdva) * 100.0 * xdva ** 2, -100.0 * ydva

    def ret_to_dva(self, xret, yret):
        return np.sign(xret) * np.sqrt(np.abs(xret) / 100.0), -yret / 100.0


def ramp_source():
    """A picture whose gray level reads off x: 0 at -20 dva, 1 at +20"""
    return ImageStimulus(np.tile(np.linspace(0, 1, SCENE_PX), (SCENE_PX, 1)))


def ramp_at(x_dva):
    """What `ramp_source` shows at a scene x"""
    return (x_dva + HALF) / (2 * HALF)


def scene_of(source=None, **kwargs):
    return Scene(ramp_source() if source is None else source,
                 fov=(SCENE_PX, SCENE_PX), **kwargs)


def implant_at(x_um=0, y_um=0, encoder=True):
    """An implant whose single electrode sits where we want to look"""
    return ProsthesisSystem(
        PointSource(x_um, y_um, 0),
        encoder=AmplitudeEncoder(amp_range=(0, AMP_MAX)) if encoder else None)


def model_for(scene, **kwargs):
    # An explicit `vfmap`: the retinotopy is what the expected values below
    # are computed through, so it cannot be left to a default.
    params = {'rho': 200, 'xrange': (-3, 3), 'yrange': (-3, 3), 'step': 1,
              'vfmap': Curcio1990Map()}
    params.update(kwargs)
    return ScoreboardModel(scene=scene, **params).build()


def seen_by(model, implant, gaze=None):
    """The gray level each electrode was handed, in electrode order

    Read back off the amplitudes the encoder produced, which is the only place
    the sampled scene shows up once registration is over.
    """
    trial = _scene_driven_implant(model, implant, gaze)
    view = trial.stim._spatial_view()
    return np.asarray(view.data, dtype=float).reshape(
        (len(trial.electrode_names), -1)) / AMP_MAX


def test_the_model_supplies_its_own_vfmap():
    """The caller never names a retinotopy; the model already has one"""
    scene = scene_of()
    implant = implant_at(*Curcio1990Map().dva_to_ret(6.0, 0.0))
    npt.assert_almost_equal(seen_by(model_for(scene), implant),
                            [[ramp_at(6.0)]], decimal=4)
    # Give the model a different retinotopy and the same electrode reads a
    # different part of the scene, with nothing else changing and nothing
    # about the map appearing at the call site:
    watson = model_for(scene, vfmap=Watson2014Map())
    npt.assert_equal(np.allclose(seen_by(watson, implant), ramp_at(6.0)),
                     False)


@pytest.mark.parametrize('x_dva', [-8.0, 2.5, 7.0])
def test_a_nonlinear_retinal_map_still_registers(x_dva):
    """Not 280 um/dva, and not linear either

    The electrode is placed at the retinal image of a known visual-field
    position, so the pixel it should see is known independently of the map.
    """
    vfmap = SquareMap()
    implant = implant_at(*vfmap.dva_to_ret(x_dva, 0.0))
    model = model_for(scene_of(), vfmap=vfmap)
    npt.assert_almost_equal(seen_by(model, implant), [[ramp_at(x_dva)]],
                            decimal=4)


def test_gaze_moves_the_scene_past_the_implant():
    """Gaze is the scene point on the fovea, so scene = visual field + gaze"""
    model = model_for(scene_of())
    implant = implant_at(0, 0)
    for gaze_x in (-4.0, 0.0, 6.0):
        npt.assert_almost_equal(seen_by(model, implant, gaze=(gaze_x, 0)),
                                [[ramp_at(gaze_x)]], decimal=4)
    # Units at the boundary, and the same answer through them:
    npt.assert_almost_equal(seen_by(model, implant, gaze=(6, 0) * dva),
                            seen_by(model, implant, gaze=(6.0, 0.0)))
    # Several electrodes keep their separation in the visual field whatever
    # the gaze: shifting gaze shifts what all of them see by the same amount.
    grid = ProsthesisSystem(ElectrodeGrid((1, 3), 280),
                            encoder=AmplitudeEncoder(amp_range=(0, AMP_MAX)))
    here = seen_by(model, grid).ravel()
    there = seen_by(model, grid, gaze=(2, 0)).ravel()
    npt.assert_almost_equal(np.diff(here), np.diff(there), decimal=4)
    npt.assert_almost_equal(there - here, ramp_at(2) - ramp_at(0), decimal=4)


def test_y_orientation_survives_the_map():
    """Row 0 of the scene is +y in the visual field, both sides of the map"""
    data = np.tile(np.linspace(0, 1, SCENE_PX).reshape((-1, 1)),
                   (1, SCENE_PX))
    model = model_for(scene_of(ImageStimulus(data)))
    vfmap = Curcio1990Map()
    for y_dva in (5.0, -5.0):
        implant = implant_at(*vfmap.dva_to_ret(0.0, y_dva))
        npt.assert_almost_equal(seen_by(model, implant),
                                [[(HALF - y_dva) / (2 * HALF)]], decimal=4)


def test_color_becomes_luminance_only_at_the_device():
    """The scene stays RGB; the electrode gets one number"""
    rgb = np.zeros((SCENE_PX, SCENE_PX, 3))
    rgb[..., 0] = 1.0  # pure red everywhere
    scene = scene_of(ImageStimulus(rgb))
    npt.assert_almost_equal(scene._sample_at(0.0, 0.0)[0, :, 0], [1, 0, 0],
                            decimal=5)
    # ... and the luminance of pure red reaches the implant:
    npt.assert_almost_equal(seen_by(model_for(scene), implant_at(0, 0)),
                            [[0.2125]], decimal=3)


def test_scene_driven_prediction_leaves_the_implant_alone():
    """Predicting what someone sees is a question, not an assignment"""
    model = model_for(scene_of())
    implant = implant_at(0, 0)
    npt.assert_equal(implant.stim, None)
    model.predict_percept(implant)
    npt.assert_equal(implant.stim, None)
    # ... and an implant that already had a stimulus keeps exactly that one:
    implant.stim = BiphasicPulse(20, 0.45)
    before = implant.stim.data.copy()
    model.predict_percept(implant)
    npt.assert_array_equal(implant.stim.data, before)


def test_a_scene_driven_stimulus_still_goes_through_the_device():
    """The stand-in implant is assigned to, not written behind

    Encoding, the safety checks and whatever else an implant does to a
    stimulus are the device's business and must not be skipped just because
    the values came from a scene rather than from the caller.
    """
    grid = ProsthesisSystem(ElectrodeGrid((1, 3), 280),
                            encoder=AmplitudeEncoder(amp_range=(0, AMP_MAX)))
    grid.deactivate('A2')
    trial = _scene_driven_implant(model_for(scene_of()), grid, None)
    npt.assert_equal('A2' in list(trial.stim.electrodes), False)
    npt.assert_equal(len(trial.stim.electrodes), 2)
    # It is a current by the time it is stored, which is the encoder having
    # run inside the setter:
    npt.assert_equal(trial.stim.unit, grid.stimulus_unit)


def test_a_scene_needs_an_encoder_and_a_retina():
    """Both failures name what is missing rather than dying downstream"""
    scene = scene_of()
    with pytest.raises(ValueError) as excinfo:
        model_for(scene).predict_percept(implant_at(0, 0, encoder=False))
    npt.assert_equal('encoder' in str(excinfo.value), True)
    # A cortical model has no retinotopy to follow an electrode out along:
    cortical = CortexScoreboard(scene=scene, rho=200, xrange=(-3, 3),
                                yrange=(-3, 3), step=1).build()
    with pytest.raises(ValueError) as excinfo:
        cortical.predict_percept(implant_at(0, 0))
    npt.assert_equal('vfmap' in str(excinfo.value), True)
    # ... and neither has a temporal-only model:
    from pulse2percept.models import Nanduri2012Temporal
    temporal = Model(temporal=Nanduri2012Temporal(), scene=scene).build()
    with pytest.raises(ValueError):
        temporal.predict_percept(implant_at(0, 0))


def test_an_unbuilt_model_says_so_before_it_samples_anything():
    """The oldest mistake is reported first, not after two newer ones"""
    unbuilt = ScoreboardModel(scene=scene_of(), rho=200, xrange=(-3, 3),
                              yrange=(-3, 3), step=1)
    with pytest.raises(NotBuiltError):
        unbuilt.predict_percept(implant_at(0, 0))
    # ... including when there is a newer mistake waiting behind it:
    with pytest.raises(NotBuiltError):
        unbuilt.predict_percept(implant_at(0, 0, encoder=False))


def test_a_scene_must_be_a_scene():
    with pytest.raises(TypeError):
        ScoreboardModel(scene=ramp_source())


def test_without_a_scene_nothing_changes():
    """The legacy path is untouched, and scene arguments are refused"""
    implant = implant_at(0, 0)
    implant.stim = BiphasicPulse(20, 0.45)
    plain = ScoreboardModel(rho=200, xrange=(-3, 3), yrange=(-3, 3),
                            step=1).build()
    npt.assert_equal(plain.scene, None)
    percept = plain.predict_percept(implant)
    npt.assert_equal(percept.is_rgb, False)
    npt.assert_equal(percept.data.ndim, 3)
    for kwargs in ({'gaze': (1, 0)}, {'vmax': 20}, {'vmin': 3}):
        with pytest.raises(ValueError):
            plain.predict_percept(implant, **kwargs)
    # A model with no stimulus at all still says nothing rather than raising:
    npt.assert_equal(plain.predict_percept(implant_at(0, 0)), None)


def test_a_scene_without_a_scotoma_returns_the_prosthetic_percept():
    """Nothing is lost, so there is nothing to compose into"""
    model = model_for(scene_of())
    percept = model.predict_percept(implant_at(0, 0))
    npt.assert_equal(percept.is_rgb, False)
    npt.assert_equal(percept.data.ndim, 3)
    # On the model's grid, not the scene's:
    npt.assert_equal(percept.shape[:2], model.spatial.grid.shape)
    # `vmax` is meaningless here and is simply unused:
    npt.assert_array_equal(
        model.predict_percept(implant_at(0, 0), vmax=20).data, percept.data)


def test_a_scene_with_a_scotoma_returns_a_composed_rgb_percept():
    scene = scene_of(scotoma=Scotoma.circle(6), scotoma_fill=0.0)
    percept = model_for(scene).predict_percept(implant_at(0, 0), vmax=20)
    npt.assert_equal(percept.is_rgb, True)
    npt.assert_equal(percept.shape, (SCENE_PX, SCENE_PX, 3, 1))
    npt.assert_equal(percept.data.min() >= 0, True)
    npt.assert_equal(percept.data.max() <= 1, True)
    # Reported on the scene's grid, in scene coordinates:
    npt.assert_almost_equal(percept.xdva, np.arange(-HALF, HALF + 1),
                            decimal=4)


def test_vmax_is_required_for_a_composed_percept():
    scene = scene_of(scotoma=Scotoma.circle(6))
    model = model_for(scene)
    with pytest.raises(ValueError) as excinfo:
        model.predict_percept(implant_at(0, 0))
    npt.assert_equal('vmax' in str(excinfo.value), True)


def test_the_intact_periphery_is_the_scene_exactly():
    """Outside the scotoma nothing is resampled, blended or rounded"""
    rng = np.random.default_rng(0)
    rgb = ImageStimulus(rng.random((SCENE_PX, SCENE_PX, 3)))
    scene = scene_of(rgb, scotoma=Scotoma.circle(6))
    percept = model_for(scene).predict_percept(implant_at(0, 0), vmax=20)
    source = rgb.data.reshape((SCENE_PX, SCENE_PX, 3))
    x, y = scene._pixel_centers()
    intact = scene.scotoma(x, y) == 0
    npt.assert_array_equal(percept.data[..., 0][intact], source[intact])


def test_the_phosphene_lands_where_the_electrode_looks():
    """Position and y orientation, all the way through the composed result"""
    vfmap = Curcio1990Map()
    scene = scene_of(scotoma=Scotoma.circle(12), scotoma_fill=0.0)
    model = model_for(scene, rho=80, xrange=(-8, 8), yrange=(-8, 8), step=0.5)
    for x_dva, y_dva in [(4.0, 0.0), (0.0, 4.0), (-4.0, 0.0), (0.0, -4.0)]:
        implant = implant_at(*vfmap.dva_to_ret(x_dva, y_dva))
        frame = model.predict_percept(implant, vmax=2).data[..., 0]
        row, col = int(round(HALF - y_dva)), int(round(x_dva + HALF))
        # Brightest where the electrode looks, dark on the opposite side:
        npt.assert_equal(frame[row, col].mean() > 0.5, True)
        npt.assert_equal(frame[SCENE_PX - 1 - row,
                               SCENE_PX - 1 - col].mean() < 0.1, True)


def test_gaze_moves_the_scotoma_and_the_phosphene_together():
    scene = scene_of(scotoma=Scotoma.circle(4), scotoma_fill=0.3)
    model = model_for(scene, rho=100, xrange=(-2, 2), yrange=(-2, 2),
                      step=0.5)
    implant = implant_at(0, 0)
    fixating = model.predict_percept(implant, vmax=2).data[..., 0]
    shifted = model.predict_percept(implant, gaze=(5, 0) * dva,
                                    vmax=2).data[..., 0]
    # The whole eye-centered pair travelled 5 degrees right across the scene:
    npt.assert_almost_equal(shifted[HALF, HALF + 5], fixating[HALF, HALF],
                            decimal=5)
    # ... and where the eye used to point is native vision again:
    source = scene.source.data.reshape((SCENE_PX, SCENE_PX))
    npt.assert_almost_equal(shifted[HALF, HALF], [source[HALF, HALF]] * 3,
                            decimal=5)


def test_a_fixed_vmax_does_not_renormalize_when_gaze_changes():
    """Nothing rescales the display behind the user's back

    The two gazes land the implant on very different parts of the scene, so
    the brightest phosphene differs. Under a fixed `vmax` the displayed peaks
    must differ too; auto-normalization would pin both at white.
    """
    scene = scene_of(scotoma=Scotoma.circle(12), scotoma_fill=0.0)
    model = model_for(scene, rho=100, xrange=(-4, 4), yrange=(-4, 4),
                      step=0.5)
    implant = implant_at(0, 0)

    def phosphene(gaze_x, **kwargs):
        """The composed pixel the foveal electrode paints

        The scotoma travels with the eye, so the fovea sits at scene x =
        ``gaze_x``. That pixel is pure phosphene; the intact periphery would
        otherwise dominate any whole-frame maximum.
        """
        percept = model.predict_percept(implant, gaze=(gaze_x, 0) * dva,
                                        **kwargs)
        return float(percept.data[HALF, HALF + gaze_x, 0, 0])

    dim, bright = phosphene(-16, vmax=200), phosphene(16, vmax=200)
    npt.assert_equal(0 < dim < bright < 1, True)
    # The ramp is 9x brighter at +16 than at -16, and that is what survives:
    npt.assert_almost_equal(bright / dim, ramp_at(16) / ramp_at(-16),
                            decimal=2)
    # A different `vmax` rescales both, which is what says it is the only
    # thing deciding the mapping:
    npt.assert_almost_equal(phosphene(16, vmax=400), bright / 2, decimal=3)


def test_a_video_scene_keeps_its_own_timing():
    """Native frames stay native, and the percept is read at their times"""
    frames = np.stack([np.full((SCENE_PX, SCENE_PX), v)
                       for v in (0.2, 0.5, 0.9)], axis=-1)
    scene = scene_of(VideoStimulus(frames, time=[0, 100, 200]),
                     scotoma=Scotoma.circle(6), scotoma_fill=0.0)
    model = model_for(scene, rho=150, xrange=(-4, 4), yrange=(-4, 4),
                      step=0.5)
    percept = model.predict_percept(implant_at(0, 0), vmax=200)
    npt.assert_equal(percept.shape, (SCENE_PX, SCENE_PX, 3, 3))
    npt.assert_almost_equal(percept.time, [0, 100, 200])
    # Brighter frames make brighter phosphenes, in the right order. Read at
    # the fovea, which is inside the scotoma and so is phosphene only:
    peaks = [percept.data[HALF, HALF, 0, f] for f in range(3)]
    npt.assert_equal(np.all(np.diff(peaks) > 0), True)
    # Outside the scotoma every video frame passes through untouched:
    npt.assert_almost_equal(percept.data[0, 0, 0], [0.2, 0.5, 0.9], decimal=6)


def test_a_temporal_percept_must_cover_the_video():
    """Nothing is extrapolated in time, in either direction"""
    source = VideoStimulus(np.zeros((5, 5, 3)), time=[0, 10, 20])
    scene = Scene(source, fov=(5, 5), scotoma=Scotoma.circle(3))
    values = np.stack([np.full((5, 5), b) for b in (0.0, 20.0)], axis=-1)
    grid = ScoreboardModel(xrange=(-2, 2), yrange=(-2, 2),
                           step=1).build().spatial.grid
    short = Percept(values, space=grid, time=[5, 15])
    with pytest.raises(ValueError) as excinfo:
        scene._compose(short, vmax=20)
    npt.assert_equal('never simulated' in str(excinfo.value), True)
    # A percept that does cover it composes, endpoints included:
    covering = Percept(values, space=grid, time=[0, 20])
    npt.assert_equal(scene._compose(covering, vmax=20).shape[-1], 3)
    # ... and so does a still percept, which has no interval to run off:
    still = Percept(values[..., :1], space=grid)
    npt.assert_equal(scene._compose(still, vmax=20).shape[-1], 3)


def test_the_time_range_check_crosses_units():
    """A percept in seconds is held against a video in milliseconds"""
    source = VideoStimulus(np.zeros((5, 5, 3)), time=[0, 10, 20])
    scene = Scene(source, fov=(5, 5), scotoma=Scotoma.circle(3))
    grid = ScoreboardModel(xrange=(-2, 2), yrange=(-2, 2),
                           step=1).build().spatial.grid
    values = np.stack([np.full((5, 5), b) for b in (0.0, 20.0)], axis=-1)
    # 0-20 ms is exactly 0-0.02 s:
    covering = Percept(values, space=grid, time=[0, 0.02], time_unit=s)
    composed = scene._compose(covering, vmax=20)
    npt.assert_equal(composed.time_unit, ms)
    npt.assert_almost_equal(composed.time, [0, 10, 20])
    short = Percept(values, space=grid, time=[0, 0.01], time_unit=s)
    with pytest.raises(ValueError):
        scene._compose(short, vmax=20)


def test_per_frame_gaze_moves_the_eye_between_video_frames():
    frames = np.repeat(ramp_source().data.reshape(
        (SCENE_PX, SCENE_PX, 1)), 3, axis=-1)
    scene = scene_of(VideoStimulus(frames, time=[0, 100, 200]))
    model = model_for(scene)
    gaze = np.array([[-6.0, 0.0], [0.0, 0.0], [6.0, 0.0]])
    seen = seen_by(model, implant_at(0, 0), gaze=gaze * dva)
    npt.assert_almost_equal(seen.ravel(), ramp_at(gaze[:, 0]), decimal=4)
    # A still scene has one frame, so there is no per-frame gaze to give it:
    with pytest.raises(ValueError):
        seen_by(model_for(scene_of()), implant_at(0, 0), gaze=gaze)


def test_find_threshold_refuses_a_scene_model():
    """Thresholding rescales a stimulus this model does not take from you"""
    model = model_for(scene_of())
    implant = implant_at(0, 0)
    implant.stim = BiphasicPulse(20, 0.45)
    with pytest.raises(NotImplementedError):
        model.find_threshold(implant, 0.1)


def test_a_scene_survives_a_deepcopy():
    """Scene is composite state, so it must not reach the sub-models"""
    from copy import deepcopy
    scene = scene_of()
    model = model_for(scene)
    copied = deepcopy(model)
    npt.assert_equal(isinstance(copied.scene, Scene), True)
    npt.assert_equal(copied.scene.fov, scene.fov)
    npt.assert_almost_equal(seen_by(copied, implant_at(0, 0)),
                            seen_by(model, implant_at(0, 0)))
    # A Model built from components carries one just as well:
    built = Model(spatial=ScoreboardSpatial(), scene=scene)
    npt.assert_equal(built.scene is scene, True)
