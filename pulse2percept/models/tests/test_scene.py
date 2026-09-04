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
                                    Implant)
from pulse2percept.models import (FadingTemporal, Model, ScoreboardModel,
                                  ScoreboardSpatial)
from pulse2percept.models.base import _scene_stim
from pulse2percept.models.cortex import ScoreboardModel as CortexScoreboard
from pulse2percept.percepts import Percept
from pulse2percept.stimuli import (AmplitudeEncoder, BiphasicPulse,
                                   BiphasicPulseTrain, ImageStimulus,
                                   VideoStimulus)
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
    kwargs.setdefault('scotoma_blend', 0)
    return Scene(ramp_source() if source is None else source,
                 fov=(SCENE_PX, SCENE_PX), **kwargs)


def implant_at(x_um=0, y_um=0, encoder=True, input_frame='eye'):
    """An implant whose single electrode sits where we want to look"""
    return Implant(
        PointSource(x_um, y_um, 0), scene_input_frame=input_frame,
        encoder=AmplitudeEncoder(amp_range=(0, AMP_MAX)) if encoder else None)


def grid_implant(input_frame='eye'):
    """Three electrodes in a row"""
    return Implant(ElectrodeGrid((1, 3), 280), scene_input_frame=input_frame,
                   encoder=AmplitudeEncoder(amp_range=(0, AMP_MAX)))


def model_for(implant, **kwargs):
    # An explicit `visual_field_map`: the retinotopy is what the expected
    # values below are computed through, so it cannot be left to a default.
    params = {'rho': 200, 'xrange': (-3, 3), 'yrange': (-3, 3), 'step': 1,
              'visual_field_map': Curcio1990Map()}
    params.update(kwargs)
    return ScoreboardModel(implant=implant, **params).build()


def seen_by(model, scene, gaze=None):
    """The gray level each electrode was handed, in electrode order

    Read back off the amplitudes the encoder produced, which is the only place
    the sampled scene shows up once registration is over.
    """
    view = _scene_stim(model, scene, gaze)._spatial_view()
    return np.asarray(view.data, dtype=float).reshape(
        (len(model.implant.electrode_names), -1)) / AMP_MAX


def test_the_model_supplies_its_own_visual_field_map():
    """The caller never names a retinotopy; the model already has one"""
    scene = scene_of()
    implant = implant_at(*Curcio1990Map().dva_to_ret(6.0, 0.0))
    npt.assert_almost_equal(seen_by(model_for(implant), scene),
                            [[ramp_at(6.0)]], decimal=4)
    # Give the model a different retinotopy and the same electrode reads a
    # different part of the scene, with nothing else changing and nothing
    # about the map appearing at the call site:
    watson = model_for(implant, visual_field_map=Watson2014Map())
    npt.assert_equal(np.allclose(seen_by(watson, scene), ramp_at(6.0)),
                     False)


@pytest.mark.parametrize('x_dva', [-8.0, 2.5, 7.0])
def test_a_nonlinear_retinal_map_still_registers(x_dva):
    """Not 280 um/dva, and not linear either"""
    visual_field_map = SquareMap()
    implant = implant_at(*visual_field_map.dva_to_ret(x_dva, 0.0))
    model = model_for(implant, visual_field_map=visual_field_map)
    npt.assert_almost_equal(seen_by(model, scene_of()), [[ramp_at(x_dva)]],
                            decimal=4)


def test_gaze_moves_the_scene_past_an_eye_coupled_implant():
    """Gaze is the scene point on the fovea, so scene = visual field + gaze"""
    scene = scene_of()
    model = model_for(implant_at(0, 0))
    for gaze_x in (-4.0, 0.0, 6.0):
        npt.assert_almost_equal(seen_by(model, scene, gaze=(gaze_x, 0)),
                                [[ramp_at(gaze_x)]], decimal=4)
    # Units at the boundary, and the same answer through them:
    npt.assert_almost_equal(seen_by(model, scene, gaze=(6, 0) * dva),
                            seen_by(model, scene, gaze=(6.0, 0.0)))
    # Several electrodes keep their separation in the visual field whatever
    # the gaze: shifting gaze shifts what all of them see by the same amount.
    on_grid = model_for(grid_implant())
    here = seen_by(on_grid, scene).ravel()
    there = seen_by(on_grid, scene, gaze=(2, 0)).ravel()
    npt.assert_almost_equal(np.diff(here), np.diff(there), decimal=4)
    npt.assert_almost_equal(there - here, ramp_at(2) - ramp_at(0), decimal=4)


def test_gaze_leaves_a_head_mounted_camera_looking_where_it_was():
    """A camera on the head does not turn when the eye does"""
    scene = scene_of()
    model = model_for(implant_at(0, 0, input_frame='head'))
    fixating = seen_by(model, scene)
    npt.assert_almost_equal(fixating, [[ramp_at(0.0)]], decimal=4)
    for gaze in ((-4.0, 0.0), (6.0, 0.0), (0.0, 5.0), (6, 0) * dva):
        npt.assert_almost_equal(seen_by(model, scene, gaze=gaze), fixating,
                                decimal=6)
    # Same for several electrodes, and gaze=(0, 0) is the fixating case:
    on_grid = model_for(grid_implant(input_frame='head'))
    npt.assert_almost_equal(seen_by(on_grid, scene, gaze=(3, -2)),
                            seen_by(on_grid, scene, gaze=(0, 0)), decimal=6)


def test_an_unknown_scene_input_frame_is_refused():
    """A typo would silently change the physics, so it raises instead"""
    with pytest.raises(ValueError):
        implant_at(0, 0, input_frame='retinal')

    # A device class may also declare a bad default, which no setter sees:
    class Typo(Implant):
        __slots__ = ()
        _default_scene_input_frame = 'retinal'

    model = model_for(Typo(PointSource(0, 0, 0),
                           encoder=AmplitudeEncoder(amp_range=(0, AMP_MAX))))
    with pytest.raises(ValueError):
        _scene_stim(model, scene_of(), None)


def test_a_camera_driven_phosphene_still_travels_with_the_eye():
    """Caspi-style: the same phosphene, drawn where the eye now points"""
    scene = scene_of(scotoma=Scotoma.circle(6), scotoma_fill=0.0)
    model = model_for(implant_at(*Curcio1990Map().dva_to_ret(2.0, 0.0),
                                 input_frame='head'),
                      rho=100, xrange=(-4, 4), yrange=(-4, 4), step=0.5)
    fixating = model.predict_percept(scene, vmax=2).data[..., 0]
    shifted = model.predict_percept(scene, gaze=(5, 0) * dva,
                                    vmax=2).data[..., 0]
    # Unchanged input, so the phosphene is the same one, five degrees right:
    npt.assert_almost_equal(shifted[HALF, HALF + 5 + 2],
                            fixating[HALF, HALF + 2], decimal=5)
    npt.assert_almost_equal(model.predict_percept(scene, gaze=(0, 0),
                                                  vmax=2).data[..., 0],
                            fixating, decimal=6)


def test_y_orientation_survives_the_map():
    """Row 0 of the scene is +y in the visual field, both sides of the map"""
    data = np.tile(np.linspace(0, 1, SCENE_PX).reshape((-1, 1)),
                   (1, SCENE_PX))
    scene = scene_of(ImageStimulus(data))
    visual_field_map = Curcio1990Map()
    for y_dva in (5.0, -5.0):
        implant = implant_at(*visual_field_map.dva_to_ret(0.0, y_dva))
        npt.assert_almost_equal(seen_by(model_for(implant), scene),
                                [[(HALF - y_dva) / (2 * HALF)]], decimal=4)


def test_color_becomes_luminance_only_at_the_device():
    """The scene stays RGB; the electrode gets one number"""
    rgb = np.zeros((SCENE_PX, SCENE_PX, 3))
    rgb[..., 0] = 1.0  # pure red everywhere
    scene = scene_of(ImageStimulus(rgb))
    npt.assert_almost_equal(scene._sample_at(0.0, 0.0)[0, :, 0], [1, 0, 0],
                            decimal=5)
    # ... and the luminance of pure red reaches the implant:
    npt.assert_almost_equal(seen_by(model_for(implant_at(0, 0)), scene),
                            [[0.2125]], decimal=3)


def test_scene_driven_prediction_leaves_the_implant_alone():
    """Predicting what someone sees is a question, not an assignment"""
    implant = implant_at(0, 0)
    model = model_for(implant)
    model.predict_percept(scene_of())
    # The implant holds no trial state to be disturbed, and the settings the
    # scene path temporarily overrides are back the way the caller left them:
    npt.assert_equal(hasattr(implant, 'stim'), False)
    npt.assert_equal(implant.preprocess, False)
    npt.assert_equal(model.implant is implant, True)


def test_a_scene_driven_stimulus_still_goes_through_the_device():
    """The sampled scene is prepared by the implant, not written behind it"""
    grid = Implant(ElectrodeGrid((1, 3), 280),
                   encoder=AmplitudeEncoder(amp_range=(0, AMP_MAX)))
    grid.deactivate('A2')
    stim = _scene_stim(model_for(grid), scene_of(), None)
    npt.assert_equal('A2' in list(stim.electrodes), False)
    npt.assert_equal(len(stim.electrodes), 2)
    # It is a current by the time it comes back, which is the encoder having
    # run inside `prepare_stim`:
    npt.assert_equal(stim.unit, grid.stimulus_unit)


def edge_source():
    """A step edge down the middle: 0 to the left of x=0, 1 to the right"""
    data = np.zeros((SCENE_PX, SCENE_PX))
    data[:, HALF + 1:] = 1.0
    return ImageStimulus(data)


def test_preprocessing_runs_on_the_picture_not_on_electrode_values():
    """An edge filter needs an image, and by sampling time there is none left"""
    scene = scene_of(edge_source())
    at_edge = implant_at(*Curcio1990Map().dva_to_ret(0.5, 0.0))
    inside = implant_at(*Curcio1990Map().dva_to_ret(10.0, 0.0))
    # Untouched, the two electrodes see the two sides of the step:
    npt.assert_almost_equal(seen_by(model_for(at_edge), scene), [[0.5]],
                            decimal=3)
    npt.assert_almost_equal(seen_by(model_for(inside), scene), [[1.0]],
                            decimal=3)
    for implant in (at_edge, inside):
        implant.preprocess = lambda stim: stim.filter('sobel')
    # Sobel puts everything at the edge and nothing in the flat interior,
    # which is the opposite ordering from the raw scene:
    npt.assert_equal(seen_by(model_for(at_edge), scene)[0, 0] > 0.3, True)
    npt.assert_almost_equal(seen_by(model_for(inside), scene), [[0.0]],
                            decimal=4)


def test_preprocessing_does_not_reach_native_vision():
    """What the device does to its input is not what the eye goes through"""
    source = ramp_source()
    scene = scene_of(source, scotoma=Scotoma.circle(6), scotoma_fill=0.0)
    implant = implant_at(*Curcio1990Map().dva_to_ret(0.0, 0.0))
    implant.preprocess = lambda stim: stim.invert()
    model = model_for(implant, rho=100)
    # The electrode is at the fovea, where the ramp reads 0.5 either way, so
    # look somewhere the inversion actually shows:
    npt.assert_almost_equal(seen_by(model, scene, gaze=(8, 0)),
                            [[1 - ramp_at(8.0)]], decimal=3)
    percept = model.predict_percept(scene, gaze=(8, 0) * dva, vmax=100)
    # Outside the scotoma: the original scene, bit for bit and uninverted.
    original = np.repeat(source.data.reshape((SCENE_PX, SCENE_PX, 1)), 3,
                         axis=-1)
    x, y = scene._pixel_centers()
    intact = scene.scotoma(x - 8, y) == 0
    npt.assert_array_equal(percept.data[..., 0][intact], original[intact])
    # ... and the caller's scene was not rewritten on the way through:
    npt.assert_array_equal(scene.source.data, source.data)


def test_preprocessing_runs_exactly_once():
    """The stand-in implant must not put the picture through it again"""
    calls = []

    def counted(stim):
        calls.append(stim)
        return stim.invert()

    scene = scene_of()
    implant = implant_at(0, 0)
    implant.preprocess = counted
    model = model_for(implant)
    model.predict_percept(scene)
    npt.assert_equal(len(calls), 1)
    # Inversion is not idempotent, so a second pass would show up as the
    # original ramp coming back:
    npt.assert_almost_equal(seen_by(model, scene, gaze=(8, 0)),
                            [[1 - ramp_at(8.0)]], decimal=3)
    # The caller's implant still preprocesses; only the stand-in was told not
    # to, and only because it had already happened:
    npt.assert_equal(implant.preprocess is counted, True)


def test_a_video_scene_is_preprocessed_the_same_way():
    frames = np.stack([np.full((SCENE_PX, SCENE_PX), v)
                       for v in (0.2, 0.8)], axis=-1)
    scene = scene_of(VideoStimulus(frames, time=[0, 100]))
    implant = implant_at(0, 0)
    implant.preprocess = lambda stim: stim.invert()
    seen = seen_by(model_for(implant), scene)
    npt.assert_equal(seen.shape, (1, 2))
    npt.assert_almost_equal(seen.ravel(), [0.8, 0.2], decimal=3)


@pytest.mark.parametrize('returns', [lambda stim: BiphasicPulse(20, 0.45),
                                     lambda stim: np.zeros((4, 4))])
def test_scene_preprocessing_must_return_a_picture(returns):
    """Crossing to current early leaves nothing to register spatially"""
    scene = scene_of()
    implant = implant_at(0, 0)
    implant.preprocess = returns
    with pytest.raises(TypeError) as excinfo:
        model_for(implant).predict_percept(scene)
    npt.assert_equal('preprocess' in str(excinfo.value), True)
    npt.assert_equal('encoder' in str(excinfo.value), True)


def test_scene_preprocessing_must_preserve_spatial_shape():
    """A resize would reinterpret `fov`, not just change what is seen"""
    scene = scene_of()
    implant = implant_at(0, 0)
    implant.preprocess = lambda stim: stim.resize((20, 20))
    with pytest.raises(ValueError) as excinfo:
        model_for(implant).predict_percept(scene)
    npt.assert_equal('shape' in str(excinfo.value), True)


def test_scene_preprocessing_must_preserve_the_frame_clock():
    """Frames are what a video scene is registered against in time"""
    frames = np.stack([np.full((SCENE_PX, SCENE_PX), v)
                       for v in (0.2, 0.8)], axis=-1)
    scene = scene_of(VideoStimulus(frames, time=[0, 100]))
    implant = implant_at(0, 0)
    implant.preprocess = lambda stim: VideoStimulus(
        stim.data.reshape(stim.vid_shape)[..., :1], time=stim.time[:1])
    with pytest.raises(ValueError) as excinfo:
        model_for(implant).predict_percept(scene)
    npt.assert_equal('frame' in str(excinfo.value), True)
    # The same instants told in seconds rather than milliseconds are the same
    # clock, and stay allowed:
    implant.preprocess = lambda stim: VideoStimulus(
        1 - stim.data.reshape(stim.vid_shape), time=stim.time / 1000 * s)
    npt.assert_almost_equal(seen_by(model_for(implant), scene).ravel(),
                            [0.8, 0.2], decimal=3)


def test_a_scene_needs_an_encoder_and_a_retina():
    """Both failures name what is missing rather than dying downstream"""
    scene = scene_of()
    with pytest.raises(ValueError) as excinfo:
        model_for(implant_at(0, 0, encoder=False)).predict_percept(scene)
    npt.assert_equal('encoder' in str(excinfo.value), True)
    # A cortical model has no retinotopy to follow an electrode out along:
    cortical = CortexScoreboard(implant=implant_at(0, 0), rho=200,
                                xrange=(-3, 3), yrange=(-3, 3),
                                step=1).build()
    with pytest.raises(ValueError) as excinfo:
        cortical.predict_percept(scene)
    npt.assert_equal('visual_field_map' in str(excinfo.value), True)
    # ... and neither has a temporal-only model:
    from pulse2percept.models import Nanduri2012Temporal
    temporal = Model(temporal=Nanduri2012Temporal()).build()
    with pytest.raises(ValueError):
        temporal.predict_percept(scene)


def test_an_unbuilt_model_builds_itself_before_it_samples_anything():
    """A scene is sampled through a bound implant, so the build comes first"""
    unbuilt = ScoreboardModel(implant=implant_at(0, 0), rho=200,
                              xrange=(-3, 3), yrange=(-3, 3), step=1)
    npt.assert_equal(unbuilt.predict_percept(scene_of()) is not None, True)
    npt.assert_equal(unbuilt.is_built, True)
    # An implant with no encoder still cannot turn gray levels into current,
    # and that is now the first thing the caller hears about:
    unbuilt = ScoreboardModel(implant=implant_at(0, 0, encoder=False),
                              rho=200, xrange=(-3, 3), yrange=(-3, 3), step=1)
    with pytest.raises(ValueError, match='encoder'):
        unbuilt.predict_percept(scene_of())


def test_an_ordinary_source_is_not_registered_as_a_scene():
    """Only a Scene takes the registration path; a picture is stimulation"""
    grid = Implant(ElectrodeGrid((3, 3), 280),
                   encoder=AmplitudeEncoder(amp_range=(0, AMP_MAX)))
    model = model_for(grid)
    # The same pixels, not wrapped in a Scene, are sampled onto the electrodes
    # and encoded rather than registered through the retinotopy, so `gaze` is
    # not a thing that can be asked of them:
    percept = model.predict_percept(ramp_source())
    npt.assert_equal(percept.is_rgb, False)
    npt.assert_equal(percept.data.ndim, 3)
    with pytest.raises(ValueError):
        model.predict_percept(ramp_source(), gaze=(1, 0))


def test_without_a_scene_nothing_changes():
    """The ordinary path is untouched, and scene arguments are refused"""
    plain = ScoreboardModel(implant=implant_at(0, 0), rho=200,
                            xrange=(-3, 3), yrange=(-3, 3), step=1).build()
    percept = plain.predict_percept(BiphasicPulse(20, 0.45))
    npt.assert_equal(percept.is_rgb, False)
    npt.assert_equal(percept.data.ndim, 3)
    for kwargs in ({'gaze': (1, 0)}, {'vmax': 20}, {'vmin': 3}):
        with pytest.raises(ValueError):
            plain.predict_percept(BiphasicPulse(20, 0.45), **kwargs)
    # No stimulus at all still says nothing rather than raising:
    npt.assert_equal(plain.predict_percept(None), None)


def test_a_scene_without_a_scotoma_returns_the_prosthetic_percept():
    """Nothing is lost, so there is nothing to compose into"""
    model = model_for(implant_at(0, 0))
    percept = model.predict_percept(scene_of())
    npt.assert_equal(percept.is_rgb, False)
    npt.assert_equal(percept.data.ndim, 3)
    # On the model's grid, not the scene's:
    npt.assert_equal(percept.shape[:2], model.spatial.grid.shape)
    # `vmax` is meaningless here and is simply unused:
    npt.assert_array_equal(
        model.predict_percept(scene_of(), vmax=20).data, percept.data)


def test_a_scene_with_a_scotoma_returns_a_composed_rgb_percept():
    scene = scene_of(scotoma=Scotoma.circle(6), scotoma_fill=0.0)
    percept = model_for(implant_at(0, 0)).predict_percept(scene, vmax=20)
    npt.assert_equal(percept.is_rgb, True)
    npt.assert_equal(percept.shape, (SCENE_PX, SCENE_PX, 3, 1))
    npt.assert_equal(percept.data.min() >= 0, True)
    npt.assert_equal(percept.data.max() <= 1, True)
    # Reported on the scene's grid, in scene coordinates:
    npt.assert_almost_equal(percept.xdva, np.arange(-HALF, HALF + 1),
                            decimal=4)


def test_vmax_is_required_for_a_composed_percept():
    scene = scene_of(scotoma=Scotoma.circle(6))
    model = model_for(implant_at(0, 0))
    with pytest.raises(ValueError) as excinfo:
        model.predict_percept(scene)
    npt.assert_equal('vmax' in str(excinfo.value), True)


def test_the_intact_periphery_is_the_scene_exactly():
    """Outside the scotoma nothing is resampled, blended or rounded"""
    rng = np.random.default_rng(0)
    rgb = ImageStimulus(rng.random((SCENE_PX, SCENE_PX, 3)))
    scene = scene_of(rgb, scotoma=Scotoma.circle(6))
    percept = model_for(implant_at(0, 0)).predict_percept(scene, vmax=20)
    source = rgb.data.reshape((SCENE_PX, SCENE_PX, 3))
    x, y = scene._pixel_centers()
    intact = scene.scotoma(x, y) == 0
    npt.assert_array_equal(percept.data[..., 0][intact], source[intact])


def test_the_phosphene_lands_where_the_electrode_looks():
    """Position and y orientation, all the way through the composed result"""
    visual_field_map = Curcio1990Map()
    scene = scene_of(scotoma=Scotoma.circle(12), scotoma_fill=0.0)
    for x_dva, y_dva in [(4.0, 0.0), (0.0, 4.0), (-4.0, 0.0), (0.0, -4.0)]:
        implant = implant_at(*visual_field_map.dva_to_ret(x_dva, y_dva))
        model = model_for(implant, rho=80, xrange=(-8, 8), yrange=(-8, 8),
                          step=0.5)
        frame = model.predict_percept(scene, vmax=2).data[..., 0]
        row, col = int(round(HALF - y_dva)), int(round(x_dva + HALF))
        # Brightest where the electrode looks, dark on the opposite side:
        npt.assert_equal(frame[row, col].mean() > 0.5, True)
        npt.assert_equal(frame[SCENE_PX - 1 - row,
                               SCENE_PX - 1 - col].mean() < 0.1, True)


def test_gaze_moves_the_scotoma_and_the_phosphene_together():
    scene = scene_of(scotoma=Scotoma.circle(4), scotoma_fill=0.3)
    model = model_for(implant_at(0, 0), rho=100, xrange=(-2, 2),
                      yrange=(-2, 2), step=0.5)
    fixating = model.predict_percept(scene, vmax=2).data[..., 0]
    shifted = model.predict_percept(scene, gaze=(5, 0) * dva,
                                    vmax=2).data[..., 0]
    # The whole eye-centered pair travelled 5 degrees right across the scene:
    npt.assert_almost_equal(shifted[HALF, HALF + 5], fixating[HALF, HALF],
                            decimal=5)
    # ... and where the eye used to point is native vision again:
    source = scene.source.data.reshape((SCENE_PX, SCENE_PX))
    npt.assert_almost_equal(shifted[HALF, HALF], [source[HALF, HALF]] * 3,
                            decimal=5)


def test_a_fixed_vmax_does_not_renormalize_when_gaze_changes():
    """Nothing rescales the display behind the user's back"""
    scene = scene_of(scotoma=Scotoma.circle(12), scotoma_fill=0.0)
    model = model_for(implant_at(0, 0), rho=100, xrange=(-4, 4),
                      yrange=(-4, 4), step=0.5)

    def phosphene(gaze_x, **kwargs):
        """The composed pixel the foveal electrode paints

        The scotoma travels with the eye, so the fovea sits at scene x =
        ``gaze_x``. That pixel is pure phosphene; the intact periphery would
        otherwise dominate any whole-frame maximum.
        """
        percept = model.predict_percept(scene, gaze=(gaze_x, 0) * dva,
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
    model = model_for(implant_at(0, 0), rho=150, xrange=(-4, 4),
                      yrange=(-4, 4), step=0.5)
    percept = model.predict_percept(scene, vmax=200)
    npt.assert_equal(percept.shape, (SCENE_PX, SCENE_PX, 3, 3))
    npt.assert_almost_equal(percept.time, [0, 100, 200])
    # Brighter frames make brighter phosphenes, in the right order. Read at
    # the fovea, which is inside the scotoma and so is phosphene only:
    peaks = [percept.data[HALF, HALF, 0, f] for f in range(3)]
    npt.assert_equal(np.all(np.diff(peaks) > 0), True)
    # Outside the scotoma every video frame passes through untouched:
    npt.assert_almost_equal(percept.data[0, 0, 0], [0.2, 0.5, 0.9], decimal=6)


def test_a_spatiotemporal_model_composes_against_a_video_scene():
    """The ordinary spatial+temporal pipeline, with nothing asked of it"""
    frames = np.stack([np.full((SCENE_PX, SCENE_PX), v)
                       for v in (0.2, 0.5, 0.9)], axis=-1)
    source = VideoStimulus(frames, time=[0, 100, 200])
    scene = Scene(source, fov=(SCENE_PX, SCENE_PX),
                  scotoma=Scotoma.circle(6), scotoma_fill=0.0)

    def spatiotemporal():
        return Model(
            spatial=ScoreboardSpatial(implant_at(0, 0), rho=200,
                                      xrange=(-4, 4), yrange=(-4, 4),
                                      step=0.5,
                                      visual_field_map=Curcio1990Map()),
            temporal=FadingTemporal()).build()

    raw = spatiotemporal().predict_percept(
        Scene(source, fov=(SCENE_PX, SCENE_PX)))
    # The premise: the two clocks really do differ, frame for frame.
    npt.assert_almost_equal(raw.time, [100, 200, 300])
    npt.assert_almost_equal(scene.time, [0, 100, 200])

    percept = spatiotemporal().predict_percept(scene, vmax=5)
    npt.assert_equal(percept.shape, (SCENE_PX, SCENE_PX, 3, 3))
    # The percept's clock describes the output, not the video's onsets:
    npt.assert_almost_equal(percept.time, [100, 200, 300])
    # ... and the native frames come through in order, one per output frame:
    npt.assert_almost_equal(percept.data[0, 0, 0], [0.2, 0.5, 0.9], decimal=5)


def test_a_temporal_stage_does_not_lose_the_visual_field_grid():
    """A percept rewritten frame by frame has not moved in the visual field"""
    model = Model(spatial=ScoreboardSpatial(implant_at(0, 0), rho=200,
                                            xrange=(-2, 2), yrange=(-2, 2),
                                            step=1),
                  temporal=FadingTemporal()).build()
    percept = model.predict_percept(
        BiphasicPulseTrain(20, 30, 0.45, stim_dur=50))
    npt.assert_equal(percept._has_space, True)
    npt.assert_almost_equal(percept.xdva, [-2, -1, 0, 1, 2])
    npt.assert_almost_equal(percept.ydva, [-2, -1, 0, 1, 2])


def test_a_single_timed_percept_is_not_broadcast_over_a_video():
    """One frame at a named instant happened then, not throughout"""
    source = VideoStimulus(np.zeros((5, 5, 3)), time=[0, 10, 20])
    scene = Scene(source, fov=(5, 5), scotoma=Scotoma.circle(3))
    grid = ScoreboardModel(implant=implant_at(0, 0), xrange=(-2, 2),
                           yrange=(-2, 2), step=1).build().spatial.grid
    at_10 = Percept(np.full((5, 5, 1), 20.0), space=grid, time=[10])
    with pytest.raises(ValueError) as excinfo:
        scene._compose(at_10, vmax=20)
    npt.assert_equal('never simulated' in str(excinfo.value), True)
    # A percept with no clock at all did not happen at any instant, so it does
    # stand behind every frame:
    timeless = Percept(np.full((5, 5, 1), 20.0), space=grid)
    npt.assert_equal(timeless.time, None)
    composed = scene._compose(timeless, vmax=20)
    npt.assert_equal(composed.shape[-1], 3)
    npt.assert_almost_equal(composed.time, [0, 10, 20])


def test_a_temporal_percept_must_cover_the_video():
    """Nothing is extrapolated in time, in either direction"""
    source = VideoStimulus(np.zeros((5, 5, 3)), time=[0, 10, 20])
    scene = Scene(source, fov=(5, 5), scotoma=Scotoma.circle(3))
    values = np.stack([np.full((5, 5), b) for b in (0.0, 20.0)], axis=-1)
    grid = ScoreboardModel(implant=implant_at(0, 0), xrange=(-2, 2),
                           yrange=(-2, 2), step=1).build().spatial.grid
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
    grid = ScoreboardModel(implant=implant_at(0, 0), xrange=(-2, 2),
                           yrange=(-2, 2), step=1).build().spatial.grid
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
    gaze = np.array([[-6.0, 0.0], [0.0, 0.0], [6.0, 0.0]])
    eye = model_for(implant_at(0, 0))
    seen = seen_by(eye, scene, gaze=gaze * dva)
    npt.assert_almost_equal(seen.ravel(), ramp_at(gaze[:, 0]), decimal=4)
    # A head-mounted camera holds still through the same eye movements:
    camera = model_for(implant_at(0, 0, input_frame='head'))
    npt.assert_almost_equal(seen_by(camera, scene, gaze=gaze * dva).ravel(),
                            [ramp_at(0.0)] * 3, decimal=4)
    # A still scene has one frame, so there is no per-frame gaze to give
    # it, whichever frame the device takes its input in:
    for model in (eye, camera):
        with pytest.raises(ValueError):
            seen_by(model, scene_of(), gaze=gaze)


def test_a_bound_implant_survives_a_deepcopy():
    """A copied model describes the same physical implant"""
    from copy import deepcopy
    implant = implant_at(0, 0)
    model = model_for(implant)
    copied = deepcopy(model)
    npt.assert_equal(copied.implant is implant, True)
    npt.assert_equal(copied.is_built, True)
    npt.assert_almost_equal(seen_by(copied, scene_of()),
                            seen_by(model, scene_of()))
