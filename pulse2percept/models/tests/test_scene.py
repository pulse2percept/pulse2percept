"""Scene-driven prediction: registering a scene onto an implant (#668)

Registration happens in the model, which has both the retinotopy and the
implant.

Scenes use one degree per pixel with an odd pixel count, so pixel centers are
on whole degrees and the center pixel is at the origin.
"""
import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.implants import (ElectrodeArray, ElectrodeGrid,
                                    Implant, PointSource)
from pulse2percept.models import FadingTemporal, Model, SpatialModel
from pulse2percept.models.retina import ScoreboardModel, ScoreboardSpatial
from pulse2percept.models.base import _placement_shift, _scene_stim
from pulse2percept.models.cortex import ScoreboardModel as CortexScoreboard
from pulse2percept.percepts import Percept
from pulse2percept.stimuli import (AmplitudeEncoder, BiphasicPulse,
                                   BiphasicPulseTrain, ImageStimulus,
                                   VideoStimulus)
from pulse2percept.topography.cortex import Polimeni2006Map
from pulse2percept.topography.retina import (Curcio1990Map, RetinalMap,
                                             Watson2014Map)
from pulse2percept.units import deg, dva, ms, s, um
from pulse2percept.vision import Gaze, Scene, Scotoma

#: Square scene: one pixel per degree, center pixel at the origin.
SCENE_PX = 41
HALF = (SCENE_PX - 1) // 2

#: Gray level 1 maps to this many uA, so amplitude / AMP_MAX = gray level.
AMP_MAX = 100.0


class SquareMap(RetinalMap):
    """A retinal map that is neither 280 um/dva nor linear

    Retinal x grows as the square of eccentricity, so a fixed um/dva ratio
    gives the wrong location.
    """

    def dva_to_ret(self, xdva, ydva):
        return np.sign(xdva) * 100.0 * xdva ** 2, -100.0 * ydva

    def ret_to_dva(self, xret, yret):
        return np.sign(xret) * np.sqrt(np.abs(xret) / 100.0), -yret / 100.0


def ramp_source():
    """Return an image with gray level 0 at x=-20 dva, 1 at x=+20 dva"""
    return ImageStimulus(np.tile(np.linspace(0, 1, SCENE_PX), (SCENE_PX, 1)))


def ramp_at(x_dva):
    """Return the `ramp_source` gray level at scene x (dva)"""
    return (x_dva + HALF) / (2 * HALF)


def scene_of(source=None, **kwargs):
    kwargs.setdefault('scotoma_blend', 0)
    return Scene(ramp_source() if source is None else source,
                 fov=(SCENE_PX, SCENE_PX), **kwargs)


def implant_at(x_um=0, y_um=0, encoder=True, input_frame='eye'):
    """Return a single-electrode implant at (x_um, y_um)"""
    return Implant(
        PointSource(x_um, y_um, 0), scene_input_frame=input_frame,
        encoder=AmplitudeEncoder(amp_range=(0, AMP_MAX)) if encoder else None)


def grid_implant(input_frame='eye'):
    """Three electrodes in a row"""
    return Implant(ElectrodeGrid((1, 3), 280), scene_input_frame=input_frame,
                   encoder=AmplitudeEncoder(amp_range=(0, AMP_MAX)))


def model_for(implant, **kwargs):
    # Explicit `visual_field_map`, since expected values are computed with it:
    params = {'rho': 200, 'xrange': (-3, 3), 'yrange': (-3, 3), 'step': 1,
              'visual_field_map': Curcio1990Map()}
    params.update(kwargs)
    return ScoreboardModel(implant=implant, **params).build()


def composed(model, scene, vmax, gaze=None, **kwargs):
    """Return the model percept rendered into the scene

    Prediction returns a percept on the model grid; `Scene.render` composes it.
    """
    percept = model.predict_percept(scene, gaze=gaze)
    return scene.render(percept=percept, gaze=gaze, vmax=vmax, **kwargs).data


def seen_by(model, scene, gaze=None):
    """Return the gray level sampled by each electrode, in electrode order

    Computed from the encoded amplitudes.
    """
    view = _scene_stim(model, scene, gaze)._spatial_view()
    return np.asarray(view.data, dtype=float).reshape(
        (len(model.implant.electrode_names), -1)) / AMP_MAX


def test_the_model_supplies_its_own_visual_field_map():
    """Scene sampling uses the model's `visual_field_map`"""
    scene = scene_of()
    implant = implant_at(*Curcio1990Map().dva_to_ret(6.0, 0.0))
    npt.assert_almost_equal(seen_by(model_for(implant), scene),
                            [[ramp_at(6.0)]], decimal=4)
    # A different map makes the same electrode sample a different location:
    watson = model_for(implant, visual_field_map=Watson2014Map())
    npt.assert_equal(np.allclose(seen_by(watson, scene), ramp_at(6.0)),
                     False)


@pytest.mark.parametrize('x_dva', [-8.0, 2.5, 7.0])
def test_a_nonlinear_retinal_map_still_registers(x_dva):
    """Registration works with a nonlinear map"""
    visual_field_map = SquareMap()
    implant = implant_at(*visual_field_map.dva_to_ret(x_dva, 0.0))
    model = model_for(implant, visual_field_map=visual_field_map)
    npt.assert_almost_equal(seen_by(model, scene_of()), [[ramp_at(x_dva)]],
                            decimal=4)


def test_gaze_moves_the_scene_past_an_eye_coupled_implant():
    """Gaze is the scene point at the fovea: scene = visual field + gaze"""
    scene = scene_of()
    model = model_for(implant_at(0, 0))
    for gaze_x in (-4.0, 0.0, 6.0):
        npt.assert_almost_equal(seen_by(model, scene, gaze=(gaze_x, 0)),
                                [[ramp_at(gaze_x)]], decimal=4)
    # Unitful gaze gives the same result:
    npt.assert_almost_equal(seen_by(model, scene, gaze=(6, 0) * dva),
                            seen_by(model, scene, gaze=(6.0, 0.0)))
    # Gaze shifts all electrodes by the same amount:
    on_grid = model_for(grid_implant())
    here = seen_by(on_grid, scene).ravel()
    there = seen_by(on_grid, scene, gaze=(2, 0)).ravel()
    npt.assert_almost_equal(np.diff(here), np.diff(there), decimal=4)
    npt.assert_almost_equal(there - here, ramp_at(2) - ramp_at(0), decimal=4)


def test_gaze_leaves_a_head_mounted_camera_looking_where_it_was():
    """Gaze does not affect a head-mounted camera"""
    scene = scene_of()
    model = model_for(implant_at(0, 0, input_frame='head'))
    fixating = seen_by(model, scene)
    npt.assert_almost_equal(fixating, [[ramp_at(0.0)]], decimal=4)
    for gaze in ((-4.0, 0.0), (6.0, 0.0), (0.0, 5.0), (6, 0) * dva):
        npt.assert_almost_equal(seen_by(model, scene, gaze=gaze), fixating,
                                decimal=6)
    # Same for several electrodes; gaze=(0, 0) is fixation:
    on_grid = model_for(grid_implant(input_frame='head'))
    npt.assert_almost_equal(seen_by(on_grid, scene, gaze=(3, -2)),
                            seen_by(on_grid, scene, gaze=(0, 0)), decimal=6)


def ramp_video_scene(n_frames=4):
    """Return a static ramp video with 100 ms frames"""
    frames = np.repeat(ramp_source().data.reshape(
        (SCENE_PX, SCENE_PX, 1)), n_frames, axis=-1)
    return scene_of(VideoStimulus(frames,
                                  time=np.arange(n_frames) * 100.0))


def test_a_gaze_trajectory_is_resolved_on_the_scenes_own_frames():
    """Sparse gaze events are expanded to one gaze per source frame"""
    scene = ramp_video_scene()
    model = model_for(implant_at(0, 0))
    sparse = Gaze([(0, 0), (6, 0)] * dva, time=[0, 200] * ms)
    expanded = np.array([(0, 0), (0, 0), (6, 0), (6, 0)], dtype=float)
    npt.assert_almost_equal(seen_by(model, scene, gaze=sparse).ravel(),
                            ramp_at(expanded[:, 0]), decimal=4)
    npt.assert_array_equal(model.predict_percept(scene, gaze=sparse).data,
                           model.predict_percept(scene,
                                                 gaze=expanded * dva).data)


def test_a_gaze_trajectory_does_not_move_a_head_mounted_camera():
    scene = ramp_video_scene()
    model = model_for(implant_at(0, 0, input_frame='head'))
    sparse = Gaze([(0, 0), (6, 0)] * dva, time=[0, 200] * ms)
    npt.assert_almost_equal(seen_by(model, scene, gaze=sparse),
                            seen_by(model, scene), decimal=6)


def test_a_gaze_trajectory_needs_a_scene_with_frame_times():
    """A gaze trajectory with a still scene raises ValueError"""
    model = model_for(implant_at(0, 0))
    gaze = Gaze([(0, 0), (6, 0)] * dva, time=[0, 200] * ms)
    with pytest.raises(ValueError):
        model.predict_percept(scene_of(), gaze=gaze)


def test_an_unknown_scene_input_frame_is_refused():
    """An unknown `scene_input_frame` raises ValueError"""
    with pytest.raises(ValueError):
        implant_at(0, 0, input_frame='retinal')

    # Also for a bad class default (bypasses the setter):
    class Typo(Implant):
        __slots__ = ()
        _default_scene_input_frame = 'retinal'

    model = model_for(Typo(PointSource(0, 0, 0),
                           encoder=AmplitudeEncoder(amp_range=(0, AMP_MAX))))
    with pytest.raises(ValueError):
        _scene_stim(model, scene_of(), None)


def test_a_camera_driven_phosphene_still_travels_with_the_eye():
    """Head-mounted camera: same phosphene, drawn where the eye points"""
    scene = scene_of(scotoma=Scotoma.circle(6), scotoma_fill=0.0)
    model = model_for(implant_at(*Curcio1990Map().dva_to_ret(2.0, 0.0),
                                 input_frame='head'),
                      rho=100, xrange=(-4, 4), yrange=(-4, 4), step=0.5)
    fixating = composed(model, scene, vmax=2)[..., 0]
    shifted = composed(model, scene, vmax=2, gaze=(5, 0) * dva)[..., 0]
    # Same input, so the phosphene is still 2 deg right of the fovea in the
    # eye-centered display:
    npt.assert_almost_equal(shifted[HALF, HALF + 2],
                            fixating[HALF, HALF + 2], decimal=5)
    npt.assert_almost_equal(composed(model, scene, vmax=2, gaze=(0, 0))[...,
                                                                       0],
                            fixating, decimal=6)
    # The model percept does not depend on gaze:
    npt.assert_array_equal(
        model.predict_percept(scene, gaze=(5, 0) * dva).data,
        model.predict_percept(scene).data)


def test_y_orientation_survives_the_map():
    """Row 0 of the scene is +y in the visual field, on both sides of y=0"""
    data = np.tile(np.linspace(0, 1, SCENE_PX).reshape((-1, 1)),
                   (1, SCENE_PX))
    scene = scene_of(ImageStimulus(data))
    visual_field_map = Curcio1990Map()
    for y_dva in (5.0, -5.0):
        implant = implant_at(*visual_field_map.dva_to_ret(0.0, y_dva))
        npt.assert_almost_equal(seen_by(model_for(implant), scene),
                                [[(HALF - y_dva) / (2 * HALF)]], decimal=4)


def test_color_becomes_luminance_only_at_the_device():
    """The scene stays RGB; the electrode gets luminance"""
    rgb = np.zeros((SCENE_PX, SCENE_PX, 3))
    rgb[..., 0] = 1.0  # pure red everywhere
    scene = scene_of(ImageStimulus(rgb))
    npt.assert_almost_equal(scene._sample_at(0.0, 0.0)[0, :, 0], [1, 0, 0],
                            decimal=5)
    # Luminance of pure red:
    npt.assert_almost_equal(seen_by(model_for(implant_at(0, 0)), scene),
                            [[0.2125]], decimal=3)


def test_scene_driven_prediction_leaves_the_implant_alone():
    """Scene prediction does not modify the implant"""
    implant = implant_at(0, 0)
    model = model_for(implant)
    model.predict_percept(scene_of())
    # No stored stimulus, and temporarily overridden settings are restored:
    npt.assert_equal(hasattr(implant, 'stim'), False)
    npt.assert_equal(implant.preprocess, False)
    npt.assert_equal(model.implant is implant, True)


def test_a_scene_driven_stimulus_still_goes_through_the_device():
    """The sampled scene goes through `implant.prepare_stim`"""
    grid = Implant(ElectrodeGrid((1, 3), 280),
                   encoder=AmplitudeEncoder(amp_range=(0, AMP_MAX)))
    grid.deactivate('A2')
    stim = _scene_stim(model_for(grid), scene_of(), None)
    npt.assert_equal('A2' in list(stim.electrodes), False)
    npt.assert_equal(len(stim.electrodes), 2)
    # The encoder converted it to current:
    npt.assert_equal(stim.unit, grid.stimulus_unit)


def edge_source():
    """Return a step edge: 0 left of x=0, 1 right of it"""
    data = np.zeros((SCENE_PX, SCENE_PX))
    data[:, HALF + 1:] = 1.0
    return ImageStimulus(data)


def test_preprocessing_runs_on_the_picture_not_on_electrode_values():
    """Preprocessing is applied to the image before sampling"""
    scene = scene_of(edge_source())
    at_edge = implant_at(*Curcio1990Map().dva_to_ret(0.5, 0.0))
    inside = implant_at(*Curcio1990Map().dva_to_ret(10.0, 0.0))
    # Without preprocessing:
    npt.assert_almost_equal(seen_by(model_for(at_edge), scene), [[0.5]],
                            decimal=3)
    npt.assert_almost_equal(seen_by(model_for(inside), scene), [[1.0]],
                            decimal=3)
    for implant in (at_edge, inside):
        implant.preprocess = lambda stim: stim.filter('sobel')
    # Sobel is bright at the edge and zero in the flat interior:
    npt.assert_equal(seen_by(model_for(at_edge), scene)[0, 0] > 0.3, True)
    npt.assert_almost_equal(seen_by(model_for(inside), scene), [[0.0]],
                            decimal=4)


def test_preprocessing_does_not_reach_native_vision():
    """Implant preprocessing does not affect native vision"""
    source = ramp_source()
    scene = scene_of(source, scotoma=Scotoma.circle(6), scotoma_fill=0.0)
    implant = implant_at(*Curcio1990Map().dva_to_ret(0.0, 0.0))
    implant.preprocess = lambda stim: stim.invert()
    model = model_for(implant, rho=100)
    # The ramp is 0.5 at the fovea whether inverted or not, so shift gaze:
    npt.assert_almost_equal(seen_by(model, scene, gaze=(8, 0)),
                            [[1 - ramp_at(8.0)]], decimal=3)
    seen = composed(model, scene, vmax=100, gaze=(8, 0) * dva)
    # Outside the scotoma: the original scene, shifted 8 deg left, uninverted:
    original = np.repeat(source.data.reshape((SCENE_PX, SCENE_PX, 1)), 3,
                         axis=-1)
    x, y = scene._pixel_centers()
    intact = (scene.scotoma(x, y) == 0)[:, :-8]
    npt.assert_almost_equal(seen[:, :-8, :, 0][intact],
                            original[:, 8:][intact], decimal=6)
    # The user's scene is unchanged:
    npt.assert_array_equal(scene.source.data, source.data)


def test_preprocessing_runs_exactly_once():
    """Preprocessing runs once per prediction"""
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
    # Inverting twice would restore the original ramp:
    npt.assert_almost_equal(seen_by(model, scene, gaze=(8, 0)),
                            [[1 - ramp_at(8.0)]], decimal=3)
    # The implant's own setting is untouched:
    npt.assert_equal(implant.preprocess is counted, True)


class _BindingCheck(AmplitudeEncoder):
    """Record the implant bound at each ``encode`` call"""
    __slots__ = ('seen',)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen = []

    def encode(self, source):
        self.seen.append(self.implant)
        return super().encode(source)


class _PreparingImplant(Implant):
    """Record every implant object that prepares a stimulus"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepared = []

    def _prepare_stim(self, *args, **kwargs):
        self.prepared.append(self)
        return super()._prepare_stim(*args, **kwargs)


def test_scene_encodes_with_the_bound_implant():
    """Scene input is prepared by the encoder's bound implant"""
    encoder = _BindingCheck(amp_range=(0, AMP_MAX))
    implant = _PreparingImplant(PointSource(0, 0, 0), encoder=encoder)
    model = model_for(implant)
    model.predict_percept(scene_of())
    npt.assert_equal(len(encoder.seen), 1)
    npt.assert_equal(encoder.seen[0] is implant, True)
    # Same implant object (a shallow copy would share `prepared` but record
    # itself):
    npt.assert_equal(len(implant.prepared), 1)
    npt.assert_equal(implant.prepared[0] is encoder.implant, True)


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
    """Preprocessing must return an image, not a current"""
    scene = scene_of()
    implant = implant_at(0, 0)
    implant.preprocess = returns
    with pytest.raises(TypeError) as excinfo:
        model_for(implant).predict_percept(scene)
    npt.assert_equal('preprocess' in str(excinfo.value), True)
    npt.assert_equal('encoder' in str(excinfo.value), True)


def test_scene_preprocessing_must_preserve_spatial_shape():
    """Preprocessing must not change the image shape (it would change `fov`)"""
    scene = scene_of()
    implant = implant_at(0, 0)
    implant.preprocess = lambda stim: stim.resize((20, 20))
    with pytest.raises(ValueError) as excinfo:
        model_for(implant).predict_percept(scene)
    npt.assert_equal('shape' in str(excinfo.value), True)


def test_scene_preprocessing_must_preserve_the_frame_clock():
    """Preprocessing must keep the video frame times"""
    frames = np.stack([np.full((SCENE_PX, SCENE_PX), v)
                       for v in (0.2, 0.8)], axis=-1)
    scene = scene_of(VideoStimulus(frames, time=[0, 100]))
    implant = implant_at(0, 0)
    implant.preprocess = lambda stim: VideoStimulus(
        stim.data.reshape(stim.vid_shape)[..., :1], time=stim.time[:1])
    with pytest.raises(ValueError) as excinfo:
        model_for(implant).predict_percept(scene)
    npt.assert_equal('frame' in str(excinfo.value), True)
    # The same frame times in seconds are allowed:
    implant.preprocess = lambda stim: VideoStimulus(
        1 - stim.data.reshape(stim.vid_shape), time=stim.time / 1000 * s)
    npt.assert_almost_equal(seen_by(model_for(implant), scene).ravel(),
                            [0.8, 0.2], decimal=3)


def test_a_scene_needs_an_encoder_and_a_spatial_model():
    """A scene requires an encoder and a spatial model"""
    scene = scene_of()
    with pytest.raises(ValueError) as excinfo:
        model_for(implant_at(0, 0, encoder=False)).predict_percept(scene)
    npt.assert_equal('encoder' in str(excinfo.value), True)
    # A temporal-only model has no electrodes:
    from pulse2percept.models.retina import Nanduri2012Temporal
    temporal = Model(temporal=Nanduri2012Temporal()).build()
    with pytest.raises(ValueError):
        temporal.predict_percept(scene)


class BareSpatial(SpatialModel):
    """A spatial model on the anatomy-neutral base"""

    def get_default_params(self):
        return {**super().get_default_params(),
                'visual_field_map': Curcio1990Map()}

    def _predict_spatial(self, electrode_array, stim):
        n_time = 1 if stim.time is None else stim.time.size
        return np.zeros((self.grid.x.size, n_time), dtype=np.float32)


def test_scene_registration_is_a_spatial_model_capability():
    """Scene sampling requires `_scene_sampling_points`

    Models that do not implement it raise NotImplementedError naming the
    model class.
    """
    scene = scene_of()
    # Cortical model:
    cortical = CortexScoreboard(implant=implant_at(0, 0), rho=200,
                                xrange=(-3, 3), yrange=(-3, 3),
                                step=1).build()
    with pytest.raises(NotImplementedError) as excinfo:
        cortical.predict_percept(scene)
    npt.assert_equal('ScoreboardSpatial' in str(excinfo.value), True)
    # Bare spatial model:
    bare = Model(spatial=BareSpatial(implant_at(0, 0), xrange=(-3, 3),
                                     yrange=(-3, 3), step=1)).build()
    with pytest.raises(NotImplementedError) as excinfo:
        bare.predict_percept(scene)
    npt.assert_equal('BareSpatial' in str(excinfo.value), True)
    # A retinal model with a cortical map raises ValueError:
    retinal = model_for(implant_at(0, 0))
    retinal.spatial.visual_field_map = Polimeni2006Map()
    with pytest.raises(ValueError) as excinfo:
        retinal.predict_percept(scene)
    npt.assert_equal('visual_field_map' in str(excinfo.value), True)


def test_scene_sampling_points_are_the_registration_the_model_uses():
    """`_scene_sampling_points` returns the coordinates used for sampling"""
    x_dva = 4.0
    visual_field_map = Curcio1990Map()
    implant = implant_at(*visual_field_map.dva_to_ret(x_dva, 0.0))
    model = model_for(implant, visual_field_map=visual_field_map)
    x_vf, y_vf = model.spatial._scene_sampling_points()
    npt.assert_almost_equal(x_vf, [x_dva])
    npt.assert_almost_equal(y_vf, [0.0])
    # Matches the gray level the encoder sampled:
    npt.assert_almost_equal(seen_by(model, scene_of()).ravel(),
                            [ramp_at(x_dva)], decimal=3)


def test_an_unbuilt_model_builds_itself_before_it_samples_anything():
    """An unbuilt model builds before sampling a scene"""
    unbuilt = ScoreboardModel(implant=implant_at(0, 0), rho=200,
                              xrange=(-3, 3), yrange=(-3, 3), step=1)
    npt.assert_equal(unbuilt.predict_percept(scene_of()) is not None, True)
    npt.assert_equal(unbuilt.is_built, True)
    # An implant without an encoder raises ValueError:
    unbuilt = ScoreboardModel(implant=implant_at(0, 0, encoder=False),
                              rho=200, xrange=(-3, 3), yrange=(-3, 3), step=1)
    with pytest.raises(ValueError, match='encoder'):
        unbuilt.predict_percept(scene_of())


def test_an_ordinary_source_is_not_registered_as_a_scene():
    """Only a Scene is registered; an image is encoded directly"""
    grid = Implant(ElectrodeGrid((3, 3), 280),
                   encoder=AmplitudeEncoder(amp_range=(0, AMP_MAX)))
    model = model_for(grid)
    # An image is mapped onto the electrodes by the encoder, so `gaze` raises
    # ValueError:
    percept = model.predict_percept(ramp_source())
    npt.assert_equal(percept.is_rgb, False)
    npt.assert_equal(percept.data.ndim, 3)
    with pytest.raises(ValueError):
        model.predict_percept(ramp_source(), gaze=(1, 0))


def test_without_a_scene_nothing_changes():
    """Without a scene, `gaze`, `vmin`, and `vmax` are rejected"""
    plain = ScoreboardModel(implant=implant_at(0, 0), rho=200,
                            xrange=(-3, 3), yrange=(-3, 3), step=1).build()
    percept = plain.predict_percept(BiphasicPulse(20, 0.45))
    npt.assert_equal(percept.is_rgb, False)
    npt.assert_equal(percept.data.ndim, 3)
    with pytest.raises(ValueError):
        plain.predict_percept(BiphasicPulse(20, 0.45), gaze=(1, 0))
    # `vmin`/`vmax` are not prediction arguments:
    for kwargs in ({'vmax': 20}, {'vmin': 3}):
        with pytest.raises(TypeError):
            plain.predict_percept(BiphasicPulse(20, 0.45), **kwargs)
    # Nothing in, None out:
    npt.assert_equal(plain.predict_percept(None), None)


def test_scene_prediction_returns_the_model_percept():
    """Scene prediction returns the model percept on the model grid"""
    model = model_for(implant_at(0, 0))
    for scene in (scene_of(),
                  scene_of(scotoma=Scotoma.circle(6), scotoma_fill=0.0)):
        percept = model.predict_percept(scene)
        npt.assert_equal(percept.is_rgb, False)
        npt.assert_equal(percept.data.ndim, 3)
        # Model grid, not scene grid:
        npt.assert_equal(percept.shape[:2], model.spatial.grid.shape)
        npt.assert_equal(percept.shape[:2] == (SCENE_PX, SCENE_PX), False)
        npt.assert_almost_equal(percept.xdva, model.spatial.grid.x[0])


def test_a_scotoma_does_not_touch_the_prosthetic_percept():
    """The scotoma does not affect the prosthetic percept"""
    model = model_for(implant_at(0, 0))
    seeing = model.predict_percept(scene_of(), gaze=(3, -1) * dva)
    for fill in (0.0, 0.6, 'inpaint'):
        blind = scene_of(scotoma=Scotoma.circle(6), scotoma_fill=fill,
                         scotoma_blend=1.5)
        npt.assert_array_equal(
            model.predict_percept(blind, gaze=(3, -1) * dva).data,
            seeing.data)
    # The rendered scene does differ:
    npt.assert_equal(np.allclose(
        composed(model, scene_of(scotoma=Scotoma.circle(6), scotoma_fill=0.0),
                 vmax=20), scene_of().render().data), False)


def test_display_range_is_not_a_prediction_argument():
    """`vmin` and `vmax` belong to `Scene.plot` and `Scene.render`"""
    scene = scene_of(scotoma=Scotoma.circle(6))
    model = model_for(implant_at(0, 0))
    for kwargs in ({'vmax': 20}, {'vmin': 3}):
        with pytest.raises(TypeError):
            model.predict_percept(scene, **kwargs)
    # `render` defaults vmax to the percept max:
    percept = model.predict_percept(scene)
    npt.assert_array_equal(
        scene.render(percept=percept).data,
        scene.render(percept=percept, vmax=percept.data.max()).data)


def test_rendering_a_scene_with_a_scotoma_gives_a_composed_rgb_percept():
    scene = scene_of(scotoma=Scotoma.circle(6), scotoma_fill=0.0)
    model = model_for(implant_at(0, 0))
    rendered = scene.render(percept=model.predict_percept(scene), vmax=20)
    npt.assert_equal(rendered.is_rgb, True)
    npt.assert_equal(rendered.shape, (SCENE_PX, SCENE_PX, 3, 1))
    npt.assert_equal(rendered.data.min() >= 0, True)
    npt.assert_equal(rendered.data.max() <= 1, True)
    # Render grid, in scene coordinates:
    npt.assert_almost_equal(rendered.xdva, np.arange(-HALF, HALF + 1),
                            decimal=4)


def test_an_inpainted_scotoma_cannot_hold_a_prosthetic_percept():
    """Inpainting cannot hold a phosphene; a numeric fill works"""
    model = model_for(implant_at(0, 0))
    filled = scene_of(scotoma=Scotoma.circle(6), scotoma_fill='inpaint')
    with pytest.raises(ValueError):
        filled.render(percept=model.predict_percept(filled), vmax=20)
    numeric = scene_of(scotoma=Scotoma.circle(6), scotoma_fill=0.0)
    npt.assert_equal(
        numeric.render(percept=model.predict_percept(numeric),
                       vmax=20).is_rgb, True)


def test_the_intact_periphery_is_the_scene_exactly():
    """Outside the scotoma, the rendered scene equals the source exactly"""
    rng = np.random.default_rng(0)
    rgb = ImageStimulus(rng.random((SCENE_PX, SCENE_PX, 3)))
    scene = scene_of(rgb, scotoma=Scotoma.circle(6))
    seen = composed(model_for(implant_at(0, 0)), scene, vmax=20)
    source = rgb.data.reshape((SCENE_PX, SCENE_PX, 3))
    x, y = scene._pixel_centers()
    intact = scene.scotoma(x, y) == 0
    npt.assert_array_equal(seen[..., 0][intact], source[intact])


def test_the_phosphene_lands_where_the_electrode_looks():
    """Phosphene position and y orientation in the rendered scene"""
    visual_field_map = Curcio1990Map()
    scene = scene_of(scotoma=Scotoma.circle(12), scotoma_fill=0.0)
    for x_dva, y_dva in [(4.0, 0.0), (0.0, 4.0), (-4.0, 0.0), (0.0, -4.0)]:
        implant = implant_at(*visual_field_map.dva_to_ret(x_dva, y_dva))
        model = model_for(implant, rho=80, xrange=(-8, 8), yrange=(-8, 8),
                          step=0.5)
        frame = composed(model, scene, vmax=2)[..., 0]
        row, col = int(round(HALF - y_dva)), int(round(x_dva + HALF))
        # Bright at the electrode location, dark on the opposite side:
        npt.assert_equal(frame[row, col].mean() > 0.5, True)
        npt.assert_equal(frame[SCENE_PX - 1 - row,
                               SCENE_PX - 1 - col].mean() < 0.1, True)


def test_gaze_moves_the_scotoma_and_the_phosphene_together():
    scene = scene_of(scotoma=Scotoma.circle(4), scotoma_fill=0.3)
    model = model_for(implant_at(0, 0), rho=100, xrange=(-2, 2),
                      yrange=(-2, 2), step=0.5)
    fixating = composed(model, scene, vmax=2)[..., 0]
    shifted = composed(model, scene, vmax=2, gaze=(5, 0) * dva)[..., 0]
    # Scotoma and phosphene stay at the center of the eye-centered display:
    npt.assert_almost_equal(shifted[HALF, HALF], fixating[HALF, HALF],
                            decimal=5)
    # The scene shifts 5 deg left:
    source = scene.source.data.reshape((SCENE_PX, SCENE_PX))
    npt.assert_almost_equal(shifted[HALF, HALF - 10],
                            [source[HALF, HALF - 5]] * 3, decimal=5)
    npt.assert_almost_equal(shifted[HALF, HALF - 10],
                            fixating[HALF, HALF - 5], decimal=5)


def test_a_fixed_vmax_does_not_renormalize_when_gaze_changes():
    """A fixed vmax is not renormalized when gaze changes"""
    scene = scene_of(scotoma=Scotoma.circle(12), scotoma_fill=0.0)
    model = model_for(implant_at(0, 0), rho=100, xrange=(-4, 4),
                      yrange=(-4, 4), step=0.5)

    def phosphene(gaze_x, **kwargs):
        """Return the rendered center (foveal) pixel

        The display is eye-centered, so the center pixel is pure phosphene
        (the intact periphery would dominate a whole-frame max).
        """
        seen = composed(model, scene, gaze=(gaze_x, 0) * dva, **kwargs)
        return float(seen[HALF, HALF, 0, 0])

    dim, bright = phosphene(-16, vmax=200), phosphene(16, vmax=200)
    npt.assert_equal(0 < dim < bright < 1, True)
    # The ramp is 9x brighter at +16 than at -16:
    npt.assert_almost_equal(bright / dim, ramp_at(16) / ramp_at(-16),
                            decimal=2)
    # Doubling `vmax` halves the brightness:
    npt.assert_almost_equal(phosphene(16, vmax=400), bright / 2, decimal=3)


def test_a_video_scene_keeps_its_own_timing():
    """A video scene keeps its frame times"""
    frames = np.stack([np.full((SCENE_PX, SCENE_PX), v)
                       for v in (0.2, 0.5, 0.9)], axis=-1)
    scene = scene_of(VideoStimulus(frames, time=[0, 100, 200]),
                     scotoma=Scotoma.circle(6), scotoma_fill=0.0)
    model = model_for(implant_at(0, 0), rho=150, xrange=(-4, 4),
                      yrange=(-4, 4), step=0.5)
    percept = scene.render(percept=model.predict_percept(scene), vmax=200)
    npt.assert_equal(percept.shape, (SCENE_PX, SCENE_PX, 3, 3))
    npt.assert_almost_equal(percept.time, [0, 100, 200])
    # Brighter frames give brighter phosphenes (fovea is inside the scotoma,
    # so phosphene only):
    peaks = [percept.data[HALF, HALF, 0, f] for f in range(3)]
    npt.assert_equal(np.all(np.diff(peaks) > 0), True)
    # Outside the scotoma, video frames are unchanged:
    npt.assert_almost_equal(percept.data[0, 0, 0], [0.2, 0.5, 0.9], decimal=6)


def test_a_spatiotemporal_model_composes_against_a_video_scene():
    """A spatiotemporal model renders against a video scene"""
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
    # Percept and scene frame times differ:
    npt.assert_almost_equal(raw.time, [100, 200, 300])
    npt.assert_almost_equal(scene.time, [0, 100, 200])
    # Frames are paired by source frame time, not frame count:
    npt.assert_almost_equal(raw.metadata['source_frame_time'], [0, 100, 200])

    percept = scene.render(
        percept=spatiotemporal().predict_percept(scene), vmax=5)
    npt.assert_equal(percept.shape, (SCENE_PX, SCENE_PX, 3, 3))
    # Rendered times are the percept's output times:
    npt.assert_almost_equal(percept.time, [100, 200, 300])
    # Source frames appear in order, one per output frame:
    npt.assert_almost_equal(percept.data[0, 0, 0], [0.2, 0.5, 0.9], decimal=5)


def test_a_one_frame_video_keeps_its_source_clock():
    """A one-frame video keeps its source frame time"""
    source = VideoStimulus(np.full((SCENE_PX, SCENE_PX, 1), 0.5), time=[0])
    scene = Scene(source, fov=(SCENE_PX, SCENE_PX),
                  scotoma=Scotoma.circle(6), scotoma_fill=0.0)
    model = Model(spatial=ScoreboardSpatial(implant_at(0, 0), rho=200,
                                            xrange=(-4, 4), yrange=(-4, 4),
                                            step=0.5,
                                            visual_field_map=Curcio1990Map()),
                  temporal=FadingTemporal()).build()
    percept = model.predict_percept(scene)
    npt.assert_almost_equal(percept.metadata['source_frame_time'], [0])
    # Labeled at the frame end, paired by source frame time:
    rendered = scene.render(percept=percept, vmax=5)
    npt.assert_equal(rendered.shape[-1], 1)
    npt.assert_almost_equal(rendered.time, percept.time)


def test_a_temporal_stage_does_not_lose_the_visual_field_grid():
    """A temporal stage keeps the percept's visual field grid"""
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
    """A single timed percept frame is not broadcast over a video"""
    source = VideoStimulus(np.zeros((5, 5, 3)), time=[0, 10, 20])
    scene = Scene(source, fov=(5, 5), scotoma=Scotoma.circle(3))
    grid = ScoreboardModel(implant=implant_at(0, 0), xrange=(-2, 2),
                           yrange=(-2, 2), step=1).build().spatial.grid
    at_10 = Percept(np.full((5, 5, 1), 20.0), space=grid, time=[10])
    with pytest.raises(ValueError) as excinfo:
        scene.render(percept=at_10, vmax=20)
    npt.assert_equal('never simulated' in str(excinfo.value), True)
    # A percept without time is broadcast to every frame:
    timeless = Percept(np.full((5, 5, 1), 20.0), space=grid)
    npt.assert_equal(timeless.time, None)
    seen = scene.render(percept=timeless, vmax=20)
    npt.assert_equal(seen.shape[-1], 3)
    npt.assert_almost_equal(seen.time, [0, 10, 20])


def test_a_temporal_percept_must_cover_the_video():
    """A percept must cover the video's time range (no extrapolation)"""
    source = VideoStimulus(np.zeros((5, 5, 3)), time=[0, 10, 20])
    scene = Scene(source, fov=(5, 5), scotoma=Scotoma.circle(3))
    values = np.stack([np.full((5, 5), b) for b in (0.0, 20.0)], axis=-1)
    grid = ScoreboardModel(implant=implant_at(0, 0), xrange=(-2, 2),
                           yrange=(-2, 2), step=1).build().spatial.grid
    short = Percept(values, space=grid, time=[5, 15])
    with pytest.raises(ValueError) as excinfo:
        scene.render(percept=short, vmax=20)
    npt.assert_equal('never simulated' in str(excinfo.value), True)
    # A covering percept works, endpoints included:
    covering = Percept(values, space=grid, time=[0, 20])
    npt.assert_equal(scene.render(percept=covering, vmax=20).shape[-1], 3)
    # So does a percept without time:
    still = Percept(values[..., :1], space=grid)
    npt.assert_equal(scene.render(percept=still, vmax=20).shape[-1], 3)


def test_the_time_range_check_crosses_units():
    """The time range check converts a percept in s to a video in ms"""
    source = VideoStimulus(np.zeros((5, 5, 3)), time=[0, 10, 20])
    scene = Scene(source, fov=(5, 5), scotoma=Scotoma.circle(3))
    grid = ScoreboardModel(implant=implant_at(0, 0), xrange=(-2, 2),
                           yrange=(-2, 2), step=1).build().spatial.grid
    values = np.stack([np.full((5, 5), b) for b in (0.0, 20.0)], axis=-1)
    # 0-20 ms is exactly 0-0.02 s:
    covering = Percept(values, space=grid, time=[0, 0.02], time_unit=s)
    seen = scene.render(percept=covering, vmax=20)
    npt.assert_equal(seen.time_unit, ms)
    npt.assert_almost_equal(seen.time, [0, 10, 20])
    short = Percept(values, space=grid, time=[0, 0.01], time_unit=s)
    with pytest.raises(ValueError):
        scene.render(percept=short, vmax=20)


def labeled_percept(time, source=None, time_unit=ms):
    """Return frames of brightness 1, 2, 3, optionally with source times"""
    grid = ScoreboardModel(implant=implant_at(0, 0), xrange=(-2, 2),
                           yrange=(-2, 2), step=1).build().spatial.grid
    values = np.stack([np.full((5, 5), b) for b in (1.0, 2.0, 3.0)], axis=-1)
    meta = None if source is None else {'source_frame_time': source}
    return Percept(values, space=grid, time=time, time_unit=time_unit,
                   metadata=meta)


def test_equal_frame_counts_do_not_pair_frames():
    """Equal frame counts alone do not pair frames"""
    scene = Scene(VideoStimulus(np.zeros((5, 5, 3)), time=[0, 10, 20]),
                  fov=(5, 5), scotoma=Scotoma.circle(3))
    for source in (None, [0, 20, 40]):
        percept = labeled_percept([5, 15, 25], source=source)
        with pytest.raises(ValueError) as excinfo:
            scene.render(percept=percept, vmax=3)
        npt.assert_equal('never simulated' in str(excinfo.value), True)
        # All times are checked, even for one frame:
        with pytest.raises(ValueError):
            scene._prosthetic_frames(percept, frame=1)


def test_source_provenance_pairs_frames_by_index():
    """Source frame times pair percept frame k with scene frame k"""
    scene = Scene(VideoStimulus(np.zeros((5, 5, 3)), time=[0, 10, 20]),
                  fov=(5, 5), scotoma=Scotoma.circle(3))
    percept = labeled_percept([5, 15, 25], source=[0, 10, 20])
    frames, time, unit = scene._prosthetic_frames(percept)
    npt.assert_almost_equal(frames[2, 2], [1, 2, 3])
    npt.assert_almost_equal(time, [5, 15, 25])
    frames, time, _ = scene._prosthetic_frames(percept, frame=1)
    npt.assert_almost_equal(frames[2, 2], [2])
    npt.assert_almost_equal(time, [15])
    npt.assert_almost_equal(scene.render(percept=percept, vmax=3).time,
                            [5, 15, 25])
    # Source frame times are in ms; percept times may be in s:
    percept = labeled_percept([0.005, 0.015, 0.025], source=[0, 10, 20],
                              time_unit=s)
    frames, time, unit = scene._prosthetic_frames(percept)
    npt.assert_almost_equal(frames[2, 2], [1, 2, 3])
    npt.assert_almost_equal(time, [0.005, 0.015, 0.025])
    npt.assert_equal(unit, s)
    # A scene in s matches source frame times in ms:
    in_s = Scene(VideoStimulus(np.zeros((5, 5, 3)),
                               time=np.array([0, 0.01, 0.02]) * s),
                 fov=(5, 5), scotoma=Scotoma.circle(3))
    frames, _, _ = in_s._prosthetic_frames(
        labeled_percept([5, 15, 25], source=[0, 10, 20]))
    npt.assert_almost_equal(frames[2, 2], [1, 2, 3])


def test_a_covering_percept_is_interpolated_at_scene_times():
    scene = Scene(VideoStimulus(np.zeros((5, 5, 3)), time=[0, 10, 20]),
                  fov=(5, 5), scotoma=Scotoma.circle(3))
    percept = labeled_percept([0, 20, 40])
    frames, time, _ = scene._prosthetic_frames(percept)
    npt.assert_almost_equal(frames[2, 2], [1, 1.5, 2])
    npt.assert_almost_equal(time, [0, 10, 20])


def test_per_frame_gaze_moves_the_eye_between_video_frames():
    frames = np.repeat(ramp_source().data.reshape(
        (SCENE_PX, SCENE_PX, 1)), 3, axis=-1)
    scene = scene_of(VideoStimulus(frames, time=[0, 100, 200]))
    gaze = np.array([[-6.0, 0.0], [0.0, 0.0], [6.0, 0.0]])
    eye = model_for(implant_at(0, 0))
    seen = seen_by(eye, scene, gaze=gaze * dva)
    npt.assert_almost_equal(seen.ravel(), ramp_at(gaze[:, 0]), decimal=4)
    # Gaze does not affect a head-mounted camera:
    camera = model_for(implant_at(0, 0, input_frame='head'))
    npt.assert_almost_equal(seen_by(camera, scene, gaze=gaze * dva).ravel(),
                            [ramp_at(0.0)] * 3, decimal=4)
    # Per-frame gaze with a still scene raises ValueError:
    for model in (eye, camera):
        with pytest.raises(ValueError):
            seen_by(model, scene_of(), gaze=gaze)


def test_a_bound_implant_survives_a_deepcopy():
    """A deep-copied model keeps the same implant object"""
    from copy import deepcopy
    implant = implant_at(0, 0)
    model = model_for(implant)
    copied = deepcopy(model)
    npt.assert_equal(copied.implant is implant, True)
    npt.assert_equal(copied.is_built, True)
    npt.assert_almost_equal(seen_by(copied, scene_of()),
                            seen_by(model, scene_of()))


def offset_implant(input_frame='eye'):
    """Return three electrodes with a non-central device-local origin"""
    array = ElectrodeArray([PointSource(0, 0, 0), PointSource(280, 0, 0),
                            PointSource(560, 0, 0)])
    return Implant(array, scene_input_frame=input_frame,
                   encoder=AmplitudeEncoder(amp_range=(0, AMP_MAX)))


def placed_coords(model, implant):
    """Return the placed tissue coordinates (um) of the electrodes"""
    stim = _scene_stim(model, scene_of(), None)
    x, y, z = model.spatial._electrode_coords(implant.electrode_array, stim)
    return np.column_stack((x, y, z)).astype(float)


def test_implant_position_defaults_to_the_tissue_origin():
    """Default `implant_position` leaves the implant unmoved"""
    scene = scene_of()
    implant = implant_at(*Curcio1990Map().dva_to_ret(3.0, 1.0))
    for visual_field_map in (Curcio1990Map(), SquareMap()):
        here = model_for(implant, visual_field_map=visual_field_map)
        there = model_for(implant, visual_field_map=visual_field_map,
                          implant_position=(0, 0))
        npt.assert_array_equal(seen_by(there, scene), seen_by(here, scene))
        npt.assert_array_equal(there.predict_percept(scene).data,
                               here.predict_percept(scene).data)
        npt.assert_array_equal(placed_coords(there, implant),
                               implant.electrode_array.coordinates())


@pytest.mark.parametrize('visual_field_map', [Curcio1990Map(), SquareMap()])
def test_implant_position_lands_on_the_local_origin(visual_field_map):
    """`implant_position` places the array's local (0, 0), not its centroid"""
    implant = offset_implant()
    model = model_for(implant, visual_field_map=visual_field_map,
                      implant_position=(6, -2) * dva)
    placed = placed_coords(model, implant)
    # The electrode at the local origin lands on the requested location
    # (the centroid is 280 um away):
    npt.assert_almost_equal(visual_field_map.ret_to_dva(placed[0, 0],
                                                        placed[0, 1]),
                            (6.0, -2.0), decimal=4)
    npt.assert_almost_equal(placed[0, :2],
                            visual_field_map.dva_to_ret(6.0, -2.0), decimal=3)
    # Same result with a position in um:
    same = model_for(implant, visual_field_map=visual_field_map,
                     implant_position=visual_field_map.dva_to_ret(
                         6.0, -2.0) * um)
    npt.assert_almost_equal(placed_coords(same, implant), placed, decimal=3)


@pytest.mark.parametrize('visual_field_map', [Curcio1990Map(), SquareMap()])
def test_implant_position_translates_not_warps(visual_field_map):
    """`implant_position` is a rigid translation"""
    implant = offset_implant()
    before = implant.electrode_array.coordinates()
    placed = placed_coords(model_for(implant,
                                     visual_field_map=visual_field_map,
                                     implant_position=(7, 2) * dva), implant)
    npt.assert_almost_equal(np.diff(placed, axis=0), np.diff(before, axis=0),
                            decimal=2)
    shift = placed - before
    npt.assert_almost_equal(shift - shift[0], 0, decimal=2)
    # The implant object is unchanged:
    npt.assert_array_equal(implant.electrode_array.coordinates(), before)


def test_implant_rotation_turns_the_array_about_its_own_origin():
    """`implant_rotation` rotates about the local (0, 0), counter-clockwise"""
    implant = offset_implant()
    before = implant.electrode_array.coordinates()
    placed = placed_coords(model_for(implant, implant_rotation=90), implant)
    # The electrode at the origin stays put; the one at +280 um x moves to
    # +280 um y:
    npt.assert_almost_equal(placed[0], before[0], decimal=6)
    npt.assert_almost_equal(placed[1, :2], (0, 280), decimal=6)
    # Rigid: distances and z unchanged, implant object unchanged:
    npt.assert_almost_equal(np.linalg.norm(np.diff(placed[:, :2], axis=0),
                                           axis=1),
                            np.linalg.norm(np.diff(before[:, :2], axis=0),
                                           axis=1), decimal=6)
    npt.assert_almost_equal(placed[:, 2], before[:, 2], decimal=6)
    npt.assert_array_equal(implant.electrode_array.coordinates(), before)
    # Unitful angle gives the same result:
    npt.assert_almost_equal(
        placed_coords(model_for(implant, implant_rotation=90 * deg), implant),
        placed, decimal=6)


def test_implant_rotation_happens_before_the_translation():
    """Rotation is applied before translation"""
    implant = offset_implant()
    before = implant.electrode_array.coordinates()
    model = model_for(implant, implant_rotation=30,
                      implant_position=(400, -100) * um)
    placed = placed_coords(model, implant)
    th = np.deg2rad(30)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    npt.assert_almost_equal(placed[:, :2],
                            (R @ before[:, :2].T).T + [400, -100], decimal=3)
    # The opposite order gives a different result:
    npt.assert_equal(np.allclose(placed[:, :2],
                                 (R @ (before[:, :2] + [400, -100]).T).T),
                     False)


def test_implant_position_moves_scene_sampling_and_the_percept_alike():
    scene = scene_of(scotoma=Scotoma.circle(14), scotoma_fill=0.0)
    implant = implant_at(0, 0)
    grid = {'rho': 80, 'xrange': (-8, 8), 'yrange': (-8, 8), 'step': 0.5}
    fovea = model_for(implant, **grid)
    placed = model_for(implant, implant_position=(4, -3) * dva, **grid)
    # Same gray level as gazing 4 deg right and 3 deg down:
    npt.assert_almost_equal(seen_by(placed, scene_of()),
                            seen_by(fovea, scene_of(), gaze=(4, -3) * dva),
                            decimal=4)
    # The phosphene moves there too (row is -y, column is +x):
    here = composed(fovea, scene, vmax=2)[..., 0]
    there = composed(placed, scene, vmax=2)[..., 0]
    npt.assert_almost_equal(there[HALF + 3, HALF + 4], here[HALF, HALF],
                            decimal=4)


def test_implant_depth_translates_depth_without_flattening_the_array():
    """`implant_depth` is added to each electrode's local z"""
    array = ElectrodeArray([PointSource(0, 0, 0), PointSource(280, 0, 50),
                            PointSource(560, 0, -20)])
    implant = Implant(array,
                      encoder=AmplitudeEncoder(amp_range=(0, AMP_MAX)))
    before = implant.electrode_array.coordinates()
    placed = placed_coords(model_for(implant, implant_depth=150 * um), implant)
    npt.assert_almost_equal(placed[:, :2], before[:, :2], decimal=4)
    npt.assert_almost_equal(placed[:, 2], before[:, 2] + 150, decimal=3)
    # Local non-planarity survives:
    npt.assert_almost_equal(np.diff(placed[:, 2]), np.diff(before[:, 2]),
                            decimal=3)
    npt.assert_array_equal(implant.electrode_array.coordinates(), before)


def test_one_implant_can_be_placed_in_several_models_at_once():
    """Placement is stored on the model, so one implant can be reused"""
    implant = offset_implant()
    before = implant.electrode_array.coordinates()
    near = model_for(implant, implant_position=(2, 0) * dva, implant_depth=0)
    far = model_for(implant, implant_position=(8, 0) * dva,
                    implant_depth=200 * um)
    a, b = placed_coords(near, implant), placed_coords(far, implant)
    npt.assert_equal(np.allclose(a, b), False)
    npt.assert_almost_equal(np.diff(a, axis=0), np.diff(b, axis=0), decimal=2)
    npt.assert_almost_equal(b[:, 2] - a[:, 2], 200, decimal=3)
    npt.assert_array_equal(implant.electrode_array.coordinates(), before)
    # Placements are independent:
    npt.assert_almost_equal(placed_coords(near, implant), a, decimal=6)


def test_a_cortical_implant_is_placed_by_visual_field_position():
    """(6, -2) dva places the implant at the V1 location of that point"""
    visual_field_map = Polimeni2006Map(regions=['v1'])
    model = CortexScoreboard(implant=implant_at(0, 0),
                             implant_position=(6, -2) * dva,
                             visual_field_map=visual_field_map)
    shift = _placement_shift(model.spatial, um)
    npt.assert_almost_equal(shift[:2],
                            visual_field_map.dva_to_v1(6.0, -2.0), decimal=2)
    npt.assert_almost_equal(shift[2], 0, decimal=6)


def test_an_ambiguous_cortical_placement_is_refused():
    """A dva position with several cortical regions is ambiguous"""
    model = CortexScoreboard(implant=implant_at(0, 0), regions=['v1', 'v2'],
                             implant_position=(6, -2) * dva)
    with pytest.raises(NotImplementedError):
        _placement_shift(model.spatial, um)
    # A position in um works:
    model.spatial.implant_position = (1000, 0) * um
    npt.assert_almost_equal(_placement_shift(model.spatial, um),
                            (1000, 0, 0), decimal=6)


def test_implant_position_and_location_noise_stay_separate():
    """`implant_position` and `location_noise` are independent"""
    scene = scene_of()
    implant = grid_implant()
    plain = model_for(implant)
    placed = model_for(implant, implant_position=(5, 0) * dva)
    noisy = model_for(implant, location_noise=1.0)
    both = model_for(implant, implant_position=(5, 0) * dva,
                     location_noise=1.0)
    # `location_noise` displaces the percept, not the scene sampling points:

    npt.assert_array_equal(seen_by(noisy, scene), seen_by(plain, scene))
    npt.assert_array_equal(seen_by(both, scene), seen_by(placed, scene))
