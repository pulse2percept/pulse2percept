"""Registering a scene against an implant through a retinal map

Phase 3 of #668: an image that states a ``fov`` is a picture *of* somewhere in
the visual field, so its pixels reach the electrodes that actually see them
rather than being stretched across the implant's bounding box.
"""
import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.implants import (ElectrodeGrid, PointSource,
                                    ProsthesisSystem)
from pulse2percept.stimuli import (AmplitudeEncoder, ImageStimulus,
                                   VideoStimulus)
from pulse2percept.stimuli.encoders import _as_luminance
from pulse2percept.topography import Curcio1990Map, RetinalMap, Watson2014Map
from pulse2percept.units import dva


class SquareMap(RetinalMap):
    """A retinal map that is neither 280 um/dva nor linear

    Retinal x grows as the square of eccentricity, so a registration that
    assumes any fixed micron-per-degree ratio lands in the wrong place.
    """

    def dva_to_ret(self, xdva, ydva):
        return np.sign(xdva) * 100.0 * xdva ** 2, -100.0 * ydva

    def ret_to_dva(self, xret, yret):
        return np.sign(xret) * np.sqrt(np.abs(xret) / 100.0), -yret / 100.0


def one_electrode_at(x_um, y_um):
    """An implant whose single electrode sits where we want to look"""
    return ProsthesisSystem(PointSource(x_um, y_um, 0))


#: A square scene laid out so that one pixel is exactly one degree and the
#: center pixel sits on the origin: pixel column c is at x = c - HALF dva, and
#: row r at y = HALF - r. That keeps the expected values below plain
#: arithmetic, independent of the pixel-center convention `dva_to_pixel`
#: already owns (and that Phase 1 tests).
SCENE_PX = 41
HALF = (SCENE_PX - 1) // 2


def ramp_scene():
    """A scene whose gray level reads off x: 0 at -20 dva, 1 at +20"""
    data = np.tile(np.linspace(0, 1, SCENE_PX), (SCENE_PX, 1))
    return ImageStimulus(data, fov=(SCENE_PX, SCENE_PX))


def ramp_at(x_dva):
    """What `ramp_scene` reads at a visual-field x"""
    return (x_dva + HALF) / (2 * HALF)


def sampled(implant, stim, **kwargs):
    """The one gray level a one-electrode implant sees"""
    return float(implant.reshape_stim(stim, **kwargs).data.ravel()[0])


def test_centered_implant_and_gaze_sample_the_center_pixel():
    """An electrode on the fovea, looking at the origin, sees the middle"""
    npt.assert_almost_equal(
        sampled(one_electrode_at(0, 0), ramp_scene(), vfmap=Curcio1990Map()),
        ramp_at(0), decimal=6)


@pytest.mark.parametrize('x_dva', [-5.0, 2.5, 7.0])
def test_implant_offset_moves_through_the_map(x_dva):
    """A retinal offset lands where the map says it lands, not where we guess

    The electrode is placed at the retinal image of a known visual-field
    position, so the pixel it should see is known independently of the map.
    """
    vfmap = Curcio1990Map()
    x_ret, y_ret = vfmap.dva_to_ret(x_dva, 0.0)
    implant = one_electrode_at(x_ret, y_ret)
    npt.assert_almost_equal(sampled(implant, ramp_scene(), vfmap=vfmap),
                            ramp_at(x_dva), decimal=5)


def test_gaze_moves_the_scene_not_the_implant():
    """Gaze is the scene point on the fovea, so scene = visual field + gaze"""
    scene = ramp_scene()
    vfmap = Curcio1990Map()
    implant = one_electrode_at(0, 0)
    for gaze_x in (-4.0, 0.0, 6.0):
        # A foveal electrode sees exactly whatever the eye is pointing at:
        npt.assert_almost_equal(
            sampled(implant, scene, vfmap=vfmap, gaze=(gaze_x, 0)),
            ramp_at(gaze_x), decimal=5)
    # Several electrodes keep their separation in the visual field whatever
    # the gaze: shifting gaze shifts what all of them see by the same amount.
    grid = ProsthesisSystem(ElectrodeGrid((1, 3), 280))
    here = grid.reshape_stim(scene, vfmap=vfmap).data.ravel()
    there = grid.reshape_stim(scene, vfmap=vfmap, gaze=(2, 0)).data.ravel()
    npt.assert_almost_equal(np.diff(here), np.diff(there), decimal=5)
    npt.assert_almost_equal(there - here, ramp_at(2) - ramp_at(0), decimal=5)


def test_registration_uses_the_supplied_map():
    """Not 280 um/dva, and not a linear map either"""
    scene = ramp_scene()
    # 4 dva out sits at 100 * 4**2 = 1600 um on this map:
    implant = one_electrode_at(1600.0, 0)
    npt.assert_almost_equal(sampled(implant, scene, vfmap=SquareMap()),
                            ramp_at(4.0), decimal=5)
    # Curcio would have read the same electrode as 1600/280 = 5.71 dva:
    npt.assert_almost_equal(sampled(implant, scene, vfmap=Curcio1990Map()),
                            ramp_at(1600.0 / 280.0), decimal=5)


def test_y_axis_orientation_survives_the_map():
    """Row 0 of the scene is +y in the visual field, both sides of the map"""
    data = np.tile(np.linspace(0, 1, SCENE_PX).reshape((-1, 1)),
                   (1, SCENE_PX))
    scene = ImageStimulus(data, fov=(SCENE_PX, SCENE_PX))
    vfmap = Curcio1990Map()
    # Row r sits at y = HALF - r, so the small values are up in the field:
    for y_dva in (5.0, -5.0):
        x_ret, y_ret = vfmap.dva_to_ret(0.0, y_dva)
        npt.assert_almost_equal(sampled(one_electrode_at(x_ret, y_ret), scene,
                                        vfmap=vfmap),
                                (HALF - y_dva) / (2 * HALF), decimal=5)


def test_outside_the_scene_is_dark():
    """An electrode looking past the edge of the scene sees nothing"""
    vfmap = Curcio1990Map()
    x_ret, _ = vfmap.dva_to_ret(40.0, 0.0)
    npt.assert_almost_equal(sampled(one_electrode_at(x_ret, 0), ramp_scene(),
                                    vfmap=vfmap), 0.0)


def test_rgb_survives_sampling_and_greys_at_the_encoder():
    """Color reaches the electrodes; the encoder makes it one number"""
    rgb = np.zeros((21, 21, 3))
    rgb[..., 0] = 1.0  # pure red everywhere
    scene = ImageStimulus(rgb, fov=(20, 20))
    values = one_electrode_at(0, 0)._sample_source(scene,
                                                   vfmap=Curcio1990Map())
    # Three channels per electrode per frame, still red:
    npt.assert_equal(values.shape, (1, 3, 1))
    npt.assert_almost_equal(values[0, :, 0], [1, 0, 0], decimal=5)
    # ... and the luminance of pure red only at the encoder boundary:
    npt.assert_almost_equal(_as_luminance(values), [[0.2125]], decimal=4)


@pytest.mark.parametrize('registered', [False, True])
def test_sampling_rgb_then_greying_matches_greying_first(registered):
    """Moving rgb2gray after interpolation must not move the numbers

    Both steps are linear, so they commute; this is what says the seam is a
    refactor and not a change of result.
    """
    rng = np.random.default_rng(0)
    rgb = rng.random((17, 23, 3))
    implant = ProsthesisSystem(ElectrodeGrid((4, 5), 300))
    kwargs = {'vfmap': Curcio1990Map()} if registered else {}
    fov = (30, 20) if registered else None
    scene = ImageStimulus(rgb, fov=fov)
    late = implant.reshape_stim(scene, **kwargs).data
    gray = ImageStimulus(scene.rgb2gray().data.reshape((17, 23)), fov=fov)
    early = implant.reshape_stim(gray, **kwargs).data
    npt.assert_almost_equal(late, early, decimal=5)


def test_video_registers_frame_by_frame():
    """A fixating eye sees the same scene region in every frame"""
    n_frames = 4
    vid = np.stack([np.tile(np.linspace(0, 1, 21), (21, 1)) * (f + 1) / 4
                    for f in range(n_frames)], axis=-1)
    scene = VideoStimulus(vid, fov=(20, 20), time=np.arange(n_frames) * 10.0)
    out = one_electrode_at(0, 0).reshape_stim(scene, vfmap=Curcio1990Map())
    npt.assert_equal(out.data.shape, (1, n_frames))
    npt.assert_almost_equal(out.time, np.arange(n_frames) * 10.0)
    # The center pixel is 0.5 scaled by the frame's own factor:
    npt.assert_almost_equal(out.data.ravel(),
                            0.5 * (np.arange(n_frames) + 1) / 4, decimal=5)


def test_gaze_may_move_between_frames():
    """One gaze per frame moves the eye across a still scene"""
    frames = np.repeat(ramp_scene().data.reshape((SCENE_PX, SCENE_PX, 1)),
                       3, axis=-1)
    scene = VideoStimulus(frames, fov=(SCENE_PX, SCENE_PX), time=[0, 10, 20])
    implant = one_electrode_at(0, 0)
    gaze = np.array([[-6.0, 0.0], [0.0, 0.0], [6.0, 0.0]])
    out = implant.reshape_stim(scene, vfmap=Curcio1990Map(), gaze=gaze)
    npt.assert_almost_equal(out.data.ravel(), ramp_at(gaze[:, 0]), decimal=5)
    # A static gaze is not the same thing, which is what says the per-frame
    # values were actually used:
    static = implant.reshape_stim(scene, vfmap=Curcio1990Map(), gaze=(0, 0))
    npt.assert_equal(np.allclose(out.data, static.data), False)
    with pytest.raises(ValueError):
        implant.reshape_stim(scene, vfmap=Curcio1990Map(),
                             gaze=np.zeros((2, 2)))


def test_fov_without_a_map_fails_loudly():
    """A registered scene must never fall back on the stretch"""
    scene = ramp_scene()
    implant = one_electrode_at(0, 0)
    with pytest.raises(ValueError) as excinfo:
        implant.reshape_stim(scene)
    npt.assert_equal('field of view' in str(excinfo.value), True)
    npt.assert_equal('vfmap' in str(excinfo.value), True)
    # The same guard on the path an implant takes on its own:
    implant.encoder = AmplitudeEncoder()
    with pytest.raises(ValueError):
        implant.stim = scene


def test_a_map_without_a_fov_fails_loudly():
    """Registration needs a scene to register, not just a picture"""
    implant = one_electrode_at(0, 0)
    plain = ImageStimulus(np.ones((8, 8)))
    with pytest.raises(ValueError) as excinfo:
        implant.reshape_stim(plain, vfmap=Curcio1990Map())
    npt.assert_equal('field of view' in str(excinfo.value), True)
    with pytest.raises(ValueError):
        implant.reshape_stim(plain, gaze=(1, 0))
    # ... and an encoder asked to register without an implant to place:
    with pytest.raises(ValueError):
        AmplitudeEncoder().encode(plain, vfmap=Curcio1990Map())


def test_encoder_registers_the_same_way_reshape_does():
    """encode(vfmap=, gaze=) is the explicit path to the same sampling"""
    scene = ramp_scene()
    implant = ProsthesisSystem(ElectrodeGrid((3, 3), 300))
    vfmap = Curcio1990Map()
    enc = AmplitudeEncoder(amp_range=(0, 50)).encode(
        scene, implant=implant, vfmap=vfmap, gaze=(2, -1) * dva)
    direct = AmplitudeEncoder(amp_range=(0, 50)).encode(
        implant.reshape_stim(scene, vfmap=vfmap, gaze=(2, -1)))
    npt.assert_almost_equal(enc.data, direct.data, decimal=4)
    npt.assert_equal(list(enc.electrodes), list(implant.electrode_names))


def test_unitful_gaze_and_a_nonlinear_map_together():
    """Units at the boundary, and no 280 um/dva assumption behind it"""
    scene = ramp_scene()
    implant = one_electrode_at(0, 0)
    plain = sampled(implant, scene, vfmap=Watson2014Map(), gaze=(3, 0))
    unitful = sampled(implant, scene, vfmap=Watson2014Map(),
                      gaze=(3, 0) * dva)
    npt.assert_almost_equal(plain, unitful)
    npt.assert_almost_equal(plain, ramp_at(3.0), decimal=5)


def test_legacy_stretch_is_untouched():
    """A picture without a fov still spans the electrode bounding box"""
    implant = ProsthesisSystem(ElectrodeGrid((5, 5), 400))
    data = np.tile(np.linspace(0, 1, 5), (5, 1))
    out = implant.reshape_stim(ImageStimulus(data))
    # One image pixel per electrode column, so the ramp comes back verbatim:
    npt.assert_almost_equal(out.data.ravel().reshape((5, 5))[0],
                            np.linspace(0, 1, 5), decimal=5)
    npt.assert_equal(out.time, None)


#: A 4x4 scene one degree per pixel: columns sit at x = -1.5, -0.5, 0.5, 1.5
#: and rows at y = 1.5, 0.5, -0.5, -1.5, so the outer extent runs to +/-2.
EDGE_PX = 4
EDGE_EXTENT = EDGE_PX / 2


def edge_scene(along='x'):
    """A 4x4 scene whose value reads off the pixel index, 0.25 to 1.0

    Nonzero everywhere, so that a sample falling off the scene (which reads 0)
    cannot be mistaken for a pixel of the scene.
    """
    ramp = np.linspace(0.25, 1.0, EDGE_PX)
    data = (np.tile(ramp, (EDGE_PX, 1)) if along == 'x'
            else np.tile(ramp.reshape((-1, 1)), (1, EDGE_PX)))
    return ImageStimulus(data, fov=(EDGE_PX, EDGE_PX))


def seen_at(scene, x_dva, y_dva=0.0):
    """What a foveal electrode sees when the eye points at a scene location

    Gaze does the moving here, so this reads the scene at (x_dva, y_dva)
    exactly, through the whole retina -> visual field -> scene chain.
    """
    return sampled(one_electrode_at(0, 0), scene, vfmap=Curcio1990Map(),
                   gaze=(x_dva, y_dva))


@pytest.mark.parametrize('sign', [1, -1])
def test_the_whole_stated_fov_belongs_to_the_scene(sign):
    """The outer half-pixel border is scene, not background

    A fov is the image's *outer* extent, so it reaches half a pixel past the
    outermost pixel centers. Interpolation stops at those centers, so without
    care that border strip comes back black -- a strip of the scene the user
    said was there.
    """
    scene = edge_scene('x')
    # The value of the edge pixel this side of the scene:
    edge_value = 1.0 if sign > 0 else 0.25
    last_center = sign * (EDGE_EXTENT - 0.5)
    npt.assert_almost_equal(seen_at(scene, last_center), edge_value, decimal=5)
    # Between the last pixel center and the outer edge: still the scene, and
    # it takes the value of the pixel it is inside rather than extrapolating:
    npt.assert_almost_equal(seen_at(scene, sign * (EDGE_EXTENT - 0.25)),
                            edge_value, decimal=5)
    # Exactly on the outer edge is the last point that is still the scene:
    npt.assert_almost_equal(seen_at(scene, sign * EDGE_EXTENT), edge_value,
                            decimal=5)
    # Just past it there is no scene left to sample:
    npt.assert_almost_equal(seen_at(scene, sign * (EDGE_EXTENT + 0.01)), 0.0)


@pytest.mark.parametrize('sign', [1, -1])
def test_the_vertical_fov_reaches_its_edges_too(sign):
    """Same border rule on y, where row 0 is +y"""
    scene = edge_scene('y')
    # Row 0 holds 0.25 and sits at the top, so +y is the small value:
    edge_value = 0.25 if sign > 0 else 1.0
    for y in (sign * (EDGE_EXTENT - 0.5), sign * (EDGE_EXTENT - 0.25),
              sign * EDGE_EXTENT):
        npt.assert_almost_equal(seen_at(scene, 0.0, y), edge_value, decimal=5)
    npt.assert_almost_equal(seen_at(scene, 0.0, sign * (EDGE_EXTENT + 0.01)),
                            0.0)


def test_a_corner_outside_the_fov_is_outside_even_on_one_axis():
    """Inside on x is not inside: a point off the scene in y is off it"""
    scene = edge_scene('x')
    npt.assert_almost_equal(seen_at(scene, 0.0, 1.9), 0.625, decimal=5)
    npt.assert_almost_equal(seen_at(scene, 0.0, 2.1), 0.0)
    npt.assert_almost_equal(seen_at(scene, 2.1, 1.9), 0.0)


def test_interior_sampling_still_interpolates():
    """Clamping the border must not flatten the inside of the scene"""
    scene = edge_scene('x')
    # Halfway between the two middle pixel centers (0.5 and 0.75):
    npt.assert_almost_equal(seen_at(scene, 0.0), 0.625, decimal=5)
    # A quarter of the way from the second pixel to the third:
    npt.assert_almost_equal(seen_at(scene, -0.25), 0.5625, decimal=5)


@pytest.mark.parametrize('gaze', [(np.nan, 0), (0, np.inf), (-np.inf, 0)])
def test_non_finite_gaze_is_refused(gaze):
    """A blank percept is not the right answer to 'where was the eye?'"""
    with pytest.raises(ValueError):
        one_electrode_at(0, 0).reshape_stim(ramp_scene(),
                                            vfmap=Curcio1990Map(), gaze=gaze)
