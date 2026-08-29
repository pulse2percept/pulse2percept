"""Placing a picture in the visual field (#668)

The scenes here are laid out one degree per pixel with an odd pixel count, so
that pixel centers land on whole degrees and the center pixel sits on the
origin. Expected values are then plain arithmetic rather than a restatement of
the pixel-center convention `Scene.dva_to_pixel` already owns.
"""
import numpy as np
import numpy.testing as npt
import pytest
import matplotlib.pyplot as plt

from pulse2percept.percepts import Percept
from pulse2percept.stimuli import ImageStimulus, LogoBVL, VideoStimulus
from pulse2percept.units import dva, ms, s
from pulse2percept.vision import Scene, Scotoma

SCENE_PX = 41
HALF = (SCENE_PX - 1) // 2


def ramp_scene(**kwargs):
    """A scene whose gray level reads off x: 0 at -20 dva, 1 at +20"""
    data = np.tile(np.linspace(0, 1, SCENE_PX), (SCENE_PX, 1))
    return Scene(ImageStimulus(data), fov=(SCENE_PX, SCENE_PX), **kwargs)


def ramp_at(x_dva):
    """What `ramp_scene` shows at a scene x"""
    return (x_dva + HALF) / (2 * HALF)


def seen_at(scene, x_dva, y_dva=0.0):
    """What the fovea sees when the eye points at a scene location

    Gaze does the moving, so this reads the scene at exactly
    ``(x_dva, y_dva)`` through the same path an electrode's would take.
    """
    return float(np.ravel(scene._device_input(0.0, 0.0,
                                              gaze=(x_dva, y_dva)))[0])


def test_fov_may_be_a_scalar_a_pair_or_unitful():
    """A scalar is the horizontal extent, with square angular pixels"""
    source = ImageStimulus(np.zeros((10, 20)))
    npt.assert_almost_equal(Scene(source, fov=40).fov, (40.0, 20.0))
    npt.assert_almost_equal(Scene(source, fov=40 * dva).fov, (40.0, 20.0))
    npt.assert_almost_equal(Scene(source, fov=(40, 15)).fov, (40.0, 15.0))
    npt.assert_almost_equal(Scene(source, fov=(40, 15) * dva).fov,
                            (40.0, 15.0))


@pytest.mark.parametrize('fov', [0, -5, np.nan, np.inf, (10, 0), (10, np.nan),
                                 (1, 2, 3)])
def test_a_scene_needs_a_real_field_of_view(fov):
    """A scene with no extent, or an infinite one, is not somewhere"""
    with pytest.raises(ValueError):
        Scene(ImageStimulus(np.zeros((8, 8))), fov=fov)


def test_a_scene_requires_a_fov_at_all():
    """There is no such thing as a scene that does not say where it is"""
    with pytest.raises(TypeError):
        Scene(ImageStimulus(np.zeros((8, 8))))


def test_an_ordinary_image_is_wrapped():
    """Convenience only: a bare array is a picture, so treat it as one"""
    scene = Scene(np.zeros((10, 20)), fov=40)
    npt.assert_equal(isinstance(scene.source, ImageStimulus), True)
    npt.assert_equal(scene.shape, (10, 20))
    # An ImageStimulus subclass is left exactly as it came:
    logo = LogoBVL()
    npt.assert_equal(Scene(logo, fov=40).source is logo, True)


def test_pixel_coordinates_address_centers_inside_the_outer_extent():
    """The fov is the outer edge; pixel centers sit half a pixel inside it"""
    scene = ramp_scene()
    # The center pixel of an odd grid is on the origin:
    x, y = scene.pixel_to_dva(HALF, HALF)
    npt.assert_almost_equal([x, y], [0.0, 0.0])
    # The outermost pixel centers are half a degree inside +/-20.5:
    npt.assert_almost_equal(scene.pixel_to_dva(0, 0), [-HALF, HALF])
    npt.assert_almost_equal(scene.pixel_to_dva(SCENE_PX - 1, SCENE_PX - 1),
                            [HALF, -HALF])
    # ... and dva_to_pixel is the exact inverse, fractions included:
    for col, row in [(0, 0), (3.5, 17.25), (SCENE_PX - 1, SCENE_PX - 1)]:
        back = scene.dva_to_pixel(*scene.pixel_to_dva(col, row))
        npt.assert_almost_equal(back, [col, row], decimal=9)


def test_row_zero_is_the_top_of_the_visual_field():
    """Getting this backwards flips the world and nothing else notices"""
    data = np.tile(np.linspace(0, 1, SCENE_PX).reshape((-1, 1)),
                   (1, SCENE_PX))
    scene = Scene(ImageStimulus(data), fov=(SCENE_PX, SCENE_PX))
    for y_dva in (5.0, -5.0):
        # Row r sits at y = HALF - r, so the small values are up in the field:
        npt.assert_almost_equal(seen_at(scene, 0.0, y_dva),
                                (HALF - y_dva) / (2 * HALF), decimal=5)


def test_sampling_depends_only_on_where_you_look():
    """Scene sampling is a pure function of the eye-centered position"""
    scene = ramp_scene()
    for x_vf, gaze_x in [(4.0, 0.0), (0.0, 4.0), (10.0, -6.0), (-3.0, 7.0)]:
        npt.assert_almost_equal(
            float(np.ravel(scene._device_input(x_vf, 0.0,
                                               gaze=(gaze_x, 0)))[0]),
            ramp_at(x_vf + gaze_x), decimal=5)


def test_outside_the_scene_there_is_nothing():
    """Looking past the edge sees no picture, and none is invented"""
    scene = ramp_scene()
    npt.assert_almost_equal(seen_at(scene, 40.0), 0.0)
    npt.assert_almost_equal(seen_at(scene, 0.0, -40.0), 0.0)


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
    return Scene(ImageStimulus(data), fov=(EDGE_PX, EDGE_PX))


@pytest.mark.parametrize('sign', [1, -1])
def test_the_whole_stated_fov_belongs_to_the_scene(sign):
    """The outer half-pixel border is scene, not background"""
    scene = edge_scene('x')
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


def test_color_survives_sampling_and_greys_only_at_the_device():
    """Three channels reach the sampler; one number leaves for the device"""
    rgb = np.zeros((21, 21, 3))
    rgb[..., 0] = 1.0  # pure red everywhere
    scene = Scene(ImageStimulus(rgb), fov=(20, 20))
    values = scene._sample_at(0.0, 0.0)
    npt.assert_equal(values.shape, (1, 3, 1))
    npt.assert_almost_equal(values[0, :, 0], [1, 0, 0], decimal=5)
    # ... and the luminance of pure red only at the device boundary:
    npt.assert_almost_equal(scene._device_input(0.0, 0.0), [[0.2125]],
                            decimal=4)


def test_sampling_rgb_then_greying_matches_greying_first():
    """Moving rgb2gray after interpolation must not move the numbers"""
    rng = np.random.default_rng(0)
    rgb = ImageStimulus(rng.random((17, 23, 3)))
    x = np.linspace(-10, 10, 7)
    y = np.linspace(-6, 6, 7)
    late = Scene(rgb, fov=(30, 20))._device_input(x, y)
    gray = ImageStimulus(rgb.rgb2gray().data.reshape((17, 23)))
    early = Scene(gray, fov=(30, 20))._device_input(x, y)
    npt.assert_almost_equal(late, early, decimal=5)


def test_an_image_scene_has_one_frame_and_no_clock():
    scene = ramp_scene()
    npt.assert_equal(scene.time, None)
    npt.assert_equal(scene.n_frames, 1)
    npt.assert_equal(scene._device_input(0.0, 0.0).shape, (1, 1))


def test_a_video_scene_keeps_its_frames_and_its_clock():
    """A fixating eye sees the same scene region in every frame"""
    n_frames = 4
    vid = np.stack([np.tile(np.linspace(0, 1, 21), (21, 1)) * (f + 1) / 4
                    for f in range(n_frames)], axis=-1)
    source = VideoStimulus(vid, time=np.arange(n_frames) * 10.0)
    scene = Scene(source, fov=(20, 20))
    npt.assert_almost_equal(scene.time, [0, 10, 20, 30])
    npt.assert_equal(scene.time_unit, ms)
    npt.assert_equal(scene.n_frames, n_frames)
    values = scene._device_input(0.0, 0.0)
    npt.assert_equal(values.shape, (1, n_frames))
    # The center pixel is 0.5 scaled by the frame's own factor:
    npt.assert_almost_equal(values.ravel(),
                            0.5 * (np.arange(n_frames) + 1) / 4, decimal=5)


def test_a_scene_reports_its_sources_clock_verbatim():
    """The scene does not re-time anything; the source owns the clock"""
    source = VideoStimulus(np.zeros((4, 4, 2)), time=[0, 0.05] * s)
    scene = Scene(source, fov=(4, 4))
    npt.assert_equal(scene.time_unit, source.time_unit)
    npt.assert_almost_equal(scene.time, source.time)
    npt.assert_almost_equal(scene.time, [0, 50])


def test_gaze_may_move_between_frames():
    """One gaze per frame moves the eye across a video"""
    frames = np.repeat(np.tile(np.linspace(0, 1, SCENE_PX),
                               (SCENE_PX, 1))[..., np.newaxis], 3, axis=-1)
    scene = Scene(VideoStimulus(frames, time=[0, 10, 20]),
                  fov=(SCENE_PX, SCENE_PX))
    gaze = np.array([[-6.0, 0.0], [0.0, 0.0], [6.0, 0.0]])
    moving = scene._device_input(0.0, 0.0, gaze=gaze)
    npt.assert_almost_equal(moving.ravel(), ramp_at(gaze[:, 0]), decimal=5)
    # A static gaze is not the same thing, which is what says the per-frame
    # values were actually used:
    static = scene._device_input(0.0, 0.0, gaze=(0, 0))
    npt.assert_equal(np.allclose(moving, static), False)
    with pytest.raises(ValueError):
        scene._device_input(0.0, 0.0, gaze=np.zeros((2, 2)))


@pytest.mark.parametrize('gaze', [(np.nan, 0), (0, np.inf), (-np.inf, 0)])
def test_non_finite_gaze_is_refused(gaze):
    """A blank sample is not the right answer to 'where was the eye?'"""
    with pytest.raises(ValueError):
        ramp_scene()._device_input(0.0, 0.0, gaze=gaze)


def test_unitful_gaze_reads_the_same_place():
    scene = ramp_scene()
    npt.assert_almost_equal(seen_at(scene, 3.0),
                            float(np.ravel(scene._device_input(
                                0.0, 0.0, gaze=(3, 0) * dva))[0]))


def test_without_a_scotoma_native_vision_is_the_scene_exactly():
    scene = Scene(LogoBVL(), fov=40 * dva)
    native = scene._native_rgb()
    npt.assert_equal(native.shape, (576, 720, 3, 1))
    # The logo is RGBA, so alpha is blended against black and nothing else:
    source = scene.source.data.reshape(scene.source.img_shape)
    expected = source[..., :3] * source[..., 3:4]
    npt.assert_almost_equal(native[..., 0], expected, decimal=6)


def test_a_grayscale_scene_becomes_rgb_without_changing_intensity():
    gray = np.linspace(0, 1, SCENE_PX ** 2).reshape((SCENE_PX, SCENE_PX))
    scene = Scene(ImageStimulus(gray), fov=(SCENE_PX, SCENE_PX))
    native = scene._native_rgb()
    npt.assert_equal(native.shape, (SCENE_PX, SCENE_PX, 3, 1))
    npt.assert_almost_equal(native[..., 0], np.stack([gray] * 3, axis=-1),
                            decimal=6)


@pytest.mark.parametrize('fill', [0.0, 0.35, 1.0])
def test_a_complete_scotoma_shows_the_fill_and_nothing_else(fill):
    scene = ramp_scene(scotoma=Scotoma.circle(3), scotoma_fill=fill)
    native = scene._native_rgb()[..., 0]
    npt.assert_almost_equal(native[HALF, HALF], [fill] * 3, decimal=6)
    # Outside the scotoma the scene passes through bit for bit:
    source = np.repeat(scene.source.data.reshape(
        (SCENE_PX, SCENE_PX, 1)), 3, axis=-1)
    x, y = scene._pixel_centers()
    intact = scene.scotoma(x, y) == 0
    npt.assert_array_equal(native[intact], source[intact])


def test_a_graded_scotoma_mixes_linearly():
    scene = ramp_scene(scotoma=Scotoma(lambda x, y: np.full(np.shape(x), 0.5)),
                       scotoma_fill=0.4)
    native = scene._native_rgb()[..., 0]
    source = scene.source.data.reshape((SCENE_PX, SCENE_PX))
    npt.assert_almost_equal(native[HALF, HALF],
                            [0.5 * source[HALF, HALF] + 0.5 * 0.4] * 3,
                            decimal=6)


def test_the_scotoma_is_eye_centered_so_gaze_moves_the_scene_past_it():
    scene = ramp_scene(scotoma=Scotoma.circle(3), scotoma_fill=0.0)
    fixating = scene._native_rgb()[..., 0]
    shifted = scene._native_rgb(gaze=(5, 0) * dva)[..., 0]
    # The blind spot travelled 5 degrees right across the scene:
    npt.assert_almost_equal(shifted[HALF, HALF + 5], [0.0] * 3, decimal=6)
    npt.assert_almost_equal(fixating[HALF, HALF], [0.0] * 3, decimal=6)
    # ... and where the eye used to point is native vision again:
    source = scene.source.data.reshape((SCENE_PX, SCENE_PX))
    npt.assert_almost_equal(shifted[HALF, HALF], [source[HALF, HALF]] * 3,
                            decimal=6)


def test_scotoma_does_not_change_what_the_device_sees():
    """A camera does not go blind where its wearer has"""
    x = np.array([-15.0, -8.0, -2.0, 0.0, 3.0, 9.0, 18.0])
    y = np.array([0.0, 6.0, -4.0, 0.0, 2.0, -9.0, 5.0])
    plain = ramp_scene()
    for fill in (0.0, 0.75, 1.0):
        blind = ramp_scene(scotoma=Scotoma.circle(10), scotoma_fill=fill)
        npt.assert_array_equal(blind._sample_at(x, y), plain._sample_at(x, y))
        npt.assert_array_equal(blind._device_input(x, y),
                               plain._device_input(x, y))
    # The test only means something if some of those points are lost and some
    # are not, and if the source actually varies across them:
    loss = blind.scotoma(x, y)
    npt.assert_equal(loss.max() == 1 and loss.min() == 0, True)
    seen = np.ravel(plain._device_input(x, y))
    npt.assert_equal(np.unique(seen).size, x.size)
    # ... and native vision does change, which is what says `fill` was live:
    npt.assert_equal(np.allclose(blind._native_rgb(), plain._native_rgb()),
                     False)


def test_inpainting_a_constant_surround_gives_back_the_constant():
    """Nothing to extrapolate from a flat field but the flat field"""
    flat = np.full((31, 31), 0.6)
    scene = Scene(ImageStimulus(flat), fov=(31, 31),
                  scotoma=Scotoma.circle(5), scotoma_fill='inpaint')
    npt.assert_almost_equal(scene._native_rgb()[..., 0], 0.6, decimal=6)


@pytest.mark.parametrize('blend', [0, 2])
def test_the_inpainted_fill_knows_nothing_of_what_it_covers(blend):
    """Only the visible surround may reach what is drawn, blurred or not"""
    scotoma = Scotoma.circle(6)
    lost = ramp_scene(scotoma=scotoma)._loss_at((0, 0)) > 0
    base = np.tile(np.linspace(0, 1, SCENE_PX), (SCENE_PX, 1))
    rng = np.random.default_rng(0)
    sources = [np.where(lost, hidden, base)
               for hidden in (np.zeros_like(base), rng.random(base.shape))]
    views = [Scene(ImageStimulus(src), fov=(SCENE_PX, SCENE_PX),
                   scotoma=scotoma, scotoma_fill='inpaint',
                   scotoma_blend=blend)._native_rgb()
             for src in sources]
    npt.assert_equal(np.allclose(*sources), False)
    npt.assert_array_equal(*views)


def test_inpainting_works_in_color_and_stays_a_display_intensity():
    rgb = np.stack([np.tile(np.linspace(0, 1, 31), (31, 1)),
                    np.tile(np.linspace(1, 0, 31), (31, 1)).T,
                    np.full((31, 31), 0.5)], axis=-1)
    scene = Scene(ImageStimulus(rgb), fov=(31, 31),
                  scotoma=Scotoma.circle(4), scotoma_fill='inpaint',
                  scotoma_blend=1.5)
    native = scene._native_rgb()
    npt.assert_equal(native.shape, (31, 31, 3, 1))
    npt.assert_equal(np.all(np.isfinite(native)), True)
    npt.assert_equal(native.min() >= 0 and native.max() <= 1, True)
    npt.assert_almost_equal(native[..., 2, 0], 0.5, decimal=6)


def test_a_zero_percept_composes_to_plain_inpainted_native_vision():
    """`_compose` and `_native_rgb` fill the scotoma the same way"""
    rgb = np.stack([np.tile(np.linspace(0.1, 0.9, 31), (31, 1)),
                    np.tile(np.linspace(0.9, 0.2, 31), (31, 1)).T,
                    np.full((31, 31), 0.4)], axis=-1)
    scene = Scene(ImageStimulus(rgb), fov=(31, 31), scotoma=Scotoma.circle(4),
                  scotoma_fill='inpaint', scotoma_blend=1.5)
    # No phosphene anywhere, so the lost region is the fill and nothing else:
    dark = Percept(np.zeros((31, 31, 1)), space=scene._grid())
    npt.assert_almost_equal(scene._compose(dark, vmax=1).data,
                            scene._native_rgb(), decimal=6)


def test_inpainting_does_not_change_what_the_device_sees():
    x = np.array([-15.0, -2.0, 0.0, 3.0, 18.0])
    y = np.array([0.0, 6.0, 0.0, 2.0, 5.0])
    plain = ramp_scene()
    blind = ramp_scene(scotoma=Scotoma.circle(10), scotoma_fill='inpaint')
    npt.assert_array_equal(blind._sample_at(x, y), plain._sample_at(x, y))
    npt.assert_array_equal(blind._device_input(x, y),
                           plain._device_input(x, y))


def test_scotoma_and_fill_are_validated():
    source = ImageStimulus(np.zeros((8, 8)))
    with pytest.raises(TypeError):
        Scene(source, fov=8, scotoma='circle')
    for fill in (-0.1, 1.5, np.nan):
        with pytest.raises(ValueError):
            Scene(source, fov=8, scotoma_fill=fill)
    for blend in (-1, -0.1, np.nan, np.inf):
        with pytest.raises(ValueError):
            Scene(source, fov=8, scotoma_blend=blend)
    for fill in ('blur', 'INPAINT', 'inpainting'):
        with pytest.raises(ValueError):
            Scene(source, fov=8, scotoma_fill=fill)
    blind = Scene(source, fov=8, scotoma=Scotoma.circle(100),
                  scotoma_fill='inpaint')
    with pytest.raises(ValueError):
        blind._native_rgb()


def test_no_blend_leaves_the_boundary_as_sharp_as_the_scotoma_is():
    scotoma = Scotoma.circle(6)
    default = ramp_scene(scotoma=scotoma, scotoma_fill=0.3)
    explicit = ramp_scene(scotoma=scotoma, scotoma_fill=0.3, scotoma_blend=0)
    npt.assert_equal(default.scotoma_blend, 0.0)
    npt.assert_array_equal(explicit._native_rgb(), default._native_rgb())
    npt.assert_array_equal(np.unique(explicit._rendered_loss_at((0, 0))),
                           [0.0, 1.0])


def test_blending_softens_the_boundary_and_only_the_boundary():
    """One degree per pixel here, so a sigma of 2 px is 2 dva"""
    scene = ramp_scene(scotoma=Scotoma.circle(6), scotoma_blend=2)
    loss = scene._rendered_loss_at((0, 0))
    # Lost out to the scotoma's own 6 dva edge; the feather is outside it:
    npt.assert_almost_equal(loss[HALF, HALF], 1.0, decimal=12)
    npt.assert_almost_equal(loss[HALF, HALF + 6], 1.0, decimal=12)
    npt.assert_almost_equal(loss[HALF, HALF + 15], 0.0, decimal=12)
    npt.assert_equal(0.05 < loss[HALF, HALF + 8] < 0.95, True)
    profile = loss[HALF, HALF:HALF + 16]
    npt.assert_array_less(np.diff(profile), 1e-12)


def test_blending_reads_the_loss_field_past_the_frame_edge():
    """A scotoma that covers no pixel can still darken the frame"""
    def edge(blend):
        flat = np.full((SCENE_PX, SCENE_PX), 0.8)
        scene = Scene(ImageStimulus(flat), fov=(SCENE_PX, SCENE_PX),
                      scotoma=Scotoma.circle(4, center=(-HALF - 5, 0)),
                      scotoma_fill=0.0, scotoma_blend=blend)
        return scene._native_rgb()[HALF, 0, 0, 0]
    # Entirely off the left of the frame, so unblended it takes nothing;
    # blurred, its edge is a pixel away and reaches in:
    npt.assert_almost_equal(edge(0), 0.8, decimal=6)
    npt.assert_equal(edge(2) < 0.75, True)


def test_a_softened_boundary_shows_in_the_composed_percept():
    """The blur reaches the prosthetic path, not just native vision"""
    scene = ramp_scene(scotoma=Scotoma.circle(6), scotoma_fill=0.0,
                       scotoma_blend=2)
    bright = Percept(np.ones((SCENE_PX, SCENE_PX, 1)), space=scene._grid())
    seen = scene._compose(bright, vmax=1).data[..., 0]
    npt.assert_equal(seen[HALF, HALF, 0] > 0.98, True)
    npt.assert_almost_equal(seen[HALF, HALF + 15, 0], ramp_at(15), decimal=6)
    npt.assert_equal(ramp_at(8) < seen[HALF, HALF + 8, 0] < 1.0, True)


def test_blending_is_rendering_only():
    x = np.array([-15.0, -6.0, 0.0, 4.0, 12.0])
    y = np.array([0.0, 3.0, 0.0, -5.0, 7.0])
    scotoma = Scotoma.circle(6)
    sharp = ramp_scene(scotoma=scotoma, scotoma_blend=0)
    soft = ramp_scene(scotoma=scotoma, scotoma_blend=3)
    npt.assert_array_equal(soft._sample_at(x, y), sharp._sample_at(x, y))
    npt.assert_array_equal(soft._device_input(x, y), sharp._device_input(x, y))
    npt.assert_array_equal(soft.scotoma(x, y), sharp.scotoma(x, y))
    npt.assert_equal(np.allclose(soft._native_rgb(), sharp._native_rgb()),
                     False)


def test_plot_draws_native_vision_where_the_eye_is_pointing():
    """The drawn image is the residual view, on visual-field axes"""
    scene = ramp_scene(scotoma=Scotoma.circle(3), scotoma_fill=0.0)
    ax = scene.plot(gaze=(5, 0) * dva)
    drawn = ax.images[-1].get_array()
    npt.assert_almost_equal(drawn, scene._native_rgb(gaze=(5, 0))[..., 0],
                            decimal=6)
    # Row 0 of the drawn array is the top of the field, and the axes say dva:
    npt.assert_equal(ax.images[-1].origin, 'upper')
    npt.assert_almost_equal(ax.get_xlim(), (-HALF, HALF))
    npt.assert_equal('degrees of visual angle' in ax.get_xlabel(), True)
    plt.close('all')


def test_plot_picks_a_frame_of_a_video_scene():
    frames = np.stack([np.full((6, 6), v) for v in (0.2, 0.8)], axis=-1)
    scene = Scene(VideoStimulus(frames, time=[0, 10]), fov=(6, 6))
    for frame, value in enumerate((0.2, 0.8)):
        ax = scene.plot(frame=frame)
        npt.assert_almost_equal(ax.images[-1].get_array()[0, 0],
                                [value] * 3, decimal=6)
        plt.close('all')
    with pytest.raises(ValueError):
        scene.plot(frame=2)


def test_play_animates_a_video_scene_and_refuses_a_still_one():
    frames = np.stack([np.full((6, 6), v) for v in (0.2, 0.8)], axis=-1)
    scene = Scene(VideoStimulus(frames, time=[0, 10]), fov=(6, 6),
                  scotoma=Scotoma.circle(1))
    ani = scene.play()
    npt.assert_equal(ani._frame_data.shape, (6, 6, 3, 2))
    plt.close('all')
    with pytest.raises(ValueError):
        ramp_scene().play()
