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
from pulse2percept.stimuli import ImageStimulus, VideoStimulus, samples
from pulse2percept.topography import Grid2D
from pulse2percept.units import dva, ms, s
from pulse2percept.vision import Scene, Scotoma
from pulse2percept.vision.scene import (_raster_axes, _raster_step,
                                        _ring_radii)

SCENE_PX = 41
HALF = (SCENE_PX - 1) // 2


def ramp_scene(**kwargs):
    """A scene whose gray level reads off x: 0 at -20 dva, 1 at +20"""
    data = np.tile(np.linspace(0, 1, SCENE_PX), (SCENE_PX, 1))
    kwargs.setdefault('scotoma_blend', 0)
    return Scene(ImageStimulus(data), fov=(SCENE_PX, SCENE_PX), **kwargs)


def rgba_source():
    """An opaque red square on a transparent surround that is also red

    The surround's color is what a background must override, so that reading
    it back says the alpha was honored rather than merely copied.
    """
    img = np.zeros((8, 8, 4))
    img[..., 0] = 1.0
    img[2:6, 2:6, 3] = 1.0
    return ImageStimulus(img)


def ramp_at(x_dva):
    """What `ramp_scene` shows at a scene x"""
    return (x_dva + HALF) / (2 * HALF)


def rendered_loss(scene, gaze=(0, 0)):
    """The drawn loss map, on the scene's own raster"""
    return scene._rendered_loss_on(*scene._axes, gaze)


def rendered(scene, **kwargs):
    """`Scene.render` on the source raster, as a bare RGB array"""
    return scene.render(**kwargs).data


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
    logo = samples.logo_bvl()
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
    scene = Scene(samples.logo_bvl(), fov=40 * dva)
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


def test_the_inpainted_fill_knows_nothing_of_what_it_covers():
    """Only the visible surround may reach the filled region"""
    scotoma = Scotoma.circle(6)
    lost = rendered_loss(ramp_scene(scotoma=scotoma)) > 0
    base = np.tile(np.linspace(0, 1, SCENE_PX), (SCENE_PX, 1))
    rng = np.random.default_rng(0)
    sources = [np.where(lost, hidden, base)
               for hidden in (np.zeros_like(base), rng.random(base.shape))]
    views = [Scene(ImageStimulus(src), fov=(SCENE_PX, SCENE_PX),
                   scotoma=scotoma, scotoma_fill='inpaint')._native_rgb()
             for src in sources]
    npt.assert_equal(np.allclose(*sources), False)
    npt.assert_array_equal(*views)


def test_inpainting_works_in_color_and_stays_a_display_intensity():
    rgb = np.stack([np.tile(np.linspace(0, 1, 31), (31, 1)),
                    np.tile(np.linspace(1, 0, 31), (31, 1)).T,
                    np.full((31, 31), 0.5)], axis=-1)
    scene = Scene(ImageStimulus(rgb), fov=(31, 31),
                  scotoma=Scotoma.circle(4), scotoma_fill='inpaint')
    native = scene._native_rgb()
    npt.assert_equal(native.shape, (31, 31, 3, 1))
    npt.assert_equal(np.all(np.isfinite(native)), True)
    npt.assert_equal(native.min() >= 0 and native.max() <= 1, True)
    npt.assert_almost_equal(native[..., 2, 0], 0.5, decimal=6)


def test_an_inpainted_scene_refuses_to_compose_a_prosthetic_percept():
    """Inpainted scenes refuse prosthetic composition."""
    rgb = np.stack([np.tile(np.linspace(0.1, 0.9, 31), (31, 1)),
                    np.tile(np.linspace(0.9, 0.2, 31), (31, 1)).T,
                    np.full((31, 31), 0.4)], axis=-1)
    scene = Scene(ImageStimulus(rgb), fov=(31, 31), scotoma=Scotoma.circle(4),
                  scotoma_fill='inpaint')
    dark = Percept(np.zeros((31, 31, 1)), space=scene._grid())
    for draw in (scene.render, lambda **kw: scene.plot(**kw)):
        with pytest.raises(ValueError):
            draw(percept=dark, vmax=1)
    plt.close('all')
    # The same scene with a numeric fill composes, and native vision is
    # unaffected either way:
    numeric = Scene(ImageStimulus(rgb), fov=(31, 31),
                    scotoma=Scotoma.circle(4), scotoma_fill=0.0)
    npt.assert_equal(rendered(numeric, percept=dark, vmax=1).shape,
                     (31, 31, 3, 1))
    npt.assert_equal(np.all(np.isfinite(scene._native_rgb())), True)


def test_inpainting_ignores_the_blend():
    """A softened boundary would mix the covered pixels back into the fill"""
    views = [ramp_scene(scotoma=Scotoma.circle(6), scotoma_fill='inpaint',
                        scotoma_blend=blend)._native_rgb()
             for blend in (0, 5)]
    npt.assert_array_equal(*views)
    numeric = [ramp_scene(scotoma=Scotoma.circle(6), scotoma_fill=0.0,
                          scotoma_blend=blend)._native_rgb()
               for blend in (0, 5)]
    npt.assert_equal(np.allclose(*numeric), False)


def test_inpainting_does_not_change_what_the_device_sees():
    x = np.array([-15.0, -2.0, 0.0, 3.0, 18.0])
    y = np.array([0.0, 6.0, 0.0, 2.0, 5.0])
    plain = ramp_scene()
    blind = ramp_scene(scotoma=Scotoma.circle(10), scotoma_fill='inpaint')
    npt.assert_array_equal(blind._sample_at(x, y), plain._sample_at(x, y))
    npt.assert_array_equal(blind._device_input(x, y),
                           plain._device_input(x, y))


@pytest.mark.parametrize('background, surround', [
    (0, (0, 0, 0)),
    (1, (1, 1, 1)),
    (0.5, (0.5, 0.5, 0.5)),
    ((1, 1, 1), (1, 1, 1)),
    ((0.2, 0.4, 0.6), (0.2, 0.4, 0.6)),
])
def test_a_transparent_source_shows_the_scenes_background(background,
                                                          surround):
    scene = Scene(rgba_source(), fov=8, background=background)
    frames = scene._frames()[..., 0]
    npt.assert_almost_equal(frames[0, 0], surround, decimal=6)
    npt.assert_almost_equal(frames[4, 4], (1, 0, 0), decimal=6)


def test_background_does_nothing_without_transparency():
    gray = np.linspace(0, 1, 64).reshape((8, 8))
    rgb = np.stack([gray, gray[::-1], np.full((8, 8), 0.3)], axis=-1)
    for source in (gray, rgb):
        plain = Scene(ImageStimulus(source), fov=8)
        white = Scene(ImageStimulus(source), fov=8, background=1)
        npt.assert_array_equal(plain._frames(), white._frames())


@pytest.mark.parametrize('background', [-0.1, 1.5, np.nan, (0, 0),
                                        (0, 0, 0, 0)])
def test_a_bad_background_is_refused(background):
    with pytest.raises(ValueError):
        Scene(ImageStimulus(np.zeros((8, 8))), fov=8, background=background)


def drawn_on_fresh_axes(scene, **kwargs):
    """Plot onto axes of its own, so artists cannot accumulate across calls"""
    return scene.plot(ax=plt.subplots()[1], **kwargs)


def ring_radii(ax, center=(0, 0)):
    """Read the eccentricities back off the drawn rings"""
    offsets = [np.asarray(line.get_data()) - np.reshape(center, (2, 1))
               for line in ax.get_lines()]
    return sorted(np.hypot(*offset).mean() for offset in offsets)


def video_scene(**kwargs):
    """Two frames of the ramp, one second apart"""
    ramp = np.tile(np.linspace(0, 1, SCENE_PX), (SCENE_PX, 1))
    frames = np.stack([ramp, ramp[::-1]], axis=-1)
    return Scene(VideoStimulus(frames, time=[0, 1000]),
                 fov=(SCENE_PX, SCENE_PX), **kwargs)


def test_rings_are_drawn_only_when_asked():
    scene = ramp_scene()
    for rings in (False, None):
        npt.assert_equal(len(drawn_on_fresh_axes(scene, rings=rings).lines), 0)
    # A 41-degree field holds four 5-degree rings:
    ax = drawn_on_fresh_axes(scene, rings=True)
    npt.assert_almost_equal(ring_radii(ax), [5, 10, 15, 20], decimal=6)
    npt.assert_equal([t.get_text() for t in ax.texts],
                     ['5\N{DEGREE SIGN} ecc', '10\N{DEGREE SIGN} ecc',
                      '15\N{DEGREE SIGN} ecc', '20\N{DEGREE SIGN} ecc'])
    # Understated by default, so they read as a reference grid:
    line = ax.get_lines()[0]
    npt.assert_equal(line.get_linestyle(), '--')
    npt.assert_equal(line.get_linewidth() < 1, True)
    plt.close('all')


def test_rings_takes_a_spacing_or_the_eccentricities_themselves():
    scene = ramp_scene()
    npt.assert_almost_equal(ring_radii(drawn_on_fresh_axes(scene, rings=10)),
                            [10, 20], decimal=6)
    ax = drawn_on_fresh_axes(scene, rings=[15, 5, 25])
    # Explicit eccentricities are drawn as asked, in order, even the one that
    # falls outside the field:
    npt.assert_almost_equal(ring_radii(ax), [5, 15, 25], decimal=6)
    plt.close('all')


def test_rings_take_a_color():
    scene = ramp_scene()
    ax = drawn_on_fresh_axes(scene, rings=True, ring_color='white')
    for artist in ax.get_lines() + list(ax.texts):
        npt.assert_equal(artist.get_color(), 'white')
    plt.close('all')
    # It reaches the rasterized overlay the player is handed, too:
    video = video_scene()
    black = video.play(rings=[10], ring_color='black')._frame_data
    white = video.play(rings=[10], ring_color='white')._frame_data
    npt.assert_equal(np.allclose(black, white), False)
    plt.close('all')


def test_rings_mark_eccentricity_so_they_follow_the_fovea():
    scene = ramp_scene()
    ax = drawn_on_fresh_axes(scene, gaze=(3, -2) * dva, rings=True)
    npt.assert_almost_equal(ring_radii(ax, center=(3, -2)),
                            [5, 10, 15, 20], decimal=6)
    npt.assert_array_equal(ax.images[-1].get_array(),
                           scene._native_rgb(gaze=(3, -2))[..., 0])
    plt.close('all')


def test_rings_fit_the_shorter_half_of_the_field():
    tall = Scene(ImageStimulus(np.zeros((40, 20))), fov=(20, 40))
    npt.assert_almost_equal(ring_radii(drawn_on_fresh_axes(tall, rings=4)),
                            [4, 8], decimal=6)
    # Nothing fits inside a field smaller than one step:
    small = Scene(ImageStimulus(np.zeros((8, 8))), fov=8)
    npt.assert_equal(len(drawn_on_fresh_axes(small, rings=True).lines), 0)
    plt.close('all')


@pytest.mark.parametrize('rings', [0, -5, np.nan, [], [5, 0], [5, np.inf]])
def test_bad_rings_are_refused(rings):
    with pytest.raises(ValueError):
        drawn_on_fresh_axes(ramp_scene(), rings=rings)
    plt.close('all')


def test_play_paints_readable_rings_into_the_frames_the_player_shows():
    """The player's canvas covers the figure, so rings must be in the frames

    White scene, 3 pixels per degree, so a 10-degree ring is 30 pixels out.
    """
    white = np.full((120, 120), 1.0)
    scene = Scene(VideoStimulus(np.stack([white, white], axis=-1),
                                time=[0, 1000]), fov=(40, 40))
    plain = scene.play()._frame_data
    ringed = scene.play(rings=[10])._frame_data
    npt.assert_array_equal(plain, scene._native_rgb())
    contrast = (plain - ringed).max(axis=(2, 3))
    # The ring has to read against what it is drawn on, not merely differ:
    npt.assert_equal(contrast.max() > 0.4, True)
    npt.assert_equal(np.count_nonzero(contrast > 0.2) > 50, True)
    rows, cols = np.nonzero(contrast > 0.2)
    radius = np.hypot(cols - 60, rows - 60)
    npt.assert_equal(20 < radius.min() < 32, True)
    # The label sits above the top of the ring, and the corners stay clean:
    npt.assert_equal((contrast[:30] > 0.2).any(), True)
    npt.assert_almost_equal(contrast[0, 0], 0.0, decimal=6)
    plt.close('all')


def test_play_refuses_rings_on_a_gaze_that_moves():
    """The player draws its static artists once, so they cannot follow"""
    scene = video_scene()
    with pytest.raises(ValueError):
        scene.play(gaze=[(0, 0), (5, 5)], rings=True)
    # The same moving gaze is fine without them:
    scene.play(gaze=[(0, 0), (5, 5)])
    plt.close('all')


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
    source = ImageStimulus(np.zeros((8, 8)))
    npt.assert_equal(Scene(source, fov=8).scotoma_blend, 0.5)
    npt.assert_equal(Scene(source, fov=8, scotoma_blend=1.5 * dva)
                     .scotoma_blend, 1.5)
    sharp = ramp_scene(scotoma=Scotoma.circle(6), scotoma_fill=0.3,
                       scotoma_blend=0)
    npt.assert_array_equal(np.unique(rendered_loss(sharp)),
                           [0.0, 1.0])


def test_blending_softens_the_boundary_and_only_the_boundary():
    """One degree per pixel here, so a sigma of 2 dva is 2 px"""
    scene = ramp_scene(scotoma=Scotoma.circle(6), scotoma_blend=2)
    loss = rendered_loss(scene)
    # 3 sigmas: inside the lost field, 4 sigmas: outside it
    npt.assert_equal(loss[HALF, HALF] > 0.98, True)
    npt.assert_almost_equal(loss[HALF, HALF + 15], 0.0, decimal=12)
    npt.assert_equal(0.05 < loss[HALF, HALF + 6] < 0.95, True)
    profile = loss[HALF, HALF:HALF + 16]
    npt.assert_array_less(np.diff(profile), 1e-12)


def test_a_numeric_fill_has_no_hard_contour_at_the_boundary():
    """A flat scene behind a blurred scotoma is a ramp, not a step"""
    flat = np.full((SCENE_PX, SCENE_PX), 0.9)
    scene = Scene(ImageStimulus(flat), fov=(SCENE_PX, SCENE_PX),
                  scotoma=Scotoma.circle(6), scotoma_fill=0.0,
                  scotoma_blend=3)
    profile = scene._native_rgb()[HALF, HALF:, 0, 0]
    npt.assert_almost_equal(profile[-1], 0.9, decimal=6)
    npt.assert_equal(0.25 < profile[6] / 0.9 < 0.75, True)
    npt.assert_equal(np.abs(np.diff(profile)).max() < 0.15, True)


def test_blending_reads_the_loss_field_past_the_frame_edge():
    """A scotoma that covers no pixel can still darken the frame"""
    def edge(blend):
        flat = np.full((SCENE_PX, SCENE_PX), 0.8)
        scene = Scene(ImageStimulus(flat), fov=(SCENE_PX, SCENE_PX),
                      scotoma=Scotoma.circle(4, center=(-HALF - 5, 0)),
                      scotoma_fill=0.0, scotoma_blend=blend)
        return scene._native_rgb()[HALF, 0, 0, 0]
    npt.assert_almost_equal(edge(0), 0.8, decimal=6)
    npt.assert_equal(edge(2) < 0.75, True)


def test_a_softened_boundary_shows_in_the_composed_percept():
    """The blur reaches the prosthetic path, not just native vision"""
    scene = ramp_scene(scotoma=Scotoma.circle(6), scotoma_fill=0.0,
                       scotoma_blend=2)
    bright = Percept(np.ones((SCENE_PX, SCENE_PX, 1)), space=scene._grid())
    seen = rendered(scene, percept=bright, vmax=1)[..., 0]
    npt.assert_equal(seen[HALF, HALF, 0] > 0.98, True)
    npt.assert_almost_equal(seen[HALF, HALF + 15, 0], ramp_at(15), decimal=6)
    npt.assert_equal(ramp_at(6) < seen[HALF, HALF + 6, 0] < 1.0, True)


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


def test_the_blended_boundary_is_angular_not_pixel_sized():
    """The same softness in degrees on two very different rasters"""
    scene = ramp_scene(scotoma=Scotoma.circle(6), scotoma_blend=2)
    xs = np.linspace(-12, 12, 25)
    profiles = []
    for shape in ((SCENE_PX, SCENE_PX), (5 * SCENE_PX, 5 * SCENE_PX)):
        axes = _raster_axes(scene.fov, shape)
        loss = scene._rendered_loss_on(*axes, (0, 0))
        # The horizontal meridian, where the circle's edge is at x = +/-6:
        row = int(np.argmin(np.abs(axes[1])))
        profiles.append(np.interp(xs, axes[0], loss[row]))
    npt.assert_allclose(*profiles, atol=0.02)
    # The premise: the transition is gradual over several degrees, so a
    # pixel-valued sigma would have blurred five times as far on the fine one.
    npt.assert_equal(0.05 < profiles[0][xs.tolist().index(6.0)] < 0.95, True)


def test_anisotropic_pixels_get_their_own_blur_sigma():
    """0.5 x 0.2 degree pixels: 2 degrees is 4 columns but 10 rows

    One sigma for both axes would blur 2.5 times as far across as down, which
    is what makes the two directions comparable here at all.
    """
    scene = Scene(ImageStimulus(np.ones((5 * SCENE_PX, 2 * SCENE_PX))),
                  fov=(SCENE_PX, SCENE_PX), scotoma=Scotoma.circle(6),
                  scotoma_blend=2)
    xs, ys = scene._axes
    npt.assert_almost_equal(_raster_step(xs, ys), (0.5, 0.2))
    loss = scene._rendered_loss_on(xs, ys, (0, 0))
    out = np.linspace(0, 12, 61)
    row, col = int(np.argmin(np.abs(ys))), int(np.argmin(np.abs(xs)))
    # A circular scotoma softened by an angular sigma reads the same going
    # right as going up:
    along_x = np.interp(out, xs[col:], loss[row, col:])
    along_y = np.interp(out, ys[row::-1], loss[row::-1, col])
    npt.assert_allclose(along_x, along_y, atol=0.03)


def test_render_defaults_to_the_source_raster():
    """Asking to render resamples nothing unless a grid is named"""
    scene = ramp_scene(scotoma=Scotoma.circle(6), scotoma_fill=0.3)
    npt.assert_equal(scene.render().shape, (SCENE_PX, SCENE_PX, 3, 1))
    npt.assert_array_equal(scene.render().data, scene._native_rgb())


def test_render_takes_a_shape_or_a_step_but_not_both():
    scene = ramp_scene()
    npt.assert_equal(scene.render(shape=(7, 13)).shape, (7, 13, 3, 1))
    with pytest.raises(ValueError):
        scene.render(step=1, shape=(7, 13))
    for shape in ((0, 4), (4,), (4.5, 4), (4, 4, 4)):
        with pytest.raises(ValueError):
            scene.render(shape=shape)
    for step in (0, -1, np.nan, (1, 0), (1, 2, 3)):
        with pytest.raises(ValueError):
            scene.render(step=step)


def test_render_samples_no_coarser_than_the_step_asked_for():
    """`fov` still bounds the outer pixel edges, so the FOV is preserved"""
    scene = ramp_scene()
    for step in (0.5, 0.3, 4.0, (1.0, 0.25)):
        drawn = scene.render(step=step)
        dx, dy = np.asarray(step, dtype=float) * np.ones(2)
        n_rows, n_cols = drawn.shape[:2]
        npt.assert_equal(SCENE_PX / n_cols <= dx + 1e-12, True)
        npt.assert_equal(SCENE_PX / n_rows <= dy + 1e-12, True)
        # One pixel coarser would have been too coarse:
        npt.assert_equal(SCENE_PX / max(n_cols - 1, 1) > dx, True)
        # The outer edges are the FOV, not the outermost centers:
        npt.assert_almost_equal(
            (drawn.xdva[-1] - drawn.xdva[0]) * n_cols / max(n_cols - 1, 1),
            SCENE_PX, decimal=6)
    # A step that divides the field exactly does not buy a spare pixel:
    npt.assert_equal(scene.render(step=SCENE_PX / 41).shape[:2], (41, 41))
    # ... and a unitful step says the same thing:
    npt.assert_equal(scene.render(step=0.5 * dva).shape,
                     scene.render(step=0.5).shape)


def test_render_leaves_the_source_alone():
    scene = ramp_scene(scotoma=Scotoma.circle(6), scotoma_fill=0.0,
                       aperture='ellipse')
    before = scene.source.data.copy()
    bright = Percept(np.ones((SCENE_PX, SCENE_PX, 1)), space=scene._grid())
    scene.render(percept=bright, vmax=1, step=0.5)
    scene.render(step=0.5)
    npt.assert_array_equal(scene.source.data, before)
    npt.assert_equal(np.any(before > 0), True)


def test_render_composes_by_the_documented_equation():
    """(1 - loss) * native + loss * max(fill, phosphene), analytically"""
    native, fill, half = 0.8, 0.2, 0.5
    scene = Scene(ImageStimulus(np.full((9, 9), native)), fov=(9, 9),
                  scotoma=Scotoma(lambda x, y: np.full(np.shape(x), half)),
                  scotoma_fill=fill, scotoma_blend=0)
    for brightness, vmax in ((1.0, 4.0), (3.0, 4.0)):
        percept = Percept(np.full((9, 9, 1), brightness),
                          space=scene._grid())
        phosphene = brightness / vmax
        npt.assert_almost_equal(
            scene.render(percept=percept, vmax=vmax).data[4, 4, :, 0],
            (1 - half) * native + half * max(fill, phosphene), decimal=6)
    # Renders at a different resolution agree, the composition being pointwise:
    percept = Percept(np.full((9, 9, 1), 3.0), space=scene._grid())
    npt.assert_almost_equal(
        scene.render(percept=percept, vmax=4, shape=(45, 45)).data[22, 22, 0,
                                                                  0],
        scene.render(percept=percept, vmax=4).data[4, 4, 0, 0], decimal=6)


def test_the_aperture_is_support_and_not_scene_data():
    """Black is valid data; outside the ellipse is undefined support"""
    # A black square in the middle of a white field, so a blacked-out pixel
    # and a black source pixel cannot be confused:
    data = np.ones((21, 21))
    data[9:12, 9:12] = 0.0
    scene = Scene(ImageStimulus(data), fov=(21, 21), aperture='ellipse')
    before = scene.source.data.copy()
    drawn = scene.render().data[..., 0]
    npt.assert_array_equal(scene.source.data, before)
    # The black square inside the ellipse is ordinary black scene content:
    npt.assert_almost_equal(drawn[10, 10], 0.0)
    npt.assert_almost_equal(drawn[10, 14], [1.0] * 3, decimal=6)
    # The corners are outside the support and go black as a display result:
    npt.assert_almost_equal(drawn[0, 0], 0.0)
    # Plotting instead keeps the array and clips the artist, and the black
    # square is still black in it:
    ax = scene.plot()
    npt.assert_almost_equal(ax.images[-1].get_array()[0, 0], [1.0] * 3,
                            decimal=6)
    npt.assert_almost_equal(ax.images[-1].get_array()[10, 10], 0.0)
    npt.assert_equal(ax.images[-1].get_clip_path() is not None, True)
    plt.close('all')
    # ... and the device samples the source either way, aperture or not:
    x, y = np.array([-10.0, 0.0, 10.0]), np.array([10.0, 0.0, -10.0])
    plain = Scene(ImageStimulus(data), fov=(21, 21))
    npt.assert_array_equal(scene._device_input(x, y),
                           plain._device_input(x, y))


def test_plotting_keeps_each_layer_at_its_own_resolution():
    """A 60 x 80 source and a 301 x 301 percept stay two artists"""
    rng = np.random.default_rng(0)
    scene = Scene(ImageStimulus(rng.random((60, 80))), fov=45 * dva,
                  scotoma=Scotoma.circle(8), scotoma_fill=0.0,
                  scotoma_blend=0.5)
    # A checkerboard at the percept's own step, dimmer in its top half: both
    # would be gone if the patch were first resampled onto the source raster.
    data = np.zeros((301, 301, 1))
    data[::2, ::2, 0] = 8.0
    data[:150] *= 0.25
    percept = Percept(data, space=Grid2D((-3, 3), (-3, 3), step=6 / 300))
    ax = scene.plot(percept=percept, vmax=8)
    wide, patch = (image.get_array() for image in ax.images)
    npt.assert_equal(wide.shape, (60, 80, 3))
    npt.assert_equal(patch.shape, (301, 301, 3))
    # The patch covers the percept's own extent, not the whole field:
    npt.assert_almost_equal(ax.images[1].get_extent(),
                            (-3.01, 3.01, -3.01, 3.01), decimal=6)
    npt.assert_almost_equal(ax.images[0].get_extent(),
                            (-22.5, 22.5, -16.875, 16.875), decimal=6)
    # Pixel-for-pixel checkerboard, and the dim half is at the top:
    npt.assert_equal(patch[150, ::2, 0].min() > patch[150, 1::2, 0].max(),
                     True)
    npt.assert_equal(patch[0, 0, 0] < patch[-1, 0, 0], True)
    # Nothing was resampled onto a common raster: the whole field at the
    # percept's step would be far bigger than the two artists together.
    common = scene._render_shape(6 / 300, None)
    npt.assert_equal(np.prod(common) > 20 * (60 * 80 + 301 ** 2), True)
    plt.close('all')


def test_a_dark_percept_patch_is_ordinary_residual_vision():
    """No artificial border where the local patch ends"""
    scene = Scene(ImageStimulus(np.full((41, 41), 0.7)), fov=(41, 41),
                  scotoma=Scotoma.circle(14), scotoma_fill=0.25,
                  scotoma_blend=2)
    dark = Percept(np.zeros((21, 21, 1)),
                   space=Grid2D((-10, 10), (-10, 10), step=1))
    ax = scene.plot(percept=dark, vmax=5)
    wide, patch = (image.get_array() for image in ax.images)
    # The patch reduces to the residual view the wide layer shows, so the two
    # agree where they overlap:
    npt.assert_almost_equal(patch[0, :], wide[HALF - 10, HALF - 10:HALF + 11],
                            decimal=6)
    npt.assert_almost_equal(patch[10, 10], wide[HALF, HALF], decimal=6)
    plt.close('all')


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


def test_the_fellow_eye_mirrors_an_asymmetric_scotoma():
    """Eye-centered anatomy is reflected; the world it looks at is not"""
    # Off both meridians, so a reflection is distinguishable from a rotation:
    scotoma = Scotoma.circle(3, center=(6, -3))
    scene = ramp_scene(scotoma=scotoma, aperture='ellipse', background=0.25,
                       scotoma_fill=0.4, scotoma_blend=1.5)
    fellow = scene.fellow_eye()
    npt.assert_equal(fellow is scene, False)
    npt.assert_equal(isinstance(fellow, Scene), True)
    # The loss is 6 degrees into the other hemifield now, and still below the
    # horizontal meridian: x flips, y does not.
    npt.assert_almost_equal(float(fellow.scotoma(-6, -3)), 1.0)
    npt.assert_almost_equal(float(fellow.scotoma(-6, 3)), 0.0)
    npt.assert_almost_equal(float(fellow.scotoma(6, -3)), 0.0)
    # ... and the original is untouched:
    npt.assert_equal(scene.scotoma is scotoma, True)
    npt.assert_almost_equal(float(scene.scotoma(6, -3)), 1.0)
    # Everything that is not eye-specific carries over, source object included:
    npt.assert_equal(fellow.source is scene.source, True)
    npt.assert_equal(fellow.fov, scene.fov)
    npt.assert_equal(fellow.aperture, scene.aperture)
    npt.assert_almost_equal(fellow.background, scene.background)
    npt.assert_equal(fellow.scotoma_fill, scene.scotoma_fill)
    npt.assert_equal(fellow.scotoma_blend, scene.scotoma_blend)


def test_the_fellow_eye_of_an_intact_field_is_intact_too():
    fellow = ramp_scene().fellow_eye()
    npt.assert_equal(fellow.scotoma, None)
    # The picture itself is not flipped: both eyes see the same world:
    npt.assert_array_equal(fellow._native_rgb(), ramp_scene()._native_rgb())


def ellipse_scene(**kwargs):
    """`ramp_scene` seen through an elliptical aperture (here, a disc)"""
    return ramp_scene(aperture='ellipse', **kwargs)


def test_a_rectangular_aperture_is_the_default_and_fills_the_frame():
    scene = ramp_scene()
    npt.assert_equal(scene.aperture, 'rectangle')
    npt.assert_equal('aperture' in repr(scene), False)
    # Every pixel of the rectangle still shows the ramp, corners included:
    native = scene._native_rgb()[..., 0]
    npt.assert_almost_equal(native[0, 0, 0], ramp_at(-HALF), decimal=6)
    npt.assert_almost_equal(native[0, -1, 0], ramp_at(HALF), decimal=6)


def test_a_square_fov_ellipse_keeps_the_center_and_blacks_out_the_corners():
    """Blacking out is `render`'s doing: a display result, not scene data"""
    scene = ellipse_scene()
    npt.assert_equal(scene.aperture, 'ellipse')
    npt.assert_equal("aperture='ellipse'" in repr(scene), True)
    rect = rendered(ramp_scene())[..., 0]
    circ = rendered(scene)[..., 0]
    # Inside the disc nothing changed; the four corners went black:
    npt.assert_array_equal(circ[HALF, :, 0], rect[HALF, :, 0])
    npt.assert_array_equal(circ[:, HALF, 0], rect[:, HALF, 0])
    for row, col in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        npt.assert_almost_equal(circ[row, col], 0.0)
    # The right-hand corners are the bright end of the ramp, so they say the
    # aperture blacked them out rather than the source being dark there:
    npt.assert_almost_equal(rect[0, -1, 0], ramp_at(HALF), decimal=6)
    npt.assert_almost_equal(rect[-1, -1, 0], ramp_at(HALF), decimal=6)
    # ... and the underlying residual view is untouched by the aperture:
    npt.assert_array_equal(scene._native_rgb(), ramp_scene()._native_rgb())


@pytest.mark.parametrize('aperture', ['circle', 'circular', 'ELLIPSE', '',
                                      None, 0, ['ellipse']])
def test_a_bad_aperture_is_refused(aperture):
    with pytest.raises(ValueError):
        ramp_scene(aperture=aperture)


def test_an_ellipse_takes_a_semiaxis_from_each_side_of_the_field():
    """A 61 x 41 field is apertured by both dimensions, not by the shorter"""
    data = np.tile(np.linspace(0.2, 1, 61), (41, 1))
    scene = Scene(ImageStimulus(data), fov=(61, 41), aperture='ellipse')
    lit = rendered(scene)[..., 0, 0] > 0
    x, y = scene._pixel_centers()
    npt.assert_array_equal(lit, (x / 30.5) ** 2 + (y / 20.5) ** 2 <= 1)
    # The aperture reaches much farther sideways than up: 25 dva out along
    # the horizontal is kept, though a disc of radius min(fov) / 2 would have
    # cut it, while the same x lifted 15 dva is outside:
    npt.assert_equal(lit[20, 55], True)          # (25, 0) dva
    npt.assert_equal(lit[5, 55], False)          # (25, 15) dva
    # ... and the semiaxes themselves are in, while the corners are not:
    for row, col in ((20, 0), (20, -1), (0, 30), (-1, 30)):
        npt.assert_equal(lit[row, col], True)
    for row, col in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        npt.assert_equal(lit[row, col], False)


def test_the_aperture_is_eye_centered_and_follows_gaze():
    data = np.tile(np.linspace(0.2, 1, SCENE_PX), (SCENE_PX, 1))
    scene = Scene(ImageStimulus(data), fov=(SCENE_PX, SCENE_PX),
                  aperture='ellipse')
    lit = rendered(scene, gaze=(8, -5) * dva)[..., 0, 0] > 0
    x, y = scene._pixel_centers()
    # scene = eye-centered + gaze, so the disc is centered on the gaze point:
    npt.assert_array_equal(lit, (x - 8) ** 2 + (y + 5) ** 2 <= 20.5 ** 2)


def test_a_transparent_background_does_not_leak_outside_the_aperture():
    """Outside the aperture is black even when the scene's ground is white"""
    scene = Scene(rgba_source(), fov=(8, 8), background=1, aperture='ellipse')
    native = rendered(scene)[..., 0]
    npt.assert_almost_equal(native[0, 0], 0.0)
    # ... while the background still shows through inside it:
    npt.assert_almost_equal(native[0, 4], [1.0, 1.0, 1.0], decimal=6)


def test_the_aperture_does_not_change_what_a_device_is_given():
    """The critical invariant: a rendering boundary, not a sampling geometry"""
    x = np.array([-19.0, -8.0, 0.0, 6.5, 18.0, 30.0])
    y = np.array([17.0, -3.0, 0.0, 11.25, -19.0, 0.0])
    rect, ellip = ramp_scene(), ellipse_scene()
    for gaze in (None, (6, -4) * dva, (-9.5, 12) * dva):
        npt.assert_array_equal(ellip._sample_at(x, y, gaze=gaze),
                               rect._sample_at(x, y, gaze=gaze))
        npt.assert_array_equal(ellip._device_input(x, y, gaze=gaze),
                               rect._device_input(x, y, gaze=gaze))
    npt.assert_array_equal(ellip.dva_to_pixel(x, y), rect.dva_to_pixel(x, y))
    npt.assert_array_equal(ellip.pixel_to_dva(x, y), rect.pixel_to_dva(x, y))
    npt.assert_array_equal(ellip._frames(), rect._frames())


def test_the_aperture_leaves_the_scotoma_alone_inside_it():
    scotoma = Scotoma.circle(6)
    rect = ramp_scene(scotoma=scotoma, scotoma_fill=0.5)
    ellip = ramp_scene(scotoma=scotoma, scotoma_fill=0.5, aperture='ellipse')
    npt.assert_array_equal(ellip.scotoma(0.0, 0.0), rect.scotoma(0.0, 0.0))
    inside = np.hypot(*ellip._pixel_centers()) <= 20.5
    npt.assert_array_equal(ellip._native_rgb()[..., 0][inside],
                           rect._native_rgb()[..., 0][inside])


def test_the_aperture_clips_a_composed_percept_only_when_it_is_drawn():
    scotoma = Scotoma.circle(6)
    bright = Percept(np.ones((SCENE_PX, SCENE_PX, 1)),
                     space=ramp_scene()._grid())
    rect = ramp_scene(scotoma=scotoma, scotoma_fill=0.0)
    ellip = ramp_scene(scotoma=scotoma, scotoma_fill=0.0, aperture='ellipse')
    seen_rect = rendered(rect, percept=bright, vmax=1)[..., 0]
    seen_ellip = rendered(ellip, percept=bright, vmax=1)[..., 0]
    inside = np.hypot(*ellip._pixel_centers()) <= 20.5
    npt.assert_array_equal(seen_ellip[inside], seen_rect[inside])
    npt.assert_almost_equal(seen_ellip[0, -1], 0.0)
    npt.assert_almost_equal(seen_rect[0, -1, 0], ramp_at(HALF), decimal=6)
    # Plotting clips instead: the same elliptical support, applied to the
    # artist rather than written into its array.
    ax = ellip.plot(percept=bright, vmax=1)
    npt.assert_array_equal(ax.images[0].get_array(),
                           rect._native_rgb()[..., 0])
    npt.assert_equal(ax.images[0].get_clip_path() is not None, True)
    plt.close('all')


def test_a_percept_can_be_plotted_in_the_context_of_the_whole_field():
    """A small phosphene, drawn on black at the scale of the ocular field"""
    space = Scene(ImageStimulus(np.zeros((9, 9))), fov=(9, 9))._grid()
    data = np.zeros((9, 9, 1))
    data[4, 4] = 20.0
    phosphene = Percept(data, space=space)
    scene = ellipse_scene()
    ax = scene.plot(percept=phosphene, vmax=20)
    wide, patch = (image.get_array() for image in ax.images)
    # Each layer keeps its own resolution:
    npt.assert_equal(wide.shape, (SCENE_PX, SCENE_PX, 3))
    npt.assert_equal(patch.shape, (9, 9, 3))
    # The percept lands on the fovea at its own size, not stretched to the FOV
    npt.assert_almost_equal(ax.images[1].get_extent(),
                            (-4.5, 4.5, -4.5, 4.5), decimal=6)
    npt.assert_almost_equal(patch[4, 4], [1.0, 1.0, 1.0], decimal=6)
    npt.assert_almost_equal(patch[4, 7], 0.0)
    # ... and native vision is not underneath it, since nothing is lost here:
    npt.assert_almost_equal(wide, 0.0)
    plt.close('all')
    # Brightness is in arbitrary units, so a display range is required:
    with pytest.raises(ValueError):
        scene.plot(percept=phosphene)
    # ... and a display range with nothing to map onto it is not silently
    # ignored, either way round:
    with pytest.raises(ValueError):
        scene.plot(vmax=20)
    with pytest.raises(ValueError):
        scene.plot(vmin=5)


def test_plotting_a_percept_over_a_scotoma_is_the_composed_view():
    """On a shared raster the drawn layers are the dense composition"""
    scene = ramp_scene(scotoma=Scotoma.circle(6), scotoma_fill=0.0)
    bright = Percept(np.ones((SCENE_PX, SCENE_PX, 1)), space=scene._grid())
    ax = scene.plot(percept=bright, vmax=1)
    npt.assert_almost_equal(ax.images[1].get_array(),
                            rendered(scene, percept=bright, vmax=1)[..., 0],
                            decimal=6)
    # The wide layer underneath it is residual native vision:
    npt.assert_almost_equal(ax.images[0].get_array(),
                            scene._native_rgb()[..., 0], decimal=6)
    plt.close('all')


def test_rings_still_land_on_the_fovea_the_aperture_is_centered_on():
    scene = ellipse_scene()
    ax = scene.plot(gaze=(7, -3) * dva, rings=[10])
    # The ring is a circle of radius 10 about the gaze point:
    ring = ax.lines[-1]
    xs, ys = ring.get_xdata(), ring.get_ydata()
    npt.assert_almost_equal([xs.min(), xs.max()], [-3.0, 17.0], decimal=6)
    npt.assert_almost_equal([ys.min(), ys.max()], [-13.0, 7.0], decimal=6)
    # The outermost ring `rings=True` asks for sits inside that same boundary:
    npt.assert_almost_equal(_ring_radii(True, scene.fov).max(), 20.0)
    plt.close('all')


def test_the_aperture_reaches_the_frames_the_player_shows():
    frames = np.stack([np.full((9, 9), v) for v in (0.4, 0.8)], axis=-1)
    scene = Scene(VideoStimulus(frames, time=[0, 10]), fov=(9, 9),
                  aperture='ellipse')
    ani = scene.play()
    npt.assert_almost_equal(ani._frame_data[0, 0, :, 0], 0.0)
    npt.assert_almost_equal(ani._frame_data[4, 4, :, 1], 0.8, decimal=6)
    plt.close('all')


def gray_video_scene(n_frames=3, **kwargs):
    """A ramp scene dimmed by a different factor in every frame"""
    ramp = np.tile(np.linspace(0, 1, SCENE_PX), (SCENE_PX, 1))
    frames = np.stack([ramp * w for w in np.linspace(0.4, 1.0, n_frames)],
                      axis=-1)
    kwargs.setdefault('scotoma_blend', 0)
    return Scene(VideoStimulus(frames, time=np.arange(n_frames) * 10.0),
                 fov=(SCENE_PX, SCENE_PX), **kwargs)


def rgb_video_scene(n_frames=3, **kwargs):
    """Three channels that never agree, so a channel mix-up shows up"""
    rgb = np.stack([np.tile(np.linspace(0, 1, SCENE_PX), (SCENE_PX, 1)),
                    np.tile(np.linspace(1, 0, SCENE_PX), (SCENE_PX, 1)),
                    np.full((SCENE_PX, SCENE_PX), 0.5)], axis=-1)
    frames = np.stack([rgb * w for w in np.linspace(0.4, 1.0, n_frames)],
                      axis=-1)
    kwargs.setdefault('scotoma_blend', 0)
    return Scene(VideoStimulus(frames, time=np.arange(n_frames) * 10.0),
                 fov=(SCENE_PX, SCENE_PX), **kwargs)


def ramped_percept(scene, n_frames):
    """A percept on the scene's own grid, brighter with every frame"""
    frame = np.tile(np.linspace(0, 1, SCENE_PX), (SCENE_PX, 1))
    data = np.stack([frame * w for w in np.linspace(0.3, 1.0, n_frames)],
                    axis=-1)
    return Percept(data, space=scene._grid(),
                   time=np.arange(n_frames) * 10.0)


def frame_scene(scene, f, **kwargs):
    """A still scene of frame ``f``, built exactly like the video one"""
    frames = scene.source.data.reshape(scene.source.vid_shape)
    return Scene(ImageStimulus(frames[..., f]), fov=scene.fov,
                 scotoma=scene.scotoma, scotoma_fill=scene.scotoma_fill,
                 scotoma_blend=scene.scotoma_blend,
                 aperture=scene.aperture, **kwargs)


def test_composing_a_grayscale_video_keeps_the_canonical_rgb_layout():
    """A gray source is composed as RGB without ever differing by channel"""
    n = 3
    scene = gray_video_scene(n, scotoma=Scotoma.circle(6), scotoma_fill=0.0)
    seen = rendered(scene, percept=ramped_percept(scene, n), vmax=1)
    npt.assert_equal(seen.shape, (SCENE_PX, SCENE_PX, 3, n))
    npt.assert_equal(np.asarray(seen).dtype, np.float32)
    npt.assert_array_equal(seen[:, :, 0, :], seen[:, :, 1, :])
    npt.assert_array_equal(seen[:, :, 0, :], seen[:, :, 2, :])
    # Where nothing is lost, native vision passes through untouched:
    source = scene.source.data.reshape(scene.source.vid_shape)
    x, y = scene._pixel_centers()
    intact = scene.scotoma(x, y) == 0
    for f in range(n):
        npt.assert_array_equal(seen[intact, 0, f], source[intact, f])


def test_composing_an_rgb_video_keeps_every_channel_where_it_was():
    n = 3
    scene = rgb_video_scene(n, scotoma=Scotoma.circle(6), scotoma_fill=0.2)
    seen = rendered(scene, percept=ramped_percept(scene, n), vmax=1)
    npt.assert_equal(seen.shape, (SCENE_PX, SCENE_PX, 3, n))
    source = scene.source.data.reshape(scene.source.vid_shape)
    x, y = scene._pixel_centers()
    intact = scene.scotoma(x, y) == 0
    npt.assert_array_equal(seen[intact], source[intact])
    # Inside complete loss the fill and the phosphene are all that is left,
    # whichever is brighter (dim phosphene in frame 0, bright one in frame -1):
    for f in (0, n - 1):
        phosphene = float(ramped_percept(scene, n).data[HALF, HALF, f])
        npt.assert_almost_equal(seen[HALF, HALF, :, f],
                                [max(0.2, phosphene)] * 3, decimal=6)


@pytest.mark.parametrize('blend', [0, 2])
def test_composition_follows_a_gaze_that_moves_between_frames(blend):
    """Frame f of a moving-gaze composition is frame f composed on its own"""
    n = 3
    scene = gray_video_scene(n, scotoma=Scotoma.circle(6), scotoma_fill=0.0,
                             scotoma_blend=blend)
    gaze = np.array([[-7.0, 2.0], [0.0, 0.0], [8.0, -3.0]])
    percept = ramped_percept(scene, n)
    moving = rendered(scene, percept=percept, vmax=1, gaze=gaze)
    for f in range(n):
        still = frame_scene(scene, f)
        alone = rendered(
            still, percept=Percept(percept.data[..., f:f + 1],
                                   space=still._grid()),
            vmax=1, gaze=gaze[f])
        npt.assert_almost_equal(moving[..., f], alone[..., 0], decimal=6)
    # Reusing the pixel raster must not freeze the geometry:
    static = rendered(scene, percept=percept, vmax=1, gaze=gaze[1])
    npt.assert_equal(np.allclose(moving, static), False)


def test_a_static_gaze_composes_every_frame_of_a_video():
    n = 4
    scene = gray_video_scene(n, scotoma=Scotoma.circle(5), scotoma_fill=0.1,
                             scotoma_blend=2)
    percept = ramped_percept(scene, n)
    seen = rendered(scene, percept=percept, vmax=1, gaze=(4.0, -1.0))
    npt.assert_equal(seen.shape, (SCENE_PX, SCENE_PX, 3, n))
    for f in range(n):
        still = frame_scene(scene, f)
        alone = rendered(
            still, percept=Percept(percept.data[..., f:f + 1],
                                   space=still._grid()),
            vmax=1, gaze=(4.0, -1.0))
        npt.assert_almost_equal(seen[..., f], alone[..., 0], decimal=6)
    # The frames are not copies of one another:
    npt.assert_equal(np.allclose(seen[..., 0], seen[..., -1]), False)


def test_an_elliptical_aperture_survives_composing_a_video():
    n = 3
    scene = gray_video_scene(n, scotoma=Scotoma.circle(6), scotoma_fill=0.0,
                             aperture='ellipse')
    percept = ramped_percept(scene, n)
    seen = rendered(scene, percept=percept, vmax=1)
    npt.assert_equal(seen.shape, (SCENE_PX, SCENE_PX, 3, n))
    npt.assert_equal(np.asarray(seen).dtype, np.float32)
    npt.assert_array_equal(seen[0, 0], 0)
    npt.assert_array_equal(seen[-1, -1], 0)
    # Inside the disc the aperture changes nothing:
    rect = gray_video_scene(n, scotoma=Scotoma.circle(6), scotoma_fill=0.0)
    inside = ~scene._aperture_mask(*scene._axes, (0, 0))
    npt.assert_array_equal(seen[inside],
                           rendered(rect, percept=percept, vmax=1)[inside])


@pytest.mark.parametrize('blend', [0, 2])
def test_the_loss_composition_renders_is_float32(blend):
    """A float64 loss map would upcast the whole float32 blend"""
    scene = ramp_scene(scotoma=Scotoma.circle(6), scotoma_fill=0.0,
                       scotoma_blend=blend)
    npt.assert_equal(rendered_loss(scene).dtype, np.float32)
    npt.assert_equal(rendered_loss(ramp_scene()).dtype, np.float32)


def test_repeated_pixel_center_queries_read_the_same_raster():
    scene = ramp_scene(scotoma=Scotoma.circle(6), scotoma_blend=2)
    x, y = scene._pixel_centers()
    npt.assert_equal(x.shape, (SCENE_PX,) * 2)
    again = scene._pixel_centers()
    npt.assert_array_equal(again[0], x)
    npt.assert_array_equal(again[1], y)
    # The pixel centers are the source raster's own axes:
    npt.assert_array_equal(x[0], scene._axes[0])
    npt.assert_array_equal(y[:, 0], scene._axes[1])
