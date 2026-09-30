"""Tests for Scene (#668)

Test scenes use 1 dva per pixel and an odd pixel count, so pixel centers land
on whole degrees and the center pixel sits on the origin.
"""
import numpy as np
import numpy.testing as npt
import pytest
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

from pulse2percept.percepts import Percept
from pulse2percept.stimuli import ImageStimulus, VideoStimulus, samples
from pulse2percept.topography import Grid2D
from pulse2percept.units import dva, ms, s
from pulse2percept.vision import Scene, Scotoma
from pulse2percept.vision import scene as scene_module
from pulse2percept.vision.scene import (_raster_axes, _raster_extent,
                                        _raster_step)

SCENE_PX = 41
HALF = (SCENE_PX - 1) // 2


def ramp_scene(**kwargs):
    """Scene whose gray level is a ramp in x: 0 at -20 dva, 1 at +20 dva"""
    data = np.tile(np.linspace(0, 1, SCENE_PX), (SCENE_PX, 1))
    kwargs.setdefault('scotoma_blend', 0)
    return Scene(ImageStimulus(data), fov=(SCENE_PX, SCENE_PX), **kwargs)


def rgba_source():
    """Opaque red square on a transparent red surround

    The background must replace the surround's red if alpha is applied.
    """
    img = np.zeros((8, 8, 4))
    img[..., 0] = 1.0
    img[2:6, 2:6, 3] = 1.0
    return ImageStimulus(img)


def ramp_at(x_dva):
    """Returns the `ramp_scene` value at scene x (dva)"""
    return (x_dva + HALF) / (2 * HALF)


def rendered_loss(scene):
    """Returns the drawn loss map on the default FOV raster"""
    return scene._rendered_loss_on(*scene._view_axes())


def rendered(scene, **kwargs):
    """`Scene.render` on the source raster, as a bare RGB array"""
    return scene.render(**kwargs).data


def seen_at(scene, x_dva, y_dva=0.0):
    """Returns the device input at the fovea with gaze at ``(x_dva, y_dva)``

    Uses the same sampling path as an electrode.
    """
    return float(np.ravel(scene._device_input(0.0, 0.0,
                                              gaze=(x_dva, y_dva)))[0])


def test_fov_may_be_a_scalar_a_pair_or_unitful():
    """A scalar fov gives a square window"""
    source = ImageStimulus(np.zeros((10, 20)))
    npt.assert_almost_equal(Scene(source, fov=40).fov, (40.0, 40.0))
    npt.assert_almost_equal(Scene(source, fov=40 * dva).fov, (40.0, 40.0))
    npt.assert_almost_equal(Scene(source, fov=(40, 15)).fov, (40.0, 15.0))
    npt.assert_almost_equal(Scene(source, fov=(40, 15) * dva).fov,
                            (40.0, 15.0))


@pytest.mark.parametrize('fov', [0, -5, np.nan, np.inf, (10, 0), (10, np.nan),
                                 (1, 2, 3)])
def test_a_scene_needs_a_real_field_of_view(fov):
    """Zero, negative, non-finite, or wrong-length fov is a ValueError"""
    with pytest.raises(ValueError):
        Scene(ImageStimulus(np.zeros((8, 8))), fov=fov)


def test_a_scene_requires_a_fov_at_all():
    """Scene requires a field of view"""
    with pytest.raises(TypeError):
        Scene(ImageStimulus(np.zeros((8, 8))))


def test_an_ordinary_image_is_wrapped():
    """A bare array is wrapped in an ImageStimulus"""
    scene = Scene(np.zeros((10, 20)), fov=40)
    npt.assert_equal(isinstance(scene.source, ImageStimulus), True)
    npt.assert_equal(scene.shape, (10, 20))
    # An ImageStimulus subclass is used as is:
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
    # dva_to_pixel is the exact inverse, including fractional pixels:
    for col, row in [(0, 0), (3.5, 17.25), (SCENE_PX - 1, SCENE_PX - 1)]:
        back = scene.dva_to_pixel(*scene.pixel_to_dva(col, row))
        npt.assert_almost_equal(back, [col, row], decimal=9)


def test_row_zero_is_the_top_of_the_visual_field():
    """Row 0 is the top of the field (+y)"""
    data = np.tile(np.linspace(0, 1, SCENE_PX).reshape((-1, 1)),
                   (1, SCENE_PX))
    scene = Scene(ImageStimulus(data), fov=(SCENE_PX, SCENE_PX))
    for y_dva in (5.0, -5.0):
        # Row r sits at y = HALF - r, so small values are at +y:
        npt.assert_almost_equal(seen_at(scene, 0.0, y_dva),
                                (HALF - y_dva) / (2 * HALF), decimal=5)


def test_sampling_depends_only_on_where_you_look():
    """Sampled value depends only on eye-centered position plus gaze"""
    scene = ramp_scene()
    for x_vf, gaze_x in [(4.0, 0.0), (0.0, 4.0), (10.0, -6.0), (-3.0, 7.0)]:
        npt.assert_almost_equal(
            float(np.ravel(scene._device_input(x_vf, 0.0,
                                               gaze=(gaze_x, 0)))[0]),
            ramp_at(x_vf + gaze_x), decimal=5)


def test_outside_the_scene_there_is_nothing():
    """Points outside the scene return 0"""
    scene = ramp_scene()
    npt.assert_almost_equal(seen_at(scene, 40.0), 0.0)
    npt.assert_almost_equal(seen_at(scene, 0.0, -40.0), 0.0)


#: 4x4 scene at 1 dva/pixel: columns at x = -1.5, -0.5, 0.5, 1.5, rows at
#: y = 1.5, 0.5, -0.5, -1.5; outer extent is +/-2 dva.
EDGE_PX = 4
EDGE_EXTENT = EDGE_PX / 2


def edge_scene(along='x'):
    """4x4 scene ramping from 0.25 to 1.0 along one axis

    Nonzero everywhere, so off-scene samples (0) are distinguishable.
    """
    ramp = np.linspace(0.25, 1.0, EDGE_PX)
    data = (np.tile(ramp, (EDGE_PX, 1)) if along == 'x'
            else np.tile(ramp.reshape((-1, 1)), (1, EDGE_PX)))
    return Scene(ImageStimulus(data), fov=(EDGE_PX, EDGE_PX))


@pytest.mark.parametrize('sign', [1, -1])
def test_the_whole_stated_fov_belongs_to_the_scene(sign):
    """The outer half-pixel border belongs to the scene"""
    scene = edge_scene('x')
    edge_value = 1.0 if sign > 0 else 0.25
    last_center = sign * (EDGE_EXTENT - 0.5)
    npt.assert_almost_equal(seen_at(scene, last_center), edge_value, decimal=5)
    # Between last pixel center and outer edge, the edge pixel value is used
    # (no extrapolation):
    npt.assert_almost_equal(seen_at(scene, sign * (EDGE_EXTENT - 0.25)),
                            edge_value, decimal=5)
    # The outer edge itself is inside:
    npt.assert_almost_equal(seen_at(scene, sign * EDGE_EXTENT), edge_value,
                            decimal=5)
    # Just past it is outside:
    npt.assert_almost_equal(seen_at(scene, sign * (EDGE_EXTENT + 0.01)), 0.0)


@pytest.mark.parametrize('sign', [1, -1])
def test_the_vertical_fov_reaches_its_edges_too(sign):
    """Same border rule on y, where row 0 is +y"""
    scene = edge_scene('y')
    # Row 0 holds 0.25 and sits at +y:
    edge_value = 0.25 if sign > 0 else 1.0
    for y in (sign * (EDGE_EXTENT - 0.5), sign * (EDGE_EXTENT - 0.25),
              sign * EDGE_EXTENT):
        npt.assert_almost_equal(seen_at(scene, 0.0, y), edge_value, decimal=5)
    npt.assert_almost_equal(seen_at(scene, 0.0, sign * (EDGE_EXTENT + 0.01)),
                            0.0)


def test_a_corner_outside_the_fov_is_outside_even_on_one_axis():
    """A point outside the fov on either axis is outside"""
    scene = edge_scene('x')
    npt.assert_almost_equal(seen_at(scene, 0.0, 1.9), 0.625, decimal=5)
    npt.assert_almost_equal(seen_at(scene, 0.0, 2.1), 0.0)
    npt.assert_almost_equal(seen_at(scene, 2.1, 1.9), 0.0)


def test_interior_sampling_still_interpolates():
    """Border clamping leaves interior interpolation intact"""
    scene = edge_scene('x')
    # Halfway between the two middle pixel centers (0.5 and 0.75):
    npt.assert_almost_equal(seen_at(scene, 0.0), 0.625, decimal=5)
    # A quarter of the way from the second pixel to the third:
    npt.assert_almost_equal(seen_at(scene, -0.25), 0.5625, decimal=5)


def test_color_survives_sampling_and_greys_only_at_the_device():
    """Sampling keeps RGB; device input is converted to gray"""
    rgb = np.zeros((21, 21, 3))
    rgb[..., 0] = 1.0  # pure red everywhere
    scene = Scene(ImageStimulus(rgb), fov=(20, 20))
    values = scene._sample_at(0.0, 0.0)
    npt.assert_equal(values.shape, (1, 3, 1))
    npt.assert_almost_equal(values[0, :, 0], [1, 0, 0], decimal=5)
    # Device input is the luminance of pure red:
    npt.assert_almost_equal(scene._device_input(0.0, 0.0), [[0.2125]],
                            decimal=4)


def test_sampling_rgb_then_greying_matches_greying_first():
    """rgb2gray after interpolation matches rgb2gray before it"""
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
    """Video scene keeps its frames and frame times"""
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
    # Center pixel is 0.5 times the per-frame scale factor:
    npt.assert_almost_equal(values.ravel(),
                            0.5 * (np.arange(n_frames) + 1) / 4, decimal=5)


def test_a_scene_reports_its_sources_clock_verbatim():
    """Scene keeps the source's frame times and time unit"""
    source = VideoStimulus(np.zeros((4, 4, 2)), time=[0, 0.05] * s)
    scene = Scene(source, fov=(4, 4))
    npt.assert_equal(scene.time_unit, source.time_unit)
    npt.assert_almost_equal(scene.time, source.time)
    npt.assert_almost_equal(scene.time, [0, 50])


def test_gaze_may_move_between_frames():
    """A (n_frames, 2) gaze array applies one gaze per frame"""
    frames = np.repeat(np.tile(np.linspace(0, 1, SCENE_PX),
                               (SCENE_PX, 1))[..., np.newaxis], 3, axis=-1)
    scene = Scene(VideoStimulus(frames, time=[0, 10, 20]),
                  fov=(SCENE_PX, SCENE_PX))
    gaze = np.array([[-6.0, 0.0], [0.0, 0.0], [6.0, 0.0]])
    moving = scene._device_input(0.0, 0.0, gaze=gaze)
    npt.assert_almost_equal(moving.ravel(), ramp_at(gaze[:, 0]), decimal=5)
    # Static gaze gives a different result, so per-frame gaze was used:
    static = scene._device_input(0.0, 0.0, gaze=(0, 0))
    npt.assert_equal(np.allclose(moving, static), False)
    with pytest.raises(ValueError):
        scene._device_input(0.0, 0.0, gaze=np.zeros((2, 2)))


@pytest.mark.parametrize('gaze', [(np.nan, 0), (0, np.inf), (-np.inf, 0)])
def test_non_finite_gaze_is_refused(gaze):
    """Non-finite gaze is a ValueError"""
    with pytest.raises(ValueError):
        ramp_scene()._device_input(0.0, 0.0, gaze=gaze)


def test_unitful_gaze_reads_the_same_place():
    scene = ramp_scene()
    npt.assert_almost_equal(seen_at(scene, 3.0),
                            float(np.ravel(scene._device_input(
                                0.0, 0.0, gaze=(3, 0) * dva))[0]))


def test_without_a_scotoma_native_vision_is_the_scene_exactly():
    # 576 x 720 logo: a 40-degree window is its central 576 columns
    scene = Scene(samples.logo_bvl(), fov=40 * dva)
    native = scene._native_rgb()
    npt.assert_equal(native.shape, (576, 576, 3, 1))
    # The logo is RGBA; alpha is blended against black:
    source = scene.source.data.reshape(scene.source.img_shape)
    expected = source[..., :3] * source[..., 3:4]
    npt.assert_almost_equal(native[..., 0], expected[:, 72:648], decimal=6)


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
    # Outside the scotoma the scene is unchanged:
    source = np.repeat(scene.source.data.reshape(
        (SCENE_PX, SCENE_PX, 1)), 3, axis=-1)
    x, y = scene._pixel_centers()
    intact = scene.scotoma(x, y) == 0
    npt.assert_array_equal(native[intact], source[intact])


@pytest.mark.parametrize('fill, rgb', [
    ('gray', (128 / 255,) * 3),
    ('red', (1.0, 0.0, 0.0)),
    ('#336699', (0x33 / 255, 0x66 / 255, 0x99 / 255)),
    ('0.5', (0.5, 0.5, 0.5)),
])
def test_a_matplotlib_color_fills_the_scotoma_with_that_color(fill, rgb):
    scene = ramp_scene(scotoma=Scotoma.circle(3), scotoma_fill=fill)
    npt.assert_almost_equal(scene.scotoma_fill, rgb, decimal=6)
    # Each channel gets its own value:
    npt.assert_almost_equal(scene._native_rgb()[HALF, HALF, :, 0], rgb,
                            decimal=6)


def test_a_color_fill_survives_a_round_trip_through_the_constructor():
    """Color fill is kept when a Scene is rebuilt (e.g., `fellow_eye`)"""
    scene = ramp_scene(scotoma=Scotoma.circle(3), scotoma_fill='red')
    npt.assert_almost_equal(scene.fellow_eye().scotoma_fill, (1.0, 0.0, 0.0))
    npt.assert_almost_equal(Scene(scene.source, fov=scene.fov,
                                  scotoma_fill=(0.2, 0.4, 0.6)).scotoma_fill,
                            (0.2, 0.4, 0.6))


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
    # The scotoma stays at the FOV center (fovea):
    npt.assert_almost_equal(fixating[HALF, HALF], [0.0] * 3, decimal=6)
    npt.assert_almost_equal(shifted[HALF, HALF], [0.0] * 3, decimal=6)
    # The scene moved 5 dva left:
    source = scene.source.data.reshape((SCENE_PX, SCENE_PX))
    npt.assert_almost_equal(shifted[HALF, HALF - 5], [source[HALF, HALF]] * 3,
                            decimal=6)
    npt.assert_almost_equal(shifted[HALF, HALF + 5], [ramp_at(10)] * 3,
                            decimal=6)


def test_scotoma_does_not_change_what_the_device_sees():
    """Scotoma affects native vision only, not device input"""
    x = np.array([-15.0, -8.0, -2.0, 0.0, 3.0, 9.0, 18.0])
    y = np.array([0.0, 6.0, -4.0, 0.0, 2.0, -9.0, 5.0])
    plain = ramp_scene()
    for fill in (0.0, 0.75, 1.0):
        blind = ramp_scene(scotoma=Scotoma.circle(10), scotoma_fill=fill)
        npt.assert_array_equal(blind._sample_at(x, y), plain._sample_at(x, y))
        npt.assert_array_equal(blind._device_input(x, y),
                               plain._device_input(x, y))
    # Some points must be lost, some intact, and all source values distinct:
    loss = blind.scotoma(x, y)
    npt.assert_equal(loss.max() == 1 and loss.min() == 0, True)
    seen = np.ravel(plain._device_input(x, y))
    npt.assert_equal(np.unique(seen).size, x.size)
    # Native vision does change:
    npt.assert_equal(np.allclose(blind._native_rgb(), plain._native_rgb()),
                     False)


def test_inpainting_a_constant_surround_gives_back_the_constant():
    """Inpainting a flat field returns the flat value"""
    flat = np.full((31, 31), 0.6)
    scene = Scene(ImageStimulus(flat), fov=(31, 31),
                  scotoma=Scotoma.circle(5), scotoma_fill='inpaint')
    npt.assert_almost_equal(scene._native_rgb()[..., 0], 0.6, decimal=6)


def test_the_inpainted_fill_knows_nothing_of_what_it_covers():
    """Inpainted fill depends only on the visible surround"""
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
    """Composing a percept onto an inpainted scene is a ValueError"""
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
    # A numeric fill composes; native vision is finite either way:
    numeric = Scene(ImageStimulus(rgb), fov=(31, 31),
                    scotoma=Scotoma.circle(4), scotoma_fill=0.0)
    npt.assert_equal(rendered(numeric, percept=dark, vmax=1).shape,
                     (31, 31, 3, 1))
    npt.assert_equal(np.all(np.isfinite(scene._native_rgb())), True)


def test_inpainting_ignores_the_blend():
    """Inpainting ignores scotoma_blend, so covered pixels stay out"""
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
    """Plots onto new axes"""
    return scene.plot(ax=plt.subplots()[1], **kwargs)


def ring_radii(ax, center=(0, 0)):
    """Returns the sorted radii of the drawn (dashed) rings"""
    offsets = [np.asarray(line.get_data()) - np.reshape(center, (2, 1))
               for line in ax.get_lines() if line.get_linestyle() == '--']
    return sorted(np.hypot(*offset).mean() for offset in offsets)


def meridian_ends(ax):
    """Returns start and end points of the drawn (solid) meridians"""
    return [np.asarray(line.get_data())[:, [0, -1]].T
            for line in ax.get_lines() if line.get_linestyle() == '-']


def video_scene(**kwargs):
    """Two frames of the ramp, one second apart"""
    ramp = np.tile(np.linspace(0, 1, SCENE_PX), (SCENE_PX, 1))
    frames = np.stack([ramp, ramp[::-1]], axis=-1)
    return Scene(VideoStimulus(frames, time=[0, 1000]),
                 fov=(SCENE_PX, SCENE_PX), **kwargs)


def test_rings_are_drawn_only_when_asked():
    scene = ramp_scene()
    for off in (False, None):
        ax = drawn_on_fresh_axes(scene, rings=off, meridians=off)
        npt.assert_equal(len(ax.lines) + len(ax.texts), 0)
    # A 41-dva field gets the doubling sequence up to 20 dva:
    ax = drawn_on_fresh_axes(scene, rings=True)
    npt.assert_almost_equal(ring_radii(ax), [1.25, 2.5, 5, 10, 20], decimal=6)
    npt.assert_equal([t.get_text() for t in ax.texts],
                     [f'{r}\N{DEGREE SIGN}'
                      for r in ('1.25', '2.5', '5', '10', '20')])
    line = ax.get_lines()[0]
    npt.assert_equal(line.get_linestyle(), '--')
    npt.assert_equal(line.get_linewidth() < 1, True)
    plt.close('all')


def test_rings_takes_a_spacing_or_the_eccentricities_themselves():
    scene = ramp_scene()
    npt.assert_almost_equal(ring_radii(drawn_on_fresh_axes(scene, rings=10)),
                            [10, 20], decimal=6)
    ax = drawn_on_fresh_axes(scene, rings=[15, 5, 25])
    # Explicit eccentricities are all drawn, even outside the field:
    npt.assert_almost_equal(ring_radii(ax), [5, 15, 25], decimal=6)
    plt.close('all')


def test_grid_takes_a_color():
    scene = ramp_scene()
    ax = drawn_on_fresh_axes(scene, rings=True, meridians=True,
                             grid_color='white')
    for artist in ax.get_lines() + list(ax.texts):
        npt.assert_equal(artist.get_color(), 'white')
    plt.close('all')
    # Also applies to the rasterized overlay in `play`:
    video = video_scene()
    black = video.play(rings=[10], grid_color='black')._frame_data
    white = video.play(rings=[10], grid_color='white')._frame_data
    npt.assert_equal(np.allclose(black, white), False)
    plt.close('all')


def test_grid_stays_on_the_fovea_at_the_center_of_the_fov():
    scene = ramp_scene()
    ax = drawn_on_fresh_axes(scene, gaze=(3, -2) * dva, rings=True,
                             meridians=True, view='eye')
    # Gaze moves the scene; the grid stays in the eye-centered FOV:
    npt.assert_almost_equal(ring_radii(ax), [1.25, 2.5, 5, 10, 20],
                            decimal=6)
    for start, _ in meridian_ends(ax):
        npt.assert_almost_equal(start, (0, 0))
    npt.assert_array_equal(ax.images[-1].get_array(),
                           scene._native_rgb(gaze=(3, -2))[..., 0])
    plt.close('all')


def test_meridians_follow_the_cartesian_polar_angle_convention():
    """0 deg is +x, 90 deg is +y, counterclockwise; lines reach the edge"""
    ax = drawn_on_fresh_axes(ramp_scene(), gaze=(3, -2) * dva,
                             meridians=[0, 90, 180, 270], view='eye')
    ends = [end for _, end in meridian_ends(ax)]
    npt.assert_almost_equal(ends, [(20.5, 0), (0, 20.5), (-20.5, 0),
                                   (0, -20.5)])
    plt.close('all')


def test_meridians_take_a_spacing_or_the_angles_themselves():
    scene = ramp_scene()

    def angles(**kwargs):
        ax = drawn_on_fresh_axes(scene, **kwargs)
        return sorted(np.rad2deg(np.arctan2(*(end - start)[::-1])) % 360
                      for start, end in meridian_ends(ax))

    npt.assert_almost_equal(angles(meridians=True), np.arange(0, 360, 45))
    npt.assert_almost_equal(angles(meridians=30), np.arange(0, 360, 30))
    npt.assert_almost_equal(angles(meridians=[90, 0, 45]), [0, 45, 90])
    # No rings, so no ring labels:
    npt.assert_equal(len(drawn_on_fresh_axes(scene, meridians=True).texts), 0)
    plt.close('all')


def test_rings_fit_the_shorter_half_of_the_field():
    tall = Scene(ImageStimulus(np.zeros((40, 20))), fov=(20, 40))
    npt.assert_almost_equal(ring_radii(drawn_on_fresh_axes(tall, rings=4)),
                            [4, 8], decimal=6)
    small = Scene(ImageStimulus(np.zeros((8, 8))), fov=8)
    npt.assert_almost_equal(ring_radii(drawn_on_fresh_axes(small, rings=True)),
                            [1.25, 2.5], decimal=6)
    # No ring fits if the half-field is smaller than one step:
    npt.assert_equal(len(drawn_on_fresh_axes(small, rings=5).lines), 0)
    plt.close('all')


@pytest.mark.parametrize('rings', [0, -5, np.nan, np.inf, [], [5, 0],
                                   [5, np.inf], [np.nan]])
def test_bad_rings_are_refused(rings):
    with pytest.raises(ValueError):
        drawn_on_fresh_axes(ramp_scene(), rings=rings)
    plt.close('all')


@pytest.mark.parametrize('meridians', [0, -45, np.nan, np.inf, [],
                                       [0, np.nan]])
def test_bad_meridians_are_refused(meridians):
    with pytest.raises(ValueError):
        drawn_on_fresh_axes(ramp_scene(), meridians=meridians)
    plt.close('all')


def test_play_paints_a_readable_grid_into_the_frames_the_player_shows():
    """play() draws the grid into the frame data

    White scene at 3 px/dva, so a 10-dva ring is 30 px from the center.
    """
    white = np.full((120, 120), 1.0)
    scene = Scene(VideoStimulus(np.stack([white, white], axis=-1),
                                time=[0, 1000]), fov=(40, 40))
    plain = scene.play()._frame_data
    ringed = scene.play(rings=[10])._frame_data
    npt.assert_array_equal(plain, scene._native_rgb())
    contrast = (plain - ringed).max(axis=(2, 3))
    # The ring must have visible contrast:
    npt.assert_equal(contrast.max() > 0.3, True)
    npt.assert_equal(np.count_nonzero(contrast > 0.2) > 50, True)
    rows, cols = np.nonzero(contrast > 0.2)
    radius = np.hypot(cols - 60, rows - 60)
    npt.assert_equal(20 < radius.min() < 32, True)
    # The label sits outside the ring; corners are unchanged:
    npt.assert_equal((contrast[:30] > 0.2).any(), True)
    npt.assert_almost_equal(contrast[0, 0], 0.0, decimal=6)
    # Meridians too; 0 deg runs along the row through the fovea:
    ruled = scene.play(meridians=[0])._frame_data
    contrast = (plain - ruled).max(axis=(2, 3))
    npt.assert_equal((contrast[58:62, 62:] > 0.1).any(axis=0).all(), True)
    npt.assert_almost_equal(contrast[:50].max(), 0.0, decimal=6)
    plt.close('all')


def test_play_keeps_the_grid_still_on_a_gaze_that_moves():
    """With view='eye', the grid does not move with gaze"""
    white = np.full((SCENE_PX, SCENE_PX), 1.0)
    # 21-dva FOV on a 41-dva white scene, so both frames are white:
    scene = Scene(VideoStimulus(np.stack([white, white], axis=-1),
                                time=[0, 1000]), fov=(21, 21),
                  extent=(-20.5, 20.5, -20.5, 20.5))
    gaze = [(0, 0), (5, 5)]
    plain = scene.play(gaze=gaze, view='eye')._frame_data
    ringed = scene.play(gaze=gaze, rings=[10], view='eye')._frame_data
    painted = (plain - ringed).max(axis=2) > 0.2
    npt.assert_equal(painted.any(), True)
    # The white source covers the ring in both frames:
    npt.assert_array_equal(painted[..., 0], painted[..., 1])
    plt.close('all')


def aligned_percept():
    """Percept aligned to `video_scene`'s frames, time stamps at frame ends"""
    grid = Grid2D((-8, 8), (-8, 8), step=1)
    ramp = (grid.x + 8) / 16
    return Percept(np.stack([ramp, 3 * ramp[::-1]], axis=-1), space=grid,
                   time=[1000, 2000],
                   metadata={'source_frame_time': [0, 1000]})


@pytest.fixture
def played(monkeypatch):
    """Records the Percept passed to `Percept.play` and its kwargs"""
    seen = {}
    original = Percept.play

    def play(self, *args, **kwargs):
        seen.update(percept=self, kwargs=kwargs)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Percept, 'play', play)
    return seen


@pytest.mark.parametrize('rings', [False, [3]])
def test_play_shows_what_render_composes(played, rings):
    scene = video_scene(scotoma=Scotoma.circle(6), scotoma_fill=0.2,
                        scotoma_blend=0)
    percept = aligned_percept()
    frames = []
    for vmin, vmax in ((0, 3), (1, 2)):
        rendered = scene.render(percept=percept, vmax=vmax, vmin=vmin)
        ani = scene.play(percept=percept, vmax=vmax, vmin=vmin, rings=rings)
        if not rings:
            npt.assert_allclose(ani._frame_data, rendered.data, atol=1e-6)
        frames.append(ani._frame_data)
        # vmin/vmax are used in rendering and not passed to the player:
        npt.assert_equal('vmax' in played['kwargs'], False)
        npt.assert_equal('vmin' in played['kwargs'], False)
        # Frame-end time stamps are kept:
        npt.assert_almost_equal(played['percept'].time, [1000, 2000])
        npt.assert_equal(played['percept'].time_unit, ms)
    npt.assert_equal(np.abs(frames[0] - frames[1]).max() > 0.1, True)
    # The percept is composed into the frames:
    npt.assert_equal(np.abs(frames[0] - scene.play()._frame_data).max() > 0.1,
                     True)
    plt.close('all')


def test_play_paints_the_grid_over_the_composed_frames():
    scene = video_scene(scotoma=Scotoma.circle(6), scotoma_fill=0.2,
                        scotoma_blend=0)
    percept = aligned_percept()
    composed = scene.render(percept=percept, vmax=3).data
    ringed = scene.play(percept=percept, vmax=3, rings=[3])._frame_data
    changed = np.abs(ringed - composed).max(axis=(2, 3)) > 0.1
    rows, cols = np.nonzero(changed)
    # A 3-dva ring lies inside the 6-dva scotoma, over the phosphenes:
    npt.assert_equal(rows.size > 0, True)
    npt.assert_equal(np.hypot(rows - HALF, cols - HALF).min() < 6, True)
    # Below the ring and its label, frames match the composed ones:
    far = np.zeros((SCENE_PX, SCENE_PX), dtype=bool)
    far[HALF + 6:] = True
    npt.assert_allclose(ringed[far], composed[far], atol=1e-6)
    plt.close('all')


def test_play_forwards_gaze_to_render():
    scene = video_scene(scotoma=Scotoma.circle(6), scotoma_fill=0.2)
    percept = aligned_percept()
    for gaze in ((3, -2), [(0, 0), (4, 1)]):
        npt.assert_allclose(
            scene.play(percept=percept, vmax=3, gaze=gaze,
                       view='eye')._frame_data,
            scene.render(percept=percept, vmax=3, gaze=gaze).data, atol=1e-6)
    plt.close('all')


def test_play_keeps_its_positional_arguments():
    scene = video_scene()
    npt.assert_array_equal(scene.play((0, 0), [10])._frame_data,
                           scene.play(gaze=(0, 0), rings=[10])._frame_data)
    plt.close('all')


@pytest.mark.parametrize('rings', [False, [3]])
def test_play_passes_its_player_options_explicitly(played, rings):
    scene = video_scene()
    options = dict(fps=2, repeat=False, annotate_time=False, fmt='jpg',
                   title='PRIMA')
    ani = scene.play(rings=rings, **options)
    npt.assert_equal(played['kwargs'], {**options, 'ax': None})
    npt.assert_equal(ani._fig._suptitle.get_text(), 'PRIMA')
    npt.assert_equal(ani._labels, None)
    # Unknown options are a TypeError:
    for name in ('cmap', 'colorbar'):
        with pytest.raises(TypeError):
            scene.play(**{name: False})
    plt.close('all')


def test_an_omitted_vmax_is_the_whole_percepts_maximum():
    """Default vmax is the maximum over all frames"""
    scene = video_scene(scotoma=Scotoma.circle(6), scotoma_fill=0.2,
                        scotoma_blend=0)
    percept = aligned_percept()
    # Frame 0 peaks at 1, frame 1 at 3:
    npt.assert_almost_equal(percept.data[..., 0].max(), 1)
    npt.assert_almost_equal(percept.data.max(), 3)
    npt.assert_array_equal(scene.play(percept=percept)._frame_data,
                           scene.play(percept=percept, vmax=3)._frame_data)
    for frame in (0, 1):
        auto = scene.plot(percept=percept, frame=frame).images[1]
        fixed = scene.plot(percept=percept, frame=frame, vmax=3,
                           vmin=0).images[1]
        npt.assert_array_equal(auto.get_array(), fixed.get_array())
        plt.close('all')
    # Explicit limits are used when given:
    clipped = scene.play(percept=percept, vmax=1)._frame_data
    auto = scene.play(percept=percept)._frame_data
    npt.assert_equal(np.abs(clipped - auto).max() > 0.1, True)
    # A blank percept is drawn as with any vmax > 0:
    blank = Percept(np.zeros_like(percept.data), space=Grid2D((-8, 8), (-8, 8),
                                                              step=1),
                    time=percept.time, metadata=percept.metadata)
    npt.assert_array_equal(scene.play(percept=blank)._frame_data,
                           scene.play(percept=blank, vmax=1)._frame_data)
    auto = scene.plot(percept=blank).images[1].get_array()
    fixed = scene.plot(percept=blank, vmax=1).images[1].get_array()
    npt.assert_array_equal(auto, fixed)
    # An explicit empty range is a ValueError:
    with pytest.raises(ValueError):
        scene.plot(percept=blank, vmin=0, vmax=0)
    plt.close('all')


def test_play_inherits_the_composition_rules_of_render():
    scene = video_scene(scotoma=Scotoma.circle(6), scotoma_fill='inpaint')
    with pytest.raises(ValueError) as excinfo:
        scene.play(percept=aligned_percept(), vmax=3)
    npt.assert_equal('inpaint' in str(excinfo.value), True)
    with pytest.raises(ValueError):
        # vmax without a percept:
        video_scene().play(vmax=3)
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
    for fill in ('blur', 'INPAINT', 'inpainting', 'definitely-not-a-color'):
        with pytest.raises(ValueError):
            Scene(source, fov=8, scotoma_fill=fill)
    for fill in ((0.1, 0.2), (0.1, 0.2, 0.3, 0.4), (0.5, 1.5, 0.5)):
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
    """At 1 dva/px, a sigma of 2 dva is 2 px"""
    scene = ramp_scene(scotoma=Scotoma.circle(6), scotoma_blend=2)
    loss = rendered_loss(scene)
    # 3 sigmas inside the edge is fully lost, 4 sigmas outside is intact:
    npt.assert_equal(loss[HALF, HALF] > 0.98, True)
    npt.assert_almost_equal(loss[HALF, HALF + 15], 0.0, decimal=12)
    npt.assert_equal(0.05 < loss[HALF, HALF + 6] < 0.95, True)
    profile = loss[HALF, HALF:HALF + 16]
    npt.assert_array_less(np.diff(profile), 1e-12)


def test_a_numeric_fill_has_no_hard_contour_at_the_boundary():
    """A blended scotoma over a flat scene gives a gradual ramp"""
    flat = np.full((SCENE_PX, SCENE_PX), 0.9)
    scene = Scene(ImageStimulus(flat), fov=(SCENE_PX, SCENE_PX),
                  scotoma=Scotoma.circle(6), scotoma_fill=0.0,
                  scotoma_blend=3)
    profile = scene._native_rgb()[HALF, HALF:, 0, 0]
    npt.assert_almost_equal(profile[-1], 0.9, decimal=6)
    npt.assert_equal(0.25 < profile[6] / 0.9 < 0.75, True)
    npt.assert_equal(np.abs(np.diff(profile)).max() < 0.15, True)


def test_blending_reads_the_loss_field_past_the_frame_edge():
    """Blending includes scotoma loss outside the frame"""
    def edge(blend):
        flat = np.full((SCENE_PX, SCENE_PX), 0.8)
        scene = Scene(ImageStimulus(flat), fov=(SCENE_PX, SCENE_PX),
                      scotoma=Scotoma.circle(4, center=(-HALF - 5, 0)),
                      scotoma_fill=0.0, scotoma_blend=blend)
        return scene._native_rgb()[HALF, 0, 0, 0]
    npt.assert_almost_equal(edge(0), 0.8, decimal=6)
    npt.assert_equal(edge(2) < 0.75, True)


def test_a_softened_boundary_shows_in_the_composed_percept():
    """Blending also applies to the composed percept"""
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
    """Blend sigma is in dva, so the profile matches across resolutions"""
    scene = ramp_scene(scotoma=Scotoma.circle(6), scotoma_blend=2)
    xs = np.linspace(-12, 12, 25)
    profiles = []
    for shape in ((SCENE_PX, SCENE_PX), (5 * SCENE_PX, 5 * SCENE_PX)):
        axes = _raster_axes(scene.extent, shape)
        loss = scene._rendered_loss_on(*axes)
        # Horizontal meridian, where the circle's edge is at x = +/-6 dva:
        row = int(np.argmin(np.abs(axes[1])))
        profiles.append(np.interp(xs, axes[0], loss[row]))
    npt.assert_allclose(*profiles, atol=0.02)
    # The transition spans several dva, so a sigma in px would differ 5x:
    npt.assert_equal(0.05 < profiles[0][xs.tolist().index(6.0)] < 0.95, True)


def test_anisotropic_pixels_get_their_own_blur_sigma():
    """0.5 x 0.2 dva pixels: a 2-dva sigma is 4 columns but 10 rows"""
    half = SCENE_PX / 2
    scene = Scene(ImageStimulus(np.ones((5 * SCENE_PX, 2 * SCENE_PX))),
                  fov=(SCENE_PX, SCENE_PX), extent=(-half, half, -half, half),
                  scotoma=Scotoma.circle(6), scotoma_blend=2)
    xs, ys = scene._axes
    npt.assert_almost_equal(_raster_step(xs, ys), (0.5, 0.2))
    loss = scene._rendered_loss_on(xs, ys)
    out = np.linspace(0, 12, 61)
    row, col = int(np.argmin(np.abs(ys))), int(np.argmin(np.abs(xs)))
    # Circular scotoma: profile along +x matches profile along +y:
    along_x = np.interp(out, xs[col:], loss[row, col:])
    along_y = np.interp(out, ys[row::-1], loss[row::-1, col])
    npt.assert_allclose(along_x, along_y, atol=0.03)


def test_render_defaults_to_the_source_raster():
    """render() without shape or step uses the source raster"""
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
    """Pixel size is <= step; `fov` still bounds the outer pixel edges"""
    scene = ramp_scene()
    for step in (0.5, 0.3, 4.0, (1.0, 0.25)):
        drawn = scene.render(step=step)
        dx, dy = np.asarray(step, dtype=float) * np.ones(2)
        n_rows, n_cols = drawn.shape[:2]
        npt.assert_equal(SCENE_PX / n_cols <= dx + 1e-12, True)
        npt.assert_equal(SCENE_PX / n_rows <= dy + 1e-12, True)
        # One pixel fewer would exceed the step:
        npt.assert_equal(SCENE_PX / max(n_cols - 1, 1) > dx, True)
        # Outer pixel edges (not centers) span the FOV:
        npt.assert_almost_equal(
            (drawn.xdva[-1] - drawn.xdva[0]) * n_cols / max(n_cols - 1, 1),
            SCENE_PX, decimal=6)
    # A step that divides the field exactly adds no extra pixel:
    npt.assert_equal(scene.render(step=SCENE_PX / 41).shape[:2], (41, 41))
    # Unitful step gives the same shape:
    npt.assert_equal(scene.render(step=0.5 * dva).shape,
                     scene.render(step=0.5).shape)


def test_render_leaves_the_source_alone():
    scene = ramp_scene(scotoma=Scotoma.circle(6), scotoma_fill=0.0,
                       aperture='round')
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
    # Composition is pointwise, so a finer render agrees:
    percept = Percept(np.full((9, 9, 1), 3.0), space=scene._grid())
    fine = scene.render(percept=percept, vmax=4, shape=(45, 45))
    npt.assert_almost_equal(fine.data[22, 22, 0, 0],
                            scene.render(percept=percept,
                                         vmax=4).data[4, 4, 0, 0], decimal=6)


def test_a_color_fill_composes_a_percept_channel_by_channel():
    """max(fill, phosphene) is applied per channel"""
    native, fill, half = 0.8, (1.0, 0.0, 0.0), 0.5
    scene = Scene(ImageStimulus(np.full((9, 9), native)), fov=(9, 9),
                  scotoma=Scotoma(lambda x, y: np.full(np.shape(x), half)),
                  scotoma_fill='red', scotoma_blend=0)
    percept = Percept(np.full((9, 9, 1), 2.0), space=scene._grid())
    phosphene = 2.0 / 4.0
    lost = [max(c, phosphene) for c in fill]
    npt.assert_almost_equal(
        scene.render(percept=percept, vmax=4).data[4, 4, :, 0],
        [(1 - half) * native + half * c for c in lost], decimal=6)


def test_the_aperture_is_support_and_not_scene_data():
    """Aperture masks the display only; black inside it is scene data"""
    # Black square on a white field, to tell masked pixels from black data:
    data = np.ones((21, 21))
    data[9:12, 9:12] = 0.0
    scene = Scene(ImageStimulus(data), fov=(21, 21), aperture='round')
    before = scene.source.data.copy()
    drawn = scene.render().data[..., 0]
    npt.assert_array_equal(scene.source.data, before)
    npt.assert_almost_equal(drawn[10, 10], 0.0)
    npt.assert_almost_equal(drawn[10, 14], [1.0] * 3, decimal=6)
    # render() blacks out the corners outside the aperture:
    npt.assert_almost_equal(drawn[0, 0], 0.0)
    # plot() keeps the array and clips the artist instead:
    ax = scene.plot()
    npt.assert_almost_equal(ax.images[-1].get_array()[0, 0], [1.0] * 3,
                            decimal=6)
    npt.assert_almost_equal(ax.images[-1].get_array()[10, 10], 0.0)
    npt.assert_equal(ax.images[-1].get_clip_path() is not None, True)
    plt.close('all')
    # Device input ignores the aperture:
    x, y = np.array([-10.0, 0.0, 10.0]), np.array([10.0, 0.0, -10.0])
    plain = Scene(ImageStimulus(data), fov=(21, 21))
    npt.assert_array_equal(scene._device_input(x, y),
                           plain._device_input(x, y))


@pytest.mark.parametrize('view', ['eye', 'scene'])
def test_plotting_keeps_each_layer_at_its_own_resolution(view):
    """A 60 x 80 source and a 301 x 301 percept are drawn as separate images"""
    rng = np.random.default_rng(0)
    scene = Scene(ImageStimulus(rng.random((60, 80))), fov=(45, 33.75) * dva,
                  scotoma=Scotoma.circle(8), scotoma_fill=0.0,
                  scotoma_blend=0.5)
    # Checkerboard at the percept's step, dimmer in the top half; resampling
    # onto the source raster would erase both:
    data = np.zeros((301, 301, 1))
    data[::2, ::2, 0] = 8.0
    data[:150] *= 0.25
    percept = Percept(data, space=Grid2D((-3, 3), (-3, 3), step=6 / 300))
    ax = scene.plot(percept=percept, vmax=8, view=view)
    wide, patch = (image.get_array() for image in ax.images[-2:])
    # Source layers stay at source resolution:
    for image in ax.images[:-1]:
        npt.assert_equal(image.get_array().shape, (60, 80, 3))
    npt.assert_equal(patch.shape, (301, 301, 3))
    # The patch covers only the percept's extent:
    npt.assert_almost_equal(ax.images[-1].get_extent(),
                            (-3.01, 3.01, -3.01, 3.01), decimal=6)
    npt.assert_almost_equal(ax.images[-2].get_extent(),
                            (-22.5, 22.5, -16.875, 16.875), decimal=6)
    # Checkerboard is intact and the dim half is at the top:
    npt.assert_equal(patch[150, ::2, 0].min() > patch[150, 1::2, 0].max(),
                     True)
    npt.assert_equal(patch[0, 0, 0] < patch[-1, 0, 0], True)
    # A common raster at the percept's step would be much larger than both
    # images combined:
    common = scene._render_shape(6 / 300, None)
    npt.assert_equal(np.prod(common) > 20 * (60 * 80 + 301 ** 2), True)
    plt.close('all')


@pytest.mark.parametrize('view', ['eye', 'scene'])
def test_a_dark_percept_patch_is_ordinary_residual_vision(view):
    """A dark percept patch matches the residual view at its border"""
    scene = Scene(ImageStimulus(np.full((41, 41), 0.7)), fov=(41, 41),
                  scotoma=Scotoma.circle(14), scotoma_fill=0.25,
                  scotoma_blend=2)
    dark = Percept(np.zeros((21, 21, 1)),
                   space=Grid2D((-10, 10), (-10, 10), step=1))
    ax = scene.plot(percept=dark, vmax=5, view=view)
    wide, patch = (image.get_array() for image in ax.images[-2:])
    # Patch and wide layer agree where they overlap:
    npt.assert_almost_equal(patch[0, :], wide[HALF - 10, HALF - 10:HALF + 11],
                            decimal=6)
    npt.assert_almost_equal(patch[10, 10], wide[HALF, HALF], decimal=6)
    plt.close('all')


def test_plot_draws_native_vision_where_the_eye_is_pointing():
    """plot() draws native vision at the given gaze, on dva axes"""
    scene = ramp_scene(scotoma=Scotoma.circle(3), scotoma_fill=0.0)
    ax = scene.plot(gaze=(5, 0) * dva, view='eye')
    drawn = ax.images[-1].get_array()
    npt.assert_almost_equal(drawn, scene._native_rgb(gaze=(5, 0))[..., 0],
                            decimal=6)
    # Row 0 is the top of the field; axes are in dva:
    npt.assert_equal(ax.images[-1].origin, 'upper')
    npt.assert_almost_equal(ax.get_xlim(), (-HALF - 0.5, HALF + 0.5))
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


@pytest.mark.parametrize('view', ('scene', 'eye'))
def test_play_saves_a_gif_with_pillow(tmp_path, view):
    Image = pytest.importorskip('PIL.Image')
    frames = np.stack([np.full((6, 6), v) for v in (0.2, 0.8, 0.5)], axis=-1)
    scene = Scene(VideoStimulus(frames, time=[0, 10, 30]), fov=(6, 6))
    fname = str(tmp_path / 'scene.gif')
    scene.play(view=view, gaze=[(0, 0), (1, 0), (2, 0)]).save(
        fname, writer='pillow')
    plt.close('all')
    with Image.open(fname) as gif:
        npt.assert_equal(gif.n_frames, 3)


def test_the_fellow_eye_mirrors_an_asymmetric_scotoma():
    """fellow_eye() mirrors the scotoma in x; the scene is unchanged"""
    # Off both meridians, to tell a reflection from a rotation:
    scotoma = Scotoma.circle(3, center=(6, -3))
    scene = ramp_scene(scotoma=scotoma, aperture='round', background=0.25,
                       scotoma_fill=0.4, scotoma_blend=1.5)
    fellow = scene.fellow_eye()
    npt.assert_equal(fellow is scene, False)
    npt.assert_equal(isinstance(fellow, Scene), True)
    # x flips, y does not:
    npt.assert_almost_equal(float(fellow.scotoma(-6, -3)), 1.0)
    npt.assert_almost_equal(float(fellow.scotoma(-6, 3)), 0.0)
    npt.assert_almost_equal(float(fellow.scotoma(6, -3)), 0.0)
    # The original is unchanged:
    npt.assert_equal(scene.scotoma is scotoma, True)
    npt.assert_almost_equal(float(scene.scotoma(6, -3)), 1.0)
    # All other parameters carry over, including the source object:
    npt.assert_equal(fellow.source is scene.source, True)
    npt.assert_equal(fellow.fov, scene.fov)
    npt.assert_equal(fellow.aperture, scene.aperture)
    npt.assert_almost_equal(fellow.background, scene.background)
    npt.assert_equal(fellow.scotoma_fill, scene.scotoma_fill)
    npt.assert_equal(fellow.scotoma_blend, scene.scotoma_blend)


def test_the_fellow_eye_of_an_intact_field_is_intact_too():
    fellow = ramp_scene().fellow_eye()
    npt.assert_equal(fellow.scotoma, None)
    # The source image is not flipped:
    npt.assert_array_equal(fellow._native_rgb(), ramp_scene()._native_rgb())


def round_scene(**kwargs):
    """`ramp_scene` with a round aperture (a disc for a square fov)"""
    return ramp_scene(aperture='round', **kwargs)


def test_a_rectangular_aperture_is_the_default_and_fills_the_frame():
    scene = ramp_scene()
    npt.assert_equal(scene.aperture, 'rectangular')
    npt.assert_equal('aperture' in repr(scene), False)
    # Corners still show the ramp:
    native = scene._native_rgb()[..., 0]
    npt.assert_almost_equal(native[0, 0, 0], ramp_at(-HALF), decimal=6)
    npt.assert_almost_equal(native[0, -1, 0], ramp_at(HALF), decimal=6)


def test_a_square_fov_ellipse_keeps_the_center_and_blacks_out_the_corners():
    """render() blacks out the corners; native vision is unchanged"""
    scene = round_scene()
    npt.assert_equal(scene.aperture, 'round')
    npt.assert_equal("aperture='round'" in repr(scene), True)
    rect = rendered(ramp_scene())[..., 0]
    circ = rendered(scene)[..., 0]
    # Inside the disc nothing changes; the four corners are black:
    npt.assert_array_equal(circ[HALF, :, 0], rect[HALF, :, 0])
    npt.assert_array_equal(circ[:, HALF, 0], rect[:, HALF, 0])
    for row, col in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        npt.assert_almost_equal(circ[row, col], 0.0)
    # Right-hand corners are bright in the source:
    npt.assert_almost_equal(rect[0, -1, 0], ramp_at(HALF), decimal=6)
    npt.assert_almost_equal(rect[-1, -1, 0], ramp_at(HALF), decimal=6)
    npt.assert_array_equal(scene._native_rgb(), ramp_scene()._native_rgb())


@pytest.mark.parametrize('aperture', ['circle', 'ellipse', 'ROUND', '',
                                      None, 0, ['round']])
def test_a_bad_aperture_is_refused(aperture):
    with pytest.raises(ValueError):
        ramp_scene(aperture=aperture)


def test_an_ellipse_takes_a_semiaxis_from_each_side_of_the_field():
    """A 61 x 41 dva field gets an ellipse with semiaxes 30.5 and 20.5 dva"""
    data = np.tile(np.linspace(0.2, 1, 61), (41, 1))
    scene = Scene(ImageStimulus(data), fov=(61, 41), aperture='round')
    lit = rendered(scene)[..., 0, 0] > 0
    x, y = scene._pixel_centers()
    npt.assert_array_equal(lit, (x / 30.5) ** 2 + (y / 20.5) ** 2 <= 1)
    # (25, 0) dva is inside, though a disc of radius min(fov) / 2 would cut it:
    npt.assert_equal(lit[20, 55], True)          # (25, 0) dva
    npt.assert_equal(lit[5, 55], False)          # (25, 15) dva
    # Semiaxis endpoints are inside, corners are not:
    for row, col in ((20, 0), (20, -1), (0, 30), (-1, 30)):
        npt.assert_equal(lit[row, col], True)
    for row, col in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        npt.assert_equal(lit[row, col], False)


def test_the_aperture_is_eye_centered_and_stays_on_the_fovea():
    world = (-40.5, 40.5, -40.5, 40.5)
    scene = Scene(ImageStimulus(np.tile(np.linspace(0.2, 1, 81), (81, 1))),
                  fov=(SCENE_PX, SCENE_PX), extent=world, aperture='round')
    lit = rendered(scene, gaze=(8, -5) * dva)[..., 0, 0] > 0
    x, y = np.meshgrid(*scene._view_axes())
    # The disc is inscribed in the FOV regardless of gaze:
    npt.assert_array_equal(lit, x ** 2 + y ** 2 <= 20.5 ** 2)


def test_a_transparent_background_does_not_leak_outside_the_aperture():
    """Outside the aperture is black even with a white background"""
    scene = Scene(rgba_source(), fov=(8, 8), background=1, aperture='round')
    native = rendered(scene)[..., 0]
    npt.assert_almost_equal(native[0, 0], 0.0)
    # Inside, the background shows through:
    npt.assert_almost_equal(native[0, 4], [1.0, 1.0, 1.0], decimal=6)


def test_the_aperture_does_not_change_what_a_device_is_given():
    """Aperture affects rendering only, not sampling or device input"""
    x = np.array([-19.0, -8.0, 0.0, 6.5, 18.0, 30.0])
    y = np.array([17.0, -3.0, 0.0, 11.25, -19.0, 0.0])
    rect, ellip = ramp_scene(), round_scene()
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
    ellip = ramp_scene(scotoma=scotoma, scotoma_fill=0.5, aperture='round')
    npt.assert_array_equal(ellip.scotoma(0.0, 0.0), rect.scotoma(0.0, 0.0))
    inside = np.hypot(*ellip._pixel_centers()) <= 20.5
    npt.assert_array_equal(ellip._native_rgb()[..., 0][inside],
                           rect._native_rgb()[..., 0][inside])


def test_the_aperture_clips_a_composed_percept_only_when_it_is_drawn():
    scotoma = Scotoma.circle(6)
    bright = Percept(np.ones((SCENE_PX, SCENE_PX, 1)),
                     space=ramp_scene()._grid())
    rect = ramp_scene(scotoma=scotoma, scotoma_fill=0.0)
    ellip = ramp_scene(scotoma=scotoma, scotoma_fill=0.0, aperture='round')
    seen_rect = rendered(rect, percept=bright, vmax=1)[..., 0]
    seen_ellip = rendered(ellip, percept=bright, vmax=1)[..., 0]
    inside = np.hypot(*ellip._pixel_centers()) <= 20.5
    npt.assert_array_equal(seen_ellip[inside], seen_rect[inside])
    npt.assert_almost_equal(seen_ellip[0, -1], 0.0)
    npt.assert_almost_equal(seen_rect[0, -1, 0], ramp_at(HALF), decimal=6)
    # plot() clips the artist instead of masking the array:
    ax = ellip.plot(percept=bright, vmax=1, view='eye')
    npt.assert_array_equal(ax.images[0].get_array(),
                           rect._native_rgb()[..., 0])
    npt.assert_equal(ax.images[0].get_clip_path() is not None, True)
    plt.close('all')


def test_a_percept_can_be_plotted_in_the_context_of_the_whole_field():
    """A small percept is drawn at its own extent on a black field"""
    space = Scene(ImageStimulus(np.zeros((9, 9))), fov=(9, 9))._grid()
    data = np.zeros((9, 9, 1))
    data[4, 4] = 20.0
    phosphene = Percept(data, space=space)
    scene = round_scene()
    ax = scene.plot(percept=phosphene, vmax=20, view='eye')
    wide, patch = (image.get_array() for image in ax.images)
    # Each layer keeps its own resolution:
    npt.assert_equal(wide.shape, (SCENE_PX, SCENE_PX, 3))
    npt.assert_equal(patch.shape, (9, 9, 3))
    # The percept is centered on the fovea at its own size:
    npt.assert_almost_equal(ax.images[1].get_extent(),
                            (-4.5, 4.5, -4.5, 4.5), decimal=6)
    npt.assert_almost_equal(patch[4, 4], [1.0, 1.0, 1.0], decimal=6)
    npt.assert_almost_equal(patch[4, 7], 0.0)
    # Native vision of a black source is black:
    npt.assert_almost_equal(wide, 0.0)
    plt.close('all')
    # Default vmax is the percept's maximum:
    auto = scene.plot(percept=phosphene, view='eye').images[1].get_array()
    npt.assert_almost_equal(auto, patch, decimal=6)
    plt.close('all')
    # vmin or vmax without a percept is a ValueError:
    with pytest.raises(ValueError):
        scene.plot(vmax=20)
    with pytest.raises(ValueError):
        scene.plot(vmin=5)


def test_plotting_a_percept_over_a_scotoma_is_the_composed_view():
    """On the scene's raster, plot() matches render()"""
    scene = ramp_scene(scotoma=Scotoma.circle(6), scotoma_fill=0.0)
    bright = Percept(np.ones((SCENE_PX, SCENE_PX, 1)), space=scene._grid())
    ax = scene.plot(percept=bright, vmax=1, view='eye')
    npt.assert_almost_equal(ax.images[1].get_array(),
                            rendered(scene, percept=bright, vmax=1)[..., 0],
                            decimal=6)
    # The wide layer is residual native vision:
    npt.assert_almost_equal(ax.images[0].get_array(),
                            scene._native_rgb()[..., 0], decimal=6)
    plt.close('all')


def test_rings_still_land_on_the_fovea_the_aperture_is_centered_on():
    scene = round_scene()
    ax = scene.plot(gaze=(7, -3) * dva, rings=[10], view='eye')
    # 10-dva ring centered on the fovea (FOV center):
    ring = ax.lines[-1]
    xs, ys = ring.get_xdata(), ring.get_ydata()
    npt.assert_almost_equal([xs.min(), xs.max()], [-10.0, 10.0], decimal=6)
    npt.assert_almost_equal([ys.min(), ys.max()], [-10.0, 10.0], decimal=6)
    # The outermost default ring fits inside the aperture:
    npt.assert_almost_equal(scene._grid_geometry(True, False)[0].max(), 20.0)
    plt.close('all')


def test_the_grid_is_clipped_to_an_elliptical_aperture():
    scene = round_scene()
    ax = scene.plot(gaze=(7, -3) * dva, rings=[10, 30], meridians=True)
    for artist in ax.get_lines() + list(ax.texts):
        npt.assert_equal(artist.get_clip_path() is not None, True)
    plt.close('all')
    # play() blanks the overlay outside the aperture:
    frames = np.stack([np.ones((SCENE_PX, SCENE_PX))] * 2, axis=-1)
    video = Scene(VideoStimulus(frames, time=[0, 1000]),
                  fov=(SCENE_PX, SCENE_PX), aperture='round')
    ruled = video.play(meridians=True, view='eye')._frame_data
    outside = video._aperture_mask(*video._view_axes())
    npt.assert_array_equal(ruled[outside], 0)
    plt.close('all')


def test_the_aperture_reaches_the_frames_the_player_shows():
    frames = np.stack([np.full((9, 9), v) for v in (0.4, 0.8)], axis=-1)
    scene = Scene(VideoStimulus(frames, time=[0, 10]), fov=(9, 9),
                  aperture='round')
    ani = scene.play(view='eye')
    npt.assert_almost_equal(ani._frame_data[0, 0, :, 0], 0.0)
    npt.assert_almost_equal(ani._frame_data[4, 4, :, 1], 0.8, decimal=6)
    plt.close('all')


def gray_video_scene(n_frames=3, **kwargs):
    """Ramp scene scaled by a different factor in each frame"""
    ramp = np.tile(np.linspace(0, 1, SCENE_PX), (SCENE_PX, 1))
    frames = np.stack([ramp * w for w in np.linspace(0.4, 1.0, n_frames)],
                      axis=-1)
    kwargs.setdefault('scotoma_blend', 0)
    return Scene(VideoStimulus(frames, time=np.arange(n_frames) * 10.0),
                 fov=(SCENE_PX, SCENE_PX), **kwargs)


def rgb_video_scene(n_frames=3, **kwargs):
    """RGB video with distinct channels, to detect channel mix-ups"""
    rgb = np.stack([np.tile(np.linspace(0, 1, SCENE_PX), (SCENE_PX, 1)),
                    np.tile(np.linspace(1, 0, SCENE_PX), (SCENE_PX, 1)),
                    np.full((SCENE_PX, SCENE_PX), 0.5)], axis=-1)
    frames = np.stack([rgb * w for w in np.linspace(0.4, 1.0, n_frames)],
                      axis=-1)
    kwargs.setdefault('scotoma_blend', 0)
    return Scene(VideoStimulus(frames, time=np.arange(n_frames) * 10.0),
                 fov=(SCENE_PX, SCENE_PX), **kwargs)


def ramped_percept(scene, n_frames):
    """Percept on the scene's grid, brighter in each frame"""
    frame = np.tile(np.linspace(0, 1, SCENE_PX), (SCENE_PX, 1))
    data = np.stack([frame * w for w in np.linspace(0.3, 1.0, n_frames)],
                    axis=-1)
    return Percept(data, space=scene._grid(),
                   time=np.arange(n_frames) * 10.0)


def frame_scene(scene, f, **kwargs):
    """Still scene of frame ``f`` with the video scene's parameters"""
    frames = scene.source.data.reshape(scene.source.vid_shape)
    return Scene(ImageStimulus(frames[..., f]), fov=scene.fov,
                 scotoma=scene.scotoma, scotoma_fill=scene.scotoma_fill,
                 scotoma_blend=scene.scotoma_blend,
                 aperture=scene.aperture, **kwargs)


def test_composing_a_grayscale_video_keeps_the_canonical_rgb_layout():
    """A gray source is composed as RGB with identical channels"""
    n = 3
    scene = gray_video_scene(n, scotoma=Scotoma.circle(6), scotoma_fill=0.0)
    seen = rendered(scene, percept=ramped_percept(scene, n), vmax=1)
    npt.assert_equal(seen.shape, (SCENE_PX, SCENE_PX, 3, n))
    npt.assert_equal(np.asarray(seen).dtype, np.float32)
    npt.assert_array_equal(seen[:, :, 0, :], seen[:, :, 1, :])
    npt.assert_array_equal(seen[:, :, 0, :], seen[:, :, 2, :])
    # Outside the scotoma, native vision is unchanged:
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
    # Under complete loss, max(fill, phosphene); phosphene is dim in frame 0
    # and bright in frame -1:
    for f in (0, n - 1):
        phosphene = float(ramped_percept(scene, n).data[HALF, HALF, f])
        npt.assert_almost_equal(seen[HALF, HALF, :, f],
                                [max(0.2, phosphene)] * 3, decimal=6)


@pytest.mark.parametrize('blend', [0, 2])
def test_composition_follows_a_gaze_that_moves_between_frames(blend):
    """With per-frame gaze, frame f matches frame f composed alone"""
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
    # Differs from a static gaze:
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
    # Frames differ:
    npt.assert_equal(np.allclose(seen[..., 0], seen[..., -1]), False)


def test_an_elliptical_aperture_survives_composing_a_video():
    n = 3
    scene = gray_video_scene(n, scotoma=Scotoma.circle(6), scotoma_fill=0.0,
                             aperture='round')
    percept = ramped_percept(scene, n)
    seen = rendered(scene, percept=percept, vmax=1)
    npt.assert_equal(seen.shape, (SCENE_PX, SCENE_PX, 3, n))
    npt.assert_equal(np.asarray(seen).dtype, np.float32)
    npt.assert_array_equal(seen[0, 0], 0)
    npt.assert_array_equal(seen[-1, -1], 0)
    # Inside the disc, same as a rectangular aperture:
    rect = gray_video_scene(n, scotoma=Scotoma.circle(6), scotoma_fill=0.0)
    inside = ~scene._aperture_mask(*scene._view_axes())
    npt.assert_array_equal(seen[inside],
                           rendered(rect, percept=percept, vmax=1)[inside])


def frames_evaluated(monkeypatch):
    """Records the number of frames sampled from source and percept"""
    counts = {'source': [], 'percept': []}
    source_on, percept_on = Scene._source_on, scene_module._percept_on

    def counted_source(self, xs, ys, gaze_xy=(0.0, 0.0), frame=None):
        out = source_on(self, xs, ys, gaze_xy, frame=frame)
        counts['source'].append(out.shape[-1])
        return out

    def counted_percept(prosthetic, frames, xs, ys):
        counts['percept'].append(frames.shape[-1])
        return percept_on(prosthetic, frames, xs, ys)

    monkeypatch.setattr(Scene, '_source_on', counted_source)
    monkeypatch.setattr(scene_module, '_percept_on', counted_percept)
    return counts


@pytest.mark.parametrize('view', ['eye', 'scene'])
def test_plotting_one_frame_evaluates_only_that_frame(monkeypatch, view):
    """plot(frame=k) evaluates only frame k"""
    n = 6
    scene = gray_video_scene(n, scotoma=Scotoma.circle(6), scotoma_fill=0.0,
                             scotoma_blend=2)
    percept = ramped_percept(scene, n)
    counts = frames_evaluated(monkeypatch)
    ax = scene.plot(percept=percept, vmax=1, frame=4, ax=plt.subplots()[1],
                    view=view)
    # One source frame each for patch and wide layer, one percept frame:
    npt.assert_equal(counts['source'], [1, 1])
    npt.assert_equal(counts['percept'], [1])
    # Frame 4 matches the full render:
    dense = rendered(scene, percept=percept, vmax=1)
    npt.assert_almost_equal(ax.images[-1].get_array(), dense[..., 4],
                            decimal=6)
    npt.assert_almost_equal(ax.images[-2].get_array(),
                            scene._native_rgb()[..., 4], decimal=6)
    plt.close('all')


@pytest.mark.parametrize('view', ['eye', 'scene'])
def test_a_still_scene_narrows_to_the_requested_percept_frame(monkeypatch,
                                                              view):
    """A still scene evaluates one source and one percept frame"""
    n = 5
    scene = ramp_scene(scotoma=Scotoma.circle(6), scotoma_fill=0.0)
    percept = ramped_percept(scene, n)
    counts = frames_evaluated(monkeypatch)
    ax = scene.plot(percept=percept, vmax=1, frame=3, ax=plt.subplots()[1],
                    view=view)
    npt.assert_equal(counts['percept'], [1])
    npt.assert_equal(counts['source'], [1, 1])
    npt.assert_almost_equal(ax.images[-1].get_array(),
                            rendered(scene, percept=percept,
                                     vmax=1)[..., 3], decimal=6)
    plt.close('all')


@pytest.mark.parametrize('view', ['eye', 'scene'])
def test_plotting_a_video_frame_without_a_percept_reads_one_frame(monkeypatch,
                                                                  view):
    n = 6
    scene = gray_video_scene(n, scotoma=Scotoma.circle(6), scotoma_fill=0.2)
    counts = frames_evaluated(monkeypatch)
    ax = scene.plot(frame=2, ax=plt.subplots()[1], view=view)
    npt.assert_equal(counts['source'], [1])
    npt.assert_almost_equal(ax.images[-1].get_array(),
                            scene._native_rgb()[..., 2], decimal=6)
    plt.close('all')


def test_render_still_rasterizes_the_whole_video(monkeypatch):
    """render() evaluates all frames"""
    n = 6
    scene = gray_video_scene(n, scotoma=Scotoma.circle(6), scotoma_fill=0.0)
    percept = ramped_percept(scene, n)
    counts = frames_evaluated(monkeypatch)
    npt.assert_equal(scene.render(percept=percept, vmax=1).shape[-1], n)
    npt.assert_equal(counts['source'], [n])
    npt.assert_equal(counts['percept'], [n])


def test_narrowing_aligns_the_percept_at_that_scene_time_alone():
    """With frame=k, the percept is resampled at frame k's time only"""
    n = 4
    ramp = np.tile(np.linspace(0, 1, SCENE_PX), (SCENE_PX, 1))
    frames = np.stack([ramp * w for w in np.linspace(0.4, 1.0, n)], axis=-1)
    scene = Scene(VideoStimulus(frames, time=[0.0, 10.0, 20.0, 30.0]),
                  fov=(SCENE_PX, SCENE_PX))
    # Different time points, so the percept must be resampled:
    percept = Percept(np.stack([ramp * b for b in (0.0, 3.0)], axis=-1),
                      space=scene._grid(), time=[-5.0, 35.0])
    full, full_time, unit = scene._prosthetic_frames(percept)
    npt.assert_equal(full.shape[-1], n)
    for f in range(n):
        one, one_time, one_unit = scene._prosthetic_frames(percept, frame=f)
        npt.assert_equal(one.shape[-1], 1)
        npt.assert_array_equal(one[..., 0], full[..., f])
        npt.assert_almost_equal(one_time, np.asarray(full_time)[f:f + 1])
        npt.assert_equal(one_unit, unit)
    # A percept that does not cover the whole video is a ValueError for any
    # frame:
    short = Percept(percept.data, space=scene._grid(), time=[0.0, 20.0])
    for frame in (None, 0):
        with pytest.raises(ValueError):
            scene._prosthetic_frames(short, frame=frame)


@pytest.mark.parametrize('blend', [0, 2])
def test_the_loss_composition_renders_is_float32(blend):
    """Loss map is float32, to avoid upcasting the composition"""
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
    # Pixel centers match the source raster axes:
    npt.assert_array_equal(x[0], scene._axes[0])
    npt.assert_array_equal(y[:, 0], scene._axes[1])


def test_a_blank_scene_is_a_black_elliptical_field_by_default():
    scene = Scene.blank()
    npt.assert_equal(scene.fov, (45.0, 45.0))
    npt.assert_equal(scene.shape, (512, 512))
    npt.assert_equal(scene.aperture, 'round')
    data = scene.source.data
    npt.assert_equal(np.all(np.isfinite(data)), True)
    npt.assert_array_equal(data, 0.0)


def test_a_blank_scene_is_an_ordinary_black_scene():
    """Scene.blank() behaves like a Scene of zeros"""
    blank = Scene.blank(fov=45, aperture='rectangular')
    plain = Scene(np.zeros(blank.shape), fov=45, aperture='rectangular')
    npt.assert_equal(blank.fov, plain.fov)
    npt.assert_equal(blank.shape, plain.shape)
    for x, y in [(0, 0), (-10, 4), (12, -8)]:
        npt.assert_almost_equal(blank._sample_at(x, y), plain._sample_at(x, y))
        npt.assert_almost_equal(blank._device_input(x, y),
                                plain._device_input(x, y))
    npt.assert_almost_equal(rendered(blank), rendered(plain))


def test_a_blank_scene_keeps_the_fov_it_is_given():
    npt.assert_equal(Scene.blank(fov=(60, 40) * dva).fov, (60.0, 40.0))
    # The backing raster is square, so a scalar fov stays square:
    npt.assert_equal(Scene.blank(fov=60).fov, (60.0, 60.0))


def test_the_blank_raster_is_not_the_callers_business():
    """Scene.blank() does not accept a shape"""
    with pytest.raises(TypeError):
        Scene.blank(shape=(32, 48))
    # Output resolution is set in render(); fov is unchanged:
    scene = Scene.blank(fov=45)
    npt.assert_equal(scene.render(shape=(1080, 1920)).shape[:2], (1080, 1920))
    npt.assert_equal(scene.fov, (45.0, 45.0))


def test_an_explicit_aperture_overrides_the_blank_default():
    npt.assert_equal(Scene.blank(aperture='rectangular').aperture,
                     'rectangular')
    npt.assert_equal(Scene.blank().aperture, 'round')


@pytest.mark.parametrize('view', ['eye', 'scene'])
def test_a_blank_scene_does_not_resample_a_finer_percept(view):
    """A percept finer than the blank raster keeps its resolution"""
    scene = Scene.blank(fov=45 * dva)
    # Finer than the 45 / 512 dva blank raster:
    space = Grid2D((-2, 2), (-2, 2), step=0.02)
    data = np.zeros(space.x.shape + (1,))
    data[100, 100] = 5.0
    ax = scene.plot(percept=Percept(data, space=space), vmax=5, view=view)
    wide, patch = (image.get_array() for image in ax.images[-2:])
    npt.assert_equal(wide.shape, (512, 512, 3))
    npt.assert_equal(patch.shape, (201, 201, 3))
    npt.assert_almost_equal(ax.images[-1].get_extent(),
                            (-2.01, 2.01, -2.01, 2.01), decimal=6)
    npt.assert_almost_equal(patch[100, 100], [1.0, 1.0, 1.0], decimal=6)
    plt.close('all')


def test_a_blank_scene_gives_a_device_black():
    scene = Scene.blank(fov=45 * dva)
    for x, y in [(0, 0), (-20, 0), (0, 20), (10, -10), (22, 22)]:
        npt.assert_almost_equal(seen_at(scene, x, y), 0.0)


# World extent vs. field of view
#
# `WORLD_PX` pixels at 1 dva each, so `WORLD` spans 81 dva and pixel centers
# land on whole degrees. Red encodes scene x, green scene y.
WORLD_PX = 81
WORLD = (-40.5, 40.5, -40.5, 40.5)


def coordinate_image():
    """RGB image with R = (x + 40) / 80 and G = (y + 40) / 80, scene dva"""
    ramp = np.linspace(0, 1, WORLD_PX)
    red = np.tile(ramp, (WORLD_PX, 1))
    green = np.tile(ramp[::-1, np.newaxis], (1, WORLD_PX))
    return np.stack([red, green, np.zeros_like(red)], axis=-1)


def world_scene(source=None, **kwargs):
    """21-dva FOV on the 81-dva coordinate image"""
    kwargs.setdefault('fov', (21, 21))
    kwargs.setdefault('scotoma_blend', 0)
    return Scene(ImageStimulus(coordinate_image()) if source is None
                 else source, extent=WORLD, **kwargs)


def outer_extent(percept):
    """Returns ``(left, right, bottom, top)`` of a rendered percept"""
    return _raster_extent(percept.xdva, percept.ydva[::-1])


@pytest.mark.parametrize('shape, extent', [
    # Landscape: the height spans the FOV, the width keeps the aspect ratio
    ((173, 320), (-20 * 320 / 173, 20 * 320 / 173, -20, 20)),
    # Portrait: the width spans the FOV
    ((320, 173), (-20, 20, -20 * 320 / 173, 20 * 320 / 173)),
    # Square: extent and FOV coincide
    ((64, 64), (-20, 20, -20, 20)),
])
def test_a_scalar_fov_is_a_square_window_inside_the_inferred_extent(shape,
                                                                    extent):
    scene = Scene(np.zeros(shape), fov=40 * dva)
    npt.assert_almost_equal(scene.fov, (40, 40))
    npt.assert_almost_equal(scene.extent, extent)
    # Square pixels:
    dx, dy = scene._angular_pixel
    npt.assert_almost_equal(dx, dy)
    # The render covers the FOV, not the extent:
    npt.assert_almost_equal(outer_extent(scene.render()), (-20, 20, -20, 20),
                            decimal=5)
    npt.assert_equal(scene.render().shape[:2], (min(shape), min(shape)))


def test_a_landscape_video_source_infers_a_74_by_40_degree_extent():
    video = VideoStimulus(np.zeros((173, 320, 2)), time=[0, 100])
    scene = Scene(video, fov=40 * dva)
    left, right, bottom, top = scene.extent
    npt.assert_almost_equal((right - left, top - bottom), (74.0, 40.0),
                            decimal=1)
    npt.assert_almost_equal(outer_extent(scene.render()), (-20, 20, -20, 20),
                            decimal=5)


@pytest.mark.parametrize('shape, fov, size', [
    # FOV wider than the source aspect ratio: width sets the extent
    ((100, 100), (40, 30), (40, 40)),
    # FOV taller: height sets the extent
    ((173, 320), (40, 30), (30 * 320 / 173, 30)),
    ((50, 100), (40, 30), (60, 30)),
])
def test_a_rectangular_fov_gets_the_smallest_extent_that_contains_it(
        shape, fov, size):
    scene = Scene(np.zeros(shape), fov=fov * dva)
    left, right, bottom, top = scene.extent
    npt.assert_almost_equal((right - left, top - bottom), size)
    npt.assert_almost_equal((right - left) / (top - bottom),
                            shape[1] / shape[0])
    npt.assert_almost_equal(outer_extent(scene.render()),
                            (-fov[0] / 2, fov[0] / 2, -fov[1] / 2, fov[1] / 2),
                            decimal=5)


def test_an_explicit_extent_is_kept_exactly():
    scene = Scene(np.zeros((60, 100)), extent=(-50, 50, -30, 30) * dva,
                  fov=40 * dva)
    npt.assert_equal(scene.extent, (-50.0, 50.0, -30.0, 30.0))
    npt.assert_equal(scene.fov, (40.0, 40.0))
    npt.assert_almost_equal(scene.pixel_to_dva(0, 0), (-49.5, 29.5))
    # Off-center extents and non-square pixels are allowed:
    shifted = Scene(np.zeros((10, 10)), extent=(0, 20, -5, 5), fov=10)
    npt.assert_almost_equal(shifted.pixel_to_dva(0, 0), (1.0, 4.5))
    npt.assert_almost_equal(shifted._angular_pixel, (2.0, 1.0))


@pytest.mark.parametrize('shape, extent', [
    # The shorter dimension spans the scalar, the other keeps the aspect ratio
    ((173, 320), (-22.5 * 320 / 173, 22.5 * 320 / 173, -22.5, 22.5)),
    ((320, 173), (-22.5, 22.5, -22.5 * 320 / 173, 22.5 * 320 / 173)),
    ((64, 64), (-22.5, 22.5, -22.5, 22.5)),
])
def test_a_scalar_extent_spans_the_shorter_source_dimension(shape, extent):
    scene = Scene(np.zeros(shape), fov=40, extent=45 * dva)
    npt.assert_almost_equal(scene.extent, extent)
    dx, dy = scene._angular_pixel
    npt.assert_almost_equal(dx, dy)
    npt.assert_equal(scene.fov, (40.0, 40.0))
    # Unitless values are dva; an extent smaller than the FOV is allowed:
    npt.assert_almost_equal(Scene(np.zeros(shape), fov=40, extent=45).extent,
                            extent)
    small = Scene(np.zeros(shape), fov=40, extent=10)
    npt.assert_almost_equal(min(small.extent[1] - small.extent[0],
                                small.extent[3] - small.extent[2]), 10)


@pytest.mark.parametrize('extent', [(0, 1, 2), (1, 0, 0, 1), (0, 1, 1, 1),
                                    (0, np.inf, 0, 1), (0, 1, np.nan, 1),
                                    0, -45, np.nan, np.inf])
def test_a_bad_extent_is_refused(extent):
    with pytest.raises(ValueError):
        Scene(np.zeros((8, 8)), fov=8, extent=extent)


def test_gaze_moves_the_window_through_a_fixed_world():
    scene = world_scene()
    source = coordinate_image()
    for gx, gy in ((0, 0), (10, -5), (-17, 12)):
        seen = scene.render(gaze=(gx, gy) * dva)
        # Eye-centered axes regardless of gaze:
        npt.assert_almost_equal(outer_extent(seen),
                                (-10.5, 10.5, -10.5, 10.5), decimal=5)
        # Source pixels map one to one; scene x = gx is column gx + 40:
        col, row = 40 + gx - 10, 40 - gy - 10
        npt.assert_almost_equal(seen.data[..., 0],
                                source[row:row + 21, col:col + 21],
                                decimal=6)
    npt.assert_equal(scene.extent, WORLD)


def test_the_aperture_is_inscribed_in_the_fov_and_stays_on_fixation():
    scene = world_scene(aperture='round')
    x, y = np.meshgrid(*scene._view_axes())
    for gaze in ((0, 0), (10, -5)):
        lit = scene.render(gaze=gaze).data[..., 0].sum(axis=-1) > 0
        npt.assert_array_equal(lit, x ** 2 + y ** 2 <= 10.5 ** 2)
    # A rectangular aperture shows the whole window:
    white = world_scene(np.ones((WORLD_PX, WORLD_PX)))
    npt.assert_array_equal(white.render(gaze=(10, -5)).data, 1.0)


def test_the_scotoma_stays_on_fixation_at_its_angular_size():
    scene = world_scene(np.ones((WORLD_PX, WORLD_PX)),
                        scotoma=Scotoma.circle(5), scotoma_fill=0.0)
    x, y = np.meshgrid(*scene._view_axes())
    for gaze in ((0, 0), (10, -5), (-17, 12)):
        lost = scene.render(gaze=gaze).data[..., 0, 0] < 0.5
        npt.assert_array_equal(lost, x ** 2 + y ** 2 <= 25)
    # Radius stays 5 dva at a finer render step:
    fine = scene.render(gaze=(10, -5), step=0.25)
    x, y = np.meshgrid(fine.xdva, fine.ydva[::-1])
    npt.assert_array_equal(fine.data[..., 0, 0] < 0.5, x ** 2 + y ** 2 <= 25)


def test_image_and_video_sources_see_the_same_world():
    image = world_scene()
    frames = np.stack([coordinate_image()] * 2, axis=-1)
    video = world_scene(VideoStimulus(frames, time=[0, 100]))
    npt.assert_equal(video.extent, image.extent)
    gaze = [(10, -5), (-17, 12)]
    moving = video.render(gaze=gaze * dva).data
    for f, g in enumerate(gaze):
        npt.assert_array_equal(moving[..., f],
                               image.render(gaze=g).data[..., 0])


def test_render_and_the_eye_view_show_the_same_window():
    frames = np.stack([coordinate_image()] * 2, axis=-1)
    scene = world_scene(VideoStimulus(frames, time=[0, 100]),
                        scotoma=Scotoma.circle(3), scotoma_fill=0.2)
    gaze = [(10, -5), (-17, 12)]
    rendered_frames = scene.render(gaze=gaze).data
    # Full 21-dva FOV, including edge pixels:
    fov = (-10.5, 10.5)
    for grid in ({}, {'rings': [5]}):
        ani = scene.play(gaze=gaze, view='eye', **grid)
        if not grid:
            npt.assert_array_equal(ani._frame_data, rendered_frames)
        npt.assert_almost_equal(ani._image.axes.get_xlim(), fov)
        npt.assert_almost_equal(ani._image.axes.get_ylim(), fov)
    for f in range(2):
        ax = scene.plot(gaze=gaze, frame=f, ax=plt.subplots()[1],
                        view='eye')
        npt.assert_almost_equal(ax.images[0].get_array(),
                                rendered_frames[..., f], decimal=6)
        npt.assert_almost_equal(ax.get_xlim(), fov)
        npt.assert_almost_equal(ax.get_ylim(), fov)
    plt.close('all')


def test_a_percept_stays_registered_with_what_the_device_samples():
    """Rendered percept and device input use the same scene coordinates"""
    scene = world_scene(scotoma=Scotoma.circle(8), scotoma_fill=0.0)
    # A single bright pixel at eye-centered (3, 2):
    space = Grid2D((-6, 6), (-6, 6), step=1)
    data = np.zeros(space.x.shape + (1,))
    data[(space.y == 2) & (space.x == 3)] = 1.0
    percept = Percept(data, space=space)
    for gaze in ((0, 0), (10, -5)):
        seen = scene.render(percept=percept, gaze=gaze, vmax=1).data[..., 0]
        # Eye (3, 2) is column 10 + 3 and row 10 - 2 of the 21 x 21 window:
        npt.assert_almost_equal(seen[8, 13], 1.0, decimal=6)
        npt.assert_almost_equal(seen[12, 7], 0.0, decimal=6)
        # Outside the scotoma, the render matches device input:
        gray = scene._device_input(-9.0, 0.0, gaze=gaze)
        npt.assert_almost_equal(rgb2gray(seen[10:11, 1:2]).item(),
                                np.ravel(gray)[0], decimal=6)


def test_the_device_reads_the_world_outside_the_fov():
    """`fov` limits the display only, not device sampling"""
    # Eye (15, -12) at gaze (10, -5) is scene (25, -17): inside the 81-dva
    # extent, outside a 21-dva FOV:
    x, y, gaze = 15.0, -12.0, (10, -5)
    expected = [(25 + 40) / 80, (-17 + 40) / 80, 0.0]
    scenes = [world_scene(fov=fov, aperture=aperture)
              for fov in ((21, 21), (5, 5), (81, 81))
              for aperture in ('rectangular', 'round')]
    sampled = [scene._sample_at(x, y, gaze=gaze) for scene in scenes]
    for values in sampled:
        npt.assert_almost_equal(np.ravel(values), expected, decimal=6)
        npt.assert_array_equal(values, sampled[0])
    npt.assert_array_equal(
        world_scene(fov=(5, 5))._device_input(x, y, gaze=gaze),
        world_scene(fov=(81, 81))._device_input(x, y, gaze=gaze))


def test_gaze_finds_a_source_placed_off_center():
    """An off-center `extent` is not recentered"""
    source = np.random.default_rng(0).uniform(0.1, 1.0, (21, 21))
    # Pixel centers at scene x = 30..50, y = -10..10:
    scene = Scene(ImageStimulus(source), fov=(21, 21),
                  extent=(29.5, 50.5, -10.5, 10.5))
    # Gaze (0, 0) shows nothing:
    npt.assert_array_equal(scene.render().data, 0.0)
    # Gaze at its center shows the whole source:
    npt.assert_almost_equal(scene.render(gaze=(40, 0)).data[..., 0, 0],
                            source, decimal=6)
    # Gaze at its left edge shows its left half in the right of the FOV:
    seen = scene.render(gaze=(30, 0)).data[..., 0, 0]
    npt.assert_array_equal(seen[:, :10], 0.0)
    npt.assert_almost_equal(seen[:, 10:], source[:, :11], decimal=6)


def test_a_fov_past_the_edge_of_the_world_is_black():
    """Outside `extent` is black, not the background color"""
    source = np.ones((WORLD_PX, WORLD_PX, 4))
    source[..., 3] = 0.0
    # Transparent 81-dva source on a white background:
    scene = world_scene(ImageStimulus(source), background=1, fov=(41, 41))
    seen = scene.render(gaze=(30, 0)).data[..., 0, 0]
    x, _ = np.meshgrid(*scene._view_axes())
    npt.assert_array_equal(seen[x + 30 < 40.5], 1.0)
    npt.assert_array_equal(seen[x + 30 > 40.5], 0.0)
    # Device input is black too:
    npt.assert_almost_equal(seen_at(scene, 45.0), 0.0)


# Scene view: fixed world axes, the FOV moving with gaze
#
# World pixel (row, col) is at scene (col - 40, 40 - row).
WORLD_X, WORLD_Y = np.meshgrid(np.arange(-40.0, 41.0),
                               np.arange(40.0, -41.0, -1.0))


def in_fov(gaze, aperture='rectangular', half=10.5):
    """Returns a mask of world pixels inside a 21-dva FOV at ``gaze``"""
    x, y = WORLD_X - gaze[0], WORLD_Y - gaze[1]
    if aperture == 'round':
        return x ** 2 + y ** 2 <= half ** 2
    return (np.abs(x) <= half) & (np.abs(y) <= half)


def clip_bounds(ax, artist):
    """Returns the artist's clip bounds ``(left, bottom, right, top)``"""
    path = artist.get_clip_path()
    # Matplotlib turns a rectangular clip path into a clip box:
    bbox = (artist.get_clip_box() if path is None
            else path.get_fully_transformed_path().get_extents())
    return tuple(ax.transData.inverted().transform(bbox.get_points()).ravel())


def world_video(n_frames=2, source=None, **kwargs):
    """`world_scene` as a video, one second per frame"""
    frame = coordinate_image() if source is None else source
    frames = np.stack([frame] * n_frames, axis=-1)
    return world_scene(VideoStimulus(frames, time=np.arange(n_frames) * 1000),
                       **kwargs)


def test_the_scene_view_is_the_default_and_spans_the_extent():
    scene = world_scene()
    source = coordinate_image()
    for gaze in ((0, 0), (10, -5), (35, 0)):
        ax = drawn_on_fresh_axes(scene, gaze=gaze * dva)
        # Axes span the extent regardless of gaze:
        npt.assert_almost_equal(ax.get_xlim(), WORLD[:2])
        npt.assert_almost_equal(ax.get_ylim(), WORLD[2:])
        # Source is drawn at `extent`, at source resolution:
        for image in ax.images[:2]:
            npt.assert_almost_equal(image.get_extent(), WORLD)
        npt.assert_almost_equal(ax.images[0].get_array(), 0.25 * source,
                                decimal=6)
        npt.assert_almost_equal(ax.images[1].get_array(), source, decimal=6)
    plt.close('all')
    with pytest.raises(ValueError):
        scene.plot(view='world')
    with pytest.raises(ValueError):
        world_video().play(view='head')


@pytest.mark.parametrize('aperture', ['rectangular', 'round'])
def test_the_fov_moves_through_the_fixed_scene(aperture):
    gaze = [(0, 0), (10, -5), (35, 0)]
    scene = world_video(len(gaze), aperture=aperture)
    source = coordinate_image()
    frames = scene.play(gaze=gaze)._frame_data
    # Frames cover `extent` only, even where the FOV extends past it:
    npt.assert_equal(frames.shape, (WORLD_PX, WORLD_PX, 3, len(gaze)))
    for f, g in enumerate(gaze):
        inside = in_fov(g, aperture)
        # Outside the FOV, pixels are dimmed to 0.25:
        npt.assert_almost_equal(frames[inside, :, f], source[inside],
                                decimal=6)
        npt.assert_almost_equal(frames[~inside, :, f], 0.25 * source[~inside],
                                decimal=6)
    # `plot` clips the full-intensity layer to the FOV:
    ax = drawn_on_fresh_axes(scene, gaze=gaze, frame=1)
    npt.assert_almost_equal(clip_bounds(ax, ax.images[1]),
                            (-0.5, -15.5, 20.5, 5.5), decimal=6)
    # The dimmed context is not clipped:
    npt.assert_equal(ax.images[0].get_clip_path(), None)
    npt.assert_equal(ax.images[0].get_clip_box().bounds, ax.bbox.bounds)
    plt.close('all')


@pytest.mark.parametrize('alpha', [0, 0.25, 1])
def test_context_alpha_dims_only_the_scene_outside_the_fov(alpha):
    scene = world_video()
    source = coordinate_image()
    inside = in_fov((10, -5))
    frames = scene.play(gaze=(10, -5), context_alpha=alpha)._frame_data
    npt.assert_almost_equal(frames[~inside, :, 0], alpha * source[~inside],
                            decimal=6)
    npt.assert_almost_equal(frames[inside, :, 0], source[inside], decimal=6)
    ax = drawn_on_fresh_axes(scene, gaze=(10, -5), context_alpha=alpha)
    npt.assert_almost_equal(ax.images[0].get_array(), alpha * source,
                            decimal=6)
    plt.close('all')
    # Eye view and render ignore it:
    npt.assert_array_equal(
        scene.play(gaze=(10, -5), context_alpha=alpha,
                   view='eye')._frame_data,
        scene.render(gaze=(10, -5)).data)
    plt.close('all')


@pytest.mark.parametrize('alpha', [-0.1, 1.5, np.nan, np.inf])
def test_a_bad_context_alpha_is_refused(alpha):
    with pytest.raises(ValueError):
        world_scene().plot(context_alpha=alpha)
    with pytest.raises(ValueError):
        world_video().play(context_alpha=alpha)
    plt.close('all')
    # The eye view ignores context_alpha, so no error:
    world_scene().plot(context_alpha=alpha, view='eye')
    world_video().play(context_alpha=alpha, view='eye')
    plt.close('all')


@pytest.mark.parametrize('scotoma_fill', [0.0, 'inpaint'])
def test_plot_kwargs_reach_the_fov_in_the_scene_view(scotoma_fill):
    scene = world_scene(scotoma=Scotoma.circle(4), scotoma_fill=scotoma_fill)
    ax = drawn_on_fresh_axes(scene, gaze=(10, -5), alpha=0.5,
                             interpolation='nearest')
    context, fov = ax.images
    npt.assert_equal(fov.get_alpha(), 0.5)
    npt.assert_equal(fov.get_interpolation(), 'nearest')
    # The context layer uses `context_alpha` only:
    npt.assert_equal(context.get_alpha(), None)
    npt.assert_equal(context.get_interpolation() == 'nearest', False)
    npt.assert_almost_equal(ax.get_xlim(), WORLD[:2])
    plt.close('all')


def test_the_scotoma_moves_with_gaze_at_its_angular_size():
    white = np.ones((WORLD_PX, WORLD_PX))
    scene = world_scene(white, scotoma=Scotoma.circle(5), scotoma_fill=0.0)
    for gx, gy in ((0, 0), (10, -5)):
        ax = drawn_on_fresh_axes(scene, gaze=(gx, gy))
        lost = ax.images[1].get_array()[..., 0] < 0.5
        npt.assert_array_equal(lost, np.hypot(WORLD_X - gx, WORLD_Y - gy) <= 5)
    plt.close('all')
    video = world_video(source=white, scotoma=Scotoma.circle(5),
                        scotoma_fill=0.0)
    frames = video.play(gaze=[(0, 0), (10, -5)], context_alpha=1)._frame_data
    for f, (gx, gy) in enumerate(((0, 0), (10, -5))):
        npt.assert_array_equal(frames[..., 0, f] < 0.5,
                               (WORLD_X - gx) ** 2 + (WORLD_Y - gy) ** 2 <= 25)
    plt.close('all')


def point_percept():
    """Percept with one bright pixel at eye-centered (3, 2) dva"""
    space = Grid2D((-6, 6), (-6, 6), step=1)
    data = np.zeros(space.x.shape + (1,))
    data[(space.y == 2) & (space.x == 3)] = 1.0
    return Percept(data, space=space)


@pytest.mark.parametrize('scotoma', [None, Scotoma.circle(8)])
def test_a_percept_is_drawn_where_its_fovea_points(scotoma):
    scene = world_scene(scotoma=scotoma, scotoma_fill=0.0)
    percept = point_percept()
    ax = drawn_on_fresh_axes(scene, percept=percept, gaze=(10, -5), vmax=1)
    patch = ax.images[-1]
    # Eye-view patch shifted by gaze:
    left, right, bottom, top = patch.get_extent()
    npt.assert_almost_equal((left, right, bottom, top),
                            (3.5, 16.5, -11.5, 1.5))
    row, col = np.unravel_index(np.argmax(patch.get_array()[..., 0]),
                                patch.get_array().shape[:2])
    # Eye (3, 2) at gaze (10, -5) is scene (13, -3):
    npt.assert_almost_equal((left + col + 0.5, top - row - 0.5), (13, -3))
    npt.assert_almost_equal(clip_bounds(ax, patch), (-0.5, -15.5, 20.5, 5.5),
                            decimal=6)
    plt.close('all')
    video = world_video(scotoma=scotoma, scotoma_fill=0.0)
    frames = video.play(percept=percept, gaze=(10, -5), vmax=1)._frame_data
    # Scene (13, -3) is row 43, column 53:
    npt.assert_almost_equal(frames[43, 53, :, 0], 1.0, decimal=6)
    plt.close('all')


def test_a_percept_without_a_scotoma_is_still_shown_on_black():
    """Without a scotoma, the percept is drawn on black inside the FOV"""
    scene = world_video()
    source = coordinate_image()
    inside = in_fov((10, -5))
    ax = drawn_on_fresh_axes(scene, percept=point_percept(), gaze=(10, -5),
                             vmax=1)
    npt.assert_almost_equal(ax.images[1].get_array(), 0.0)
    npt.assert_almost_equal(ax.images[0].get_array(), 0.25 * source,
                            decimal=6)
    plt.close('all')
    frames = scene.play(percept=point_percept(), gaze=(10, -5),
                        vmax=1)._frame_data[..., 0]
    lit = frames.max(axis=-1) > 0
    npt.assert_equal(np.count_nonzero(lit[inside]), 1)
    npt.assert_almost_equal(frames[~inside], 0.25 * source[~inside],
                            decimal=6)
    plt.close('all')


def test_the_scene_view_grid_follows_gaze():
    scene = world_scene()
    ax = drawn_on_fresh_axes(scene, gaze=(10, -5), rings=[5, 8],
                             meridians=[0, 90])
    # Rings are centered on the gaze position:
    npt.assert_almost_equal(ring_radii(ax, center=(10, -5)), [5, 8],
                            decimal=6)
    ends = meridian_ends(ax)
    for start, _ in ends:
        npt.assert_almost_equal(start, (10, -5))
    # Meridians reach the edge of the shifted FOV:
    npt.assert_almost_equal([end for _, end in ends], [(20.5, -5), (10, 5.5)])
    npt.assert_almost_equal(clip_bounds(ax, ax.get_lines()[0]),
                            (-0.5, -15.5, 20.5, 5.5), decimal=6)
    plt.close('all')
    # The eye view keeps the grid at the origin:
    ax = drawn_on_fresh_axes(scene, gaze=(10, -5), rings=[5], view='eye')
    npt.assert_almost_equal(ring_radii(ax), [5], decimal=6)
    plt.close('all')
    # play() centers the grid on each frame's gaze, clipped to that FOV:
    video = world_video(source=np.ones((WORLD_PX, WORLD_PX)))
    gaze = [(0, 0), (6, -4)]
    plain = video.play(gaze=gaze)._frame_data
    ringed = video.play(gaze=gaze, rings=[5])._frame_data
    painted = np.abs(plain - ringed).max(axis=2) > 0.05
    npt.assert_equal(painted[..., 0].any(), True)
    npt.assert_array_equal(np.roll(painted[..., 0], (4, 6), axis=(0, 1)),
                           painted[..., 1])
    npt.assert_equal(painted[~in_fov((0, 0)), 0].any(), False)
    plt.close('all')


def test_play_in_the_scene_view_keeps_the_display_clock(played):
    """Scene-view play() keeps the percept's frame-end time stamps"""
    scene = video_scene(scotoma=Scotoma.circle(6), scotoma_fill=0.2)
    scene.play(percept=aligned_percept(), vmax=3, gaze=[(0, 0), (4, 1)])
    npt.assert_almost_equal(played['percept'].time, [1000, 2000])
    npt.assert_equal(played['percept'].time_unit, ms)
    npt.assert_almost_equal(outer_extent(played['percept']), scene.extent)
    plt.close('all')


def test_inpainting_cannot_see_past_the_fov_in_either_view():
    """Inpainting uses only pixels inside the FOV, in both views"""
    kwargs = dict(scotoma=Scotoma.circle(20), scotoma_fill='inpaint')
    scene, video = world_scene(**kwargs), world_video(**kwargs)
    draws = (scene.render, lambda: scene.plot(view='eye'), scene.plot,
             lambda: video.play(view='eye'), video.play)
    for draw in draws:
        with pytest.raises(ValueError):
            draw()
        plt.close('all')


def inpainted_views(scene, gaze):
    """Returns eye-view render, scene-view plot image, and played frames"""
    seen = scene.render(gaze=gaze).data[..., 0]
    ax = drawn_on_fresh_axes(scene, gaze=gaze)
    wide = ax.images[1]
    frames = scene.play(gaze=gaze)._frame_data[..., 0]
    plt.close('all')
    return seen, wide, frames


@pytest.mark.parametrize('gaze', [(10, -5), (35, 0)])
def test_the_scene_view_inpaints_what_the_eye_view_does(gaze):
    """Scene view inpaints the same eye-centered raster as the eye view"""
    scene = world_video(scotoma=Scotoma.circle(4), scotoma_fill='inpaint')
    seen, wide, frames = inpainted_views(scene, gaze)
    # `plot` draws the eye-view raster, shifted by gaze:
    npt.assert_almost_equal(wide.get_array(), seen, decimal=6)
    gx, gy = gaze
    npt.assert_almost_equal(wide.get_extent(),
                            (gx - 10.5, gx + 10.5, gy - 10.5, gy + 10.5))
    # Lattices align, so `play` shows the same pixels. Eye raster (row, col)
    # is world (30 - gy + row, 30 + gx + col):
    rows, cols = np.arange(21) + 30 - gy, np.arange(21) + 30 + gx
    keep = cols < WORLD_PX
    npt.assert_almost_equal(frames[np.ix_(rows, cols[keep])], seen[:, keep],
                            decimal=6)


def test_inpainting_uses_the_eye_raster_when_the_lattices_differ():
    """Source pixels 45 / 173 dva, render pixels 40 / 154 dva: no shared
    pixel centers"""
    y, x = np.mgrid[0:1:173j, 0:1:320j]
    frame = np.stack([x, y, 0.5 + 0.4 * np.sin(6 * x) * np.cos(4 * y)],
                     axis=-1)
    video = VideoStimulus(np.stack([frame] * 2, axis=-1), time=[0, 100])
    scene = Scene(video, extent=45 * dva, fov=40 * dva,
                  scotoma=Scotoma.circle(5), scotoma_fill='inpaint')
    npt.assert_equal(scene.render().shape[:2], (154, 154))
    gaze = (3.3, -1.7)
    seen, wide, frames = inpainted_views(scene, gaze)
    npt.assert_almost_equal(wide.get_array(), seen, decimal=6)
    # `play` shows the eye-view image, bilinearly interpolated onto source
    # pixels inside the FOV:
    xs, ys = scene._axes
    eye_x, eye_y = np.meshgrid(xs - gaze[0], ys - gaze[1])
    inside = (np.abs(eye_x) < 19.5) & (np.abs(eye_y) < 19.5)
    view_x, view_y = scene._view_axes()
    col = np.interp(eye_x[inside], view_x, np.arange(view_x.size))
    row = np.interp(-eye_y[inside], -view_y, np.arange(view_y.size))
    c0, r0 = np.floor(col).astype(int), np.floor(row).astype(int)
    fc, fr = (col - c0)[:, None], (row - r0)[:, None]
    c1, r1 = np.minimum(c0 + 1, 153), np.minimum(r0 + 1, 153)
    top = (1 - fc) * seen[r0, c0] + fc * seen[r0, c1]
    bottom = (1 - fc) * seen[r1, c0] + fc * seen[r1, c1]
    expected = (1 - fr) * top + fr * bottom
    npt.assert_allclose(frames[inside], expected, atol=1e-5)
