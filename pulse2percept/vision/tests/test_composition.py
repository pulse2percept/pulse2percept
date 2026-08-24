"""Composing native and prosthetic vision (#668)

The scenes here are laid out one degree per pixel with an odd pixel count, so
that pixel centers land on whole degrees and the center pixel sits on the
origin. Expected values are then plain arithmetic rather than a restatement of
the pixel-center convention Phase 1 already owns.
"""
import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.percepts import Percept
from pulse2percept.stimuli import ImageStimulus, VideoStimulus
from pulse2percept.topography import Grid2D
from pulse2percept.units import dva, ms, s
from pulse2percept.vision import Scotoma, compose_amd

SCENE_PX = 21
HALF = (SCENE_PX - 1) // 2


def scene_rgb(n_frames=None, seed=0):
    """An RGB scene, every pixel a different color, one degree per pixel"""
    rng = np.random.default_rng(seed)
    shape = (SCENE_PX, SCENE_PX, 3)
    if n_frames is None:
        return ImageStimulus(rng.random(shape), fov=(SCENE_PX, SCENE_PX))
    return VideoStimulus(rng.random((*shape, n_frames)),
                         fov=(SCENE_PX, SCENE_PX),
                         time=np.arange(n_frames) * 10.0)


def pixel_of(x_dva, y_dva=0.0):
    """The (row, col) of the scene pixel centered on a visual-field point"""
    return int(round(HALF - y_dva)), int(round(x_dva + HALF))


def percept_on(values, x_range=(-4, 4), y_range=(-4, 4), step=2, time=None,
               time_unit=ms):
    """A brightness percept on a known dva grid

    ``values`` is indexed the way percept data is: row 0 is the largest y.
    """
    grid = Grid2D(x_range, y_range, step=step)
    values = np.asarray(values, dtype=float)
    if values.ndim == 2:
        values = values[..., np.newaxis]
    npt.assert_equal(values.shape[:2], grid.shape)
    return Percept(values, space=grid, time=time, time_unit=time_unit)


def uniform_percept(brightness, **kwargs):
    """A percept of one brightness everywhere on its grid"""
    grid_shape = Grid2D(kwargs.get('x_range', (-4, 4)),
                        kwargs.get('y_range', (-4, 4)),
                        step=kwargs.get('step', 2)).shape
    return percept_on(np.full(grid_shape, float(brightness)), **kwargs)


def test_intact_periphery_is_the_scene_exactly():
    """Outside the scotoma nothing is resampled, blended or rounded"""
    scene = scene_rgb()
    combined = compose_amd(scene, uniform_percept(20), Scotoma.circle(3),
                           vmax=20)
    source = scene.data.reshape((SCENE_PX, SCENE_PX, 3))
    # A corner is far outside both the scotoma and the model grid:
    npt.assert_array_equal(combined.data[0, 0, :, 0], source[0, 0])
    # Every pixel the scotoma does not touch, to the last bit:
    x, y = np.meshgrid(np.arange(-HALF, HALF + 1), np.arange(HALF, -HALF - 1,
                                                             -1))
    intact = Scotoma.circle(3)(x, y) == 0
    npt.assert_array_equal(combined.data[..., 0][intact],
                           source[intact])
    # The output lives on the scene's grid, in scene coordinates:
    npt.assert_equal(combined.shape, (SCENE_PX, SCENE_PX, 3, 1))
    npt.assert_almost_equal(combined.xdva, np.arange(-HALF, HALF + 1),
                            decimal=4)
    npt.assert_almost_equal(combined.ydva, np.arange(-HALF, HALF + 1),
                            decimal=4)


@pytest.mark.parametrize('fill', [0.0, 0.35, 1.0])
def test_complete_loss_without_a_percept_is_the_fill(fill):
    combined = compose_amd(scene_rgb(), uniform_percept(0), Scotoma.circle(3),
                           vmax=20, scotoma_fill=fill)
    npt.assert_almost_equal(combined.data[pixel_of(0)][:, 0], [fill] * 3,
                            decimal=6)


@pytest.mark.parametrize('brightness, expected',
                         [(0, 0.0), (5, 0.25), (20, 1.0), (30, 1.0),
                          (-5, 0.0)])
def test_brightness_maps_onto_the_display_range(brightness, expected):
    """vmin -> 0, vmax -> 1, clipped outside"""
    combined = compose_amd(scene_rgb(), uniform_percept(brightness),
                           Scotoma.circle(3), vmax=20)
    npt.assert_almost_equal(combined.data[pixel_of(0)][:, 0],
                            [expected] * 3, decimal=5)


def test_vmin_shifts_the_display_range():
    combined = compose_amd(scene_rgb(), uniform_percept(15),
                           Scotoma.circle(3), vmin=10, vmax=20)
    npt.assert_almost_equal(combined.data[pixel_of(0)][:, 0], [0.5] * 3,
                            decimal=5)


def test_the_gap_between_percept_and_scotoma_shows_the_fill():
    """Inside the scotoma but outside the implant's reach is not native vision

    The architectural question from #668: a scotoma wider than the percept
    leaves a ring that is neither intact nor stimulated.
    """
    scotoma = Scotoma.circle(8)
    combined = compose_amd(scene_rgb(), uniform_percept(20), scotoma, vmax=20,
                           scotoma_fill=0.2)
    # Inside the model grid (which reaches 4 dva) the phosphene shows:
    npt.assert_almost_equal(combined.data[pixel_of(0)][:, 0], [1.0] * 3,
                            decimal=5)
    # Between the edge of the grid and the edge of the scotoma, the fill:
    npt.assert_almost_equal(combined.data[pixel_of(6)][:, 0], [0.2] * 3,
                            decimal=5)
    # And outside the scotoma, the scene again:
    source = scene_rgb().data.reshape((SCENE_PX, SCENE_PX, 3))
    npt.assert_array_equal(combined.data[pixel_of(9)][:, 0],
                           source[pixel_of(9)])


def test_the_percept_need_not_be_centered_on_the_scotoma():
    """An offset model grid lands where its own coordinates say"""
    percept = uniform_percept(20, x_range=(4, 8), y_range=(-2, 2))
    combined = compose_amd(scene_rgb(), percept, Scotoma.circle(10), vmax=20,
                           scotoma_fill=0.0)
    npt.assert_almost_equal(combined.data[pixel_of(6)][:, 0], [1.0] * 3,
                            decimal=5)
    # The fovea is inside the scotoma but outside this percept:
    npt.assert_almost_equal(combined.data[pixel_of(0)][:, 0], [0.0] * 3,
                            decimal=6)
    # ... and so is the mirror-image position:
    npt.assert_almost_equal(combined.data[pixel_of(-6)][:, 0], [0.0] * 3,
                            decimal=6)


def test_y_orientation_survives_the_grid():
    """A percept bright at the top of its grid is bright above fixation

    `Grid2D` meshes its y axis reversed, so percept row 0 holds the largest y
    while `ydva` ascends. Getting that backwards flips the world upside down
    and nothing else notices.
    """
    values = np.zeros((5, 5))
    values[0] = 20.0  # row 0 == largest y == above fixation
    percept = percept_on(values)
    combined = compose_amd(scene_rgb(), percept, Scotoma.circle(10), vmax=20,
                           scotoma_fill=0.0)
    npt.assert_almost_equal(combined.data[pixel_of(0, 4)][:, 0], [1.0] * 3,
                            decimal=5)
    npt.assert_almost_equal(combined.data[pixel_of(0, -4)][:, 0], [0.0] * 3,
                            decimal=6)


def test_gaze_moves_scotoma_and_percept_together():
    """Gaze moves the scene past the eye, keeping the two in step"""
    percept = uniform_percept(20, x_range=(-2, 2), y_range=(-2, 2))
    scotoma = Scotoma.circle(4)
    fixating = compose_amd(scene_rgb(), percept, scotoma, vmax=20,
                           scotoma_fill=0.3)
    shifted = compose_amd(scene_rgb(), percept, scotoma, vmax=20,
                          scotoma_fill=0.3, gaze=(5, 0) * dva)
    # Everything the eye-centered pair paints has moved 5 degrees right:
    npt.assert_almost_equal(shifted.data[pixel_of(5)][:, 0],
                            fixating.data[pixel_of(0)][:, 0], decimal=6)
    npt.assert_almost_equal(shifted.data[pixel_of(8)][:, 0],
                            fixating.data[pixel_of(3)][:, 0], decimal=6)
    # The phosphene still sits in the middle of the scotoma, not beside it:
    npt.assert_almost_equal(shifted.data[pixel_of(5)][:, 0], [1.0] * 3,
                            decimal=5)
    npt.assert_almost_equal(shifted.data[pixel_of(8)][:, 0], [0.3] * 3,
                            decimal=5)
    # ... and where the eye used to be looking is native vision again:
    source = scene_rgb().data.reshape((SCENE_PX, SCENE_PX, 3))
    npt.assert_array_equal(shifted.data[pixel_of(0)][:, 0],
                           source[pixel_of(0)])


def test_a_graded_scotoma_mixes_linearly():
    scene = scene_rgb()
    half = Scotoma(lambda x, y: np.full(np.shape(x), 0.5))
    combined = compose_amd(scene, uniform_percept(10), half, vmax=20)
    source = scene.data.reshape((SCENE_PX, SCENE_PX, 3))
    # lost = max(0, 0.5) = 0.5, mixed half and half with the scene:
    npt.assert_almost_equal(combined.data[pixel_of(0)][:, 0],
                            0.5 * source[pixel_of(0)] + 0.5 * 0.5, decimal=6)


def test_a_grayscale_scene_becomes_rgb_without_changing_intensity():
    gray = np.linspace(0, 1, SCENE_PX ** 2).reshape((SCENE_PX, SCENE_PX))
    scene = ImageStimulus(gray, fov=(SCENE_PX, SCENE_PX))
    combined = compose_amd(scene, uniform_percept(0), Scotoma.circle(1),
                           vmax=20)
    npt.assert_equal(combined.is_rgb, True)
    row, col = pixel_of(8, 8)
    npt.assert_almost_equal(combined.data[row, col, :, 0],
                            [gray[row, col]] * 3, decimal=6)


def test_an_image_scene_takes_the_percepts_timing():
    """A still scene has no clock, so the percept sets one"""
    values = np.stack([np.full((5, 5), b) for b in (0.0, 10.0, 20.0)],
                      axis=-1)
    percept = percept_on(values, time=[0, 25, 50])
    scene = scene_rgb()
    combined = compose_amd(scene, percept, Scotoma.circle(10), vmax=20,
                           scotoma_fill=0.0)
    npt.assert_equal(combined.shape, (SCENE_PX, SCENE_PX, 3, 3))
    npt.assert_almost_equal(combined.time, [0, 25, 50])
    npt.assert_almost_equal(combined.data[pixel_of(0)][0], [0.0, 0.5, 1.0],
                            decimal=5)
    # The scene is repeated behind every frame rather than resampled:
    source = scene.data.reshape((SCENE_PX, SCENE_PX, 3))
    for f in range(3):
        npt.assert_array_equal(combined.data[0, 0, :, f], source[0, 0])


def test_a_video_keeps_its_own_timing():
    """Native frames stay native, and the percept is read at their times"""
    scene = scene_rgb(n_frames=3)
    values = np.stack([np.full((5, 5), b) for b in (0.0, 20.0)], axis=-1)
    percept = percept_on(values, time=[0, 20])
    combined = compose_amd(scene, percept, Scotoma.circle(10), vmax=20,
                           scotoma_fill=0.0)
    npt.assert_equal(combined.shape, (SCENE_PX, SCENE_PX, 3, 3))
    npt.assert_almost_equal(combined.time, [0, 10, 20])
    # The percept is interpolated onto the video's clock, not the other way:
    npt.assert_almost_equal(combined.data[pixel_of(0)][0], [0.0, 0.5, 1.0],
                            decimal=5)
    # Outside the scotoma every video frame passes through untouched:
    source = scene.data.reshape((SCENE_PX, SCENE_PX, 3, 3))
    npt.assert_array_equal(combined.data[0, 0], source[0, 0])


def test_a_still_percept_broadcasts_across_a_video():
    scene = scene_rgb(n_frames=3)
    combined = compose_amd(scene, uniform_percept(20), Scotoma.circle(10),
                           vmax=20)
    npt.assert_equal(combined.shape[-1], 3)
    npt.assert_almost_equal(combined.time, [0, 10, 20])
    npt.assert_almost_equal(combined.data[pixel_of(0)],
                            np.ones((3, 3)), decimal=5)


def test_per_frame_gaze_moves_the_eye_between_frames():
    scene = scene_rgb(n_frames=3)
    percept = uniform_percept(20, x_range=(-2, 2), y_range=(-2, 2))
    gaze = np.array([[-6.0, 0.0], [0.0, 0.0], [6.0, 0.0]])
    combined = compose_amd(scene, percept, Scotoma.circle(3), vmax=20,
                           gaze=gaze * dva, scotoma_fill=0.0)
    for f, gx in enumerate(gaze[:, 0]):
        npt.assert_almost_equal(combined.data[pixel_of(gx)][:, f], [1.0] * 3,
                                decimal=5)
    with pytest.raises(ValueError):
        compose_amd(scene, percept, Scotoma.circle(3), vmax=20,
                    gaze=np.zeros((2, 2)))


def test_compose_amd_does_not_touch_its_inputs():
    scene = scene_rgb()
    before = scene.data.copy()
    percept = uniform_percept(20)
    percept_before = percept.data.copy()
    compose_amd(scene, percept, Scotoma.circle(3), vmax=20)
    npt.assert_array_equal(scene.data, before)
    npt.assert_array_equal(percept.data, percept_before)


def test_compose_amd_rejects_inputs_it_cannot_place():
    scene = scene_rgb()
    percept = uniform_percept(20)
    scotoma = Scotoma.circle(3)
    # A picture with no visual geometry:
    with pytest.raises(ValueError):
        compose_amd(ImageStimulus(np.zeros((5, 5, 3))), percept, scotoma,
                    vmax=20)
    # A percept with nowhere to be:
    with pytest.raises(ValueError):
        compose_amd(scene, Percept(np.zeros((1, 1, 3)), time=[0, 10, 20]),
                    scotoma, vmax=20)
    # An RGB percept, which no model produces yet:
    with pytest.raises(ValueError):
        compose_amd(scene, Percept(np.zeros((5, 5, 3, 1)),
                                   space=Grid2D((-4, 4), (-4, 4), step=2)),
                    scotoma, vmax=20)
    # Wrong types:
    with pytest.raises(TypeError):
        compose_amd(np.zeros((5, 5, 3)), percept, scotoma, vmax=20)
    with pytest.raises(TypeError):
        compose_amd(scene, np.zeros((5, 5, 1)), scotoma, vmax=20)
    with pytest.raises(TypeError):
        compose_amd(scene, percept, 'circle', vmax=20)


@pytest.mark.parametrize('kwargs', [{'vmax': 0}, {'vmax': -1},
                                    {'vmax': np.inf}, {'vmax': np.nan},
                                    {'vmax': 20, 'vmin': 20},
                                    {'vmax': 20, 'vmin': 25},
                                    {'vmax': 20, 'vmin': np.nan},
                                    {'vmax': 20, 'scotoma_fill': 1.5},
                                    {'vmax': 20, 'scotoma_fill': -0.1},
                                    {'vmax': 20, 'scotoma_fill': np.nan}])
def test_compose_amd_rejects_a_mapping_it_cannot_display(kwargs):
    with pytest.raises(ValueError):
        compose_amd(scene_rgb(), uniform_percept(20), Scotoma.circle(3),
                    **kwargs)


def test_compose_amd_needs_a_percept_with_extent():
    """A grid with no width or height cannot be interpolated onto a scene"""
    strip = Percept(np.zeros((1, 5, 1)),
                    space=Grid2D((-4, 4), (0, 0), step=2))
    with pytest.raises(ValueError):
        compose_amd(scene_rgb(), strip, Scotoma.circle(3), vmax=20)


def test_frames_always_arrive_with_a_clock():
    """Neither input can have frames and no time axis

    Both number their frames when nothing better is given, which is why
    composition never has to guess when they happen.
    """
    npt.assert_almost_equal(percept_on(np.zeros((5, 5, 2))).time, [0, 1])
    npt.assert_almost_equal(
        VideoStimulus(np.zeros((4, 4, 3, 2)), fov=(4, 4)).time, [0, 1])


def test_unitful_arguments_are_accepted():
    plain = compose_amd(scene_rgb(), uniform_percept(20), Scotoma.circle(3),
                        vmax=20, gaze=(2, -1))
    unitful = compose_amd(scene_rgb(), uniform_percept(20),
                          Scotoma.circle(3 * dva), vmax=20,
                          gaze=(2, -1) * dva)
    npt.assert_array_equal(plain.data, unitful.data)


def test_output_percept_plays_and_plots():
    """The result is an ordinary RGB percept, with everything that implies"""
    scene = scene_rgb(n_frames=3)
    combined = compose_amd(scene, uniform_percept(20), Scotoma.circle(3),
                           vmax=20)
    npt.assert_equal(combined.data.min() >= 0, True)
    npt.assert_equal(combined.data.max() <= 1, True)
    npt.assert_equal(combined[..., 5 * ms].shape, (SCENE_PX, SCENE_PX, 3))
    ani = combined.play()
    npt.assert_equal(ani._frame_data.shape, (SCENE_PX, SCENE_PX, 3, 3))


def test_a_percept_without_a_space_is_not_a_place():
    """Pixel indices are not degrees, however much they look like them

    `Percept` fills an omitted spatial axis with 0, 1, 2, ...; composing those
    as visual-field coordinates puts the percept somewhere plausible-looking
    and wrong, which is the one failure that would not announce itself.
    """
    nowhere = Percept(np.zeros((5, 5, 1)))
    npt.assert_array_equal(nowhere.xdva, [0, 1, 2, 3, 4])
    npt.assert_equal(nowhere._has_space, False)
    with pytest.raises(ValueError):
        compose_amd(scene_rgb(), nowhere, Scotoma.circle(3), vmax=20)
    # The same data, told where it is, composes fine:
    somewhere = Percept(np.zeros((5, 5, 1)),
                        space=Grid2D((-4, 4), (-4, 4), step=2))
    npt.assert_equal(somewhere._has_space, True)
    compose_amd(scene_rgb(), somewhere, Scotoma.circle(3), vmax=20)


def test_the_output_keeps_the_clock_it_was_given():
    """A percept counted in seconds does not come back in milliseconds"""
    values = np.stack([np.full((5, 5), b) for b in (0.0, 20.0)], axis=-1)
    percept = percept_on(values, time=[0, 0.05], time_unit=s)
    combined = compose_amd(scene_rgb(), percept, Scotoma.circle(10), vmax=20)
    npt.assert_equal(combined.time_unit, s)
    npt.assert_almost_equal(combined.time, [0, 0.05])
    npt.assert_almost_equal(combined.times(ms), [0, 50])
    # A video owns the clock instead, and its unit comes along the same way:
    video = scene_rgb(n_frames=3)
    combined = compose_amd(video, percept, Scotoma.circle(10), vmax=20)
    npt.assert_equal(combined.time_unit, video.time_unit)
    npt.assert_almost_equal(combined.time, video.time)
