"""Tests for BinocularScene (#668)

Test scenes use 1 dva per pixel and an odd pixel count, as in `test_scene`,
so pixel centers land on whole degrees.
"""
import numpy as np
import numpy.testing as npt
import pytest
import matplotlib.pyplot as plt

from pulse2percept.percepts import Percept
from pulse2percept.stimuli import ImageStimulus, VideoStimulus
from pulse2percept.units import dva
from pulse2percept.vision import BinocularScene, Scene, Scotoma

SCENE_PX = 21
HALF = (SCENE_PX - 1) // 2


def flat_scene(level=0.5, px=SCENE_PX, **kwargs):
    """Uniform gray scene"""
    data = np.full((px, px), float(level))
    kwargs.setdefault('scotoma_blend', 0)
    return Scene(ImageStimulus(data), fov=(px, px), **kwargs)


def spot_percept(scene, x_dva=0.0, y_dva=0.0, brightness=10.0):
    """Percept with one bright pixel at an eye-centered location (dva)"""
    data = np.zeros((SCENE_PX, SCENE_PX, 1))
    col, row = scene.dva_to_pixel(x_dva, y_dva)
    data[int(round(float(row))), int(round(float(col)))] = brightness
    return Percept(data, space=scene._grid())


def clipped_away(ax, point_dva):
    """Returns True if the last image's clip path excludes ``point_dva``"""
    clip = ax.images[-1].get_clip_path()
    if clip is None:
        return False
    return not clip.get_fully_transformed_path().contains_point(
        ax.transData.transform(point_dva))


def test_the_two_eyes_are_stored_as_given_and_not_swapped():
    left, right = flat_scene(0.2), flat_scene(0.8)
    binocular = BinocularScene(left=left, right=right)
    npt.assert_equal(binocular.left is left, True)
    npt.assert_equal(binocular.right is right, True)
    # Positional order is left, right:
    npt.assert_equal(BinocularScene(left, right).left is left, True)
    npt.assert_equal('left' in repr(binocular), True)
    npt.assert_equal('right' in repr(binocular), True)


@pytest.mark.parametrize('bad', [None, 0.5, np.zeros((4, 4)),
                                 ImageStimulus(np.zeros((4, 4)))])
def test_only_scenes_can_be_an_eye(bad):
    scene = flat_scene()
    with pytest.raises(TypeError):
        BinocularScene(left=bad, right=scene)
    with pytest.raises(TypeError):
        BinocularScene(left=scene, right=bad)


def test_the_two_eyes_are_independent_channels():
    """Eyes may differ in source, FOV, shape, scotoma, and aperture"""
    left = Scene(ImageStimulus(np.full((9, 9), 0.3)), fov=(30, 30),
                 scotoma=Scotoma.circle(6), scotoma_fill=0.0,
                 aperture='round')
    right = Scene(VideoStimulus(np.zeros((15, 21, 2)), time=[0, 10]),
                  fov=(42, 30), background=1)
    binocular = BinocularScene(left=left, right=right)
    npt.assert_equal(binocular.left.fov, (30.0, 30.0))
    npt.assert_equal(binocular.right.fov, (42.0, 30.0))
    npt.assert_equal(binocular.left.shape, (9, 9))
    npt.assert_equal(binocular.right.shape, (15, 21))
    npt.assert_equal(binocular.left.aperture, 'round')
    npt.assert_equal(binocular.right.aperture, 'rectangular')
    npt.assert_equal(binocular.right.scotoma, None)


def test_plot_draws_the_two_eyes_in_order_on_their_own_axes():
    binocular = BinocularScene(left=flat_scene(0.2), right=flat_scene(0.8))
    ax_left, ax_right = binocular.plot()
    npt.assert_equal(ax_left is not ax_right, True)
    npt.assert_equal(ax_left.get_title(), 'left eye')
    npt.assert_equal(ax_right.get_title(), 'right eye')
    npt.assert_almost_equal(ax_left.images[-1].get_array()[HALF, HALF],
                            [0.2] * 3, decimal=6)
    npt.assert_almost_equal(ax_right.images[-1].get_array()[HALF, HALF],
                            [0.8] * 3, decimal=6)
    plt.close('all')
    # Supplied axes are used, in the same order:
    _, axes = plt.subplots(1, 2)
    npt.assert_equal(binocular.plot(axes=axes), tuple(axes))
    plt.close('all')
    with pytest.raises(ValueError):
        binocular.plot(axes=plt.subplots(1, 3)[1])
    plt.close('all')


def test_a_percept_can_be_shown_in_one_eye_only():
    """The eye without a percept shows its scene, on either side"""
    left, right = flat_scene(0.2), flat_scene(0.8)
    binocular = BinocularScene(left=left, right=right)
    percept = spot_percept(left, x_dva=-4.0)

    ax_left, ax_right = binocular.plot(left_percept=percept, vmax=10)
    # Implanted eye: phosphene on black; fellow eye: its scene:
    npt.assert_almost_equal(ax_left.images[-1].get_array()[HALF, HALF - 4],
                            [1.0] * 3, decimal=6)
    npt.assert_almost_equal(ax_left.images[-1].get_array()[HALF, HALF], 0.0)
    npt.assert_almost_equal(ax_right.images[-1].get_array()[HALF, HALF],
                            [0.8] * 3, decimal=6)
    plt.close('all')

    ax_left, ax_right = binocular.plot(right_percept=percept, vmax=10)
    npt.assert_almost_equal(ax_left.images[-1].get_array()[HALF, HALF],
                            [0.2] * 3, decimal=6)
    npt.assert_almost_equal(ax_right.images[-1].get_array()[HALF, HALF - 4],
                            [1.0] * 3, decimal=6)
    plt.close('all')


def test_two_percepts_stay_two_percepts():
    """Percepts in the two eyes are not combined"""
    left, right = flat_scene(0.0), flat_scene(0.0)
    binocular = BinocularScene(left=left, right=right)
    on_the_left = spot_percept(left, x_dva=-6.0)
    on_the_right = spot_percept(right, x_dva=6.0)
    ax_left, ax_right = binocular.plot(left_percept=on_the_left,
                                       right_percept=on_the_right, vmax=10)
    seen_left = ax_left.images[-1].get_array()
    seen_right = ax_right.images[-1].get_array()
    # Each eye shows only its own phosphene:
    npt.assert_almost_equal(seen_left[HALF, HALF - 6], [1.0] * 3, decimal=6)
    npt.assert_almost_equal(seen_left[HALF, HALF + 6], 0.0)
    npt.assert_almost_equal(seen_right[HALF, HALF + 6], [1.0] * 3, decimal=6)
    npt.assert_almost_equal(seen_right[HALF, HALF - 6], 0.0)
    # Each panel matches Scene.plot for that eye:
    npt.assert_almost_equal(seen_left,
                            left.plot(percept=on_the_left,
                                      vmax=10).images[-1].get_array(),
                            decimal=6)
    plt.close('all')


def test_an_omitted_vmax_is_shared_by_both_eyes():
    """Default vmax is the maximum over both percepts"""
    left, right = flat_scene(0.0), flat_scene(0.0)
    binocular = BinocularScene(left=left, right=right)
    dim = spot_percept(left, x_dva=-6.0, brightness=5.0)
    bright = spot_percept(right, x_dva=6.0, brightness=10.0)
    ax_left, ax_right = binocular.plot(left_percept=dim, right_percept=bright)
    npt.assert_almost_equal(ax_left.images[-1].get_array()[HALF, HALF - 6],
                            [0.5] * 3, decimal=6)
    npt.assert_almost_equal(ax_right.images[-1].get_array()[HALF, HALF + 6],
                            [1.0] * 3, decimal=6)
    plt.close('all')
    # Blank percepts in both eyes are drawn black:
    blank = spot_percept(left, brightness=0.0)
    ax_left, ax_right = binocular.plot(left_percept=blank, right_percept=blank)
    npt.assert_almost_equal(ax_left.images[-1].get_array(), 0.0)
    npt.assert_almost_equal(ax_right.images[-1].get_array(), 0.0)
    plt.close('all')
    # vmin is compared against the shared maximum:
    for peak in (5.0, 10.0):
        spot = spot_percept(left, brightness=peak)
        with pytest.raises(ValueError):
            binocular.plot(left_percept=spot, right_percept=spot, vmin=10)
        plt.close('all')


def test_each_eye_keeps_its_own_aperture():
    """Each panel is clipped to its own aperture"""
    left = flat_scene(0.6, aperture='round')
    right = flat_scene(0.6)
    ax_left, ax_right = BinocularScene(left=left, right=right).plot()
    # The corner is outside the round aperture, so it is clipped (not black):
    npt.assert_equal([clipped_away(ax, (-HALF, HALF))
                      for ax in (ax_left, ax_right)], [True, False])
    for ax in (ax_left, ax_right):
        npt.assert_almost_equal(ax.images[-1].get_array()[0, 0], [0.6] * 3,
                                decimal=6)
    # render() sets it to black:
    npt.assert_almost_equal(left.render().data[0, 0, :, 0], 0.0)
    npt.assert_almost_equal(right.render().data[0, 0, :, 0], [0.6] * 3,
                            decimal=6)
    plt.close('all')


def test_two_fovs_are_drawn_on_one_angular_scale():
    """Eye view shares limits: max width and max height over both FOVs

    One FOV is wide, the other tall, so swapping width and height would give
    different limits.
    """
    left = Scene(ImageStimulus(np.full((21, 31), 0.4)), fov=(31, 21))
    right = Scene(ImageStimulus(np.full((51, 41), 0.4)), fov=(41, 51))
    ax_left, ax_right = BinocularScene(left=left, right=right).plot(
        view='eye')
    npt.assert_almost_equal(ax_left.get_xlim(), ax_right.get_xlim())
    npt.assert_almost_equal(ax_left.get_ylim(), ax_right.get_ylim())
    # Limits are outer FOV edges (not pixel centers), per dimension:
    npt.assert_almost_equal(ax_left.get_xlim(), (-20.5, 20.5))   # max width
    npt.assert_almost_equal(ax_left.get_ylim(), (-25.5, 25.5))   # max height
    # Each image keeps its own extent:
    npt.assert_almost_equal(ax_left.images[-1].get_extent(),
                            (-15.5, 15.5, -10.5, 10.5))
    npt.assert_almost_equal(ax_right.images[-1].get_extent(),
                            (-20.5, 20.5, -25.5, 25.5))
    plt.close('all')


def test_the_scene_view_shares_the_union_of_both_extents():
    """Scene view shares limits: the union of both extents"""
    source = np.full((21, 21), 0.4)
    left = Scene(ImageStimulus(source), fov=11, extent=(-30, 10, -5, 15))
    right = Scene(ImageStimulus(source), fov=11, extent=(-10, 20, -25, 5))
    binocular = BinocularScene(left=left, right=right)
    for ax in binocular.plot(gaze=(3, -2) * dva):
        npt.assert_almost_equal(ax.get_xlim(), (-30, 20))
        npt.assert_almost_equal(ax.get_ylim(), (-25, 15))
    plt.close('all')
    # The eye view keeps the shared FOV:
    for ax in binocular.plot(gaze=(3, -2) * dva, view='eye'):
        npt.assert_almost_equal(ax.get_xlim(), (-5.5, 5.5))
        npt.assert_almost_equal(ax.get_ylim(), (-5.5, 5.5))
    plt.close('all')
    with pytest.raises(ValueError):
        binocular.plot(view='fused')


def test_context_alpha_reaches_both_eyes():
    source = np.full((21, 21), 0.8)
    binocular = BinocularScene(
        left=Scene(ImageStimulus(source), fov=11, extent=21),
        right=Scene(ImageStimulus(source), fov=11, extent=21))
    for alpha in (0, 0.5):
        for ax in binocular.plot(context_alpha=alpha):
            npt.assert_almost_equal(ax.images[0].get_array(), alpha * 0.8,
                                    decimal=6)
        plt.close('all')


def test_plot_lays_out_only_the_figure_it_made(monkeypatch):
    binocular = BinocularScene(left=flat_scene(), right=flat_scene())
    laid_out = []
    monkeypatch.setattr(plt.Figure, 'tight_layout',
                        lambda self, *a, **kw: laid_out.append(self))
    binocular.plot()
    npt.assert_equal(len(laid_out), 1)
    plt.close('all')
    # A user-supplied figure is not laid out:
    _, axes = plt.subplots(1, 2)
    binocular.plot(axes=axes)
    npt.assert_equal(len(laid_out), 1)
    plt.close('all')


def test_a_shared_gaze_moves_both_eyes_together():
    scotoma = Scotoma.circle(4)
    left = flat_scene(0.7, scotoma=scotoma, scotoma_fill=0.0)
    right = flat_scene(0.7, scotoma=scotoma, scotoma_fill=0.0)
    axes = BinocularScene(left=left, right=right).plot(gaze=(6, 0) * dva,
                                                       view='eye')
    for ax, scene in zip(axes, (left, right)):
        npt.assert_almost_equal(ax.images[-1].get_array(),
                                scene._native_rgb(gaze=(6, 0))[..., 0],
                                decimal=6)
    plt.close('all')


def test_the_grid_reaches_both_eyes_about_each_fovea():
    binocular = BinocularScene(left=flat_scene(), right=flat_scene())
    axes = binocular.plot(gaze=(2, 1) * dva, rings=[4], meridians=[0, 90],
                          grid_color='red', view='eye')
    for ax in axes:
        lines = ax.get_lines()
        npt.assert_equal([line.get_linestyle() for line in lines],
                         ['--', '-', '-'])
        # Eye-centered, so independent of gaze:
        ring = np.asarray(lines[0].get_data())
        npt.assert_almost_equal(np.hypot(ring[0], ring[1]), 4)
        for line in lines[1:]:
            npt.assert_almost_equal(np.asarray(line.get_data())[:, 0], (0, 0))
        npt.assert_equal({line.get_color() for line in lines}, {'red'})
    plt.close('all')


def test_it_has_no_implant_or_model_behavior():
    """BinocularScene has no implant, model, or fusion methods"""
    binocular = BinocularScene(left=flat_scene(), right=flat_scene())
    for absent in ('eye', 'implant', 'predict_percept', 'play', 'fuse',
                   '_compose', '_device_input', '_sample_at'):
        npt.assert_equal(hasattr(binocular, absent), False)
    # Scene has no eye attribute; BinocularScene stores which eye is which:
    npt.assert_equal(hasattr(binocular.left, 'eye'), False)


def packed_stereo(rows=10, half_cols=20, channels=None):
    """Side-by-side frame: left half 0, right half 1"""
    shape = (rows, 2 * half_cols) + (() if channels is None else (channels,))
    stereo = np.zeros(shape, dtype=np.float32)
    stereo[:, half_cols:] = 1.0
    return stereo


def packed_ramp(rows=10, half_cols=20):
    """Side-by-side ramp with all pixel values distinct"""
    n_px = rows * 2 * half_cols
    return (np.arange(n_px, dtype=np.float32) / n_px).reshape(rows, -1)


def test_side_by_side_splits_left_from_right():
    stereo = packed_ramp()
    binocular = BinocularScene.from_side_by_side(stereo, fov=(40, 20) * dva)
    for eye, half in ((binocular.left, stereo[:, :20]),
                      (binocular.right, stereo[:, 20:])):
        npt.assert_equal(eye.shape, (10, 20))
        # Exact match, so any flip or resampling fails:
        npt.assert_array_equal(eye.source.data.reshape(eye.shape), half)


@pytest.mark.parametrize('channels', [3, 4])
def test_side_by_side_splits_columns_not_channels(channels):
    stereo = packed_stereo(rows=6, half_cols=4, channels=channels)
    stereo[:, 4:, 1:] = 0.25
    binocular = BinocularScene.from_side_by_side(stereo, fov=(40, 20) * dva)
    for eye in (binocular.left, binocular.right):
        npt.assert_equal(eye.source.img_shape, (6, 4, channels))
        npt.assert_equal(eye.shape, (6, 4))
    # All channels are kept, including alpha:
    right = binocular.right.source.data.reshape(6, 4, channels)
    npt.assert_array_equal(right[..., 0], np.ones((6, 4), dtype=np.float32))
    npt.assert_array_equal(right[..., 1:], np.full((6, 4, channels - 1), 0.25,
                                                   dtype=np.float32))
    npt.assert_array_equal(binocular.left.source.data.reshape(6, 4, channels),
                           np.zeros((6, 4, channels), dtype=np.float32))


@pytest.mark.parametrize('n_cols', [21, 7])
def test_an_odd_width_has_no_seam(n_cols):
    stereo = np.zeros((10, n_cols))
    with pytest.raises(ValueError) as excinfo:
        BinocularScene.from_side_by_side(stereo, fov=40 * dva)
    npt.assert_equal('even' in str(excinfo.value), True)


def test_a_scalar_fov_describes_one_eye_not_the_packed_frame():
    """Extent is inferred per eye, not from the packed frame"""
    binocular = BinocularScene.from_side_by_side(packed_stereo(), fov=40)
    for eye in (binocular.left, binocular.right):
        npt.assert_equal(eye.shape, (10, 20))
        # 40 dva over 10 rows, so 80 dva over 20 columns:
        npt.assert_almost_equal(eye.fov, (40.0, 40.0))
        npt.assert_almost_equal(eye.extent, (-40.0, 40.0, -20.0, 20.0))


def test_an_explicit_fov_pair_goes_to_both_eyes():
    binocular = BinocularScene.from_side_by_side(packed_stereo(),
                                                 fov=(60, 40) * dva)
    npt.assert_almost_equal(binocular.left.fov, (60.0, 40.0))
    npt.assert_almost_equal(binocular.right.fov, (60.0, 40.0))


def test_scene_kwargs_reach_both_eyes():
    binocular = BinocularScene.from_side_by_side(packed_stereo(), fov=40 * dva,
                                                 aperture='round',
                                                 background=0.5,
                                                 scotoma=Scotoma.circle(8))
    for eye in (binocular.left, binocular.right):
        npt.assert_equal(eye.aperture, 'round')
        npt.assert_almost_equal(eye.background, (0.5, 0.5, 0.5))
        npt.assert_equal(eye.scotoma is not None, True)


def test_an_image_stimulus_is_a_stereo_source():
    stereo = ImageStimulus(packed_stereo())
    binocular = BinocularScene.from_side_by_side(stereo, fov=40 * dva)
    npt.assert_equal(binocular.left.shape, (10, 20))
    npt.assert_almost_equal(binocular.left.source.data.max(), 0.0)
    npt.assert_almost_equal(binocular.right.source.data.min(), 1.0)


def test_both_halves_inherit_the_packed_frame_metadata():
    stereo = ImageStimulus(packed_stereo(), metadata={'foo': 'bar'})
    binocular = BinocularScene.from_side_by_side(stereo, fov=40 * dva)
    for eye in (binocular.left, binocular.right):
        npt.assert_equal(eye.source.metadata['foo'], 'bar')


def test_stereo_video_is_out_of_scope():
    video = VideoStimulus(np.zeros((10, 40, 3)), time=[0, 1, 2])
    with pytest.raises(TypeError):
        BinocularScene.from_side_by_side(video, fov=40 * dva)
