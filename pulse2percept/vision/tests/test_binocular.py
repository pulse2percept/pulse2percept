"""Two independent monocular views, held side by side (#668)

Scenes here are laid out one degree per pixel with an odd pixel count, as in
`test_scene`, so pixel centers land on whole degrees.
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
    """A uniform gray field, so a percept drawn on it is the only structure"""
    data = np.full((px, px), float(level))
    kwargs.setdefault('scotoma_blend', 0)
    return Scene(ImageStimulus(data), fov=(px, px), **kwargs)


def spot_percept(scene, x_dva=0.0, y_dva=0.0, brightness=10.0):
    """A single bright grid point at an eye-centered location"""
    data = np.zeros((SCENE_PX, SCENE_PX, 1))
    col, row = scene.dva_to_pixel(x_dva, y_dva)
    data[int(round(float(row))), int(round(float(col)))] = brightness
    return Percept(data, space=scene._grid())


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
    """Different source, FOV, shape, scotoma and aperture on the two sides"""
    left = Scene(ImageStimulus(np.full((9, 9), 0.3)), fov=(30, 30),
                 scotoma=Scotoma.circle(6), scotoma_fill=0.0,
                 aperture='ellipse')
    right = Scene(VideoStimulus(np.zeros((15, 21, 2)), time=[0, 10]),
                  fov=(42, 30), background=1)
    binocular = BinocularScene(left=left, right=right)
    npt.assert_equal(binocular.left.fov, (30.0, 30.0))
    npt.assert_equal(binocular.right.fov, (42.0, 30.0))
    npt.assert_equal(binocular.left.shape, (9, 9))
    npt.assert_equal(binocular.right.shape, (15, 21))
    npt.assert_equal(binocular.left.aperture, 'ellipse')
    npt.assert_equal(binocular.right.aperture, 'rectangle')
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
    """The fellow eye shows its own scene, whichever side is implanted"""
    left, right = flat_scene(0.2), flat_scene(0.8)
    binocular = BinocularScene(left=left, right=right)
    percept = spot_percept(left, x_dva=-4.0)

    ax_left, ax_right = binocular.plot(left_percept=percept, vmax=10)
    # The implanted eye shows the phosphene on black, the fellow eye its scene
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
    """No fusion: neither eye is averaged, maxed or otherwise combined"""
    left, right = flat_scene(0.0), flat_scene(0.0)
    binocular = BinocularScene(left=left, right=right)
    on_the_left = spot_percept(left, x_dva=-6.0)
    on_the_right = spot_percept(right, x_dva=6.0)
    ax_left, ax_right = binocular.plot(left_percept=on_the_left,
                                       right_percept=on_the_right, vmax=10)
    seen_left = ax_left.images[-1].get_array()
    seen_right = ax_right.images[-1].get_array()
    # Each eye shows its own phosphene, and only its own:
    npt.assert_almost_equal(seen_left[HALF, HALF - 6], [1.0] * 3, decimal=6)
    npt.assert_almost_equal(seen_left[HALF, HALF + 6], 0.0)
    npt.assert_almost_equal(seen_right[HALF, HALF + 6], [1.0] * 3, decimal=6)
    npt.assert_almost_equal(seen_right[HALF, HALF - 6], 0.0)
    # ... and each panel matches that eye's scene drawn on its own:
    npt.assert_almost_equal(seen_left,
                            left.plot(percept=on_the_left,
                                      vmax=10).images[-1].get_array(),
                            decimal=6)
    plt.close('all')


def test_each_eye_keeps_its_own_aperture():
    left = flat_scene(0.6, aperture='ellipse')
    right = flat_scene(0.6)
    ax_left, ax_right = BinocularScene(left=left, right=right).plot()
    # The corner is inside the rectangle but outside the ellipse:
    npt.assert_almost_equal(ax_left.images[-1].get_array()[0, 0], 0.0)
    npt.assert_almost_equal(ax_right.images[-1].get_array()[0, 0],
                            [0.6] * 3, decimal=6)
    plt.close('all')


def test_two_fovs_are_drawn_on_one_angular_scale():
    """A 31-degree field must not be stretched to look like a 41-degree one

    Both fields are oblong and oblong the other way round, so the widest one
    horizontally is the shorter one vertically. Reading a single dimension for
    both axes, or swapping width and height, lands on different numbers.
    """
    left = Scene(ImageStimulus(np.full((21, 31), 0.4)), fov=(31, 21))
    right = Scene(ImageStimulus(np.full((51, 41), 0.4)), fov=(41, 51))
    ax_left, ax_right = BinocularScene(left=left, right=right).plot()
    npt.assert_almost_equal(ax_left.get_xlim(), ax_right.get_xlim())
    npt.assert_almost_equal(ax_left.get_ylim(), ax_right.get_ylim())
    # Both axes span the wider field's stated outer extent, per dimension,
    # rather than the outermost pixel centers:
    npt.assert_almost_equal(ax_left.get_xlim(), (-20.5, 20.5))   # max width
    npt.assert_almost_equal(ax_left.get_ylim(), (-25.5, 25.5))   # max height
    # ... and the images keep the extents their own geometry gives them:
    npt.assert_almost_equal(ax_left.images[-1].get_extent(),
                            (-15.5, 15.5, -10.5, 10.5))
    npt.assert_almost_equal(ax_right.images[-1].get_extent(),
                            (-20.5, 20.5, -25.5, 25.5))
    plt.close('all')


def test_plot_lays_out_only_the_figure_it_made(monkeypatch):
    binocular = BinocularScene(left=flat_scene(), right=flat_scene())
    laid_out = []
    monkeypatch.setattr(plt.Figure, 'tight_layout',
                        lambda self, *a, **kw: laid_out.append(self))
    binocular.plot()
    npt.assert_equal(len(laid_out), 1)
    plt.close('all')
    # A caller's figure is left as the caller arranged it:
    _, axes = plt.subplots(1, 2)
    binocular.plot(axes=axes)
    npt.assert_equal(len(laid_out), 1)
    plt.close('all')


def test_a_shared_gaze_moves_both_eyes_together():
    scotoma = Scotoma.circle(4)
    left = flat_scene(0.7, scotoma=scotoma, scotoma_fill=0.0)
    right = flat_scene(0.7, scotoma=scotoma, scotoma_fill=0.0)
    axes = BinocularScene(left=left, right=right).plot(gaze=(6, 0) * dva)
    for ax, scene in zip(axes, (left, right)):
        npt.assert_almost_equal(ax.images[-1].get_array(),
                                scene._native_rgb(gaze=(6, 0))[..., 0],
                                decimal=6)
    plt.close('all')


def test_it_has_no_implant_or_model_behavior():
    """A container for two views, not a dispatcher"""
    binocular = BinocularScene(left=flat_scene(), right=flat_scene())
    for absent in ('eye', 'implant', 'predict_percept', 'play', 'fuse',
                   '_compose', '_device_input', '_sample_at'):
        npt.assert_equal(hasattr(binocular, absent), False)
    # A Scene is monocular; eye identity lives in the container, not the scene:
    npt.assert_equal(hasattr(binocular.left, 'eye'), False)


def packed_stereo(rows=10, half_cols=20, channels=None):
    """A side-by-side frame whose halves are unmistakably different"""
    shape = (rows, 2 * half_cols) + (() if channels is None else (channels,))
    stereo = np.zeros(shape, dtype=np.float32)
    stereo[:, half_cols:] = 1.0
    return stereo


def test_side_by_side_splits_left_from_right():
    stereo = packed_stereo()
    binocular = BinocularScene.from_side_by_side(stereo, fov=(40, 20) * dva)
    for eye, level in ((binocular.left, 0.0), (binocular.right, 1.0)):
        npt.assert_equal(eye.shape, (10, 20))
        npt.assert_almost_equal(eye.source.data.reshape(eye.shape),
                                np.full((10, 20), level))
    # The halves are drawn as given, neither swapped nor flipped:
    npt.assert_almost_equal(binocular.left._native_rgb().max(), 0.0)
    npt.assert_almost_equal(binocular.right._native_rgb().min(), 1.0)


def test_side_by_side_splits_columns_not_channels():
    stereo = packed_stereo(rows=6, half_cols=4, channels=3)
    stereo[:, 4:, 1:] = 0.25
    binocular = BinocularScene.from_side_by_side(stereo, fov=(40, 20) * dva)
    for eye in (binocular.left, binocular.right):
        npt.assert_equal(eye.source.img_shape, (6, 4, 3))
        npt.assert_equal(eye.shape, (6, 4))
    right = binocular.right.source.data.reshape(6, 4, 3)
    npt.assert_almost_equal(right[..., 0], 1.0)
    npt.assert_almost_equal(right[..., 1:], 0.25)


@pytest.mark.parametrize('n_cols', [21, 7])
def test_an_odd_width_has_no_seam(n_cols):
    stereo = np.zeros((10, n_cols))
    with pytest.raises(ValueError) as excinfo:
        BinocularScene.from_side_by_side(stereo, fov=40 * dva)
    npt.assert_equal('even' in str(excinfo.value), True)


def test_a_scalar_fov_describes_one_eye_not_the_packed_frame():
    """The packing geometry has no visual-field meaning"""
    binocular = BinocularScene.from_side_by_side(packed_stereo(), fov=40)
    for eye in (binocular.left, binocular.right):
        npt.assert_equal(eye.shape, (10, 20))
        # 40 degrees across 20 columns, so 20 degrees down 10 rows:
        npt.assert_almost_equal(eye.fov, (40.0, 20.0))


def test_an_explicit_fov_pair_goes_to_both_eyes():
    binocular = BinocularScene.from_side_by_side(packed_stereo(),
                                                 fov=(60, 40) * dva)
    npt.assert_almost_equal(binocular.left.fov, (60.0, 40.0))
    npt.assert_almost_equal(binocular.right.fov, (60.0, 40.0))


def test_scene_kwargs_reach_both_eyes():
    binocular = BinocularScene.from_side_by_side(packed_stereo(), fov=40 * dva,
                                                 aperture='ellipse',
                                                 background=0.5,
                                                 scotoma=Scotoma.circle(8))
    for eye in (binocular.left, binocular.right):
        npt.assert_equal(eye.aperture, 'ellipse')
        npt.assert_almost_equal(eye.background, (0.5, 0.5, 0.5))
        npt.assert_equal(eye.scotoma is not None, True)


def test_an_image_stimulus_is_a_stereo_source():
    stereo = ImageStimulus(packed_stereo())
    binocular = BinocularScene.from_side_by_side(stereo, fov=40 * dva)
    npt.assert_equal(binocular.left.shape, (10, 20))
    npt.assert_almost_equal(binocular.left.source.data.max(), 0.0)
    npt.assert_almost_equal(binocular.right.source.data.min(), 1.0)


def test_stereo_video_is_out_of_scope():
    video = VideoStimulus(np.zeros((10, 40, 3)), time=[0, 1, 2])
    with pytest.raises(TypeError):
        BinocularScene.from_side_by_side(video, fov=40 * dva)
