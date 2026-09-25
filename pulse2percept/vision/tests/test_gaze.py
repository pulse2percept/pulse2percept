"""Timestamped fixation events (#916)"""
import numpy as np
import numpy.testing as npt
import pytest
import matplotlib.pyplot as plt

from pulse2percept.percepts import Percept
from pulse2percept.stimuli import ImageStimulus, VideoStimulus
from pulse2percept.topography import Grid2D
from pulse2percept.units import DimensionMismatchError, dva, ms, s, um
from pulse2percept.vision import Gaze, Scene, Scotoma

SCENE_PX = 41
# One frame every 100 ms, so events at 0/200/300 land on frames 0, 2 and 3:
FRAME_TIMES = np.arange(5) * 100.0
EVENTS = [(0, 0), (6, 2), (-4, 3)]
EVENT_TIMES = [0, 200, 300]
# What EVENTS/EVENT_TIMES expand to on FRAME_TIMES:
EXPANDED = np.array([(0, 0), (0, 0), (6, 2), (-4, 3), (-4, 3)], dtype=float)


def video_scene(**kwargs):
    """Five frames of a moving ramp, 100 ms apart"""
    ramp = np.tile(np.linspace(0, 1, SCENE_PX), (SCENE_PX, 1))
    frames = np.stack([np.roll(ramp, 4 * f, axis=1)
                       for f in range(FRAME_TIMES.size)], axis=-1)
    kwargs.setdefault('scotoma', Scotoma.circle(5 * dva))
    kwargs.setdefault('scotoma_fill', 0.5)
    kwargs.setdefault('scotoma_blend', 0)
    return Scene(VideoStimulus(frames, time=FRAME_TIMES),
                 fov=(SCENE_PX, SCENE_PX), **kwargs)


def trajectory():
    return Gaze(EVENTS * dva, time=EVENT_TIMES * ms)


def test_gaze_validates_its_arguments():
    with pytest.raises(ValueError):  # not (n, 2)
        Gaze([0, 0] * dva, time=[0] * ms)
    with pytest.raises(ValueError):  # one timestamp per fixation
        Gaze([(0, 0), (1, 1)] * dva, time=[0] * ms)
    with pytest.raises(ValueError):  # finite positions
        Gaze([(0, 0), (np.nan, 1)] * dva, time=[0, 10] * ms)
    with pytest.raises(ValueError):  # finite times
        Gaze([(0, 0), (1, 1)] * dva, time=[0, np.inf] * ms)
    with pytest.raises(ValueError):  # strictly increasing
        Gaze([(0, 0), (1, 1)] * dva, time=[10, 10] * ms)
    with pytest.raises(ValueError):
        Gaze([(0, 0), (1, 1)] * dva, time=[10, 0] * ms)


def test_an_empty_trajectory_is_refused():
    """No fixation at all says nothing about where the eye was pointing"""
    with pytest.raises(ValueError):
        Gaze(np.zeros((0, 2)) * dva, time=np.zeros(0) * ms)


def test_time_must_be_counted_in_a_unit_of_time():
    """As for Percept: a length or an angle is not a clock"""
    with pytest.raises(DimensionMismatchError):
        Gaze([(0, 0), (6, 2)] * dva, time=[0, 400] * um)
    with pytest.raises(DimensionMismatchError):
        Gaze([(0, 0), (6, 2)] * dva, time=[0, 400] * dva)


def test_a_trajectory_cannot_be_edited_after_construction():
    """Mutating it would change what an already-resolved gaze meant"""
    positions = np.array(EVENTS, dtype=float)
    times = np.array(EVENT_TIMES, dtype=float)
    gaze = Gaze(positions * dva, time=times * ms)
    with pytest.raises(ValueError):
        gaze.positions[0, 0] = 99
    with pytest.raises(ValueError):
        gaze.time[0] = 99
    # Nor through the arrays it was built from:
    positions[0, 0] = 99
    times[0] = 99
    npt.assert_almost_equal(gaze.positions, EVENTS)
    npt.assert_almost_equal(gaze.time, EVENT_TIMES)


def test_a_fixation_starts_at_its_timestamp_and_is_held():
    """Right-continuous: a frame on an event already sees the new fixation"""
    npt.assert_almost_equal(trajectory()._at(FRAME_TIMES, ms), EXPANDED)
    # Between events the previous fixation stands, and the last one is held
    # past the end of the trajectory:
    npt.assert_almost_equal(trajectory()._at([199.9, 200, 5000], ms),
                            [(0, 0), (6, 2), (-4, 3)])


def test_gaze_before_the_first_event_is_undefined():
    gaze = Gaze([(0, 0), (6, 2)] * dva, time=[100, 200] * ms)
    with pytest.raises(ValueError):
        gaze._at([0, 100, 200], ms)


def test_bare_times_are_milliseconds_and_units_convert():
    """Seconds and milliseconds describe the same trajectory"""
    bare = Gaze(EVENTS * dva, time=EVENT_TIMES)
    npt.assert_equal(bare.time_unit, ms)
    npt.assert_almost_equal(bare.time, EVENT_TIMES)
    in_seconds = Gaze(EVENTS * dva, time=[0, 0.2, 0.3] * s)
    npt.assert_equal(in_seconds.time_unit, s)
    npt.assert_almost_equal(in_seconds.time, [0, 0.2, 0.3])
    for gaze in (bare, in_seconds):
        npt.assert_almost_equal(gaze._at(FRAME_TIMES, ms), EXPANDED)
    # The target clock may count in something else again:
    npt.assert_almost_equal(in_seconds._at(FRAME_TIMES / 1000.0, s), EXPANDED)


def test_sparse_events_render_like_the_expanded_trajectory():
    """Sparse events stand in for one gaze per frame"""
    scene = video_scene()
    npt.assert_array_equal(scene.render(gaze=trajectory()).data,
                           scene.render(gaze=EXPANDED * dva).data)


def test_sparse_events_place_a_percept_like_the_expanded_trajectory():
    scene = video_scene()
    percept = Percept(np.random.rand(9, 9, FRAME_TIMES.size),
                      space=Grid2D((-4, 4), (-4, 4), step=1),
                      time=FRAME_TIMES)
    sparse = scene.render(percept=percept, gaze=trajectory(), vmax=1)
    dense = scene.render(percept=percept, gaze=EXPANDED * dva, vmax=1)
    npt.assert_array_equal(sparse.data, dense.data)


@pytest.mark.parametrize('frame', range(FRAME_TIMES.size))
def test_plot_draws_the_fixation_held_at_that_frame(frame):
    scene = video_scene()
    drawn = scene.plot(gaze=trajectory(), frame=frame).images[-1].get_array()
    expected = scene.plot(gaze=EXPANDED[frame] * dva,
                          frame=frame).images[-1].get_array()
    npt.assert_almost_equal(np.asarray(drawn), np.asarray(expected))
    plt.close('all')


def test_play_accepts_a_trajectory_but_still_refuses_moving_rings():
    scene = video_scene()
    scene.play(gaze=trajectory())
    with pytest.raises(ValueError):
        scene.play(gaze=trajectory(), rings=True)
    plt.close('all')


def test_a_still_scene_has_no_clock_to_resolve_against():
    scene = Scene(ImageStimulus(np.zeros((8, 8))), fov=(8, 8))
    for call in (scene.render, scene.plot):
        with pytest.raises(ValueError):
            call(gaze=trajectory())
    plt.close('all')


def test_the_output_clock_and_the_rendered_frames_never_disagree():
    """One rule decides the frame count, whatever the scene and percept"""
    grid = Grid2D((-4, 4), (-4, 4), step=1)
    untimed = Percept(np.random.rand(9, 9, 1), space=grid)
    timed = Percept(np.random.rand(9, 9, FRAME_TIMES.size), space=grid,
                    time=FRAME_TIMES)
    still = Scene(ImageStimulus(np.zeros((8, 8))), fov=(8, 8))
    for scene in (video_scene(), still):
        for percept in (None, untimed, timed):
            kwargs = {} if percept is None else {'percept': percept,
                                                 'vmax': 1}
            n_drawn = scene.render(**kwargs).data.shape[-1]
            time, _, n_clock = scene._output_clock(percept)
            npt.assert_equal(n_clock, n_drawn)
            npt.assert_equal(scene._n_display_frames(percept), n_drawn)
            if time is not None:
                npt.assert_equal(np.size(time), n_drawn)


def test_gaze_resolves_on_scene_frames_not_on_percept_response_times():
    """A temporal model's output times label the render, but not the gaze

    Its frames are reported one frame period late, which would shift every
    fixation by a frame if gaze were resolved against them.
    """
    scene = video_scene()
    late = Percept(np.random.rand(9, 9, FRAME_TIMES.size),
                   space=Grid2D((-4, 4), (-4, 4), step=1),
                   time=FRAME_TIMES + 100.0,
                   metadata={'source_frame_time': FRAME_TIMES})
    npt.assert_almost_equal(scene._output_clock(late)[0], FRAME_TIMES)
    npt.assert_array_equal(
        scene.render(percept=late, gaze=trajectory(), vmax=1).data,
        scene.render(percept=late, gaze=EXPANDED * dva, vmax=1).data)


def test_a_still_scene_resolves_against_a_timed_percept():
    """With no scene clock the percept's own frame times are the output clock"""
    scene = Scene(ImageStimulus(np.zeros((8, 8))), fov=(8, 8),
                  scotoma=Scotoma.circle(2 * dva), scotoma_fill=0.5,
                  scotoma_blend=0)
    percept = Percept(np.random.rand(9, 9, FRAME_TIMES.size),
                      space=Grid2D((-4, 4), (-4, 4), step=1),
                      time=FRAME_TIMES)
    npt.assert_array_equal(
        scene.render(percept=percept, gaze=trajectory(), vmax=1).data,
        scene.render(percept=percept, gaze=EXPANDED * dva, vmax=1).data)
