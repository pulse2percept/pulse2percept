"""Timestamped fixation events (#916)"""
import numpy as np
import numpy.testing as npt
import pytest
import matplotlib.pyplot as plt

from pulse2percept.percepts import Percept
from pulse2percept.stimuli import ImageStimulus, VideoStimulus
from pulse2percept.topography import Grid2D
from pulse2percept.units import DimensionMismatchError, dva, ms, s, um
from pulse2percept.vision import FixationalMotion, Gaze, Scene, Scotoma

SCENE_PX = 41
# One frame every 100 ms, so events at 0/200/300 ms land on frames 0, 2, 3:
FRAME_TIMES = np.arange(5) * 100.0
EVENTS = [(0, 0), (6, 2), (-4, 3)]
EVENT_TIMES = [0, 200, 300]
# EVENTS/EVENT_TIMES expanded onto FRAME_TIMES:
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
    """Gaze requires at least one fixation"""
    with pytest.raises(ValueError):
        Gaze(np.zeros((0, 2)) * dva, time=np.zeros(0) * ms)


def test_time_must_be_counted_in_a_unit_of_time():
    """Non-time units for 'time' are a DimensionMismatchError"""
    with pytest.raises(DimensionMismatchError):
        Gaze([(0, 0), (6, 2)] * dva, time=[0, 400] * um)
    with pytest.raises(DimensionMismatchError):
        Gaze([(0, 0), (6, 2)] * dva, time=[0, 400] * dva)


def test_a_trajectory_cannot_be_edited_after_construction():
    """positions and time are read-only copies"""
    positions = np.array(EVENTS, dtype=float)
    times = np.array(EVENT_TIMES, dtype=float)
    gaze = Gaze(positions * dva, time=times * ms)
    with pytest.raises(ValueError):
        gaze.positions[0, 0] = 99
    with pytest.raises(ValueError):
        gaze.time[0] = 99
    # Changing the input arrays has no effect:
    positions[0, 0] = 99
    times[0] = 99
    npt.assert_almost_equal(gaze.positions, EVENTS)
    npt.assert_almost_equal(gaze.time, EVENT_TIMES)


def test_a_fixation_starts_at_its_timestamp_and_is_held():
    """A frame exactly on an event time uses the new fixation"""
    npt.assert_almost_equal(trajectory()._at(FRAME_TIMES, ms), EXPANDED)
    # Between events the previous fixation holds; the last one holds past the
    # end:
    npt.assert_almost_equal(trajectory()._at([199.9, 200, 5000], ms),
                            [(0, 0), (6, 2), (-4, 3)])


def test_gaze_before_the_first_event_is_undefined():
    gaze = Gaze([(0, 0), (6, 2)] * dva, time=[100, 200] * ms)
    with pytest.raises(ValueError):
        gaze._at([0, 100, 200], ms)


def test_bare_times_are_milliseconds_and_units_convert():
    """Bare times are ms; times in s give the same trajectory"""
    bare = Gaze(EVENTS * dva, time=EVENT_TIMES)
    npt.assert_equal(bare.time_unit, ms)
    npt.assert_almost_equal(bare.time, EVENT_TIMES)
    in_seconds = Gaze(EVENTS * dva, time=[0, 0.2, 0.3] * s)
    npt.assert_equal(in_seconds.time_unit, s)
    npt.assert_almost_equal(in_seconds.time, [0, 0.2, 0.3])
    for gaze in (bare, in_seconds):
        npt.assert_almost_equal(gaze._at(FRAME_TIMES, ms), EXPANDED)
    # Query times may use a different unit:
    npt.assert_almost_equal(in_seconds._at(FRAME_TIMES / 1000.0, s), EXPANDED)


def test_rows_carry_their_own_timestamps():
    """(x, y, time) rows match positions + time"""
    reference = trajectory()
    rows = [(0, 0, 0), (6 * dva, 2 * dva, 200 * ms),
            (-4 * dva, 3 * dva, 0.3 * s)]
    for gaze in (Gaze(rows), Gaze([(0, 0, 0), (6, 2, 200), (-4, 3, 300)]),
                 Gaze(np.array([(0, 0, 0), (6, 2, 200), (-4, 3, 300)]))):
        npt.assert_almost_equal(gaze.positions, reference.positions)
        npt.assert_almost_equal(gaze.time, EVENT_TIMES)
        npt.assert_equal(gaze.time_unit, ms)
        npt.assert_almost_equal(gaze._at(FRAME_TIMES, ms), EXPANDED)
        with pytest.raises(ValueError):
            gaze.time[0] = 99
    # A nonzero bare entry keeps its default unit (dva, ms):
    mixed = Gaze([(1, 0, 0), (2 * dva, 0, 1 * s)])
    npt.assert_almost_equal(mixed.positions, [(1, 0), (2, 0)])
    npt.assert_almost_equal(mixed.time, [0, 1000])


def test_rows_are_validated():
    with pytest.raises(ValueError):  # (x, y) without time
        Gaze([(0, 0), (1, 1)])
    with pytest.raises(ValueError):  # ragged rows
        Gaze([(0, 0, 0), (1, 1)])
    with pytest.raises(ValueError):  # not rows at all
        Gaze([0, 0, 0])
    with pytest.raises(ValueError):  # empty
        Gaze([])
    with pytest.raises(ValueError):  # rows with time, and time too
        Gaze([(0, 0, 0), (1, 1, 10)], time=[0, 10])
    with pytest.raises(ValueError):
        Gaze([(0, 0, 0), (1 * dva, 1 * dva, 10 * ms)], time=[0, 10] * ms)
    with pytest.raises(ValueError):  # strictly increasing
        Gaze([(0, 0, 10), (1, 1, 10)])
    with pytest.raises(ValueError):  # finite
        Gaze([(0, np.nan, 0), (1, 1, 10)])
    with pytest.raises(DimensionMismatchError):  # time column in dva
        Gaze([(0, 0, 0), (1 * dva, 1 * dva, 10 * dva)])
    with pytest.raises(DimensionMismatchError):  # position in ms
        Gaze([(0, 0, 0), (1 * ms, 1 * dva, 10 * ms)])
    with pytest.raises(DimensionMismatchError):  # one unit for all columns
        Gaze([(0, 0, 0), (1, 1, 10)] * dva)


def test_sparse_events_render_like_the_expanded_trajectory():
    """Sparse events render the same as one gaze per frame"""
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


def test_play_accepts_a_trajectory_with_or_without_rings():
    scene = video_scene()
    scene.play(gaze=trajectory())
    # Rings are eye-centered, like the displayed FOV:
    scene.play(gaze=trajectory(), rings=True)
    plt.close('all')


def test_a_still_scene_samples_a_trajectory_at_its_events():
    scene = Scene(ImageStimulus(np.zeros((8, 8))), fov=(8, 8))
    npt.assert_almost_equal(scene.render(gaze=trajectory()).time, EVENT_TIMES)
    # Fixational motion is zero at every event, so it is refused:
    gaze = Gaze(EVENTS * dva, time=EVENT_TIMES * ms,
                fixational_motion=FixationalMotion(seed=3))
    for call in (scene.render, scene.plot, scene.play):
        with pytest.raises(ValueError):
            call(gaze=gaze)
    plt.close('all')


def test_the_output_clock_and_the_rendered_frames_never_disagree():
    """_output_clock frame count matches render() for any scene and percept"""
    grid = Grid2D((-4, 4), (-4, 4), step=1)
    untimed = Percept(np.random.rand(9, 9, 1), space=grid)
    timed = Percept(np.random.rand(9, 9, FRAME_TIMES.size), space=grid,
                    time=FRAME_TIMES)
    still = Scene(ImageStimulus(np.zeros((8, 8))), fov=(8, 8))
    for scene in (video_scene(), still):
        for percept in (None, untimed, timed):
            for gaze in (None, trajectory()):
                kwargs = {} if percept is None else {'percept': percept,
                                                     'vmax': 1}
                n_drawn = scene.render(gaze=gaze, **kwargs).data.shape[-1]
                time, _, n_clock = scene._output_clock(percept, gaze)
                npt.assert_equal(n_clock, n_drawn)
                if time is not None:
                    npt.assert_equal(np.size(time), n_drawn)


def test_gaze_resolves_on_scene_frames_not_on_percept_response_times():
    """Gaze is resolved at scene frame times, not percept time stamps

    Percept frames are stamped one frame late, which would shift every
    fixation by one frame.
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


def test_play_resolves_gaze_as_render_does():
    scene = video_scene()
    late = Percept(np.random.rand(9, 9, FRAME_TIMES.size),
                   space=Grid2D((-4, 4), (-4, 4), step=1),
                   time=FRAME_TIMES + 100.0,
                   metadata={'source_frame_time': FRAME_TIMES})
    npt.assert_allclose(
        scene.play(percept=late, gaze=trajectory(), vmax=1,
                   view='eye')._frame_data,
        scene.render(percept=late, gaze=EXPANDED * dva, vmax=1).data,
        atol=1e-6)
    plt.close('all')


def test_a_still_scene_resolves_against_a_timed_percept():
    """A still scene resolves gaze at the percept's frame times"""
    scene = Scene(ImageStimulus(np.zeros((8, 8))), fov=(8, 8),
                  scotoma=Scotoma.circle(2 * dva), scotoma_fill=0.5,
                  scotoma_blend=0)
    percept = Percept(np.random.rand(9, 9, FRAME_TIMES.size),
                      space=Grid2D((-4, 4), (-4, 4), step=1),
                      time=FRAME_TIMES)
    npt.assert_array_equal(
        scene.render(percept=percept, gaze=trajectory(), vmax=1).data,
        scene.render(percept=percept, gaze=EXPANDED * dva, vmax=1).data)


# Fixational motion; onsets off the 10 ms grid:
ROWS = [(0, 0, 0), (-29 * dva, -7 * dva, 605 * ms),
        (22 * dva, -10 * dva, 1270 * ms)]
# The internal grid, so each microsaccade is one step:
GRID = np.arange(0, 20001, 10.0)


def wandering(seed=42, **kwargs):
    return Gaze(ROWS, fixational_motion=FixationalMotion(seed=seed, **kwargs))


def offsets(seed, **kwargs):
    """Returns 20 s of offsets from one fixation at the origin, in dva"""
    gaze = Gaze([(0, 0)] * dva, time=[0] * ms,
                fixational_motion=FixationalMotion(seed=seed, **kwargs))
    return gaze._at(GRID, ms)


def test_fixational_motion_validates_its_arguments():
    for motion in ('drift', FixationalMotion, 42):
        with pytest.raises(TypeError):
            Gaze(ROWS, fixational_motion=motion)
    with pytest.raises(TypeError):
        FixationalMotion(drift=1)
    with pytest.raises(TypeError):
        FixationalMotion(microsaccades=None)
    for seed in (-1, 1.5, True, '42'):
        with pytest.raises(ValueError):
            FixationalMotion(seed=seed)


def test_without_fixational_motion_gaze_is_unchanged():
    """No motion, or motion with both components off, holds each fixation"""
    npt.assert_equal(trajectory().fixational_motion, None)
    still = FixationalMotion(drift=False, microsaccades=False, seed=1)
    gaze = Gaze(EVENTS * dva, time=EVENT_TIMES * ms, fixational_motion=still)
    npt.assert_array_equal(gaze._at(FRAME_TIMES, ms), EXPANDED)


def test_fixational_motion_keeps_the_intended_fixations():
    gaze = wandering()
    positions, time = gaze.positions.copy(), gaze.time.copy()
    moved = gaze._at(np.arange(0, 2000, 40.0), ms)
    npt.assert_array_equal(gaze.positions, positions)
    npt.assert_array_equal(gaze.time, time)
    with pytest.raises(ValueError):
        gaze.positions[0, 0] = 99
    with pytest.raises(ValueError):
        gaze.time[0] = 99
    # Rows and positions + time give the same motion:
    split = Gaze(positions * dva, time=time * ms,
                 fixational_motion=gaze.fixational_motion)
    npt.assert_array_equal(split._at(np.arange(0, 2000, 40.0), ms), moved)


def test_fixational_motion_is_reproducible():
    times = np.arange(0, 2000, 40.0)
    npt.assert_array_equal(wandering(7)._at(times, ms),
                           wandering(7)._at(times, ms))
    npt.assert_equal(np.array_equal(wandering(7)._at(times, ms),
                                    wandering(8)._at(times, ms)), False)
    # Without a seed, one is drawn once and reused:
    unseeded = Gaze(ROWS, fixational_motion=FixationalMotion())
    first = unseeded._at(times, ms)
    npt.assert_array_equal(unseeded._at(times, ms), first)
    npt.assert_array_equal(
        wandering(unseeded.fixational_motion.seed)._at(times, ms), first)


def test_fixational_motion_does_not_depend_on_how_it_is_sampled():
    """Sampling order, subsets, and frame rate leave the trajectory as is"""
    gaze = wandering()
    npt.assert_array_equal(gaze._at([100, 200], ms)[1],
                           gaze._at([200], ms)[0])
    at_25hz = np.arange(0, 3000, 40.0)
    expected = gaze._at(at_25hz, ms)
    one_by_one = [gaze._at([t], ms)[0] for t in at_25hz[::-1]][::-1]
    npt.assert_array_equal(one_by_one, expected)
    shuffled = np.random.default_rng(0).permutation(at_25hz.size)
    npt.assert_array_equal(gaze._at(at_25hz[shuffled], ms),
                           expected[shuffled])
    # 1 kHz output, compared at the 25 Hz frame times:
    npt.assert_array_equal(gaze._at(np.arange(0, 3000, 1.0), ms)[::40],
                           expected)


def test_the_offset_resets_at_each_fixation_onset():
    gaze = wandering()
    npt.assert_array_equal(gaze._at([0, 605, 1270], ms), gaze.positions)
    # The motion follows time since onset, not the absolute clock:
    later = Gaze(gaze.positions * dva, time=(gaze.time + 3) * ms,
                 fixational_motion=gaze.fixational_motion)
    times = np.arange(0, 2000, 7.0)
    npt.assert_allclose(later._at(times + 3, ms), gaze._at(times, ms),
                        rtol=0, atol=1e-12)


def test_seconds_and_milliseconds_give_the_same_motion():
    gaze = wandering()
    in_seconds = Gaze([(0, 0, 0), (-29 * dva, -7 * dva, 0.605 * s),
                       (22 * dva, -10 * dva, 1.27 * s)],
                      fixational_motion=gaze.fixational_motion)
    times = np.arange(0, 3000, 40.0)
    expected = gaze._at(times, ms)
    for query, unit in ((times, ms), (times / 1000, s)):
        npt.assert_allclose(in_seconds._at(query, unit), expected, rtol=0,
                            atol=1e-12)


def test_the_last_fixation_keeps_moving_after_its_onset():
    gaze = wandering()
    late = gaze._at([5000, 60000], ms) - gaze.positions[-1]
    npt.assert_equal(np.all(np.isfinite(late)), True)
    npt.assert_equal(np.all(np.hypot(*late.T) > 0), True)
    npt.assert_equal(np.all(np.hypot(*late.T) < 0.5), True)


def test_drift_is_temporally_correlated():
    """Correlated over 10 ms, unlike white noise; decorrelated over 2 s"""
    for seed in range(5):
        x = offsets(seed, microsaccades=False)[50:, 0]
        npt.assert_equal(np.corrcoef(x[:-1], x[1:])[0, 1] > 0.9, True)
        npt.assert_equal(np.corrcoef(x[:-200], x[200:])[0, 1] < 0.5, True)


def test_fixational_motion_stays_near_the_target():
    moved = np.array([offsets(seed) for seed in range(20)])
    npt.assert_equal(np.hypot(*moved.T).max() < 0.5, True)
    # No systematic drift away from fixation:
    npt.assert_equal(np.all(np.abs(moved.reshape(-1, 2).mean(0)) < 0.05),
                     True)


def test_microsaccades_are_rare_jumps_larger_than_drift():
    jumps = np.array([np.hypot(*np.diff(offsets(seed, drift=False), axis=0).T)
                      for seed in range(20)])
    sizes = jumps[jumps > 0]
    npt.assert_equal(np.all((sizes >= 0.1) & (sizes <= 0.3)), True)
    rate = sizes.size / (20 * GRID[-1] / 1000)  # per s
    npt.assert_equal(0.7 < rate < 1.3, True)
    for row in jumps:
        onsets = np.flatnonzero(row) * 10.0  # ms
        npt.assert_equal(onsets.min() >= 150 - 10, True)
        npt.assert_equal(np.diff(onsets).min() >= 150 - 10, True)
    drift_steps = np.hypot(*np.diff(offsets(0, microsaccades=False),
                                    axis=0).T)
    npt.assert_equal(drift_steps.max() < 0.05, True)


def test_a_fixed_seed_alternates_drift_and_microsaccades():
    """Small drift steps, with a few large microsaccades"""
    steps = np.hypot(*np.diff(offsets(42)[:300], axis=0).T)  # 3 s
    large = np.flatnonzero(steps > 0.05)
    npt.assert_equal(1 <= large.size <= 8, True)
    npt.assert_equal(np.all(np.diff(large) >= 14), True)
    small = np.delete(steps, large)
    npt.assert_equal(np.all((small > 0) & (small < 0.05)), True)


def test_render_and_play_resolve_the_same_fixational_motion():
    scene = video_scene()
    gaze = Gaze(EVENTS * dva, time=EVENT_TIMES * ms,
                fixational_motion=FixationalMotion(seed=3))
    moved = gaze._at(FRAME_TIMES, ms)
    npt.assert_equal(np.array_equal(moved, EXPANDED), False)
    npt.assert_array_equal(scene.render(gaze=gaze).data,
                           scene.render(gaze=moved * dva).data)
    percept = Percept(np.random.rand(9, 9, FRAME_TIMES.size),
                      space=Grid2D((-4, 4), (-4, 4), step=1),
                      time=FRAME_TIMES)
    npt.assert_allclose(
        scene.play(percept=percept, gaze=gaze, vmax=1,
                   view='eye')._frame_data,
        scene.render(percept=percept, gaze=moved * dva, vmax=1).data,
        atol=1e-6)
    plt.close('all')
