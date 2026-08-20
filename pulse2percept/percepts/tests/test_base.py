from pulse2percept.topography import Grid2D
from pulse2percept.percepts import Percept
from pulse2percept.units import (DimensionMismatchError, Hz, kHz, ms, s, uA,
                                 um, us)
from skimage.io import imread
from skimage import img_as_float
import imageio
from imageio import mimread
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Subplot
import matplotlib.pyplot as plt
import json
import os
import warnings
import re
import numpy as np
import pytest
import numpy.testing as npt
import matplotlib
matplotlib.use('Agg')


def player(ani):
    """The config the JavaScript player was handed"""
    html = ani.to_jshtml()
    return json.loads(re.search(r'var cfg = (\{.*?\});', html, re.S).group(1))


def test_Percept():
    # Automatic axes:
    ndarray = np.arange(15).reshape((3, 5, 1))
    percept = Percept(ndarray, metadata='meta')
    npt.assert_equal(percept.shape, ndarray.shape)
    npt.assert_equal(percept.metadata, 'meta')
    npt.assert_equal(hasattr(percept, 'xdva'), True)
    npt.assert_almost_equal(percept.xdva, np.arange(ndarray.shape[1]))
    npt.assert_equal(hasattr(percept, 'ydva'), True)
    npt.assert_almost_equal(percept.ydva, np.arange(ndarray.shape[0]))
    # Singleton dimensions can be None:
    npt.assert_equal(hasattr(percept, 'time'), True)
    npt.assert_equal(percept.time, None)

    # Specific labels:
    percept = Percept(ndarray, time=0.4)
    npt.assert_almost_equal(percept.time, [0.4])
    percept = Percept(ndarray, time=[0.4])
    npt.assert_almost_equal(percept.time, [0.4])

    # Labels from a grid.
    y_range = (-1, 1)
    x_range = (-2, 2)
    grid = Grid2D(x_range, y_range)
    percept = Percept(ndarray, space=grid)
    npt.assert_almost_equal(percept.xdva, grid._xflat)
    npt.assert_almost_equal(percept.ydva, grid._yflat)
    npt.assert_equal(percept.time, None)
    grid = Grid2D(x_range, y_range)
    percept = Percept(ndarray, space=grid, time=0)
    npt.assert_almost_equal(percept.xdva, grid._xflat)
    npt.assert_almost_equal(percept.ydva, grid._yflat)
    npt.assert_almost_equal(percept.time, [0])

    # Gray levels
    for n_gray in [2, 4]:
        percept = Percept(np.arange(49, dtype=float).reshape((7, 7, 1)),
                          n_gray=n_gray)
        npt.assert_equal(len(np.unique(percept.data)), n_gray)

    with pytest.raises(TypeError):
        Percept(ndarray, space={'x': [0, 1, 2], 'y': [0, 1, 2, 3, 4]})
    with pytest.raises(ValueError):
        Percept(ndarray, n_gray=1.2)
    with pytest.raises(ValueError):
        Percept(ndarray, n_gray=-3)

    # Noise:
    data = np.arange(100, dtype=float).reshape((5, 5, 4))
    npt.assert_almost_equal(Percept(data, noise=0).data, data)
    npt.assert_almost_equal(Percept(data, noise=0.0).data, data)
    for noise in [0.5, 1.0]:
        percept = Percept(data, noise=noise)
        n_white = sum(np.isclose(percept.data.ravel(), 99.0))
        n_black = sum(np.isclose(percept.data.ravel(), 0.0))
        npt.assert_equal(abs(n_white - 0.5 * noise * data.size) <= 2, True)
        npt.assert_equal(abs(n_black - 0.5 * noise * data.size) <= 2, True)


def test_Percept__iter__():
    ndarray = np.zeros((2, 4, 3))
    ndarray[..., 1] = 1
    ndarray[..., 2] = 2
    percept = Percept(ndarray)
    for i, frame in enumerate(percept):
        npt.assert_equal(frame.shape, (2, 4))
        npt.assert_almost_equal(frame, i)


def test_Percept_argmax():
    percept = Percept(np.arange(30).reshape((3, 5, 2)))
    npt.assert_almost_equal(percept.argmax(), 29)
    npt.assert_almost_equal(percept.argmax(axis="frames"), 1)
    with pytest.raises(TypeError):
        percept.argmax(axis=(0, 1))
    with pytest.raises(ValueError):
        percept.argmax(axis='invalid')


def test_Percept_max():
    percept = Percept(np.arange(30).reshape((3, 5, 2)))
    npt.assert_almost_equal(percept.max(), 29)
    npt.assert_almost_equal(percept.max(axis="frames"),
                            percept.data[..., 1])
    npt.assert_almost_equal(percept.max(),
                            percept.data.ravel()[percept.argmax()])
    npt.assert_almost_equal(percept.max(axis='frames'),
                            percept.data[..., percept.argmax(axis='frames')])
    with pytest.raises(TypeError):
        percept.max(axis=(0, 1))
    with pytest.raises(ValueError):
        percept.max(axis='invalid')


def test_Percept_plot():
    y_range = (-1, 1)
    x_range = (-2, 2)
    grid = Grid2D(x_range, y_range)
    percept = Percept(np.arange(15).reshape((3, 5, 1)), space=grid)

    # Basic usage of pcolor:
    ax = percept.plot(kind='pcolor')
    npt.assert_equal(isinstance(ax, Subplot), True)
    npt.assert_almost_equal(ax.axis(), [*x_range, *y_range])
    frame = percept.max(axis='frames')
    npt.assert_almost_equal(ax.collections[0].get_clim(),
                            [frame.min(), frame.max()])

    # Basic usage of hex:
    ax = percept.plot(kind='hex')
    npt.assert_equal(isinstance(ax, Subplot), True)
    npt.assert_almost_equal(ax.axis(), [percept.xdva[0], percept.xdva[-1],
                                        percept.ydva[0], percept.ydva[-1]])
    npt.assert_almost_equal(ax.collections[0].get_clim(),
                            [percept.data[..., 0].min(),
                             percept.data[..., 0].max()])

    # Verify color map:
    npt.assert_equal(ax.collections[0].cmap, plt.cm.gray)

    # Specify figsize:
    ax = percept.plot(kind='pcolor', figsize=(6, 4))
    npt.assert_almost_equal(ax.figure.get_size_inches(), (6, 4))

    # Test vmin and vmax
    ax.clear()
    ax = percept.plot(vmin=2, vmax=4)
    npt.assert_equal(ax.collections[0].get_clim(), (2., 4.))

    # Invalid calls:
    with pytest.raises(ValueError):
        percept.plot(kind='invalid')
    with pytest.raises(TypeError):
        percept.plot(ax='invalid')


@ pytest.mark.parametrize('n_frames', (2, 3, 10, 14))
def test_Percept_play(n_frames):
    ndarray = np.random.rand(2, 4, n_frames)
    percept = Percept(ndarray)
    ani = percept.play()
    npt.assert_equal(isinstance(ani, FuncAnimation), True)
    npt.assert_equal(len(list(ani.frame_seq)), n_frames)
    # The animation renders as a self-contained HTML player:
    html = ani.to_jshtml()
    npt.assert_equal('p2p-anim' in html, True)
    npt.assert_equal(f'"n": {n_frames}' in html, True)
    # Time is annotated in the title unless turned off:
    npt.assert_equal(f't = {percept.time[-1]:.2f} ms' in html, True)
    html = percept.play(annotate_time=False).to_jshtml()
    npt.assert_equal(f't = {percept.time[-1]:.2f} ms' in html, False)


def test_Percept_play_single_frame():
    """A percept with a single time point has no frame rate of its own"""
    percept = Percept(np.random.rand(4, 4, 1), time=[3.5])
    html = percept.play().to_jshtml()
    npt.assert_equal('"n": 1' in html, True)
    npt.assert_equal('t = 3.50 ms' in html, True)
    # Without a time axis it is not an animation at all:
    with pytest.raises(ValueError):
        Percept(np.random.rand(4, 4, 1)).play()


def test_Percept_play_fmt():
    percept = Percept(np.random.rand(8, 8, 4))
    # A percept is scalar, so the lossless default already costs only one byte
    # per pixel -- and JPEG rings around high-contrast phosphenes:
    npt.assert_equal('data:image/jpeg;base64,' in percept.play().to_jshtml(),
                     False)
    npt.assert_equal('data:image/jpeg;base64,' in
                     percept.play(fmt='jpg').to_jshtml(), True)
    with pytest.raises(ValueError):
        percept.play(fmt='gif')


@ pytest.mark.parametrize('dtype', (np.float32, np.uint8))
def test_Percept_save(dtype, tmp_path):
    ndarray = np.arange(256, dtype=dtype).repeat(31).reshape((-1, 16, 16))
    percept = Percept(ndarray.transpose((2, 0, 1)))

    # Save multiple frames as a gif or movie:
    for name in ['test.mp4', 'test.avi', 'test.mov', 'test.wmv', 'test.gif']:
        fname = str(tmp_path / name)
        percept.save(fname, vmin=0, vmax=255)
        npt.assert_equal(os.path.isfile(fname), True)
        # Normalized to [0, 255] with some loss of precision:
        for mov in mimread(fname):
            npt.assert_equal(np.min(mov) <= 10, True)
            npt.assert_equal(np.max(mov) >= 240, True)

    # Cannot save multiple frames image:
    fname = str(tmp_path / 'test.jpg')
    with pytest.raises(ValueError):
        percept.save(fname, vmin=0, vmax=255)

    # But, can save single frame as image:
    percept = Percept(ndarray[..., :1])
    for name in ['test.jpg', 'test.png', 'test.tif', 'test.gif']:
        fname = str(tmp_path / name)
        percept.save(fname, vmin=0, vmax=255)
        npt.assert_equal(os.path.isfile(fname), True)
        img = img_as_float(imread(fname))
        npt.assert_almost_equal(np.min(img), 0, decimal=3)
        npt.assert_almost_equal(np.max(img), 1.0, decimal=3)


def test_Percept_save_single_frame(tmp_path):
    """A percept with a single time point has no frame rate of its own"""
    percept = Percept(np.random.rand(16, 16, 1), time=[3.5])
    for name in ['test.mp4', 'test.avi', 'test.gif']:
        fname = str(tmp_path / name)
        percept.save(fname, vmin=0, vmax=1)
        npt.assert_equal(len(mimread(fname)), 1)
    # An explicit frame rate is still honored:
    fname = str(tmp_path / 'fps.mp4')
    percept.save(fname, fps=12, vmin=0, vmax=1)
    npt.assert_equal(len(mimread(fname)), 1)


def test_Percept_fps_units(tmp_path):
    """A frame rate is a frequency, however it is spelled

    .. versionadded:: 0.10.0
    """
    # One second of percept, sampled at 100 Hz:
    percept = Percept(np.random.rand(8, 8, 100), time=np.arange(0, 1000, 10))

    def interval(**kwargs):
        """The frame delay (ms) the HTML player was configured with"""
        html = percept.play(**kwargs).to_jshtml()
        return float(re.search(r'"interval": ([0-9.]+)', html).group(1))

    # 33.33 ms, which is what 30 frames per second asks for:
    npt.assert_almost_equal(interval(fps=30), 1000 / 30, decimal=6)
    for spelling in (30 * Hz, 0.03 * kHz):
        npt.assert_almost_equal(interval(fps=spelling), interval(fps=30),
                                decimal=12)

    # ... and the same on the way out to a file:
    fname = str(tmp_path / 'fps.mp4')
    percept.save(fname, fps=30 * Hz, vmin=0, vmax=1)
    npt.assert_equal(len(mimread(fname)), 30)
    percept.save(fname, fps=0.03 * kHz, vmin=0, vmax=1)
    npt.assert_equal(len(mimread(fname)), 30)

    # Nothing else is a frame rate:
    for wrong in (30 * ms, 30 * uA):
        with pytest.raises(DimensionMismatchError):
            percept.play(fps=wrong)
        with pytest.raises(DimensionMismatchError):
            percept.save(fname, fps=wrong)


def test_Percept_units():
    """A percept's time axis knows what unit it is written in

    .. versionadded:: 0.10.0
    """
    data = np.zeros((3, 3, 2))
    # Milliseconds unless told otherwise, and a bare time axis keeps the
    # meaning it has always had:
    percept = Percept(data, time=[0, 10])
    npt.assert_equal(percept.time_unit, ms)
    npt.assert_almost_equal(percept.time, [0, 10])

    # A unitful time axis is normalized *into* the percept's unit rather than
    # changing it. Deterministic, and the same rule as everywhere else in p2p:
    percept = Percept(data, time=[0, 0.01] * s)
    npt.assert_equal(percept.time_unit, ms)
    npt.assert_allclose(percept.time, [0, 10], rtol=1e-12)
    # ... including a sequence built one element at a time:
    npt.assert_allclose(Percept(data, time=[0 * ms, 10000 * us]).time,
                        [0, 10], rtol=1e-12)

    # Storing in another unit is the caller's choice -- a model passes its own
    # `time_unit` -- and then bare numbers mean *that* unit:
    percept = Percept(data, time=[0, 0.01], time_unit=s)
    npt.assert_equal(percept.time_unit, s)
    npt.assert_allclose(percept.time, [0, 0.01], rtol=1e-12)
    npt.assert_allclose(percept.times(ms), [0, 10], rtol=1e-12)
    # `times()` with no unit hands back the stored array, unconverted:
    npt.assert_allclose(percept.times(), [0, 0.01], rtol=0, atol=0)
    npt.assert_equal(percept.time_quantity.unit, s)
    npt.assert_allclose(percept.time_quantity.to_value(ms), [0, 10],
                        rtol=1e-12)
    # A quantity handed to a percept that stores seconds lands in seconds:
    npt.assert_allclose(Percept(data, time=[0, 10] * ms, time_unit=s).time,
                        [0, 0.01], rtol=1e-12)

    # Nothing to express in any unit without a time axis:
    spatial = Percept(np.zeros((3, 3, 1)))
    npt.assert_equal(spatial.time, None)
    npt.assert_equal(spatial.times(s), None)
    npt.assert_equal(spatial.time_quantity, None)
    npt.assert_equal(spatial.time_unit, ms)

    # `data` is perceived brightness in arbitrary units, so it has no unit of
    # its own and gains none here:
    npt.assert_equal(hasattr(percept, 'unit'), False)

    # `time_unit` has to be a unit, and a unit of time:
    with pytest.raises(TypeError):
        Percept(data, time=[0, 10], time_unit='ms')
    with pytest.raises(DimensionMismatchError):
        Percept(data, time=[0, 10], time_unit=um)
    # ... and so does `time` itself:
    with pytest.raises(DimensionMismatchError):
        Percept(data, time=[0, 10] * um)


def test_Percept_animates_in_wall_clock_time(tmp_path, monkeypatch):
    """The label is in the percept's unit, the frame rate is in real time

    Two percepts describing the same 50 Hz sequence play at the same speed
    whether they were written down in milliseconds or in seconds.
    """
    data = np.random.rand(4, 4, 3)
    milli = Percept(data, time=[0, 20, 40])
    second = Percept(data, time=[0, 0.02, 0.04], time_unit=s)

    # `play`: same delay between frames...
    milli_ani, second_ani = milli.play(), second.play()
    npt.assert_almost_equal(milli_ani._interval, second_ani._interval)
    npt.assert_almost_equal(milli_ani._interval, 20)
    # ... but each labelled in its own unit:
    npt.assert_equal('t = 40.00 ms' in milli_ani.to_jshtml(), True)
    npt.assert_equal('t = 0.04 s' in second_ani.to_jshtml(), True)
    # An explicit `fps` still wins over both:
    fixed = second.play(fps=50)
    npt.assert_almost_equal(fixed._interval, 20)

    # `save`: same frame rate, so the movies run for the same length of time.
    seen = []
    monkeypatch.setattr(imageio, 'mimwrite',
                        lambda fname, data, **kwargs: seen.append(kwargs))
    for percept in (milli, second):
        percept.save(str(tmp_path / 'test.mp4'), vmin=0, vmax=1)
    npt.assert_equal(len(seen), 2)
    npt.assert_almost_equal(seen[0]['fps'], 50)
    npt.assert_almost_equal(seen[1]['fps'], 50)

    # A percept whose time axis is in seconds is not a ragged one either: an
    # irregular axis plays at the speed it was recorded at, and that speed is
    # counted in milliseconds.
    ragged = Percept(data, time=[0, 0.02, 0.05], time_unit=s)
    npt.assert_almost_equal(player(ragged.play())['intervals'], [20, 30, 30])
    # A movie file runs at a single frame rate, so this one needs an `fps`:
    with pytest.raises(NotImplementedError):
        ragged.save(str(tmp_path / 'ragged.mp4'), vmin=0, vmax=1)


def pulse_train_percept(n_pulses=3, period=1000.0 / 6):
    """A percept whose time axis is as ragged as a pulse train's

    Two 0.45 ms phases and a 0.1 ms interphase gap every ``period`` ms, so the
    time steps span three orders of magnitude. Each pulse lights up one frame,
    which is what makes a dropped frame visible.
    """
    time, bright = [], []
    for i in range(n_pulses):
        onset = i * period
        time += [onset, onset + 0.45, onset + 0.55, onset + 1.0]
        bright += [0, 1, 0, 0]
    data = np.zeros((4, 4, len(time)))
    data[..., :] = np.asarray(bright)
    return Percept(data, time=time)


@pytest.mark.parametrize('fps', (15, 30, 60))
def test_Percept_play_fps_is_display_rate(fps):
    """Changing fps resamples without changing playback duration."""
    # One second of percept, sampled at 100 Hz:
    percept = Percept(np.random.rand(4, 4, 100), time=np.arange(0, 1000, 10))
    cfg = player(percept.play(fps=fps))
    # One display frame per 1/fps of a second ...
    npt.assert_equal(cfg['n'], fps)
    npt.assert_almost_equal(cfg['interval'], 1000.0 / fps)
    # ... and still one second of animation:
    npt.assert_almost_equal(np.sum(cfg['intervals']), 1000.0, decimal=6)
    # The percept's own rate keeps every frame, and takes just as long:
    native = player(percept.play())
    npt.assert_equal(native['n'], 100)
    npt.assert_almost_equal(np.sum(native['intervals']), 1000.0, decimal=6)


def test_Percept_play_zero_order_hold():
    """Display resampling uses zero-order hold"""
    data = np.zeros((2, 2, 4))
    data[..., :] = [0.0, 0.25, 0.5, 1.0]
    # 40 ms of percept: four frames of 10 ms each.
    percept = Percept(data, time=[0, 10, 20, 30])
    # 50 fps samples it at t = 0 and 20 ms ...
    ani = percept.play(fps=50)
    npt.assert_equal(ani._frame_data.shape[-1], 2)
    npt.assert_almost_equal(ani._frame_data[0, 0], [0.0, 0.5])
    # ... 100 fps lands on every frame, and 200 fps holds each one for two
    # samples:
    npt.assert_almost_equal(percept.play(fps=100)._frame_data[0, 0],
                            [0.0, 0.25, 0.5, 1.0])
    npt.assert_almost_equal(percept.play(fps=200)._frame_data[0, 0],
                            [0, 0, 0.25, 0.25, 0.5, 0.5, 1.0, 1.0])
    # A display sample that falls between two percept frames shows the earlier
    # one, never an average of the two: at 25 ms per display frame, t = 25 ms
    # shows the frame from t = 20 ms (0.5), not the 0.75 that averaging it
    # with the frame from t = 30 ms would give:
    held = percept.play(fps=40)._frame_data[0, 0]
    npt.assert_almost_equal(held, [0.0, 0.5])
    # The label follows the frame that is held, not the display clock:
    npt.assert_equal(player(percept.play(fps=50))['labels'],
                     ['t = 0.00 ms', 't = 20.00 ms'])


def test_Percept_play_irregular_time():
    """Irregular percept times produce unequal frame durations."""
    period = 1000.0 / 6
    percept = pulse_train_percept(n_pulses=3, period=period)
    cfg = player(percept.play())
    # Every frame is kept ...
    npt.assert_equal(cfg['n'], percept.time.size)
    # ... each shown for as long as the axis says, 165 ms gaps included. The
    # last frame is held for the interval in front of it, so it is seen:
    pulse = [0.45, 0.1, 0.45]
    steps = pulse + [period - 1.0]
    npt.assert_almost_equal(cfg['intervals'], steps * 2 + pulse + [0.45],
                            decimal=6)
    # ... which adds up to the wall-clock time the percept covers:
    npt.assert_almost_equal(np.sum(cfg['intervals']),
                            percept.time[-1] - percept.time[0] + 0.45,
                            decimal=6)


def test_Percept_play_irregular_time_fps():
    """Irregular percepts can be resampled onto a regular clock."""
    percept = pulse_train_percept(n_pulses=3, period=1000.0 / 6)
    step = 1000.0 / 60
    cfg = player(percept.play(fps=60))
    # The percept ends with its last pulse, 334.33 ms in, so 60 fps buys 20
    # frames of equal length, however ragged the percept's own axis is:
    npt.assert_equal(cfg['n'], 20)
    npt.assert_almost_equal(cfg['intervals'], [step] * 20)
    # ... covering the percept's own duration, to within one display frame:
    duration = percept.time[-1] - percept.time[0] + 0.45
    npt.assert_array_less(abs(np.sum(cfg['intervals']) - duration), step)


def test_Percept_play_brief_events_are_missed():
    """Events between display samples are not interpolated."""
    percept = pulse_train_percept()
    brightest = percept.data.max()
    # At its own rate, every pulse is on screen:
    npt.assert_almost_equal(percept.play()._frame_data.max(), brightest)
    # A 0.45 ms pulse every 166.67 ms almost never coincides with a display
    # sample, so the pulses are simply not seen:
    frames = percept.play(fps=30)._frame_data
    npt.assert_equal(frames.shape[-1], 10)
    npt.assert_almost_equal(frames.max(), 0)
    # No interpolation either: every display frame is one percept frame,
    # copied verbatim:
    values = np.unique(percept.play(fps=1000)._frame_data)
    npt.assert_equal(np.isin(values, np.unique(percept.data)).all(), True)


def test_Percept_save_fps_resamples(tmp_path, monkeypatch):
    """Export fps changes frame count, not movie duration."""
    seen = []
    monkeypatch.setattr(imageio, 'mimwrite',
                        lambda fname, data, **kwargs: seen.append((data,
                                                                   kwargs)))
    # One second of percept, sampled at 100 Hz:
    percept = Percept(np.random.rand(16, 16, 100), time=np.arange(0, 1000, 10))
    for fps in (15, 30, 60, None):
        percept.save(str(tmp_path / 'test.mp4'), fps=fps, vmin=0, vmax=1)
    for (data, kwargs), fps in zip(seen, (15, 30, 60, 100)):
        n_frames = fps if fps != 100 else 100
        npt.assert_equal(len(data), n_frames)
        npt.assert_almost_equal(kwargs['fps'], fps)
        # Frame count over frame rate is one second of movie, every time:
        npt.assert_almost_equal(len(data) / kwargs['fps'], 1.0, decimal=6)

    # An irregular percept is written out the same way. Its final frame uses
    # the same preceding-interval display convention as play():
    seen.clear()
    percept = pulse_train_percept(n_pulses=3, period=1000.0 / 6)
    duration = (percept.time[-1] - percept.time[0] + 0.45) / 1000.0
    percept.save(str(tmp_path / 'pulses.mp4'), fps=30, vmin=0, vmax=1)
    npt.assert_equal(len(seen[0][0]), 10)
    npt.assert_array_less(abs(len(seen[0][0]) / 30.0 - duration), 1 / 30.0)


def test_Percept_play_keeps_the_last_frame():
    """Hold the final frame for the preceding interval."""
    data = np.zeros((2, 2, 3))
    data[..., :] = [0.0, 0.5, 1.0]
    percept = Percept(data, time=[0, 20, 50])
    cfg = player(percept.play())
    npt.assert_equal(cfg['n'], 3)
    npt.assert_equal(cfg['labels'][-1], 't = 50.00 ms')
    # The last frame is on screen for as long as the interval in front of it:
    npt.assert_almost_equal(cfg['intervals'], [20, 30, 30])
    npt.assert_almost_equal(percept.play()._frame_data[0, 0], [0, 0.5, 1.0])
    # ... and a display clock fine enough to resolve it reaches it, which a
    # zero-length last frame would not allow at any rate:
    for fps in (40, 100, 1000):
        frames = percept.play(fps=fps)._frame_data[0, 0]
        npt.assert_almost_equal(frames[-1], 1.0)
        npt.assert_equal(player(percept.play(fps=fps))['labels'][-1],
                         't = 50.00 ms')
    # A clock too coarse to resolve it misses it, like any other frame: at
    # 25 fps the 80 ms percept is sampled at 0 and 40 ms only.
    npt.assert_almost_equal(percept.play(fps=25)._frame_data[0, 0], [0, 0.5])


def test_Percept_play_rejects_unordered_time(tmp_path):
    """Playback requires strictly increasing time points."""
    percept = Percept(np.random.rand(2, 2, 3), time=[0, 30, 10])
    for fps in (None, 30):
        with pytest.raises(ValueError):
            percept.play(fps=fps)
    with pytest.raises(ValueError):
        percept.save(str(tmp_path / 'test.mp4'), fps=30, vmin=0, vmax=1)


def test_Percept_play_save_do_not_mutate(tmp_path):
    """Displaying a percept must leave the percept alone"""
    percept = pulse_train_percept()
    data, time = percept.data.copy(), percept.time.copy()
    for fps in (None, 10, 1000):
        percept.play(fps=fps).to_jshtml()
    percept.save(str(tmp_path / 'test.mp4'), fps=30, vmin=0, vmax=1)
    npt.assert_almost_equal(percept.data, data)
    npt.assert_almost_equal(percept.time, time)


def test_Percept_play_units_equivalent():
    """The same timeline in another unit is the same animation"""
    data = np.random.rand(4, 4, 4)
    milli = Percept(data, time=[0, 0.45, 0.55, 166.67])
    second = Percept(data, time=np.asarray([0, 0.45, 0.55, 166.67]) / 1000,
                     time_unit=s)
    for fps in (None, 6, 30):
        milli_cfg, second_cfg = (player(p.play(fps=fps))
                                 for p in (milli, second))
        npt.assert_equal(milli_cfg['n'], second_cfg['n'])
        npt.assert_almost_equal(milli_cfg['intervals'],
                                second_cfg['intervals'], decimal=6)


def test_Percept_getitem_time():
    """A number on the time axis is a time, not a frame index"""
    data = np.arange(24, dtype=float).reshape((2, 3, 4))
    percept = Percept(data, time=[0.0, 10.0, 20.0, 30.0])
    # One time point drops the time axis, the way a scalar index does on any
    # other axis ...
    npt.assert_equal(percept[..., 10.0].shape, (2, 3))
    npt.assert_equal(percept[:, 0, 10.0].shape, (2,))
    npt.assert_equal(np.isscalar(percept[0, 1, 10.0]), True)
    # ... and a stored time point comes back verbatim:
    npt.assert_almost_equal(percept[..., 10.0], data[..., 1])
    # One in between is interpolated between its neighbors:
    npt.assert_almost_equal(percept[..., 5.0],
                            (data[..., 0] + data[..., 1]) / 2)
    npt.assert_almost_equal(percept[0, 1, 5.0],
                            (data[0, 1, 0] + data[0, 1, 1]) / 2)
    # Beyond the ends, the closest stored frame is held:
    npt.assert_almost_equal(percept[..., -5.0], data[..., 0])
    npt.assert_almost_equal(percept[..., 99.0], data[..., -1])
    # Space is indexed the NumPy way, and an index that stops short of the
    # time axis returns the whole time series:
    npt.assert_almost_equal(percept[0, 1], data[0, 1])
    npt.assert_almost_equal(percept[0], data[0])
    # A float64 percept is not interpolated down to float32:
    npt.assert_equal(percept[..., 5.0].dtype, np.float64)


def test_Percept_getitem_multiple_times():
    """Lists, slices and masks of time points"""
    data = np.arange(24, dtype=float).reshape((2, 3, 4))
    percept = Percept(data, time=[0.0, 10.0, 20.0, 30.0])
    # A list of time points is interpolated onto:
    npt.assert_almost_equal(percept[..., [5.0, 15.0]],
                            np.stack([(data[..., 0] + data[..., 1]) / 2,
                                      (data[..., 1] + data[..., 2]) / 2],
                                     axis=-1))
    # One requested time point is still a time axis when asked for as a list:
    npt.assert_equal(percept[..., [5.0]].shape, (2, 3, 1))
    npt.assert_equal(percept[0, 0, [5.0]].shape, (1,))
    # A stepped slice is a time range, not a range of frame indices:
    npt.assert_equal(percept[..., 0:30:5].shape, (2, 3, 6))
    npt.assert_almost_equal(percept[0, 0, 0:30:10], data[0, 0, :3])
    # A stepless slice takes the stored frames by position:
    npt.assert_almost_equal(percept[..., :], data)
    with pytest.raises(ValueError):
        percept[..., 0:20]
    # A boolean mask selects stored frames without interpolating:
    npt.assert_almost_equal(percept[..., percept.time < 20], data[..., :2])
    # One of the wrong length is a shape error. It must not be read as the
    # times t=1 and t=0, which is what its True and False would interpolate to:
    with pytest.raises(IndexError):
        percept[..., np.array([True, False])]


def test_Percept_getitem_irregular_time():
    """Interpolation follows the recorded times, however unevenly spaced"""
    data = np.arange(8, dtype=float).reshape((2, 1, 4))
    percept = Percept(data, time=[0.0, 1.0, 10.0, 100.0])
    npt.assert_almost_equal(
        percept[0, 0, 5.5],
        data[0, 0, 1] + 0.5 * (data[0, 0, 2] - data[0, 0, 1]))
    npt.assert_almost_equal(percept[0, 0, 1.0], data[0, 0, 1])


def test_Percept_getitem_units():
    """A time point may be bare or unitful"""
    data = np.arange(24, dtype=float).reshape((2, 3, 4))
    percept = Percept(data, time=[0.0, 10.0, 20.0, 30.0])
    npt.assert_almost_equal(percept[..., 15 * ms], percept[..., 15.0])
    npt.assert_almost_equal(percept[..., 0.015 * s], percept[..., 15.0])
    # A percept that counts seconds reads bare numbers as seconds:
    in_s = Percept(data, time=[0.0, 0.01, 0.02, 0.03], time_unit=s)
    npt.assert_almost_equal(in_s[..., 0.015], percept[..., 15.0])
    npt.assert_almost_equal(in_s[..., 15 * ms], percept[..., 15.0])
    with pytest.raises(DimensionMismatchError):
        percept[..., 15 * uA]


def test_Percept_getitem_no_time():
    """Without a time axis, indexing is ordinary NumPy indexing"""
    data = np.arange(6, dtype=float).reshape((2, 3, 1))
    percept = Percept(data)
    npt.assert_equal(percept.time, None)
    npt.assert_almost_equal(percept[..., 0], data[..., 0])
    npt.assert_almost_equal(percept[0, 1, 0], data[0, 1, 0])
    npt.assert_almost_equal(percept[:, :, 0:1], data[:, :, 0:1])
    with pytest.raises(IndexError):
        percept[..., 1.5]
    # A time axis that was filled in automatically is still a time axis:
    auto = Percept(np.arange(24, dtype=float).reshape((2, 3, 4)))
    npt.assert_almost_equal(auto.time, [0, 1, 2, 3])
    npt.assert_almost_equal(auto[0, 0, 1.5], 1.5)


def test_Percept_play_clim():
    """Explicit limits set the color scale and leave the timeline alone"""
    percept = Percept(np.random.rand(4, 4, 10) * 20,
                      time=np.arange(10) * 10.0)
    auto = percept.play()
    npt.assert_almost_equal(auto._image.get_clim(), (0, percept.data.max()))
    fixed = percept.play(vmin=-1, vmax=50)
    npt.assert_almost_equal(fixed._image.get_clim(), (-1, 50))
    npt.assert_almost_equal(player(fixed)['intervals'],
                            player(auto)['intervals'])
    npt.assert_equal(fixed._frame_data.shape, auto._frame_data.shape)
    with pytest.raises(ValueError):
        percept.play(vmin=1, vmax=0)


def test_Percept_play_clim_ignores_fps():
    """The color scale spans the percept, not the frames the display samples"""
    data = np.zeros((4, 4, 10))
    data[..., 1] = 20.0
    percept = Percept(data, time=np.arange(10) * 10.0)
    # 20 fps samples t = 0 and 50 ms, missing the flash at t = 10 ms ...
    npt.assert_equal(percept.play(fps=20)._frame_data.max(), 0)
    # ... but the brightness scale still knows about it:
    npt.assert_almost_equal(percept.play(fps=20)._image.get_clim(), (0, 20))
    npt.assert_almost_equal(percept.play()._image.get_clim(), (0, 20))


def test_Percept_save_common_clim(tmp_path):
    """Two percepts saved on a common range come back on a common scale"""
    dim = Percept(np.linspace(0, 5, 256).reshape((16, 16, 1)))
    bright = Percept(np.linspace(0, 20, 256).reshape((16, 16, 1)))
    for percept, name in ((dim, 'dim.png'), (bright, 'bright.png')):
        percept.save(str(tmp_path / name), shape=(16, 16), vmin=0, vmax=20)
        loaded = Percept.load(str(tmp_path / name))
        # 8 bits of gray spread over a range of 20:
        npt.assert_allclose(loaded.data, percept.data, atol=20 / 255)
    # The dim percept does not get stretched to fill the gray levels:
    npt.assert_equal(imread(str(tmp_path / 'dim.png')).max() < 255, True)
    npt.assert_equal(imread(str(tmp_path / 'bright.png')).max(), 255)


def test_Percept_save_clim_edge_cases(tmp_path):
    """Clipping, negative values, constant data, and nonsensical ranges"""
    percept = Percept(np.linspace(-5, 5, 256).reshape((16, 16, 1)))
    # Negative values are just the bottom of the range:
    fname = str(tmp_path / 'signed.png')
    percept.save(fname, shape=(16, 16), vmin=-5, vmax=5)
    npt.assert_allclose([Percept.load(fname).data.min(),
                         Percept.load(fname).data.max()], [-5, 5], atol=0.05)
    # Everything below vmin is clipped to it:
    fname = str(tmp_path / 'clipped.png')
    percept.save(fname, shape=(16, 16), vmin=0, vmax=5)
    npt.assert_almost_equal(Percept.load(fname).data.min(), 0)
    npt.assert_equal(np.mean(Percept.load(fname).data == 0) > 0.4, True)
    # A constant percept has no range to stretch, and does not divide by zero:
    fname = str(tmp_path / 'constant.png')
    with pytest.warns(UserWarning):
        Percept(np.full((16, 16, 1), 3.0)).save(fname, shape=(16, 16))
    npt.assert_almost_equal(Percept.load(fname).data, 3.0)
    with pytest.raises(ValueError):
        percept.save(str(tmp_path / 'bad.png'), vmin=5, vmax=0)


def test_Percept_save_clim_ignores_fps(tmp_path):
    """Export rate cannot change how bright the movie comes out"""
    data = np.zeros((16, 16, 10))
    data[..., :] = np.linspace(0, 1, 10)
    # A flash that only the faster export rate samples:
    data[..., 1] = 10.0
    percept = Percept(data, time=np.arange(10) * 10.0)
    with pytest.warns(UserWarning):
        percept.save(str(tmp_path / 'slow.gif'), shape=(16, 16), fps=20)
    with pytest.warns(UserWarning):
        percept.save(str(tmp_path / 'fast.gif'), shape=(16, 16), fps=100)
    slow, fast = (mimread(str(tmp_path / n)) for n in ('slow.gif', 'fast.gif'))
    npt.assert_equal((len(slow), len(fast)), (2, 10))
    # Same percept frame, same gray levels:
    npt.assert_array_equal(slow[1], fast[5])


def test_Percept_save_warns_without_clim(tmp_path):
    """Automatic normalization is not a shared scale, and says so"""
    percept = Percept(np.random.rand(16, 16, 1))
    with pytest.warns(UserWarning, match='Pass'):
        percept.save(str(tmp_path / 'auto.png'), shape=(16, 16))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        percept.save(str(tmp_path / 'fixed.png'), shape=(16, 16), vmin=0,
                     vmax=1)


def test_Percept_load_image(tmp_path):
    """A static image is a single-frame percept with no time axis"""
    fname = str(tmp_path / 'p.png')
    percept = Percept(np.linspace(0, 20, 256).reshape((16, 16, 1)))
    percept.save(fname, shape=(16, 16), vmin=0, vmax=20)
    loaded = Percept.load(fname)
    npt.assert_equal(loaded.shape, (16, 16, 1))
    npt.assert_equal(loaded.time, None)
    npt.assert_allclose(loaded.data, percept.data, atol=20 / 255)
    # A media file does not record where the pixels sit:
    npt.assert_almost_equal(loaded.xdva, np.arange(16))
    grid = Grid2D((-1, 1), (-1, 1), step=2 / 15)
    npt.assert_almost_equal(Percept.load(fname, space=grid).xdva, grid._xflat)


@pytest.mark.parametrize('ext', ('.gif', '.mp4'))
def test_Percept_load_video(ext, tmp_path):
    """A GIF or movie is a multi-frame percept, timed by its frame rate"""
    fname = str(tmp_path / f'p{ext}')
    data = np.zeros((16, 16, 5))
    data[..., :] = np.linspace(0, 20, 5)
    percept = Percept(data, time=np.arange(5) * 100.0)
    percept.save(fname, shape=(32, 32), vmin=0, vmax=20)
    loaded = Percept.load(fname, vmin=0, vmax=20)
    npt.assert_equal(loaded.shape[-1], 5)
    npt.assert_almost_equal(loaded.time, percept.time)
    # Quantization and, for a movie, the codec cost a fraction of a gray level:
    npt.assert_allclose([loaded.data[..., i].mean() for i in range(5)],
                        np.linspace(0, 20, 5), atol=0.5)


def test_Percept_load_timing(tmp_path):
    """Explicit timing overrides what the file records"""
    fname = str(tmp_path / 'p.gif')
    percept = Percept(np.random.rand(16, 16, 4), time=np.arange(4) * 100.0)
    percept.save(fname, shape=(16, 16), vmin=0, vmax=1)
    npt.assert_almost_equal(Percept.load(fname).time, [0, 100, 200, 300])
    npt.assert_almost_equal(Percept.load(fname, fps=50).time, [0, 20, 40, 60])
    npt.assert_almost_equal(Percept.load(fname, fps=50 * Hz).time,
                            [0, 20, 40, 60])
    npt.assert_almost_equal(Percept.load(fname, time=[0, 1, 2, 3]).time,
                            [0, 1, 2, 3])
    npt.assert_almost_equal(
        Percept.load(fname, time=np.arange(4) / 1000 * s).time, [0, 1, 2, 3])


def test_Percept_load_variable_frame_durations(tmp_path):
    """A GIF that holds a different duration per frame has no frame rate"""
    fname = str(tmp_path / 'ragged.gif')
    frames = [np.full((16, 16), level, dtype=np.uint8) for level in range(4)]
    imageio.mimwrite(fname, frames, duration=[100, 300, 50, 200])
    with pytest.raises(ValueError):
        Percept.load(fname)
    # Saying when the frames happen is what the error asks for:
    with pytest.warns(UserWarning):
        loaded = Percept.load(fname, time=[0, 100, 400, 450])
    npt.assert_almost_equal(loaded.time, [0, 100, 400, 450])


@pytest.mark.parametrize('fps', (0, -30, np.nan, np.inf))
def test_Percept_load_rejects_bad_fps(fps, tmp_path):
    """A frame rate that is not a finite positive number is not a frame rate

    ``nan`` would give the percept a nan time axis and ``inf`` an all-zero
    one, neither of which announces itself later.
    """
    fname = str(tmp_path / 'p.gif')
    percept = Percept(np.random.rand(16, 16, 4), time=np.arange(4) * 100.0)
    percept.save(fname, shape=(16, 16), vmin=0, vmax=1)
    with pytest.raises(ValueError):
        Percept.load(fname, fps=fps)


def test_Percept_load_grayscale(tmp_path):
    """Color input is reduced to one brightness per pixel"""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    imageio.imwrite(str(tmp_path / 'rgb.png'), rgb)
    with pytest.warns(UserWarning):
        loaded = Percept.load(str(tmp_path / 'rgb.png'))
    npt.assert_equal(loaded.shape, (8, 8, 1))
    # The luminance of pure red:
    npt.assert_allclose(loaded.data, 0.2125, atol=1e-3)
    # Alpha is blended against black, as elsewhere in p2p:
    rgba = np.full((8, 8, 4), 255, dtype=np.uint8)
    rgba[..., 3] = 128
    imageio.imwrite(str(tmp_path / 'rgba.png'), rgba)
    with pytest.warns(UserWarning):
        loaded = Percept.load(str(tmp_path / 'rgba.png'))
    npt.assert_allclose(loaded.data, 128 / 255, atol=0.01)


def test_Percept_load_range_precedence(tmp_path):
    """Explicit limits beat file metadata, which beats the file name"""
    percept = Percept(np.linspace(0, 20, 256).reshape((16, 16, 1)))
    # Metadata says [0, 20] and the file name says [0, 5]:
    fname = str(tmp_path / 'p__p2p_vmin=0.0_vmax=5.0.png')
    percept.save(fname, shape=(16, 16), vmin=0, vmax=20)
    npt.assert_allclose(Percept.load(fname).data.max(), 20, atol=0.1)
    npt.assert_allclose(Percept.load(fname, vmax=100).data.max(), 100,
                        atol=0.5)
    # A BMP has nowhere to record the range, so the file name is what is left:
    named = str(tmp_path / 'q__p2p_vmin=0.0_vmax=5.0.bmp')
    percept.save(named, shape=(16, 16), vmin=0, vmax=20)
    npt.assert_allclose(Percept.load(named).data.max(), 5, atol=0.05)


def test_Percept_load_unknown_range_warns(tmp_path):
    """An unrecoverable range leaves the data normalized, and says so"""
    fname = str(tmp_path / 'plain.bmp')
    percept = Percept(np.linspace(0, 20, 256).reshape((16, 16, 1)))
    percept.save(fname, shape=(16, 16), vmin=0, vmax=20)
    with pytest.warns(UserWarning, match='brightness range'):
        loaded = Percept.load(fname)
    npt.assert_almost_equal([loaded.data.min(), loaded.data.max()], [0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        recovered = Percept.load(fname, vmin=0, vmax=20)
    npt.assert_allclose(recovered.data, percept.data, atol=20 / 255)
