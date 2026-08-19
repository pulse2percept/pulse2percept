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
        percept.save(fname)
        npt.assert_equal(os.path.isfile(fname), True)
        # Normalized to [0, 255] with some loss of precision:
        for mov in mimread(fname):
            npt.assert_equal(np.min(mov) <= 10, True)
            npt.assert_equal(np.max(mov) >= 240, True)

    # Cannot save multiple frames image:
    fname = str(tmp_path / 'test.jpg')
    with pytest.raises(ValueError):
        percept.save(fname)

    # But, can save single frame as image:
    percept = Percept(ndarray[..., :1])
    for name in ['test.jpg', 'test.png', 'test.tif', 'test.gif']:
        fname = str(tmp_path / name)
        percept.save(fname)
        npt.assert_equal(os.path.isfile(fname), True)
        img = img_as_float(imread(fname))
        npt.assert_almost_equal(np.min(img), 0, decimal=3)
        npt.assert_almost_equal(np.max(img), 1.0, decimal=3)


def test_Percept_save_single_frame(tmp_path):
    """A percept with a single time point has no frame rate of its own"""
    percept = Percept(np.random.rand(16, 16, 1), time=[3.5])
    for name in ['test.mp4', 'test.avi', 'test.gif']:
        fname = str(tmp_path / name)
        percept.save(fname)
        npt.assert_equal(len(mimread(fname)), 1)
    # An explicit frame rate is still honored:
    fname = str(tmp_path / 'fps.mp4')
    percept.save(fname, fps=12)
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
    percept.save(fname, fps=30 * Hz)
    npt.assert_equal(len(mimread(fname)), 30)
    percept.save(fname, fps=0.03 * kHz)
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
        percept.save(str(tmp_path / 'test.mp4'))
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
        ragged.save(str(tmp_path / 'ragged.mp4'))


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
    """`fps` buys frames, not speed: the time axis owns the duration

    Regression test: `fps` used to be handed to the player as the delay
    between two frames, so asking for a higher frame rate played the same
    frames faster instead of sampling them more finely.
    """
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
    """A display frame shows the most recent percept frame, unblended"""
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
    """An irregular time axis plays with its own unequal time steps

    Regression test: a pulse train's time axis used to be rejected outright,
    because the player could only be given a single frame delay.
    """
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
    """The same irregular percept, on a regular display clock"""
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
    """A pulse between two display samples is dropped, not pooled

    Sampling faster is the fix; smearing the pulse across the frames around it
    would put brightness on the screen at a time the model never computed it.
    """
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
    """Changing the export rate changes the frame count, not the duration

    Regression test: `fps` used to be passed straight to the writer while
    every percept frame was written out, so a higher rate produced a shorter
    movie of the same frames.
    """
    seen = []
    monkeypatch.setattr(imageio, 'mimwrite',
                        lambda fname, data, **kwargs: seen.append((data,
                                                                   kwargs)))
    # One second of percept, sampled at 100 Hz:
    percept = Percept(np.random.rand(16, 16, 100), time=np.arange(0, 1000, 10))
    for fps in (15, 30, 60, None):
        percept.save(str(tmp_path / 'test.mp4'), fps=fps)
    for (data, kwargs), fps in zip(seen, (15, 30, 60, 100)):
        n_frames = fps if fps != 100 else 100
        npt.assert_equal(len(data), n_frames)
        npt.assert_almost_equal(kwargs['fps'], fps)
        # Frame count over frame rate is one second of movie, every time:
        npt.assert_almost_equal(len(data) / kwargs['fps'], 1.0, decimal=6)

    # An irregular percept is written out the same way. Its last time point
    # ends it, so the movie does not run past the pulse train it shows:
    seen.clear()
    percept = pulse_train_percept(n_pulses=3, period=1000.0 / 6)
    duration = (percept.time[-1] - percept.time[0] + 0.45) / 1000.0
    percept.save(str(tmp_path / 'pulses.mp4'), fps=30)
    npt.assert_equal(len(seen[0][0]), 10)
    npt.assert_array_less(abs(len(seen[0][0]) / 30.0 - duration), 1 / 30.0)


def test_Percept_play_keeps_the_last_frame():
    """The last time point of an irregular percept is model output

    `TemporalModel.predict_percept(t_percept=[0, 20, 50])` asks for brightness
    at those three instants, so the frame at 50 ms is a percept like any
    other, not an endpoint that ends the animation. Giving it no duration
    would drop it from playback entirely.
    """
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
    """A percept whose time axis backtracks cannot be laid out on a clock"""
    percept = Percept(np.random.rand(2, 2, 3), time=[0, 30, 10])
    for fps in (None, 30):
        with pytest.raises(ValueError):
            percept.play(fps=fps)
    with pytest.raises(ValueError):
        percept.save(str(tmp_path / 'test.mp4'), fps=30)


def test_Percept_play_save_do_not_mutate(tmp_path):
    """Displaying a percept must leave the percept alone"""
    percept = pulse_train_percept()
    data, time = percept.data.copy(), percept.time.copy()
    for fps in (None, 10, 1000):
        percept.play(fps=fps).to_jshtml()
    percept.save(str(tmp_path / 'test.mp4'), fps=30)
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
