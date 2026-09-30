from pulse2percept.topography import Grid2D
from pulse2percept.percepts import Percept
from pulse2percept.units import (DimensionMismatchError, Hz, kHz, ms, s, uA,
                                 um, us)
from skimage.io import imread
from skimage import img_as_float
import imageio
from imageio import mimread
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg
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
    """Return the JavaScript player config"""
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


def grid_lines(ax, linestyle):
    """Return drawn rings ('--') or meridians ('-') as (2, n) arrays"""
    return [np.asarray(line.get_data()) for line in ax.get_lines()
            if line.get_linestyle() == linestyle]


def test_Percept_plot_grid_is_centered_on_the_visual_field_origin():
    """Grid is centered at (0, 0), not at the center of an asymmetric field"""
    grid = Grid2D((-15, 5), (-4, 10), step=0.5)
    percept = Percept(np.random.rand(*grid.x.shape, 1), space=grid)
    ax = percept.plot(ax=plt.subplots()[1], rings=True,
                      meridians=[0, 90, 180, 270])
    # Automatic rings stop at the nearest edge (y = -4):
    radii = [np.hypot(*ring).mean() for ring in grid_lines(ax, '--')]
    npt.assert_almost_equal(radii, [1.25, 2.5])
    npt.assert_equal([t.get_text() for t in ax.texts],
                     ['1.25\N{DEGREE SIGN}', '2.5\N{DEGREE SIGN}'])
    # 0 deg is +x, 90 deg is +y; each runs from the fovea to the field edge:
    ends = [line[:, [0, -1]].T for line in grid_lines(ax, '-')]
    npt.assert_almost_equal(ends, [[(0, 0), (5, 0)], [(0, 0), (0, 10)],
                                   [(0, 0), (-15, 0)], [(0, 0), (0, -4)]])
    # Annotation only:
    npt.assert_equal(len(ax.collections), 1)
    plt.close('all')


def test_Percept_plot_grid_on_a_field_that_excludes_the_fovea():
    grid = Grid2D((-15, -3), (-2, 10), step=0.5)
    percept = Percept(np.random.rand(*grid.x.shape, 1), space=grid)
    ax = percept.plot(ax=plt.subplots()[1], rings=True, meridians=True)
    # Rings span the eccentricities the field covers (3 to ~18 dva):
    radii = [np.hypot(*ring).mean() for ring in grid_lines(ax, '--')]
    npt.assert_almost_equal(radii, [5, 10])
    # Only meridians that cross the field are drawn, from where they enter:
    meridians = grid_lines(ax, '-')
    angles = [np.rad2deg(np.arctan2(m[1, -1], m[0, -1])) % 360
              for m in meridians]
    npt.assert_almost_equal(angles, [135, 180])
    for m in meridians:
        npt.assert_equal(np.all(m[0] <= -3 + 1e-9), True)
    plt.close('all')


def test_Percept_grid_needs_visual_field_coordinates():
    """Rings and meridians require a percept built with space"""
    bare = Percept(np.random.rand(3, 5, 2), time=[0, 10])
    temporal = Percept(np.random.rand(1, 1, 4), time=[0, 1, 2, 3])
    for percept in (bare, temporal):
        for grid in ({'rings': True}, {'meridians': True}):
            with pytest.raises(ValueError):
                percept.plot(ax=plt.subplots()[1], **grid)
            with pytest.raises(ValueError):
                percept.play(**grid)
    # No grid requested:
    ax = bare.plot(ax=plt.subplots()[1], rings=False, meridians=None)
    npt.assert_equal(len(ax.lines) + len(ax.texts), 0)
    npt.assert_equal(len(bare.play(rings=None)._layers), 1)
    plt.close('all')


def test_Percept_play_shows_the_grid_as_a_still_layer():
    """play draws the grid as a static image layer"""
    grid = Grid2D((-4, 4), (-2, 2), step=0.5)
    percept = Percept(np.random.rand(*grid.x.shape, 3), space=grid,
                      time=[0, 10, 20])
    plain = percept.play()
    ruled = percept.play(meridians=[0], grid_color='red')
    npt.assert_equal(len(ruled._layers), 2)
    frames, overlay = ruled._layers
    npt.assert_equal(frames.data is percept.data, True)
    npt.assert_array_equal(overlay.index, 0)
    npt.assert_equal(overlay.data.shape[2:], (4, 1))
    # 0 deg runs along the middle row from the fovea to the right edge:
    alpha = overlay.data[..., 3, 0]
    n_rows, n_cols = alpha.shape
    middle = alpha[n_rows // 2 - 2:n_rows // 2 + 2]
    npt.assert_equal(middle[:, n_cols // 2 + 2:].max(axis=0).min() > 0.2, True)
    npt.assert_almost_equal(middle[:, :n_cols // 2 - 2].max(), 0)
    npt.assert_almost_equal(alpha[:n_rows // 4].max(), 0)
    # Red, and one extra sprite sheet in the HTML player:
    npt.assert_almost_equal(overlay.data[alpha > 0.5, :3, 0].mean(0),
                            (1, 0, 0), decimal=2)
    count = 'data:image/png;base64'
    npt.assert_equal(ruled.to_jshtml().count(count),
                     plain.to_jshtml().count(count) + 1)
    plt.close('all')


@ pytest.mark.parametrize('n_frames', (2, 3, 10, 14))
def test_Percept_play(n_frames):
    ndarray = np.random.rand(2, 4, n_frames)
    percept = Percept(ndarray)
    ani = percept.play()
    npt.assert_equal(isinstance(ani, FuncAnimation), True)
    npt.assert_equal(len(list(ani.frame_seq)), n_frames)
    # Renders as a self-contained HTML player:
    html = ani.to_jshtml()
    npt.assert_equal('p2p-anim' in html, True)
    npt.assert_equal(f'"n": {n_frames}' in html, True)
    # Time is annotated in the title unless turned off:
    npt.assert_equal(f't = {percept.time[-1]:.2f} ms' in html, True)
    html = percept.play(annotate_time=False).to_jshtml()
    npt.assert_equal(f't = {percept.time[-1]:.2f} ms' in html, False)


@pytest.mark.parametrize('own_axes', (True, False))
def test_Percept_play_title_is_independent_of_time_annotation(own_axes):
    percept = Percept(np.random.rand(2, 4, 3))
    for annotate_time in (True, False):
        ax = None if own_axes else plt.subplots()[1]
        ani = percept.play(title='PRIMA', annotate_time=annotate_time, ax=ax)
        npt.assert_equal(ani._fig._suptitle.get_text(), 'PRIMA')
        npt.assert_equal(ani._labels is not None, annotate_time)
    npt.assert_equal(percept.play()._fig._suptitle, None)
    plt.close('all')


def test_Percept_omitted_vmax_is_the_whole_percepts_maximum():
    """Default vmax is the maximum over all frames"""
    data = np.zeros((3, 5, 3))
    data[..., 0] = 1
    data[1, 1, 2] = 4
    data[0, 0, 2] = -1
    percept = Percept(data, space=Grid2D((-2, 2), (-1, 1)), time=[0, 1, 2])
    npt.assert_almost_equal(percept.play()._image.get_clim(), (0, 4))
    npt.assert_almost_equal(percept.play(vmax=2)._image.get_clim(), (0, 2))
    # `plot` keeps the drawn frame's minimum as vmin:
    npt.assert_almost_equal(percept.plot().collections[0].get_clim(), (-1, 4))
    plt.close('all')
    npt.assert_almost_equal(
        percept.plot(vmin=1, vmax=2).collections[0].get_clim(), (1, 2))
    plt.close('all')


def test_Percept_play_single_frame():
    """play works for a single time point"""
    percept = Percept(np.random.rand(4, 4, 1), time=[3.5])
    html = percept.play().to_jshtml()
    npt.assert_equal('"n": 1' in html, True)
    npt.assert_equal('t = 3.50 ms' in html, True)
    # No time axis:
    with pytest.raises(ValueError):
        Percept(np.random.rand(4, 4, 1)).play()


@pytest.mark.parametrize('gridded', (True, False))
def test_Percept_play_uses_visual_field_axes(gridded):
    """play uses the same dva axes as plot"""
    grid = Grid2D((-4, 4), (-2, 2), step=1) if gridded else None
    shape = grid.x.shape if gridded else (5, 9)
    percept = Percept(np.random.rand(*shape, 3), space=grid,
                      time=[0., 10., 20.])
    played = percept.play(colorbar=False)._layers[0].image.axes
    plotted = percept.plot(ax=plt.subplots()[1])
    npt.assert_almost_equal(played.get_xlim(), plotted.get_xlim())
    npt.assert_almost_equal(played.get_ylim(), plotted.get_ylim())
    npt.assert_equal(played.get_xlabel(), 'x (degrees of visual angle)')
    npt.assert_equal(played.get_ylabel(), 'y (degrees of visual angle)')
    if gridded:
        npt.assert_almost_equal(played.get_xlim(), (-4, 4))
        npt.assert_almost_equal(played.get_ylim(), (-2, 2))
        # Extent spans pixel edges, half a step outside the centers:
        npt.assert_almost_equal(played.images[0].get_extent(),
                                (-4.5, 4.5, -2.5, 2.5))


def test_Percept_play_orientation_matches_plot():
    """play and plot draw the first data row at the same place"""
    grid = Grid2D((-4, 4), (-2, 2), step=1)
    data = np.zeros((*grid.x.shape, 1))
    data[0, 0, 0] = 1.0
    percept = Percept(data, space=grid, time=[0.])

    def brightest(ax):
        # New Agg canvas: `play` closes its figure, which leaves no renderer
        # on some Matplotlib versions:
        canvas = FigureCanvasAgg(ax.figure)
        canvas.draw()
        img = np.asarray(canvas.buffer_rgba())[..., :3].mean(-1)
        box, height = ax.get_window_extent(canvas.get_renderer()), img.shape[0]
        r0, r1 = int(height - box.y1) + 2, int(height - box.y0) - 2
        c0, c1 = int(box.x0) + 2, int(box.x1) - 2
        patch = img[r0:r1, c0:c1]
        # Centroid of lit pixels (robust to antialiasing):
        rows, cols = np.nonzero(patch >= 0.5 * patch.max())
        return ax.transData.inverted().transform(
            (c0 + cols.mean(), height - (r0 + rows.mean())))

    animation = percept.play(colorbar=False, annotate_time=False)
    animation._func(0)
    played = brightest(animation._layers[0].image.axes)
    plotted = brightest(percept.plot(ax=plt.subplots()[1]))
    npt.assert_allclose(played, plotted, atol=0.3)
    # `grid.x[0, 0], grid.y[0, 0]` is the top-left corner. Axis limits clip
    # the corner cell, so check the quadrant only:
    npt.assert_array_less(played[0], 0)
    npt.assert_array_less(0, played[1])
    npt.assert_equal((grid.x[0, 0] < 0, grid.y[0, 0] > 0), (True, True))


def test_Percept_play_fmt():
    percept = Percept(np.random.rand(8, 8, 4))
    # Default is lossless PNG (JPEG rings around high-contrast phosphenes):
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
        fname = percept.save(str(tmp_path / name), vmin=0, vmax=255)
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
        fname = percept.save(str(tmp_path / name), vmin=0, vmax=255)
        npt.assert_equal(os.path.isfile(fname), True)
        img = img_as_float(imread(fname))
        npt.assert_almost_equal(np.min(img), 0, decimal=3)
        npt.assert_almost_equal(np.max(img), 1.0, decimal=3)


def test_Percept_save_single_frame(tmp_path):
    """save works for a single time point"""
    percept = Percept(np.random.rand(16, 16, 1), time=[3.5])
    for name in ['test.mp4', 'test.avi', 'test.gif']:
        fname = percept.save(str(tmp_path / name), vmin=0, vmax=1)
        npt.assert_equal(len(mimread(fname)), 1)
    # Explicit fps:
    fname = percept.save(str(tmp_path / 'fps.mp4'), fps=12, vmin=0, vmax=1)
    npt.assert_equal(len(mimread(fname)), 1)


def test_Percept_fps_units(tmp_path):
    """fps accepts any frequency unit

    .. versionadded:: 0.10.0
    """
    # One second of percept, sampled at 100 Hz:
    percept = Percept(np.random.rand(8, 8, 100), time=np.arange(0, 1000, 10))

    def interval(**kwargs):
        """Return the HTML player frame delay (ms)"""
        html = percept.play(**kwargs).to_jshtml()
        return float(re.search(r'"interval": ([0-9.]+)', html).group(1))

    # 30 fps = 33.33 ms:
    npt.assert_almost_equal(interval(fps=30), 1000 / 30, decimal=6)
    for spelling in (30 * Hz, 0.03 * kHz):
        npt.assert_almost_equal(interval(fps=spelling), interval(fps=30),
                                decimal=12)

    # Same for save:
    fname = str(tmp_path / 'fps.mp4')
    npt.assert_equal(
        len(mimread(percept.save(fname, fps=30 * Hz, vmin=0, vmax=1))), 30)
    npt.assert_equal(
        len(mimread(percept.save(fname, fps=0.03 * kHz, vmin=0, vmax=1))), 30)

    # Non-frequency units:
    for wrong in (30 * ms, 30 * uA):
        with pytest.raises(DimensionMismatchError):
            percept.play(fps=wrong)
        with pytest.raises(DimensionMismatchError):
            percept.save(fname, fps=wrong)


def test_Percept_units():
    """Percept time units

    .. versionadded:: 0.10.0
    """
    data = np.zeros((3, 3, 2))
    # Default: ms:
    percept = Percept(data, time=[0, 10])
    npt.assert_equal(percept.time_unit, ms)
    npt.assert_almost_equal(percept.time, [0, 10])

    # A unitful time axis is converted to `time_unit`:
    percept = Percept(data, time=[0, 0.01] * s)
    npt.assert_equal(percept.time_unit, ms)
    npt.assert_allclose(percept.time, [0, 10], rtol=1e-12)
    # Also for a list of quantities:
    npt.assert_allclose(Percept(data, time=[0 * ms, 10000 * us]).time,
                        [0, 10], rtol=1e-12)

    # With another `time_unit`, bare numbers are in that unit:
    percept = Percept(data, time=[0, 0.01], time_unit=s)
    npt.assert_equal(percept.time_unit, s)
    npt.assert_allclose(percept.time, [0, 0.01], rtol=1e-12)
    npt.assert_allclose(percept.times(ms), [0, 10], rtol=1e-12)
    # `times()` without a unit returns the stored array:
    npt.assert_allclose(percept.times(), [0, 0.01], rtol=0, atol=0)
    npt.assert_equal(percept.time_quantity.unit, s)
    npt.assert_allclose(percept.time_quantity.to_value(ms), [0, 10],
                        rtol=1e-12)
    # Quantities are converted to s:
    npt.assert_allclose(Percept(data, time=[0, 10] * ms, time_unit=s).time,
                        [0, 0.01], rtol=1e-12)

    # No time axis:
    spatial = Percept(np.zeros((3, 3, 1)))
    npt.assert_equal(spatial.time, None)
    npt.assert_equal(spatial.times(s), None)
    npt.assert_equal(spatial.time_quantity, None)
    npt.assert_equal(spatial.time_unit, ms)

    # Brightness is in arbitrary units (no `unit` attribute):
    npt.assert_equal(hasattr(percept, 'unit'), False)

    # `time_unit` must be a time unit:
    with pytest.raises(TypeError):
        Percept(data, time=[0, 10], time_unit='ms')
    with pytest.raises(DimensionMismatchError):
        Percept(data, time=[0, 10], time_unit=um)
    # So must `time`:
    with pytest.raises(DimensionMismatchError):
        Percept(data, time=[0, 10] * um)


def test_Percept_animates_in_wall_clock_time(tmp_path, monkeypatch):
    """Labels use the percept's time unit; frame rate uses wall-clock time"""
    data = np.random.rand(4, 4, 3)
    milli = Percept(data, time=[0, 20, 40])
    second = Percept(data, time=[0, 0.02, 0.04], time_unit=s)

    # `play`: same frame delay:
    milli_ani, second_ani = milli.play(), second.play()
    npt.assert_almost_equal(milli_ani._interval, second_ani._interval)
    npt.assert_almost_equal(milli_ani._interval, 20)
    # Labels in each percept's unit:
    npt.assert_equal('t = 40.00 ms' in milli_ani.to_jshtml(), True)
    npt.assert_equal('t = 0.04 s' in second_ani.to_jshtml(), True)
    # Explicit `fps` overrides:
    fixed = second.play(fps=50)
    npt.assert_almost_equal(fixed._interval, 20)

    # `save`: same frame rate:
    seen = []
    monkeypatch.setattr(imageio, 'mimwrite',
                        lambda fname, data, **kwargs: seen.append(kwargs))
    for percept in (milli, second):
        percept.save(str(tmp_path / 'test.mp4'), vmin=0, vmax=1)
    npt.assert_equal(len(seen), 2)
    npt.assert_almost_equal(seen[0]['fps'], 50)
    npt.assert_almost_equal(seen[1]['fps'], 50)

    # Irregular time in s plays with intervals in ms:
    ragged = Percept(data, time=[0, 0.02, 0.05], time_unit=s)
    npt.assert_almost_equal(player(ragged.play())['intervals'], [20, 30, 30])
    # Movies require a fixed frame rate:
    with pytest.raises(NotImplementedError):
        ragged.save(str(tmp_path / 'ragged.mp4'), vmin=0, vmax=1)


def pulse_train_percept(n_pulses=3, period=1000.0 / 6):
    """Return a percept with pulse-train timing

    Two 0.45 ms phases and a 0.1 ms interphase gap every ``period`` ms. Each
    pulse lights up one frame.
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
    # One display frame per 1/fps s:
    npt.assert_equal(cfg['n'], fps)
    npt.assert_almost_equal(cfg['interval'], 1000.0 / fps)
    # Still 1 s of animation:
    npt.assert_almost_equal(np.sum(cfg['intervals']), 1000.0, decimal=6)
    # Native rate keeps every frame, same duration:
    native = player(percept.play())
    npt.assert_equal(native['n'], 100)
    npt.assert_almost_equal(np.sum(native['intervals']), 1000.0, decimal=6)


def test_Percept_play_does_not_resample_the_data():
    """Display sampling selects frames by index without copying data"""
    percept = Percept(np.random.rand(4, 4, 8), time=np.arange(8) * 10.0)
    for fps, index in [(None, np.arange(8)),
                       (200, np.repeat(np.arange(8), 2)),
                       (50, [0, 2, 4, 6])]:
        layer = percept.play(fps=fps)._layers[0]
        npt.assert_equal(layer.data is percept.data, True)
        npt.assert_equal(layer.index, index)
    # Repeated and skipped frames:
    npt.assert_almost_equal(percept.play(fps=200)._frame_data,
                            np.repeat(percept.data, 2, axis=-1))
    npt.assert_almost_equal(percept.play(fps=50)._frame_data,
                            percept.data[..., ::2])


def test_Percept_play_zero_order_hold():
    """Display resampling uses zero-order hold"""
    data = np.zeros((2, 2, 4))
    data[..., :] = [0.0, 0.25, 0.5, 1.0]
    # 40 ms of percept: four frames of 10 ms each.
    percept = Percept(data, time=[0, 10, 20, 30])
    # 50 fps samples t = 0 and 20 ms:
    ani = percept.play(fps=50)
    npt.assert_equal(ani._frame_data.shape[-1], 2)
    npt.assert_almost_equal(ani._frame_data[0, 0], [0.0, 0.5])
    # 100 fps hits every frame; 200 fps shows each frame twice:
    npt.assert_almost_equal(percept.play(fps=100)._frame_data[0, 0],
                            [0.0, 0.25, 0.5, 1.0])
    npt.assert_almost_equal(percept.play(fps=200)._frame_data[0, 0],
                            [0, 0, 0.25, 0.25, 0.5, 0.5, 1.0, 1.0])
    # A sample between frames shows the earlier frame (t = 25 ms -> 0.5):
    held = percept.play(fps=40)._frame_data[0, 0]
    npt.assert_almost_equal(held, [0.0, 0.5])
    # Labels show the held frame's time:
    npt.assert_equal(player(percept.play(fps=50))['labels'],
                     ['t = 0.00 ms', 't = 20.00 ms'])


def test_Percept_play_irregular_time():
    """Irregular percept times produce unequal frame durations."""
    period = 1000.0 / 6
    percept = pulse_train_percept(n_pulses=3, period=period)
    cfg = player(percept.play())
    # Every frame is kept:
    npt.assert_equal(cfg['n'], percept.time.size)
    # Each frame lasts until the next; the last frame lasts as long as the
    # preceding interval:
    pulse = [0.45, 0.1, 0.45]
    steps = pulse + [period - 1.0]
    npt.assert_almost_equal(cfg['intervals'], steps * 2 + pulse + [0.45],
                            decimal=6)
    # Total equals the percept duration:
    npt.assert_almost_equal(np.sum(cfg['intervals']),
                            percept.time[-1] - percept.time[0] + 0.45,
                            decimal=6)


def test_Percept_play_irregular_time_fps():
    """Irregular percepts can be resampled onto a regular clock."""
    percept = pulse_train_percept(n_pulses=3, period=1000.0 / 6)
    step = 1000.0 / 60
    cfg = player(percept.play(fps=60))
    # 334.33 ms at 60 fps gives 20 equal frames:
    npt.assert_equal(cfg['n'], 20)
    npt.assert_almost_equal(cfg['intervals'], [step] * 20)
    # Duration matches to within one display frame:
    duration = percept.time[-1] - percept.time[0] + 0.45
    npt.assert_array_less(abs(np.sum(cfg['intervals']) - duration), step)


def test_Percept_play_brief_events_are_missed():
    """Events between display samples are not interpolated."""
    percept = pulse_train_percept()
    brightest = percept.data.max()
    # At native rate, every pulse is shown:
    npt.assert_almost_equal(percept.play()._frame_data.max(), brightest)
    # 0.45 ms pulses every 166.67 ms fall between 30 fps samples:
    frames = percept.play(fps=30)._frame_data
    npt.assert_equal(frames.shape[-1], 10)
    npt.assert_almost_equal(frames.max(), 0)
    # No interpolation: display frames are percept frames:
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
        # 1 s of movie:
        npt.assert_almost_equal(len(data) / kwargs['fps'], 1.0, decimal=6)

    # Irregular percept; the last frame uses the preceding interval, as in
    # play():
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
    # Last frame lasts as long as the preceding interval:
    npt.assert_almost_equal(cfg['intervals'], [20, 30, 30])
    npt.assert_almost_equal(percept.play()._frame_data[0, 0], [0, 0.5, 1.0])
    # A fine enough display clock reaches the last frame:
    for fps in (40, 100, 1000):
        frames = percept.play(fps=fps)._frame_data[0, 0]
        npt.assert_almost_equal(frames[-1], 1.0)
        npt.assert_equal(player(percept.play(fps=fps))['labels'][-1],
                         't = 50.00 ms')
    # At 25 fps, the 80 ms percept is sampled at 0 and 40 ms only:
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
    """play and save do not modify the percept"""
    percept = pulse_train_percept()
    data, time = percept.data.copy(), percept.time.copy()
    for fps in (None, 10, 1000):
        percept.play(fps=fps).to_jshtml()
    percept.save(str(tmp_path / 'test.mp4'), fps=30, vmin=0, vmax=1)
    npt.assert_almost_equal(percept.data, data)
    npt.assert_almost_equal(percept.time, time)


def test_Percept_play_units_equivalent():
    """The same timeline in ms or s gives the same animation"""
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
    # A scalar time drops the time axis:
    npt.assert_equal(percept[..., 10.0].shape, (2, 3))
    npt.assert_equal(percept[:, 0, 10.0].shape, (2,))
    npt.assert_equal(np.isscalar(percept[0, 1, 10.0]), True)
    # Stored time points are returned as is:
    npt.assert_almost_equal(percept[..., 10.0], data[..., 1])
    # Intermediate times are interpolated:
    npt.assert_almost_equal(percept[..., 5.0],
                            (data[..., 0] + data[..., 1]) / 2)
    npt.assert_almost_equal(percept[0, 1, 5.0],
                            (data[0, 1, 0] + data[0, 1, 1]) / 2)
    # Beyond the ends, the nearest frame is held:
    npt.assert_almost_equal(percept[..., -5.0], data[..., 0])
    npt.assert_almost_equal(percept[..., 99.0], data[..., -1])
    # NumPy indexing for space; a shorter index returns the time series:
    npt.assert_almost_equal(percept[0, 1], data[0, 1])
    npt.assert_almost_equal(percept[0], data[0])
    # float64 stays float64:
    npt.assert_equal(percept[..., 5.0].dtype, np.float64)


def test_Percept_getitem_multiple_times():
    """Lists, slices and masks of time points"""
    data = np.arange(24, dtype=float).reshape((2, 3, 4))
    percept = Percept(data, time=[0.0, 10.0, 20.0, 30.0])
    # A list of times is interpolated:
    npt.assert_almost_equal(percept[..., [5.0, 15.0]],
                            np.stack([(data[..., 0] + data[..., 1]) / 2,
                                      (data[..., 1] + data[..., 2]) / 2],
                                     axis=-1))
    # A one-element list keeps the time axis:
    npt.assert_equal(percept[..., [5.0]].shape, (2, 3, 1))
    npt.assert_equal(percept[0, 0, [5.0]].shape, (1,))
    # A stepped slice is a time range:
    npt.assert_equal(percept[..., 0:30:5].shape, (2, 3, 6))
    npt.assert_almost_equal(percept[0, 0, 0:30:10], data[0, 0, :3])
    # A stepless slice selects frames by position:
    npt.assert_almost_equal(percept[..., :], data)
    with pytest.raises(ValueError):
        percept[..., 0:20]
    # A boolean mask selects stored frames:
    npt.assert_almost_equal(percept[..., percept.time < 20], data[..., :2])
    # A mask of the wrong length raises IndexError (not read as t=1, t=0):
    with pytest.raises(IndexError):
        percept[..., np.array([True, False])]


def test_Percept_getitem_irregular_time():
    """Interpolation uses irregular time points"""
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
    # With time_unit=s, bare numbers are in s:
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
    # An automatic time axis is still a time axis:
    auto = Percept(np.arange(24, dtype=float).reshape((2, 3, 4)))
    npt.assert_almost_equal(auto.time, [0, 1, 2, 3])
    npt.assert_almost_equal(auto[0, 0, 1.5], 1.5)


def test_Percept_play_clim():
    """vmin/vmax set the color scale without changing the timeline"""
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
    """The color scale spans all frames, not only the displayed ones"""
    data = np.zeros((4, 4, 10))
    data[..., 1] = 20.0
    percept = Percept(data, time=np.arange(10) * 10.0)
    # 20 fps samples t = 0 and 50 ms, missing the flash at t = 10 ms:
    npt.assert_equal(percept.play(fps=20)._frame_data.max(), 0)
    # The color scale still includes it:
    npt.assert_almost_equal(percept.play(fps=20)._image.get_clim(), (0, 20))
    npt.assert_almost_equal(percept.play()._image.get_clim(), (0, 20))


def test_Percept_save_common_clim(tmp_path):
    """Percepts saved with the same range load on the same scale"""
    dim = Percept(np.linspace(0, 5, 256).reshape((16, 16, 1)))
    bright = Percept(np.linspace(0, 20, 256).reshape((16, 16, 1)))
    for percept, name in ((dim, 'dim.png'), (bright, 'bright.png')):
        percept.save(str(tmp_path / name), shape=(16, 16), vmin=0, vmax=20)
        loaded = Percept.load(str(tmp_path / name))
        # 8-bit gray over a range of 20:
        npt.assert_allclose(loaded.data, percept.data, atol=20 / 255)
    # The dim percept is not stretched:
    npt.assert_equal(imread(str(tmp_path / 'dim.png')).max() < 255, True)
    npt.assert_equal(imread(str(tmp_path / 'bright.png')).max(), 255)


def test_Percept_save_clim_edge_cases(tmp_path):
    """Clipping, negative values, constant data, and nonsensical ranges"""
    percept = Percept(np.linspace(-5, 5, 256).reshape((16, 16, 1)))
    # Negative values:
    fname = str(tmp_path / 'signed.png')
    percept.save(fname, shape=(16, 16), vmin=-5, vmax=5)
    npt.assert_allclose([Percept.load(fname).data.min(),
                         Percept.load(fname).data.max()], [-5, 5], atol=0.05)
    # Values below vmin are clipped:
    fname = str(tmp_path / 'clipped.png')
    percept.save(fname, shape=(16, 16), vmin=0, vmax=5)
    npt.assert_almost_equal(Percept.load(fname).data.min(), 0)
    npt.assert_equal(np.mean(Percept.load(fname).data == 0) > 0.4, True)
    # Constant percept (no division by zero):
    fname = str(tmp_path / 'constant.png')
    with pytest.warns(UserWarning):
        Percept(np.full((16, 16, 1), 3.0)).save(fname, shape=(16, 16))
    npt.assert_almost_equal(Percept.load(fname).data, 3.0)
    with pytest.raises(ValueError):
        percept.save(str(tmp_path / 'bad.png'), vmin=5, vmax=0)


def test_Percept_save_clim_ignores_fps(tmp_path):
    """Export fps does not change movie brightness"""
    data = np.zeros((16, 16, 10))
    data[..., :] = np.linspace(0, 1, 10)
    # A flash sampled only at the faster fps:
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
    """save warns if vmin/vmax are omitted"""
    percept = Percept(np.random.rand(16, 16, 1))
    with pytest.warns(UserWarning, match='Pass'):
        percept.save(str(tmp_path / 'auto.png'), shape=(16, 16))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        percept.save(str(tmp_path / 'fixed.png'), shape=(16, 16), vmin=0,
                     vmax=1)


def test_Percept_save_keeps_an_explicit_clim(tmp_path):
    """Movie file names store an explicit brightness range for load"""
    data = np.zeros((16, 16, 5))
    data[..., :] = np.linspace(0, 0.35, 5)
    percept = Percept(data, time=np.arange(5) * 100.0)
    fname = percept.save(str(tmp_path / 'percept.mp4'), shape=(32, 32),
                         vmax=0.5)
    # Both bounds are in the returned file name:
    npt.assert_equal(os.path.basename(fname),
                     'percept__p2p_vmin=0.0_vmax=0.5.mp4')
    npt.assert_equal(os.path.isfile(fname), True)
    # load needs no arguments and does not warn:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        loaded = Percept.load(fname)
    # vmax=0.5 is the encoding scale, not the data maximum:
    npt.assert_allclose(loaded.data.max(), 0.35, atol=0.02)
    npt.assert_allclose([loaded.data[..., i].mean() for i in range(5)],
                        np.linspace(0, 0.35, 5), atol=0.02)


def test_Percept_save_range_tag(tmp_path):
    """The file name stores the range only if metadata cannot"""
    percept = Percept(np.linspace(0, 20, 256).reshape((16, 16, 1)))
    # PNG metadata stores the range; name unchanged:
    fname = percept.save(str(tmp_path / 'p.png'), shape=(16, 16), vmin=0,
                         vmax=20)
    npt.assert_equal(os.path.basename(fname), 'p.png')
    npt.assert_allclose(Percept.load(fname).data.max(), 20, atol=0.1)
    # Automatic normalization does not rename:
    with pytest.warns(UserWarning):
        auto = percept.save(str(tmp_path / 'auto.bmp'), shape=(16, 16))
    npt.assert_equal(os.path.basename(auto), 'auto.bmp')
    # An existing tag is replaced, not appended:
    stale = str(tmp_path / 'stale__p2p_vmin=0.0_vmax=1.0.bmp')
    tagged = percept.save(stale, shape=(16, 16), vmin=0, vmax=20)
    npt.assert_equal(os.path.basename(tagged),
                     'stale__p2p_vmin=0.0_vmax=20.0.bmp')
    npt.assert_equal(os.path.isfile(stale), False)
    npt.assert_allclose(Percept.load(tagged).data.max(), 20, atol=0.1)
    # Also for an automatically resolved range:
    with pytest.warns(UserWarning):
        redone = percept.save(tagged, shape=(16, 16))
    npt.assert_equal(os.path.basename(redone),
                     'stale__p2p_vmin=0.0_vmax=20.0.bmp')


def test_Percept_load_image(tmp_path):
    """A static image loads as one frame with time=None"""
    fname = str(tmp_path / 'p.png')
    percept = Percept(np.linspace(0, 20, 256).reshape((16, 16, 1)))
    percept.save(fname, shape=(16, 16), vmin=0, vmax=20)
    loaded = Percept.load(fname)
    npt.assert_equal(loaded.shape, (16, 16, 1))
    npt.assert_equal(loaded.time, None)
    npt.assert_allclose(loaded.data, percept.data, atol=20 / 255)
    # Media files store no spatial coordinates:
    npt.assert_almost_equal(loaded.xdva, np.arange(16))
    grid = Grid2D((-1, 1), (-1, 1), step=2 / 15)
    npt.assert_almost_equal(Percept.load(fname, space=grid).xdva, grid._xflat)


@pytest.mark.parametrize('ext', ('.gif', '.mp4'))
def test_Percept_load_video(ext, tmp_path):
    """A GIF or movie loads with times from its frame rate"""
    data = np.zeros((16, 16, 5))
    data[..., :] = np.linspace(0, 20, 5)
    percept = Percept(data, time=np.arange(5) * 100.0)
    fname = percept.save(str(tmp_path / f'p{ext}'), shape=(32, 32), vmin=0,
                         vmax=20)
    loaded = Percept.load(fname, vmin=0, vmax=20)
    npt.assert_equal(loaded.shape[-1], 5)
    npt.assert_almost_equal(loaded.time, percept.time)
    # Quantization and codec error:
    npt.assert_allclose([loaded.data[..., i].mean() for i in range(5)],
                        np.linspace(0, 20, 5), atol=0.5)


def test_Percept_load_timing(tmp_path):
    """time and fps override the file's frame rate"""
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
    """A GIF with variable frame durations requires explicit time"""
    fname = str(tmp_path / 'ragged.gif')
    frames = [np.full((16, 16), level, dtype=np.uint8) for level in range(4)]
    imageio.mimwrite(fname, frames, duration=[100, 300, 50, 200])
    with pytest.raises(ValueError):
        Percept.load(fname)
    with pytest.warns(UserWarning):
        loaded = Percept.load(fname, time=[0, 100, 400, 450])
    npt.assert_almost_equal(loaded.time, [0, 100, 400, 450])


@pytest.mark.parametrize('fps', (0, -30, np.nan, np.inf))
def test_Percept_load_rejects_bad_fps(fps, tmp_path):
    """load raises ValueError for a non-positive or non-finite fps"""
    fname = str(tmp_path / 'p.gif')
    percept = Percept(np.random.rand(16, 16, 4), time=np.arange(4) * 100.0)
    percept.save(fname, shape=(16, 16), vmin=0, vmax=1)
    with pytest.raises(ValueError):
        Percept.load(fname, fps=fps)


def test_Percept_load_grayscale(tmp_path):
    """Color input is converted to grayscale"""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    imageio.imwrite(str(tmp_path / 'rgb.png'), rgb)
    with pytest.warns(UserWarning):
        loaded = Percept.load(str(tmp_path / 'rgb.png'))
    npt.assert_equal(loaded.shape, (8, 8, 1))
    # Luminance of pure red:
    npt.assert_allclose(loaded.data, 0.2125, atol=1e-3)
    # Alpha is blended against black:
    rgba = np.full((8, 8, 4), 255, dtype=np.uint8)
    rgba[..., 3] = 128
    imageio.imwrite(str(tmp_path / 'rgba.png'), rgba)
    with pytest.warns(UserWarning):
        loaded = Percept.load(str(tmp_path / 'rgba.png'))
    npt.assert_allclose(loaded.data, 128 / 255, atol=0.01)


def test_Percept_load_range_precedence(tmp_path):
    """Explicit vmin/vmax > file metadata > file name"""
    from PIL.PngImagePlugin import PngInfo
    gray = np.linspace(0, 255, 256).astype(np.uint8).reshape((16, 16))
    # Hand-written file with conflicting ranges: metadata [0, 20], name
    # [0, 5]:
    info = PngInfo()
    info.add_text('Comment', '__p2p_vmin=0.0_vmax=20.0')
    fname = str(tmp_path / 'p__p2p_vmin=0.0_vmax=5.0.png')
    imageio.imwrite(fname, gray, pnginfo=info)
    npt.assert_allclose(Percept.load(fname).data.max(), 20, atol=0.1)
    npt.assert_allclose(Percept.load(fname, vmax=100).data.max(), 100,
                        atol=0.5)
    # BMP has no metadata, so the file name is used:
    named = str(tmp_path / 'q__p2p_vmin=0.0_vmax=5.0.bmp')
    imageio.imwrite(named, gray)
    npt.assert_allclose(Percept.load(named).data.max(), 5, atol=0.05)


def test_Percept_load_unknown_range_warns(tmp_path):
    """load warns and keeps [0, 1] values if the range is unknown"""
    percept = Percept(np.linspace(0, 20, 256).reshape((16, 16, 1)))
    # No vmin/vmax, so no range is stored and the name is unchanged:
    with pytest.warns(UserWarning, match='Normalizing'):
        fname = percept.save(str(tmp_path / 'plain.bmp'), shape=(16, 16))
    npt.assert_equal(fname, str(tmp_path / 'plain.bmp'))
    with pytest.warns(UserWarning, match='encoded'):
        loaded = Percept.load(fname)
    npt.assert_almost_equal([loaded.data.min(), loaded.data.max()], [0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        recovered = Percept.load(fname, vmin=0, vmax=20)
    npt.assert_allclose(recovered.data, percept.data, atol=20 / 255)


def test_Percept_load_half_a_range_raises(tmp_path):
    """load raises ValueError if only one of vmin/vmax is known"""
    percept = Percept(np.linspace(0, 20, 256).reshape((16, 16, 1)))
    with pytest.warns(UserWarning):
        fname = percept.save(str(tmp_path / 'plain.bmp'), shape=(16, 16))
    with pytest.raises(ValueError, match="'vmin' is unknown"):
        Percept.load(fname, vmax=20)
    with pytest.raises(ValueError, match="'vmax' is unknown"):
        Percept.load(fname, vmin=0)
    # Both bounds:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        npt.assert_allclose(Percept.load(fname, vmin=0, vmax=20).data,
                            percept.data, atol=20 / 255)


def rgb_percept(n_frames=3, shape=(4, 6), **kwargs):
    """Return an RGB percept with distinct values on a known dva grid"""
    n_rows, n_cols = shape
    data = np.linspace(0, 1, n_rows * n_cols * 3 * n_frames, dtype=float)
    grid = Grid2D((-2, 2), (-1, 1),
                  step=[4 / (n_cols - 1), 2 / (n_rows - 1)])
    return Percept(data.reshape((n_rows, n_cols, 3, n_frames)), space=grid,
                   **kwargs)


@pytest.mark.parametrize('shape', [(4, 6), (4, 6, 4, 2), (4, 6, 3, 2, 2),
                                   (4, 6, 3, 3, 3)])
def test_Percept_rejects_shapes_that_are_neither_gray_nor_rgb(shape):
    with pytest.raises(ValueError):
        Percept(np.zeros(shape))


def test_Percept_rgb_shape_contract():
    gray = Percept(np.zeros((4, 6, 2)))
    npt.assert_equal(gray.is_rgb, False)
    npt.assert_equal(gray.shape, (4, 6, 2))
    rgb = Percept(np.zeros((4, 6, 3, 2)))
    npt.assert_equal(rgb.is_rgb, True)
    npt.assert_equal(rgb.shape, (4, 6, 3, 2))
    # A 3D array is grayscale even if its last axis has size 3:
    npt.assert_equal(Percept(np.zeros((4, 6, 3))).is_rgb, False)
    # `space` describes (Y, X) only:
    grid = Grid2D((-2, 2), (-1, 1))
    npt.assert_equal(Percept(np.zeros((*grid.x.shape, 3, 2)),
                             space=grid).xdva.size, grid.x.shape[1])


def test_Percept_rgb_still_and_multiframe():
    still = Percept(np.zeros((4, 6, 3, 1)))
    npt.assert_equal(still.shape, (4, 6, 3, 1))
    npt.assert_equal(still.time, None)
    movie = rgb_percept(n_frames=3, time=[0, 10, 20])
    npt.assert_equal(movie.shape, (4, 6, 3, 3))
    npt.assert_almost_equal(movie.time, [0, 10, 20])


def test_Percept_rgb_indexing_and_frames():
    percept = rgb_percept(n_frames=3, time=[0, 10, 20])
    # A frame keeps its color channels:
    npt.assert_equal(percept[..., 0].shape, (4, 6, 3))
    npt.assert_almost_equal(percept[..., 0], percept.data[..., 0])
    # A time between frames is interpolated:
    npt.assert_almost_equal(percept[..., 5.0],
                            0.5 * (percept.data[..., 0] +
                                   percept.data[..., 1]))
    # One pixel's color over time:
    npt.assert_equal(percept[0, 1].shape, (3, 3))
    for i, frame in enumerate(percept):
        npt.assert_equal(frame.shape, (4, 6, 3))
        npt.assert_almost_equal(frame, percept.data[..., i])
    # No brightest pixel or frame for RGB:
    for axis in (None, 'frames'):
        with pytest.raises(ValueError):
            percept.argmax(axis=axis)
        with pytest.raises(ValueError):
            percept.max(axis=axis)
    npt.assert_almost_equal(percept.data.max(), 1.0)


def test_Percept_rgb_plot():
    percept = rgb_percept(n_frames=1)
    ax = percept.plot()
    npt.assert_equal(isinstance(ax, Subplot), True)
    npt.assert_almost_equal(ax.axis(), [-2, 2, -1, 1])
    # Drawn as an image without a colormap:
    npt.assert_equal(len(ax.images), 1)
    npt.assert_equal(len(ax.collections), 0)
    drawn = ax.images[0].get_array()
    npt.assert_equal(drawn.shape, (4, 6, 3))
    npt.assert_almost_equal(drawn, percept.data[..., 0])
    # Row 0 at the top, as with `pcolor` for grayscale:
    left, right, bottom, top = ax.images[0].get_extent()
    npt.assert_equal(top > bottom, True)
    npt.assert_equal(right > left, True)
    # vmin, vmax, cmap raise ValueError for RGB:
    for kwargs in ({'vmin': 0}, {'vmax': 1}, {'cmap': 'viridis'}):
        with pytest.raises(ValueError):
            percept.plot(**kwargs)
    # hexbin requires one value per pixel:
    with pytest.raises(ValueError):
        percept.plot(kind='hex')


def test_Percept_rgb_plot_will_not_pick_a_frame():
    """plot raises ValueError for a multi-frame RGB percept"""
    percept = rgb_percept(n_frames=2, time=[0, 10])
    with pytest.raises(ValueError):
        percept.plot()


def test_Percept_rgb_play():
    percept = rgb_percept(n_frames=3, time=[0, 10, 20])
    ani = percept.play()
    npt.assert_equal(isinstance(ani, FuncAnimation), True)
    # RGB frames, no colorbar:
    npt.assert_equal(ani._frame_data.shape, (4, 6, 3, 3))
    npt.assert_equal(len(ani._fig.axes), 1)
    npt.assert_equal(player(ani)['n'], 3)
    with pytest.raises(ValueError):
        percept.play(vmin=0)


@pytest.mark.parametrize('value', [1.8, -0.1, np.nan, np.inf])
def test_Percept_rgb_rejects_values_outside_the_display_range(value):
    """RGB values outside [0, 1] or non-finite raise ValueError"""
    data = np.full((4, 6, 3, 2), 0.5)
    data[1, 2, 0, 1] = value
    with pytest.raises(ValueError):
        Percept(data)


def test_Percept_rgb_accepts_the_ends_of_the_display_range():
    percept = Percept(np.array([0.0, 1.0] * 12).reshape((4, 2, 3, 1)))
    npt.assert_equal(percept.is_rgb, True)
    npt.assert_almost_equal((percept.data.min(), percept.data.max()), (0, 1))
    # Brightness percepts are unbounded:
    npt.assert_almost_equal(Percept(np.full((4, 6, 2), 40.0)).data.max(), 40)


def test_Percept_rgb_save_still_roundtrip(tmp_path):
    fname = str(tmp_path / 'still.png')
    percept = Percept(np.linspace(0, 1, 16 * 16 * 3).reshape((16, 16, 3, 1)))
    # No warning for RGB:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = percept.save(fname, shape=(16, 16))
    npt.assert_equal(out, fname)
    loaded = Percept.load(out, as_gray=False)
    npt.assert_equal(loaded.is_rgb, True)
    npt.assert_equal(loaded.shape, (16, 16, 3, 1))
    # 8-bit quantization error:
    npt.assert_allclose(loaded.data, percept.data, atol=1 / 255)
    # Default loads as brightness:
    npt.assert_equal(Percept.load(out).shape, (16, 16, 1))


def test_Percept_rgb_save_movie_roundtrip(tmp_path):
    fname = str(tmp_path / 'movie.gif')
    data = np.linspace(0, 1, 16 * 16 * 3 * 4).reshape((16, 16, 3, 4))
    percept = Percept(data, time=np.arange(4) * 100.0)
    out = percept.save(fname, shape=(16, 16))
    loaded = Percept.load(out, as_gray=False)
    npt.assert_equal(loaded.shape, (16, 16, 3, 4))
    npt.assert_almost_equal(loaded.time, [0, 100, 200, 300])
    npt.assert_allclose(loaded.data, percept.data, atol=1 / 255)
    with pytest.raises(ValueError):
        percept.save(str(tmp_path / 'x.gif'), vmin=0, vmax=1)


def test_Percept_rgb_rejects_brightness_operations():
    data = np.random.rand(4, 6, 3, 2)
    with pytest.raises(ValueError):
        Percept(data, n_gray=4)
    # Brightness percept:
    npt.assert_equal(len(np.unique(Percept(np.random.rand(4, 6, 2),
                                           n_gray=4).data)), 4)


def test_Percept_load_rgb_rejects_a_brightness_range(tmp_path):
    fname = str(tmp_path / 'rgb.png')
    imageio.imwrite(fname, np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        Percept.load(fname, as_gray=False, vmin=0, vmax=20)
    # Loads without a warning:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        npt.assert_equal(Percept.load(fname, as_gray=False).shape,
                         (8, 8, 3, 1))


def test_Percept_rgb_temporal_plot():
    """A 1x1 RGB percept plots one line per channel"""
    percept = Percept(np.random.rand(1, 1, 3, 5), time=np.arange(5) * 10.0)
    ax = percept.plot()
    npt.assert_equal(len(ax.lines), 3)
    npt.assert_almost_equal(ax.lines[0].get_ydata(), percept.data[0, 0, 0])


def test_model_prediction_stays_grayscale():
    """Model predictions are grayscale (Y, X, T)"""
    from pulse2percept.implants.retina import ArgusII
    from pulse2percept.models.retina import ScoreboardModel
    model = ScoreboardModel(implant=ArgusII(), rho=200, xrange=(-4, 4),
                            yrange=(-4, 4), step=1).build()
    percept = model.predict_percept({'A8': 30})
    npt.assert_equal(percept.is_rgb, False)
    npt.assert_equal(percept.data.ndim, 3)
    npt.assert_equal(percept.shape, (9, 9, 1))
    # Nonzero phosphene:
    npt.assert_equal(percept.data.max() > 0, True)
