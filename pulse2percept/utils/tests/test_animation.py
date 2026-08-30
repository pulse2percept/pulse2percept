import base64
import json
import re
from io import BytesIO

import numpy as np
import numpy.testing as npt
import pytest

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

from pulse2percept.units import (DimensionMismatchError, Hz, dva, kHz, ms, uA)
from pulse2percept.utils import HTMLAnimation, frame_interval
from pulse2percept.utils.animation import (MAX_SPRITE_PX,
                                           SINGLE_FRAME_INTERVAL,
                                           _frame_timeline, _sprite_grid,
                                           _frame_shape, _weight2css,
                                           _check_fmt)


def make_ani(data, labels=None, interval=25.0, repeat=True, colorbar=False,
             fmt='png', intervals=None):
    """Set up an HTMLAnimation the same way ``Percept.play`` does"""
    fig, ax = plt.subplots(figsize=(8, 5))
    frame0 = np.zeros(data.shape[:-1])
    mat = ax.imshow(frame0, cmap='gray', vmin=0, vmax=data.max())
    if colorbar:
        fig.colorbar(mat)
    plt.close(fig)
    # Either a single delay for all frames, or one delay per frame:
    timing = ({'interval': interval} if intervals is None
              else {'intervals': intervals})
    return HTMLAnimation(fig, lambda d: mat, iter(range(data.shape[-1])),
                         save_count=data.shape[-1], repeat=repeat, image=mat,
                         frame_data=data, labels=labels, fmt=fmt, **timing)


def parse(html):
    """Pull the player config and the two embedded images out of the HTML"""
    cfg = json.loads(re.search(r'var cfg = (\{.*?\});', html, re.S).group(1))
    imgs = [Image.open(BytesIO(base64.b64decode(b64)))
            for b64 in re.findall(r'data:image/\w+;base64,([A-Za-z0-9+/=]+)',
                                  html)]
    # These animations have a single animated image; read its geometry as the
    # config's own, the way the player did before it could stack layers:
    return {**cfg, **cfg['layers'][0]}, imgs[0], imgs[1]


def tile(cfg, sheet, i):
    """Cut frame ``i`` out of the sprite sheet"""
    col, row = i % cfg['ncols'], i // cfg['ncols']
    return np.asarray(sheet)[row * cfg['sh']:row * cfg['sh'] + cfg['fh'],
                             col * cfg['sw']:col * cfg['sw'] + cfg['fw']]


def test_sprite_grid():
    # A single frame needs a single tile:
    npt.assert_equal(_sprite_grid(1, 10, 10), (1, 1))
    for n_frames in [2, 3, 7, 16, 94, 1000]:
        for height, width in [(10, 10), (3, 100), (100, 3)]:
            n_rows, n_cols = _sprite_grid(n_frames, height, width)
            # Every frame must fit on the sheet, with at most one partial row:
            npt.assert_equal(n_rows * n_cols >= n_frames, True)
            npt.assert_equal((n_rows - 1) * n_cols < n_frames, True)
            # Tiling must keep the sheet smaller than a single stack of
            # frames, which is what browsers care about:
            npt.assert_equal(max(n_rows * height, n_cols * width) <=
                             max(n_frames * height, width), True)


def test_frame_shape():
    # Frames are never upsampled:
    npt.assert_equal(_frame_shape((10, 20), 5, (100, 200)), (10, 20))
    # ... but are downsampled to the size at which they are displayed:
    npt.assert_equal(_frame_shape((100, 200), 5, (50, 100)), (50, 100))
    # Aspect ratio is preserved:
    npt.assert_equal(_frame_shape((100, 200), 5, (50, 400)), (50, 100))
    # Huge stacks are shrunk until the sheet fits what browsers can decode,
    # padding included:
    for pad_to in (1, 8, 16):
        height, width = _frame_shape((2000, 2000), 1000, (2000, 2000), pad_to)
        pad_h = int(np.ceil(height / pad_to)) * pad_to
        pad_w = int(np.ceil(width / pad_to)) * pad_to
        n_rows, n_cols = _sprite_grid(1000, pad_h, pad_w)
        npt.assert_equal(max(n_rows * pad_h, n_cols * pad_w) <= MAX_SPRITE_PX,
                         True)


def test_weight2css():
    npt.assert_equal(_weight2css('normal'), 'normal')
    npt.assert_equal(_weight2css('light'), 'normal')
    npt.assert_equal(_weight2css('bold'), 'bold')
    npt.assert_equal(_weight2css('demibold'), 'bold')
    npt.assert_equal(_weight2css(700), '700')


def test_check_fmt():
    npt.assert_equal(_check_fmt('jpg'), 'jpg')
    npt.assert_equal(_check_fmt('JPEG'), 'jpg')
    npt.assert_equal(_check_fmt('PNG'), 'png')
    for fmt in ['gif', 'webp', 'gzip', None]:
        with pytest.raises(ValueError):
            _check_fmt(fmt)


def test_frame_interval():
    # Inferred from the time axis:
    npt.assert_almost_equal(frame_interval([0, 10, 20, 30]), 10)
    npt.assert_almost_equal(frame_interval([0, 0.5, 1.0]), 0.5)
    # 'fps' wins over the time axis:
    npt.assert_almost_equal(frame_interval([0, 10, 20], fps=25), 40)
    # A single frame has no time step of its own, but must still animate:
    npt.assert_almost_equal(frame_interval([0]), SINGLE_FRAME_INTERVAL)
    npt.assert_almost_equal(frame_interval([0], fps=10), 100)
    # A non-homogeneous time axis needs an explicit 'fps':
    with pytest.raises(NotImplementedError):
        frame_interval([0, 1, 10])
    npt.assert_almost_equal(frame_interval([0, 1, 10], fps=20), 50)
    # 'tol' decides how much jitter still counts as homogeneous:
    npt.assert_almost_equal(frame_interval([0, 10, 20.005], tol=1), 10)
    with pytest.raises(NotImplementedError):
        frame_interval([0, 10, 20.005], tol=1e-6)


def test_frame_interval_fps_units():
    """A frame rate is a frequency, however it is spelled"""
    bare = frame_interval([0, 10, 20], fps=25)
    for spelling in (25 * Hz, 0.025 * kHz):
        npt.assert_allclose(frame_interval([0, 10, 20], fps=spelling), bare,
                            rtol=1e-12)
    # ... and nothing else is a frame rate:
    for wrong in (30 * ms, 30 * uA, 30 * dva):
        with pytest.raises(DimensionMismatchError):
            frame_interval([0, 10, 20], fps=wrong)


def test_frame_timeline():
    """`fps` decides how often the timeline is sampled, not how fast it runs"""
    # At its own rate, every frame is shown for its own time step:
    timeline = _frame_timeline([0, 10, 20, 30])
    npt.assert_equal(timeline.indices, [0, 1, 2, 3])
    npt.assert_almost_equal(timeline.times, [0, 10, 20, 30])
    npt.assert_almost_equal(timeline.intervals, [10, 10, 10, 10])
    # A single frame has no time step at all:
    timeline = _frame_timeline([7.5])
    npt.assert_equal(timeline.indices, [0])
    npt.assert_almost_equal(timeline.intervals, [SINGLE_FRAME_INTERVAL])
    # An irregular axis keeps its own unequal time steps:
    timeline = _frame_timeline([0, 0.45, 0.55, 166.67])
    npt.assert_almost_equal(timeline.intervals, [0.45, 0.1, 166.12, 166.12],
                            decimal=6)

    # 40 ms of animation stays 40 ms of animation, whatever the display rate:
    for fps, n_frames in [(25, 1), (50, 2), (100, 4), (200, 8)]:
        timeline = _frame_timeline([0, 10, 20, 30], fps=fps)
        npt.assert_equal(timeline.indices.size, n_frames)
        npt.assert_almost_equal(timeline.intervals, [1000.0 / fps] * n_frames)
        npt.assert_almost_equal(timeline.intervals.sum(), 40, decimal=6)
    # Zero-order hold: each display frame repeats the most recent one that was
    # due, and frames in between are dropped rather than blended in:
    npt.assert_equal(_frame_timeline([0, 10, 20, 30], fps=200).indices,
                     [0, 0, 1, 1, 2, 2, 3, 3])
    npt.assert_equal(_frame_timeline([0, 10, 20, 30], fps=50).indices, [0, 2])
    # An irregular axis resamples the same way. The frame that is only up
    # between 0.45 and 0.55 ms falls between two display samples and is never
    # shown, which is what a 30 fps display would do:
    npt.assert_equal(_frame_timeline([0, 0.45, 0.55, 166.67], fps=30).indices,
                     [0, 2, 2, 2, 2, 2, 3, 3, 3, 3])

    # A frame rate is a frequency, however it is spelled ...
    for spelling in (25 * Hz, 0.025 * kHz):
        npt.assert_equal(_frame_timeline([0, 10, 20], fps=spelling).indices,
                         _frame_timeline([0, 10, 20], fps=25).indices)
    # ... and nothing else is one:
    for wrong in (30 * ms, 30 * uA, 30 * dva):
        with pytest.raises(DimensionMismatchError):
            _frame_timeline([0, 10, 20], fps=wrong)
    for wrong in (0, -30):
        with pytest.raises(ValueError):
            _frame_timeline([0, 10, 20], fps=wrong)
    with pytest.raises(ValueError):
        _frame_timeline([])


def test_frame_timeline_rejects_unordered_time():
    """A frame cannot come up before the one in front of it

    `searchsorted` assumes a sorted axis and would quietly pick the wrong
    frames for anything else, and a negative interval would go straight to the
    player as a negative delay.
    """
    for wrong in ([0, 10, 5], [0, 10, 10], [10, 0], [0, np.nan, 10],
                  [0, np.inf]):
        for fps in (None, 30):
            with pytest.raises(ValueError):
                _frame_timeline(wrong, fps=fps)


def test_frame_timeline_last_frame():
    """The last frame is held for the interval in front of it

    A time axis says when each frame comes up, not when the last one goes
    away. Reading its final time point as an endpoint instead would give that
    frame no duration at all, and a frame nobody can see is not a frame -- a
    percept's time points are the instants a model was evaluated at, so the
    last one is model output like any other.
    """
    for time in ([0, 10, 20], [0, 10, 30], [0, 0.45, 0.55, 166.67]):
        timeline = _frame_timeline(time)
        npt.assert_almost_equal(timeline.intervals[-1],
                                timeline.intervals[-2])
        npt.assert_array_less(0, timeline.intervals)
        # `n` frames of `dt` take `n * dt`, which is what a video means by `n`
        # frames:
        npt.assert_almost_equal(_frame_timeline([0, 10, 20]).intervals.sum(),
                                30)
        # ... and a display clock fine enough to resolve it reaches it,
        # which a zero-length last frame would not allow at any rate:
        npt.assert_equal(_frame_timeline(time, fps=1000).indices[-1],
                         len(time) - 1)


def test_frame_timeline_does_not_mutate():
    """The timeline is handed out, so it cannot alias the caller's axis"""
    time = np.array([0.0, 10.0, 30.0])
    timeline = _frame_timeline(time)
    timeline.times[0] = 999
    npt.assert_almost_equal(time, [0, 10, 30])


@pytest.mark.parametrize('n_frames', (1, 2, 5, 17))
@pytest.mark.parametrize('fmt', ('png', 'jpg'))
def test_HTMLAnimation_sprite_sheet(n_frames, fmt):
    data = np.random.rand(6, 8, n_frames)
    ani = make_ani(data, fmt=fmt)
    cfg, bg, sheet = parse(ani.to_jshtml())
    npt.assert_equal(cfg['n'], n_frames)
    # Frames are embedded at their native size (they are magnified for
    # display, and the browser can do that for free):
    npt.assert_equal((cfg['fh'], cfg['fw']), (6, 8))
    # The sheet must be large enough to hold every frame:
    n_rows = int(np.ceil(n_frames / cfg['ncols']))
    npt.assert_equal(sheet.size, (cfg['ncols'] * cfg['sw'],
                                  n_rows * cfg['sh']))
    if fmt == 'png':
        # Scalar data is shipped as a palettized PNG (one byte per pixel),
        # with no padding needed between frames:
        npt.assert_equal(sheet.mode, 'P')
        npt.assert_equal((cfg['sh'], cfg['sw']), (cfg['fh'], cfg['fw']))
    else:
        # A gray colormap needs no chroma, so 8x8 DCT blocks are the unit
        # that frames must be aligned to:
        npt.assert_equal(sheet.mode, 'L')
        npt.assert_equal((cfg['sh'], cfg['sw']), (8, 8))
    # The image is blitted into the figure, which is otherwise static:
    npt.assert_equal(bg.size, (800, 500))
    x, y, w, h = cfg['rect']
    npt.assert_equal(x >= 0 and y >= 0, True)
    npt.assert_equal(x + w <= 800 and y + h <= 500, True)


@pytest.mark.parametrize('shape', ((6, 8), (61, 91), (37, 37), (13, 100)))
def test_HTMLAnimation_rect_covers_image(shape):
    """The frame must not leave a gap along the edge of the image

    Rounding the *size* of the destination rect (rather than its edges) can
    leave a row or column of figure background exposed, which shows up as a
    bright line along the edge of the percept.
    """
    ani = make_ani(np.random.rand(*shape, 3))
    cfg, bg, _ = parse(ani.to_jshtml())
    bbox = ani._layers[0].image.get_window_extent()
    height = bg.size[1]
    x, y, w, h = cfg['rect']
    npt.assert_equal(x <= bbox.x0 and x + w >= bbox.x1, True)
    npt.assert_equal(y <= height - bbox.y1, True)
    npt.assert_equal(y + h >= height - bbox.y0, True)
    # ... but must not overshoot by more than a pixel on either side, which
    # would visibly stretch the percept:
    npt.assert_equal(w - (bbox.x1 - bbox.x0) < 2, True)
    npt.assert_equal(h - (bbox.y1 - bbox.y0) < 2, True)


def test_HTMLAnimation_frame_values():
    """Every frame must land on the sheet with the right gray levels"""
    n_frames = 7
    data = np.linspace(0, 1, 4 * 5 * n_frames).reshape((4, 5, n_frames))
    cfg, _, sheet = parse(make_ani(data, fmt='png').to_jshtml())
    for i in range(n_frames):
        # Matplotlib quantizes to 256 levels before the colormap lookup:
        expected = np.clip(data[..., i] / data.max() * 256, 0, 255)
        npt.assert_equal(tile(cfg, sheet, i), expected.astype(np.uint8))
    # JPEG is lossy, but must still be visually indistinguishable:
    cfg, _, sheet = parse(make_ani(data, fmt='jpg').to_jshtml())
    for i in range(n_frames):
        expected = np.clip(data[..., i] / data.max() * 256, 0, 255)
        npt.assert_array_less(np.abs(tile(cfg, sheet, i).astype(float) -
                                     expected), 16)


@pytest.mark.parametrize('fmt', ('png', 'jpg'))
def test_HTMLAnimation_rgb(fmt):
    data = np.linspace(0, 1, 8 * 8 * 3 * 5).reshape((8, 8, 3, 5))
    cfg, _, sheet = parse(make_ani(data, fmt=fmt).to_jshtml())
    npt.assert_equal(sheet.mode, 'RGB')
    npt.assert_equal((cfg['fh'], cfg['fw']), (8, 8))
    if fmt == 'jpg':
        # Color needs chroma, which is subsampled in 16x16 macroblocks:
        npt.assert_equal((cfg['sh'], cfg['sw']), (16, 16))
    for i in range(5):
        expected = (data[..., i] * 255).astype(np.uint8)
        if fmt == 'png':
            npt.assert_equal(tile(cfg, sheet, i), expected)
        else:
            npt.assert_array_less(np.abs(tile(cfg, sheet, i).astype(float) -
                                         expected), 24)


@pytest.mark.parametrize('fmt', ('png', 'jpg'))
def test_HTMLAnimation_rgba(fmt):
    """Four channels are RGBA, not RGB

    ``Image.fromarray(sheet, mode='RGB')`` on an RGBA array does not drop the
    alpha channel: PIL reads the 4-bytes-per-pixel buffer 3 bytes at a time, so
    the channels shear across every row and the sheet turns to garbage.
    """
    data = np.zeros((6, 8, 4, 5), dtype=np.float32)
    data[:, :, 0, :] = 1.0     # pure red ...
    data[:, :, 3, :] = 0.25    # ... at 25% opacity
    cfg, _, sheet = parse(make_ani(data, fmt=fmt).to_jshtml())
    npt.assert_equal((cfg['fh'], cfg['fw']), (6, 8))
    for i in range(5):
        tile_i = tile(cfg, sheet, i).astype(float)
        if fmt == 'png':
            # PNG carries the alpha channel, and the canvas composites it:
            npt.assert_equal(sheet.mode, 'RGBA')
            npt.assert_array_less(
                np.abs(tile_i - [255, 0, 0, 64]).max(axis=-1), 2)
        else:
            # JPEG cannot, so the frames are flattened onto the white axes
            # background, which is what Matplotlib rasterizes as well:
            npt.assert_equal(sheet.mode, 'RGB')
            npt.assert_array_less(
                np.abs(tile_i - [255, 191, 191]).max(axis=-1), 8)


@pytest.mark.parametrize('shape', ((65, 97), (30, 40), (16, 16), (13, 11)))
def test_HTMLAnimation_no_frame_bleed(shape):
    """JPEG blocks must never straddle two frames of the sprite sheet

    Without padding each frame out to a whole number of blocks, a bright frame
    bleeds into the edge of the dark frame next to it on the sheet.
    """
    n_frames = 8
    for data in [np.zeros((*shape, n_frames)),
                 np.zeros((*shape, 3, n_frames))]:
        data[..., 1::2] = 1.0    # alternate pitch-black and pure-white frames
        cfg, _, sheet = parse(make_ani(data, fmt='jpg').to_jshtml())
        for i in range(n_frames):
            npt.assert_array_less(
                np.abs(tile(cfg, sheet, i).astype(float) - i % 2 * 255), 2)


def test_HTMLAnimation_labels():
    data = np.random.rand(4, 4, 3)
    labels = ['t = 0.00 ms', 't = 1.00 ms', 't = 2.00 ms']
    cfg, _, _ = parse(make_ani(data, labels=labels).to_jshtml())
    npt.assert_equal(cfg['labels'], labels)
    npt.assert_equal(cfg['title'] is not None, True)
    # The title band sits above the image and spans the whole figure:
    npt.assert_equal(cfg['title']['rect'][1] + cfg['title']['rect'][3] <=
                     cfg['rect'][1], True)
    # Without labels, the player leaves the title alone:
    cfg, _, _ = parse(make_ani(data).to_jshtml())
    npt.assert_equal(cfg['title'], None)
    npt.assert_equal(cfg['labels'], [])


def title_ink(label, dpi=100):
    """The pixels a Matplotlib-rendered axes title actually covers"""
    rendered = []
    for text in (label, ''):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(np.zeros((4, 4)), cmap='gray')
        ax.set_title(text)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=dpi)
        plt.close(fig)
        rendered.append(np.asarray(Image.open(buf).convert('L'), dtype=int))
    rows, cols = np.where(rendered[0] != rendered[1])
    return rows.min(), rows.max(), cols.min(), cols.max()


@pytest.mark.parametrize('label', ('t = 0.00 ms', 't = 123.45 ms', 'gjpqy AWM'))
def test_HTMLAnimation_title_band_covers_text(label):
    """The band the player clears must cover a whole line of text

    The player redraws the title on the canvas for every frame, but can only
    erase what it drew itself. A band that is too short (an empty ``Text`` has
    a degenerate bounding box, so measuring it directly gives a zero-height
    band) leaves every title standing, and they pile up into an unreadable
    smear after a few frames.
    """
    cfg, _, _ = parse(make_ani(np.random.rand(4, 4, 3),
                               labels=[label] * 3).to_jshtml())
    _, top, _, height = cfg['title']['rect']
    y0, y1, x0, x1 = title_ink(label)
    npt.assert_equal(top <= y0 and y1 < top + height, True)
    # ... and the text must be anchored where Matplotlib would put it:
    npt.assert_equal(cfg['title']['align'], 'center')
    npt.assert_array_less(abs((x0 + x1) / 2 - cfg['title']['x']), 2)


def test_HTMLAnimation_title_not_in_background():
    """A title left on the axes must not be baked into the background

    The background is a static ``<img>`` underneath the canvas, so anything
    rendered into it survives the player's ``clearRect`` and shows through
    every frame. Two ways to get a title onto the axes before the HTML is
    built: pass in an axes that already has one, or render the inherited
    ``FuncAnimation`` first (which leaves the last frame's title behind).
    """
    data = np.random.rand(4, 4, 3)
    labels = ['t = 0.00 ms', 't = 1.00 ms', 't = 2.00 ms']
    ani = make_ani(data, labels=labels)
    reference = np.asarray(parse(ani.to_jshtml())[1])
    for stale in ('t = 999.00 ms', 'A title'):
        ani = make_ani(data, labels=labels)
        ani._layers[0].image.axes.set_title(stale)
        npt.assert_equal(np.asarray(parse(ani.to_jshtml())[1]), reference)
        # The caller's title is left the way it was found:
        npt.assert_equal(ani._layers[0].image.axes.get_title(), stale)
    # Without labels the player leaves the title alone, so a title on the axes
    # belongs in the background:
    ani = make_ani(data)
    ani._layers[0].image.axes.set_title('A title')
    npt.assert_equal(np.any(np.asarray(parse(ani.to_jshtml())[1]) != reference),
                     True)


def test_HTMLAnimation_playback():
    data = np.random.rand(4, 4, 3)
    # 'repeat' picks the default loop mode:
    npt.assert_equal(parse(make_ani(data).to_jshtml())[0]['mode'], 'loop')
    once = parse(make_ani(data, repeat=False).to_jshtml())[0]
    npt.assert_equal(once['mode'], 'once')
    ani = make_ani(data, interval=40.0)
    npt.assert_almost_equal(parse(ani.to_jshtml())[0]['interval'], 40.0)
    # 'fps' and 'default_mode' override the animation's own settings:
    cfg, _, _ = parse(ani.to_jshtml(fps=10, default_mode='reflect'))
    npt.assert_almost_equal(cfg['interval'], 100.0)
    npt.assert_equal(cfg['mode'], 'reflect')


def test_HTMLAnimation_per_frame_intervals():
    """Frames of unequal duration are what an irregular time axis needs"""
    data = np.random.rand(4, 4, 3)
    intervals = [0.45, 165.67, 0.45]
    cfg, _, _ = parse(make_ani(data, intervals=intervals).to_jshtml())
    npt.assert_almost_equal(cfg['intervals'], intervals)
    # Matplotlib only knows a single frame delay, which is all its own
    # machinery (`save`, `to_html5_video`) can express:
    ani = make_ani(data, intervals=intervals)
    npt.assert_almost_equal(ani._interval, np.mean(intervals))
    # A constant delay is still the default:
    cfg, _, _ = parse(make_ani(data, interval=40.0).to_jshtml())
    npt.assert_almost_equal(cfg['intervals'], [40.0] * 3)
    # An explicit `fps` overrides the animation's own timing, as it does in
    # Matplotlib -- it does not resample the frames:
    cfg, _, _ = parse(make_ani(data, intervals=intervals).to_jshtml(fps=10))
    npt.assert_almost_equal(cfg['intervals'], [100.0] * 3)
    npt.assert_equal(cfg['n'], 3)
    # There must be exactly one delay per frame:
    with pytest.raises(ValueError):
        make_ani(data, intervals=[10, 20])



def test_HTMLAnimation_smoothing():
    # Strongly magnified frames are drawn with nearest-neighbor, just like
    # Matplotlib's 'antialiased' interpolation:
    small = parse(make_ani(np.random.rand(4, 4, 3)).to_jshtml())[0]
    npt.assert_equal(small['smooth'], False)
    # Frames that are shown at roughly their native size are interpolated:
    large = parse(make_ani(np.random.rand(300, 300, 3)).to_jshtml())[0]
    npt.assert_equal(large['smooth'], True)


def test_HTMLAnimation_html():
    data = np.random.rand(4, 4, 3)
    html = make_ani(data, labels=['a', 'b', 'c']).to_jshtml()
    # Self-contained: no external resources, and everything is scoped to a
    # unique id so that several animations can live in the same notebook:
    npt.assert_equal('http://' in html or 'https://' in html, False)
    uids = set(re.findall(r'id="(p2p-anim-[0-9a-f]+)"', html))
    npt.assert_equal(len(uids), 1)
    npt.assert_equal(html.count(uids.pop()) > 5, True)
    npt.assert_equal(make_ani(data).to_jshtml() != html, True)
    # No placeholder was left unsubstituted:
    npt.assert_equal(re.search(r'\$(uid|bg|sheet|config|width|height)', html),
                     None)


def test_HTMLAnimation_fmt():
    data = np.random.rand(20, 20, 5)
    png = make_ani(data, fmt='png').to_jshtml()
    jpg = make_ani(data, fmt='jpg').to_jshtml()
    npt.assert_equal('data:image/png;base64,' in png, True)
    npt.assert_equal('data:image/jpeg;base64,' in jpg, True)
    # The background is always a PNG: it is mostly text and thin lines, which
    # is exactly what JPEG is bad at:
    npt.assert_equal(jpg.count('data:image/png;base64,'), 1)
    # 'jpeg' is accepted as an alias, and an unknown format is rejected:
    ani = make_ani(np.random.rand(4, 4, 2), fmt='JPEG')
    npt.assert_equal(ani._fmt, 'jpg')
    npt.assert_equal('data:image/jpeg;base64,' in ani.to_jshtml(), True)
    with pytest.raises(ValueError):
        make_ani(data, fmt='gif')


def test_HTMLAnimation_caching():
    ani = make_ani(np.random.rand(4, 4, 3))
    html = ani.to_jshtml()
    # The same call returns the identical (cached) player:
    npt.assert_equal(ani.to_jshtml(), html)
    npt.assert_equal(ani._repr_html_(), html)
    # ... but changing the playback settings rebuilds it:
    npt.assert_equal(ani.to_jshtml(fps=1) != html, True)


def test_HTMLAnimation_matplotlib_compat():
    data = np.random.rand(4, 4, 3)
    ani = make_ani(data)
    npt.assert_equal(isinstance(ani, FuncAnimation), True)
    npt.assert_equal(len(list(ani.frame_seq)), 3)
    npt.assert_equal('p2p-anim' in ani.to_jshtml(), True)
    # Without frame data there is nothing to accelerate, so Matplotlib's own
    # (slow) player is used:
    ani = make_ani(data)
    ani._layers = None
    html = ani.to_jshtml()
    npt.assert_equal('<script' in html, True)
    npt.assert_equal('p2p-anim' in html, False)
