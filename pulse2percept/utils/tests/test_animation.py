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
    """Return the player config and the two embedded images"""
    cfg = json.loads(re.search(r'var cfg = (\{.*?\});', html, re.S).group(1))
    imgs = [Image.open(BytesIO(base64.b64decode(b64)))
            for b64 in re.findall(r'data:image/\w+;base64,([A-Za-z0-9+/=]+)',
                                  html)]
    # Single animated image, so merge its layer config into the top level:
    return {**cfg, **cfg['layers'][0]}, imgs[0], imgs[1]


def tile(cfg, sheet, i):
    """Return frame ``i`` of the sprite sheet"""
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
            # Tiled sheet must be no larger than a single stack of frames:
            npt.assert_equal(max(n_rows * height, n_cols * width) <=
                             max(n_frames * height, width), True)


def test_frame_shape():
    # Frames are never upsampled:
    npt.assert_equal(_frame_shape((10, 20), 5, (100, 200)), (10, 20))
    # ... but are downsampled to the size at which they are displayed:
    npt.assert_equal(_frame_shape((100, 200), 5, (50, 100)), (50, 100))
    # Aspect ratio is preserved:
    npt.assert_equal(_frame_shape((100, 200), 5, (50, 400)), (50, 100))
    # Huge stacks are shrunk until the padded sheet fits MAX_SPRITE_PX:
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
    # A single frame has no time step:
    npt.assert_almost_equal(frame_interval([0]), SINGLE_FRAME_INTERVAL)
    npt.assert_almost_equal(frame_interval([0], fps=10), 100)
    # A non-homogeneous time axis needs an explicit 'fps':
    with pytest.raises(NotImplementedError):
        frame_interval([0, 1, 10])
    npt.assert_almost_equal(frame_interval([0, 1, 10], fps=20), 50)
    # 'tol' sets how much jitter counts as homogeneous:
    npt.assert_almost_equal(frame_interval([0, 10, 20.005], tol=1), 10)
    with pytest.raises(NotImplementedError):
        frame_interval([0, 10, 20.005], tol=1e-6)


def test_frame_interval_fps_units():
    """fps accepts plain Hz or any frequency unit"""
    bare = frame_interval([0, 10, 20], fps=25)
    for spelling in (25 * Hz, 0.025 * kHz):
        npt.assert_allclose(frame_interval([0, 10, 20], fps=spelling), bare,
                            rtol=1e-12)
    # Other units are rejected:
    for wrong in (30 * ms, 30 * uA, 30 * dva):
        with pytest.raises(DimensionMismatchError):
            frame_interval([0, 10, 20], fps=wrong)


def test_frame_timeline():
    """fps sets the display sampling rate without changing duration"""
    # Without fps, each frame is shown for its own time step:
    timeline = _frame_timeline([0, 10, 20, 30])
    npt.assert_equal(timeline.indices, [0, 1, 2, 3])
    npt.assert_almost_equal(timeline.times, [0, 10, 20, 30])
    npt.assert_almost_equal(timeline.intervals, [10, 10, 10, 10])
    # A single frame has no time step:
    timeline = _frame_timeline([7.5])
    npt.assert_equal(timeline.indices, [0])
    npt.assert_almost_equal(timeline.intervals, [SINGLE_FRAME_INTERVAL])
    # An irregular axis keeps its unequal time steps:
    timeline = _frame_timeline([0, 0.45, 0.55, 166.67])
    npt.assert_almost_equal(timeline.intervals, [0.45, 0.1, 166.12, 166.12],
                            decimal=6)

    # Duration stays 40 ms at any display rate:
    for fps, n_frames in [(25, 1), (50, 2), (100, 4), (200, 8)]:
        timeline = _frame_timeline([0, 10, 20, 30], fps=fps)
        npt.assert_equal(timeline.indices.size, n_frames)
        npt.assert_almost_equal(timeline.intervals, [1000.0 / fps] * n_frames)
        npt.assert_almost_equal(timeline.intervals.sum(), 40, decimal=6)
    # Zero-order hold: each display frame shows the most recent source frame;
    # skipped frames are dropped, not blended:
    npt.assert_equal(_frame_timeline([0, 10, 20, 30], fps=200).indices,
                     [0, 0, 1, 1, 2, 2, 3, 3])
    npt.assert_equal(_frame_timeline([0, 10, 20, 30], fps=50).indices, [0, 2])
    # Irregular axis: the frame shown from 0.45 to 0.55 ms falls between two
    # 30 fps display samples and is dropped:
    npt.assert_equal(_frame_timeline([0, 0.45, 0.55, 166.67], fps=30).indices,
                     [0, 2, 2, 2, 2, 2, 3, 3, 3, 3])

    # fps accepts any frequency unit:
    for spelling in (25 * Hz, 0.025 * kHz):
        npt.assert_equal(_frame_timeline([0, 10, 20], fps=spelling).indices,
                         _frame_timeline([0, 10, 20], fps=25).indices)
    # Other units are rejected:
    for wrong in (30 * ms, 30 * uA, 30 * dva):
        with pytest.raises(DimensionMismatchError):
            _frame_timeline([0, 10, 20], fps=wrong)
    for wrong in (0, -30):
        with pytest.raises(ValueError):
            _frame_timeline([0, 10, 20], fps=wrong)
    with pytest.raises(ValueError):
        _frame_timeline([])


def test_frame_timeline_rejects_unordered_time():
    """Time axes that are unsorted, repeated, or non-finite are rejected"""
    for wrong in ([0, 10, 5], [0, 10, 10], [10, 0], [0, np.nan, 10],
                  [0, np.inf]):
        for fps in (None, 30):
            with pytest.raises(ValueError):
                _frame_timeline(wrong, fps=fps)


def test_frame_timeline_last_frame():
    """The last frame is held for the preceding interval"""
    for time in ([0, 10, 20], [0, 10, 30], [0, 0.45, 0.55, 166.67]):
        timeline = _frame_timeline(time)
        npt.assert_almost_equal(timeline.intervals[-1],
                                timeline.intervals[-2])
        npt.assert_array_less(0, timeline.intervals)
        # `n` frames of `dt` take `n * dt`:
        npt.assert_almost_equal(_frame_timeline([0, 10, 20]).intervals.sum(),
                                30)
        # A fine display clock reaches the last frame:
        npt.assert_equal(_frame_timeline(time, fps=1000).indices[-1],
                         len(time) - 1)


def test_frame_timeline_does_not_mutate():
    """The timeline does not alias the input time array"""
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
    # Frames are embedded at native size; the browser magnifies them:
    npt.assert_equal((cfg['fh'], cfg['fw']), (6, 8))
    # The sheet holds every frame:
    n_rows = int(np.ceil(n_frames / cfg['ncols']))
    npt.assert_equal(sheet.size, (cfg['ncols'] * cfg['sw'],
                                  n_rows * cfg['sh']))
    if fmt == 'png':
        # Scalar data is a palettized PNG (one byte per pixel), unpadded:
        npt.assert_equal(sheet.mode, 'P')
        npt.assert_equal((cfg['sh'], cfg['sw']), (cfg['fh'], cfg['fw']))
    else:
        # Gray colormap has no chroma, so frames align to 8x8 DCT blocks:
        npt.assert_equal(sheet.mode, 'L')
        npt.assert_equal((cfg['sh'], cfg['sw']), (8, 8))
    # The image rect lies inside the static background:
    npt.assert_equal(bg.size, (800, 500))
    x, y, w, h = cfg['rect']
    npt.assert_equal(x >= 0 and y >= 0, True)
    npt.assert_equal(x + w <= 800 and y + h <= 500, True)


@pytest.mark.parametrize('shape', ((6, 8), (61, 91), (37, 37), (13, 100)))
def test_HTMLAnimation_rect_covers_image(shape):
    """The drawn rect covers the whole image, with no gap at the edges"""
    ani = make_ani(np.random.rand(*shape, 3))
    cfg, bg, _ = parse(ani.to_jshtml())
    bbox = ani._image.get_window_extent()
    height = bg.size[1]
    x, y, w, h = cfg['rect']
    npt.assert_equal(x <= bbox.x0 and x + w >= bbox.x1, True)
    npt.assert_equal(y <= height - bbox.y1, True)
    npt.assert_equal(y + h >= height - bbox.y0, True)
    # Overshoot is less than 2 pixels:
    npt.assert_equal(w - (bbox.x1 - bbox.x0) < 2, True)
    npt.assert_equal(h - (bbox.y1 - bbox.y0) < 2, True)


def test_HTMLAnimation_frame_values():
    """Every frame is on the sheet with the correct gray levels"""
    n_frames = 7
    data = np.linspace(0, 1, 4 * 5 * n_frames).reshape((4, 5, n_frames))
    cfg, _, sheet = parse(make_ani(data, fmt='png').to_jshtml())
    for i in range(n_frames):
        # Matplotlib quantizes to 256 levels before the colormap lookup:
        expected = np.clip(data[..., i] / data.max() * 256, 0, 255)
        npt.assert_equal(tile(cfg, sheet, i), expected.astype(np.uint8))
    # JPEG is lossy, but within 16 gray levels:
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
        # Chroma is subsampled in 16x16 macroblocks:
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
    """Four-channel data is encoded as RGBA

    ``Image.fromarray(sheet, mode='RGB')`` on an RGBA array reads 4-byte pixels
    3 bytes at a time, which scrambles the channels.
    """
    data = np.zeros((6, 8, 4, 5), dtype=np.float32)
    data[:, :, 0, :] = 1.0     # pure red ...
    data[:, :, 3, :] = 0.25    # ... at 25% opacity
    cfg, _, sheet = parse(make_ani(data, fmt=fmt).to_jshtml())
    npt.assert_equal((cfg['fh'], cfg['fw']), (6, 8))
    for i in range(5):
        tile_i = tile(cfg, sheet, i).astype(float)
        if fmt == 'png':
            # PNG keeps alpha for the canvas to composite:
            npt.assert_equal(sheet.mode, 'RGBA')
            npt.assert_array_less(
                np.abs(tile_i - [255, 0, 0, 64]).max(axis=-1), 2)
        else:
            # JPEG frames are flattened onto the white axes background, as
            # Matplotlib rasterizes them:
            npt.assert_equal(sheet.mode, 'RGB')
            npt.assert_array_less(
                np.abs(tile_i - [255, 191, 191]).max(axis=-1), 8)


@pytest.mark.parametrize('shape', ((65, 97), (30, 40), (16, 16), (13, 11)))
def test_HTMLAnimation_no_frame_bleed(shape):
    """JPEG frames do not bleed into neighboring frames on the sheet"""
    n_frames = 8
    for data in [np.zeros((*shape, n_frames)),
                 np.zeros((*shape, 3, n_frames))]:
        data[..., 1::2] = 1.0    # alternate black and white frames
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
    # The title band is above the image:
    npt.assert_equal(cfg['title']['rect'][1] + cfg['title']['rect'][3] <=
                     cfg['rect'][1], True)
    # Without labels, the title is not redrawn:
    cfg, _, _ = parse(make_ani(data).to_jshtml())
    npt.assert_equal(cfg['title'], None)
    npt.assert_equal(cfg['labels'], [])


def title_ink(label, dpi=100):
    """Return the pixel extent of a Matplotlib-rendered axes title"""
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
    """The cleared title band covers a full line of text

    A band that is too short leaves old titles on the canvas, where they pile
    up over frames.
    """
    cfg, _, _ = parse(make_ani(np.random.rand(4, 4, 3),
                               labels=[label] * 3).to_jshtml())
    _, top, _, height = cfg['title']['rect']
    y0, y1, x0, x1 = title_ink(label)
    npt.assert_equal(top <= y0 and y1 < top + height, True)
    # Text is anchored where Matplotlib puts it:
    npt.assert_equal(cfg['title']['align'], 'center')
    npt.assert_array_less(abs((x0 + x1) / 2 - cfg['title']['x']), 2)


def test_HTMLAnimation_title_not_in_background():
    """With labels, an existing axes title is excluded from the background

    The background is a static ``<img>``, so a title rendered into it shows
    through every frame.
    """
    data = np.random.rand(4, 4, 3)
    labels = ['t = 0.00 ms', 't = 1.00 ms', 't = 2.00 ms']
    ani = make_ani(data, labels=labels)
    reference = np.asarray(parse(ani.to_jshtml())[1])
    for stale in ('t = 999.00 ms', 'A title'):
        ani = make_ani(data, labels=labels)
        ani._image.axes.set_title(stale)
        npt.assert_equal(np.asarray(parse(ani.to_jshtml())[1]), reference)
        # The existing title is restored:
        npt.assert_equal(ani._image.axes.get_title(), stale)
    # Without labels, the axes title is part of the background:
    ani = make_ani(data)
    ani._image.axes.set_title('A title')
    npt.assert_equal(np.any(np.asarray(parse(ani.to_jshtml())[1]) != reference),
                     True)


def make_indexed(data, index):
    """Return an HTMLAnimation whose display frames index into ``data``"""
    fig, ax = plt.subplots(figsize=(8, 5))
    mat = ax.imshow(np.zeros(data.shape[:-1]), cmap='gray', vmin=0,
                    vmax=data.max())
    plt.close(fig)
    return HTMLAnimation(fig, lambda d: mat, iter(range(len(index))),
                         save_count=len(index), image=mat, frame_data=data,
                         frame_index=[index], interval=25.0, fmt='png')


def test_HTMLAnimation_packs_each_frame_once():
    """Only displayed source frames are packed, each once"""
    n_src = 6
    data = np.linspace(0, 1, 4 * 5 * n_src).reshape((4, 5, n_src))
    # Repeated frames reuse one tile:
    repeated = np.repeat(np.arange(n_src), 2)
    cfg, _, sheet = parse(make_indexed(data, repeated).to_jshtml())
    npt.assert_equal(cfg['n'], 2 * n_src)
    npt.assert_equal(cfg['map'], list(repeated))
    # Skipped frames are omitted, and the map is renumbered:
    skipped = [0, 2, 4]
    small_cfg, _, small = parse(make_indexed(data, skipped).to_jshtml())
    npt.assert_equal(small_cfg['map'], [0, 1, 2])
    for i, src in enumerate(skipped):
        # Matplotlib quantizes to 256 levels before the colormap lookup:
        expected = np.clip(data[..., src] / data.max() * 256, 0, 255)
        npt.assert_equal(tile(small_cfg, small, i), expected.astype(np.uint8))
    # Fewer frames give a smaller sheet:
    npt.assert_equal(np.prod(small.size) < np.prod(sheet.size), True)


def test_HTMLAnimation_playback():
    data = np.random.rand(4, 4, 3)
    # 'repeat' picks the default loop mode:
    npt.assert_equal(parse(make_ani(data).to_jshtml())[0]['mode'], 'loop')
    once = parse(make_ani(data, repeat=False).to_jshtml())[0]
    npt.assert_equal(once['mode'], 'once')
    ani = make_ani(data, interval=40.0)
    npt.assert_almost_equal(parse(ani.to_jshtml())[0]['interval'], 40.0)
    # 'fps' and 'default_mode' override the animation settings:
    cfg, _, _ = parse(ani.to_jshtml(fps=10, default_mode='reflect'))
    npt.assert_almost_equal(cfg['interval'], 100.0)
    npt.assert_equal(cfg['mode'], 'reflect')


def test_HTMLAnimation_per_frame_intervals():
    """Per-frame intervals are passed to the player"""
    data = np.random.rand(4, 4, 3)
    intervals = [0.45, 165.67, 0.45]
    cfg, _, _ = parse(make_ani(data, intervals=intervals).to_jshtml())
    npt.assert_almost_equal(cfg['intervals'], intervals)
    # Matplotlib (`save`, `to_html5_video`) uses the mean delay:
    ani = make_ani(data, intervals=intervals)
    npt.assert_almost_equal(ani._interval, np.mean(intervals))
    # A constant delay is still the default:
    cfg, _, _ = parse(make_ani(data, interval=40.0).to_jshtml())
    npt.assert_almost_equal(cfg['intervals'], [40.0] * 3)
    # Explicit `fps` overrides the timing (as in Matplotlib) without
    # resampling frames:
    cfg, _, _ = parse(make_ani(data, intervals=intervals).to_jshtml(fps=10))
    npt.assert_almost_equal(cfg['intervals'], [100.0] * 3)
    npt.assert_equal(cfg['n'], 3)
    # Requires one delay per frame:
    with pytest.raises(ValueError):
        make_ani(data, intervals=[10, 20])



def test_HTMLAnimation_smoothing():
    # Strongly magnified frames use nearest-neighbor, as in Matplotlib's
    # 'antialiased' interpolation:
    small = parse(make_ani(np.random.rand(4, 4, 3)).to_jshtml())[0]
    npt.assert_equal(small['smooth'], False)
    # Frames shown near native size are interpolated:
    large = parse(make_ani(np.random.rand(300, 300, 3)).to_jshtml())[0]
    npt.assert_equal(large['smooth'], True)


def test_HTMLAnimation_html():
    data = np.random.rand(4, 4, 3)
    html = make_ani(data, labels=['a', 'b', 'c']).to_jshtml()
    # No external resources, and everything is scoped to a unique id so that
    # several animations can share a notebook:
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
    # The background is always PNG (text and thin lines compress poorly as
    # JPEG):
    npt.assert_equal(jpg.count('data:image/png;base64,'), 1)
    # 'jpeg' is an alias; unknown formats are rejected:
    ani = make_ani(np.random.rand(4, 4, 2), fmt='JPEG')
    npt.assert_equal(ani._fmt, 'jpg')
    npt.assert_equal('data:image/jpeg;base64,' in ani.to_jshtml(), True)
    with pytest.raises(ValueError):
        make_ani(data, fmt='gif')


def test_HTMLAnimation_caching():
    ani = make_ani(np.random.rand(4, 4, 3))
    html = ani.to_jshtml()
    # The same call returns the cached player:
    npt.assert_equal(ani.to_jshtml(), html)
    npt.assert_equal(ani._repr_html_(), html)
    # Changing playback settings rebuilds it:
    npt.assert_equal(ani.to_jshtml(fps=1) != html, True)


def test_HTMLAnimation_matplotlib_compat():
    data = np.random.rand(4, 4, 3)
    ani = make_ani(data)
    npt.assert_equal(isinstance(ani, FuncAnimation), True)
    npt.assert_equal(len(list(ani.frame_seq)), 3)
    npt.assert_equal('p2p-anim' in ani.to_jshtml(), True)
    # Without frame data, Matplotlib's player is used:
    ani = make_ani(data)
    ani._layers = None
    html = ani.to_jshtml()
    npt.assert_equal('<script' in html, True)
    npt.assert_equal('p2p-anim' in html, False)


def make_layered_ani(src, percept, src_index):
    """Return two animated images in one figure, as in
    ``play_stimulus_percept``"""
    fig, axes = plt.subplots(ncols=2)
    images = [ax.imshow(np.zeros(data.shape[:-1]), cmap='gray', vmin=0,
                        vmax=data.max())
              for ax, data in zip(axes, (src, percept))]
    plt.close(fig)
    n_frames = percept.shape[-1]
    return HTMLAnimation(fig, lambda i: images, iter(range(n_frames)),
                         save_count=n_frames, image=images,
                         frame_data=[src, percept],
                         frame_index=[src_index, None], interval=25.0,
                         fmt='png')


def parse_layers(html):
    """Return the player config and one sprite sheet per animated image"""
    cfg = json.loads(re.search(r'var cfg = (\{.*?\});', html, re.S).group(1))
    sheets = [Image.open(BytesIO(base64.b64decode(b64)))
              for b64 in re.findall(r'data:image/\w+;base64,([A-Za-z0-9+/=]+)',
                                    html)]
    # The first embedded image is the static background:
    return cfg, sheets[1:]


def test_HTMLAnimation_layers():
    """Each animated image has its own sheet and frame map"""
    src = np.linspace(0, 1, 4 * 5 * 3).reshape((4, 5, 3))
    percept = np.linspace(0, 1, 6 * 6 * 4).reshape((6, 6, 4))
    src_index = [0, 0, 1, 2]
    cfg, sheets = parse_layers(
        make_layered_ani(src, percept, src_index).to_jshtml())
    npt.assert_equal(cfg['n'], 4)
    npt.assert_equal(len(sheets), 2)
    # The source uses a frame map; the percept advances one frame at a time:
    npt.assert_equal(cfg['layers'][0]['map'], src_index)
    npt.assert_equal(cfg['layers'][1]['map'], None)
    # Each panel is drawn from its own sheet into its own rect:
    for layer, sheet, data in zip(cfg['layers'], sheets, (src, percept)):
        for i in range(data.shape[-1]):
            expected = np.clip(data[..., i] / data.max() * 256, 0, 255)
            npt.assert_equal(tile(layer, sheet, i), expected.astype(np.uint8))
    left, right = cfg['layers']
    npt.assert_equal(left['rect'][0] + left['rect'][2] <= right['rect'][0],
                     True)


@pytest.mark.parametrize('fmt', ('png', 'jpg'))
def test_HTMLAnimation_overlapping_layers(fmt):
    """A transparent layer drawn over another keeps its alpha"""
    fig, ax = plt.subplots()
    frames = np.random.rand(6, 8, 3)
    overlay = np.zeros((6, 8, 4, 1), dtype=np.float32)
    overlay[2, :, :, 0] = (1, 0, 0, 1)
    images = [ax.imshow(np.zeros((6, 8)), cmap='gray', vmin=0, vmax=1),
              ax.imshow(overlay[..., 0])]
    plt.close(fig)
    html = HTMLAnimation(fig, lambda i: images, iter(range(3)), save_count=3,
                         image=images, frame_data=[frames, overlay],
                         frame_index=[None, [0, 0, 0]], fmt=fmt).to_jshtml()
    cfg, (under, over) = parse_layers(html)
    npt.assert_equal(cfg['layers'][0]['rect'], cfg['layers'][1]['rect'])
    # The top layer stays RGBA PNG even when the layer below is JPEG:
    npt.assert_equal(under.format, 'JPEG' if fmt == 'jpg' else 'PNG')
    npt.assert_equal((over.format, over.mode), ('PNG', 'RGBA'))
    alpha = tile(cfg['layers'][1], over, 0)[..., 3]
    npt.assert_equal(alpha[2].min(), 255)
    npt.assert_equal(np.delete(alpha, 2, axis=0).max(), 0)
    # All layers are cleared before any is drawn:
    draw = re.search(r'function draw\(\) \{(.*?)\n  \}', html, re.S).group(1)
    loops = [m.start() for m in re.finditer(r'cfg\.layers\.forEach', draw)]
    npt.assert_equal(len(loops), 2)
    npt.assert_equal(loops[0] < draw.index('clearRect') < loops[1] <
                     draw.index('drawImage'), True)


def test_HTMLAnimation_clips_images_to_their_axes():
    """Images are clipped to their axes and stay out of the title band

    Pixel-edge extents reach half a pixel past axes set to pixel centers, as in
    `Percept.play`.
    """
    fig, ax = plt.subplots()
    data = np.random.rand(4, 8, 3)
    im = ax.imshow(np.zeros((4, 8)), cmap='gray', vmin=0, vmax=1,
                   extent=(-0.5, 7.5, -0.5, 3.5))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 3)
    ax.set_title('t = 0')
    plt.close(fig)
    html = HTMLAnimation(fig, lambda i: im, iter(range(3)), save_count=3,
                         image=im, frame_data=data, labels=['a', 'b', 'c'],
                         fmt='png').to_jshtml()
    cfg, _, _ = parse(html)
    title, rect, crop = cfg['title'], cfg['rect'], cfg['crop']
    # The drawn rect is the axes box, below the title band:
    box = ax.get_window_extent()
    height = fig.bbox.height
    npt.assert_allclose(rect, [box.x0, height - box.y1, box.width,
                               box.height], atol=1)
    npt.assert_equal(rect[1] >= title['rect'][1] + title['rect'][3], True)
    # The source is cropped by half a data pixel on each side:
    npt.assert_equal((cfg['fw'], cfg['fh']), (8, 4))
    npt.assert_allclose(crop, [0.5, 0.5, 7, 3], atol=0.05)


def test_HTMLAnimation_layers_agree_on_frame_count():
    src = np.random.rand(4, 5, 3)
    percept = np.random.rand(6, 6, 4)
    # A mapped layer's length is the length of its map:
    make_layered_ani(src, percept, [0, 1, 2, 2])
    # All layers require the same number of display frames:
    with pytest.raises(ValueError):
        make_layered_ani(src, percept, [0, 1, 2])
    with pytest.raises(ValueError):
        make_layered_ani(src, percept, None)


def test_HTMLAnimation_single_layer_aliases():
    """Single-image animations expose _image and _frame_data"""
    data = np.random.rand(4, 4, 3)
    ani = make_ani(data)
    npt.assert_equal(ani._image is ani._layers[0].image, True)
    npt.assert_almost_equal(ani._frame_data, data)
