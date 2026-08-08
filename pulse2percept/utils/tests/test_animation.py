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

from pulse2percept.utils import HTMLAnimation
from pulse2percept.utils.animation import (MAX_SPRITE_PX, _sprite_grid,
                                           _frame_shape, _weight2css)


def make_ani(data, labels=None, interval=25.0, repeat=True, colorbar=False):
    """Set up an HTMLAnimation the same way ``Percept.play`` does"""
    fig, ax = plt.subplots(figsize=(8, 5))
    frame0 = np.zeros(data.shape[:-1])
    mat = ax.imshow(frame0, cmap='gray', vmin=0, vmax=data.max())
    if colorbar:
        fig.colorbar(mat)
    plt.close(fig)
    return HTMLAnimation(fig, lambda d: mat, iter(range(data.shape[-1])),
                         interval=interval, save_count=data.shape[-1],
                         repeat=repeat, image=mat, frame_data=data,
                         labels=labels)


def parse(html):
    """Pull the player config and the two embedded PNGs out of the HTML"""
    cfg = json.loads(re.search(r'var cfg = (\{.*?\});', html, re.S).group(1))
    pngs = [Image.open(BytesIO(base64.b64decode(b64)))
            for b64 in re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',
                                  html)]
    return cfg, pngs[0], pngs[1]


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
            npt.assert_equal(max(n_rows * height, n_cols * width)
                             <= max(n_frames * height, width), True)


def test_frame_shape():
    # Frames are never upsampled:
    npt.assert_equal(_frame_shape((10, 20), 5, (100, 200)), (10, 20))
    # ... but are downsampled to the size at which they are displayed:
    npt.assert_equal(_frame_shape((100, 200), 5, (50, 100)), (50, 100))
    # Aspect ratio is preserved:
    npt.assert_equal(_frame_shape((100, 200), 5, (50, 400)), (50, 100))
    # Huge stacks are shrunk until the sheet fits what browsers can decode:
    height, width = _frame_shape((2000, 2000), 1000, (2000, 2000))
    n_rows, n_cols = _sprite_grid(1000, height, width)
    npt.assert_equal(max(n_rows * height, n_cols * width) <= MAX_SPRITE_PX,
                     True)


def test_weight2css():
    npt.assert_equal(_weight2css('normal'), 'normal')
    npt.assert_equal(_weight2css('light'), 'normal')
    npt.assert_equal(_weight2css('bold'), 'bold')
    npt.assert_equal(_weight2css('demibold'), 'bold')
    npt.assert_equal(_weight2css(700), '700')


@pytest.mark.parametrize('n_frames', (1, 2, 5, 17))
def test_HTMLAnimation_sprite_sheet(n_frames):
    data = np.random.rand(6, 8, n_frames)
    ani = make_ani(data)
    cfg, bg, sheet = parse(ani.to_jshtml())
    npt.assert_equal(cfg['n'], n_frames)
    # Frames are embedded at their native size (they are magnified for
    # display, and the browser can do that for free):
    npt.assert_equal((cfg['fh'], cfg['fw']), (6, 8))
    # The sheet must be large enough to hold every frame:
    n_rows = int(np.ceil(n_frames / cfg['ncols']))
    npt.assert_equal(sheet.size, (cfg['ncols'] * 8, n_rows * 6))
    # Scalar data is shipped as a palettized PNG (one byte per pixel):
    npt.assert_equal(sheet.mode, 'P')
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
    bbox = ani._image.get_window_extent()
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
    ani = make_ani(data)
    cfg, _, sheet = parse(ani.to_jshtml())
    sheet = np.asarray(sheet)
    for i in range(n_frames):
        col, row = i % cfg['ncols'], i // cfg['ncols']
        tile = sheet[row * 4:(row + 1) * 4, col * 5:(col + 1) * 5]
        # Matplotlib quantizes to 256 levels before the colormap lookup:
        expected = np.clip(data[..., i] / data.max() * 256, 0, 255)
        npt.assert_equal(tile, expected.astype(np.uint8))


def test_HTMLAnimation_rgb():
    data = np.random.rand(4, 6, 3, 5)
    ani = make_ani(data)
    cfg, _, sheet = parse(ani.to_jshtml())
    npt.assert_equal(sheet.mode, 'RGB')
    npt.assert_equal((cfg['fh'], cfg['fw']), (4, 6))
    sheet = np.asarray(sheet)
    for i in range(5):
        col, row = i % cfg['ncols'], i // cfg['ncols']
        tile = sheet[row * 4:(row + 1) * 4, col * 6:(col + 1) * 6]
        npt.assert_equal(tile, (data[..., i] * 255).astype(np.uint8))


def test_HTMLAnimation_labels():
    data = np.random.rand(4, 4, 3)
    labels = ['t = 0.00 ms', 't = 1.00 ms', 't = 2.00 ms']
    cfg, _, _ = parse(make_ani(data, labels=labels).to_jshtml())
    npt.assert_equal(cfg['labels'], labels)
    npt.assert_equal(cfg['title'] is not None, True)
    # The title band sits above the image and spans the whole figure:
    npt.assert_equal(cfg['title']['rect'][1] + cfg['title']['rect'][3]
                     <= cfg['rect'][1], True)
    # Without labels, the player leaves the title alone:
    cfg, _, _ = parse(make_ani(data).to_jshtml())
    npt.assert_equal(cfg['title'], None)
    npt.assert_equal(cfg['labels'], [])


def test_HTMLAnimation_playback():
    data = np.random.rand(4, 4, 3)
    # 'repeat' picks the default loop mode:
    npt.assert_equal(parse(make_ani(data).to_jshtml())[0]['mode'], 'loop')
    npt.assert_equal(parse(make_ani(data, repeat=False).to_jshtml())[0]['mode'],
                     'once')
    ani = make_ani(data, interval=40.0)
    npt.assert_almost_equal(parse(ani.to_jshtml())[0]['interval'], 40.0)
    # 'fps' and 'default_mode' override the animation's own settings:
    cfg, _, _ = parse(ani.to_jshtml(fps=10, default_mode='reflect'))
    npt.assert_almost_equal(cfg['interval'], 100.0)
    npt.assert_equal(cfg['mode'], 'reflect')


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
    ani._frame_data = None
    html = ani.to_jshtml()
    npt.assert_equal('<script' in html, True)
    npt.assert_equal('p2p-anim' in html, False)
