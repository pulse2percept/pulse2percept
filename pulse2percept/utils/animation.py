""":py:class:`~pulse2percept.utils.animation.HTMLAnimation`

Fast, dependency-free HTML/JavaScript animations.

:py:class:`HTMLAnimation` renders the static parts of the figure exactly once
and packs all frames into a single, color-mapped sprite sheet that is blitted
into a ``<canvas>`` by a small vanilla-JavaScript player. This is typically two
orders of magnitude faster and produces much smaller notebooks and doc pages.

The sheet is encoded as JPEG by default, which roughly halves it again; pass
``fmt='png'`` if you need the frames to be pixel-exact.
"""
import base64
from collections import namedtuple
from io import BytesIO
from json import dumps
from string import Template
from uuid import uuid4

import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_hex, to_rgba
from PIL import Image

from ..units import Hz, as_value

__all__ = ['HTMLAnimation', 'FrameTimeline', 'frame_interval',
           'frame_timeline']

# Frames are packed into a single sprite sheet. Browsers put a cap on the size
# of an image they are willing to decode; 8192px per side is safe everywhere,
# including on mobile:
MAX_SPRITE_PX = 8192

# Matplotlib's 'antialiased' interpolation falls back to 'nearest' once an
# image is magnified by more than this factor (see ``Image._make_image``). The
# canvas player mirrors that behavior so that percepts keep their crisp,
# pixelated look:
MAX_SMOOTH_UPSAMPLE = 3

# Number of gray/color levels in the sprite sheet. Matplotlib quantizes to 256
# levels before the colormap lookup as well, so nothing is lost here:
N_LEVELS = 256

# PNG compression level. Anything above this buys a few percent in size for
# several times the encoding time:
PNG_COMPRESS_LEVEL = 3

# JPEG quality. High enough that the artifacts stay invisible next to the blur
# of a phosphene (a mean error of about 1 gray level out of 255), while still
# cutting the size of the sprite sheet roughly in half:
JPEG_QUALITY = 90

# Frames are padded so that no JPEG block ever straddles two frames of the
# sprite sheet, which would bleed one frame into the next. Grayscale sheets are
# coded in 8x8 DCT blocks; color sheets additionally use 4:2:0 chroma
# subsampling, whose macroblocks are 16x16:
JPEG_BLOCK = 8
JPEG_MACROBLOCK = 16

# Frame duration (in ms) to fall back on for single-frame animations, which
# have no time step of their own:
SINGLE_FRAME_INTERVAL = 1000.0 / 30

# The title is blank while the animation is built, and an empty ``Text`` has a
# degenerate bounding box. Its geometry is therefore measured on a probe
# string, which must carry both an ascender and a descender so that the band
# the player clears covers a full line of text:
TITLE_PROBE = 'Ag'

# Pixels of slack added to that band, to catch antialiasing:
TITLE_MARGIN = 1


def frame_interval(time, fps=None, tol=1e-2):
    """Determine the delay between two frames of an animation

    .. versionadded:: 0.10.0

    Parameters
    ----------
    time : array_like
        The time points of the animation (in ms)
    fps : float or None
        Frames per second. If None, the interval is inferred from ``time``,
        which is not supported for a non-homogeneous time axis.

        May be given as a plain number of hertz or as a unitful frequency
        (e.g. ``0.03 * kHz``); see :py:mod:`pulse2percept.units`.
    tol : float, optional
        Tolerance within which two time steps count as equal

    Returns
    -------
    interval : float
        The delay between two frames (in ms). A single-frame animation has no
        time step of its own and falls back on ``SINGLE_FRAME_INTERVAL``.

    """
    # Every ``play`` in p2p ends up here for its frame timing, so this is the
    # one place a frame rate has to be turned into a plain number of hertz. A
    # rate is a frequency, so `30 * Hz` and `0.03 * kHz` are the same argument
    # as `30`, and `30 * ms` is not an argument at all:
    fps = as_value(fps, Hz, 'fps')
    if fps is not None:
        return 1000.0 / fps
    interval = np.diff(np.asarray(time, dtype=np.float64))
    if interval.size == 0:
        # A single frame has no time step, and there is nothing to advance to,
        # so any interval will do:
        return SINGLE_FRAME_INTERVAL
    # Compare the steps against each other rather than quantizing each one onto
    # a grid of `tol`: a step that lands exactly on a grid boundary (33.365 ms
    # against tol=1e-2, which is what a 29.97 fps percept comes out at) rounds
    # up or down depending on floating-point noise far below `tol`, and an
    # evenly spaced axis then looks like two different steps.
    spread = float(interval.max() - interval.min())
    if spread > tol:
        raise NotImplementedError(
            f"Cannot infer the frame rate from a non-homogeneous time axis "
            f"(time steps range over {spread:g} ms, more than tol={tol:g}). "
            f"Pass 'fps' instead.")
    return float(interval[0])


#: The frames of an animation, laid out on a display clock: which frame of the
#: source data to show (``indices``), when it comes up (``times``, in ms), and
#: how long it stays up (``intervals``, in ms).
FrameTimeline = namedtuple('FrameTimeline', ['indices', 'times', 'intervals'])


def frame_timeline(time, fps=None):
    """Lay the frames of an animation out on a display clock

    .. versionadded:: 0.10.0

    ``time`` owns the timing: an animation that spans three seconds takes
    three seconds to play, whatever ``fps`` says. ``fps`` only decides how
    often that timeline is sampled for display.

    Parameters
    ----------
    time : array_like
        The time points of the animation (in ms), in increasing order. May be
        non-homogeneous (e.g., the short phases and long gaps of a pulse
        train).
    fps : float or None
        The rate at which the timeline is sampled for display. If None, every
        frame is shown for as long as ``time`` says it lasts. Otherwise the
        timeline is resampled onto a regular clock of ``1000 / fps`` ms with a
        zero-order hold: each display frame shows the most recent frame that
        was due. Either way, the animation lasts the same wall-clock time.

        May be given as a plain number of hertz or as a unitful frequency
        (e.g. ``0.03 * kHz``); see :py:mod:`pulse2percept.units`.

    Returns
    -------
    timeline : FrameTimeline
        A named tuple of ``indices`` (which frame of ``time`` to show),
        ``times`` (when each displayed frame comes up, in ms), and
        ``intervals`` (how long each displayed frame stays up, in ms).

    Notes
    -----
    *  The last frame's duration does not follow from ``time`` alone (there is
       no timestamp after it to end it), so it is held for as long as the
       frame before it. A single-frame animation falls back on
       ``SINGLE_FRAME_INTERVAL``.
    *  Frames are never interpolated, pooled, or otherwise mixed: an event
       that falls between two display samples is simply missed, exactly as it
       would be on a display of that frame rate. Sample faster if that matters.

    Examples
    --------
    Playing a percept at its own rate shows every frame for its own time step:

    >>> frame_timeline([0, 10, 30]).indices
    array([0, 1, 2])
    >>> frame_timeline([0, 10, 30]).intervals
    array([10., 20., 20.])

    Halving the display rate halves the number of frames, not the duration:

    >>> frame_timeline([0, 10, 20, 30], fps=50).indices
    array([0, 2])

    """
    fps = as_value(fps, Hz, 'fps')
    # A copy: the timeline is handed out, and the caller's time axis is not
    # ours to modify:
    time = np.array(time, dtype=np.float64).ravel()
    if time.size == 0:
        raise ValueError("'time' must have at least one time point.")
    intervals = np.diff(time)
    # The last frame has no timestamp after it, so it is held for as long as
    # the one before it. That is also what makes a homogeneous axis last
    # exactly `n_frames` time steps:
    intervals = np.append(intervals, intervals[-1] if intervals.size
                          else SINGLE_FRAME_INTERVAL)
    if fps is None:
        return FrameTimeline(np.arange(time.size), time, intervals)
    if fps <= 0:
        raise ValueError(f"'fps' must be greater than zero, not {fps}.")
    step = 1000.0 / fps
    # Same wall-clock duration, sampled at the display rate. Rounded rather
    # than rounded up, so that an axis which fits the display clock exactly
    # cannot pick up a spurious extra frame from floating-point noise:
    n_frames = max(1, int(np.floor(intervals.sum() / step + 0.5)))
    times = time[0] + np.arange(n_frames) * step
    # Zero-order hold: show whichever frame was most recently due. Values are
    # never blended, so a brief event between two samples is missed:
    indices = np.clip(np.searchsorted(time, times, side='right') - 1, 0,
                      time.size - 1)
    return FrameTimeline(indices, times, np.full(n_frames, step))


def _weight2css(weight):
    """Translate a Matplotlib font weight into a CSS font weight"""
    if isinstance(weight, (int, np.integer)):
        return str(int(weight))
    if str(weight).lower() in ('semibold', 'demibold', 'demi', 'bold', 'heavy',
                               'extra bold', 'black'):
        return 'bold'
    return 'normal'


def _check_fmt(fmt):
    """Normalize and validate the sprite sheet image format"""
    known = {'jpg': 'jpg', 'jpeg': 'jpg', 'png': 'png'}
    normalized = known.get(str(fmt).lower())
    if normalized is None:
        raise ValueError(f"Unknown image format '{fmt}'. Choose either 'jpg' "
                         f"or 'png'.")
    return normalized


def _round_up(value, multiple):
    """Round ``value`` up to the next multiple of ``multiple``"""
    return int(np.ceil(value / multiple)) * multiple


def _sprite_grid(n_frames, height, width):
    """Lay out ``n_frames`` frames in a roughly square sprite sheet

    Returns the number of rows and columns of the sheet. A square-ish sheet
    keeps both dimensions as small as possible, which is what browsers care
    about.
    """
    n_cols = max(1, int(np.ceil(np.sqrt(n_frames * height / width))))
    n_cols = min(n_frames, n_cols)
    n_rows = int(np.ceil(n_frames / n_cols))
    return n_rows, n_cols


def _frame_shape(data_shape, n_frames, max_shape, pad_to=1):
    """Determine the size at which each frame is embedded

    Frames are never upsampled (the browser does that for free) and are
    downsampled if they are either larger than the area they are displayed in
    or too large to fit in a sprite sheet. ``pad_to`` is the multiple that
    each tile is padded to on the sheet.
    """
    height, width = data_shape
    # No point in shipping more pixels than are actually displayed:
    scale = min(1.0, max_shape[0] / height, max_shape[1] / width)
    # Shrink further if the sheet would exceed what browsers can decode:
    for _ in range(20):
        out_h = max(1, int(round(height * scale)))
        out_w = max(1, int(round(width * scale)))
        n_rows, n_cols = _sprite_grid(n_frames, _round_up(out_h, pad_to),
                                      _round_up(out_w, pad_to))
        if max(n_rows * _round_up(out_h, pad_to),
               n_cols * _round_up(out_w, pad_to)) <= MAX_SPRITE_PX:
            break
        scale *= 0.75
    return out_h, out_w


def _is_gray(lut):
    """Whether a color lookup table contains nothing but shades of gray"""
    return bool((lut[:, 0] == lut[:, 1]).all() and
                (lut[:, 1] == lut[:, 2]).all())


def _quantize(data, norm):
    """Convert (Y, X, T) scalar data into (T, Y, X) colormap indices"""
    vmin = 0.0 if norm.vmin is None else float(norm.vmin)
    vmax = 1.0 if norm.vmax is None else float(norm.vmax)
    if vmax > vmin:
        scaled = (np.asarray(data, dtype=np.float32) - vmin) / (vmax - vmin)
    else:
        scaled = np.zeros(np.shape(data), dtype=np.float32)
    # Same quantization as ``Colormap.__call__``: the level is the integer part
    # of x * N, clipped to the last level:
    idx = np.clip(scaled * N_LEVELS, 0, N_LEVELS - 1).astype(np.uint8)
    return np.ascontiguousarray(idx.transpose((2, 0, 1)))


def _color_lut(cmap):
    """The colormap as a 256x3 table of 8-bit RGB values"""
    lut = cmap((np.arange(N_LEVELS) + 0.5) / N_LEVELS)
    return (np.asarray(lut)[:, :3] * 255).astype(np.uint8)


def _sprite_sheet(data, norm, cmap, max_shape, fmt, bg_color=(255, 255, 255)):
    """Color-map every frame and pack them all into a single image

    Parameters
    ----------
    data : ndarray
        Either (Y, X, T) scalar data, (Y, X, 3, T) RGB data, or (Y, X, 4, T)
        RGBA data in [0, 1]
    norm : matplotlib.colors.Normalize
        The normalization of the animated image (ignored for RGB(A) data)
    cmap : matplotlib.colors.Colormap
        The colormap of the animated image (ignored for RGB(A) data)
    max_shape : (height, width)
        Frames are downsampled to at most this size
    fmt : {'jpg', 'png'}
        Whether to encode the sheet as (lossy) JPEG or (lossless) PNG
    bg_color : (r, g, b), optional
        The 8-bit color that RGBA frames are flattened onto when the sheet
        cannot carry an alpha channel (i.e., for JPEG)

    Returns
    -------
    sheet : dict
        The encoded sheet ('data', 'mime') and its geometry: the number of
        frames per row ('ncols'), the size of a frame ('fw', 'fh'), and the
        distance between two frames on the sheet ('sw', 'sh')
    """
    rgb = np.ndim(data) == 4
    rgba = rgb and np.shape(data)[-2] == 4
    n_frames = np.shape(data)[-1]
    if rgb:
        # (Y, X, C, T) -> (T, Y, X, C), clipped the same way Matplotlib clips
        # out-of-range RGB values:
        scaled = np.clip(np.asarray(data, dtype=np.float32), 0, 1) * 255
        frames = np.ascontiguousarray(
            scaled.transpose((3, 0, 1, 2)).astype(np.uint8))
    else:
        frames = _quantize(data, norm)
    if rgba and fmt == 'jpg':
        # JPEG has no alpha channel, so the frames have to be flattened onto
        # what they are drawn on top of. This is what Matplotlib rasterizes as
        # well; the PNG path keeps the alpha and lets the canvas composite it:
        alpha = frames[..., 3:].astype(np.float32) / 255.0
        flat = frames[..., :3] * alpha + np.asarray(bg_color, dtype=np.float32)\
            * (1.0 - alpha)
        frames = np.ascontiguousarray(np.round(flat).astype(np.uint8))
        rgba = False
    # JPEG has no palette, so scalar data must carry its colors itself. Gray
    # colormaps stay single-channel; anything else is expanded to RGB:
    if fmt == 'jpg' and not rgb:
        lut = _color_lut(cmap)
        frames = lut[..., 0][frames] if _is_gray(lut) else lut[frames]
    if fmt != 'jpg':
        pad_to = 1
    else:
        pad_to = JPEG_MACROBLOCK if frames.ndim == 4 else JPEG_BLOCK
    out_h, out_w = _frame_shape(frames.shape[1:3], n_frames, max_shape, pad_to)
    if (out_h, out_w) != frames.shape[1:3]:
        # Downsample in index space, which is what Matplotlib does as well (it
        # resamples the normalized data before the colormap lookup):
        frames = np.stack([
            np.asarray(Image.fromarray(frame).resize((out_w, out_h),
                                                     Image.BILINEAR))
            for frame in frames])
    # Pad the tiles so that JPEG macroblocks cannot straddle two frames.
    # Repeating the edge pixel keeps the padding from ringing into the frame:
    stride_h, stride_w = _round_up(out_h, pad_to), _round_up(out_w, pad_to)
    if (stride_h, stride_w) != (out_h, out_w):
        pad = [(0, 0), (0, stride_h - out_h), (0, stride_w - out_w)]
        frames = np.pad(frames, pad + [(0, 0)] * (frames.ndim - 3),
                        mode='edge')
    # Tile the frames into a single sheet, filling it row by row:
    n_rows, n_cols = _sprite_grid(n_frames, stride_h, stride_w)
    n_pad = n_rows * n_cols - n_frames
    if n_pad:
        frames = np.concatenate([frames, np.zeros((n_pad, *frames.shape[1:]),
                                                  dtype=np.uint8)])
    sheet = frames.reshape((n_rows, n_cols, *frames.shape[1:]))
    sheet = sheet.swapaxes(1, 2).reshape((n_rows * stride_h, n_cols * stride_w,
                                          *frames.shape[3:]))
    buf = BytesIO()
    if fmt == 'jpg':
        Image.fromarray(sheet, mode='RGB' if sheet.ndim == 3 else 'L').save(
            buf, format='jpeg', quality=JPEG_QUALITY)
    elif rgb:
        Image.fromarray(sheet, mode='RGBA' if rgba else 'RGB').save(
            buf, format='png', compress_level=PNG_COMPRESS_LEVEL)
    else:
        # Ship the colormap as a PNG palette: this keeps the sheet at one byte
        # per pixel no matter how colorful the colormap is:
        img = Image.fromarray(sheet, mode='P')
        img.putpalette(_color_lut(cmap).ravel())
        img.save(buf, format='png', compress_level=PNG_COMPRESS_LEVEL)
    return {'data': buf.getvalue(),
            'mime': 'image/jpeg' if fmt == 'jpg' else 'image/png',
            'ncols': n_cols, 'fw': out_w, 'fh': out_h,
            'sw': stride_w, 'sh': stride_h}


def _background(fig, im):
    """Render everything but the animated image itself

    Returns the PNG bytes of the static figure along with its pixel size.
    """
    visible = im.get_visible()
    im.set_visible(False)
    buf = BytesIO()
    try:
        # ``bbox_inches`` and ``dpi`` must be pinned: the geometry below is in
        # figure pixels, and 'tight' would crop the canvas:
        fig.savefig(buf, format='png', dpi=fig.dpi, bbox_inches=None)
    finally:
        im.set_visible(visible)
    width, height = Image.open(buf).size
    return buf.getvalue(), width, height


def _bg_color(ax):
    """The 8-bit color that the animated image sits on top of

    Only matters for RGBA frames on a JPEG sheet, which cannot carry an alpha
    channel and therefore have to be flattened onto their background.
    """
    axes_rgba = np.asarray(to_rgba(ax.get_facecolor()), dtype=np.float32)
    # A transparent axes patch lets the figure show through:
    fig_rgb = np.asarray(to_rgba(ax.figure.get_facecolor()),
                         dtype=np.float32)[:3]
    alpha = axes_rgba[3]
    return tuple((axes_rgba[:3] * alpha + fig_rgb * (1 - alpha)) * 255)


def _title_geometry(title, width, height, dpi):
    """Locate the title so that the player can redraw it for every frame

    Returns a dict of canvas coordinates/styles, or None if the title cannot be
    measured (e.g., because the figure has not been drawn yet).

    The title is blank at this point so that it does not end up in the static
    background, and an empty ``Text`` has a degenerate bounding box: it sits on
    the baseline and has no height at all. Both the anchor the text is aligned
    to and the line box that has to be cleared before every frame are therefore
    measured on ``TITLE_PROBE`` instead.
    """
    text = title.get_text()
    try:
        title.set_text(TITLE_PROBE)
        bbox = title.get_window_extent()
    except (RuntimeError, ValueError, AttributeError):
        return None
    finally:
        title.set_text(text)
    align = title.get_horizontalalignment()
    if align == 'left':
        x = bbox.x0
    elif align == 'right':
        x = bbox.x1
    else:
        align, x = 'center', (bbox.x0 + bbox.x1) / 2
    return {
        # Only the title's own line box is cleared, so a suptitle or anything
        # else on the figure stays untouched. The band spans the entire figure
        # because the text grows to both sides of its anchor:
        'rect': [0, round(height - bbox.y1 - TITLE_MARGIN), width,
                 max(1, round(bbox.y1 - bbox.y0 + 2 * TITLE_MARGIN))],
        'x': round(x),
        'y': round(height - (bbox.y0 + bbox.y1) / 2),
        'align': align,
        'font': (f'{_weight2css(title.get_fontweight())} '
                 f'{title.get_fontsize() * dpi / 72.0:.1f}px '
                 f'"DejaVu Sans", Verdana, sans-serif'),
        'color': to_hex(title.get_color()),
    }


_PLAYER = Template("""
<div class="p2p-anim" id="$uid">
  <div class="p2p-stage">
    <img class="p2p-bg" src="data:image/png;base64,$bg" alt="">
    <canvas class="p2p-canvas" width="$width" height="$height"></canvas>
  </div>
  <div class="p2p-controls">
    <button class="p2p-btn" data-p2p-go="first"
            title="First frame">&#9198;</button>
    <button class="p2p-btn" data-p2p-go="prev"
            title="Previous frame">&#9194;</button>
    <button class="p2p-btn p2p-toggle" title="Play/Pause">&#9654;</button>
    <button class="p2p-btn" data-p2p-go="next"
            title="Next frame">&#9193;</button>
    <button class="p2p-btn" data-p2p-go="last"
            title="Last frame">&#9197;</button>
    <input class="p2p-slider" type="range" min="0" max="$last" value="0"
           step="1">
    <span class="p2p-count">1/$n_frames</span>
    <select class="p2p-mode" title="Loop mode">
      <option value="once">Once</option>
      <option value="loop">Loop</option>
      <option value="reflect">Reflect</option>
    </select>
  </div>
</div>
<style>
#$uid { display: inline-block; max-width: 100%; width: ${width}px;
        font-family: sans-serif; font-size: 13px; }
#$uid .p2p-stage { position: relative; line-height: 0; }
#$uid .p2p-bg { width: 100%; height: auto; display: block; }
#$uid .p2p-canvas { position: absolute; left: 0; top: 0;
                    width: 100%; height: 100%; }
#$uid .p2p-controls { display: flex; align-items: center; gap: 4px;
                      padding-top: 4px; }
#$uid .p2p-btn { cursor: pointer; border: 1px solid rgba(128,128,128,0.4);
                 border-radius: 3px; background: transparent; color: inherit;
                 padding: 1px 6px; font-size: 13px; line-height: 1.4; }
#$uid .p2p-btn:hover { background: rgba(128,128,128,0.2); }
#$uid .p2p-slider { flex: 1 1 auto; min-width: 40px; margin: 0 4px; }
#$uid .p2p-count { font-variant-numeric: tabular-nums; opacity: 0.7;
                   white-space: nowrap; }
#$uid .p2p-mode { background: transparent; color: inherit; font-size: 12px;
                  border-radius: 3px;
                  border: 1px solid rgba(128,128,128,0.4); }
</style>
<script>
(function () {
  var cfg = $config;
  var root = document.getElementById("$uid");
  if (!root) { return; }
  var ctx = root.querySelector(".p2p-canvas").getContext("2d");
  var slider = root.querySelector(".p2p-slider");
  var counter = root.querySelector(".p2p-count");
  var mode = root.querySelector(".p2p-mode");
  var toggle = root.querySelector(".p2p-toggle");
  var frame = 0, dir = 1, timer = null, playing = false, sheet = new Image();

  function draw() {
    if (!sheet.complete || !sheet.naturalWidth) { return; }
    var col = frame % cfg.ncols, row = (frame - col) / cfg.ncols;
    ctx.imageSmoothingEnabled = cfg.smooth;
    // Frames with an alpha channel are composited onto the static background,
    // so the previous frame has to go first or they stack up:
    ctx.clearRect(cfg.rect[0], cfg.rect[1], cfg.rect[2], cfg.rect[3]);
    ctx.drawImage(sheet, col * cfg.sw, row * cfg.sh, cfg.fw, cfg.fh,
                  cfg.rect[0], cfg.rect[1], cfg.rect[2], cfg.rect[3]);
    if (cfg.title) {
      ctx.clearRect(cfg.title.rect[0], cfg.title.rect[1],
                    cfg.title.rect[2], cfg.title.rect[3]);
      ctx.font = cfg.title.font;
      ctx.fillStyle = cfg.title.color;
      ctx.textAlign = cfg.title.align;
      ctx.textBaseline = "middle";
      ctx.fillText(cfg.labels[frame], cfg.title.x, cfg.title.y);
    }
  }

  function show(i) {
    frame = Math.max(0, Math.min(cfg.n - 1, i));
    slider.value = frame;
    counter.textContent = (frame + 1) + "/" + cfg.n;
    draw();
  }

  function pause() {
    if (timer !== null) { clearTimeout(timer); timer = null; }
    playing = false;
    toggle.innerHTML = "&#9654;";
  }

  function play() {
    if (playing) { return; }
    if (mode.value === "once" && frame === cfg.n - 1) { show(0); }
    playing = true;
    toggle.innerHTML = "&#10074;&#10074;";
    tick();
  }

  // Every frame is shown for its own time step, so a percept with an
  // irregular time axis (say, a pulse train) plays at the speed it was
  // recorded at. Browsers clamp timeouts to a few milliseconds, so time steps
  // shorter than that are stretched:
  function tick() {
    timer = setTimeout(function () {
      timer = null;
      advance();
      if (playing) { tick(); }
    }, Math.max(1, cfg.intervals[frame]));
  }

  function advance() {
    var next = frame + dir;
    if (next > cfg.n - 1 || next < 0) {
      if (mode.value === "loop") {
        next = dir > 0 ? 0 : cfg.n - 1;
      } else if (mode.value === "reflect" && cfg.n > 1) {
        dir = -dir;
        next = frame + dir;
      } else {
        pause();
        return;
      }
    }
    show(next);
  }

  root.querySelectorAll("[data-p2p-go]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      pause();
      dir = 1;
      var go = btn.dataset.p2pGo;
      show(go === "first" ? 0 : go === "last" ? cfg.n - 1 :
           go === "next" ? frame + 1 : frame - 1);
    });
  });
  toggle.addEventListener("click", function () {
    if (playing) { pause(); } else { play(); }
  });
  slider.addEventListener("input", function () {
    pause();
    dir = 1;
    show(parseInt(slider.value, 10));
  });
  mode.value = cfg.mode;
  sheet.onload = draw;
  sheet.src = "data:$sheet_mime;base64,$sheet";
  show(0);
})();
</script>
""")


class HTMLAnimation(FuncAnimation):
    """A :py:class:`~matplotlib.animation.FuncAnimation` with a fast player

    Behaves exactly like ``FuncAnimation`` (including ``save`` and
    ``to_html5_video``), but renders to HTML through a self-contained
    JavaScript player instead of Matplotlib's ``to_jshtml``. Instead of
    re-rendering the whole figure once per frame, the static parts of the
    figure are rendered once and all frames are shipped as a single
    color-mapped sprite sheet.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    fig, func, frames, *args, **kwargs :
        Passed to :py:class:`~matplotlib.animation.FuncAnimation`
    image : matplotlib.image.AxesImage
        The image artist that is updated by ``func``. Its position, colormap,
        and normalization determine how the frames are drawn
    frame_data : ndarray
        Either (Y, X, T) scalar data, (Y, X, 3, T) RGB data, or (Y, X, 4, T)
        RGBA data, matching what ``func`` displays in ``image``
    labels : list of str or None
        Per-frame titles. If None, the title is left alone
    fmt : {'jpg', 'png'}, optional
        Whether to encode the frames as JPEG or PNG. JPEG is typically an
        order of magnitude smaller, PNG is lossless
    intervals : array_like or None, optional
        How long each frame stays up (in ms), one value per frame. If None,
        every frame is shown for ``interval`` ms. Frames of unequal duration
        are what lets an animation follow an irregular time axis; see
        :py:func:`~pulse2percept.utils.frame_timeline`.

        .. versionadded:: 0.10.0

    Notes
    -----
    *  Frames are quantized to 256 levels and embedded at most at the size at
       which they are displayed, exactly like Matplotlib would rasterize them.
    *  The per-frame title is drawn by the browser, so it uses DejaVu Sans if
       available and falls back to the default sans-serif font otherwise.
    """

    def __init__(self, fig, func, frames=None, *args, image=None,
                 frame_data=None, labels=None, fmt='jpg', intervals=None,
                 **kwargs):
        self._image = image
        self._frame_data = frame_data
        self._labels = labels
        self._fmt = _check_fmt(fmt)
        self._html = None
        if intervals is not None:
            intervals = np.array(intervals, dtype=np.float64).ravel()
            n_frames = (None if frame_data is None
                        else np.shape(frame_data)[-1])
            if n_frames is not None and intervals.size != n_frames:
                raise ValueError(f"'intervals' must have one value per "
                                 f"frame ({n_frames}), not "
                                 f"{intervals.size}.")
            # Matplotlib has a single frame delay, which is all the inherited
            # machinery (``save``, ``to_html5_video``) can express. The mean
            # keeps a movie the same length as the animation it came from:
            kwargs.setdefault('interval', float(intervals.mean()))
        self._intervals = intervals
        super().__init__(fig, func, frames, *args, **kwargs)

    def _display_intervals(self, fps, n_frames):
        """How long each frame stays up (in ms), one value per frame"""
        if fps is not None:
            # An explicit frame rate overrides the animation's own timing,
            # which is what Matplotlib's ``to_jshtml(fps=...)`` means:
            return np.full(n_frames, 1000.0 / fps)
        if self._intervals is not None:
            return self._intervals
        return np.full(n_frames, float(self._interval))

    def _build_html(self, intervals, default_mode):
        """Assemble the self-contained HTML player"""
        fig = self._fig
        im = self._image
        if im.norm.vmin is None or im.norm.vmax is None:
            # The image is hidden while the background is rendered, so it will
            # never get a chance to autoscale itself:
            im.norm.autoscale_None(np.asarray(self._frame_data))
        title_artist = im.axes.title
        old_title = title_artist.get_text()
        try:
            if self._labels is not None:
                # The player draws the title itself, on a canvas that sits on
                # top of the static background:
                title_artist.set_text('')
            bg, width, height = _background(fig, im)
            # ``get_window_extent`` is only meaningful once the figure has been
            # drawn, which ``_background`` just did:
            bbox = im.get_window_extent()
            title = None
            if self._labels is not None:
                title = _title_geometry(title_artist, width, height, fig.dpi)
        finally:
            title_artist.set_text(old_title)
        left, right = int(np.floor(bbox.x0)), int(np.ceil(bbox.x1))
        top, bottom = int(np.floor(height - bbox.y1)), \
            int(np.ceil(height - bbox.y0))
        rect = [left, top, max(1, right - left), max(1, bottom - top)]
        sheet = _sprite_sheet(self._frame_data, im.norm, im.cmap,
                              (rect[3], rect[2]), self._fmt,
                              bg_color=_bg_color(im.axes))
        config = {
            'n': int(np.shape(self._frame_data)[-1]),
            'ncols': sheet['ncols'],
            'fw': sheet['fw'],
            'fh': sheet['fh'],
            'sw': sheet['sw'],
            'sh': sheet['sh'],
            'rect': rect,
            # Mirror Matplotlib's 'antialiased' interpolation, which switches
            # to nearest-neighbor once the image is strongly magnified:
            'smooth': (rect[2] <= MAX_SMOOTH_UPSAMPLE * sheet['fw'] and
                       rect[3] <= MAX_SMOOTH_UPSAMPLE * sheet['fh']),
            # The player advances frame by frame, so it needs every delay;
            # the scalar is kept for whoever reads the config:
            'interval': float(np.mean(intervals)),
            'intervals': [float(i) for i in intervals],
            'mode': default_mode,
            'title': title,
            'labels': self._labels if title is not None else [],
        }
        return _PLAYER.safe_substitute(
            uid=f'p2p-anim-{uuid4().hex}', width=width, height=height,
            n_frames=config['n'], last=config['n'] - 1, config=dumps(config),
            bg=base64.b64encode(bg).decode('ascii'),
            sheet_mime=sheet['mime'],
            sheet=base64.b64encode(sheet['data']).decode('ascii'))

    def to_jshtml(self, fps=None, embed_frames=True, default_mode=None):
        """Generate an HTML representation of the animation

        Parameters
        ----------
        fps : float or None
            Frames per second. If None, every frame is shown for as long as
            the animation's own timing says it lasts. Otherwise all frames are
            shown for ``1000 / fps`` ms, which changes how fast the animation
            plays; resample the frames themselves (see
            :py:func:`~pulse2percept.utils.frame_timeline`) to keep its
            duration. May be given as a unitful frequency (e.g. ``30 * Hz``);
            see :py:mod:`pulse2percept.units`.
        embed_frames : bool
            Unused; frames are always embedded.
        default_mode : {'loop', 'once', 'reflect'} or None
            What the animation should do once it has played through. If None,
            uses 'loop' or 'once', depending on ``repeat``.
        """
        # Before the fallback below: Matplotlib takes a plain number of hertz,
        # so a quantity has to be converted whichever player renders it.
        fps = as_value(fps, Hz, 'fps')
        if self._image is None or self._frame_data is None:
            # Nothing to accelerate, fall back on Matplotlib:
            return super().to_jshtml(fps=fps, embed_frames=embed_frames,
                                     default_mode=default_mode)
        if default_mode is None:
            default_mode = 'loop' if self._repeat else 'once'
        intervals = self._display_intervals(
            fps, int(np.shape(self._frame_data)[-1]))
        # We are rendering the animation ourselves, so silence Matplotlib's
        # "Animation was deleted without rendering anything" warning:
        self._draw_was_started = True
        # Rendering the sprite sheet is the expensive part, so only do it once:
        key = (tuple(intervals), default_mode)
        if self._html is None or self._html[0] != key:
            self._html = (key, self._build_html(intervals, default_mode))
        return self._html[1]

    def _repr_html_(self):
        """HTML representation for IPython, Jupyter, and Sphinx-Gallery"""
        return self.to_jshtml()
