""":py:func:`~pulse2percept.plotting.plot_stimulus_percept`,
   :py:func:`~pulse2percept.plotting.play_stimulus_percept`,
   :py:func:`~pulse2percept.plotting.plot_implant_percept`,
   :py:func:`~pulse2percept.plotting.play_implant_percept`

Side-by-side views of a model's input and its predicted percept.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.transforms import Bbox

from ..implants.electrode_arrays import _peak_drive, _stim_fill
from ..models.base import Model, SpatialModel
from ..percepts.base import Percept, _pixel_extent, _reject_rgb, _resolve_clim
from ..stimuli import ImageStimulus, Stimulus, VideoStimulus
from ..stimuli.base import _has_time_axis
from ..units import ms
from ..utils import HTMLAnimation
from ..utils import _visual_field as vf
from ..utils.animation import _frame_timeline
from ..utils.constants import ZORDER

__all__ = ['play_implant_percept', 'play_stimulus_percept',
           'plot_implant_percept', 'plot_stimulus_percept']

# Size of a two-panel figure (in inches), wide enough for two square panels
# side by side:
FIGSIZE = (10, 4)

# Electrode fill, as in ``Implant.plot(stim_cmap=True)``:
STIM_CMAP = 'YlOrRd'
# Electrode labels sit above the electrode (offset in points), so that they
# do not cover its fill:
LABEL_OFFSET = (0, 6)
LABEL_KWARGS = {'ha': 'center', 'va': 'bottom', 'color': 'black',
                'zorder': ZORDER['annotate'],
                'bbox': {'boxstyle': 'square,pad=0.1', 'ec': 'none',
                         'fc': (1, 1, 1, 0.7)}}


def _panel_axes(axes, figsize, layout=None):
    """Two Axes side by side: stimulus on the left, percept on the right"""
    if axes is None:
        return plt.subplots(ncols=2, figsize=figsize or FIGSIZE,
                            layout=layout)[1]
    axes = np.asarray(axes).ravel()
    if axes.size != 2:
        raise ValueError(f"'axes' must be two Axes (stimulus, percept), not "
                         f"{axes.size}.")
    for ax in axes:
        if not isinstance(ax, Axes):
            raise TypeError(f"'axes' must contain Matplotlib Axes, not "
                            f"{type(ax)}.")
    return axes


def _reject_non_visual(stim):
    """Returns the TypeError for a stimulus that is not an image or video"""
    return TypeError(
        f"Cannot show a {type(stim).__name__} next to the percept. Pass the "
        f"image or video that went into the model, not the electrical "
        f"stimulus an encoder made of it.")


def _source_frames(stim, times):
    """Returns the source frames and the frame index at each display time (ms)

    Zero-order hold: each display time shows the source frame active at that
    time. Times before the source start show its first frame, times after its
    end show its last. A still image has a single frame.
    """
    if isinstance(stim, VideoStimulus):
        if stim.time is None:
            raise ValueError(
                "Cannot animate a video with time=None next to the percept: "
                "without a time axis there is nothing to hold its frames on. "
                "Give the video a time axis, or pass a single frame as an "
                "ImageStimulus.")
        frames = stim._frames()
    elif isinstance(stim, ImageStimulus):
        frames = stim.data.reshape(stim.img_shape)[..., np.newaxis]
    else:
        raise _reject_non_visual(stim)
    src = stim.times(ms)
    if src is None or frames.shape[-1] == 1:
        return frames[..., :1], np.zeros(np.size(times), dtype=np.intp)
    idx = np.searchsorted(src, np.asarray(times, dtype=float),
                          side='right') - 1
    return frames, np.clip(idx, 0, frames.shape[-1] - 1)


def _source_times(percept, timeline):
    """Returns the source time (ms) shown at each display frame

    A temporal model stores the onset of the source-video frame that each
    output frame summarizes (``metadata['source_frame_time']``); the output
    time is that frame's end.
    """
    onsets = (percept.metadata or {}).get('source_frame_time')
    if onsets is None:
        return timeline.times
    return np.asarray(onsets, dtype=np.float64)[timeline.indices]


def _image_artist(ax, frames, vmin=None, vmax=None):
    """Returns an empty image artist for the animation ``frames``"""
    blank = np.zeros_like(frames[..., 0])
    if blank.ndim == 3:
        # RGB frames use no colormap or color scale:
        return ax.imshow(blank)
    return ax.imshow(blank, cmap='gray', vmin=vmin, vmax=vmax)


def plot_stimulus_percept(stim, percept, axes=None, figsize=None,
                          titles=('Stimulus', 'Percept'), stim_kwargs=None,
                          percept_kwargs=None):
    """Plot an image next to the percept it produced

    Draws ``stim`` and ``percept`` side by side, each with its own ``plot``
    method. For a video, use
    :py:func:`~pulse2percept.plotting.play_stimulus_percept`.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    stim : :py:class:`~pulse2percept.stimuli.ImageStimulus`
        The image that went into the model, not the electrical stimulus an
        encoder made of it.
    percept : :py:class:`~pulse2percept.percepts.Percept`
        The percept the model predicted.
    axes : list of two matplotlib.axes.Axes, optional
        Axes to draw into, stimulus first. If None, a new figure is created.
    figsize : ``(width, height)``, optional
        Size of that new figure (in inches). Ignored if ``axes`` is given.
    titles : (str, str), optional
        Titles for the two panels.
    stim_kwargs, percept_kwargs : dict, optional
        Passed on to the stimulus' and the percept's ``plot`` method.

    Returns
    -------
    axes : np.ndarray of matplotlib.axes.Axes
        The two Axes that were drawn into.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use('Agg')
    >>> import pulse2percept as p2p
    >>> stim = p2p.stimuli.samples.logo_ucsb(resize=(24, 32))
    >>> model = p2p.models.retina.ScoreboardModel(
    ...     p2p.implants.retina.ArgusII(), xrange=(-4, 4), yrange=(-4, 4),
    ...     step=0.5)
    >>> percept = model.predict_percept(stim)
    >>> axes = p2p.plotting.plot_stimulus_percept(stim, percept)
    >>> [ax.get_title() for ax in axes]
    ['Stimulus', 'Percept']

    """
    if isinstance(stim, VideoStimulus):
        raise TypeError(
            "A video and the percept it produced have no single frame that "
            "stands for both of them: the brightest frame of one need not "
            "line up with the brightest frame of the other. Use "
            "play_stimulus_percept() instead.")
    if not isinstance(stim, ImageStimulus):
        raise _reject_non_visual(stim)
    axes = _panel_axes(axes, figsize, layout='constrained')
    stim.plot(ax=axes[0], **(stim_kwargs or {}))
    percept.plot(ax=axes[1], **(percept_kwargs or {}))
    for ax, title in zip(axes, titles):
        ax.set_title(title)
    return axes


def play_stimulus_percept(stim, percept, fps=None, axes=None, figsize=None,
                          titles=('Stimulus', 'Percept'), repeat=True,
                          annotate_time=True, colorbar=True, fmt='png',
                          vmin=None, vmax=None):
    """Animate a stimulus next to the percept it produced

    Both panels use the percept's time axis: each percept frame is paired
    with the source frame active at the same time (zero-order hold), so
    source and percept may have different frame rates. A temporal percept
    frame that summarizes a source frame ends when that frame ends, and is
    paired with it (``metadata['source_frame_time']``). ``fps`` resamples the
    display as in :py:meth:`~pulse2percept.percepts.Percept.play`.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    stim : :py:class:`~pulse2percept.stimuli.ImageStimulus` or
           :py:class:`~pulse2percept.stimuli.VideoStimulus`
        The image or video that went into the model, not the electrical
        stimulus an encoder made of it.
    percept : :py:class:`~pulse2percept.percepts.Percept`
        The percept the model predicted. Must have a time axis.
    fps : float, optional
        Display frame rate in Hz. If None, use the percept's recorded timing.
        May also be given as a unitful frequency (e.g., ``30 * Hz``).
    axes : list of two matplotlib.axes.Axes, optional
        Axes to animate in, stimulus first. If None, a new figure is created.
    figsize : ``(width, height)``, optional
        Size of that new figure (in inches). Ignored if ``axes`` is given.
    titles : (str, str), optional
        Titles for the two panels.
    repeat : bool, optional
        Whether to repeat the animation.
    annotate_time : bool, optional
        Whether to show the current time above the two panels.
    colorbar : bool, optional
        Whether to show a brightness colorbar next to the percept. An RGB
        percept never gets one.
    fmt : {'png', 'jpg'}, optional
        Image format used to encode animation frames. 'jpg' gives smaller
        notebooks and doc pages (useful for a video source); 'png' is
        lossless.
    vmin, vmax : float, optional
        Brightness limits for the percept. By default, ``vmin=0`` and ``vmax``
        is the maximum brightness across the percept. Not available for an RGB
        percept (values are shown as is).

    Returns
    -------
    ani : :py:class:`~pulse2percept.utils.HTMLAnimation`
        The animation.

    Notes
    -----
    Neither stimulus nor percept is resampled: the source keeps its own frames,
    and ``fps`` only controls which of them are shown when.

    """
    if percept.time is None:
        raise ValueError("Cannot animate a percept with time=None. Use "
                         "plot_stimulus_percept() instead.")
    timeline = _frame_timeline(percept.times(ms), fps=fps)
    idx = timeline.indices
    src, src_idx = _source_frames(stim, _source_times(percept, timeline))
    # No constrained layout: the player measures the figure without the time
    # label, and a re-layout would shift the panels.
    axes = _panel_axes(axes, figsize)
    fig = axes[0].figure
    im_stim = _image_artist(axes[0], src, vmin=0, vmax=float(np.max(src)))
    if percept.is_rgb:
        if vmin is not None or vmax is not None:
            raise _reject_rgb('vmin/vmax', ' Its RGB values are shown as '
                                           'they are.')
        im_percept = _image_artist(axes[1], percept.data)
    else:
        vmin, vmax = _resolve_clim(percept.data, vmin, vmax, auto_vmin=0)
        im_percept = _image_artist(axes[1], percept.data, vmin=vmin, vmax=vmax)
        if colorbar:
            cbar = fig.colorbar(im_percept, ax=axes[1])
            cbar.ax.set_ylabel('Phosphene brightness (a.u.)', rotation=-90,
                               va='center')
    for ax, title in zip(axes, titles):
        ax.set_title(title)
    # Both panels share one time axis, so the time is shown once above them:
    clock = labels = None
    if annotate_time:
        clock = fig.suptitle('')
        labels = [f't = {t:.2f} {percept.time_unit}'
                  for t in percept.time[idx]]

    def update(i):
        if clock is not None:
            clock.set_text(labels[i])
        im_stim.set_data(src[..., src_idx[i]])
        im_percept.set_data(percept.data[..., idx[i]])
        return im_stim, im_percept

    def data_gen():
        yield from range(idx.size)

    plt.rcParams["animation.html"] = 'jshtml'
    plt.close(fig)
    # The player shows source frame ``src_idx[i]`` at display frame ``i``:
    return HTMLAnimation(fig, update, data_gen, repeat=repeat,
                         intervals=timeline.intervals, save_count=idx.size,
                         image=[im_stim, im_percept],
                         frame_data=[src, percept.data],
                         frame_index=[src_idx, idx], labels=labels,
                         title=clock, fmt=fmt)


def _electrode_stim(percept):
    """Returns the electrode stimulus stored in ``percept.metadata['stim']``

    For a composite model, the stimulus is stored in the intermediate percept
    stored there.
    """
    stim = percept
    while isinstance(stim, Percept):
        stim = (stim.metadata or {}).get('stim')
    if not isinstance(stim, Stimulus):
        raise ValueError("The percept does not record the stimulus that "
                         "produced it in metadata['stim']. Pass a percept "
                         "returned by the model's predict_percept().")
    return stim


def _is_causal(model):
    """Returns whether ``model`` integrates stimulation over time

    A spatial-only model uses the stimulus state at each frame. Any other
    model's percept at t depends on the stimulation delivered up to t.
    """
    if isinstance(model, Model):
        return model.has_time
    return not isinstance(model, SpatialModel)


def _frame_intervals(times, start):
    """Returns ``(lo, hi)``, where percept frame k ends interval
    ``(lo[k], hi[k]]``

    The first interval starts at stimulus onset ``start``. A frame at or
    before onset is an instant.
    """
    return np.concatenate(([min(start, times[0])], times[:-1])), times


def _electrode_drive(stim, times=None, causal=False):
    """Returns the absolute drive, (n_electrodes, n_times), at percept
    ``times``

    All times are in ms.

    * ``times=None``: peak over the whole stimulus, as one column.
    * ``causal=False`` (spatial-only model): the drive at each time. Columns
      are held until the next one (zero-order hold); a frame-level view
      (``stim._spatial_view()``) is off once the stimulus ends.
    * ``causal=True``: the peak over ``(times[k - 1], times[k]]``, i.e., what
      was delivered since the previous frame; the first interval starts at
      stimulus onset. A plain waveform is linear between its samples.
    """
    view = stim._spatial_view()
    data = np.abs(np.asarray(view.data, dtype=np.float64)).reshape(
        len(view.electrodes), -1)
    if times is None:
        return _peak_drive(data, axis=1)[:, np.newaxis]
    times = np.asarray(times, dtype=np.float64).ravel()
    if view.time is None:
        return np.repeat(data[:, :1], times.size, axis=1)
    t = view.times(ms)
    if view is stim and causal:
        signed = np.asarray(view.data, dtype=np.float64).reshape(data.shape)
        lo, hi = _frame_intervals(times, t[0])
        # A linear waveform peaks at an interval end or at a sample inside:
        drive = np.array([np.maximum(np.abs(np.interp(lo, t, row, 0, 0)),
                                     np.abs(np.interp(hi, t, row, 0, 0)))
                          for row in signed]).T
        k = np.clip(np.searchsorted(hi, t, side='left'), 0, hi.size - 1)
        inside = (lo[k] < t) & (t < hi[k])
        np.maximum.at(drive, k[inside], data[:, inside].T)
        return drive.T
    # Each column lasts until the next; a frame-level view ends with the
    # stimulus:
    end = (stim.times(ms)[-1] if view is not stim and _has_time_axis(stim)
           else np.inf)
    ends = np.append(t[1:], end)
    if not causal:
        idx = np.searchsorted(t, times, side='right') - 1
        on = idx >= 0
        on[on] = times[on] < ends[idx[on]]
        return data[:, np.clip(idx, 0, t.size - 1)] * on
    lo, hi = _frame_intervals(times, t[0])
    # An instant takes the frame that was up just before it:
    lo = np.where(lo < hi, lo, np.nextafter(hi, -np.inf))
    drive = np.zeros((data.shape[0], times.size))
    for k in range(times.size):
        cols = (t < hi[k]) & (ends > lo[k])
        if np.any(cols):
            drive[:, k] = data[:, cols].max(axis=1)
    return drive


def _drive_norm(drive):
    """Returns one color scale for all frames, from 0 to the peak drive"""
    vmax = float(np.max(drive)) if np.size(drive) else 0.0
    return Normalize(vmin=0, vmax=vmax if vmax > 0 else 1.0)


def _drive_label(stim):
    """Returns the colorbar label for the electrode drive"""
    unit = stim._spatial_view().unit
    if unit is None or unit.dimension.is_dimensionless:
        return 'Electrode drive (a.u.)'
    return f'Amplitude ({unit})'


def _implant_panel(model, ax):
    """Plot ``model`` with its placed implant on ``ax``

    Returns ``(collection, facecolors, edgecolors)`` for every electrode
    collection the implant drew, colors as drawn.
    """
    before = set(map(id, ax.collections))
    model.plot(ax=ax, show_implant=True)
    panel = []
    for coll in ax.collections:
        if id(coll) in before or not hasattr(coll, '_stim_patches'):
            continue
        n = len(coll.get_paths())
        panel.append((coll,
                      np.broadcast_to(coll.get_facecolor(), (n, 4)).copy(),
                      np.broadcast_to(coll.get_edgecolor(), (n, 4)).copy()))
    return panel


def _paint(panel, electrodes, drive, cmap, norm, only_active=False):
    """Fill each electrode's stimulus patch by its ``drive``

    ``only_active`` makes every other patch transparent.
    """
    for coll, base_fc, base_ec in panel:
        fc, ec = base_fc.copy(), base_ec.copy()
        if only_active:
            fc[:], ec[:] = 0, 0
        for name, amp in zip(electrodes, drive):
            k = coll._stim_patches.get(name)
            if k is not None and amp > 0:
                fc[k] = _stim_fill(cmap, norm, amp)
                ec[k] = base_ec[k]
        coll.set_facecolor(fc)
        coll.set_edgecolor(ec)


def _labels(model, panel, electrodes):
    """Returns one hidden label per electrode, in its collection's
    coordinates"""
    labels = {}
    for coll, _, _ in panel:
        for name in electrodes:
            if name in coll._stim_patches and name not in labels:
                e = model.implant.electrode_array[name]
                labels[name] = coll.axes.annotate(
                    str(name), (e.x, e.y), xycoords=coll.get_transform(),
                    xytext=LABEL_OFFSET, textcoords='offset points',
                    visible=False, **LABEL_KWARGS)
    return labels


def _overlay(ax, panel, labels, electrodes, states, cmap, norm):
    """RGBA renders of the stimulated electrodes, one per row of ``states``

    Only the electrode collections and labels are drawn, so the rest of the
    panel stays vector graphics underneath. Returns ``(frames, extent)``:
    (Y, X, 4, n_states) uint8 at the figure's resolution, cropped to the
    implant, and its data extent.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    colls = [coll for coll, _, _ in panel]
    for text in labels.values():
        text.set_visible(True)
    box = Bbox.intersection(
        Bbox.union([a.get_window_extent(renderer)
                    for a in colls + list(labels.values())]), ax.bbox)
    if box is None:
        box = ax.bbox
    width, height = int(round(fig.bbox.width)), int(round(fig.bbox.height))
    left = max(int(np.floor(box.x0)), 0)
    right = min(int(np.ceil(box.x1)), width)
    top = max(int(np.floor(height - box.y1)), 0)
    bottom = min(int(np.ceil(height - box.y0)), height)
    where = {name: i for i, name in enumerate(electrodes)}
    # State-major, so the (Y, X, 4, n_states) view below needs no copy:
    frames = np.empty((len(states), bottom - top, right - left, 4),
                      dtype=np.uint8)
    for s, state in enumerate(states):
        _paint(panel, electrodes, state, cmap, norm, only_active=True)
        canvas = RendererAgg(width, height, fig.dpi)
        for coll in colls:
            coll.draw(canvas)
        for name, text in labels.items():
            text.set_visible(state[where[name]] > 0)
            text.draw(canvas)
        frames[s] = np.asarray(canvas.buffer_rgba())[top:bottom, left:right]
    # The static background shows the implant as drawn, without labels:
    for coll, fc, ec in panel:
        coll.set_facecolor(fc)
        coll.set_edgecolor(ec)
    for text in labels.values():
        text.remove()
    (x0, y0), (x1, y1) = ax.transData.inverted().transform(
        [(left, height - bottom), (right, height - top)])
    return np.moveaxis(frames, 0, -1), (x0, x1, y0, y1)


def _grid_kwargs(rings=False, meridians=False, grid_color=vf.GRID_COLOR):
    """Returns the visual-field grid options of ``Percept.play``"""
    return rings, meridians, grid_color


def _percept_panel(percept, ax, vmin, vmax, colorbar):
    """Returns the image artist for an animated percept, as in
    ``Percept.play``"""
    spatial = percept.xdva is not None and percept.ydva is not None
    extent = _pixel_extent(percept.xdva, percept.ydva) if spatial else None
    blank = np.zeros_like(percept.data[..., 0])
    if percept.is_rgb:
        if vmin is not None or vmax is not None:
            raise _reject_rgb('vmin/vmax', ' Its RGB values are shown as '
                                           'they are.')
        im = ax.imshow(blank, origin='upper', extent=extent)
    else:
        vmin, vmax = _resolve_clim(percept.data, vmin, vmax, auto_vmin=0)
        im = ax.imshow(blank, cmap='gray', vmin=vmin, vmax=vmax,
                       origin='upper', extent=extent)
        if colorbar:
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Phosphene brightness (a.u.)', rotation=-90,
                               va='center')
    if spatial:
        percept._label_axes(ax)
    return im


def plot_implant_percept(model, percept, axes=None, figsize=None,
                         titles=('Implant', 'Percept'), annotate=False,
                         colorbar=True, percept_kwargs=None):
    """Plot the stimulated implant next to the percept it produced

    The left panel shows ``model`` with its implant at the model-side
    placement, each electrode filled by its absolute drive. The drive comes
    from ``percept.metadata['stim']``, the prepared stimulus that entered the
    model. Encoded stimuli (e.g., from
    :py:class:`~pulse2percept.stimuli.TraceEncoder`) are shown at their
    frame-level modulation, not at individual pulse phases.

    For a percept with more than one frame, use
    :py:func:`~pulse2percept.plotting.play_implant_percept`.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    model : Model or BaseModel
        The :py:class:`~pulse2percept.models.Model` or
        :py:class:`~pulse2percept.models.BaseModel` that predicted
        ``percept``. Its ``plot(show_implant=True)`` draws the left panel.
    percept : :py:class:`~pulse2percept.percepts.Percept`
        The percept the model predicted, timeless or a single frame. A
        timeless percept shows the peak drive over the whole stimulus; a
        single frame at time t shows the drive at t (for a model with time,
        the peak from stimulus onset to t).
    axes : list of two matplotlib.axes.Axes, optional
        Axes to draw into, implant first. If None, a new figure is created.
    figsize : ``(width, height)``, optional
        Size of that new figure (in inches). Ignored if ``axes`` is given.
    titles : (str, str), optional
        Titles for the two panels.
    annotate : bool, optional
        Whether to label the stimulated electrodes.
    colorbar : bool, optional
        Whether to show a drive colorbar next to the implant.
    percept_kwargs : dict, optional
        Passed on to :py:meth:`~pulse2percept.percepts.Percept.plot`.

    Returns
    -------
    axes : np.ndarray of matplotlib.axes.Axes
        The two Axes that were drawn into.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use('Agg')
    >>> import pulse2percept as p2p
    >>> model = p2p.models.retina.ScoreboardModel(
    ...     p2p.implants.retina.ArgusII(), xrange=(-4, 4), yrange=(-4, 4),
    ...     step=0.5)
    >>> percept = model.predict_percept({'A3': 20})
    >>> axes = p2p.plotting.plot_implant_percept(model, percept)
    >>> [ax.get_title() for ax in axes]
    ['Implant', 'Percept']

    """
    if percept.time is not None and np.size(percept.time) > 1:
        raise ValueError("A percept with more than one frame has no single "
                         "time point to show the implant at. Use "
                         "play_implant_percept() instead.")
    stim = _electrode_stim(percept)
    times = None if percept.time is None else percept.times(ms)
    drive = _electrode_drive(stim, times, causal=_is_causal(model))[:, 0]
    electrodes = list(stim.electrodes)
    cmap, norm = plt.get_cmap(STIM_CMAP), _drive_norm(drive)
    axes = _panel_axes(axes, figsize, layout='constrained')
    panel = _implant_panel(model, axes[0])
    _paint(panel, electrodes, drive, cmap, norm)
    if annotate:
        active = [name for name, amp in zip(electrodes, drive) if amp > 0]
        for text in _labels(model, panel, active).values():
            text.set_visible(True)
    if colorbar:
        cbar = axes[0].figure.colorbar(ScalarMappable(norm, cmap),
                                       ax=axes[0])
        cbar.set_label(_drive_label(stim))
    percept.plot(ax=axes[1], **(percept_kwargs or {}))
    for ax, title in zip(axes, titles):
        ax.set_title(title)
    return axes


def play_implant_percept(model, percept, fps=None, axes=None, figsize=None,
                         titles=('Implant', 'Percept'), annotate=False,
                         repeat=True, annotate_time=True, colorbar=True,
                         vmin=None, vmax=None, percept_kwargs=None):
    """Animate the stimulated implant next to the percept it produced

    Both panels use the percept's time axis. Each percept frame at time t is
    shown with the electrode drive from ``percept.metadata['stim']``:

    *  A spatial-only model: the drive at t. Frame-level modulation (e.g.,
       from :py:class:`~pulse2percept.stimuli.TraceEncoder` or an image
       encoder) is held between its frames (zero-order hold).
    *  A model with time: the peak absolute drive since the previous percept
       frame, i.e., over ``(t_prev, t]``; for the first frame, since stimulus
       onset. A frame-level modulation counts every frame active in that
       interval; a plain waveform counts every pulse, without resolving pulse
       phases. For automatic output times, this is the interval the percept
       frame summarizes. This is the delivered drive, not the percept: a
       stateful model's frame may still show decay from earlier stimulation
       while the implant is off.

    ``fps`` resamples the percept frames; the implant panel follows the
    displayed percept frame. All frames share one color scale, from 0 to the
    peak drive.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    model : Model or BaseModel
        The :py:class:`~pulse2percept.models.Model` or
        :py:class:`~pulse2percept.models.BaseModel` that predicted
        ``percept``. Its ``plot(show_implant=True)`` draws the left panel.
    percept : :py:class:`~pulse2percept.percepts.Percept`
        The percept the model predicted. Must have a time axis.
    fps : float, optional
        Display frame rate in Hz. If None, use the percept's recorded timing.
        May also be given as a unitful frequency (e.g., ``30 * Hz``).
    axes : list of two matplotlib.axes.Axes, optional
        Axes to animate in, implant first. If None, a new figure is created.
    figsize : ``(width, height)``, optional
        Size of that new figure (in inches). Ignored if ``axes`` is given.
    titles : (str, str), optional
        Titles for the two panels.
    annotate : bool, optional
        Whether to label electrodes while they are stimulated.
    repeat : bool, optional
        Whether to repeat the animation.
    annotate_time : bool, optional
        Whether to show the current time above the two panels.
    colorbar : bool, optional
        Whether to show a drive colorbar next to the implant and a brightness
        colorbar next to the percept. An RGB percept never gets one.
    vmin, vmax : float, optional
        Brightness limits for the percept. By default, ``vmin=0`` and ``vmax``
        is the maximum brightness across the percept.
    percept_kwargs : dict, optional
        ``rings``, ``meridians``, and ``grid_color``, as in
        :py:meth:`~pulse2percept.percepts.Percept.play`.

    Returns
    -------
    ani : :py:class:`~pulse2percept.utils.HTMLAnimation`
        The animation.

    Notes
    -----
    The implant layer is rendered once per distinct electrode state, as 8-bit
    RGBA at the figure's resolution (about 0.36 MB per state for a 300 x 300
    pixel implant), so memory grows with the number of distinct states.

    """
    if percept.time is None:
        raise ValueError("Cannot animate a percept with time=None. Use "
                         "plot_implant_percept() instead.")
    rings, meridians, grid_color = _grid_kwargs(**(percept_kwargs or {}))
    grid = percept._grid_geometry(rings, meridians)
    stim = _electrode_stim(percept)
    timeline = _frame_timeline(percept.times(ms), fps=fps)
    idx = timeline.indices
    electrodes = list(stim.electrodes)
    # One drive per percept frame:
    drive = _electrode_drive(stim, percept.times(ms),
                             causal=_is_causal(model))
    # The color scale spans all percept frames, regardless of ``fps``:
    cmap, norm = plt.get_cmap(STIM_CMAP), _drive_norm(drive)
    drive = drive[:, idx]
    states, state_idx = np.unique(drive.T, axis=0, return_inverse=True)
    # NumPy 2.x returns a 2D inverse for axis-wise unique:
    state_idx = np.ravel(state_idx)
    axes = _panel_axes(axes, figsize, layout='constrained')
    fig = axes[0].figure
    panel = _implant_panel(model, axes[0])
    if colorbar:
        cbar = fig.colorbar(ScalarMappable(norm, cmap), ax=axes[0])
        cbar.set_label(_drive_label(stim))
    im_percept = _percept_panel(percept, axes[1], vmin, vmax, colorbar)
    for ax, title in zip(axes, titles):
        ax.set_title(title)
    clock = labels = None
    if annotate_time:
        labels = [f't = {t:.2f} {percept.time_unit}'
                  for t in percept.time[idx]]
        clock = fig.suptitle(labels[0])
    # Lay out once, then freeze: the player measures the figure without the
    # time label, and a re-layout would shift the panels.
    fig.canvas.draw()
    fig.set_layout_engine('none')
    if clock is not None:
        clock.set_text('')
    im_grid = None
    if grid is not None:
        # A still image layer, as in ``Percept.play``:
        im_grid = percept._grid_layer(axes[1], grid, grid_color,
                                      im_percept.get_zorder() + 1)
    active = []
    if annotate:
        active = [name for name, on in zip(electrodes, np.any(drive > 0, 1))
                  if on]
    frames, extent = _overlay(axes[0], panel, _labels(model, panel, active),
                              electrodes, states, cmap, norm)
    # Keep the limits the model plot chose:
    xlim, ylim = axes[0].get_xlim(), axes[0].get_ylim()
    im_implant = axes[0].imshow(frames[..., state_idx[0]], origin='upper',
                                extent=extent, zorder=ZORDER['annotate'] + 1)
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)
    axes[0].set_autoscale_on(False)

    def update(i):
        if clock is not None:
            clock.set_text(labels[i])
        im_implant.set_data(frames[..., state_idx[i]])
        im_percept.set_data(percept.data[..., idx[i]])
        return im_implant, im_percept

    def data_gen():
        yield from range(idx.size)

    images = [im_implant, im_percept]
    frame_data = [frames, percept.data]
    frame_index = [state_idx, idx]
    if im_grid is not None:
        images.append(im_grid)
        frame_data.append(np.asarray(im_grid.get_array())[..., np.newaxis])
        frame_index.append(np.zeros_like(idx))
    plt.rcParams["animation.html"] = 'jshtml'
    plt.close(fig)
    # PNG keeps the implant layer transparent around the electrodes:
    return HTMLAnimation(fig, update, data_gen, repeat=repeat,
                         intervals=timeline.intervals, save_count=idx.size,
                         image=images, frame_data=frame_data,
                         frame_index=frame_index, labels=labels,
                         title=clock, fmt='png')
