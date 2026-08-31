""":py:func:`~pulse2percept.plotting.plot_stimulus_percept`,
   :py:func:`~pulse2percept.plotting.play_stimulus_percept`

Views that span more than one object: what went into the model next to what
came out of it.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ..percepts.base import _reject_rgb, _resolve_clim
from ..stimuli import ImageStimulus, VideoStimulus
from ..units import ms
from ..utils import HTMLAnimation
from ..utils.animation import _frame_timeline

__all__ = ['play_stimulus_percept', 'plot_stimulus_percept']

# Size of a two-panel figure (in inches), wide enough for two square panels
# side by side:
FIGSIZE = (10, 4)


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
    """The error for a stimulus that is not the source the model was shown"""
    return TypeError(
        f"Cannot show a {type(stim).__name__} next to the percept. Pass the "
        f"image or video that went into the model, not the electrical "
        f"stimulus an encoder made of it.")


def _source_frames(stim, times):
    """Source frames, and which of them to show at each display time (in ms)

    Zero-order hold: a display time shows the source frame that is up at that
    physical time. Times before the source starts hold its first frame, times
    that outlast it hold its last. A still image contributes a single frame.
    """
    if isinstance(stim, VideoStimulus):
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


def _image_artist(ax, frames, vmin=None, vmax=None):
    """An empty image artist for the animation to write ``frames`` into"""
    blank = np.zeros_like(frames[..., 0])
    if blank.ndim == 3:
        # RGB frames carry their own colors; no colormap, no color scale:
        return ax.imshow(blank)
    return ax.imshow(blank, cmap='gray', vmin=vmin, vmax=vmax)


def plot_stimulus_percept(stim, percept, axes=None, figsize=None,
                          titles=('Stimulus', 'Percept'), stim_kwargs=None,
                          percept_kwargs=None):
    """Plot an image next to the percept it produced

    Draws ``stim`` and ``percept`` side by side, each with its own ``plot``
    method. A video has no single frame that stands for the whole sequence,
    and neither does the percept it produced, so use
    :py:func:`~pulse2percept.plotting.play_stimulus_percept` for those.

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
    figsize : (width, height), optional
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
    >>> stim = p2p.stimuli.LogoUCSB(resize=(24, 32))
    >>> model = p2p.models.ScoreboardModel(implant=p2p.implants.ArgusII(),
    ...                                    xrange=(-4, 4), yrange=(-4, 4),
    ...                                    step=0.5).build()
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

    Both panels run off a single clock. The percept's time axis is
    authoritative: every displayed percept frame is paired with the source
    frame that is up at the same physical time (zero-order hold), so a source
    and a percept sampled at different rates stay in register, and a still
    image stays put. ``fps`` resamples the whole presentation, exactly as in
    :py:meth:`~pulse2percept.percepts.Percept.play`.

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
    figsize : (width, height), optional
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
        Image format used to encode animation frames. 'jpg' keeps notebooks
        and doc pages smaller, which matters most for a video source; 'png'
        keeps the frames pixel-exact.
    vmin, vmax : float, optional
        Brightness limits for the percept. By default, ``vmin=0`` and ``vmax``
        is the maximum brightness across the percept. Not available for an RGB
        percept, whose values are shown as they are.

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
    src, src_idx = _source_frames(stim, timeline.times)
    # No constrained layout: the player measures the figure with the time
    # annotation blanked out, and a re-flow would move the panels under it.
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
    # Both panels run off one clock, so the time is annotated once, above them:
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
    # Both panels are handed to the player, which shows the source frame that
    # ``src_idx`` holds at each display frame:
    return HTMLAnimation(fig, update, data_gen, repeat=repeat,
                         intervals=timeline.intervals, save_count=idx.size,
                         image=[im_stim, im_percept],
                         frame_data=[src, percept.data],
                         frame_index=[src_idx, idx], labels=labels,
                         title=clock, fmt=fmt)
