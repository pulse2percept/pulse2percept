"""Drawing a :py:class:`~pulse2percept.stimuli.Stimulus`

Imported by :py:meth:`~pulse2percept.stimuli.Stimulus.plot` rather than at the
top of :py:mod:`~pulse2percept.stimuli.base`, so that the stimulus itself does
not depend on Matplotlib.
"""
import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from ..units import mW, mm, uA
from ..utils.constants import DT


def _cell_edges(t):
    """Turn sample times into the cell edges a heatmap colors between"""
    t = np.asarray(t, dtype=float)
    if t.size == 1:
        # No neighbor to split an interval with; one sample is one time step:
        return np.array([t[0] - DT / 2, t[0] + DT / 2])
    return np.concatenate(([t[0]], 0.5 * (t[:-1] + t[1:]), [t[-1]]))


def _times(stim, time):
    """Resolve a requested time range into an index and its x values"""
    # The user can ask for a range, slice, or list of time points, which are
    # either interpolated or loaded directly.
    if time is None:
        # Ask for a slice instead of `stim.time` to avoid interpolation:
        time = slice(None)
    # A range, a list of time points, or the endpoints and step of a slice
    # may all be given as quantities:
    time = stim._as_time(time)
    if isinstance(time, tuple):
        t_idx = (stim.time > time[0]) & (stim.time < time[1])
        # Include the end points (might have to be interpolated):
        t_vals = [time[0]] + list(stim.time[t_idx]) + [time[1]]
        t_idx = t_vals
    elif isinstance(time, (list, np.ndarray)):
        t_idx = time
        t_vals = time
    elif isinstance(time, slice):
        t_vals = stim._slice_times(time)
        if t_vals is None:
            # Every stored sample, taken by position:
            t_idx = time
            t_vals = stim.time[time]
        else:
            t_idx = t_vals
    elif time == Ellipsis:
        t_idx = time
        t_vals = stim.time[t_idx]
    else:
        raise TypeError(f'"time" must be a tuple, slice, list, or NumPy '
                        f'array, not {type(time)}.')
    return t_idx, t_vals


def _value_label(stim):
    """What the stimulus values are, as an axis or colorbar label"""
    if stim.unit.dimension.is_dimensionless:
        return 'Value'
    if stim.unit == uA:
        # Spelled the way Matplotlib renders it:
        return r'Amplitude ($\mu$A)'
    if stim.unit.dimension == (mW / mm ** 2).dimension:
        # Optical stimuli use irradiance rather than current amplitude.
        return f'Irradiance ({stim.unit})'
    return f'Amplitude ({stim.unit})'


def _traces(stim, electrodes, t_idx, t_vals, fmt, ax):
    """Draw one waveform per electrode, each in its own Axes"""
    owns_figure = ax is None
    axes = ax
    if axes is None:
        if len(electrodes) == 1:
            axes = plt.gca()
        else:
            axes = plt.subplots(nrows=len(electrodes),
                                figsize=(8, 1.2 * len(electrodes)),
                                layout='constrained')[1]
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for i, ax in enumerate(axes):
        if not isinstance(ax, Axes):
            raise TypeError(f"'ax' must be a list of subplots, but "
                            f"ax[{i}] is {type(ax)}.")
    if len(axes) != len(electrodes):
        raise ValueError(f"Number of subplots ({len(axes)}) must be equal "
                         f"to the number of electrodes "
                         f"({len(electrodes)}).")
    for ax, electrode in zip(axes, electrodes):
        # Slice or interpolate stimulus:
        slc = stim[electrode, t_idx]
        ax.plot(t_vals, np.squeeze(slc), fmt, linewidth=2)
        # Turn off the ugly box spines:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([slc.min(), 0, slc.max()])
        x_pad = 0.02 * (t_vals[-1] - t_vals[0])
        ax.set_xlim(t_vals[0] - x_pad, t_vals[-1] + x_pad)
        y_pad = np.maximum(1, 0.02 * (slc.max() - slc.min()))
        ax.set_ylim(slc.min() - y_pad, slc.max() + y_pad)
        ax.set_ylabel(electrode)
    # Only the bottom subplot carries the shared time axis:
    axes[-1].set_xticks(np.linspace(t_vals[0], t_vals[-1], num=5))
    axes[-1].set_xlabel(f'Time ({stim.time_unit})')
    if owns_figure:
        axes[-1].figure.supylabel(_value_label(stim))
    if len(axes) == 1:
        return axes[0]
    return axes


def _heatmap(stim, electrodes, t_idx, t_vals, ax):
    """Draw the stimulus as a single electrode-by-time image"""
    if isinstance(ax, (list, np.ndarray)):
        raise TypeError(f"A heatmap is drawn in a single Axes, but 'ax' "
                        f"is a sequence of {len(ax)}.")
    electrodes = list(electrodes)
    owns_figure = ax is None
    if ax is None:
        # Give every electrode a readable row of its own:
        height = float(np.clip(0.18 * len(electrodes), 2.5, 12))
        ax = plt.subplots(figsize=(8, height), layout='constrained')[1]
    elif not isinstance(ax, Axes):
        raise TypeError(f"'ax' must be a Matplotlib Axes, not {type(ax)}.")
    data = np.atleast_2d(stim[electrodes, t_idx])
    t_vals = np.asarray(t_vals, dtype=float)
    vmax = np.max(np.abs(data))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    if data.min() < 0:
        cmap, vmin = 'RdBu_r', -vmax
    else:
        cmap, vmin = 'viridis', 0.0
    mesh = ax.pcolormesh(_cell_edges(t_vals), np.arange(len(data) + 1),
                         data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_yticks(np.arange(len(data)) + 0.5,
                  labels=[str(e) for e in electrodes])
    # Read top to bottom, in the order the electrodes were asked for:
    ax.invert_yaxis()
    ax.set_xlabel(f'Time ({stim.time_unit})')
    ax.set_ylabel('Electrode')
    if owns_figure:
        ax.figure.colorbar(mesh, ax=ax, label=_value_label(stim))
    return ax


def plot_stimulus(stim, electrodes=None, time=None, fmt='k-', ax=None,
                  kind=None):
    """Implements :py:meth:`~pulse2percept.stimuli.Stimulus.plot`"""
    if stim.time is None:
        # Cannot plot stimulus with single time point:
        raise NotImplementedError
    if kind is None:
        if isinstance(ax, (list, np.ndarray)):
            kind = 'traces'
        elif electrodes is None and len(stim.electrodes) > 1:
            kind = 'heatmap'
        else:
            kind = 'traces'
    elif kind not in ('traces', 'heatmap'):
        raise ValueError(f"Unknown kind '{kind}'. Choose from 'traces' or "
                         f"'heatmap'.")
    if electrodes is None:
        electrodes = stim.electrodes
    elif isinstance(electrodes, (int, str)):
        electrodes = [electrodes]
    t_idx, t_vals = _times(stim, time)
    if kind == 'heatmap':
        return _heatmap(stim, electrodes, t_idx, t_vals, ax)
    return _traces(stim, electrodes, t_idx, t_vals, fmt, ax)
