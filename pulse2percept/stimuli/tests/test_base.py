import subprocess
import sys
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from copy import copy, deepcopy
from collections import OrderedDict as ODict
from matplotlib.axes import Subplot
import matplotlib.pyplot as plt

from pulse2percept.stimuli import Stimulus
from pulse2percept.stimuli import (BiphasicPulse, BiphasicPulseTrain,
                                   MonophasicPulse)
from pulse2percept.stimuli import ImageStimulus
from pulse2percept.stimuli import VideoStimulus
from pulse2percept.stimuli._merge import merge_time_axes
from pulse2percept.stimuli.base import _interp_rows
from pulse2percept.units import (DimensionMismatchError, Quantity,
                                 dimensionless, mA, ms, uA, us)
# `s` is a loop variable elsewhere in this module, so import the unit
# under a name that cannot be shadowed by one:
from pulse2percept.units import s as sec
from pulse2percept.utils.constants import DT
from pulse2percept.utils.testing import assert_warns_msg


def test_Stimulus():
    # Slots:
    npt.assert_equal(hasattr(Stimulus(1), '__slots__'), True)
    npt.assert_equal(hasattr(Stimulus(1), '__dict__'), False)
    # One electrode:
    stim = Stimulus(3)
    npt.assert_equal(stim.shape, (1, 1))
    npt.assert_equal(stim.electrodes, [0])
    npt.assert_equal(stim.time, None)
    # One electrode with a name:
    stim = Stimulus(3, electrodes='AA001')
    npt.assert_equal(stim.shape, (1, 1))
    npt.assert_equal(stim.electrodes, ['AA001'])
    npt.assert_equal(stim.time, None)
    # Ten electrodes, one will be trimmed:
    stim = Stimulus(np.arange(10), compress=True)
    npt.assert_equal(stim.shape, (9, 1))
    npt.assert_equal(stim.electrodes, np.arange(1, 10))
    npt.assert_equal(stim.time, None)
    # Electrodes + specific time, time will be trimmed:
    stim = Stimulus(np.ones((4, 3)), time=[-3, -2, -1], compress=True)
    npt.assert_equal(stim.shape, (4, 2))
    npt.assert_equal(stim.time, [-3, -1])
    # Electrodes + specific time, but don't trim:
    stim = Stimulus(np.ones((4, 3)), time=[-3, -2, -1], compress=False)
    npt.assert_equal(stim.shape, (4, 3))
    npt.assert_equal(stim.time, [-3, -2, -1])
    # Specific names:
    stim = Stimulus({'A1': 3, 'C5': 8})
    npt.assert_equal(stim.shape, (2, 1))
    npt.assert_equal(np.sort(stim.electrodes), np.sort(['A1', 'C5']))
    npt.assert_equal(stim.time, None)
    # Specific names, renamed:
    stim = Stimulus({'A1': 3, 'C5': 8}, electrodes=['B7', 'B8'])
    npt.assert_equal(stim.shape, (2, 1))
    npt.assert_equal(np.sort(stim.electrodes), np.sort(['B7', 'B8']))
    npt.assert_equal(stim.time, None)
    # Electrodes x time, time will be trimmed:
    stim = Stimulus(np.ones((6, 100)), compress=True)
    npt.assert_equal(stim.shape, (6, 2))
    # Single electrode in time:
    stim = Stimulus([[1, 5, 7, 2, 4]])
    npt.assert_equal(stim.electrodes, [0])
    npt.assert_equal(stim.shape, (1, 5))
    # Specific electrode in time:
    stim = Stimulus({'C3': [[1, 4, 4, 3, 6]]})
    npt.assert_equal(stim.electrodes, ['C3'])
    npt.assert_equal(stim.shape, (1, 5))
    # Multiple specific electrodes in time:
    stim = Stimulus({'C3': [[0, 1, 2, 3]],
                     'F4': [[4, -1, 4, -1]]})
    # Stimulus from a Stimulus (might happen in ProsthesisSystem):
    stim = Stimulus(Stimulus(4), electrodes='B3')
    npt.assert_equal(stim.shape, (1, 1))
    npt.assert_equal(stim.electrodes, ['B3'])
    npt.assert_equal(stim.time, None)
    # Saves metadata:
    metadata = {'a': 0, 'b': 1}
    stim = Stimulus(3, metadata=metadata)
    npt.assert_equal(stim.metadata['user'], metadata)
    # List of lists instead of 2D NumPy array:
    stim = Stimulus([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], compress=True)
    npt.assert_equal(stim.shape, (2, 2))
    npt.assert_equal(stim.electrodes, [0, 1])
    npt.assert_equal(stim.time, [0, 4])
    # Tuple of tuples instead of 2D NumPy array:
    stim = Stimulus(((1, 1, 1, 1, 1), (1, 1, 1, 1, 1)), compress=True)
    npt.assert_equal(stim.shape, (2, 2))
    npt.assert_equal(stim.electrodes, [0, 1])
    npt.assert_equal(stim.time, [0, 4])
    # Zero activation:
    source = np.zeros((2, 4))
    stim = Stimulus(source, compress=True)
    npt.assert_equal(stim.shape, (0, 2))
    npt.assert_equal(stim.time, [0, source.shape[1] - 1])
    stim = Stimulus(source, compress=False)
    npt.assert_equal(stim.shape, source.shape)
    npt.assert_equal(stim.time, np.arange(source.shape[1]))
    # Annoying but possible:
    stim = Stimulus([])
    npt.assert_equal(stim.time, None)
    npt.assert_equal(len(stim.data), 0)
    npt.assert_equal(len(stim.electrodes), 0)
    npt.assert_equal(stim.shape, (0,))

    # Rename electrodes:
    stim = Stimulus(np.ones((2, 5)), compress=True)
    npt.assert_equal(stim.electrodes, [0, 1])
    stim = Stimulus(stim, electrodes=['A3', 'B8'])
    npt.assert_equal(stim.electrodes, ['A3', 'B8'])
    npt.assert_equal(stim.time, [0, 4])

    # Individual stimuli might already have electrode names:
    stim = Stimulus([Stimulus(1, electrodes='B1')])
    npt.assert_equal(stim.electrodes, ['B1'])
    # Duplicate names will be fixed (with a warning message):
    stim = Stimulus([Stimulus(1), Stimulus(2)])
    npt.assert_equal(stim.electrodes, [0, 1])
    # When passing a dict and the stimuli already have electrode names, the
    # keys of the dict prevail:
    stim = Stimulus({'A1': Stimulus(1, electrodes='B2')})
    npt.assert_equal(stim.electrodes, ['A1'])

    # Specify new time points:
    stim = Stimulus(np.ones((2, 5)), compress=True)
    npt.assert_equal(stim.time, [0, 4])
    stim = Stimulus(stim, time=np.array(stim.time) / 10.0)
    npt.assert_equal(stim.electrodes, [0, 1])
    npt.assert_almost_equal(stim.time, [0, 0.4])

    # Charge-balanced:
    npt.assert_equal(Stimulus(0).is_charge_balanced, True)
    npt.assert_equal(Stimulus(1).is_charge_balanced, False)
    npt.assert_equal(Stimulus([0, 0]).is_charge_balanced, True)
    npt.assert_equal(Stimulus([[0, 0]]).is_charge_balanced, True)
    npt.assert_equal(Stimulus([1, -1]).is_charge_balanced, False)
    npt.assert_equal(Stimulus([[1, -1]]).is_charge_balanced, True)
    npt.assert_equal(Stimulus([[1, -1], [0, 0.5]]).is_charge_balanced, False)

    # Not allowed:
    with pytest.raises(ValueError):
        # First one doesn't have time:
        stim = Stimulus({'A2': 1, 'C3': [[1, 2, 3]]})
    with pytest.raises(ValueError):
        # Invalid source type:
        stim = Stimulus(np.ones((3, 4, 5, 6)))
    with pytest.raises(TypeError):
        # Invalid source type:
        stim = Stimulus("invalid")
    with pytest.raises(ValueError):
        # Wrong number of electrodes:
        stim = Stimulus([3, 4], electrodes='A1')
    with pytest.raises(ValueError):
        # Wrong number of time points:
        stim = Stimulus(np.ones((3, 5)), time=[0, 1, 2])
    with pytest.raises(ValueError):
        # Can't force time:
        stim = Stimulus(3, time=[0.4])
    assert_warns_msg(UserWarning, Stimulus, None, [[1, 2, 3]], time=[1, 2, 1.9])


def test_Stimulus_time_resolution():
    # Time is stored as float64 while data stays float32. float32 reaches a
    # resolution of DT at t = 8.4 s, past which two time points a time step
    # apart are no longer distinguishable:
    stim = Stimulus(np.ones((2, 3)), time=[0, DT, 2 * DT])
    npt.assert_equal(stim.time.dtype, np.float64)
    npt.assert_equal(stim.data.dtype, np.float32)
    far = 30000.0
    stim = Stimulus(np.ones((2, 3)), time=[far, far + DT, far + 2 * DT])
    npt.assert_almost_equal(np.diff(stim.time), DT)
    # Slicing the time axis does not round it back down either:
    npt.assert_almost_equal(stim[0, far + DT], 1)


def test_Stimulus_nonmonotonic_warning():
    # The warning names the offending points rather than dumping the whole
    # time axis, which for a long stimulus ran to megabytes:
    time = np.arange(1000, dtype=float)
    time[500] = time[499]
    with pytest.warns(UserWarning, match='strictly monotonically') as record:
        Stimulus(np.ones((2, 1000)), time=time)
    msg = str(record[0].message)
    npt.assert_equal(len(msg) < 500, True)
    npt.assert_equal('t[499]' in msg, True)
    npt.assert_equal('1 of 1000' in msg, True)


def test_Stimulus_compress():
    data = np.zeros((2, 7))
    data[0, 0] = 1
    stim = Stimulus(data)
    npt.assert_equal(stim.shape, (2, 7))
    npt.assert_equal(stim.is_compressed, False)
    stim.compress()
    npt.assert_equal(stim.is_compressed, True)
    # Compress gets rid of the second electrode, and only keeps the signal
    # edges:
    npt.assert_equal(stim.shape, (1, 3))
    npt.assert_almost_equal(stim.time, [0, 1, 6])
    # Repeated calls don't change the outcome:
    stim.compress()
    npt.assert_equal(stim.is_compressed, True)
    npt.assert_equal(stim.shape, (1, 3))
    npt.assert_almost_equal(stim.time, [0, 1, 6])

    # All zeros:
    stim = Stimulus(np.zeros((3, 6)))
    npt.assert_equal(stim.shape, ((3, 6)))
    stim.compress()
    # Empty:
    npt.assert_equal(stim.shape, (0, 2))
    npt.assert_almost_equal(stim.time, [0, 5])

    # Compress has no effect:
    time = [3, 6, 7, 9, 10]
    stim = Stimulus(np.eye(len(time)), time=time)
    npt.assert_equal(stim.shape, (len(time), len(time)))
    npt.assert_almost_equal(stim.time, time)
    npt.assert_equal(stim.is_compressed, False)
    stim.compress()
    npt.assert_equal(stim.is_compressed, True)
    npt.assert_equal(stim.shape, (len(time), len(time)))
    npt.assert_almost_equal(stim.time, time)

    with pytest.raises(AttributeError):
        stim.is_compressed = True


def test_Stimulus_append():
    # Basic usage:
    stim = Stimulus([[0, 1, 0]], time=[0, 1, 2])
    stim2 = Stimulus([[0, 2]], time=[0, 0.5])
    comb = stim.append(stim2)
    # End point of stim and starting point of stim2 will be merged:
    npt.assert_almost_equal(comb.data, [[0, 1, 0, 2]])
    npt.assert_almost_equal(comb.time, [0, 1, 2, 2.5])

    # When other stimulus is shifted:
    comb = stim.append(stim2 >> 10)
    npt.assert_almost_equal(comb.time, [0, 1, 2, 12, 12.5])

    with pytest.raises(TypeError):
        # 'other' must be Stimulus:
        stim.append(np.array([[0, 1, 2]]))
    with pytest.raises(ValueError):
        # other cannot have time=None:
        stim.append(Stimulus(3))
    with pytest.raises(ValueError):
        # self cannot have time=None:
        Stimulus(3).append(stim)
    with pytest.raises(ValueError):
        stim.append(Stimulus([[1, 2]], electrodes='B1'))
    with pytest.raises(NotImplementedError):
        # negative time axis:
        stim.append(Stimulus([[0, 2]], time=[-1, 0]))


def test_Stimulus_plot():
    # Stimulus with one electrode
    stim = Stimulus([[0, -10, 10, -10, 10, -10, 0]],
                    time=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    for time in [None, Ellipsis, slice(None)]:
        # Different ways to plot all data points:
        fig, ax = plt.subplots()
        stim.plot(time=time, ax=ax)
        npt.assert_equal(isinstance(ax, Subplot), True)
        npt.assert_almost_equal(ax.get_yticks(), [stim.data.min(), 0,
                                                  stim.data.max()])
        npt.assert_equal(len(ax.lines), 1)
        npt.assert_almost_equal(ax.lines[0].get_data()[1].min(),
                                stim.data.min())
        npt.assert_almost_equal(ax.lines[0].get_data()[1].max(),
                                stim.data.max())
        plt.close(fig)

    # Plot a range of time values (times are sliced, but end points are
    # interpolated):
    fig, ax = plt.subplots()
    ax = stim.plot(time=(0.2, 0.6), ax=ax)
    npt.assert_equal(isinstance(ax, Subplot), True)
    npt.assert_equal(len(ax.lines), 1)
    t_vals = ax.lines[0].get_data()[0]
    npt.assert_almost_equal(t_vals[0], 0.2)
    npt.assert_almost_equal(t_vals[-1], 0.6)
    plt.close(fig)

    # Plot exact time points:
    t_vals = [0.2, 0.3, 0.4]
    fig, ax = plt.subplots()
    stim.plot(time=t_vals, ax=ax)
    npt.assert_equal(isinstance(ax, Subplot), True)
    npt.assert_equal(len(ax.lines), 1)
    npt.assert_almost_equal(ax.lines[0].get_data()[0], t_vals)
    npt.assert_almost_equal(ax.lines[0].get_data()[1],
                            np.squeeze(stim[:, t_vals]))

    # Plot multiple electrodes with string names:
    for n_electrodes in [2, 3, 4]:
        stim = Stimulus(np.random.rand(n_electrodes, 20),
                        electrodes=[f'E{i}' for i in range(n_electrodes)])
        fig, axes = plt.subplots(ncols=n_electrodes)
        stim.plot(ax=axes)
        npt.assert_equal(isinstance(axes, (list, np.ndarray)), True)
        for ax, electrode in zip(axes, stim.electrodes):
            npt.assert_equal(isinstance(ax, Subplot), True)
            npt.assert_equal(len(ax.lines), 1)
            npt.assert_equal(ax.get_ylabel(), electrode)
            npt.assert_almost_equal(ax.lines[0].get_data()[0], stim.time)
            npt.assert_almost_equal(ax.lines[0].get_data()[1],
                                    stim[electrode, :])
        plt.close(fig)

    # Invalid calls:
    with pytest.raises(TypeError):
        stim.plot(electrodes=1.2)
    with pytest.raises(TypeError):
        stim.plot(time=0)
    with pytest.raises(TypeError):
        stim.plot(ax='as')
    with pytest.raises(TypeError):
        stim.plot(time='0 0.1')
    with pytest.raises(ValueError):
        stim.plot(kind='waterfall')
    with pytest.raises(NotImplementedError):
        Stimulus(np.ones(10)).plot()
    with pytest.raises(ValueError):
        stim = Stimulus(np.ones((3, 10)))
        _, axes = plt.subplots(nrows=4)
        stim.plot(ax=axes)
    with pytest.raises(TypeError):
        stim = Stimulus(np.ones((3, 10)))
        _, axes = plt.subplots(nrows=3)
        axes[1] = 0
        stim.plot(ax=axes)
    with pytest.raises(TypeError):
        # A heatmap has nowhere to put a second Axes:
        _, axes = plt.subplots(nrows=3)
        Stimulus(np.ones((3, 10))).plot(ax=axes, kind='heatmap')
    plt.close('all')


def test_Stimulus_plot_kind_default():
    """A named electrode asks for detail, a whole implant for an overview"""
    single = Stimulus([[0, -10, 10, 0]], time=[0, 1, 2, 3])
    npt.assert_equal(len(single.plot().lines), 1)
    plt.close('all')
    multi = Stimulus(np.random.rand(4, 8),
                     electrodes=[f'E{i}' for i in range(4)])
    ax = multi.plot()
    npt.assert_equal(isinstance(ax, Subplot), True)
    npt.assert_equal(len(ax.lines), 0)
    npt.assert_equal(len(ax.collections), 1)
    plt.close('all')
    # Naming electrodes means traces, however many are named:
    axes = multi.plot(electrodes=['E0', 'E1', 'E2'])
    npt.assert_equal(len(axes), 3)
    npt.assert_equal([ax.get_ylabel() for ax in axes], ['E0', 'E1', 'E2'])
    plt.close('all')
    # Handing over an Axes per electrode says the same thing:
    _, axes = plt.subplots(nrows=4)
    multi.plot(ax=axes)
    npt.assert_equal([len(ax.lines) for ax in axes], [1, 1, 1, 1])
    plt.close('all')
    # `kind` overrides all of that:
    npt.assert_equal(len(multi.plot(kind='traces')), 4)
    plt.close('all')
    ax = multi.plot(electrodes=['E2', 'E0'], kind='heatmap')
    npt.assert_equal([t.get_text() for t in ax.get_yticklabels()],
                     ['E2', 'E0'])
    plt.close('all')
    with pytest.raises(TypeError):
        # ... including into a request a heatmap cannot honor:
        _, axes = plt.subplots(nrows=4)
        multi.plot(ax=axes, kind='heatmap')
    plt.close('all')


def test_Stimulus_plot_electrode_order():
    """Electrodes are shown in the order they were asked for, not stored in"""
    stim = Stimulus(np.arange(12, dtype=float).reshape((3, 4)),
                    electrodes=['A1', 'A2', 'A3'], time=[0, 1, 2, 3])
    axes = stim.plot(electrodes=['A3', 'A1'])
    npt.assert_equal([ax.get_ylabel() for ax in axes], ['A3', 'A1'])
    npt.assert_almost_equal(axes[0].lines[0].get_data()[1], stim['A3'])
    plt.close('all')
    # Electrodes may also be named by index:
    axes = stim.plot(electrodes=[2, 0])
    npt.assert_almost_equal(axes[0].lines[0].get_data()[1], stim['A3'])
    npt.assert_almost_equal(axes[1].lines[0].get_data()[1], stim['A1'])
    plt.close('all')
    ax = stim.plot(electrodes=[2, 0], kind='heatmap')
    npt.assert_almost_equal(np.asarray(ax.collections[0].get_array()),
                            stim.data[[2, 0], :])
    plt.close('all')


def test_Stimulus_plot_heatmap_time_is_not_uniform():
    """A compressed pulse train is not sampled at a constant rate

    Drawing it with equal-width columns would stretch the DT-wide edges of a
    pulse until they look as long as the gaps between pulses.
    """
    stim = Stimulus({'A1': BiphasicPulseTrain(20, 50, 0.45, stim_dur=100),
                     'A2': BiphasicPulseTrain(20, 25, 0.45, stim_dur=100)})
    dt = np.diff(stim.time)
    npt.assert_equal(dt.max() > 100 * dt.min(), True)
    x = stim.plot().collections[0].get_coordinates()[0, :, 0]
    # One cell per stored sample, spanning exactly the stimulus duration:
    npt.assert_equal(len(x), len(stim.time) + 1)
    npt.assert_almost_equal([x[0], x[-1]], [stim.time[0], stim.time[-1]])
    # Cells are as wide as the intervals they stand for, so a pulse edge stays
    # a pulse edge and the gaps stay long:
    npt.assert_almost_equal(x[1:-1], (stim.time[:-1] + stim.time[1:]) / 2)
    widths = np.diff(x)
    npt.assert_equal(widths.min() < DT, True)
    npt.assert_equal(widths.max() > 10, True)
    plt.close('all')
    # A time selection is drawn on the same footing:
    x = stim.plot(time=(1 * ms, 3 * ms)).collections[0]
    x = x.get_coordinates()[0, :, 0]
    npt.assert_almost_equal([x[0], x[-1]], [1, 3])
    plt.close('all')


def test_Stimulus_plot_heatmap_color_scale():
    # A signed stimulus is normalized symmetrically around zero, so that the
    # middle of the colormap is "no current":
    signed = Stimulus([[0, -30, 10, 0], [0, 5, -2, 0]], time=[0, 1, 2, 3])
    mesh = signed.plot().collections[0]
    npt.assert_almost_equal([mesh.norm.vmin, mesh.norm.vmax], [-30, 30])
    plt.close('all')
    # Nonnegative data has no sign to show and is not drawn as if it did:
    unsigned = Stimulus([[0, 1, 2, 3], [0, 2, 4, 6]], time=[0, 1, 2, 3])
    unsigned_mesh = unsigned.plot().collections[0]
    npt.assert_almost_equal([unsigned_mesh.norm.vmin, unsigned_mesh.norm.vmax],
                            [0, 6])
    npt.assert_equal(unsigned_mesh.cmap.name == mesh.cmap.name, False)
    plt.close('all')
    # An all-zero stimulus has no magnitude to scale by, and must not blow up:
    mesh = Stimulus(np.zeros((3, 4)), time=[0, 1, 2, 3]).plot().collections[0]
    npt.assert_equal(mesh.norm.vmin < mesh.norm.vmax, True)
    plt.close('all')


def test_Stimulus_plot_heatmap_electrode_selection():
    stim = Stimulus(np.arange(12, dtype=float).reshape((3, 4)),
                    electrodes=['A1', 'A2', 'A3'], time=[0, 1, 2, 3])
    ax = stim.plot(electrodes='A2', kind='heatmap')
    npt.assert_equal([t.get_text() for t in ax.get_yticklabels()], ['A2'])
    npt.assert_almost_equal(np.asarray(ax.collections[0].get_array()).ravel(),
                            stim['A2'])
    plt.close('all')
    ax = stim.plot(electrodes=['A1', 'A3'], kind='heatmap')
    npt.assert_almost_equal(np.asarray(ax.collections[0].get_array()),
                            stim.data[[0, 2], :])
    plt.close('all')


def test_Stimulus_plot_leaves_the_callers_figure_alone():
    """A supplied Axes is the whole canvas `plot` gets

    Positions are only compared between Axes `plot` was not given, so this
    does not pin down what a plot looks like -- only that the rest of the
    figure survives it.
    """
    stim = Stimulus(np.random.rand(3, 8), electrodes=['E0', 'E1', 'E2'])
    for kwargs in ({'kind': 'heatmap'}, {'electrodes': ['E0']}):
        fig, (mine, theirs) = plt.subplots(ncols=2)
        before = theirs.get_position().bounds
        stim.plot(ax=mine, **kwargs)
        npt.assert_almost_equal(theirs.get_position().bounds, before)
        # A figure-wide layout engine would move every Axes at draw time:
        npt.assert_equal(fig.get_layout_engine(), None)
        # Nothing figure-level was added either: no shared label, and no
        # colorbar Axes squeezing in next to the caller's own:
        npt.assert_equal(fig.get_supylabel(), '')
        npt.assert_equal(fig.texts, [])
        npt.assert_equal(fig.axes, [mine, theirs])
        plt.close(fig)
    # Owning the figure is what earns `plot` the decoration:
    ax = stim.plot()
    npt.assert_equal(ax.collections[0].colorbar is not None, True)
    npt.assert_equal(stim.plot(kind='traces')[0].figure.get_supylabel(),
                     r'Amplitude ($\mu$A)')
    plt.close('all')


def _unique_timepoints(stim, data):
    data['data'] = np.array([[1, 0, 1, 0, 2, 0, 1]])
    data['time'] = np.array([0, 1, 1.5, 2, 2.1, 2.10000000000001, 2.2])
    data['electrodes'] = np.arange(1)
    stim._stim = data


def test_Stimulus__stim():
    stim = Stimulus(3)
    # User could try and motify the data container after the constructor, which
    # would lead to inconsistencies between data, electrodes, time. The new
    # property setting mechanism prevents that.
    # Requires dict:
    with pytest.raises(AttributeError):
        stim._stim = np.array([0, 1])
    # Dict must have all required fields:
    fields = ['data', 'electrodes', 'time']
    for field in fields:
        _fields = deepcopy(fields)
        _fields.remove(field)
        with pytest.raises(AttributeError):
            stim._stim = {f: None for f in _fields}
    # Data must be a 2-D NumPy array:
    data = {f: None for f in fields}
    with pytest.raises(ValueError):
        data['data'] = np.ones(3)
        stim._stim = data
    # Data rows must match electrodes:
    with pytest.raises(ValueError):
        data['data'] = np.ones((3, 4))
        data['time'] = np.arange(4)
        data['electrodes'] = np.arange(2)
        stim._stim = data
    # Data columns must match time:
    with pytest.raises(ValueError):
        data['data'] = np.ones((3, 4))
        data['electrodes'] = np.arange(3)
        data['time'] = np.arange(7)
        stim._stim = data
    # Time points must be unique:
    assert_warns_msg(UserWarning, _unique_timepoints, None, stim, data)
    # But if you do all the things right, you can reset the stimulus by hand:
    data['data'] = np.ones((3, 1))
    data['electrodes'] = np.arange(3)
    data['time'] = None
    stim._stim = data

    data['data'] = np.ones((3, 1))
    data['electrodes'] = np.arange(3)
    data['time'] = np.arange(1)
    stim._stim = data

    data['data'] = np.ones((3, 4))
    data['electrodes'] = np.arange(3)
    data['time'] = np.array([0, 1, 1 + DT, 2])
    stim._stim = data


def test_Stimulus___eq__():
    # Two Stimulus objects created from the same source data are considered
    # equal:
    for source in [3, [], np.ones(3), [3, 4, 5], np.ones((3, 6))]:
        npt.assert_equal(Stimulus(source) == Stimulus(source), True)
    stim = Stimulus(np.ones((2, 3)), compress=True)
    # Compressed vs uncompressed:
    npt.assert_equal(stim == Stimulus(np.ones((2, 3)), compress=False), False)
    npt.assert_equal(stim != Stimulus(np.ones((2, 3)), compress=False), True)
    # Different electrode names:
    npt.assert_equal(stim == Stimulus(stim, electrodes=[0, 'A2']), False)
    # Different time points:
    npt.assert_equal(stim == Stimulus(stim, time=[0, 3], compress=True), False)
    # Different data shape:
    npt.assert_equal(stim == Stimulus(np.ones((2, 4))), False)
    npt.assert_equal(stim == Stimulus(np.ones(2)), False)
    # Different data points:
    npt.assert_equal(stim == Stimulus(np.ones((2, 3)) * 1.1, compress=True),
                     False)
    # Different shape
    npt.assert_equal(stim == Stimulus(np.ones((2, 5))), False)
    # Different type:
    npt.assert_equal(stim == ODict(), False)
    npt.assert_equal(stim != ODict(), True)
    # Time vs no time:
    npt.assert_equal(Stimulus(2) == stim, False)
    # Annoying but possible:
    npt.assert_equal(Stimulus([]), Stimulus(()))


def test_Stimulus___getitem__():
    stim = Stimulus(1 + np.arange(12).reshape((3, 4)))
    # Slicing:
    npt.assert_equal(stim[:], stim.data)
    npt.assert_equal(stim[...], stim.data)
    npt.assert_equal(stim[:, :], stim.data)
    npt.assert_equal(stim[:2], stim.data[:2])
    npt.assert_equal(stim[:, 0.0], stim.data[:, 0].reshape((-1, 1)))
    npt.assert_equal(stim[0, :], stim.data[0, :])
    npt.assert_equal(stim[0, ...], stim.data[0, ...])
    npt.assert_equal(stim[..., 0], stim.data[..., 0].reshape((-1, 1)))
    # More advanced slicing of time is possible, but needs a step size:
    with pytest.raises(ValueError):
        stim[:, 2:5]
    with pytest.raises(ValueError):
        stim[:, :3]
    with pytest.raises(ValueError):
        stim[:, 2:]
    npt.assert_almost_equal(stim[0, 1.2:1.65:0.15], [[2.2, 2.35, 2.5]])
    npt.assert_almost_equal(stim[0, :0.6:0.2], [[1.0, 1.2, 1.4]])
    npt.assert_almost_equal(stim[0, 2.7::0.2], [[3.7, 3.9]])
    npt.assert_almost_equal(stim[0, ::2.6], [[1.0, 3.6]])
    # Single element:
    npt.assert_equal(stim[0, 0], stim.data[0, 0])
    # Interpolating time:
    npt.assert_almost_equal(stim[0, 2.6], 3.6)
    npt.assert_almost_equal(stim[..., 2.3], np.array([[3.3], [7.3], [11.3]]),
                            decimal=3)
    # The second dimension is not a column index!
    npt.assert_almost_equal(stim[0, 0], 1.0)
    npt.assert_almost_equal(stim[0, [0, 1]], np.array([[1.0, 2.0]]))
    npt.assert_almost_equal(stim[0, [0.21, 1]], np.array([[1.21, 2.0]]))
    npt.assert_almost_equal(stim[[0, 1], [0.21, 1]],
                            np.array([[1.21, 2.0], [5.21, 6.0]]))

    # "Valid" index errors:
    with pytest.raises(IndexError):
        stim[10, :]
    with pytest.raises(IndexError):
        stim[3.3, 0]

    # Times can be extrapolated (take on value of end points):
    stim = Stimulus(1 + np.arange(12).reshape((3, 4)))
    npt.assert_almost_equal(stim[0, 9.901], 4)
    # If time=None, you cannot interpolate/extrapolate:
    stim = Stimulus([3, 4, 5])
    npt.assert_almost_equal(stim[0], stim.data[0, 0])
    with pytest.raises(ValueError):
        stim[0, 0.2]

    # With a single time point, interpolate is still possible:
    stim = Stimulus(np.arange(3).reshape((-1, 1)))
    npt.assert_almost_equal(stim[0], stim.data[0, 0])
    npt.assert_almost_equal(stim[0, 0], stim.data[0, 0])
    npt.assert_almost_equal(stim[0, 3.33], stim.data[0, 0])

    # Annoying but possible:
    stim = Stimulus([])
    npt.assert_almost_equal(stim[:], stim.data)
    with pytest.raises(IndexError):
        stim[0]

    # Electrodes by string:
    stim = Stimulus([[0, 1], [2, 3]], electrodes=['A1', 'B2'])
    npt.assert_almost_equal(stim['A1'], [0, 1])
    npt.assert_almost_equal(stim['A1', :], [0, 1])
    npt.assert_almost_equal(stim[['A1', 'B2'], 0], [[0], [2]])
    npt.assert_almost_equal(stim[['A1', 'B2'], :], stim.data)

    # Electrodes by slice:
    stim = Stimulus(np.arange(10))
    npt.assert_almost_equal(stim[1::3], np.array([[1], [4], [7]]))

    # Binary arrays:
    stim = Stimulus(np.arange(6).reshape((2, 3)),
                    electrodes=['A1', 'B2'],
                    time=[0.1, 0.3, 0.5])
    npt.assert_almost_equal(stim[stim.electrodes != 'A1', :], [[3, 4, 5]])
    npt.assert_almost_equal(stim[stim.electrodes == 'B2', :], [[3, 4, 5]])
    npt.assert_almost_equal(stim[stim.electrodes == 'C9', :], np.zeros((0, 3)))
    npt.assert_almost_equal(stim[stim.electrodes == 'C9', 0.1].size, 0)
    npt.assert_almost_equal(stim[stim.electrodes == 'B2', 0.1001], 3.0005,
                            decimal=3)
    npt.assert_almost_equal(stim[stim.electrodes == 'B2', 0.2], 3.5)
    npt.assert_almost_equal(stim[:, stim.time < 0.4], [[0, 1], [3, 4]])
    npt.assert_almost_equal(stim[stim.electrodes == 'B2', stim.time < 0.4],
                            [3, 4])
    npt.assert_almost_equal(stim[:, stim.time > 0.6], np.zeros((2, 0)))
    npt.assert_almost_equal(stim['A1', stim.time > 0.6].size, 0)
    npt.assert_almost_equal(stim['A1', np.isclose(stim.time, 0.3)], [1])


def test_Stimulus_merge():
    # We can stack multiple stimuli together - their time axes will be merged:
    stim1 = Stimulus([[0, 1, 2, 3, 4]], time=[0, 1, 2, 3, 4])
    stim2 = Stimulus([[0, 1, 2]], time=[-0.5, 1.5, 4.5])
    merge = Stimulus([stim1, stim2])
    npt.assert_almost_equal(merge.time, np.unique(np.hstack((stim1.time,
                                                             stim2.time))),
                            decimal=6)
    npt.assert_almost_equal(merge[0, [0, -1]], stim1[0, [0, -1]])
    npt.assert_almost_equal(merge[1, [0, -1]], stim2[0, [0, -1]])

    # We can keep stacking - even when nested:
    stim3 = Stimulus([[14]], time=[9.7])
    merge2 = Stimulus([merge, stim3])
    npt.assert_almost_equal(merge2.time, np.unique((np.hstack((stim1.time,
                                                               stim2.time,
                                                               stim3.time)))),
                            decimal=6)
    npt.assert_almost_equal(merge2[0, [0, -1]], stim1[0, [0, -1]])
    npt.assert_almost_equal(merge2[1, [0, -1]], stim2[0, [0, -1]])
    npt.assert_almost_equal(merge2[2, [0, -1]], stim3[0, [0, -1]])


@pytest.mark.parametrize('scalar', (12345.678, -2.3, np.pi))
def test_Stimulus_arithmetic(scalar):
    stim = Stimulus([[0, 21, -13, 0, 0]], time=[0, 1, 2, 3, 4])
    npt.assert_almost_equal((stim + scalar).data,
                            stim.data + scalar, decimal=5)
    npt.assert_almost_equal((scalar + stim).data,
                            scalar + stim.data, decimal=5)
    npt.assert_almost_equal((stim - scalar).data,
                            stim.data - scalar, decimal=5)
    npt.assert_almost_equal((scalar - stim).data,
                            scalar - stim.data, decimal=5)
    npt.assert_almost_equal((stim * scalar).data,
                            stim.data * scalar, decimal=5)
    npt.assert_almost_equal((scalar * stim).data,
                            scalar * stim.data, decimal=5)
    npt.assert_almost_equal((stim / scalar).data,
                            stim.data / scalar, decimal=5)
    npt.assert_almost_equal((-stim).data,
                            -1 * stim.data, decimal=5)
    npt.assert_almost_equal((stim >> scalar).time,
                            stim.time + scalar, decimal=5)
    npt.assert_almost_equal((stim << scalar).time,
                            stim.time - scalar, decimal=5)
    # 10 / stim is not supported because it will always give a division by
    # zero error:
    with pytest.raises(TypeError):
        s = scalar / stim
    with pytest.raises(TypeError):
        s = stim + stim
    with pytest.raises(TypeError):
        s = stim - stim
    with pytest.raises(TypeError):
        s = stim * stim
    with pytest.raises(TypeError):
        s = stim / stim
    with pytest.raises(TypeError):
        s = stim + [1, 1]
    with pytest.raises(TypeError):
        s = stim * np.array([2, 3])
    with pytest.raises(TypeError):
        s = stim >> np.array([2, 3])
    with pytest.raises(TypeError):
        s = stim << np.array([2, 3])


def test_Stimulus_remove():
    stim = Stimulus([[0, 1, 2], [3, 4, 5]], electrodes=['A1', 'C3'])
    npt.assert_equal('A1' in stim.electrodes, True)
    npt.assert_equal('C3' in stim.electrodes, True)
    stim.remove('A1')
    npt.assert_equal('A1' in stim.electrodes, False)
    npt.assert_equal('C3' in stim.electrodes, True)

    # Electrode 0 must be removable: `0` is falsy, but it is a valid index:
    stim = Stimulus([[0, 1, 2], [3, 4, 5]])
    stim.remove(0)
    npt.assert_equal(stim.shape, (1, 3))
    npt.assert_equal(stim.electrodes, [1])
    npt.assert_almost_equal(stim.data, [[3, 4, 5]])
    # Removing index 0 by index or by name gives the same result:
    stim = Stimulus([[0, 1, 2], [3, 4, 5]], electrodes=['A1', 'C3'])
    stim.remove(0)
    npt.assert_equal(stim.electrodes, ['C3'])

    # Removing a list of electrodes:
    stim = Stimulus([[0, 1, 2], [3, 4, 5]], electrodes=['A1', 'C3'])
    stim.remove(['A1', 'C3'])
    npt.assert_equal(stim.shape, (0, 3))

    # Removing "nothing" is a no-op. ProsthesisSystem relies on this when none
    # of its electrodes are deactivated:
    for nothing in (None, [], (), np.array([])):
        stim = Stimulus([[0, 1, 2], [3, 4, 5]])
        stim.remove(nothing)
        npt.assert_equal(stim.shape, (2, 3))
        npt.assert_equal(stim.electrodes, [0, 1])

    # After 'all', `electrodes` must stay an array of the same dtype, so that
    # it can still be indexed with a boolean mask:
    stim = Stimulus([[0, 1, 2], [3, 4, 5]], electrodes=['A1', 'C3'])
    dtype = stim.electrodes.dtype
    stim.remove('all')
    npt.assert_equal(isinstance(stim.electrodes, np.ndarray), True)
    npt.assert_equal(stim.electrodes.dtype, dtype)
    npt.assert_equal(stim.shape, (0, 3))
    npt.assert_equal(stim.electrodes[np.zeros(0, dtype=bool)].size, 0)

    # Unknown electrodes are an error:
    stim = Stimulus([[0, 1, 2], [3, 4, 5]], electrodes=['A1', 'C3'])
    with pytest.raises(ValueError):
        stim.remove('B2')


def test_Stimulus_duplicate_electrodes():
    # Duplicate names are replaced with their integer index:
    with pytest.warns(UserWarning, match='Duplicate electrode names'):
        stim = Stimulus(np.ones((3, 2)), electrodes=['AA', 'AA', 'BB'])
    npt.assert_equal(stim.electrodes, ['AA', '1', 'BB'])
    # Integer names stay integers:
    with pytest.warns(UserWarning, match='Duplicate electrode names'):
        stim = Stimulus([Stimulus(1), Stimulus(2)])
    npt.assert_equal(stim.electrodes, [0, 1])
    # The integer replacements must not be truncated by a string dtype that is
    # too narrow to hold them - the "fixed" names would be duplicates again:
    n_el = 200
    with pytest.warns(UserWarning, match='Duplicate electrode names'):
        stim = Stimulus(np.ones((n_el, 2)), electrodes=['A'] * n_el)
    npt.assert_equal(len(np.unique(stim.electrodes)), n_el)
    npt.assert_equal(stim.electrodes[0], 'A')
    npt.assert_equal(stim.electrodes[-1], str(n_el - 1))


def test_Stimulus_shift_without_time():
    # Shifting a stimulus that has no time component must be reported as such,
    # not as a TypeError from adding a scalar to None:
    stim = Stimulus(3)
    npt.assert_equal(stim.time, None)
    for shift in (lambda: stim.shift(1.0), lambda: stim.pad(1.0),
                  lambda: stim >> 1.0, lambda: stim << 1.0):
        with pytest.raises(ValueError):
            shift()
    # Stimuli that do have a time axis are unaffected:
    stim = Stimulus([[0, 1, 2]], time=[0, 1, 2])
    npt.assert_almost_equal((stim >> 1.5).time, [1.5, 2.5, 3.5])
    npt.assert_almost_equal((stim << 1.5).time, [-1.5, -0.5, 0.5])


def test_Stimulus_shift():
    stim = Stimulus([[0, 1, 2], [3, 4, 5]], time=[0, 1, 2],
                    electrodes=['A1', 'B2'], metadata='meta')
    # Forwards, nowhere, and backwards: a stimulus may live at negative times,
    # so a negative shift is not an error:
    npt.assert_almost_equal(stim.shift(10).time, [10, 11, 12])
    npt.assert_almost_equal(stim.shift(0).time, [0, 1, 2])
    npt.assert_almost_equal(stim.shift(-10).time, [-10, -9, -8])
    # The operators are aliases for `shift`:
    for dt in (-2.5, 0, 2.5):
        npt.assert_almost_equal((stim >> dt).time, stim.shift(dt).time)
        npt.assert_almost_equal((stim << dt).time, stim.shift(-dt).time)
    # A unitful shift is converted into the stimulus' own time unit, in either
    # direction; a quantity that is not a time is refused:
    npt.assert_almost_equal(stim.shift(0.02 * sec).time, [20, 21, 22])
    npt.assert_almost_equal(stim.shift(-1000 * us).time, [-1, 0, 1])
    with pytest.raises(DimensionMismatchError):
        stim.shift(5 * uA)
    # Everything but the time axis survives, and the original stays put:
    shifted = stim.shift(10)
    npt.assert_almost_equal(shifted.data, stim.data)
    npt.assert_equal(shifted.electrodes, stim.electrodes)
    npt.assert_equal(shifted.unit, stim.unit)
    npt.assert_equal(shifted.time_unit, stim.time_unit)
    npt.assert_equal(shifted.metadata, stim.metadata)
    npt.assert_almost_equal(stim.time, [0, 1, 2])
    # A pulse is defined by its parameters, so shifting one hands back a
    # plain stimulus rather than a pulse whose `stim_dur` contradicts the
    # time axis it now has (see test_pulses.py):
    pulse = BiphasicPulse(-20, 1)
    shifted = pulse.shift(5)
    npt.assert_equal(type(shifted), Stimulus)
    npt.assert_almost_equal(shifted.time, pulse.time + 5)
    npt.assert_almost_equal(shifted.data, pulse.data)


def test_Stimulus_pad():
    # `duration` is the time the padded stimulus ends at, not an amount of
    # time to add: a pulse shifted into the middle of a 10 s window keeps its
    # shifted times and gains zero-valued endpoints at t=0 and t=10000:
    pulse = BiphasicPulse(-20, 1)
    padded = pulse.shift(3000).pad(10000)
    npt.assert_almost_equal(padded.time[0], 0)
    npt.assert_almost_equal(padded.time[-1], 10000)
    npt.assert_almost_equal(padded.data[:, 0], 0)
    npt.assert_almost_equal(padded.data[:, -1], 0)
    npt.assert_almost_equal(padded.time[1:-1], pulse.time + 3000)
    npt.assert_almost_equal(padded.data[:, 1:-1], pulse.data)
    npt.assert_equal(type(padded), Stimulus)

    # A stimulus that already starts at t=0 only gets trailing padding:
    stim = Stimulus([[1, 0]], time=[0, 2])
    npt.assert_almost_equal(stim.pad(4).time, [0, 2, 4])
    npt.assert_almost_equal(stim.pad(4).data, [[1, 0, 0]])

    # One that starts before t=0 keeps its beginning: padding must not crop or
    # rewrite negative-time data:
    neg = Stimulus([[1, 0]], time=[-3, 2])
    npt.assert_almost_equal(neg.pad(4).time, [-3, 2, 4])
    npt.assert_almost_equal(neg.pad(4).data, [[1, 0, 0]])

    # Padding to the duration the stimulus already has adds only the missing
    # leading zero, and does not duplicate the endpoint it already has:
    stim = Stimulus([[0, 2]], time=[1, 2])
    npt.assert_almost_equal(stim.pad(stim.duration).time, [0, 1, 2])
    npt.assert_almost_equal(stim.pad(stim.duration).data, [[0, 0, 2]])

    # `pad` never truncates:
    with pytest.raises(ValueError):
        stim.pad(1.5)

    # A unitful duration is converted into the stimulus' own time unit:
    stim = Stimulus([[0, 1, 0]], time=[1, 2, 3])
    npt.assert_almost_equal(stim.pad(0.01 * sec).time, [0, 1, 2, 3, 10])
    with pytest.raises(DimensionMismatchError):
        stim.pad(5 * uA)

    # An endpoint can only be added next to a data point that is already zero:
    # the stimulus is interpolated between its time points, so a zero next to
    # a nonzero endpoint would be a ramp rather than padding.
    with pytest.raises(ValueError):
        Stimulus([[1, 0]], time=[1, 2]).pad(4)
    with pytest.raises(ValueError):
        Stimulus([[0, 1]], time=[0, 2]).pad(4)
    # Every electrode has to be zero there, not just the first one:
    with pytest.raises(ValueError):
        Stimulus([[0, 0], [0, 1]], time=[0, 2]).pad(4)
    # ... and "zero" means exactly zero:
    with pytest.raises(ValueError):
        Stimulus([[0, 1e-9]], time=[0, 2]).pad(4)
    # Padding that isn't needed at that end doesn't care what the data is:
    npt.assert_almost_equal(Stimulus([[1, 0]], time=[0, 2]).pad(2).time, [0, 2])

    # Every electrode gets a zero, not just the first one:
    multi = Stimulus([[0, 1, 0], [0, 3, 0]], time=[1, 2, 3],
                     electrodes=['A1', 'B2'], metadata='meta')
    padded = multi.pad(5)
    npt.assert_almost_equal(padded.time, [0, 1, 2, 3, 5])
    npt.assert_almost_equal(padded.data,
                            [[0, 0, 1, 0, 0], [0, 0, 3, 0, 0]])
    # The original is untouched, and the copy keeps everything else:
    npt.assert_almost_equal(multi.time, [1, 2, 3])
    npt.assert_almost_equal(multi.data, [[0, 1, 0], [0, 3, 0]])
    npt.assert_equal(padded.electrodes, multi.electrodes)
    npt.assert_equal(padded.unit, multi.unit)
    npt.assert_equal(padded.time_unit, multi.time_unit)
    npt.assert_equal(padded.metadata, multi.metadata)

    # A pad that has nothing to add still returns an independent copy, rather
    # than a stimulus sharing the buffers of the original:
    noop = Stimulus([[0, 1, 0]], time=[0, 2, 4])
    padded = noop.pad(noop.duration)
    npt.assert_almost_equal(padded.time, noop.time)
    npt.assert_almost_equal(padded.data, noop.data)
    npt.assert_equal(np.shares_memory(padded.data, noop.data), False)
    npt.assert_equal(np.shares_memory(padded.time, noop.time), False)

    # Padding a compressed stimulus does not make it uncompressed:
    compressed = Stimulus([[0, 1, 2, 0]], time=[0, 1, 2, 3], compress=True)
    npt.assert_equal(compressed.is_compressed, True)
    npt.assert_equal(compressed.pad(9).is_compressed, True)
    npt.assert_equal(multi.pad(5).is_compressed, False)


def test_Stimulus_no_global_side_effects():
    # Importing the module must not change NumPy's print options for the rest
    # of the user's session (`Stimulus` used to call `np.set_printoptions` at
    # module level). This has to run in a subprocess, because by the time this
    # test executes the module has long been imported.
    code = ("import numpy as np;"
            "before = np.get_printoptions();"
            "import pulse2percept.stimuli.base;"
            "after = np.get_printoptions();"
            "print('UNCHANGED' if before == after "
            "else f'CHANGED: {before} -> {after}')")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, check=True)
    npt.assert_equal(out.stdout.strip().splitlines()[-1], 'UNCHANGED')
    # Long arrays are still abbreviated in the repr, which is what the global
    # print options used to (redundantly) take care of:
    stim = Stimulus(np.arange(1000).reshape((10, 100)))
    npt.assert_equal('...' in repr(stim), True)
    npt.assert_equal('\n'.join(repr(stim).split()).count('electrodes='), 1)


def test_merge_time_axes_merge_tolerance():
    # Test issue where not enough unique points were collected
    # Leading to interpolation to corrupt stimuli data.
    # See: https://github.com/pulse2percept/pulse2percept/issues/392
    a = BiphasicPulseTrain(20, 1, 0.45)
    b = BiphasicPulseTrain(30, 1, 0.45)

    stim = Stimulus({"A2": a, "A10": b})
    unique_points = np.unique(stim.data)

    # Assert no value goes close to 1/3 or -1/3, i.e. a corrupted data point
    npt.assert_equal(np.isclose(1/3, unique_points, atol=0.1).any(), False)
    npt.assert_equal(np.isclose(-1/3, unique_points, atol=0.1).any(), False)


def test_merge_time_axes_float32_resolution():
    # Time is stored as float32, whose resolution is coarser than the absolute
    # merge tolerance for t > ~10 ms. Two stimuli that sample the same instant
    # then hand us time points a few ulps apart, which used to survive the
    # merge as separate columns: they were closer together than DT (so the
    # merged stimulus was not strictly increasing anymore) and interpolating
    # one stimulus at the other's time point invented data values halfway up a
    # pulse edge.
    freqs = (10, 11, 12, 13, 20, 30, 41)
    trains = {f'A{f}': BiphasicPulseTrain(f, 10, 0.45, stim_dur=1000)
              for f in freqs}
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        stim = Stimulus(trains)
    # Every pair of time points is at least a time step apart:
    npt.assert_equal(np.diff(stim.time.astype(np.float64)) >= 0.95 * DT, True)
    # And no data value was invented that is not in one of the sources:
    src_amps = np.unique(np.concatenate([t.data.ravel()
                                         for t in trains.values()]))
    npt.assert_equal(np.isin(np.unique(stim.data), src_amps), True)


def test_merge_time_axes_keeps_distinct_points():
    # The magnitude-scaled tolerance must never merge time points that are a
    # genuine time step apart, however large `t` gets:
    for t0 in (0.0, 10.0, 100.0, 1000.0, 4000.0):
        t1 = np.float32([t0, t0 + DT, t0 + 2 * DT])
        t2 = np.float32([t0, t0 + 3 * DT, t0 + 4 * DT])
        merged = merge_time_axes([np.zeros((1, 3)), np.zeros((1, 3))],
                                 [t1, t2])[1][0]
        npt.assert_almost_equal(merged, np.union1d(t1, t2), decimal=6)


def test_Stimulus_shallow_copy():
    # `append` and the arithmetic operators return a copy that shares nothing
    # mutable with the original, even though the data container is no longer
    # deep-copied first.
    # A stimulus that is defined by its samples keeps its class; one defined
    # by pulse parameters does not (see `Stimulus._derived` and
    # test_pulse_trains.py):
    stim = Stimulus([[0, 1, 0]], time=[0, 1, 2], metadata={'x': 1})
    for derive in (lambda s: s * 2, lambda s: s + 1, lambda s: -s,
                   lambda s: s >> 1.0, lambda s: s.append(s >> 1.0)):
        copied = derive(stim)
        # Same class and extra attributes:
        npt.assert_equal(type(copied), type(stim))
        npt.assert_equal(copied.unit, stim.unit)
        npt.assert_equal(copied.is_compressed, stim.is_compressed)
        # Metadata is independent. Its contents need not be identical: both
        # `append` and the operators keep any waveform parameters in sync with
        # the data, dropping them where they no longer describe it (see
        # test_pulse_trains.py):
        npt.assert_equal(copied.metadata is stim.metadata, False)
        copied.metadata['mine'] = 'changed'
        npt.assert_equal('mine' in stim.metadata, False)
        # The data container is independent, too:
        npt.assert_equal(copied._stim is stim._stim, False)
        npt.assert_equal(np.shares_memory(copied.data, stim.data), False)

    # Subclass-specific attributes survive as well:
    img = ImageStimulus(np.ones((4, 5), dtype=np.float32))
    npt.assert_equal(type(img * 2), ImageStimulus)
    npt.assert_equal((img * 2).img_shape, img.img_shape)
    npt.assert_almost_equal((img * 2).data, 2 * img.data)


@pytest.mark.parametrize('n_el, n_t, n_q', [(1, 5, 3), (2, 12, 40), (40, 8, 5),
                                            (40, 8, 400), (64, 300, 64),
                                            (33, 4, 257)])
def test_interp_rows(n_el, n_t, n_q):
    # `_interp_rows` replaces a per-electrode np.interp loop, and switches
    # between a vectorized and a looped implementation depending on the shape.
    # Both must agree with np.interp, because temporal models resolve stimulus
    # edges on a fixed simulation grid.
    #
    # Interior points are allowed to differ by a rounding: a C compiler may
    # contract `slope * dx + y0` into a single fused multiply-add inside
    # np.interp (it does on arm64), where the NumPy expression rounds twice.
    # Points that need no arithmetic must match exactly on every platform.
    rng = np.random.default_rng(n_el * 1000 + n_t * 10 + n_q)
    xp = np.unique(np.sort(rng.random(n_t).astype(np.float32) * 100))
    fp = ((rng.random((n_el, xp.size)) - 0.5) * 200).astype(np.float32)
    for x in (rng.random(n_q).astype(np.float32) * 140 - 20,   # incl. outside
              np.resize(xp, n_q),                              # exactly on knots
              np.full(n_q, xp[0], dtype=np.float32),           # left end point
              np.full(n_q, xp[-1], dtype=np.float32)):         # right end point
        expected = np.array([np.interp(x, xp, row) for row in fp])
        expected = expected.reshape((-1, x.size))
        actual = _interp_rows(x, xp, fp)
        # Scale the tolerance by the size of the data, not of the result: the
        # rounding happens on the intermediate product, which stays the size
        # of the inputs even where the result is near zero (any interpolation
        # across a zero crossing, which biphasic pulses do all the time).
        npt.assert_allclose(actual, expected, rtol=1e-12,
                            atol=1e-10 * np.abs(fp).max())
        # End points and exact knots are assigned verbatim, never computed,
        # so those must agree exactly on every platform:
        verbatim = (x <= xp[0]) | (x >= xp[-1]) | np.isin(x, xp)
        npt.assert_array_equal(actual[:, verbatim], expected[:, verbatim])


def test_interp_rows_edge_cases():
    # A single time point: np.interp returns it everywhere
    fp = np.array([[3.0], [4.0]], dtype=np.float32)
    xp = np.array([2.5], dtype=np.float32)
    x = np.array([-1.0, 2.5, 99.0], dtype=np.float32)
    npt.assert_array_equal(_interp_rows(x, xp, fp),
                           np.array([[3, 3, 3], [4, 4, 4]], dtype=np.float64))
    # No electrodes at all:
    npt.assert_equal(_interp_rows(x, np.array([0., 1.], np.float32),
                                  np.zeros((0, 2), np.float32)).shape, (0, 3))
    # A non-monotonic time axis is allowed (it only warns), but np.interp's
    # bracket search is guess-based there, so we must defer to it verbatim:
    xp = np.array([0., 1., 1., 2., 2.], dtype=np.float32)
    fp = np.array([[1., 0., 1., 0., 2.]] * 40, dtype=np.float32)
    x = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0], dtype=np.float32)
    expected = np.array([np.interp(x, xp, row) for row in fp])
    npt.assert_array_equal(_interp_rows(x, xp, fp), expected)


def test_Stimulus_getitem_many_electrodes():
    # Interpolating a stimulus with many electrodes takes the vectorized path;
    # the result must match interpolating each electrode by itself, to within
    # the one float32 ULP that a fused multiply-add inside np.interp can cost
    # (see `test_interp_rows`):
    rng = np.random.default_rng(0)
    data = rng.random((200, 25)).astype(np.float32)
    stim = Stimulus(data)
    for t in (3.7, [3.7], np.linspace(-2, 26, 37), np.asarray(stim.time)):
        # __getitem__ casts the requested time points to float32 first:
        t32 = np.float32(np.atleast_1d(t))
        expected = np.array([np.interp(t32, stim.time, row)
                             for row in data]).astype(np.float32)
        actual = np.asarray(stim[:, t]).reshape(expected.shape)
        npt.assert_almost_equal(actual, expected, decimal=6)
    # A single electrode of that stimulus must give the same values:
    npt.assert_almost_equal(stim[7, 3.7], stim[:, 3.7][7, 0])


def test_Stimulus_scalar_sequence():
    # A flat sequence of scalars takes a fast path in the constructor, which
    # must agree with the generic per-element path in every respect:
    for source in ([3, 5], (3, 5), [7], [3.5, -2.25, 0.0], [True, False],
                   [np.float32(3), np.float64(5), np.int32(7)]):
        stim = Stimulus(source)
        npt.assert_equal(stim.shape, (len(source), 1))
        npt.assert_equal(stim.time, None)
        npt.assert_equal(stim.electrodes, np.arange(len(source)))
        npt.assert_equal(stim.data.dtype, np.float32)
        npt.assert_almost_equal(stim.data.ravel(),
                                np.asarray(source, dtype=np.float32))
    # Electrode names, metadata and compression still work:
    stim = Stimulus([3, 5], electrodes=['A1', 'B2'], metadata={'x': 1})
    npt.assert_equal(stim.electrodes, ['A1', 'B2'])
    npt.assert_equal(stim.metadata['user'], {'x': 1})
    npt.assert_equal(Stimulus([0, 3, 0, 5], compress=True).shape, (2, 1))

    # An empty sequence still yields a 1-D (empty) data container:
    for source in ([], ()):
        npt.assert_equal(Stimulus(source).shape, (0,))
        npt.assert_equal(Stimulus(source).time, None)

    # Sequences that are not flat scalars must keep their old meaning: a
    # nested sequence is a single electrode in time, not several electrodes
    npt.assert_equal(Stimulus([[1, 5, 7, 2, 4]]).shape, (1, 5))
    npt.assert_equal(Stimulus([[1, 1], [1, 1]]).shape, (2, 2))
    npt.assert_equal(Stimulus(((1, 1), (1, 1))).shape, (2, 2))
    # ...and invalid elements must still raise, not be coerced. `None` in
    # particular converts to NaN if handed straight to np.asarray:
    for source in (['a', 'b'], [1, 'a'], [1, None], [1, [2, 3]]):
        with pytest.raises((TypeError, ValueError)):
            Stimulus(source)


def test_Stimulus_near_identical_time_axes():
    # Merging is skipped when all time axes are *close*, not just equal - the
    # fast path added for the common case must not tighten that tolerance.
    t1 = np.array([0., 1., 2.], dtype=np.float32)
    t2 = np.array([0., 1. + 3e-7, 2.], dtype=np.float32)
    npt.assert_equal(np.array_equal(t1, t2), False)   # differ in float32...
    npt.assert_equal(np.allclose(t1, t2), True)       # ...but within tolerance
    stim1 = Stimulus([[0., 5., 0.]], time=t1)
    stim2 = Stimulus([[0., 7., 0.]], time=t2)
    merged = Stimulus([stim1, stim2])
    # No interpolation: the first time axis is adopted verbatim
    npt.assert_array_equal(np.asarray(merged.time), t1)
    npt.assert_array_equal(merged.data, np.vstack((stim1.data, stim2.data)))
    # Genuinely different axes are still merged by interpolation:
    stim3 = Stimulus([[0., 9., 0.]], time=[0., 1.5, 2.])
    merged = Stimulus([stim1, stim3])
    npt.assert_equal(len(merged.time) > 3, True)


def test_Stimulus___eq___tolerance():
    # __eq__ compares with a tolerance; the exact-equality fast path must not
    # change that:
    npt.assert_equal(Stimulus([[1.0, 2.0]]) == Stimulus([[1.0, 2.0]]), True)
    npt.assert_equal(Stimulus([[1.0, 2.0]]) == Stimulus([[1.0, 2.0 + 1e-9]]),
                     True)
    npt.assert_equal(Stimulus([[1.0, 2.0]]) == Stimulus([[1.0, 2.5]]), False)
    # NaN never compares equal, with or without the fast path:
    npt.assert_equal(Stimulus([[np.nan, 1.0]]) == Stimulus([[np.nan, 1.0]]),
                     False)
    npt.assert_equal(Stimulus([[np.inf, 1.0]]) == Stimulus([[np.inf, 1.0]]),
                     True)


def test_Stimulus_rename_electrodes_metadata():
    # Per-electrode metadata is keyed by electrode name (BiphasicAxonMapModel
    # reads its stimulus parameters from there), so renaming the electrodes
    # has to rename those keys as well.
    stim = Stimulus({'A1': BiphasicPulseTrain(20, 30, 0.45, stim_dur=100),
                     'B3': BiphasicPulseTrain(40, 20, 0.45, stim_dur=100)})
    npt.assert_equal(sorted(stim.metadata['electrodes'].keys()), ['A1', 'B3'])

    renamed = Stimulus(stim, electrodes=['Z9', 'Y8'])
    npt.assert_equal(sorted(renamed.metadata['electrodes'].keys()), ['Y8', 'Z9'])
    for old, new in [('A1', 'Z9'), ('B3', 'Y8')]:
        npt.assert_equal(renamed.metadata['electrodes'][new],
                         stim.metadata['electrodes'][old])
    # The source must not be touched (its metadata may be shared):
    npt.assert_equal(sorted(stim.metadata['electrodes'].keys()), ['A1', 'B3'])

    # Swapping two names is a simultaneous remap, not two sequential ones:
    swapped = Stimulus(stim, electrodes=['B3', 'A1'])
    npt.assert_equal(swapped.metadata['electrodes']['B3'],
                     stim.metadata['electrodes']['A1'])
    npt.assert_equal(swapped.metadata['electrodes']['A1'],
                     stim.metadata['electrodes']['B3'])

    # Renaming a stimulus that has no per-electrode metadata is a no-op:
    plain = Stimulus(np.ones((2, 3)))
    npt.assert_equal(Stimulus(plain, electrodes=['P1', 'P2']).electrodes,
                     ['P1', 'P2'])
    npt.assert_equal(Stimulus(plain, electrodes=['P1', 'P2'])
                     .metadata['electrodes'], {})


def test_Stimulus_data_is_contiguous():
    """The data container must stay C-contiguous.

    Every Cython kernel in the library declares its stimulus argument as
    ``float32[:, ::1]``. Selecting columns, as ``compress`` does, hands back
    an F-ordered array for a multi-electrode stimulus, which used to surface
    much later as a "ndarray is not C-contiguous" from whichever kernel
    received it.
    """
    rng = np.random.default_rng(0)
    data = (rng.random((3, 5)) - 0.5).astype(np.float32)
    stim = Stimulus(data, time=np.arange(5, dtype=float) * 2)
    npt.assert_equal(stim.data.flags['C_CONTIGUOUS'], True)

    stim.compress()
    npt.assert_equal(stim.data.flags['C_CONTIGUOUS'], True)
    # ...so compressing an already-compressed stimulus works:
    stim.compress()
    npt.assert_equal(stim.data.flags['C_CONTIGUOUS'], True)

    # An F-ordered source is accepted and stored C-contiguous:
    stim = Stimulus(np.asfortranarray(data), time=np.arange(5, dtype=float))
    npt.assert_equal(stim.data.flags['C_CONTIGUOUS'], True)
    npt.assert_almost_equal(stim.data, data)


def test_Stimulus_units():
    # An electrical stimulus is stored in uA and ms, whatever it was given in:
    stim = Stimulus([500, 1000] * uA)
    npt.assert_equal(stim.unit, uA)
    npt.assert_equal(stim.time_unit, ms)
    npt.assert_almost_equal(stim.data.ravel(), [500, 1000])
    npt.assert_equal(stim.data.dtype, np.float32)
    # Every source form accepts quantities, and they all agree:
    npt.assert_almost_equal(Stimulus(5 * uA).data.ravel(), [5])
    for source in ([5, 10] * uA, np.array([5, 10]) * uA, [5 * uA, 10 * uA],
                   [0.005 * mA, 0.01 * mA], np.array([5, 10])):
        npt.assert_almost_equal(Stimulus(source).data.ravel(), [5, 10])
    npt.assert_almost_equal(
        Stimulus({'A1': 5 * uA, 'A2': 0.01 * mA}).data.ravel(), [5, 10])
    npt.assert_equal(list(Stimulus({'A1': 5 * uA}).electrodes), ['A1'])
    # Nested quantities are one electrode over time, as nested lists are:
    stim = Stimulus([[1, 2, 3] * uA, [4, 5, 6] * uA])
    npt.assert_equal(stim.shape, (2, 3))
    npt.assert_almost_equal(stim.data, [[1, 2, 3], [4, 5, 6]])
    # A time axis can be given as a quantity too:
    for time in ([0, 20] * ms, (0 * ms, 0.02 * sec), [0, 20]):
        npt.assert_almost_equal(
            Stimulus(np.ones((2, 2)), time=time).time, [0, 20])
    # Dimensional errors are caught at the boundary:
    with pytest.raises(DimensionMismatchError):
        Stimulus([5, 10] * ms)
    with pytest.raises(DimensionMismatchError):
        Stimulus({'A1': 5 * uA, 'A2': 3 * ms})
    with pytest.raises(DimensionMismatchError):
        Stimulus(np.ones((2, 2)), time=[0, 20] * uA)


def test_Stimulus_unit_views():
    stim = Stimulus([[500, 1000]] * uA, time=[0, 20] * ms)
    # The stored containers are untouched, plain numbers on a Cython-ready
    # array:
    npt.assert_equal(isinstance(stim.data, np.ndarray), True)
    npt.assert_equal(stim.data.dtype, np.float32)
    npt.assert_equal(isinstance(stim.time, np.ndarray), True)
    # The unitful views:
    npt.assert_equal(isinstance(stim.quantity, Quantity), True)
    npt.assert_equal(stim.quantity.unit, uA)
    npt.assert_almost_equal(stim.quantity.magnitude, stim.data)
    npt.assert_equal(stim.time_quantity.unit, ms)
    npt.assert_almost_equal(stim.time_quantity.magnitude, [0, 20])
    npt.assert_equal(stim.quantity == [[0.5, 1.0]] * mA, [[True, True]])
    # The converted numeric views are ordinary arrays, never quantities:
    for values in (stim.values(), stim.values(uA), stim.values(mA),
                   stim.times(), stim.times(sec)):
        npt.assert_equal(isinstance(values, np.ndarray), True)
        npt.assert_equal(values.dtype != object, True)
    npt.assert_almost_equal(stim.values(mA).ravel(), [0.5, 1.0])
    npt.assert_almost_equal(stim.values(uA).ravel(), [500, 1000])
    npt.assert_almost_equal(stim.times(sec), [0, 0.02])
    # Asking for the stored unit hands back the stored array itself, and the
    # data stays float32 through a conversion:
    npt.assert_equal(stim.values() is stim.data, True)
    npt.assert_equal(stim.values(mA).dtype, np.float32)
    # A stimulus with no time component has no time views:
    npt.assert_equal(Stimulus([1, 2]).time_quantity, None)
    npt.assert_equal(Stimulus([1, 2]).times(sec), None)
    # And a unit of the wrong dimension is refused:
    with pytest.raises(DimensionMismatchError):
        stim.values(ms)
    with pytest.raises(DimensionMismatchError):
        stim.times(uA)


def test_Stimulus_units_are_read_only():
    # The canonical storage unit is a contract, not a setting: models, safety
    # checks and Cython kernels all rely on uA/ms.
    stim = Stimulus([1, 2])
    with pytest.raises(AttributeError):
        stim.unit = mA
    with pytest.raises(AttributeError):
        stim.time_unit = sec


def test_Stimulus_dimensionless():
    # Image and video pixels are gray levels, not currents:
    img = ImageStimulus(np.ones((3, 3)))
    npt.assert_equal(img.unit, dimensionless)
    npt.assert_equal(img.unit != uA, True)
    # A copy of one is still made of gray levels:
    npt.assert_equal(Stimulus(img).unit, dimensionless)
    npt.assert_equal(deepcopy(img).unit, dimensionless)
    # But a bare Stimulus keeps its historical electrical reading:
    npt.assert_equal(Stimulus(np.ones((3, 3))).unit, uA)
    # Gray levels cannot be converted to a current without an encoder:
    with pytest.raises(DimensionMismatchError):
        img.values(uA)
    # ... and the encoder output is electrical:
    npt.assert_equal(img.encode().unit, uA)
    npt.assert_equal(img.encode().time_unit, ms)
    # Two stimuli holding the same numbers in different units are not equal:
    npt.assert_equal(Stimulus(np.ones((3, 3)).ravel()) == img, False)


def test_Stimulus_units_preserved():
    """Every operation that returns a stimulus must carry its units along"""
    elec = Stimulus(np.ones((2, 3)), time=[0, 1, 2])
    img = ImageStimulus(np.ones((3, 3)))
    for stim, unit in [(elec, uA), (img, dimensionless)]:
        produced = {
            'deepcopy': deepcopy(stim),
            'copy': stim._shallow_copy(),
            'from stimulus': Stimulus(stim),
            'multiply': stim * 2,
            'divide': stim / 2,
            'negate': -stim,
            'add': stim + 1,
            'subtract': stim - 1,
            'rsub': 1 - stim,
        }
        if stim.time is not None:
            produced['append'] = stim.append(stim >> 5)
            produced['shift'] = stim >> 20
            produced['shift back'] = stim << 1
        for name, result in produced.items():
            npt.assert_equal((name, result.unit), (name, unit))
            npt.assert_equal((name, result.time_unit), (name, ms))
        # compress and remove rewrite the data container in place:
        for name, method, args in [('compress', 'compress', ()),
                                   ('remove', 'remove', (0,))]:
            mutated = deepcopy(stim)
            getattr(mutated, method)(*args)
            npt.assert_equal((name, mutated.unit), (name, unit))
            npt.assert_equal((name, mutated.time_unit), (name, ms))
    # A collection inherits the unit its members agree on:
    npt.assert_equal(Stimulus([BiphasicPulseTrain(20, 10, 0.45)]).unit, uA)
    npt.assert_equal(
        Stimulus({'A1': ImageStimulus(np.ones((1, 1)))}).unit, dimensionless)
    # ... and refuses to guess when they disagree:
    with pytest.raises(DimensionMismatchError):
        Stimulus([ImageStimulus(np.ones((1, 1))),
                  BiphasicPulseTrain(20, 10, 0.45)])


def test_Stimulus_arithmetic_units():
    stim = Stimulus(np.ones((2, 3)) * 100, time=[0, 1, 2])
    # Adding an amplitude: a bare number means the stimulus own unit, and a
    # quantity is converted into it.
    npt.assert_almost_equal((stim + 500).data, (stim + 0.5 * mA).data)
    npt.assert_almost_equal((stim - 500).data, (stim - 0.5 * mA).data)
    npt.assert_almost_equal((500 - stim).data, (0.5 * mA - stim).data)
    npt.assert_almost_equal((stim + 0.5 * mA).data, 600 * np.ones((2, 3)))
    # Scaling: a plain number, or an explicitly dimensionless quantity.
    npt.assert_almost_equal((stim * 2).data, (stim * (2 * dimensionless)).data)
    npt.assert_almost_equal((stim / 2).data, 50 * np.ones((2, 3)))
    # Shifting in time:
    npt.assert_almost_equal((stim >> 20).time, (stim >> 0.02 * sec).time)
    npt.assert_almost_equal((stim << 1).time, (stim << 1000 * us).time)
    # A stimulus stays a stimulus: multiplying by a unit would make it a
    # charge, and that is not something this class represents.
    with pytest.raises(DimensionMismatchError):
        stim * ms
    with pytest.raises(DimensionMismatchError):
        stim * (2 * ms)
    with pytest.raises(DimensionMismatchError):
        stim / (2 * ms)
    # Nor can a time be added to a current, or a current to gray levels:
    with pytest.raises(DimensionMismatchError):
        stim + 5 * ms
    with pytest.raises(DimensionMismatchError):
        stim >> 5 * uA
    with pytest.raises(DimensionMismatchError):
        ImageStimulus(np.ones((2, 2))) + 5 * uA


def test_Stimulus_append_units():
    # `append` copies `self` and concatenates `other`'s data onto its own, so
    # without a check the result would label another stimulus' numbers with
    # this one's unit.
    elec = Stimulus(np.ones((1, 3)), time=[0, 1, 2])
    dimless = Stimulus(VideoStimulus(np.ones((1, 1, 3)), time=[0, 1, 2]))
    npt.assert_equal(elec.unit, uA)
    npt.assert_equal(dimless.unit, dimensionless)
    with pytest.raises(DimensionMismatchError):
        elec.append(dimless)
    with pytest.raises(DimensionMismatchError):
        dimless.append(elec)
    # Two stimuli that do agree still append, and keep the unit:
    combined = elec.append(elec >> 3)
    npt.assert_equal(combined.unit, uA)
    npt.assert_equal(combined.time_unit, ms)
    npt.assert_equal(dimless.append(dimless >> 3).unit, dimensionless)


def test_Stimulus_getitem_units():
    stim = Stimulus(np.arange(10, dtype=float).reshape((1, -1)),
                    time=np.arange(10, dtype=float))
    # A requested time point can be given in any unit of time:
    npt.assert_almost_equal(stim[:, 3.45], stim[:, 3.45 * ms])
    npt.assert_almost_equal(stim[:, 3.45], stim[:, 0.00345 * sec])
    # As can a list of them...
    npt.assert_almost_equal(stim[:, [1, 2]], stim[:, [1, 2] * ms])
    npt.assert_almost_equal(stim[:, [1, 2]], stim[:, [1 * ms, 2 * ms]])
    # ...and the endpoints and step of a slice:
    npt.assert_almost_equal(stim[:, 1:3:1],
                            stim[:, 0.001 * sec:0.003 * sec:1 * ms])
    npt.assert_almost_equal(stim[:, 1:3:1], stim[:, 1 * ms:3 * ms:1000 * us])
    # Interpolation still happens where it always did:
    npt.assert_almost_equal(stim[:, 3.45 * ms], 3.45, decimal=5)
    # The other indexing forms are untouched:
    npt.assert_almost_equal(stim[:, stim.time < 2].ravel(), [0, 1])
    npt.assert_equal(stim[:, ...].shape, (1, 10))
    # A current is not a point in time:
    with pytest.raises(DimensionMismatchError):
        stim[:, 3 * uA]
    with pytest.raises(DimensionMismatchError):
        stim[:, [1, 2] * uA]
    with pytest.raises(DimensionMismatchError):
        stim[:, 1 * uA:3 * uA:1 * uA]


def test_Stimulus_plot_units():
    stim = Stimulus(np.arange(10, dtype=float).reshape((1, -1)),
                    time=np.arange(10, dtype=float))
    # A range or a list of time points may be unitful:
    npt.assert_equal(isinstance(stim.plot(time=(1 * ms, 3 * ms)), Subplot),
                     True)
    npt.assert_equal(isinstance(stim.plot(time=[1, 2] * ms), Subplot), True)
    with pytest.raises(DimensionMismatchError):
        stim.plot(time=(1 * uA, 3 * uA))
    # The axes say what the stimulus is actually made of:
    ax = stim.plot()
    npt.assert_equal(ax.get_xlabel(), 'Time (ms)')
    npt.assert_equal(ax.figure.get_supylabel(), r'Amplitude ($\mu$A)')
    dimless = Stimulus(VideoStimulus(np.ones((1, 1, 3)), time=[0, 1, 2]))
    npt.assert_equal(dimless.plot().figure.get_supylabel(), 'Value')
    plt.close('all')
    # A heatmap says it on the colorbar instead:
    two = Stimulus(np.ones((2, 3)), time=[0, 1, 2])
    npt.assert_equal(two.plot().collections[0].colorbar.ax.get_ylabel(),
                     r'Amplitude ($\mu$A)')
    dimless = Stimulus(VideoStimulus(np.ones((1, 2, 3)), time=[0, 1, 2]))
    npt.assert_equal(dimless.plot().collections[0].colorbar.ax.get_ylabel(),
                     'Value')
    plt.close('all')


def test_Stimulus_time_slice():
    """A slice of the time axis means the same thing everywhere

    Slicing the time axis asks for a time *range*, which `__getitem__`
    interpolates onto. `plot` has to resolve it the same way, or the curve
    would be drawn against whatever time points happen to sit at those column
    indices.
    """
    # A ramp whose value equals its time, so a wrong x axis is visible:
    stim = Stimulus(np.arange(10, dtype=float).reshape((1, -1)),
                    time=np.arange(10, dtype=float))
    expected = np.arange(1, 4, 0.5)
    npt.assert_almost_equal(stim[:, 1:4:0.5].ravel(), expected)
    ax = stim.plot(time=slice(1, 4, 0.5))
    x, y = ax.lines[0].get_data()
    npt.assert_almost_equal(x, expected)
    npt.assert_almost_equal(y, expected)
    plt.close('all')
    # The endpoints and the step may be quantities, in both APIs:
    npt.assert_almost_equal(stim[:, 1 * ms:4 * ms:0.5 * ms].ravel(), expected)
    npt.assert_almost_equal(
        stim[:, 0.001 * sec:0.004 * sec:500 * us].ravel(), expected)
    ax = stim.plot(time=slice(1 * ms, 4 * ms, 0.5 * ms))
    npt.assert_almost_equal(ax.lines[0].get_data()[0], expected)
    plt.close('all')
    with pytest.raises(DimensionMismatchError):
        stim.plot(time=slice(1 * uA, 4 * uA, 1 * uA))
    # A slice without a step is the stored samples themselves, taken by
    # position -- the one reading that needs no interpolation:
    ax = stim.plot(time=slice(None))
    npt.assert_almost_equal(ax.lines[0].get_data()[0], stim.time)
    npt.assert_almost_equal(stim[:, :].ravel(), stim.data.ravel())
    plt.close('all')
    # And a partial slice with no step is refused identically by both:
    for call in (lambda: stim[:, 1:4], lambda: stim.plot(time=slice(1, 4))):
        with pytest.raises(ValueError):
            call()


def test_Stimulus_is_charge_balanced_needs_a_current():
    """Gray levels integrate to a number, but that number is not a charge"""
    # Not applicable, which is not the same as unbalanced:
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    vid = VideoStimulus(np.ones((2, 2, 3)) * 0.5, time=[0, 20, 40])
    for stim in (img, vid, Stimulus(img), Stimulus(vid)):
        npt.assert_equal(stim.is_charge_balanced, None)
    # A dimensionless stimulus whose values happen to sum to zero is still not
    # "balanced" -- there is nothing there to balance:
    zeros = Stimulus(VideoStimulus(np.zeros((1, 1, 3)), time=[0, 1, 2]))
    npt.assert_equal(zeros.unit, dimensionless)
    npt.assert_equal(zeros.is_charge_balanced, None)
    # Electrical stimuli answer exactly as they always have:
    npt.assert_equal(BiphasicPulse(50, 0.45).is_charge_balanced, True)
    npt.assert_equal(MonophasicPulse(50, 0.45).is_charge_balanced, False)
    npt.assert_equal(BiphasicPulseTrain(20, 50, 0.45).is_charge_balanced, True)
    npt.assert_equal(Stimulus([0]).is_charge_balanced, True)
    npt.assert_equal(Stimulus([1]).is_charge_balanced, False)
    npt.assert_equal((50 * uA * 0 + Stimulus([0])).is_charge_balanced, True)
    # Pretty-printing evaluates the property, so it must not raise on a
    # picture:
    for stim in (img, vid):
        npt.assert_equal('is_charge_balanced' in str(stim), True)


@pytest.mark.parametrize('build', [
    lambda: Stimulus(np.arange(6, dtype=np.float32).reshape((2, 3))),
    lambda: Stimulus({'A1': [0, 1, 0], 'B2': [0, 2, 0]}),
    lambda: BiphasicPulseTrain(20, 50, 0.45, stim_dur=100),
    lambda: ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4))),
    lambda: VideoStimulus(np.ones((2, 2, 3)) * 0.5, time=[0, 20, 40]),
])
def test_Stimulus_is_immutable(build):
    stim = build()
    with pytest.raises(ValueError):
        stim.data[0, 0] = 1
    if stim.time is not None:
        with pytest.raises(ValueError):
            stim.time[0] = 1
    # `ImageStimulus` names its pixels with an `ElectrodeNames`, which
    # generates them from a grid instead of storing them and so has no way to
    # set one at all:
    with pytest.raises((ValueError, TypeError)):
        stim.electrodes[0] = 'X'
    # Metadata stays writable: it is the user's, and describes the stimulus
    # rather than being it.
    stim.metadata['user'] = 'mine'
    npt.assert_equal(stim.metadata['user'], 'mine')


def test_Stimulus_owns_its_arrays():
    # Building a stimulus must neither take the caller's array away from them
    # nor leave them a handle on the stimulus:
    arr = np.ones((2, 3), dtype=np.float32)
    stim = Stimulus(arr, time=[0, 1, 2])
    npt.assert_equal(arr.flags.writeable, True)
    npt.assert_equal(np.shares_memory(arr, stim.data), False)
    arr[0, 0] = 99
    npt.assert_almost_equal(stim.data, np.ones((2, 3)))
    # Nor may one stimulus write through to another's buffers:
    copied = Stimulus(stim)
    npt.assert_equal(np.shares_memory(stim.data, copied.data), False)
    npt.assert_equal(np.shares_memory(stim.time, copied.time), False)

    # A contiguous view of a larger buffer needs no dtype or layout
    # conversion, and an ndarray subclass is handed back as a *different*
    # ndarray over the very same memory. Neither is a private array, so
    # neither may be stored as it came:
    class Tagged(np.ndarray):
        """A minimal ndarray subclass; only its type matters here"""

    big = np.arange(12, dtype=np.float32).reshape((4, 3))
    for source in (big[:2], big[:2].view(Tagged)):
        stim = Stimulus(source, time=[0, 1, 2])
        npt.assert_equal(np.shares_memory(big, stim.data), False)
        npt.assert_equal(big.flags.writeable, True)
        # ...and what comes out is an ordinary array, whatever went in:
        npt.assert_equal(type(stim.data), np.ndarray)


def test_Stimulus_deepcopy():
    # Duplicating arrays nobody can write into buys nothing, and NumPy would
    # deep-copy a read-only array into a writable one -- so the copy shares
    # the data container and only `metadata` is made independent:
    stim = BiphasicPulseTrain(20, 20, 0.45, stim_dur=100)
    # Materialize first: a train that has not generated its waveform yet has
    # no container to share, and the copy generates its own (see
    # test_pulse_trains.py):
    stim.data
    copied = deepcopy(stim)
    npt.assert_equal(copied is stim, False)
    npt.assert_equal(np.shares_memory(copied.data, stim.data), True)
    npt.assert_equal(copied.metadata is stim.metadata, False)
    copied.metadata['user'] = 'changed'
    npt.assert_equal(stim.metadata['user'], None)
    npt.assert_equal(copied.freq, stim.freq)
    npt.assert_equal(type(copied), type(stim))
    # Metadata is arbitrary user data and may point back at the stimulus it
    # describes. The copy has to resolve that as one object graph:
    stim.metadata['self'] = stim
    copied = deepcopy(stim)
    npt.assert_equal(copied.metadata['self'] is copied, True)


def test_Stimulus_immutable_operations():
    # Everything that returns or rebuilds a stimulus still works, and hands
    # back one that is immutable in its turn:
    stim = Stimulus({'A1': [0, 1, 1, 0], 'B2': [0, 0, 0, 0]},
                    time=[0, 1, 2, 3])
    derived = [stim * 2, -stim, stim + 1, stim >> 1.0, stim / 2,
               stim.append(stim >> 1.0), stim.pad(9)]
    compressed = Stimulus(stim)
    compressed.compress()
    removed = Stimulus(stim)
    removed.remove('A1')
    derived += [compressed, removed]
    for out in derived:
        npt.assert_equal(out.data.flags.writeable, False)
        npt.assert_equal(out.time.flags.writeable, False)
        npt.assert_equal(out.data.flags['C_CONTIGUOUS'], True)
    # The operations themselves are unaffected:
    npt.assert_equal(compressed.shape, (1, 3))
    npt.assert_equal(list(removed.electrodes), ['B2'])


class CountingLazy(Stimulus):
    """A stimulus that counts how often it generates its waveform

    Stands in for the parameter-backed stimuli that follow: it is defined by
    ``n_time``, not by samples, and only ``_render`` ever builds any.
    """
    __slots__ = ('n_time', 'n_renders')

    def __init__(self, n_time=4, electrodes=('A1', 'B2'), metadata=None):
        self.n_time = n_time
        self.n_renders = 0
        self._defer(electrodes, metadata=metadata)

    def _render(self):
        self.n_renders += 1
        n_el = len(self.electrodes)
        data = np.arange(n_el * self.n_time,
                         dtype=np.float32).reshape((n_el, self.n_time))
        return {'data': data, 'electrodes': self.electrodes,
                'time': np.arange(self.n_time, dtype=np.float64)}


def test_Stimulus_lazy_construction_does_not_render():
    stim = CountingLazy(metadata={'x': 1})
    npt.assert_equal(stim.n_renders, 0)
    # Everything a stimulus knows without sampling anything:
    npt.assert_equal(list(stim.electrodes), ['A1', 'B2'])
    npt.assert_equal(len(stim.electrodes), 2)
    npt.assert_equal(stim.unit, uA)
    npt.assert_equal(stim.time_unit, ms)
    npt.assert_equal(stim.metadata['user'], {'x': 1})
    npt.assert_equal(stim.is_compressed, False)
    npt.assert_equal(stim.dt, DT)
    npt.assert_equal(stim.n_renders, 0)


def test_Stimulus_lazy_renders_once():
    stim = CountingLazy()
    npt.assert_almost_equal(stim.data, [[0, 1, 2, 3], [4, 5, 6, 7]])
    npt.assert_equal(stim.n_renders, 1)
    # The cache serves every later read, of either array:
    for _ in range(3):
        npt.assert_equal(stim.data.shape, (2, 4))
        npt.assert_almost_equal(stim.time, [0, 1, 2, 3])
        npt.assert_almost_equal(stim[0, 1.5], 1.5)
        npt.assert_equal(stim.duration, 3)
    npt.assert_equal(stim.n_renders, 1)
    # Asking for `time` first renders just the same:
    other = CountingLazy()
    npt.assert_almost_equal(other.time, [0, 1, 2, 3])
    npt.assert_equal(other.n_renders, 1)
    npt.assert_almost_equal(other.data, stim.data)
    npt.assert_equal(other.n_renders, 1)


def test_Stimulus_lazy_state_is_immutable():
    # A rendered waveform is installed through the same setter as any other,
    # so it is owned, immutable and C-contiguous on the same terms:
    stim = CountingLazy()
    npt.assert_equal(stim.data.flags.writeable, False)
    npt.assert_equal(stim.time.flags.writeable, False)
    npt.assert_equal(stim.data.flags['C_CONTIGUOUS'], True)
    npt.assert_equal(stim.data.dtype, np.float32)
    npt.assert_equal(stim.time.dtype, np.float64)
    with pytest.raises(ValueError):
        stim.electrodes[0] = 'X'
    # And it is validated on the same terms, too:

    class BadShape(CountingLazy):
        __slots__ = ()

        def _render(self):
            return {'data': np.ones((5, 2), dtype=np.float32),
                    'electrodes': self.electrodes,
                    'time': np.arange(2, dtype=np.float64)}
    with pytest.raises(ValueError):
        BadShape().data


def test_Stimulus_lazy_electrodes_must_match_render():
    # Naming the electrodes up front is what lets them be read without
    # generating a waveform, so a render that disagrees with them would make
    # the answer depend on when it was asked for:
    class Renamer(CountingLazy):
        __slots__ = ()

        def _render(self):
            state = super()._render()
            return dict(state, electrodes=np.array(['C3', 'D4']))
    stim = Renamer()
    npt.assert_equal(list(stim.electrodes), ['A1', 'B2'])
    with pytest.raises(ValueError):
        stim.data


def test_Stimulus_lazy_copy_does_not_render():
    stim = CountingLazy()
    for copied in (copy(stim), deepcopy(stim)):
        npt.assert_equal(copied.n_renders, 0)
        npt.assert_equal(stim.n_renders, 0)
        npt.assert_equal(list(copied.electrodes), ['A1', 'B2'])
        npt.assert_equal(copied.n_renders, 0)
    # A copy taken after materialization shares the cached waveform, which is
    # immutable and so has nothing to gain from being duplicated:
    npt.assert_equal(stim.data.shape, (2, 4))
    shared = deepcopy(stim)
    npt.assert_equal(np.shares_memory(shared.data, stim.data), True)
    npt.assert_equal(stim.n_renders, 1)


def test_Stimulus_render_is_not_implemented_by_default():
    # A plain `Stimulus` is its waveform and never renders, so the base
    # implementation exists only to name what a subclass forgot:
    with pytest.raises(NotImplementedError):
        Stimulus._render(Stimulus(3))


def _n_renders(stim):
    """How often the `CountingLazy` entries of a collection have rendered"""
    return sum(getattr(c, 'n_renders', 0) for c, _ in stim._components)


def _lazy(stim):
    return stim._Stimulus__stim['data'] is None


# Collections that stay unmerged, each as (name, source). The expensive one
# is the last: differing frequencies means every entry lands on its own time
# axis, and merging them interpolates all of them onto the union.
COLLECTIONS = [
    ('same protocol', lambda: {'A1': CountingLazy(4, ['A1']),
                               'B2': CountingLazy(4, ['B2'])}),
    ('differing lengths', lambda: {'A1': CountingLazy(4, ['A1']),
                                   'B2': CountingLazy(7, ['B2']),
                                   'C3': CountingLazy(11, ['C3'])}),
    ('mixed representation', lambda: {'A1': CountingLazy(4, ['A1']),
                                      'B2': Stimulus([[1, 2, 3, 4]])}),
    ('list', lambda: [CountingLazy(4, ['A1']), CountingLazy(5, ['B2'])]),
    ('multi-electrode entry', lambda: [CountingLazy(4, ['x', 'y']),
                                       CountingLazy(6, ['B2'])]),
]


@pytest.mark.parametrize('name, build', COLLECTIONS,
                         ids=[c[0] for c in COLLECTIONS])
def test_Stimulus_collection_defers_the_merge(name, build):
    stim = Stimulus(build())
    npt.assert_equal(_lazy(stim), True)
    # Everything a collection knows before its entries have been sampled:
    npt.assert_equal(_n_renders(stim), 0)
    npt.assert_equal(len(stim.electrodes) > 1, True)
    npt.assert_equal(stim.unit, uA)
    npt.assert_equal(stim.time_unit, ms)
    npt.assert_equal(sorted(stim.metadata['electrodes']) != [], True)
    npt.assert_equal(stim.is_compressed, False)
    copies = [copy(stim), deepcopy(stim)]
    npt.assert_equal(_n_renders(stim), 0)
    for copied in copies:
        npt.assert_equal(_lazy(copied), True)
    # ...and the first read of the waveform is what builds one, once:
    n_lazy = sum(isinstance(c, CountingLazy) for c, _ in stim._components)
    data = stim.data
    npt.assert_equal(_n_renders(stim), n_lazy)
    for _ in range(3):
        npt.assert_equal(np.shares_memory(stim.data, data), True)
        npt.assert_equal(stim.time.dtype, np.float64)
    npt.assert_equal(_n_renders(stim), n_lazy)


@pytest.mark.parametrize('name, build', COLLECTIONS,
                         ids=[c[0] for c in COLLECTIONS])
def test_Stimulus_collection_matches_the_eager_merge(name, build,
                                                     monkeypatch):
    lazy = Stimulus(build())
    monkeypatch.setattr(Stimulus, '_defers_waveform',
                        staticmethod(lambda *a, **kw: False))
    eager = Stimulus(build())
    npt.assert_equal(_lazy(eager), False)
    npt.assert_array_equal(lazy.data, eager.data)
    npt.assert_array_equal(lazy.time, eager.time)
    npt.assert_array_equal(lazy.electrodes, eager.electrodes)


@pytest.mark.parametrize('freqs', [(20, 20), (20, 23), (20, 23, 41)])
def test_Stimulus_pulse_train_collection_merges_as_it_always_did(freqs,
                                                                 monkeypatch):
    # Differing frequencies put every train on a time axis of its own, so
    # the merge is the expensive part.
    def build():
        return {f'E{i}': BiphasicPulseTrain(f, 10 + i, 0.45, stim_dur=200)
                for i, f in enumerate(freqs)}
    lazy = Stimulus(build())
    npt.assert_equal(_lazy(lazy), True)
    monkeypatch.setattr(Stimulus, '_defers_waveform',
                        staticmethod(lambda *a, **kw: False))
    eager = Stimulus(build())
    npt.assert_array_equal(lazy.data, eager.data)
    npt.assert_array_equal(lazy.time, eager.time)


@pytest.mark.parametrize('vary', ['freq', 'amp', 'phase_dur'])
def test_Stimulus_heterogeneous_pulse_parameters_merge_unchanged(vary,
                                                                 monkeypatch):
    kwargs = {'freq': 20, 'amp': 10, 'phase_dur': 0.45, 'stim_dur': 200}
    other = dict(kwargs, **{vary: {'freq': 23, 'amp': 25,
                                   'phase_dur': 0.9}[vary]})

    def build():
        return {'A1': BiphasicPulseTrain(**kwargs),
                'B2': BiphasicPulseTrain(**other)}
    lazy = Stimulus(build())
    monkeypatch.setattr(Stimulus, '_defers_waveform',
                        staticmethod(lambda *a, **kw: False))
    npt.assert_array_equal(lazy.data, Stimulus(build()).data)


def test_Stimulus_collection_with_a_silent_child(monkeypatch):
    # A 0 Hz train is a flat row, and it still has to end where the others do:
    def build():
        return {'A1': BiphasicPulseTrain(0, 10, 0.45, stim_dur=200),
                'B2': BiphasicPulseTrain(20, 10, 0.45, stim_dur=200)}
    lazy = Stimulus(build())
    npt.assert_equal(_lazy(lazy), True)
    npt.assert_almost_equal(lazy.time[-1], 200)
    npt.assert_almost_equal(np.abs(lazy.data[0]).max(), 0)
    monkeypatch.setattr(Stimulus, '_defers_waveform',
                        staticmethod(lambda *a, **kw: False))
    npt.assert_array_equal(lazy.data, Stimulus(build()).data)


def test_Stimulus_collection_of_raw_sources_stays_eager():
    # Nothing to save: these entries are already the numbers they describe.
    for source in ({'A1': [1, 2, 3], 'B2': [4, 5, 6]}, [[1, 2], [3, 4]],
                   {'A1': 1, 'B2': 2}):
        npt.assert_equal(Stimulus(source)._components, None)
    # An explicit time axis and `compress` both ask about the merged waveform:
    stim = Stimulus({'A1': CountingLazy(3, ['A1'])}, time=[0, 1, 2])
    npt.assert_equal(stim._components, None)
    npt.assert_equal(Stimulus({'A1': CountingLazy(3, ['A1'])},
                              compress=True)._components, None)


def test_Stimulus_collection_snapshots_its_entries():
    raw = [1.0, 2.0, 3.0, 4.0]
    child = CountingLazy(4, ['A1'])
    stim = Stimulus({'A1': child, 'B2': raw})
    raw[0] = 99.0
    npt.assert_equal(stim._components[0][0] is child, False)
    npt.assert_almost_equal(stim.data[1], [1, 2, 3, 4])
    # Rendering the caller's object is not what rendered the collection:
    npt.assert_equal(child.n_renders, 0)


def test_Stimulus_collection_validates_what_it_can_without_rendering():
    child = CountingLazy(4, ['A1'])
    with pytest.raises(ValueError):
        # A scalar has no time component, the other entry does:
        Stimulus({'A1': child, 'B2': 5})
    with pytest.raises(ValueError):
        Stimulus({'A1': child}, electrodes=['A1', 'B2'])
    with pytest.raises(ValueError):
        Stimulus({'A1': child, 'B2': CountingLazy(4, ['B2'])},
                 electrodes=['only-one'])
    assert_warns_msg(UserWarning,
                     lambda: Stimulus([CountingLazy(4, ['A1']),
                                       CountingLazy(4, ['A1'])]),
                     'Duplicate electrode names detected')
    npt.assert_equal(child.n_renders, 0)


def test_Stimulus_collection_renames_without_rendering():
    child = CountingLazy(4, ['A1'])
    renamed = Stimulus({'A1': child, 'B2': CountingLazy(4, ['B2'])},
                       electrodes=['X', 'Y'])
    npt.assert_equal(_lazy(renamed), True)
    npt.assert_equal(list(renamed.electrodes), ['X', 'Y'])
    npt.assert_equal(sorted(renamed.metadata['electrodes']), ['X', 'Y'])
    npt.assert_equal(child.n_renders, 0)
    # Re-wrapping a single deferred stimulus is a rename too:
    solo = Stimulus(CountingLazy(4, ['A1']), electrodes=['Z'])
    npt.assert_equal(_lazy(solo), True)
    npt.assert_equal(list(solo.electrodes), ['Z'])
    npt.assert_equal(solo.data.shape, (1, 4))


def test_Stimulus_collection_removes_whole_entries_without_rendering():
    stim = Stimulus({'A1': CountingLazy(4, ['A1']),
                     'B2': CountingLazy(7, ['B2']),
                     'C3': CountingLazy(11, ['C3'])})
    stim.remove(['A1', 'C3'])
    npt.assert_equal(_lazy(stim), True)
    npt.assert_equal(list(stim.electrodes), ['B2'])
    npt.assert_equal(_n_renders(stim), 0)
    # Only the entry that survived is ever generated:
    npt.assert_equal(stim.data.shape, (1, 7))
    npt.assert_equal(_n_renders(stim), 1)
    empty = Stimulus({'A1': CountingLazy(4, ['A1'])})
    empty.remove('all')
    npt.assert_equal(_lazy(empty), True)
    npt.assert_equal(len(empty.electrodes), 0)
    # A copy taken before the removal keeps the entry it was built with:
    stim = Stimulus({'A1': CountingLazy(4, ['A1']),
                     'B2': CountingLazy(4, ['B2'])})
    other = stim._derived()
    other.remove('A1')
    npt.assert_equal(list(stim.electrodes), ['A1', 'B2'])
    npt.assert_equal(list(other.electrodes), ['B2'])


def test_Stimulus_collection_removing_part_of_an_entry_materializes():
    # 'x' and 'y' come from one entry, so dropping 'x' means generating it.
    stim = Stimulus([CountingLazy(4, ['x', 'y']),
                     CountingLazy(4, ['B2'])])
    stim.remove('x')
    npt.assert_equal(_lazy(stim), False)
    npt.assert_equal(list(stim.electrodes), ['y', 'B2'])
    npt.assert_equal(stim.data.shape, (2, 4))


def test_Stimulus_rewriting_a_waveform_forgets_the_components():
    # The components describe one waveform: the one they render to. An
    # operation that installs a different one has to drop them, or a stimulus
    # would go on carrying a structured source that says something else.
    stim = Stimulus({'A1': CountingLazy(4, ['A1']),
                     'B2': CountingLazy(4, ['B2'])})
    npt.assert_equal(stim.data.shape, (2, 4))
    # Rendering is the one install that keeps them -- it built that waveform:
    npt.assert_equal(stim._components is None, False)
    for rewrite in (lambda s: s + 5, lambda s: s >> 1):
        npt.assert_equal(rewrite(stim)._components, None)
    compressed = stim._shallow_copy()
    compressed.compress()
    npt.assert_equal(compressed._components, None)


@pytest.mark.parametrize('operate', [
    lambda s: s * 2, lambda s: -s, lambda s: s._without_electrodes('A1')])
def test_Stimulus_collection_exact_operations_ignore_a_cached_waveform(
        operate):
    # Reading `.data` caches a waveform, it does not rewrite one, so the
    # components stay canonical and an exact operation still works on them.
    # Whether anybody looked first must not change the outcome.
    def build():
        return Stimulus({'A1': BiphasicPulseTrain(20, 10, 0.45, stim_dur=100),
                         'B2': BiphasicPulseTrain(23, 20, 0.45,
                                                  stim_dur=100)})
    fresh, cached = build(), build()
    cached.data  # inspect the waveform, and nothing more
    a, b = operate(fresh), operate(cached)

    def described(stim):
        return [(name, type(src), src.freq, src.amp)
                for name, src in stim._structured_sources()]
    npt.assert_equal(described(a), described(b))
    npt.assert_array_equal(a.electrodes, b.electrodes)
    npt.assert_allclose(a.data, b.data, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('factor', [2, 0.5, -1, 0])
def test_Stimulus_collection_scaling_stays_deferred(factor):
    # Scaling a collection of pulse trains scales the trains, and the
    # expensive part -- merging their time axes -- still has not happened.
    def build():
        return {'A1': BiphasicPulseTrain(20, 10, 0.45, stim_dur=100),
                'B2': BiphasicPulseTrain(23, 20, 0.45, stim_dur=100)}
    stim = Stimulus(build())
    scaled = stim * factor
    npt.assert_equal(_lazy(scaled), True)
    npt.assert_equal(_lazy(stim), True)
    # The entries are still pulse trains, at the scaled amplitude:
    npt.assert_equal([type(c) for c, _ in scaled._components],
                     [BiphasicPulseTrain, BiphasicPulseTrain])
    npt.assert_almost_equal([c.amp for c, _ in scaled._components],
                            [10 * abs(factor), 20 * abs(factor)])
    # ...and so is what the models read off the metadata:
    npt.assert_almost_equal(
        [v['metadata']['amp'] for v in scaled.metadata['electrodes'].values()],
        [10 * abs(factor), 20 * abs(factor)])
    npt.assert_allclose(scaled.data, factor * Stimulus(build()).data,
                        rtol=1e-6, atol=1e-6)
    # The original is untouched, in both descriptions:
    npt.assert_almost_equal([c.amp for c, _ in stim._components], [10, 20])
    npt.assert_almost_equal(
        [v['metadata']['amp'] for v in stim.metadata['electrodes'].values()],
        [10, 20])


def test_Stimulus_collection_scaling_needs_every_entry_to_be_a_stimulus():
    # A raw entry would have to be sampled to be scaled, which is the work
    # staying unmerged exists to avoid -- so the collection gives way instead:
    stim = Stimulus({'A1': CountingLazy(4, ['A1']), 'B2': [1, 2, 3, 4]})
    scaled = stim * 2
    npt.assert_equal(scaled._components, None)
    npt.assert_almost_equal(scaled.data[1], [2, 4, 6, 8])


def test_Stimulus_collection_offset_materializes():
    # A DC offset is not something an entry's parameters express, so the
    # collection materializes and hands back a plain waveform with no
    # structured source left behind it:
    stim = Stimulus({'A1': BiphasicPulseTrain(20, 10, 0.45, stim_dur=100),
                     'B2': BiphasicPulseTrain(23, 20, 0.45, stim_dur=100)})
    out = stim + 5
    npt.assert_equal(type(out), Stimulus)
    npt.assert_equal(out._components, None)
    npt.assert_almost_equal(out.data, stim.data + 5)
    npt.assert_equal(_lazy(stim), False)
