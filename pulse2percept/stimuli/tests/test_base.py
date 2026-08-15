import subprocess
import sys
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from copy import deepcopy
from collections import OrderedDict as ODict
from matplotlib.axes import Subplot
import matplotlib.pyplot as plt

from pulse2percept.stimuli import Stimulus
from pulse2percept.stimuli import BiphasicPulseTrain
from pulse2percept.stimuli import ImageStimulus
from pulse2percept.stimuli.base import _interp_rows, merge_time_axes
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
    with pytest.raises(ValueError):
        stim >> 1.0
    with pytest.raises(ValueError):
        stim << 1.0
    # Stimuli that do have a time axis are unaffected:
    stim = Stimulus([[0, 1, 2]], time=[0, 1, 2])
    npt.assert_almost_equal((stim >> 1.5).time, [1.5, 2.5, 3.5])
    npt.assert_almost_equal((stim << 1.5).time, [-1.5, -0.5, 0.5])


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
    stim = BiphasicPulseTrain(20, 20, 0.45, stim_dur=100)
    # Negating the train flips its polarity, which `BiphasicPulseTrain` records
    # on `cathodic_first`; every other derivation leaves the flag alone:
    for derive, flips in ((lambda s: s * 2, False), (lambda s: s + 1, False),
                          (lambda s: -s, True), (lambda s: s >> 1.0, False),
                          (lambda s: s.append(s >> 1.0), False)):
        copied = derive(stim)
        # Same class and extra attributes:
        npt.assert_equal(type(copied), type(stim))
        npt.assert_equal(copied.freq, stim.freq)
        npt.assert_equal(copied.cathodic_first, stim.cathodic_first != flips)
        npt.assert_equal(copied.is_compressed, stim.is_compressed)
        # Metadata is independent. Its contents need not be identical: the
        # operators keep the pulse parameters in sync with the data (see
        # test_pulse_trains.py):
        npt.assert_equal(copied.metadata is stim.metadata, False)
        copied.metadata['user'] = 'changed'
        npt.assert_equal(stim.metadata['user'], None)
        # The data container is independent, too:
        npt.assert_equal(copied._stim is stim._stim, False)
        before = stim.data.copy()
        copied.data[:] = 0
        npt.assert_array_equal(stim.data, before)

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
