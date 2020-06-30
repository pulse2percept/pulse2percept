import numpy as np
import numpy.testing as npt
import pytest

from copy import deepcopy
from collections import OrderedDict as ODict
from matplotlib.axes import Subplot
import matplotlib.pyplot as plt

from pulse2percept.stimuli import Stimulus


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
    npt.assert_equal(stim.metadata, metadata)
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

    # Plot a range of time values (times are sliced, not interpolated):
    ax = stim.plot(time=(0.2, 0.6))
    npt.assert_equal(isinstance(ax, Subplot), True)
    npt.assert_equal(len(ax.lines), 1)
    t_vals = ax.lines[0].get_data()[0]
    npt.assert_almost_equal(t_vals[0], 0.3)
    npt.assert_almost_equal(t_vals[-1], 0.5)
    ax.clear()

    # Plot exact time points:
    t_vals = [0.2, 0.3, 0.4]
    ax = stim.plot(time=t_vals)
    npt.assert_equal(isinstance(ax, Subplot), True)
    npt.assert_equal(len(ax.lines), 1)
    npt.assert_almost_equal(ax.lines[0].get_data()[0], t_vals)
    npt.assert_almost_equal(ax.lines[0].get_data()[1],
                            np.squeeze(stim[:, t_vals]))

    # Plot multiple electrodes with string names:
    for n_electrodes in [2, 3, 4]:
        stim = Stimulus(np.random.rand(n_electrodes, 20),
                        electrodes=['E%d' % i for i in range(n_electrodes)])
        axes = stim.plot()
        npt.assert_equal(isinstance(axes, (list, np.ndarray)), True)
        for ax, electrode in zip(axes, stim.electrodes):
            npt.assert_equal(isinstance(ax, Subplot), True)
            npt.assert_equal(len(ax.lines), 1)
            npt.assert_equal(ax.get_ylabel(), electrode)
            npt.assert_almost_equal(ax.lines[0].get_data()[0], stim.time)
            npt.assert_almost_equal(ax.lines[0].get_data()[1],
                                    stim[electrode, :])

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
        stim.plot(axes=axes)
    with pytest.raises(TypeError):
        stim = Stimulus(np.ones((3, 10)))
        _, axes = plt.subplots(nrows=3)
        axes[1] = 0
        stim.plot(axes=axes)


def test_Stimulus__stim():
    stim = Stimulus(3)
    # User could try and motify the data container after the constructor, which
    # would lead to inconsistencies between data, electrodes, time. The new
    # property setting mechanism prevents that.
    # Requires dict:
    with pytest.raises(TypeError):
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
    with pytest.raises(TypeError):
        data['data'] = [1, 2]
        stim._stim = data
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
    # But if you do all the things right, you can reset the stimulus by hand:
    data['data'] = np.ones((3, 1))
    data['electrodes'] = np.arange(3)
    data['time'] = None
    stim._stim = data

    data['data'] = np.ones((3, 1))
    data['electrodes'] = np.arange(3)
    data['time'] = np.arange(1)
    stim._stim = data

    data['data'] = np.ones((3, 7))
    data['electrodes'] = np.arange(3)
    data['time'] = np.ones(7)
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

    # Extrapolating should be disabled by default:
    with pytest.raises(ValueError):
        stim[0, 9.9]
    # But you can enable it:
    stim = Stimulus(1 + np.arange(12).reshape((3, 4)), extrapolate=True)
    npt.assert_almost_equal(stim[0, 9.901], 10.901)
    # If time=None, you cannot interpolate/extrapolate:
    stim = Stimulus([3, 4, 5], extrapolate=True)
    npt.assert_almost_equal(stim[0], stim.data[0, 0])
    with pytest.raises(ValueError):
        stim[0, 0.2]

    # With a single time point, interpolate is still possible:
    stim = Stimulus(np.arange(3).reshape((-1, 1)), extrapolate=False)
    npt.assert_almost_equal(stim[0], stim.data[0, 0])
    npt.assert_almost_equal(stim[0, 0], stim.data[0, 0])
    with pytest.raises(ValueError):
        stim[0, 3.33]
    stim = Stimulus(np.arange(3).reshape((-1, 1)), extrapolate=True)
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
    npt.assert_almost_equal(stim[stim.electrodes == 'C9', 0.1], [])
    npt.assert_almost_equal(stim[stim.electrodes == 'B2', 0.1001], 3.0005,
                            decimal=3)
    npt.assert_almost_equal(stim[stim.electrodes == 'B2', 0.2], 3.5)
    npt.assert_almost_equal(stim[:, stim.time < 0.4], [[0, 1], [3, 4]])
    npt.assert_almost_equal(stim[stim.electrodes == 'B2', stim.time < 0.4],
                            [3, 4])
    npt.assert_almost_equal(stim[:, stim.time > 0.6], np.zeros((2, 0)))
    npt.assert_almost_equal(stim['A1', stim.time > 0.6], [])
    npt.assert_almost_equal(stim['A1', np.isclose(stim.time, 0.3)], [1])


def test_Stimulus_merge():
    stim = Stimulus([[1, 0.3, 0.0, 0.6, 2.0]], time=np.arange(5))
    merge = Stimulus([stim, stim])
