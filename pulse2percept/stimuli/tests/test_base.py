import numpy as np
import numpy.testing as npt
import pytest
from copy import deepcopy
from collections import OrderedDict as ODict

from pulse2percept.stimuli import Stimulus, PulseTrain


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
    stim = Stimulus(PulseTrain(0.01 / 1000, dur=0.005), compress=False)
    npt.assert_equal(stim.electrodes, [0])
    npt.assert_equal(stim.shape, (1, 500))
    stim = Stimulus(PulseTrain(0.01 / 1000, dur=0.005), compress=True)
    npt.assert_equal(stim.electrodes, [0])
    npt.assert_equal(stim.shape, (1, 8))
    # Specific electrode in time:
    stim = Stimulus({'C3': PulseTrain(0.01 / 1000, dur=0.004)}, compress=False)
    npt.assert_equal(stim.electrodes, ['C3'])
    npt.assert_equal(stim.shape, (1, 400))
    stim = Stimulus({'C3': PulseTrain(0.01 / 1000, dur=0.004)}, compress=True)
    npt.assert_equal(stim.electrodes, ['C3'])
    npt.assert_equal(stim.shape, (1, 8))
    # Multiple specific electrodes in time:
    stim = Stimulus({'C3': PulseTrain(0.01 / 1000, dur=0.004),
                     'F4': PulseTrain(0.01 / 1000, delay=0.0001, dur=0.004)},
                    compress=True)
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

    # Specify new time points:
    stim = Stimulus(np.ones((2, 5)), compress=True)
    npt.assert_equal(stim.time, [0, 4])
    stim = Stimulus(stim, time=np.array(stim.time) / 10.0)
    npt.assert_equal(stim.electrodes, [0, 1])
    npt.assert_equal(stim.time, [0, 0.4])

    # Not allowed:
    with pytest.raises(ValueError):
        # Multiple electrodes in time, different time stamps:
        stim = Stimulus([PulseTrain(0.01 / 1000, dur=0.004),
                         PulseTrain(0.005 / 1000, dur=0.002)])
    with pytest.raises(ValueError):
        # First one doesn't have time:
        stim = Stimulus({'A2': 1, 'C3': PulseTrain(0.01 / 1000, dur=0.004)})
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


@pytest.mark.parametrize('tsample', (5e-6, 1e-7))
def test_Stimulus_compress(tsample):
    # Single pulse train:
    pdur = 0.00045
    pt = PulseTrain(tsample, pulse_dur=pdur, dur=0.005)
    idata = pt.data.reshape((1, -1))
    ielec = np.array([0])
    itime = np.arange(idata.shape[-1]) * pt.tsample
    stim = Stimulus(idata, electrodes=ielec, time=itime, compress=False)
    # Compress the data. The original `tsample` shouldn't matter as long as the
    # resolution is fine enough to capture all of the pulse train:
    stim.compress()
    npt.assert_equal(stim.shape, (1, 8))
    # Electrodes are unchanged:
    npt.assert_equal(ielec, stim.electrodes)
    # First and last time step are always preserved:
    npt.assert_almost_equal(itime[0], stim.time[0])
    npt.assert_almost_equal(itime[-1], stim.time[-1])
    # The first rising edge happens at t=`pdur`:
    npt.assert_almost_equal(stim.time[1], pdur - tsample)
    npt.assert_almost_equal(stim.data[0, 1], -20)
    npt.assert_almost_equal(stim.time[2], pdur)
    npt.assert_almost_equal(stim.data[0, 2], 0)

    # Two pulse trains with slight delay/offset, and a third that's all 0:
    delay = 0.0001
    pt = PulseTrain(tsample, delay=0, dur=0.005)
    pt2 = PulseTrain(tsample, delay=delay, dur=0.005)
    pt3 = PulseTrain(tsample, amp=0, dur=0.005)
    idata = np.vstack((pt.data, pt2.data, pt3.data))
    ielec = np.array([0, 1, 2])
    itime = np.arange(idata.shape[-1]) * pt.tsample
    stim = Stimulus(idata, electrodes=ielec, time=itime, compress=False)
    # Compress the data:
    stim.compress()
    npt.assert_equal(stim.shape, (2, 16))
    # Zero electrodes should be deselected:
    npt.assert_equal(stim.electrodes, np.array([0, 1]))
    # First and last time step are always preserved:
    npt.assert_almost_equal(itime[0], stim.time[0])
    npt.assert_almost_equal(itime[-1], stim.time[-1])
    # The first rising edge happens at t=`delay`:
    npt.assert_almost_equal(stim.time[1], delay - tsample)
    npt.assert_almost_equal(stim.data[0, 1], -20)
    npt.assert_almost_equal(stim.data[1, 1], 0)
    npt.assert_almost_equal(stim.time[2], delay)
    npt.assert_almost_equal(stim.data[0, 2], -20)
    npt.assert_almost_equal(stim.data[1, 2], -20)

    # Repeated calls to compress won't change the result:
    idata = stim.data
    ielec = stim.electrodes
    itime = stim.time
    stim.compress()
    npt.assert_equal(idata, stim.data)
    npt.assert_equal(ielec, stim.electrodes)
    npt.assert_equal(itime, stim.time)

    # Tricky case is a Stimulus reduced to [] after compress:
    stim = Stimulus(np.zeros((2, 4)))
    stim.compress()
    npt.assert_equal(stim.shape, (0, 2))
    npt.assert_equal(stim[:], np.zeros((0, 2)))
    npt.assert_equal(stim[:, :], np.zeros((0, 2)))
    npt.assert_equal(stim[:, 0], [])
    npt.assert_equal(stim[:, 0.123], [])
    npt.assert_equal(stim[:, [0.25, 0.88]], [])


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
    npt.assert_equal(stim == Stimulus(np.ones((2, 3)) * 1.1), False)
    # Different type:
    npt.assert_equal(stim == np.ones((2, 3)), False)
    npt.assert_equal(stim != np.ones((2, 3)), True)
    # Annoying but possible:
    npt.assert_equal(Stimulus([]), Stimulus(()))


def test_Stimulus___getitem__():
    stim = Stimulus(1 + np.arange(12).reshape((3, 4)))
    # Slicing:
    npt.assert_equal(stim[:], stim.data)
    npt.assert_equal(stim[...], stim.data)
    npt.assert_equal(stim[:, :], stim.data)
    npt.assert_equal(stim[:2], stim.data[:2])
    npt.assert_equal(stim[:, 0], stim.data[:, 0].reshape((-1, 1)))
    npt.assert_equal(stim[0, :], stim.data[0, :])
    npt.assert_equal(stim[0, ...], stim.data[0, ...])
    npt.assert_equal(stim[..., 0], stim.data[..., 0].reshape((-1, 1)))
    # More advanced slicing is not yet implemented:
    with pytest.raises(NotImplementedError):
        # This is ambiguous because no step size is given:
        stim[1, 2:5]
    # Single element:
    npt.assert_equal(stim[0, 0], stim.data[0, 0])
    # Interpolating time:
    npt.assert_almost_equal(stim[0, 2.6], 3.6)
    npt.assert_almost_equal(stim[..., 2.3], np.array([[3.3], [7.3], [11.3]]))
    # The second dimension is not a column index!
    npt.assert_almost_equal(stim[0, 0], 1.0)
    npt.assert_almost_equal(stim[0, [0, 1]], np.array([[1.0, 2.0]]))
    npt.assert_almost_equal(stim[0, [0.21, 1]], np.array([[1.21, 2.0]]))
    npt.assert_almost_equal(stim[[0, 1], [0.21, 1]],
                            np.array([[1.21, 2.0], [5.21, 6.0]]))

    # "Valid" index errors:
    with pytest.raises(IndexError):
        stim[10, :]
    with pytest.raises(TypeError):
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
