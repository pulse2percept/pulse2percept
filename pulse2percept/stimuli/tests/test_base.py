import numpy as np
import numpy.testing as npt
import pytest
from collections import OrderedDict as ODict

from pulse2percept.stimuli import Stimulus, PulseTrain


def test_Stimulus():
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


@pytest.mark.parametrize('time', [0, 0.9, 12.5, np.array([3.5, 7.8])])
@pytest.mark.parametrize('shape', [(1, 4), (2, 2), (3, 5)])
def test_Stimulus_interp(time, shape):
    # Time is None: nothing to interpolate/extrapolate, simply return the
    # original data
    stim = Stimulus(np.ones(shape[0]))
    npt.assert_almost_equal(stim.interp(time=None).data, stim.data)

    # Single time point: nothing to interpolate/extrapolate, simply return the
    # original data
    data = np.ones(shape).reshape((-1, 1))
    stim = Stimulus(data)
    npt.assert_almost_equal(stim.interp(time=time).data, data)

    # Specific time steps:
    stim = Stimulus([np.arange(shape[1])] * shape[0], compress=False)
    npt.assert_almost_equal(stim.interp(time=time).data,
                            np.ones((shape[0], 1)) * time)
    npt.assert_almost_equal(stim.interp(time=[time]).data,
                            np.ones((shape[0], 1)) * time)
    # All time steps:
    npt.assert_almost_equal(stim.interp(time=stim.time).data, stim.data)


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
