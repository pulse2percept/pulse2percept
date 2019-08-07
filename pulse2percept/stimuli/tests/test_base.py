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
    stim = Stimulus(np.arange(10))
    npt.assert_equal(stim.shape, (9, 1))
    npt.assert_equal(stim.electrodes, np.arange(1, 10))
    npt.assert_equal(stim.time, None)
    # Electrodes + specific time, time will be trimmed:
    stim = Stimulus(np.ones((4, 3)), time=[-3, -2, -1])
    npt.assert_equal(stim.shape, (4, 2))
    npt.assert_equal(stim.time, [-3, -1])
    # Electrodes + specific time, but don't trim:
    stim = Stimulus(np.ones((4, 3)), time=[-3, -2, -1], sparsify=False)
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
    stim = Stimulus(np.ones((6, 100)))
    npt.assert_equal(stim.shape, (6, 2))
    # Single electrode in time:
    stim = Stimulus(PulseTrain(0.01 / 1000, dur=0.005))
    # Specific electrode in time:
    stim = Stimulus({'C3': PulseTrain(0.01 / 1000, dur=0.004)})
    # Multiple specific electrodes in time:
    stim = Stimulus({'C3': PulseTrain(0.01 / 1000, dur=0.004),
                     'F4': PulseTrain(0.01 / 1000, delay=0.0001, dur=0.004)})
    # Stimulus from a Stimulus (might happen in ProsthesisSystem):
    stim = Stimulus(Stimulus(4), electrodes='B3')
    npt.assert_equal(stim.shape, (1, 1))
    npt.assert_equal(stim.electrodes, ['B3'])
    npt.assert_equal(stim.time, None)
    # Saves metadata:
    metadata = {'a': 0, 'b': 1}
    stim = Stimulus(3, metadata=metadata)
    npt.assert_equal(stim.metadata, metadata)

    # Zero activation:
    source = np.zeros((2, 4))
    stim = Stimulus(source, sparsify=True)
    npt.assert_equal(stim.shape, (0, 2))
    npt.assert_equal(stim.time, [0, source.shape[1] - 1])
    stim = Stimulus(source, sparsify=False)
    npt.assert_equal(stim.shape, source.shape)
    npt.assert_equal(stim.time, np.arange(source.shape[1]))

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
def test_Stimulus_sparsify(tsample):
    # Single pulse train:
    pdur = 0.00045
    pt = PulseTrain(tsample, pulse_dur=pdur, dur=0.005)
    idata = pt.data.reshape((1, -1))
    ielec = np.array([0])
    itime = np.arange(idata.shape[-1]) * pt.tsample
    # Sparsify should compress the data. The original `tsample` shouldn't
    # matter as long as the resolution is fine enough to capture all of the
    # pulse train:
    odata, oelec, otime = Stimulus.sparsify(idata, ielec, itime)
    npt.assert_equal(odata.shape, (1, 8))
    # Electrodes are unchanged:
    npt.assert_equal(ielec, oelec)
    # First and last time step are always preserved:
    npt.assert_almost_equal(itime[0], otime[0])
    npt.assert_almost_equal(itime[-1], otime[-1])
    # The first rising edge happens at t=`pdur`:
    npt.assert_almost_equal(otime[1], pdur - tsample)
    npt.assert_almost_equal(odata[0, 1], -20)
    npt.assert_almost_equal(otime[2], pdur)
    npt.assert_almost_equal(odata[0, 2], 0)

    # Two pulse trains with slight delay/offset, and a third that's all 0:
    delay = 0.0001
    pt = PulseTrain(tsample, delay=0, dur=0.005)
    pt2 = PulseTrain(tsample, delay=delay, dur=0.005)
    pt3 = PulseTrain(tsample, amp=0, dur=0.005)
    idata = np.vstack((pt.data, pt2.data, pt3.data))
    ielec = np.array([0, 1, 2])
    itime = np.arange(idata.shape[-1]) * pt.tsample
    # Sparsify should compress the data. The original `tsample` shouldn't
    # matter as long as the resolution is fine enough to capture all of the
    # pulse train:
    odata, oelec, otime = Stimulus.sparsify(idata, ielec, itime)
    npt.assert_equal(odata.shape, (2, 16))
    # Zero electrodes should be deselected:
    npt.assert_equal(oelec, np.array([0, 1]))
    # First and last time step are always preserved:
    npt.assert_almost_equal(itime[0], otime[0])
    npt.assert_almost_equal(itime[-1], otime[-1])
    # The first rising edge happens at t=`delay`:
    npt.assert_almost_equal(otime[1], delay - tsample)
    npt.assert_almost_equal(odata[0, 1], -20)
    npt.assert_almost_equal(odata[1, 1], 0)
    npt.assert_almost_equal(otime[2], delay)
    npt.assert_almost_equal(odata[0, 2], -20)
    npt.assert_almost_equal(odata[1, 2], -20)

    # Repeated calls to sparsify won't change the result:
    idata = odata
    ielec = oelec
    itime = otime
    odata, oelec, otime = Stimulus.sparsify(idata, ielec, itime)
    npt.assert_equal(idata, odata)
    npt.assert_equal(ielec, oelec)
    npt.assert_equal(itime, otime)


# def test_Stimulus__from_ndarray():
#     # From NumPy array:
#     shape = (2, 4)
#     stim = Stimulus(np.ones(shape))
#     npt.assert_equal(stim.ndim, len(shape))
#     npt.assert_equal(stim.size, np.prod(shape))
#     # Zeros will be trimmed:
#     stim = Stimulus(np.arange(4) - 2)
#     npt.assert_equal(np.allclose(stim.coords['electrode'], [0, 1, 3]), True)
#     npt.assert_equal(np.allclose(stim.data, [-2, -1, 1]), True)
#     npt.assert_equal(len(stim), size - 1)

#     # From list:
#     stim = Stimulus([0, 0, 2] * size)
#     npt.assert_equal(len(stim), size)

#     # From tuple:
#     stim = Stimulus((0, 2, 5, 2))
#     npt.assert_equal(len(stim), 3)

#     # With coordinates:
