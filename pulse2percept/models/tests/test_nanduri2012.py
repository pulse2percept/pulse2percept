import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.models import Nanduri2012Model, Nanduri2012Spatial, Percept
from pulse2percept.implants import (DiskElectrode, PointSource, ElectrodeArray,
                                    ProsthesisSystem, ArgusI)
from pulse2percept.stimuli import Stimulus, PulseTrain
from pulse2percept.utils import FreezeError


def test_Nanduri2012Spatial():
    # Nanduri2012Spatial automatically sets `atten_a`:
    model = Nanduri2012Spatial(engine='serial', xystep=5)

    # User can set `atten_a`:
    model.atten_a = 12345
    npt.assert_equal(model.atten_a, 12345)
    model.build(atten_a=987)
    npt.assert_equal(model.atten_a, 987)

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(ArgusI()), None)

    # Zero in = zero out:
    implant = ArgusI(stim=np.zeros(16))
    percept = model.predict_percept(implant)
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [1])
    npt.assert_almost_equal(percept.data, 0)

    # Multiple frames are processed independently:
    model = Nanduri2012Spatial(engine='serial', atten_a=1000, xystep=5)
    model.build()
    percept = model.predict_percept(ArgusI(stim={'A1': [1, 2]}))
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [2])
    pmax = percept.data.max(axis=(0, 1))
    npt.assert_almost_equal(percept.data[2, 3, :], pmax)
    npt.assert_almost_equal(pmax[1] / pmax[0], 2.0)

    with pytest.raises(TypeError):
        implant = ProsthesisSystem(PointSource(0, 0, 0), stim=1)
        model = Nanduri2012Spatial(engine='serial', xystep=5).build()
        model.predict_percept(implant)


def test_Nanduri2012Model():
    model = Nanduri2012Model(engine='serial', xystep=5)
    npt.assert_equal(hasattr(model, 'has_time'), True)
    npt.assert_equal(model.has_time, True)

    # User can set `dt`:
    model.temporal.dt = 1e-5
    npt.assert_almost_equal(model.temporal.dt, 1e-5)
    model.build(dt=3e-6)
    npt.assert_almost_equal(model.temporal.dt, 3e-6)

    # User cannot add more model parameters:
    with pytest.raises(FreezeError):
        model.rho = 100

    # Some parameters exist in both spatial and temporal model. We can set them
    # both at once:
    th = 0.512
    model.set_params({'thresh_percept': th})
    npt.assert_almost_equal(model.spatial.thresh_percept, th)
    npt.assert_almost_equal(model.temporal.thresh_percept, th)
    # or individually:
    model.temporal.thresh_percept = 2 * th
    npt.assert_almost_equal(model.temporal.thresh_percept, 2 * th)


def test_Nanduri2012Model_dva2ret():
    # Nanduri model uses a linear dva2ret conversion factor:
    model = Nanduri2012Model(engine='serial', xystep=5)
    for factor in [0.0, 1.0, 2.0]:
        npt.assert_almost_equal(model.spatial.dva2ret(factor), 280.0 * factor)


def test_Nanduri2012Model_ret2dva():
    # Nanduri model uses a linear dva2ret conversion factor:
    model = Nanduri2012Model(engine='serial', xystep=5)
    for factor in [0.0, 1.0, 2.0]:
        npt.assert_almost_equal(model.spatial.ret2dva(280.0 * factor), factor)


def test_Nanduri2012Model_predict_percept():
    model = Nanduri2012Model(xrange=(0, 0), yrange=(0, 0), engine='serial')
    model.build()

    # Nothing in =  nothing out:
    implant = ArgusI(stim=None)
    npt.assert_equal(model.predict_percept(implant), None)
    implant.stim = np.zeros(16)
    npt.assert_almost_equal(model.predict_percept(implant), 0)

    # Only works for DiskElectrode arrays:
    with pytest.raises(TypeError):
        implant = ProsthesisSystem(ElectrodeArray(PointSource(0, 0, 0)))
        implant.stim = 1
        model.predict_percept(implant)
    with pytest.raises(TypeError):
        implant = ProsthesisSystem(ElectrodeArray([DiskElectrode(0, 0, 0, 100),
                                                   PointSource(100, 100, 0)]))
        implant.stim = [1, 1]
        model.predict_percept(implant)

    # Requested times must be multiples of model.dt:
    implant = ProsthesisSystem(ElectrodeArray(DiskElectrode(0, 0, 0, 260)))
    tsample = 5e-6  # sampling time step (seconds)
    implant.stim = PulseTrain(tsample)
    model.temporal.dt = 0.1
    with pytest.raises(ValueError):
        model.predict_percept(implant, t=[0.01])
    with pytest.raises(ValueError):
        model.predict_percept(implant, t=[0.01, 1.0])
    with pytest.raises(ValueError):
        model.predict_percept(implant, t=np.arange(0, 0.5, 0.1001))
    model.predict_percept(implant, t=np.arange(0, 0.5, 0.1000001))

    # Can't request the same time more than once (this would break the Cython
    # loop, because `idx_frame` is incremented after a write; also doesn't
    # make much sense):
    with pytest.raises(ValueError):
        model.predict_percept(implant, t=[0.2, 0.2])

    # It's ok to extrapolate beyond `stim` if the `extrapolate` flag is set:
    model.temporal.dt = 1e-5
    npt.assert_almost_equal(model.predict_percept(implant, t=10), 0)

    # Output shape must be determined by t_percept:
    print(model.predict_percept(implant, t=0).shape)
    npt.assert_equal(model.predict_percept(implant, t=0).shape, (1, 1, 1))
    npt.assert_equal(model.predict_percept(implant, t=[0, 1]).shape, (1, 1, 2))

    # Brightness scales differently with amplitude vs frequency:
    model.temporal.dt = 5e-6
    sdur = 1.0  # stimulus duration (seconds)
    pdur = 0.45 / 1000
    t_percept = np.arange(0, sdur, 0.005)
    bright_amp = []
    print("meow")
    for amp in np.linspace(0, 50, 5):
        implant.stim = PulseTrain(tsample, freq=20, amp=amp, dur=sdur,
                                  pulse_dur=pdur, interphase_dur=pdur)
        bright_amp.append(model.predict_percept(implant, t=t_percept).max())
    bright_amp_ref = [0.0, 0.011106647, 0.08563406, 0.1482479, 0.15539996]
    npt.assert_almost_equal(bright_amp, bright_amp_ref)

    bright_freq = []
    for freq in np.linspace(0, 100, 5):
        implant.stim = PulseTrain(tsample, freq=freq, amp=20, dur=sdur,
                                  pulse_dur=pdur, interphase_dur=pdur)
        bright_freq.append(model.predict_percept(implant, t=t_percept).max())
    bright_freq_ref = [0.0, 0.054586254, 0.10436389, 0.15269955, 0.19915082]
    npt.assert_almost_equal(bright_freq, bright_freq_ref)

    # do the same for size
