import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.implants import (DiskElectrode, PointSource, ElectrodeArray,
                                    ProsthesisSystem, ArgusI)
from pulse2percept.stimuli import Stimulus, BiphasicPulseTrain
from pulse2percept.percepts import Percept
from pulse2percept.models import (Nanduri2012Model, Nanduri2012Spatial,
                                  Nanduri2012Temporal)
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

    # Multiple frames are processed independently:
    model = Nanduri2012Spatial(engine='serial', atten_a=14000, xystep=5)
    model.build()
    percept = model.predict_percept(ArgusI(stim={'A1': [1, 2]}))
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [2])
    pmax = percept.data.max(axis=(0, 1))
    npt.assert_almost_equal(percept.data[2, 3, :], pmax)
    npt.assert_almost_equal(pmax[1] / pmax[0], 2.0)

    # Nanduri model uses a linear dva2ret conversion factor:
    for factor in [0.0, 1.0, 2.0]:
        npt.assert_almost_equal(model.dva2ret(factor), 280.0 * factor)
    for factor in [0.0, 1.0, 2.0]:
        npt.assert_almost_equal(model.ret2dva(280.0 * factor), factor)


def test_Nanduri2012Temporal():
    model = Nanduri2012Temporal()
    # User can set their own params:
    model.dt = 0.1
    npt.assert_equal(model.dt, 0.1)
    model.build(dt=1e-4)
    npt.assert_equal(model.dt, 1e-4)
    # User cannot add more model parameters:
    with pytest.raises(FreezeError):
        model.rho = 100

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(ArgusI().stim), None)

    # Zero in = zero out:
    implant = ArgusI(stim=np.zeros((16, 100)))
    percept = model.predict_percept(implant.stim, t_percept=[0, 1, 2])
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, (16, 1, 3))
    npt.assert_almost_equal(percept.data, 0)

    # Can't request the same time more than once (this would break the Cython
    # loop, because `idx_frame` is incremented after a write; also doesn't
    # make much sense):
    with pytest.raises(ValueError):
        implant.stim = np.ones((16, 100))
        model.predict_percept(implant.stim, t_percept=[0.2, 0.2])

    # Brightness scales differently with amplitude vs frequency:
    model = Nanduri2012Temporal(dt=5e-3)
    model.build()
    sdur = 1000.0  # stimulus duration (ms)
    pdur = 0.45  # (ms)
    t_percept = np.arange(0, sdur, 5)
    implant = ProsthesisSystem(ElectrodeArray(DiskElectrode(0, 0, 0, 260)))
    bright_amp = []
    for amp in np.linspace(0, 50, 5):
        # implant.stim = PulseTrain(model.dt, freq=20, amp=amp, dur=sdur,
        #                           pulse_dur=pdur, interphase_dur=pdur)
        implant.stim = BiphasicPulseTrain(20, amp, pdur, interphase_dur=pdur,
                                          stim_dur=sdur)
        percept = model.predict_percept(implant.stim, t_percept=t_percept)
        bright_amp.append(percept.data.max())
    bright_amp_ref = [0.0, 0.00890, 0.0657, 0.1500, 0.1691]
    npt.assert_almost_equal(bright_amp, bright_amp_ref, decimal=3)

    bright_freq = []
    for freq in np.linspace(0, 100, 5):
        # implant.stim = PulseTrain(model.dt, freq=freq, amp=20, dur=sdur,
        #                           pulse_dur=pdur, interphase_dur=pdur)
        implant.stim = BiphasicPulseTrain(freq, 20, pdur, interphase_dur=pdur,
                                          stim_dur=sdur)
        percept = model.predict_percept(implant.stim, t_percept=t_percept)
        bright_freq.append(percept.data.max())
    bright_freq_ref = [0.0, 0.0394, 0.0741, 0.1073, 0.1385]
    npt.assert_almost_equal(bright_freq, bright_freq_ref, decimal=3)


def test_Nanduri2012Model():
    model = Nanduri2012Model(engine='serial', xystep=5)
    npt.assert_equal(hasattr(model, 'has_time'), True)
    npt.assert_equal(model.has_time, True)

    # User can set `dt`:
    model.temporal.dt = 1e-5
    npt.assert_almost_equal(model.dt, 1e-5)
    npt.assert_almost_equal(model.temporal.dt, 1e-5)
    model.build(dt=3e-4)
    npt.assert_almost_equal(model.dt, 3e-4)
    npt.assert_almost_equal(model.temporal.dt, 3e-4)

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


def test_Nanduri2012Model_predict_percept():
    # Nothing in = nothing out:
    model = Nanduri2012Model(xrange=(0, 0), yrange=(0, 0), engine='serial')
    model.build()
    implant = ArgusI(stim=None)
    npt.assert_equal(model.predict_percept(implant), None)
    implant.stim = np.zeros(16)
    npt.assert_almost_equal(model.predict_percept(implant).data, 0)

    # Single-pixel model same as TemporalModel:
    implant = ProsthesisSystem(DiskElectrode(0, 0, 0, 100))
    # implant.stim = PulseTrain(5e-6)
    implant.stim = BiphasicPulseTrain(20, 20, 0.45, interphase_dur=0.45)
    t_percept = [0, 0.01, 1.0]
    percept = model.predict_percept(implant, t_percept=t_percept)
    temp = Nanduri2012Temporal().build()
    temp = temp.predict_percept(implant.stim, t_percept=t_percept)
    npt.assert_almost_equal(percept.data, temp.data, decimal=4)

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
    tsample = 5e-3  # sampling time step (ms)
    # implant.stim = PulseTrain(tsample)
    implant.stim = BiphasicPulseTrain(20, 20, 0.45)
    model.temporal.dt = 0.1
    with pytest.raises(ValueError):
        model.predict_percept(implant, t_percept=[0.01])
    with pytest.raises(ValueError):
        model.predict_percept(implant, t_percept=[0.01, 1.0])
    with pytest.raises(ValueError):
        model.predict_percept(implant, t_percept=np.arange(0, 0.5, 0.101))
    model.predict_percept(implant, t_percept=np.arange(0, 0.5, 1.0000001))

    # Can't request the same time more than once (this would break the Cython
    # loop, because `idx_frame` is incremented after a write; also doesn't
    # make much sense):
    with pytest.raises(ValueError):
        model.predict_percept(implant, t_percept=[0.2, 0.2])

    # It's ok to extrapolate beyond `stim` if the `extrapolate` flag is set:
    model.temporal.dt = 1e-2
    npt.assert_almost_equal(model.predict_percept(implant,
                                                  t_percept=10000).data, 0)

    # Output shape must be determined by t_percept:
    npt.assert_equal(model.predict_percept(implant, t_percept=0).shape,
                     (1, 1, 1))
    npt.assert_equal(model.predict_percept(implant, t_percept=[0, 1]).shape,
                     (1, 1, 2))

    # Brightness vs. size (use values from Nanduri paper):
    model = Nanduri2012Model(xystep=0.5, xrange=(-4, 4), yrange=(-4, 4))
    model.build()
    implant = ProsthesisSystem(ElectrodeArray(DiskElectrode(0, 0, 0, 260)))
    amp_th = 30
    bright_th = 0.107
    stim_dur = 1000.0
    pdur = 0.45
    t_percept = np.arange(0, stim_dur, 5)
    amp_factors = [1, 6]
    frames_amp = []
    for amp_f in amp_factors:
        implant.stim = BiphasicPulseTrain(20, amp_f * amp_th, pdur,
                                          interphase_dur=pdur,
                                          stim_dur=stim_dur)
        percept = model.predict_percept(implant, t_percept=t_percept)
        idx_frame = np.argmax(np.max(percept.data, axis=(0, 1)))
        brightest_frame = percept.data[..., idx_frame]
        frames_amp.append(brightest_frame)
    npt.assert_equal([np.sum(f > bright_th) for f in frames_amp], [0, 161])
    freqs = [20, 120]
    frames_freq = []
    for freq in freqs:
        implant.stim = BiphasicPulseTrain(freq, 1.25 * amp_th, pdur,
                                          interphase_dur=pdur,
                                          stim_dur=stim_dur)
        percept = model.predict_percept(implant, t_percept=t_percept)
        idx_frame = np.argmax(np.max(percept.data, axis=(0, 1)))
        brightest_frame = percept.data[..., idx_frame]
        frames_freq.append(brightest_frame)
    npt.assert_equal([np.sum(f > bright_th) for f in frames_freq], [21, 49])
