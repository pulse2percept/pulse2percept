import numpy.testing as npt
import pytest
import numpy as np
import copy
import warnings
import matplotlib.pyplot as plt

from pulse2percept.models.cortex import DynaphosModel
from pulse2percept.models.cortex.dynaphos import _pulse_train_clocks
from pulse2percept.implants import (DiskElectrode, ElectrodeArray,
                                    EnsembleImplant, ProsthesisSystem)
from pulse2percept.implants.cortex import Cortivis, Orion
from pulse2percept.topography import Polimeni2006Map
from pulse2percept.percepts import Percept
from pulse2percept.stimuli import (AmplitudeEncoder,
                                   AsymmetricBiphasicPulseTrain,
                                   BiphasicPulseTrain, ImageStimulus,
                                   Stimulus)
from pulse2percept.units import (DimensionMismatchError, Quantity, mA,
                                 ms, s, uA, um)
from pulse2percept.utils.testing import assert_warns_msg

def test_DynaphosModel():
    model = DynaphosModel(xrange=(-3, 3), yrange=(-3, 3), step=0.1).build()

    npt.assert_equal(model.regions, ['v1'])
    npt.assert_equal(model.vfmap.regions, ['v1'])

    # can't set frequency/pulse dur that don't match up
    with pytest.raises(ValueError):
        model.build(freq=300,p_dur=10)

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(Cortivis()), None)

    implant = Cortivis(x=1000, stim={e:BiphasicPulseTrain(freq=300,amp=0,phase_dur=1) for e in Cortivis().electrode_names})
    # Zero in = zero out:
    percept = model.predict_percept(implant)
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, list(model.grid.x.shape)+[51]) # 51 time points
    npt.assert_almost_equal(percept.data, 0)

    # Can't pass stimulus with no time component
    with pytest.raises(ValueError):
        model.predict_percept(Cortivis(stim=[300 for e in Cortivis().electrode_names]))

def test_predict_spatial():
    # test that no current can spread between hemispheres
    model = DynaphosModel(xrange=(-3, 3), yrange=(-3, 3), step=0.5).build()
    implant = Orion(x = 15000)
    implant.stim = {e:BiphasicPulseTrain(freq=300,amp=2000,phase_dur=0.17) for e in implant.electrode_names}
    # Check brightest frame of percept
    percept = model.predict_percept(implant).max(axis='frames')
    half = percept.shape[1] // 2
    npt.assert_equal(np.all(percept[:, half+1:] == 0), True)
    npt.assert_equal(np.all(percept[:, :half] != 0), True)

def test_temporal_predict():
    model = DynaphosModel(step=0.1).build()
    # User can set params
    model.dt = 40
    npt.assert_equal(model.dt, 40)

    implant = Cortivis(stim=np.zeros((96, 100)))

    # Can't request the same time more than once (this would break the Cython
    # loop, because `idx_frame` is incremented after a write; also doesn't
    # make much sense):
    with pytest.raises(ValueError):
        implant.stim = np.ones((96, 100))
        model.predict_percept(implant, t_percept=[0.2, 0.2])

    # Brightness scales with amplitude. The train is built on the model's own
    # clock, so that the duty cycle driving the activation is the one that
    # produced the waveform. It used not to be: a train assigned on its own
    # (rather than as {electrode: train}) carried no per-electrode metadata,
    # so this model silently simulated it at `self.freq`/`self.p_dur` however
    # it was actually built. It now reads the train itself.
    model.dt = 20
    sdur = 1000.0  # stimulus duration (ms)
    pdur = model.p_dur  # (ms)
    t_percept = np.arange(0, sdur, 20)
    implant = ProsthesisSystem(ElectrodeArray(DiskElectrode(0, 0, 0, 260)))
    bright_amp = []
    for amp in np.linspace(20, 70, 5):
        implant.stim = BiphasicPulseTrain(model.freq, amp, pdur,
                                          interphase_dur=pdur, stim_dur=sdur)
        percept = model.predict_percept(implant, t_percept=t_percept)
        bright_amp.append(percept.data.max())
    bright_amp_ref = np.array([0.0, 0.0, 0.4636, 0.7247, 0.8891])
    npt.assert_almost_equal(bright_amp, bright_amp_ref, decimal=3)
    npt.assert_equal(np.all(np.diff(bright_amp) >= 0), True)

    # Test that default models give expected values
    implant = Orion(x=15000, stim={'55': BiphasicPulseTrain(freq=300, amp=100, phase_dur=0.17)})
    percept = model.predict_percept(implant)
    npt.assert_equal(np.sum(percept.data > 0.0122), 147)
    npt.assert_equal(np.sum(percept.data > 0.0375), 96)
    npt.assert_equal(np.sum(percept.data > 0.3305), 49)
    npt.assert_equal(np.sum(percept.data > 0.8451), 39)
    npt.assert_equal(np.sum(percept.data > 0.8883), 9)

def test_deepcopy_Dynaphos():
    original = DynaphosModel()
    copied = copy.deepcopy(original)

    # Assert these are two different objects
    npt.assert_equal(id(original) != id(copied), True)

    # Assert these objects are equivalent
    npt.assert_equal(original.__dict__, copied.__dict__)

    # Assert building one object does not affect the copied
    original.build()
    npt.assert_equal(copied.is_built, False)
    # Array-aware: a plain dict comparison raises once the model is
    # built, because `array == array` cannot be coerced to a bool.
    npt.assert_raises(AssertionError, npt.assert_equal,
                      original.__dict__, copied.__dict__)

    # Assert destroying the original doesn't affect the copied
    original = None
    npt.assert_equal(copied is not None, True)

def test_dynaphos_plot():
    # make sure that plotting works before and after building
    m = DynaphosModel()
    m.plot()
    plt.close()
    m.build()
    m.plot()
    plt.close()


def test_DynaphosModel_units():
    """A unitful parameter lands on the same percept as the bare one

    `rheobase` is a current in microamps, and 0.0239 mA is 23.9 uA -- but
    23.900000000000002 after the multiplication, which is why this compares
    with a tolerance rather than for equality.
    """
    kwargs = dict(xrange=(-3, 3), yrange=(-3, 3), step=0.5)
    bare = DynaphosModel(rheobase=23.9, **kwargs).build()
    unitful = DynaphosModel(rheobase=0.0239 * mA, **kwargs).build()
    npt.assert_allclose(unitful.rheobase, 23.9, rtol=1e-12)
    npt.assert_equal(isinstance(unitful.rheobase, Quantity), False)
    implant = Cortivis(stim={'11': BiphasicPulseTrain(20, 50, 0.45,
                                                      stim_dur=100)})
    npt.assert_allclose(unitful.predict_percept(implant).data,
                        bare.predict_percept(implant).data, rtol=1e-6)
    # The model states what its numbers mean:
    npt.assert_equal((bare.stimulus_unit, bare.space_unit, bare.time_unit),
                     (uA, um, ms))
    with pytest.raises(DimensionMismatchError):
        DynaphosModel(rheobase=5 * ms)


def test_DynaphosModel_t_percept_units():
    """This model overrides `predict_percept`, so it normalizes for itself"""
    model = DynaphosModel(xrange=(-3, 3), yrange=(-3, 3), step=1).build()
    implant = Cortivis(stim={'11': BiphasicPulseTrain(20, 50, 0.45,
                                                      stim_dur=100)})
    bare = model.predict_percept(implant, t_percept=[0, 20, 40])
    for spelling in ([0, 20, 40] * ms, np.array([0, .02, .04]) * s):
        unitful = model.predict_percept(implant, t_percept=spelling)
        npt.assert_allclose(unitful.data, bare.data, rtol=1e-12)
        npt.assert_allclose(unitful.time, [0, 20, 40], rtol=1e-12)
    with pytest.raises(DimensionMismatchError):
        model.predict_percept(implant, t_percept=[0, 20] * uA)


def test_DynaphosModel_default_frame_clock_stops_at_the_stimulus():
    """The default output clock does not run past the end of the stimulus

    `arange`'s half-open end used to be nudged by the literal 1, which meant
    one *millisecond*: with a `dt` finer than that it emitted frames after the
    stimulus was over, and for a model counting in anything but milliseconds
    it would have been meaningless.
    """
    stim = BiphasicPulseTrain(20, 50, 0.1, stim_dur=10)
    implant = Cortivis(stim={'11': stim})
    kwargs = dict(xrange=(-2, 2), yrange=(-2, 2), step=1)

    # Coarser than a millisecond, which is the case the literal was written
    # for, and still the same clock it always produced:
    model = DynaphosModel(dt=2, **kwargs).build()
    npt.assert_allclose(model.predict_percept(implant).time,
                        np.arange(0, 11, 2), rtol=1e-12)

    # Finer than a millisecond, which is where it overshot:
    model = DynaphosModel(dt=0.5, **kwargs).build()
    percept = model.predict_percept(implant)
    npt.assert_allclose(percept.time, np.arange(0, 10.25, 0.5), rtol=1e-12)
    npt.assert_equal(percept.time[-1] <= implant.stim.time[-1], True)
    # The endpoint is included, not dropped:
    npt.assert_allclose(percept.time[-1], implant.stim.time[-1], rtol=1e-12)


def test_DynaphosModel_deprecated_xystep():
    # `step` was called `xystep` until 0.10.0. Dynaphos derives from
    # `BaseModel` rather than `SpatialModel` and lays out its own grid, so it
    # declares the alias itself and has to be checked separately:
    msg = "The 'xystep' parameter of DynaphosModel is deprecated"
    assert_warns_msg(DeprecationWarning, DynaphosModel, msg, xystep=1)
    with pytest.warns(DeprecationWarning):
        model = DynaphosModel(xrange=(-2, 2), yrange=(-2, 2), xystep=1)
    npt.assert_almost_equal(model.step, 1)

    assert_warns_msg(DeprecationWarning, setattr, msg, model, 'xystep', 2)
    npt.assert_almost_equal(model.step, 2)
    with pytest.warns(DeprecationWarning):
        npt.assert_almost_equal(model.xystep, 2)

    assert_warns_msg(DeprecationWarning, model.set_params, msg, xystep=1)
    npt.assert_almost_equal(model.step, 1)
    assert_warns_msg(DeprecationWarning, model.build, msg, xystep=0.5)
    npt.assert_almost_equal(model.step, 0.5)
    npt.assert_almost_equal(np.unique(np.diff(model.grid.x[0, :]))[0], 0.5)

    # Both names are the same parameter, so supplying both must raise:
    for params in ({'xystep': 1, 'step': 2}, {'step': 2, 'xystep': 1}):
        with pytest.raises(TypeError, match="same parameter"):
            DynaphosModel(**params)

    # The new name stays silent:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        model = DynaphosModel(step=1)
        model.step = 2
        npt.assert_almost_equal(model.step, 2)


def test_dynaphos_reads_the_pulse_train_itself():
    # The same train has to predict the same percept however it is assigned.
    # That used to be false: a bare train carried no per-electrode metadata
    # and was simulated on the model's default clock instead of its own.
    model = DynaphosModel(step=0.5, xrange=(-2, 2), yrange=(-2, 2)).build()
    model.dt = 20
    t_percept = np.arange(0, 200, 20)

    def predict(stim):
        implant = ProsthesisSystem(ElectrodeArray(DiskElectrode(0, 0, 0, 260)))
        implant.stim = stim
        return model.predict_percept(implant, t_percept=t_percept).data

    def train():
        return BiphasicPulseTrain(300, 100, 0.17, interphase_dur=0.17,
                                  stim_dur=200)
    bare = predict(train())
    npt.assert_array_equal(bare, predict({0: train()}))
    npt.assert_equal(np.any(bare), True)

    # User metadata says nothing about the clock, however it is written:
    corrupt = train()
    corrupt.metadata['user'] = {'freq': 1, 'amp': 0, 'phase_dur': 99}
    npt.assert_array_equal(bare, predict(corrupt))


def test_dynaphos_uses_its_defaults_for_an_arbitrary_waveform():
    # No pulse train behind the samples, so there is no clock to read and the
    # model simulates on its own -- which is what it has always done:
    model = DynaphosModel(step=0.5, xrange=(-2, 2), yrange=(-2, 2)).build()
    model.dt = 20
    implant = ProsthesisSystem(ElectrodeArray(DiskElectrode(0, 0, 0, 260)))
    implant.stim = Stimulus([[0, 100, 100, 0]], time=[0, 1, 199, 200])
    percept = model.predict_percept(implant, t_percept=np.arange(0, 200, 20))
    npt.assert_equal(np.any(percept.data), True)
    # A train whose own clock happens to be the model's gives the same answer
    # as the defaults do, which is what says the defaults were used:
    model.freq, model.p_dur = 111, 0.29
    other = model.predict_percept(implant,
                                  t_percept=np.arange(0, 200, 20)).data
    npt.assert_equal(np.allclose(percept.data, other), False)


def test_dynaphos_reads_the_clock_before_compression():
    # `compress` installs a new waveform, which is what says the trains behind
    # it no longer describe it. The parameters have to be taken first -- and
    # compression drops the electrodes driven at zero, so what is left has to
    # still line up with the right train.
    model = DynaphosModel(step=0.5, xrange=(-2, 2), yrange=(-2, 2)).build()
    model.dt = 20
    implant = ProsthesisSystem(ElectrodeArray([
        DiskElectrode(0, 0, 0, 260), DiskElectrode(1000, 0, 0, 260)]))
    implant.stim = {0: BiphasicPulseTrain(300, 0, 0.17, stim_dur=200),
                    1: BiphasicPulseTrain(300, 100, 0.17, stim_dur=200)}
    both = model.predict_percept(implant, t_percept=np.arange(0, 200, 20))
    # Only the second electrode survives compression, and it is the second
    # train's clock that has to reach the simulation:
    implant.stim = {1: BiphasicPulseTrain(300, 100, 0.17, stim_dur=200)}
    npt.assert_allclose(both.data,
                        model.predict_percept(
                            implant, t_percept=np.arange(0, 200, 20)).data)


def _ensemble_of_two_clocks():
    """Two implants driven at different frequencies, merged into one"""
    fast = Orion(stim={e: BiphasicPulseTrain(50, 300, 0.45, stim_dur=100)
                       for e in Orion().electrode_names})
    slow = Orion(x=-35000,
                 stim={e: BiphasicPulseTrain(20, 300, 0.85, stim_dur=100)
                       for e in Orion().electrode_names})
    return EnsembleImplant([fast, slow])


def test_dynaphos_reads_ensemble_clocks():
    # An ensemble keeps its members' trains rather than sampling them away,
    # so every member is simulated at the clock it was built with instead of
    # the model's own default.
    ensemble = _ensemble_of_two_clocks()
    clocks = _pulse_train_clocks(ensemble.stim)
    npt.assert_equal(len(clocks), len(ensemble.stim.electrodes))
    npt.assert_equal(sorted(set(clocks.values())), [(20, 0.85), (50, 0.45)])
    # ...and the members keep their own, rather than sharing one:
    npt.assert_equal(clocks['0-96'], (50, 0.45))
    npt.assert_equal(clocks['1-96'], (20, 0.85))


def test_dynaphos_ensemble_prediction_uses_those_clocks():
    model = DynaphosModel(xrange=(-3, 3), yrange=(-3, 3), step=1).build()
    ensemble = _ensemble_of_two_clocks()
    with_clocks = model.predict_percept(ensemble).data
    npt.assert_equal(np.any(with_clocks), True)
    # The same waveform with the trains behind it taken away is back on the
    # model's default clock
    waveform_only = _ensemble_of_two_clocks()
    waveform_only.stim = Stimulus(ensemble.stim.data,
                                  electrodes=ensemble.stim.electrodes,
                                  time=ensemble.stim.time)
    npt.assert_equal(_pulse_train_clocks(waveform_only.stim), None)
    npt.assert_equal(
        np.allclose(with_clocks,
                    model.predict_percept(waveform_only).data), False)


def test_dynaphos_clocks_are_not_read_when_structure_says_otherwise():
    # Samples with nothing behind them leave the model on its own clock:
    npt.assert_equal(
        _pulse_train_clocks(Stimulus([[0, 100, 100, 0]],
                                     time=[0, 1, 99, 100])), None)
    # A DC offset leaves no train behind, and drops the parameters with it:
    stim = Stimulus({'0': BiphasicPulseTrain(20, 100, 0.45, stim_dur=100)})
    npt.assert_equal(_pulse_train_clocks(stim + 5), None)
    # A stimulus made of something other than biphasic trains stays on the
    # defaults too:
    asym = Stimulus({'0': AsymmetricBiphasicPulseTrain(20, 100, 50, 0.45, 0.9,
                                                       stim_dur=100)})
    npt.assert_equal(_pulse_train_clocks(asym), None)


def test_dynaphos_uses_its_defaults_for_an_encoded_stimulus():
    # An encoder's schedule can change frequency from frame to frame, so there
    # is no per-electrode clock to take from it. The model stays on its own:
    implant = Cortivis(x=1000)
    encoded = AmplitudeEncoder().encode(
        ImageStimulus(np.linspace(0, 1, 64).reshape(8, 8)), implant=implant)
    npt.assert_equal(_pulse_train_clocks(encoded), None)
    # A single-electrode schedule is a structured source, and is refused on
    # the same grounds rather than read as a pulse train:
    solo = AmplitudeEncoder().encode(ImageStimulus(np.array([[0.7]])))
    npt.assert_equal(len(solo._structured_sources()), 1)
    npt.assert_equal(_pulse_train_clocks(solo), None)
