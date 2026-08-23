import warnings
from copy import deepcopy

import numpy as np
import pytest
import numpy.testing as npt
from scipy.integrate import trapezoid

from pulse2percept.stimuli import VideoStimulus
from pulse2percept.units import dimensionless
from pulse2percept.stimuli import (Stimulus, PulseTrain, BiphasicPulse,
                                   BiphasicPulseTrain,
                                   BiphasicTripletTrain,
                                   AsymmetricBiphasicPulseTrain)
from pulse2percept.stimuli.pulse_trains import _tile_pulse
from pulse2percept.utils.constants import DT
from pulse2percept.units import (DimensionMismatchError, Hz, Quantity,
                                 kHz, mA, ms, uA, us, xTh)
from pulse2percept.units import s as sec


def test_PulseTrain():
    # All zeros:
    npt.assert_almost_equal(PulseTrain(10, Stimulus(np.zeros((1, 5)))).data,
                            0)
    # Simple fake pulse:
    pulse = Stimulus([[0, -1, 0]], time=[0, 0.1, 0.2])
    for n_pulses in [2, 3, 10]:
        pt = PulseTrain(10, pulse, n_pulses=n_pulses, electrode='A4')
        npt.assert_equal(np.sum(np.isclose(pt.data, -1)), n_pulses)
        npt.assert_equal(pt.electrodes, 'A4')

    # PulseTrains can cut off/trim individual pulses if necessary:
    pt = PulseTrain(3, pulse, stim_dur=11)
    npt.assert_almost_equal(pt.time[-1], 11)
    npt.assert_almost_equal(pt[0, 11], 0)

    # Invalid calls:
    with pytest.raises(TypeError):
        # Wrong stimulus type:
        PulseTrain(10, {'a': 1})
    with pytest.raises(ValueError):
        # Pulse does not fit:
        PulseTrain(100000, pulse)
    with pytest.raises(ValueError):
        # n_pulses does not fit:
        PulseTrain(10, pulse, n_pulses=100000)
    with pytest.raises(ValueError):
        # No time component:
        PulseTrain(10, Stimulus(1))
    with pytest.raises(ValueError):
        # Empty stim:
        pulse = Stimulus([[0, 0, 0]], time=[0, 0.1, 0.2], compress=True)
        PulseTrain(10, pulse)


def test_PulseTrain_whole_pulses():
    # A train delivers only pulses it can finish. Starting one and cutting it
    # short at `stim_dur` leaves a net current behind:
    frame_dur = 1000 / 29.97
    for freq, expected in [(20, 1), (30, 1), (60, 2), (90, 3)]:
        pt = BiphasicPulseTrain(freq, 50, 0.46, stim_dur=frame_dur)
        n_pulses = np.count_nonzero(
            np.diff((pt.data[0] < 0).astype(int)) > 0)
        npt.assert_equal(n_pulses, expected)
        npt.assert_equal(pt.is_charge_balanced, True)
        npt.assert_almost_equal(pt.time[-1], frame_dur)
    # The window still holds one pulse even when the frequency asks for less
    # than one, so that a slow train is not silence:
    npt.assert_equal(np.count_nonzero(np.diff(
        (BiphasicPulseTrain(1, 50, 0.46, stim_dur=10).data[0] < 0
         ).astype(int)) > 0), 1)
    # 0 Hz is silence, though:
    npt.assert_almost_equal(BiphasicPulseTrain(0, 50, 0.46, stim_dur=10).data,
                            0)


def test_PulseTrain_time_axis():
    # Whatever the frequency, the time axis stays strictly increasing. This
    # used to fail for the rare frequency whose train ended just short of
    # `stim_dur`, leaving the trimmed end point less than DT past its
    # predecessor:
    for freq in np.linspace(1, 300, 500):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            pt = BiphasicPulseTrain(freq, 1, 0.46, stim_dur=1000 / 29.97 - DT)
        npt.assert_equal(np.all(np.diff(pt.time) > 0.95 * DT), True)


def test_PulseTrain_charge_balance_over_time():
    # Charge balance has to survive a long train. The time axis is float64
    # precisely because float32 cannot resolve a DT-wide pulse edge past
    # t = 8.4 s, which used to leave even a symmetric train unbalanced:
    for n_pulses in (6, 60, 600, 6000):
        pt = BiphasicPulseTrain(1000, 50, 0.46, n_pulses=n_pulses,
                                stim_dur=n_pulses)
        npt.assert_equal(pt.is_charge_balanced, True)
    # Every pulse edge is still exactly DT wide at the end of a 20 s train:
    pt = BiphasicPulseTrain(50, 50, 0.46, stim_dur=20000)
    npt.assert_almost_equal(np.diff(pt.time).min(), DT)


@pytest.mark.parametrize('amp', (-3, 4))
@pytest.mark.parametrize('interphase_dur', (0, np.pi))
@pytest.mark.parametrize('delay_dur', (0, np.e))
@pytest.mark.parametrize('cathodic_first', (True, False))
def test_BiphasicPulseTrain(amp, interphase_dur, delay_dur, cathodic_first):
    freq = 23.456
    stim_dur = 657.456
    phase_dur = 2
    window_dur = 1000.0 / freq
    n_pulses = int(freq * stim_dur / 1000.0)
    mid_first_pulse = delay_dur + phase_dur / 2.0
    mid_interphase = delay_dur + phase_dur + interphase_dur / 2.0
    mid_second_pulse = delay_dur + interphase_dur + 1.5 * phase_dur
    first_amp = -np.abs(amp) if cathodic_first else np.abs(amp)
    second_amp = -first_amp

    # Basic usage:
    pt = BiphasicPulseTrain(freq, amp, phase_dur,
                            interphase_dur=interphase_dur, delay_dur=delay_dur,
                            stim_dur=stim_dur, cathodic_first=cathodic_first)
    for i in range(n_pulses):
        t_win = i * window_dur
        npt.assert_almost_equal(pt[0, t_win - DT], 0)
        npt.assert_almost_equal(pt[0, t_win + mid_first_pulse], first_amp)
        if interphase_dur > 0:
            npt.assert_almost_equal(pt[0, t_win + mid_interphase], 0)
        npt.assert_almost_equal(pt[0, t_win + mid_second_pulse],
                                second_amp)
    npt.assert_almost_equal(pt.time[0], 0)
    npt.assert_almost_equal(pt.time[-1], stim_dur, decimal=2)
    npt.assert_equal(pt.cathodic_first, cathodic_first)
    npt.assert_equal(pt.is_charge_balanced,
                     np.isclose(trapezoid(pt.data, pt.time)[0], 0, atol=1e-5))

    # Zero frequency:
    pt = BiphasicPulseTrain(0, amp, phase_dur)
    npt.assert_almost_equal(pt.time, [0, 1000])
    npt.assert_almost_equal(pt.data, 0)
    # Zero amp:
    pt = BiphasicPulseTrain(freq, 0, phase_dur)
    npt.assert_almost_equal(pt.data, 0)

    # Pulse can fill the entire window (no "unique time points" error):
    pt = BiphasicPulseTrain(10, 20, 50, stim_dur=500)
    npt.assert_almost_equal(pt.time[-1], 500)
    npt.assert_equal(np.round(trapezoid(np.abs(pt.data), pt.time)[0]), 10000)

    # Specific number of pulses
    for n_pulses in [2, 4, 5]:
        pt = BiphasicPulseTrain(500, 30, 0.05, n_pulses=n_pulses, stim_dur=19)
        npt.assert_almost_equal(np.sum(np.isclose(pt.data, 30)), 2 * n_pulses)
        npt.assert_almost_equal(pt.time[-1], 19)


@pytest.mark.parametrize('amp1', (-1, 13))
@pytest.mark.parametrize('amp2', (4, -8))
@pytest.mark.parametrize('interphase_dur', (0, 1))
@pytest.mark.parametrize('delay_dur', (0, 6))
@pytest.mark.parametrize('cathodic_first', (True, False))
def test_AsymmetricBiphasicPulseTrain(amp1, amp2, interphase_dur, delay_dur,
                                      cathodic_first):
    freq = 23.456
    phase_dur1 = 2
    phase_dur2 = 4
    stim_dur = 876.311
    window_dur = 1000.0 / freq
    n_pulses = int(freq * stim_dur / 1000.0)
    mid_first_pulse = delay_dur + phase_dur1 / 2
    mid_interphase = delay_dur + phase_dur1 + interphase_dur / 2
    mid_second_pulse = delay_dur + phase_dur1 + interphase_dur + phase_dur2 / 2
    first_amp = -np.abs(amp1) if cathodic_first else np.abs(amp1)
    second_amp = np.abs(amp2) if cathodic_first else -np.abs(amp2)

    # Basic usage:
    pt = AsymmetricBiphasicPulseTrain(freq, amp1, amp2, phase_dur1, phase_dur2,
                                      interphase_dur=interphase_dur,
                                      delay_dur=delay_dur, stim_dur=stim_dur,
                                      cathodic_first=cathodic_first)
    for i in range(n_pulses):
        t_win = i * window_dur
        npt.assert_almost_equal(pt[0, t_win - DT], 0)
        npt.assert_almost_equal(pt[0, t_win + mid_first_pulse], first_amp)
        if interphase_dur > 0:
            npt.assert_almost_equal(pt[0, t_win + mid_interphase], 0)
        npt.assert_almost_equal(pt[0, t_win + mid_second_pulse], second_amp)
    npt.assert_almost_equal(pt.time[0], 0)
    npt.assert_almost_equal(pt.time[-1], stim_dur, decimal=2)
    npt.assert_equal(pt.cathodic_first, cathodic_first)
    npt.assert_equal(pt.is_charge_balanced,
                     np.isclose(trapezoid(pt.data, pt.time)[0], 0, atol=1e-5))

    # Zero frequency:
    pt = AsymmetricBiphasicPulseTrain(0, amp1, amp2, phase_dur1, phase_dur2)
    npt.assert_almost_equal(pt.time, [0, 1000])
    npt.assert_almost_equal(pt.data, [[0, 0]])
    # Zero amp:
    pt = AsymmetricBiphasicPulseTrain(freq, 0, 0, phase_dur1, phase_dur2)
    npt.assert_almost_equal(pt.data, 0)

    # Pulse can fill the entire window (no "unique time points" error):
    pt = AsymmetricBiphasicPulseTrain(10, 40, 10, 20, 80, stim_dur=500)
    npt.assert_almost_equal(pt.time[-1], 500)
    npt.assert_equal(np.round(trapezoid(np.abs(pt.data), pt.time)[0]), 8000)

    # Specific number of pulses
    for n_pulses in [2, 4, 5]:
        pt = AsymmetricBiphasicPulseTrain(500, -30, 40, 0.05, 0.05,
                                          n_pulses=n_pulses, stim_dur=19)
        npt.assert_almost_equal(np.sum(np.isclose(pt.data, 40)), 2 * n_pulses)
        npt.assert_almost_equal(pt.time[-1], 19)


@pytest.mark.parametrize('amp', (-3, 4))
@pytest.mark.parametrize('interphase_dur', (0, 1))
@pytest.mark.parametrize('interpulse_dur', (0, 1))
@pytest.mark.parametrize('delay_dur', (4, 0))
@pytest.mark.parametrize('cathodic_first', (True, False))
def test_BiphasicTripletTrain(amp, interphase_dur, interpulse_dur, delay_dur, cathodic_first):
    freq = 23.456
    stim_dur = 657.456
    phase_dur = 2
    window_dur = 1000.0 / freq
    n_pulses = int(freq * stim_dur / 1000.0)
    mid_first_pulse = delay_dur + phase_dur / 2.0
    mid_interphase = delay_dur + phase_dur + interphase_dur / 2.0
    mid_second_pulse = delay_dur + interphase_dur + 1.5 * phase_dur
    mid_interpulse = delay_dur + 2.0*phase_dur + \
        interphase_dur + interpulse_dur / 2.0
    first_amp = -np.abs(amp) if cathodic_first else np.abs(amp)
    second_amp = -first_amp

    # Basic usage:
    pt = BiphasicTripletTrain(freq, amp, phase_dur,
                              interphase_dur=interphase_dur,
                              interpulse_dur=interpulse_dur,
                              delay_dur=delay_dur, stim_dur=stim_dur,
                              cathodic_first=cathodic_first)
    for i in range(n_pulses):
        t_win = i * window_dur
        npt.assert_almost_equal(pt[0, np.floor(t_win)], 0)
        npt.assert_almost_equal(pt[0, t_win + mid_first_pulse], first_amp)
        if interphase_dur > 0:
            npt.assert_almost_equal(pt[0, t_win + mid_interphase], 0)
        npt.assert_almost_equal(pt[0, t_win + mid_second_pulse], second_amp)
        if interpulse_dur > 0:
            npt.assert_almost_equal(pt[0, mid_interpulse], 0)
    npt.assert_almost_equal(pt.time[0], 0)
    npt.assert_almost_equal(pt.time[-1], stim_dur, decimal=2)
    npt.assert_equal(pt.cathodic_first, cathodic_first)
    npt.assert_equal(pt.is_charge_balanced,
                     np.isclose(trapezoid(pt.data, pt.time)[0], 0, atol=1e-5))

    # Zero frequency:
    pt = BiphasicPulseTrain(0, amp, phase_dur)
    npt.assert_almost_equal(pt.time, [0, 1000])
    npt.assert_almost_equal(pt.data, 0)
    # Zero amp:
    pt = BiphasicPulseTrain(freq, 0, phase_dur)
    npt.assert_almost_equal(pt.data, 0)

    # Pulse can fill the entire window (no "unique time points" error):
    pt = BiphasicTripletTrain(10, 20, 100 / 6.001, stim_dur=500)
    npt.assert_almost_equal(pt.time[-1], 500)
    npt.assert_equal(np.round(trapezoid(np.abs(pt.data), pt.time)[0]), 9998)

    # Specific number of pulses
    for n_pulses in [2, 4, 5]:
        pt = BiphasicPulseTrain(500, 30, 0.05, n_pulses=n_pulses, stim_dur=19)
        npt.assert_almost_equal(np.sum(np.isclose(pt.data, 30)), 2 * n_pulses)
        npt.assert_almost_equal(pt.time[-1], 19)

# Test metadata collecting


def test_metadata():
    # A train stores what the user put there and nothing else:
    stim = BiphasicPulseTrain(10, 10, 1, metadata='userdata')
    npt.assert_equal(stim.metadata, {'user': 'userdata'})

    stim = Stimulus({'A2': BiphasicPulseTrain(10, 10, 1, metadata='userdataA2'),
                     'B1': BiphasicPulseTrain(11, 9, 2, metadata='userdataB1'),
                     'C3': BiphasicPulseTrain(12, 8, 3, metadata='userdataC3')}, metadata='stimulus_userdata')
    npt.assert_equal(stim.metadata['user'], 'stimulus_userdata')
    sources = dict(stim._structured_sources())
    npt.assert_equal(type(sources['A2']), BiphasicPulseTrain)
    npt.assert_equal(sources['B1'].freq, 11)
    npt.assert_equal(sources['C3'].metadata['user'], 'userdataC3')


@pytest.mark.parametrize('scale', [2, 0.5, 1, 0])
def test_BiphasicPulseTrain_scaling(scale):
    # A model reads `amp` off the train, so scaling the data has to scale the
    # parameter with it:
    pt = BiphasicPulseTrain(20, 10, 0.45, stim_dur=100)
    scaled = pt * scale
    npt.assert_almost_equal(scaled.amp, 10 * scale)
    npt.assert_almost_equal(scaled.data, scale * pt.data)
    # Same data as the pulse train built that way directly:
    direct = BiphasicPulseTrain(20, 10 * scale, 0.45, stim_dur=100)
    npt.assert_almost_equal(scaled.data, direct.data)
    # Every route to a scaled train agrees, and none of them touches the
    # original:
    npt.assert_almost_equal((scale * pt).amp, 10 * scale)
    if scale != 0:
        npt.assert_almost_equal((pt / (1 / scale)).amp, 10 * scale)
    npt.assert_equal(pt.amp, 10)
    # The other pulse parameters are untouched:
    for name in ('freq', 'phase_dur', 'delay_dur'):
        npt.assert_equal(getattr(scaled, name), getattr(pt, name))


def test_BiphasicPulseTrain_amp_is_a_magnitude():
    # `BiphasicPulse` takes the magnitude of `amp` and reads the polarity off
    # `cathodic_first`, so the sign of `amp` never reaches the data
    pos = BiphasicPulseTrain(20, 10, 0.45, stim_dur=100)
    neg = BiphasicPulseTrain(20, -10, 0.45, stim_dur=100)
    npt.assert_almost_equal(pos.data, neg.data)
    npt.assert_equal(pos.amp, neg.amp)
    npt.assert_equal(pos.amp, 10)
    npt.assert_equal((neg * 2).amp, 20)


def test_BiphasicPulseTrain_polarity():
    # A negative factor swaps the two phases, which is what `cathodic_first`
    # records; the amplitude keeps its magnitude:
    pt = BiphasicPulseTrain(20, 10, 0.45, stim_dur=100)
    for flipped in (-pt, pt * -1, 0 - pt, pt / -1):
        npt.assert_almost_equal(flipped.data, -pt.data)
        npt.assert_equal(flipped.amp, 10)
        npt.assert_equal(type(flipped), BiphasicPulseTrain)
        npt.assert_equal(flipped.cathodic_first, not pt.cathodic_first)
    # Flipping twice comes back to where it started:
    npt.assert_equal((-(-pt)).cathodic_first, pt.cathodic_first)
    # A flipped train is the train that was built the other way round:
    direct = BiphasicPulseTrain(20, 10, 0.45, stim_dur=100,
                                cathodic_first=False)
    npt.assert_almost_equal((-pt).data, direct.data)
    npt.assert_equal((-pt).cathodic_first, direct.cathodic_first)


@pytest.mark.parametrize('modify', [lambda pt: pt + 5, lambda pt: pt - 5,
                                    lambda pt: 5 - pt, lambda pt: 5 + pt,
                                    lambda pt: (pt + 5) * 2,
                                    lambda pt: pt * np.inf,
                                    lambda pt: pt * np.nan,
                                    lambda pt: pt / 0])
def test_BiphasicPulseTrain_stops_being_a_train(modify):
    # A DC offset is neither biphasic nor charge-balanced, and a non-finite
    # factor leaves a waveform of infinities. Neither is a pulse train, so
    # neither may go on advertising one to a model:
    pt = BiphasicPulseTrain(20, 10, 0.45, stim_dur=100, metadata='userdata')
    with np.errstate(divide='ignore', invalid='ignore'):
        modified = modify(pt)
    npt.assert_equal(type(modified), Stimulus)
    npt.assert_equal(modified._structured_sources(), None)
    # The user's own metadata is theirs, and survives:
    npt.assert_equal(modified.metadata['user'], 'userdata')
    npt.assert_equal(pt.amp, 10)


def test_BiphasicPulseTrain_shift():
    pt = BiphasicPulseTrain(20, 10, 0.45, stim_dur=100, metadata='userdata')
    for shifted in (pt >> 5, pt << 5):
        npt.assert_equal(shifted.metadata['user'], 'userdata')
        # A shifted train no longer ends where its `stim_dur` says, so what
        # comes back is a plain Stimulus (see `Stimulus._derived`):
        npt.assert_equal(type(shifted), Stimulus)
    # Adding zero changes nothing at all, so what comes back is still a train:
    for same in (pt + 0, pt - 0):
        npt.assert_equal(same.metadata['user'], 'userdata')
        npt.assert_equal(type(same), BiphasicPulseTrain)
        npt.assert_equal(same.amp, pt.amp)
        npt.assert_equal(same.cathodic_first, pt.cathodic_first)
    npt.assert_almost_equal((pt >> 5).time, pt.time + 5)


def test_Stimulus_operators_leave_user_metadata_alone():
    # What the user filed is theirs, whatever the operator does to the data:
    stim = Stimulus(np.ones((2, 3)), metadata={'note': 'mine'})
    for modified in (stim * 2, stim + 5, -stim, stim >> 1):
        npt.assert_equal(modified.metadata['user'], {'note': 'mine'})


def test_Stimulus_collection_of_mixed_sources():
    # An implant's stimulus need not be all pulse trains, and the entries that
    # are keep their identity next to the ones that are not:
    stim = Stimulus({'A1': BiphasicPulseTrain(20, 10, 0.45, stim_dur=100),
                     'B2': Stimulus([[1, 2, 3]], time=[0, 1, 2],
                                    metadata={'note': 'mine'})})
    sources = dict(stim._structured_sources())
    npt.assert_equal(type(sources['A1']), BiphasicPulseTrain)
    npt.assert_equal(type(sources['B2']), Stimulus)
    npt.assert_equal(sources['B2'].metadata['user'], {'note': 'mine'})
    # Scaling scales the train and leaves the plain entry describing itself:
    scaled = dict((stim * 3)._structured_sources())
    npt.assert_almost_equal(scaled['A1'].amp, 30)
    npt.assert_equal(scaled['B2'].metadata['user'], {'note': 'mine'})


@pytest.mark.parametrize('freq, phase_dur, interphase_dur',
                         [(20, 0.45, 0), (100, 0.45, 0.2), (13, 0.1, 0),
                          (225, 0.075, 0.075), (2000, 0.1, 0)])
def test_PulseTrain_tiling(freq, phase_dur, interphase_dur):
    # The pulse train is assembled by tiling rather than by repeatedly calling
    # `append`
    pulse = BiphasicPulse(20, phase_dur, interphase_dur=interphase_dur)
    n_pulses = 7
    window_dur = 1000.0 / freq
    shift = np.maximum(0, window_dur - pulse.time[-1])

    # Reference: the original implementation
    ref = pulse
    for _ in range(1, n_pulses):
        ref = ref.append(pulse >> shift)

    data, time = _tile_pulse(pulse, shift, n_pulses)
    npt.assert_array_equal(data, ref.data)
    npt.assert_array_equal(time, ref.time)
    npt.assert_equal(data.dtype, ref.data.dtype)
    npt.assert_equal(time.dtype, ref.time.dtype)

    # A single pulse must come out unchanged:
    data, time = _tile_pulse(pulse, shift, 1)
    npt.assert_array_equal(data, pulse.data)
    npt.assert_array_equal(time, pulse.time)


def test_PulseTrain_tiling_errors():
    # A pulse whose first and last sample differ cannot be tiled without a gap
    # between the copies (the junction points would have to be merged):
    pulse = Stimulus([[1, 2, 3]], time=[0, 0.5, 1.0])
    with pytest.raises(ValueError):
        _tile_pulse(pulse, 0.0, 3)
    # ...but with a gap it is fine:
    data, time = _tile_pulse(pulse, 5.0, 3)
    npt.assert_equal(data.shape, (1, 9))
    # A negative time axis is not supported:
    with pytest.raises(NotImplementedError):
        _tile_pulse(Stimulus([[0, 5, 0]], time=[-2.0, 0.0, 4.0]), 0.0, 3)


@pytest.mark.parametrize('cls, args, kwargs', [
    (PulseTrain, (20, BiphasicPulse(30, 0.45)), {}),
    (BiphasicPulseTrain, (20, 30, 0.45), {}),
    (AsymmetricBiphasicPulseTrain, (20, -40, 10, 1, 4), {}),
    (BiphasicTripletTrain, (20, 30, 0.45), {}),
    (BiphasicTripletTrain, (20, 30, 0.45), {'interpulse_dur': 0.5}),
])
def test_PulseTrain_electrode_name(cls, args, kwargs):
    stim = cls(*args, stim_dur=100, electrode='A1', **kwargs)
    npt.assert_equal(stim.electrodes, ['A1'])
    # Without a name, electrodes are still numbered from 0:
    stim = cls(*args, stim_dur=100, **kwargs)
    npt.assert_equal(stim.electrodes, [0])
    # And the name has to survive the trip through Stimulus():
    stim = Stimulus(cls(*args, stim_dur=100, electrode='A1', **kwargs))
    npt.assert_equal(stim.electrodes, ['A1'])


def test_pulse_train_units():
    """Equivalent unit choices must produce numerically identical trains"""
    pairs = [
        (BiphasicPulseTrain(20, 50, 0.45, interphase_dur=0.2, delay_dur=1,
                            stim_dur=200),
         BiphasicPulseTrain(0.02 * kHz, 0.05 * mA, 450 * us,
                            interphase_dur=200 * us, delay_dur=1 * ms,
                            stim_dur=0.2 * sec)),
        (AsymmetricBiphasicPulseTrain(20, -40, 10, 1, 4, stim_dur=200),
         AsymmetricBiphasicPulseTrain(20 * Hz, -0.04 * mA, 10 * uA, 1 * ms,
                                      4000 * us, stim_dur=0.2 * sec)),
        (BiphasicTripletTrain(20, 50, 0.45, interpulse_dur=1, stim_dur=200),
         BiphasicTripletTrain(0.02 * kHz, 0.05 * mA, 450 * us,
                              interpulse_dur=1000 * us, stim_dur=0.2 * sec)),
    ]
    for bare, unitful in pairs:
        npt.assert_array_equal(bare.data, unitful.data)
        npt.assert_array_equal(bare.time, unitful.time)
        npt.assert_equal(bare == unitful, True)
        npt.assert_equal(unitful.unit, uA)
        npt.assert_equal(unitful.time_unit, ms)
    # A generic PulseTrain takes its frequency and duration the same way:
    pulse = BiphasicPulse(50, 0.45)
    bare = PulseTrain(20, pulse, stim_dur=200)
    unitful = PulseTrain(0.02 * kHz, pulse, stim_dur=0.2 * sec)
    npt.assert_array_equal(bare.data, unitful.data)
    npt.assert_array_equal(bare.time, unitful.time)
    # Dimensional errors:
    with pytest.raises(DimensionMismatchError) as excinfo:
        BiphasicPulseTrain(20 * ms, 50, 0.45)
    npt.assert_equal("Parameter 'freq' expects frequency (Hz), got time"
                     in str(excinfo.value), True)
    with pytest.raises(DimensionMismatchError):
        BiphasicPulseTrain(20, 50, 0.45, stim_dur=1 * uA)
    with pytest.raises(DimensionMismatchError):
        BiphasicTripletTrain(20, 50, 0.45, interpulse_dur=1 * uA)
    with pytest.raises(DimensionMismatchError):
        PulseTrain(20 * uA, pulse)


def test_pulse_train_parameter_units():
    """A train reports plain numbers in its historical units

    BiphasicAxonMapModel and DynaphosModel read amplitude, frequency and phase
    duration off the train and feed them straight into their equations, so a
    Quantity coming back would break them.
    """
    train = BiphasicPulseTrain(0.02 * kHz, 0.05 * mA, 450 * us,
                               delay_dur=100 * us, stim_dur=0.2 * sec)
    for name, value in (('freq', 20), ('amp', 50), ('phase_dur', 0.45),
                        ('delay_dur', 0.1)):
        npt.assert_almost_equal(getattr(train, name), value)
        npt.assert_equal(isinstance(getattr(train, name), Quantity), False)
        npt.assert_equal(np.isscalar(getattr(train, name)), True)
    npt.assert_equal(isinstance(PulseTrain(0.02 * kHz,
                                           BiphasicPulse(50, 0.45)).freq,
                                Quantity), False)
    # And scaling still rewrites the amplitude it advertises:
    npt.assert_almost_equal((train * 2).amp, 100)


def test_PulseTrain_unit_provenance():
    """A generic PulseTrain is measured in whatever it tiled"""
    electrical = PulseTrain(20, BiphasicPulse(50, 0.45), stim_dur=200)
    npt.assert_equal(electrical.unit, uA)
    npt.assert_equal(electrical.time_unit, ms)
    # A dimensionless temporal stimulus stays dimensionless: tiling gray
    # levels does not turn them into a current.
    source = Stimulus(VideoStimulus(np.ones((1, 1, 5)),
                                    time=[0, 1, 2, 3, 4]))
    npt.assert_equal(source.unit, dimensionless)
    train = PulseTrain(20, source, stim_dur=200)
    npt.assert_equal(train.unit, dimensionless)
    npt.assert_equal(train.time_unit, ms)
    # A silent train is all zeros, but the zeros still mean whatever the
    # source pulse measured:
    for pulse, unit in [(BiphasicPulse(50, 0.45), uA), (source, dimensionless)]:
        silent = PulseTrain(0, pulse, stim_dur=200)
        npt.assert_almost_equal(silent.data, 0)
        npt.assert_equal(silent.unit, unit)
        npt.assert_equal(silent.time_unit, ms)
    # The specialized trains build their own electrical pulses, so they are
    # microamps whatever happens:
    for train in (BiphasicPulseTrain(20, 50, 0.45, stim_dur=200),
                  BiphasicTripletTrain(20, 50, 0.45, stim_dur=200),
                  AsymmetricBiphasicPulseTrain(20, -40, 10, 1, 4,
                                               stim_dur=200)):
        npt.assert_equal(train.unit, uA)


def _rendered(stim):
    """Whether the stimulus has generated its waveform yet

    Reads the private container, because every public attribute that could
    answer the question would generate one first.
    """
    return stim._Stimulus__stim['data'] is not None


TRAINS = [
    (PulseTrain,
     lambda: PulseTrain(20, BiphasicPulse(50, 0.45), stim_dur=30000),
     {'freq': 20, 'n_pulses': 600, 'stim_dur': 30000,
      'pulse_type': 'BiphasicPulse'}),
    (BiphasicPulseTrain,
     lambda: BiphasicPulseTrain(20, 50, 0.45, interphase_dur=0.2,
                                delay_dur=1, stim_dur=30000,
                                cathodic_first=False),
     {'freq': 20, 'amp': 50, 'phase_dur': 0.45, 'interphase_dur': 0.2,
      'delay_dur': 1, 'cathodic_first': False, 'n_pulses': 600,
      'stim_dur': 30000}),
    (AsymmetricBiphasicPulseTrain,
     lambda: AsymmetricBiphasicPulseTrain(20, 50, 20, 0.45, 0.9,
                                          stim_dur=30000),
     {'freq': 20, 'amp1': 50, 'amp2': 20, 'phase_dur1': 0.45,
      'phase_dur2': 0.9, 'interphase_dur': 0, 'delay_dur': 0,
      'cathodic_first': True, 'n_pulses': 600, 'stim_dur': 30000}),
    (BiphasicTripletTrain,
     lambda: BiphasicTripletTrain(20, 50, 0.45, interpulse_dur=1,
                                  stim_dur=30000),
     {'freq': 20, 'amp': 50, 'phase_dur': 0.45, 'interphase_dur': 0,
      'interpulse_dur': 1, 'delay_dur': 0, 'cathodic_first': True,
      'n_pulses': 600, 'stim_dur': 30000}),
]


@pytest.mark.parametrize('cls, build, params', TRAINS)
def test_pulse_train_parameters_are_canonical(cls, build, params):
    train = build()
    for name, expected in params.items():
        npt.assert_equal(getattr(train, name), expected)
    # `duration` is one of them: `stim_dur` already says where the train ends,
    # so asking must not tile 600 pulses to find out.
    npt.assert_almost_equal(train.duration, params['stim_dur'])
    npt.assert_equal(_rendered(train), False)
    npt.assert_almost_equal(train.time[-1], train.duration, decimal=3)


@pytest.mark.parametrize('cls, build, params', TRAINS)
def test_pulse_train_parameters_are_read_only(cls, build, params):
    train = build()
    for name in params:
        with pytest.raises(AttributeError):
            setattr(train, name, 1)


@pytest.mark.parametrize('cls, build, params', TRAINS)
def test_pulse_train_renders_once_and_only_when_asked(cls, build, params):
    # A 30-second train delivers 600 pulses; not one of them is sampled until
    # something asks for a waveform.
    train = build()
    npt.assert_equal(_rendered(train), False)
    for name in params:
        getattr(train, name)
    npt.assert_equal(len(train.electrodes), 1)
    repr(train)
    npt.assert_equal(_rendered(train), False)
    first = train.data
    npt.assert_equal(_rendered(train), True)
    for _ in range(3):
        npt.assert_equal(np.shares_memory(train.data, first), True)
    npt.assert_equal(train.data.flags.writeable, False)
    npt.assert_equal(train.data.dtype, np.float32)
    npt.assert_equal(train.time.dtype, np.float64)


def test_PulseTrain_snapshots_its_pulse():
    # Tiling used to copy the pulse values into the train there and then, so a
    # train must not change when the caller replaces the pulse it was built
    # from
    pulse = BiphasicPulse(50, 0.45)
    train = PulseTrain(20, pulse, stim_dur=100)
    npt.assert_equal(train.pulse is pulse, False)
    npt.assert_equal(train.pulse.amp, pulse.amp)
    npt.assert_equal(train.pulse_type, 'BiphasicPulse')


def test_PulseTrain_pulse_is_not_a_way_into_the_train():
    # `remove` and `compress` rewrite a stimulus in place
    pulse = Stimulus([[0, -50, 50, 0]], electrodes=['A1'],
                     time=[0, 0.1, 0.2, 0.3])
    train = PulseTrain(20, pulse, stim_dur=100)
    train.pulse.remove('all')
    npt.assert_equal(len(train.pulse.electrodes), 1)
    npt.assert_equal(train.data.shape[0], 1)
    # A copy renders from a pulse of its own, too:
    copied = deepcopy(train)
    copied._pulse.compress()
    npt.assert_equal(train._pulse.is_compressed, False)


@pytest.mark.parametrize('factor', [2, 0.5, -1, -2, 1, 0, 1e-3])
def test_BiphasicPulseTrain_scaling_rebuilds_the_train(factor):
    pt = BiphasicPulseTrain(20, 10, 0.45, interphase_dur=0.2, delay_dur=1,
                            stim_dur=100, metadata='userdata')
    routes = [pt * factor, factor * pt]
    if factor:
        routes.append(pt / (1 / factor))
    for scaled in routes:
        npt.assert_equal(type(scaled), BiphasicPulseTrain)
        npt.assert_almost_equal(scaled.amp, pt.amp * abs(factor))
        npt.assert_equal(scaled.cathodic_first,
                         pt.cathodic_first if factor >= 0
                         else not pt.cathodic_first)
        # Everything else is untouched:
        for name in ('freq', 'phase_dur', 'interphase_dur', 'delay_dur',
                     'n_pulses', 'stim_dur'):
            npt.assert_almost_equal(getattr(scaled, name), getattr(pt, name))
        # What the user put in the metadata is theirs, and comes along:
        npt.assert_equal(scaled.metadata['user'], 'userdata')
        # ...and the waveform is the scaled one, to within float32 rounding:
        npt.assert_allclose(scaled.data, factor * pt.data, rtol=1e-6,
                            atol=1e-6)
    # The original is untouched:
    npt.assert_equal(pt.amp, 10)
    npt.assert_equal(pt.cathodic_first, True)


def test_BiphasicPulseTrain_scaling_matches_a_train_built_that_way():
    pt = BiphasicPulseTrain(20, 10, 0.45, stim_dur=100)
    direct = BiphasicPulseTrain(20, 20, 0.45, stim_dur=100)
    npt.assert_array_equal((pt * 2).data, direct.data)
    npt.assert_array_equal((pt * 2).time, direct.time)
    npt.assert_equal((pt * 2).metadata, direct.metadata)
    flipped = BiphasicPulseTrain(20, 10, 0.45, stim_dur=100,
                                 cathodic_first=False)
    npt.assert_array_equal((-pt).data, flipped.data)


@pytest.mark.parametrize('modify', [lambda pt: pt + 5, lambda pt: pt - 5,
                                    lambda pt: 5 - pt, lambda pt: pt >> 5,
                                    lambda pt: pt << 1,
                                    lambda pt: pt * np.inf,
                                    lambda pt: pt * np.nan,
                                    lambda pt: pt / 0])
def test_BiphasicPulseTrain_inexact_operations_degrade(modify):
    pt = BiphasicPulseTrain(20, 10, 0.45, stim_dur=100)
    with np.errstate(divide='ignore', invalid='ignore'):
        modified = modify(pt)
    npt.assert_equal(type(modified), Stimulus)
    npt.assert_equal(modified._is_parametric, False)


@pytest.mark.parametrize('cls, build, params', TRAINS)
@pytest.mark.parametrize('factor', [2, 0.5, -1, -2, 1, 0, 1e-3])
def test_train_scaling_stays_a_train(cls, build, params, factor):
    train = build()
    reference = factor * np.asarray(train.data)
    for scaled in (train * factor, factor * train):
        npt.assert_equal(type(scaled), cls)
        npt.assert_equal(_rendered(scaled), False)
        npt.assert_allclose(scaled.data, reference, rtol=1e-6, atol=1e-6)
        # The train's own clock is untouched by its amplitude:
        for name in ('freq', 'n_pulses', 'stim_dur', 'phase_dur',
                     'phase_dur1', 'phase_dur2', 'interphase_dur',
                     'interpulse_dur', 'delay_dur'):
            if name in params:
                npt.assert_almost_equal(getattr(scaled, name), params[name])
        for name in ('amp', 'amp1', 'amp2'):
            if name in params:
                npt.assert_almost_equal(getattr(scaled, name),
                                        params[name] * abs(factor))
        if 'cathodic_first' in params:
            npt.assert_equal(scaled.cathodic_first,
                             params['cathodic_first'] if factor >= 0
                             else not params['cathodic_first'])
    # The original is untouched:
    for name, expected in params.items():
        npt.assert_equal(getattr(train, name), expected)


def test_PulseTrain_scaling_keeps_the_pulse_it_repeats():
    # A generic train scales the pulse it tiles rather than the tiled samples
    train = PulseTrain(20, BiphasicPulse(50, 0.45), stim_dur=200)
    scaled = train * 2
    npt.assert_equal(scaled.pulse_type, 'BiphasicPulse')
    npt.assert_almost_equal(scaled.pulse.amp, 100)
    npt.assert_almost_equal(train.pulse.amp, 50)


def test_train_scaling_survives_a_partial_last_window():
    train = BiphasicPulseTrain(23, 20, 0.45, stim_dur=100)
    npt.assert_equal(train.n_pulses, 3)
    npt.assert_equal((train * 2).n_pulses, 3)
    npt.assert_allclose((train * 2).data, 2 * train.data, rtol=1e-6,
                        atol=1e-6)
    # An explicitly requested count still comes back unchanged:
    asked = BiphasicPulseTrain(23, 20, 0.45, n_pulses=2, stim_dur=100)
    npt.assert_equal((asked * 2).n_pulses, 2)


def test_append_gives_a_plain_waveform():
    # No single frequency describes a 20 Hz train followed by a 50 Hz one, so
    # the result stops claiming to be a train at all.
    pt20 = BiphasicPulseTrain(20, 10, 0.45, stim_dur=100, metadata='mine')
    pt50 = BiphasicPulseTrain(50, 20, 0.45, stim_dur=100)
    out = pt20.append(pt50)
    npt.assert_equal(type(out), Stimulus)
    npt.assert_equal(hasattr(out, 'freq'), False)
    # ...and it is the concatenation the waveforms alone would have produced:
    plain20 = Stimulus(pt20.data, electrodes=pt20.electrodes, time=pt20.time)
    plain50 = Stimulus(pt50.data, electrodes=pt50.electrodes, time=pt50.time)
    expected = plain20.append(plain50)
    npt.assert_array_equal(out.data, expected.data)
    npt.assert_array_equal(out.time, expected.time)
    npt.assert_almost_equal(out.duration, 200)


def test_append_still_rejects_what_it_always_did():
    pt = BiphasicPulseTrain(20, 10, 0.45, stim_dur=100)
    with pytest.raises(TypeError):
        pt.append(5)
    with pytest.raises(ValueError):
        # Different electrodes:
        pt.append(BiphasicPulseTrain(20, 10, 0.45, stim_dur=100,
                                     electrode='A1'))
    with pytest.raises(ValueError):
        # `other` starts at t=0 with an amplitude this one does not end on:
        pt.append(Stimulus([[5, 5]], electrodes=pt.electrodes, time=[0, 10]))
    with pytest.raises(DimensionMismatchError):
        pt.append(VideoStimulus(np.ones((1, 1, 3))))


@pytest.mark.parametrize('amp, threshold_amp, expected',
                         [(50, None, (50, None, None, uA)),
                          (0.05 * mA, None, (50, None, None, uA)),
                          (2 * xTh, None, (2, None, 2, xTh)),
                          (2 * xTh, 80 * uA, (160, 80, 2, uA)),
                          (160 * uA, 80 * uA, (160, 80, 2, uA)),
                          (2 * xTh, 80, (160, 80, 2, uA))])
def test_BiphasicPulseTrain_threshold_relative_amp(amp, threshold_amp,
                                                   expected):
    pt = BiphasicPulseTrain(20, amp, 0.45, stim_dur=100,
                            threshold_amp=threshold_amp)
    npt.assert_almost_equal(pt.amp, expected[0])
    npt.assert_equal(pt.threshold_amp, expected[1])
    npt.assert_equal(pt.amp_factor, expected[2])
    # A threshold multiple only becomes a current once a threshold says so:
    npt.assert_equal(pt.unit, expected[3])
    npt.assert_almost_equal(np.abs(pt.data).max(), expected[0], decimal=3)


def test_BiphasicPulseTrain_threshold_amp_is_validated():
    for bad in (0, -80, np.nan, np.inf):
        with pytest.raises(ValueError):
            BiphasicPulseTrain(20, 2 * xTh, 0.45, threshold_amp=bad)
    for bad in (2 * xTh, 5 * ms):
        with pytest.raises(DimensionMismatchError):
            BiphasicPulseTrain(20, 50, 0.45, threshold_amp=bad)


def test_BiphasicPulseTrain_scaling_preserves_amp_basis():
    relative = BiphasicPulseTrain(20, 2 * xTh, 0.45, stim_dur=100,
                                  threshold_amp=80 * uA)
    npt.assert_almost_equal((relative * 2).amp_factor, 4)
    npt.assert_almost_equal((relative * 2).amp, 320)
    npt.assert_equal((relative * 2)._amp_relative, True)
    current = BiphasicPulseTrain(20, 160 * uA, 0.45, stim_dur=100)
    npt.assert_almost_equal((current * 2).amp, 320)
    npt.assert_equal((current * 2).amp_factor, None)
    npt.assert_equal((current * 2)._amp_relative, False)
    npt.assert_equal((-relative).cathodic_first, not relative.cathodic_first)
    npt.assert_almost_equal((-relative).amp_factor, 2)
    npt.assert_almost_equal((-current).amp, 160)


def test_BiphasicPulseTrain_amp_basis_survives_a_round_trip():
    pt = BiphasicPulseTrain(20, 2 * xTh, 0.45, stim_dur=100,
                            threshold_amp=80 * uA)
    for copied in (deepcopy(pt), pt * 1):
        npt.assert_equal(copied._amp_relative, True)
        npt.assert_almost_equal(copied.threshold_amp, 80)
        npt.assert_almost_equal(copied.amp_factor, 2)


@pytest.mark.parametrize('amp, threshold_amp, overridden, restored',
                         [(2 * xTh, None, (200, 2), (2, 2)),
                          (2 * xTh, 50 * uA, (200, 2), (100, 2)),
                          (160 * uA, None, (160, 1.6), (160, None)),
                          (160 * uA, 50 * uA, (160, 1.6), (160, 3.2))])
def test_BiphasicPulseTrain_threshold_override(amp, threshold_amp, overridden,
                                               restored):
    pt = BiphasicPulseTrain(20, amp, 0.45, stim_dur=100,
                            threshold_amp=threshold_amp)
    calibrated = pt._with_threshold(100)
    npt.assert_almost_equal(calibrated.amp, overridden[0])
    npt.assert_almost_equal(calibrated.amp_factor, overridden[1])
    npt.assert_almost_equal(calibrated.threshold_amp, 100)
    cleared = calibrated._with_threshold(None)
    npt.assert_almost_equal(cleared.amp, restored[0])
    npt.assert_equal(cleared.amp_factor, restored[1])
    npt.assert_almost_equal(pt.amp, restored[0])
    # A no-op override hands back the same object rather than a rebuild:
    npt.assert_equal(pt._with_threshold(None) is pt, True)


def test_BiphasicPulseTrain_scaling_keeps_the_override():
    pt = BiphasicPulseTrain(20, 2 * xTh, 0.45, stim_dur=100,
                            threshold_amp=50 * uA)._with_threshold(100)
    scaled = pt * 2
    npt.assert_almost_equal(scaled.amp, 400)
    npt.assert_almost_equal(scaled.amp_factor, 4)
    # Clearing after scaling still restores the train's own threshold:
    npt.assert_almost_equal(scaled._with_threshold(None).amp, 200)
    current = BiphasicPulseTrain(20, 160 * uA, 0.45,
                                 stim_dur=100)._with_threshold(80)
    npt.assert_almost_equal((current * 2).amp, 320)
    npt.assert_almost_equal((current * 2).amp_factor, 4)
    npt.assert_equal((current * 2)._with_threshold(None).amp_factor, None)


def test_BiphasicPulseTrain_xTh_amp_is_not_a_current():
    # No threshold, no current: the train must not invent one.
    pt = BiphasicPulseTrain(20, 2 * xTh, 0.45, stim_dur=100)
    npt.assert_equal(pt.unit, xTh)
    npt.assert_almost_equal(pt.amp, 2)
    npt.assert_almost_equal(pt.amp_factor, 2)
    npt.assert_equal(pt.threshold_amp, None)
    npt.assert_almost_equal(np.abs(pt.data).max(), 2, decimal=3)
    calibrated = pt._with_threshold(80)
    npt.assert_equal(calibrated.unit, uA)
    npt.assert_almost_equal(calibrated.amp, 160)
    npt.assert_almost_equal(calibrated.amp_factor, 2)


def test_BiphasicPulseTrain_repr_keeps_the_amp_basis():
    # Same current, different basis, so different behavior under
    # recalibration -- the repr has to tell them apart:
    relative = repr(BiphasicPulseTrain(20, 2 * xTh, 0.45,
                                       threshold_amp=80 * uA))
    current = repr(BiphasicPulseTrain(20, 160 * uA, 0.45,
                                      threshold_amp=80 * uA))
    npt.assert_equal(relative == current, False)
    npt.assert_equal('xTh' in relative, True)
    npt.assert_equal('xTh' in current, False)
    for shown in (relative, current):
        npt.assert_equal('threshold_amp' in shown, True)
    npt.assert_equal('threshold_amp' in repr(BiphasicPulseTrain(20, 50, 0.45)),
                     False)
