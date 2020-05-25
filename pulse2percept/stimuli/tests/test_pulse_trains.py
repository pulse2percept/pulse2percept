import numpy as np
import copy
import pytest
import numpy.testing as npt

from pulse2percept.stimuli import (Stimulus, TimeSeries, PulseTrain,
                                   BiphasicPulseTrain, BiphasicTripletTrain,
                                   AsymmetricBiphasicPulseTrain)
from pulse2percept.stimuli.pulse_trains import LegacyPulseTrain


def test_PulseTrain():
    # All zeros:
    npt.assert_almost_equal(PulseTrain(10, Stimulus(np.zeros((1, 5)))).data,
                            0)
    # Simple fake pulse:
    pulse = Stimulus([[0, -1, 0]], time=[0, 0.1, 0.2])
    for n_pulses in [2, 3, 10]:
        pt = PulseTrain(10, pulse, n_pulses=n_pulses)
        npt.assert_equal(np.sum(np.isclose(pt.data, -1)), n_pulses)

    # stim_dur too short:
    npt.assert_almost_equal(PulseTrain(2, pulse, stim_dur=10).data, 0)

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


@pytest.mark.parametrize('amp', (-3, 4))
@pytest.mark.parametrize('interphase_dur', (0, 1))
@pytest.mark.parametrize('delay_dur', (0, 4))
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
        npt.assert_almost_equal(pt[0, t_win], 0)
        npt.assert_almost_equal(pt[0, t_win + mid_first_pulse], first_amp)
        npt.assert_almost_equal(pt[0, t_win + mid_interphase], 0)
        npt.assert_almost_equal(pt[0, t_win + mid_second_pulse], second_amp)
    npt.assert_almost_equal(pt.time[0], 0)
    npt.assert_almost_equal(pt.time[-1], stim_dur, decimal=2)
    npt.assert_equal(pt.cathodic_first, cathodic_first)
    npt.assert_equal(pt.charge_balanced,
                     np.isclose(np.trapz(pt.data, pt.time)[0], 0, atol=1e-5))

    # Zero frequency:
    pt = BiphasicPulseTrain(0, amp, phase_dur)
    npt.assert_almost_equal(pt.time, [0, 1000])
    npt.assert_almost_equal(pt.data, 0)
    # Zero amp:
    pt = BiphasicPulseTrain(freq, 0, phase_dur)
    npt.assert_almost_equal(pt.data, 0)

    # Specific number of pulses
    for n_pulses in [2, 4, 5]:
        pt = BiphasicPulseTrain(500, 30, 0.05, n_pulses=n_pulses, stim_dur=19,
                                dt=0.05)
        npt.assert_almost_equal(np.sum(np.isclose(pt.data, 30)), n_pulses)
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
        npt.assert_almost_equal(pt[0, t_win], 0)
        npt.assert_almost_equal(pt[0, t_win + mid_first_pulse], first_amp)
        npt.assert_almost_equal(pt[0, t_win + mid_interphase], 0)
        npt.assert_almost_equal(pt[0, t_win + mid_second_pulse], second_amp)
    npt.assert_almost_equal(pt.time[0], 0)
    npt.assert_almost_equal(pt.time[-1], stim_dur, decimal=2)
    npt.assert_equal(pt.cathodic_first, cathodic_first)
    npt.assert_equal(pt.charge_balanced,
                     np.isclose(np.trapz(pt.data, pt.time)[0], 0, atol=1e-5))

    # Zero frequency:
    pt = AsymmetricBiphasicPulseTrain(0, amp1, amp2, phase_dur1, phase_dur2)
    npt.assert_almost_equal(pt.time, [0, 1000])
    npt.assert_almost_equal(pt.data, [[0, 0]])
    # Zero amp:
    pt = AsymmetricBiphasicPulseTrain(freq, 0, 0, phase_dur1, phase_dur2)
    npt.assert_almost_equal(pt.data, 0)

    # Specific number of pulses
    for n_pulses in [2, 4, 5]:
        pt = AsymmetricBiphasicPulseTrain(500, -30, 40, 0.05, 0.05,
                                          n_pulses=n_pulses, stim_dur=19,
                                          dt=0.05)
        npt.assert_almost_equal(np.sum(np.isclose(pt.data, 40)), n_pulses)
        npt.assert_almost_equal(pt.time[-1], 19)


@pytest.mark.parametrize('amp', (-3, 4))
@pytest.mark.parametrize('interphase_dur', (0, 1))
@pytest.mark.parametrize('delay_dur', (4, 0))
@pytest.mark.parametrize('cathodic_first', (True, False))
def test_BiphasicTripletTrain(amp, interphase_dur, delay_dur, cathodic_first):
    freq = 23.456
    stim_dur = 657.456
    phase_dur = 2
    window_dur = 1000.0 / freq
    n_pulses = int(freq * stim_dur / 1000.0)
    dt = 1e-6
    mid_first_pulse = delay_dur + phase_dur / 2.0
    mid_interphase = delay_dur + phase_dur + interphase_dur / 2.0
    mid_second_pulse = delay_dur + interphase_dur + 1.5 * phase_dur
    first_amp = -np.abs(amp) if cathodic_first else np.abs(amp)
    second_amp = -first_amp

    # Basic usage:
    pt = BiphasicTripletTrain(freq, amp, phase_dur,
                              interphase_dur=interphase_dur,
                              delay_dur=delay_dur, stim_dur=stim_dur,
                              cathodic_first=cathodic_first, dt=dt)
    for i in range(n_pulses):
        t_win = i * window_dur
        npt.assert_almost_equal(pt[0, t_win], 0)
        npt.assert_almost_equal(pt[0, t_win + mid_first_pulse], first_amp)
        if interphase_dur > 0:
            npt.assert_almost_equal(pt[0, t_win + mid_interphase], 0)
        npt.assert_almost_equal(pt[0, t_win + mid_second_pulse], second_amp)
    npt.assert_almost_equal(pt.time[0], 0)
    npt.assert_almost_equal(pt.time[-1], stim_dur, decimal=2)
    npt.assert_equal(pt.cathodic_first, cathodic_first)
    npt.assert_equal(pt.charge_balanced,
                     np.isclose(np.trapz(pt.data, pt.time)[0], 0, atol=1e-5))

    # Zero frequency:
    pt = BiphasicPulseTrain(0, amp, phase_dur)
    npt.assert_almost_equal(pt.time, [0, 1000])
    npt.assert_almost_equal(pt.data, 0)
    # Zero amp:
    pt = BiphasicPulseTrain(freq, 0, phase_dur)
    npt.assert_almost_equal(pt.data, 0)

    # Specific number of pulses
    for n_pulses in [2, 4, 5]:
        pt = BiphasicPulseTrain(500, 30, 0.05, n_pulses=n_pulses, stim_dur=19,
                                dt=0.05)
        npt.assert_almost_equal(np.sum(np.isclose(pt.data, 30)), n_pulses)
        npt.assert_almost_equal(pt.time[-1], 19)


def test_TimeSeries():
    # Slots:
    npt.assert_equal(hasattr(TimeSeries(1, [1]), '__slots__'), True)
    npt.assert_equal(hasattr(TimeSeries(1, [1]), '__dict__'), False)

    max_val = 2.0
    max_idx = 156
    data_orig = np.random.rand(10, 10, 1000)
    data_orig[4, 4, max_idx] = max_val
    ts = TimeSeries(1.0, data_orig)

    # Make sure function returns largest element
    tmax, vmax = ts.max()
    npt.assert_equal(tmax, max_idx)
    npt.assert_equal(vmax, max_val)

    # Make sure function returns largest frame
    tmax, fmax = ts.max_frame()
    npt.assert_equal(tmax, max_idx)
    npt.assert_equal(fmax.data, data_orig[:, :, max_idx])

    # Make sure getitem works
    npt.assert_equal(isinstance(ts[3], TimeSeries), True)
    npt.assert_equal(ts[3].data, ts.data[3])


def test_TimeSeries_resample():
    max_val = 2.0
    max_idx = 156
    data_orig = np.random.rand(10, 10, 1000)
    data_orig[4, 4, max_idx] = max_val
    ts = TimeSeries(1.0, data_orig)
    tmax, vmax = ts.max()

    # Resampling with same sampling step shouldn't change anything
    ts_new = ts.resample(ts.tsample)
    npt.assert_equal(ts_new.shape, ts.shape)
    npt.assert_equal(ts_new.duration, ts.duration)

    # Make sure resampling works
    tsample_new = 4
    ts_new = ts.resample(tsample_new)
    npt.assert_equal(ts_new.tsample, tsample_new)
    npt.assert_equal(ts_new.data.shape[-1], ts.data.shape[-1] / tsample_new)
    npt.assert_equal(ts_new.duration, ts.duration)
    tmax_new, vmax_new = ts_new.max()
    npt.assert_equal(tmax_new, tmax)
    npt.assert_equal(vmax_new, vmax)

    # Make sure resampling leaves old data unaffected (deep copy)
    ts_new.data[0, 0, 0] = max_val * 2.0
    tmax_new, vmax_new = ts_new.max()
    npt.assert_equal(tmax_new, 0)
    npt.assert_equal(vmax_new, max_val * 2.0)
    tmax, vmax = ts.max()
    npt.assert_equal(tmax, max_idx)
    npt.assert_equal(vmax, max_val)


def test_TimeSeries_append():
    max_val = 2.0
    max_idx = 156
    data_orig = np.random.rand(10, 10, 1000)
    data_orig[4, 4, max_idx] = max_val
    ts_orig = TimeSeries(1.0, data_orig)

    # Make sure adding two TimeSeries objects works:
    # Must have the right type and size
    ts = copy.deepcopy(ts_orig)
    with pytest.raises(TypeError):
        ts.append(4.0)
    with pytest.raises(ValueError):
        ts_wrong_size = TimeSeries(1.0, np.ones((2, 2)))
        ts.append(ts_wrong_size)

    # Adding messes only with the last dimension of the array
    ts = copy.deepcopy(ts_orig)
    ts.append(ts)
    npt.assert_equal(ts.shape[: -1], ts_orig.shape[: -1])
    npt.assert_equal(ts.shape[-1], ts_orig.shape[-1] * 2)

    # If necessary, the second pulse train is resampled to the first
    ts = copy.deepcopy(ts_orig)
    tsample_new = 2.0
    ts_new = ts.resample(tsample_new)
    ts.append(ts_new)
    npt.assert_equal(ts.shape[: -1], ts_orig.shape[: -1])
    npt.assert_equal(ts.shape[-1], ts_orig.shape[-1] * 2)
    ts_add = copy.deepcopy(ts_new)
    ts_add.append(ts_orig)
    npt.assert_equal(ts_add.shape[: -1], ts_new.shape[: -1])
    npt.assert_equal(ts_add.shape[-1], ts_new.shape[-1] * 2)


@pytest.mark.parametrize('pulsetype', ['cathodicfirst', 'anodicfirst'])
@pytest.mark.parametrize('delay', [0, 10 / 1000])
@pytest.mark.parametrize('pulseorder', ['pulsefirst', 'gapfirst'])
def test_LegacyPulseTrain(pulsetype, delay, pulseorder):
    dur = 0.5
    pdur = 0.45 / 1000
    tsample = 5e-6
    ampl = 20.0
    freq = 5.0

    # First an easy one (sawtooth)...
    for scale in [1.0, 5.0, 10.0]:
        pt = LegacyPulseTrain(tsample=0.1 * scale,
                              dur=1.0 * scale, freq=freq / scale,
                              amp=ampl * scale,
                              pulse_dur=0.1 * scale,
                              interphase_dur=0,
                              pulsetype='cathodicfirst',
                              pulseorder='pulsefirst')
        npt.assert_equal(np.sum(np.isclose(pt.data, ampl * scale)), freq)
        npt.assert_equal(np.sum(np.isclose(pt.data, -ampl * scale)), freq)
        npt.assert_equal(pt.data[0], -ampl * scale)
        npt.assert_equal(pt.data[-1], ampl * scale)
        npt.assert_equal(len(pt.data), 10)

    # Slots:
    npt.assert_equal(hasattr(pt, '__slots__'), True)
    npt.assert_equal(hasattr(pt, '__dict__'), False)

    # Then some more general ones...
    # Size of array given stimulus duration
    stim_size = int(np.round(dur / tsample))

    # All empty pulse trains: Expect no division by zero errors
    for amp in [0, 20]:
        p2pt = LegacyPulseTrain(freq=0, amp=amp, dur=dur,
                                pulse_dur=pdur,
                                interphase_dur=pdur,
                                tsample=tsample)
        npt.assert_equal(p2pt.data, np.zeros(stim_size))

    # Non-zero pulse trains: Expect right length, pulse order, etc.
    for freq in [9, 13.8, 20]:
        p2pt = LegacyPulseTrain(freq=freq,
                                dur=dur,
                                pulse_dur=pdur,
                                interphase_dur=pdur,
                                delay=delay,
                                tsample=tsample,
                                amp=ampl,
                                pulsetype=pulsetype,
                                pulseorder=pulseorder)

        # make sure length is correct
        npt.assert_equal(p2pt.data.size, stim_size)

        # make sure amplitude is correct
        npt.assert_equal(np.unique(p2pt.data), np.array([-ampl, 0, ampl]))

        # make sure pulse type is correct
        idx_min = np.where(p2pt.data == p2pt.data.min())
        idx_max = np.where(p2pt.data == p2pt.data.max())
        if pulsetype == 'cathodicfirst':
            # cathodicfirst should have min first
            npt.assert_equal(idx_min[0] < idx_max[0], True)
        else:
            npt.assert_equal(idx_min[0] < idx_max[0], False)

        # Make sure frequency is correct
        # Need to trim size if `freq` is not a nice number
        envelope_size = int(np.round(1.0 / float(freq) / tsample))
        single_pulse_dur = int(np.round(1.0 * pdur / tsample))
        num_pulses = int(np.floor(dur * freq))  # round down
        trim_sz = envelope_size * num_pulses
        idx_min = np.where(p2pt.data[:trim_sz] == p2pt.data.min())
        idx_max = np.where(p2pt.data[:trim_sz] == p2pt.data.max())
        npt.assert_equal(idx_max[0].shape[-1], num_pulses * single_pulse_dur)
        npt.assert_equal(idx_min[0].shape[-1], num_pulses * single_pulse_dur)

        # make sure pulse order is correct
        delay_dur = int(np.round(delay / tsample))
        envelope_dur = int(np.round((1 / freq) / tsample))
        if pulsetype == 'cathodicfirst':
            val = p2pt.data.min()  # expect min first
        else:
            val = p2pt.data.max()  # expect max first
        if pulseorder == 'pulsefirst':
            idx0 = delay_dur  # expect pulse first, then gap
        else:
            idx0 = envelope_dur - 3 * single_pulse_dur
        npt.assert_equal(p2pt.data[idx0], val)
        npt.assert_equal(p2pt.data[idx0 + envelope_dur], val)

    # Invalid values
    with pytest.raises(ValueError):
        LegacyPulseTrain(0.1, delay=-10)
    with pytest.raises(ValueError):
        LegacyPulseTrain(0.1, pulse_dur=-10)
    with pytest.raises(ValueError):
        LegacyPulseTrain(0.1, freq=1000, pulse_dur=10)
    with pytest.raises(ValueError):
        LegacyPulseTrain(0.1, pulseorder='cathodicfirst')
    with pytest.raises(ValueError):
        LegacyPulseTrain(0)

    # Smoke test envelope_size > stim_size
    LegacyPulseTrain(1, freq=0.01, dur=0.01)
