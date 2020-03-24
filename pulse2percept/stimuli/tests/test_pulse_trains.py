import numpy as np
import copy
import pytest
import numpy.testing as npt

from pulse2percept.stimuli import (TimeSeries, MonophasicPulse, BiphasicPulse,
                                   PulseTrain)


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


@pytest.mark.parametrize('ptype', ('anodic', 'cathodic'))
@pytest.mark.parametrize('sdur', [None, 5, 10, 100])
def test_MonophasicPulse(ptype, sdur):
    tsample = 1.0

    for pdur in range(10):
        for ddur in range(10):
            pulse = MonophasicPulse(ptype, pdur, tsample, ddur, sdur)
            # Slots:
            npt.assert_equal(hasattr(pulse, '__slots__'), True)
            npt.assert_equal(hasattr(pulse, '__dict__'), False)

            # Make sure the pulse length is correct:
            # When stimulus duration is not specified, stimulus must
            # tightly fit pulse + delay. When stimulus direction is
            # specified, pulse duration must match that number exactly.
            if sdur is None:
                npt.assert_equal(pulse.data.size, pdur + ddur)
            else:
                npt.assert_equal(pulse.data.size, sdur)

            # Make sure delay is correct
            delay = pulse.data[:ddur]
            npt.assert_equal(np.allclose(delay, 0.0), True)

            # Make sure pulse length and amplitude are correct
            if pdur > 0:
                # Actually, depending on settings, pulse duration might
                # be different from what was specified:
                if sdur is not None and pdur + ddur > sdur:
                    # Stim is trimmed, adjust pdur
                    actual_pdur = np.maximum(0, sdur - ddur)
                else:
                    actual_pdur = pdur

                # Find maximma/minima
                idx_min = np.isclose(pulse.data, -1.0)
                idx_max = np.isclose(pulse.data, 1.0)
                if ptype == 'anodic':
                    npt.assert_equal(np.sum(idx_max), actual_pdur)
                    if actual_pdur > 0:
                        npt.assert_equal(pulse.data.max(), 1.0)
                else:
                    npt.assert_equal(np.sum(idx_min), actual_pdur)
                    if actual_pdur > 0:
                        npt.assert_equal(pulse.data.min(), -1.0)

    # Invalid pulsetype
    with pytest.raises(ValueError):
        MonophasicPulse('anodicfirst', 10, 0.1)
    with pytest.raises(ValueError):
        MonophasicPulse('cathodicfirst', 10, 0.1)


@pytest.mark.parametrize('ptype', ('cathodicfirst', 'anodicfirst'))
@pytest.mark.parametrize('pdur', (0.25 / 1000, 0.45 / 1000))
@pytest.mark.parametrize('tsample', (5e-6, 5e-5))
def test_BiphasicPulse(ptype, pdur, tsample):
    for interphase_dur in [0, 0.25 / 1000, 0.45 / 1000, 0.65 / 1000]:
        # generate pulse
        pulse = BiphasicPulse(ptype, pdur, tsample, interphase_dur)

        # Slots:
        npt.assert_equal(hasattr(pulse, '__slots__'), True)
        npt.assert_equal(hasattr(pulse, '__dict__'), False)

        # predicted length
        pulse_gap_dur = 2 * pdur + interphase_dur

        # make sure length is correct
        npt.assert_equal(
            pulse.shape[-1], int(np.round(pulse_gap_dur / tsample)))

        # make sure amplitude is correct: negative peak,
        # zero (in case of nonnegative interphase dur),
        # positive peak
        if interphase_dur > 0:
            npt.assert_equal(np.unique(pulse.data), np.array([-1, 0, 1]))
        else:
            npt.assert_equal(np.unique(pulse.data), np.array([-1, 1]))

        # make sure pulse order is correct
        idx_min = np.where(pulse.data == pulse.data.min())
        idx_max = np.where(pulse.data == pulse.data.max())
        if ptype == 'cathodicfirst':
            # cathodicfirst should have min first
            npt.assert_equal(idx_min[0] < idx_max[0], True)
        else:
            npt.assert_equal(idx_min[0] < idx_max[0], False)

    # Invalid pulsetype
    with pytest.raises(ValueError):
        BiphasicPulse('anodic', 10, 0.1)
    with pytest.raises(ValueError):
        BiphasicPulse('cathodic', 10, 0.1)


@pytest.mark.parametrize('pulsetype', ['cathodicfirst', 'anodicfirst'])
@pytest.mark.parametrize('delay', [0, 10 / 1000])
@pytest.mark.parametrize('pulseorder', ['pulsefirst', 'gapfirst'])
def test_PulseTrain(pulsetype, delay, pulseorder):
    dur = 0.5
    pdur = 0.45 / 1000
    tsample = 5e-6
    ampl = 20.0
    freq = 5.0

    # First an easy one (sawtooth)...
    for scale in [1.0, 5.0, 10.0]:
        pt = PulseTrain(tsample=0.1 * scale,
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
        p2pt = PulseTrain(freq=0, amp=amp, dur=dur,
                          pulse_dur=pdur,
                          interphase_dur=pdur,
                          tsample=tsample)
        npt.assert_equal(p2pt.data, np.zeros(stim_size))

    # Non-zero pulse trains: Expect right length, pulse order, etc.
    for freq in [9, 13.8, 20]:
        p2pt = PulseTrain(freq=freq,
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
        PulseTrain(0.1, delay=-10)
    with pytest.raises(ValueError):
        PulseTrain(0.1, pulse_dur=-10)
    with pytest.raises(ValueError):
        PulseTrain(0.1, freq=1000, pulse_dur=10)
    with pytest.raises(ValueError):
        PulseTrain(0.1, pulseorder='cathodicfirst')
    with pytest.raises(ValueError):
        PulseTrain(0)

    # Smoke test envelope_size > stim_size
    PulseTrain(1, freq=0.01, dur=0.01)
