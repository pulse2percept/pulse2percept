import numpy as np
import copy
import pytest
import numpy.testing as npt

from pulse2percept.stimuli import (Stimulus, PulseTrain, BiphasicPulseTrain,
                                   BiphasicTripletTrain,
                                   AsymmetricBiphasicPulseTrain)


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
        npt.assert_almost_equal(pt[0, np.floor(t_win)], 0)
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
