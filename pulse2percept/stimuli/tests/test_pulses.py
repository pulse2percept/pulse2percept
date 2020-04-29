import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.stimuli import (AsymmetricBiphasicPulse, BiphasicPulse,
                                   MonophasicPulse)


@pytest.mark.parametrize('amp', (-1, 13))
@pytest.mark.parametrize('phase_dur', (2, 3))
@pytest.mark.parametrize('delay_dur', (0, 4))
def test_MonophasicPulse(amp, phase_dur, delay_dur):
    # Basic usage:
    pulse = MonophasicPulse(amp, phase_dur, delay_dur=delay_dur)
    npt.assert_almost_equal(pulse[0, 0], 0)
    npt.assert_almost_equal(pulse[0, delay_dur + phase_dur / 2.0], amp)
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], phase_dur + delay_dur)
    npt.assert_equal(pulse.cathodic, amp <= 0)
    npt.assert_equal(pulse.charge_balanced, False)

    # Custom stim dur:
    pulse = MonophasicPulse(amp, phase_dur, delay_dur=delay_dur, stim_dur=100)
    npt.assert_almost_equal(pulse[0, 0], 0)
    npt.assert_almost_equal(pulse[0, delay_dur + phase_dur / 2.0], amp)
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], 100)

    # Zero amplitude:
    pulse = MonophasicPulse(0, phase_dur, delay_dur=delay_dur)
    npt.assert_almost_equal(pulse.data, 0)
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], phase_dur + delay_dur)
    npt.assert_equal(pulse.charge_balanced, True)

    # Invalid calls:
    with pytest.raises(ValueError):
        MonophasicPulse(amp, 0)
    with pytest.raises(ValueError):
        MonophasicPulse(amp, phase_dur, delay_dur=-1)
    with pytest.raises(ValueError):
        MonophasicPulse(amp, phase_dur, delay_dur=delay_dur, stim_dur=1)


@pytest.mark.parametrize('amp', (-1, 13))
@pytest.mark.parametrize('phase_dur', (2, 3))
@pytest.mark.parametrize('interphase_dur', (0, 1))
@pytest.mark.parametrize('delay_dur', (0, 4))
@pytest.mark.parametrize('cathodic_first', (True, False))
def test_BiphasicPulse(amp, phase_dur, interphase_dur, delay_dur,
                       cathodic_first):
    mid_first_pulse = delay_dur + phase_dur / 2.0
    mid_second_pulse = delay_dur + interphase_dur + 1.5 * phase_dur
    first_amp = -np.abs(amp) if cathodic_first else np.abs(amp)
    second_amp = -first_amp
    min_dur = 2 * phase_dur + delay_dur + interphase_dur

    # Basic usage:
    pulse = BiphasicPulse(amp, phase_dur, interphase_dur=interphase_dur,
                          delay_dur=delay_dur, cathodic_first=cathodic_first)
    npt.assert_almost_equal(pulse[0, 0], 0)
    npt.assert_almost_equal(pulse[0, mid_first_pulse], first_amp)
    npt.assert_almost_equal(pulse[0, mid_second_pulse], second_amp)
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], min_dur)
    npt.assert_equal(pulse.cathodic_first, cathodic_first)
    npt.assert_equal(pulse.charge_balanced, True)

    # Custom stim dur:
    pulse = BiphasicPulse(amp, phase_dur, interphase_dur=interphase_dur,
                          delay_dur=delay_dur, cathodic_first=cathodic_first,
                          stim_dur=100)
    npt.assert_almost_equal(pulse[0, 0], 0)
    npt.assert_almost_equal(pulse[0, mid_first_pulse], first_amp)
    npt.assert_almost_equal(pulse[0, mid_second_pulse], second_amp)
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], 100)

    # Zero amplitude:
    pulse = BiphasicPulse(0, phase_dur, interphase_dur=interphase_dur,
                          delay_dur=delay_dur, cathodic_first=cathodic_first)
    npt.assert_almost_equal(pulse.data, 0)
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], min_dur)
    npt.assert_equal(pulse.charge_balanced, True)

    # Invalid calls:
    with pytest.raises(ValueError):
        BiphasicPulse(amp, 0)
    with pytest.raises(ValueError):
        BiphasicPulse(amp, phase_dur, interphase_dur=-1)
    with pytest.raises(ValueError):
        BiphasicPulse(amp, phase_dur, interphase_dur=interphase_dur,
                      delay_dur=-1)
    with pytest.raises(ValueError):
        BiphasicPulse(amp, phase_dur, interphase_dur=interphase_dur,
                      delay_dur=delay_dur, stim_dur=1)


@pytest.mark.parametrize('amp1', (-1, 13))
@pytest.mark.parametrize('amp2', (4, -8))
@pytest.mark.parametrize('phase_dur1', (2, 3))
@pytest.mark.parametrize('phase_dur2', (4, 5))
@pytest.mark.parametrize('interphase_dur', (0, 1))
@pytest.mark.parametrize('delay_dur', (0, 6))
@pytest.mark.parametrize('cathodic_first', (True, False))
def test_AsymmetricBiphasicPulse(amp1, amp2, phase_dur1, phase_dur2,
                                 interphase_dur, delay_dur, cathodic_first):
    mid_first_pulse = delay_dur + phase_dur1 / 2.0
    mid_second_pulse = delay_dur + phase_dur1 + interphase_dur + phase_dur2 / 2
    first_amp = -np.abs(amp1) if cathodic_first else np.abs(amp1)
    second_amp = np.abs(amp2) if cathodic_first else -np.abs(amp2)
    min_dur = delay_dur + phase_dur1 + interphase_dur + phase_dur2

    # Basic usage:
    pulse = AsymmetricBiphasicPulse(amp1, amp2, phase_dur1, phase_dur2,
                                    interphase_dur=interphase_dur,
                                    delay_dur=delay_dur,
                                    cathodic_first=cathodic_first)
    npt.assert_almost_equal(pulse[0, 0], 0)
    npt.assert_almost_equal(pulse[0, mid_first_pulse], first_amp)
    npt.assert_almost_equal(pulse[0, mid_second_pulse], second_amp)
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], min_dur)
    npt.assert_equal(pulse.cathodic_first, cathodic_first)
    npt.assert_equal(pulse.charge_balanced,
                     np.isclose(np.trapz(pulse.time, pulse.data), 0))

    # Custom stim dur:
    pulse = AsymmetricBiphasicPulse(amp1, amp2, phase_dur1, phase_dur2,
                                    interphase_dur=interphase_dur,
                                    delay_dur=delay_dur,
                                    cathodic_first=cathodic_first,
                                    stim_dur=100)
    npt.assert_almost_equal(pulse[0, 0], 0)
    npt.assert_almost_equal(pulse[0, mid_first_pulse], first_amp)
    npt.assert_almost_equal(pulse[0, mid_second_pulse], second_amp)
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], 100)

    # Zero amplitude:
    pulse = AsymmetricBiphasicPulse(0, 0, phase_dur1, phase_dur2,
                                    interphase_dur=interphase_dur,
                                    delay_dur=delay_dur,
                                    cathodic_first=cathodic_first)
    npt.assert_almost_equal(pulse.data, 0)
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], min_dur)
    npt.assert_equal(pulse.charge_balanced,
                     np.isclose(np.trapz(pulse.time, pulse.data), 0))

    # If both phases have the same values, it's basically a symmetric biphasic
    # pulse:
    abp = AsymmetricBiphasicPulse(amp1, amp1, phase_dur1, phase_dur1,
                                  interphase_dur=interphase_dur,
                                  delay_dur=delay_dur,
                                  cathodic_first=cathodic_first)
    bp = BiphasicPulse(amp1, phase_dur1, interphase_dur=interphase_dur,
                       delay_dur=delay_dur, cathodic_first=cathodic_first)
    bp_min_dur = phase_dur1 * 2 + interphase_dur + delay_dur
    npt.assert_almost_equal(abp[:, np.linspace(0, bp_min_dur, num=5)],
                            bp[:, np.linspace(0, bp_min_dur, num=5)])
    npt.assert_equal(abp.cathodic_first, bp.cathodic_first)

    # If one phase is zero, it's basically a monophasic pulse:
    abp = AsymmetricBiphasicPulse(amp1, 0, phase_dur1, phase_dur2,
                                  interphase_dur=interphase_dur,
                                  delay_dur=delay_dur,
                                  cathodic_first=cathodic_first)
    mono = MonophasicPulse(first_amp, phase_dur1, delay_dur=delay_dur,
                           stim_dur=min_dur)
    npt.assert_almost_equal(abp[:, np.linspace(0, min_dur, num=5)],
                            mono[:, np.linspace(0, min_dur, num=5)])
    npt.assert_equal(abp.cathodic_first, mono.cathodic)

    # Invalid calls:
    with pytest.raises(ValueError):
        AsymmetricBiphasicPulse(amp1, amp2, 0, phase_dur2)
    with pytest.raises(ValueError):
        AsymmetricBiphasicPulse(amp1, amp2, phase_dur1, 0)
    with pytest.raises(ValueError):
        AsymmetricBiphasicPulse(amp1, amp2, phase_dur1, phase_dur2,
                                interphase_dur=-1)
    with pytest.raises(ValueError):
        AsymmetricBiphasicPulse(amp1, amp2, phase_dur1, phase_dur2,
                                interphase_dur=interphase_dur, delay_dur=-1)
    with pytest.raises(ValueError):
        AsymmetricBiphasicPulse(amp1, amp2, phase_dur1, phase_dur2,
                                interphase_dur=interphase_dur,
                                delay_dur=delay_dur, stim_dur=1)
