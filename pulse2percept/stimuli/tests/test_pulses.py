import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.stimuli import (AsymmetricBiphasicPulse, BiphasicPulse,
                                   MonophasicPulse, Stimulus)
from pulse2percept.utils.constants import DT

DECIMAL = int(-np.log10(DT))


@pytest.mark.parametrize('amp', (-1, 13))
@pytest.mark.parametrize('delay_dur', (0, 2.2, np.pi))
def test_MonophasicPulse(amp, delay_dur):
    phase_dur = 3.456
    # Basic usage:
    pulse = MonophasicPulse(amp, phase_dur, delay_dur=delay_dur)
    npt.assert_almost_equal(pulse[0, 0], 0)
    npt.assert_almost_equal(pulse[0, delay_dur + phase_dur / 2.0], amp,
                            decimal=DECIMAL)
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], phase_dur + delay_dur,
                            decimal=DECIMAL)
    npt.assert_equal(pulse.cathodic, amp <= 0)
    npt.assert_equal(pulse.is_charge_balanced, False)

    # Custom stim dur:
    pulse = MonophasicPulse(amp, phase_dur, delay_dur=delay_dur, stim_dur=100)
    npt.assert_almost_equal(pulse[0, 0], 0)
    npt.assert_almost_equal(pulse[0, delay_dur + phase_dur / 2.0], amp)
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], 100)

    # Exact stim dur:
    stim_dur = phase_dur + delay_dur
    pulse = MonophasicPulse(amp, phase_dur, delay_dur=delay_dur,
                            stim_dur=stim_dur)
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], stim_dur, decimal=6)

    # Zero amplitude:
    pulse = MonophasicPulse(0, phase_dur, delay_dur=delay_dur, electrode='A1')
    npt.assert_almost_equal(pulse.data, 0)
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], phase_dur + delay_dur,
                            decimal=DECIMAL)
    npt.assert_equal(pulse.is_charge_balanced, True)
    npt.assert_equal(pulse.electrodes, 'A1')

    # You can wrap a pulse in a Stimulus to overwrite attributes:
    stim = Stimulus(pulse, electrodes='AA1')
    npt.assert_equal(stim.electrodes, 'AA1')
    # Or concatenate:
    stim = Stimulus([pulse, pulse])
    npt.assert_equal(stim.shape[0], 2)
    npt.assert_almost_equal(stim.data[0, :], stim.data[1, :])
    npt.assert_almost_equal(stim.time, pulse.time)
    npt.assert_equal(stim.electrodes, ['A1', 1])
    # Concatenate and rename:
    stim = Stimulus([pulse, pulse], electrodes=['C1', 'D2'])
    npt.assert_equal(stim.electrodes, ['C1', 'D2'])

    # Invalid calls:
    with pytest.raises(ValueError):
        MonophasicPulse(amp, 0)
    with pytest.raises(ValueError):
        MonophasicPulse(amp, phase_dur, delay_dur=-1)
    with pytest.raises(ValueError):
        MonophasicPulse(amp, phase_dur, delay_dur=delay_dur, stim_dur=1)
    with pytest.raises(ValueError):
        MonophasicPulse(amp, phase_dur, delay_dur=delay_dur,
                        electrode=['A1', 'B2'])


@pytest.mark.parametrize('amp', (-1, 13))
@pytest.mark.parametrize('interphase_dur', (0, 1.3))
@pytest.mark.parametrize('delay_dur', (0, 4.55))
@pytest.mark.parametrize('cathodic_first', (True, False))
def test_BiphasicPulse(amp, interphase_dur, delay_dur, cathodic_first):
    phase_dur = 3.19
    mid_first_pulse = delay_dur + phase_dur / 2.0
    mid_interphase = delay_dur + phase_dur + interphase_dur / 2.0
    mid_second_pulse = delay_dur + interphase_dur + 1.5 * phase_dur
    first_amp = -np.abs(amp) if cathodic_first else np.abs(amp)
    second_amp = -first_amp
    min_dur = 2 * phase_dur + delay_dur + interphase_dur

    # Basic usage:
    pulse = BiphasicPulse(amp, phase_dur, interphase_dur=interphase_dur,
                          delay_dur=delay_dur, cathodic_first=cathodic_first)
    npt.assert_almost_equal(pulse[0, 0], 0)
    npt.assert_almost_equal(pulse[0, mid_first_pulse], first_amp)
    npt.assert_almost_equal(pulse[0, mid_interphase], 0)
    npt.assert_almost_equal(pulse[0, mid_second_pulse], second_amp)
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], min_dur, decimal=3)
    npt.assert_equal(pulse.cathodic_first, cathodic_first)
    npt.assert_equal(pulse.is_charge_balanced, True)

    # Custom stim dur:
    pulse = BiphasicPulse(amp, phase_dur, interphase_dur=interphase_dur,
                          delay_dur=delay_dur, cathodic_first=cathodic_first,
                          stim_dur=100, electrode='B1')
    npt.assert_almost_equal(pulse[0, 0], 0)
    npt.assert_almost_equal(pulse[0, mid_first_pulse], first_amp)
    npt.assert_almost_equal(pulse[0, mid_interphase], 0)
    npt.assert_almost_equal(pulse[0, mid_second_pulse], second_amp)
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], 100)
    npt.assert_equal(pulse.electrodes, 'B1')

    # Exact stim dur:
    stim_dur = 2 * phase_dur + interphase_dur + delay_dur
    pulse = BiphasicPulse(amp, phase_dur, interphase_dur=interphase_dur,
                          delay_dur=delay_dur, cathodic_first=cathodic_first,
                          stim_dur=stim_dur)
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], stim_dur, decimal=6)

    # Zero amplitude:
    pulse = BiphasicPulse(0, phase_dur, interphase_dur=interphase_dur,
                          delay_dur=delay_dur, cathodic_first=cathodic_first)
    npt.assert_almost_equal(pulse.data, 0)
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], min_dur, decimal=3)
    npt.assert_equal(pulse.is_charge_balanced, True)

    # You can wrap a pulse in a Stimulus to overwrite attributes:
    stim = Stimulus(pulse, electrodes='AA1')
    npt.assert_equal(stim.electrodes, ['AA1'])
    # Or concatenate:
    stim = Stimulus([pulse, pulse])
    npt.assert_equal(stim.shape[0], 2)
    npt.assert_almost_equal(stim.data[0, :], stim.data[1, :])
    npt.assert_almost_equal(stim.time, pulse.time)
    npt.assert_equal(stim.electrodes, [0, 1])
    # Concatenate and rename:
    stim = Stimulus([pulse, pulse], electrodes=['C1', 'D2'])
    npt.assert_equal(stim.electrodes, ['C1', 'D2'])

    # Floating point math with np.unique is tricky, but this works:
    BiphasicPulse(10, np.pi, interphase_dur=np.pi, delay_dur=np.pi,
                  stim_dur=5 * np.pi)

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
    with pytest.raises(ValueError):
        BiphasicPulse(amp, phase_dur, interphase_dur=interphase_dur,
                      delay_dur=delay_dur, electrode=['A1', 'B2'])


@pytest.mark.parametrize('amp1', (-1, 13))
@pytest.mark.parametrize('amp2', (4, -8))
@pytest.mark.parametrize('interphase_dur', (0, 1))
@pytest.mark.parametrize('delay_dur', (0, 6.01))
@pytest.mark.parametrize('cathodic_first', (True, False))
def test_AsymmetricBiphasicPulse(amp1, amp2, interphase_dur, delay_dur,
                                 cathodic_first):
    phase_dur1 = 2.1
    phase_dur2 = 4.87
    mid_first_pulse = delay_dur + phase_dur1 / 2.0
    mid_interphase = delay_dur + phase_dur1 + interphase_dur / 2.0
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
    npt.assert_almost_equal(pulse[0, mid_interphase], 0)
    npt.assert_almost_equal(pulse[0, mid_second_pulse], second_amp)
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], min_dur, decimal=3)
    npt.assert_equal(pulse.cathodic_first, cathodic_first)
    npt.assert_equal(pulse.is_charge_balanced,
                     np.isclose(np.trapz(pulse.data, pulse.time)[0], 0))

    # Custom stim dur:
    pulse = AsymmetricBiphasicPulse(amp1, amp2, phase_dur1, phase_dur2,
                                    interphase_dur=interphase_dur,
                                    delay_dur=delay_dur,
                                    cathodic_first=cathodic_first,
                                    stim_dur=100, electrode='A1')
    npt.assert_almost_equal(pulse[0, 0], 0)
    npt.assert_almost_equal(pulse[0, mid_first_pulse], first_amp)
    npt.assert_almost_equal(pulse[0, mid_interphase], 0)
    npt.assert_almost_equal(pulse[0, mid_second_pulse], second_amp)
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], 100)
    npt.assert_equal(pulse.electrodes, 'A1')

    # Exact stim dur:
    stim_dur = delay_dur + phase_dur1 + interphase_dur + phase_dur2
    pulse = AsymmetricBiphasicPulse(amp1, amp2, phase_dur1, phase_dur2,
                                    interphase_dur=interphase_dur,
                                    delay_dur=delay_dur,
                                    cathodic_first=cathodic_first,
                                    stim_dur=stim_dur, electrode='A1')
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], stim_dur, decimal=6)

    # Zero amplitude:
    pulse = AsymmetricBiphasicPulse(0, 0, phase_dur1, phase_dur2,
                                    interphase_dur=interphase_dur,
                                    delay_dur=delay_dur,
                                    cathodic_first=cathodic_first)
    npt.assert_almost_equal(pulse.data, 0)
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], min_dur, decimal=3)
    npt.assert_equal(pulse.is_charge_balanced,
                     np.isclose(np.trapz(pulse.data, pulse.time)[0], 0))

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

    # You can wrap a pulse in a Stimulus to overwrite attributes:
    stim = Stimulus(pulse, electrodes='AA1')
    npt.assert_equal(stim.electrodes, 'AA1')
    # Or concatenate:
    stim = Stimulus([pulse, pulse])
    npt.assert_equal(stim.shape[0], 2)
    npt.assert_almost_equal(stim.data[0, :], stim.data[1, :])
    npt.assert_almost_equal(stim.time, pulse.time, decimal=2)
    npt.assert_equal(stim.electrodes, [0, 1])
    # Concatenate and rename:
    stim = Stimulus([pulse, pulse], electrodes=['C1', 'D2'])
    npt.assert_equal(stim.electrodes, ['C1', 'D2'])

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
    with pytest.raises(ValueError):
        AsymmetricBiphasicPulse(amp1, amp2, phase_dur1, phase_dur2,
                                interphase_dur=interphase_dur,
                                delay_dur=delay_dur, electrode=['A1', 'B2'])


@pytest.mark.parametrize('amp', (-1.234, -13))
@pytest.mark.parametrize('phase_dur', (0.022, 2.2, np.pi))
def test_pulse_append(amp, phase_dur):
    # Build a biphasic pulse from two monophasic pulses:
    mono = MonophasicPulse(amp, phase_dur)
    bi = BiphasicPulse(amp, phase_dur)
    npt.assert_equal(mono.append(-mono) == bi, True)
