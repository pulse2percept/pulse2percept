import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.stimuli import (AsymmetricBiphasicPulse, BiphasicPulse,
                                   MonophasicPulse, Stimulus)
# Not exposed in __all__, and only needed for PulseTrain:
from pulse2percept.stimuli.pulses import (LegacyMonophasicPulse,
                                          LegacyBiphasicPulse)


@pytest.mark.parametrize('amp', (-1, 13))
@pytest.mark.parametrize('delay_dur', (0, 2.2))
def test_MonophasicPulse(amp, delay_dur):
    phase_dur = 3.456
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
    pulse = MonophasicPulse(0, phase_dur, delay_dur=delay_dur, electrode='A1')
    npt.assert_almost_equal(pulse.data, 0)
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], phase_dur + delay_dur)
    npt.assert_equal(pulse.charge_balanced, True)
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
    npt.assert_equal(pulse.charge_balanced, True)

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

    # Zero amplitude:
    pulse = BiphasicPulse(0, phase_dur, interphase_dur=interphase_dur,
                          delay_dur=delay_dur, cathodic_first=cathodic_first)
    npt.assert_almost_equal(pulse.data, 0)
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], min_dur, decimal=3)
    npt.assert_equal(pulse.charge_balanced, True)

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
    npt.assert_equal(pulse.charge_balanced,
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

    # Zero amplitude:
    pulse = AsymmetricBiphasicPulse(0, 0, phase_dur1, phase_dur2,
                                    interphase_dur=interphase_dur,
                                    delay_dur=delay_dur,
                                    cathodic_first=cathodic_first)
    npt.assert_almost_equal(pulse.data, 0)
    npt.assert_almost_equal(pulse.time[0], 0)
    npt.assert_almost_equal(pulse.time[-1], min_dur, decimal=3)
    npt.assert_equal(pulse.charge_balanced,
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


@pytest.mark.parametrize('ptype', ('anodic', 'cathodic'))
@pytest.mark.parametrize('sdur', [None, 100])
def test_LegacyMonophasicPulse(ptype, sdur):
    tsample = 1.0

    for pdur in range(10):
        for ddur in range(10):
            pulse = LegacyMonophasicPulse(ptype, pdur, tsample, ddur, sdur)
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
        LegacyMonophasicPulse('anodicfirst', 10, 0.1)
    with pytest.raises(ValueError):
        LegacyMonophasicPulse('cathodicfirst', 10, 0.1)


@pytest.mark.parametrize('ptype', ('cathodicfirst', 'anodicfirst'))
def test_LegacyBiphasicPulse(ptype):
    pdur = 0.25 / 1000
    tsample = 5e-6
    for interphase_dur in [0, 0.25 / 1000, 0.45 / 1000, 0.65 / 1000]:
        # generate pulse
        pulse = LegacyBiphasicPulse(ptype, pdur, tsample, interphase_dur)

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
        LegacyBiphasicPulse('anodic', 10, 0.1)
    with pytest.raises(ValueError):
        LegacyBiphasicPulse('cathodic', 10, 0.1)
