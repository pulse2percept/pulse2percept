from contextlib import contextmanager

import numpy as np
import numpy.testing as npt
import pytest
from scipy.integrate import trapezoid

from pulse2percept.stimuli import (AsymmetricBiphasicPulse, BiphasicPulse,
                                   MonophasicPulse, Stimulus)
from pulse2percept.utils.constants import DT
from pulse2percept.units import (DimensionMismatchError, mA, ms, uA,
                                 us)
from pulse2percept.units import s as sec

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
                     np.isclose(trapezoid(pulse.data, pulse.time)[0], 0))

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
                     np.isclose(trapezoid(pulse.data, pulse.time)[0], 0))

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


def test_pulse_units():
    """Equivalent unit choices must produce numerically identical pulses"""
    # The headline case from the spec:
    npt.assert_equal(BiphasicPulse(50, 0.45) == BiphasicPulse(0.05 * mA,
                                                              450 * us), True)
    pairs = [
        (MonophasicPulse(-20, 1, delay_dur=2, stim_dur=10),
         MonophasicPulse(-0.02 * mA, 1000 * us, delay_dur=0.002 * sec,
                         stim_dur=0.01 * sec)),
        (BiphasicPulse(50, 0.45, interphase_dur=0.2, delay_dur=1,
                       stim_dur=20),
         BiphasicPulse(0.05 * mA, 450 * us, interphase_dur=200 * us,
                       delay_dur=1 * ms, stim_dur=0.02 * sec)),
        (AsymmetricBiphasicPulse(-40, 10, 1, 4, interphase_dur=1, delay_dur=2,
                                 stim_dur=15),
         AsymmetricBiphasicPulse(-0.04 * mA, 0.01 * mA, 1 * ms, 4000 * us,
                                 interphase_dur=1 * ms, delay_dur=0.002 * sec,
                                 stim_dur=15 * ms)),
    ]
    for bare, unitful in pairs:
        # Not merely close: the same arrays, bit for bit.
        npt.assert_array_equal(bare.data, unitful.data)
        npt.assert_array_equal(bare.time, unitful.time)
        npt.assert_equal(bare.data.dtype, np.float32)
        npt.assert_equal(unitful.data.dtype, np.float32)
        npt.assert_equal(bare == unitful, True)
        npt.assert_equal(unitful.unit, uA)
        npt.assert_equal(unitful.time_unit, ms)
    # A quantity of the wrong dimension is caught, and names the argument:
    with pytest.raises(DimensionMismatchError) as excinfo:
        BiphasicPulse(10 * ms, 0.45 * ms)
    npt.assert_equal("Parameter 'amp' expects electric current (uA), got time"
                     in str(excinfo.value), True)
    with pytest.raises(DimensionMismatchError):
        BiphasicPulse(50 * uA, 0.45 * uA)
    with pytest.raises(DimensionMismatchError):
        MonophasicPulse(20, 1, delay_dur=2 * uA)
    with pytest.raises(DimensionMismatchError):
        MonophasicPulse(20, 1, stim_dur=10 * uA)
    with pytest.raises(DimensionMismatchError):
        AsymmetricBiphasicPulse(-40 * ms, 10, 1, 4)
    with pytest.raises(DimensionMismatchError):
        AsymmetricBiphasicPulse(-40, 10, 1, 4, interphase_dur=1 * uA)


@contextmanager
def counting_renders(cls):
    """Count how often ``cls`` generates a waveform inside the block"""
    original = cls._render
    counts = []

    def counted(self):
        counts.append(type(self).__name__)
        return original(self)
    cls._render = counted
    try:
        yield counts
    finally:
        cls._render = original


def _rendered(stim):
    """Whether the stimulus has generated its waveform yet

    Reads the private container, because every public attribute that could
    answer the question would generate one first.
    """
    return stim._Stimulus__stim['data'] is not None


# One entry per (class, build, the parameters that define it):
PULSES = [
    (MonophasicPulse,
     lambda: MonophasicPulse(-20, 1, delay_dur=2, stim_dur=10),
     {'amp': -20, 'phase_dur': 1, 'delay_dur': 2, 'stim_dur': 10,
      'cathodic': True}),
    (MonophasicPulse,
     lambda: MonophasicPulse(13, 0.45),
     {'amp': 13, 'phase_dur': 0.45, 'delay_dur': 0, 'stim_dur': 0.45,
      'cathodic': False}),
    (BiphasicPulse,
     lambda: BiphasicPulse(-20, 1, interphase_dur=0.5, delay_dur=2,
                           stim_dur=10),
     {'amp': 20, 'phase_dur': 1, 'interphase_dur': 0.5, 'delay_dur': 2,
      'stim_dur': 10, 'cathodic_first': True}),
    (BiphasicPulse,
     lambda: BiphasicPulse(20, 0.45, cathodic_first=False),
     {'amp': 20, 'phase_dur': 0.45, 'interphase_dur': 0, 'delay_dur': 0,
      'stim_dur': 0.9, 'cathodic_first': False}),
    (AsymmetricBiphasicPulse,
     lambda: AsymmetricBiphasicPulse(-40, 10, 1, 4, interphase_dur=1,
                                     delay_dur=2, stim_dur=15),
     {'amp1': 40, 'amp2': 10, 'phase_dur1': 1, 'phase_dur2': 4,
      'interphase_dur': 1, 'delay_dur': 2, 'stim_dur': 15,
      'cathodic_first': True}),
    (AsymmetricBiphasicPulse,
     lambda: AsymmetricBiphasicPulse(40, 10, 0.45, 0.9, cathodic_first=False),
     {'amp1': 40, 'amp2': 10, 'phase_dur1': 0.45, 'phase_dur2': 0.9,
      'interphase_dur': 0, 'delay_dur': 0, 'stim_dur': 1.35,
      'cathodic_first': False}),
]


@pytest.mark.parametrize('cls, build, params', PULSES)
def test_pulse_parameters_are_canonical(cls, build, params):
    # A pulse reads back the parameters it was built from. The two amplitude
    # conventions differ on purpose: a monophasic pulse gets its polarity from
    # the sign of `amp`, a biphasic one from `cathodic_first`, so the latter
    # keeps only the magnitude (which is all of it that reaches the waveform).
    pulse = build()
    for name, expected in params.items():
        npt.assert_almost_equal(getattr(pulse, name), expected)
    # The asymmetric parameters stay distinct from one another:
    if cls is AsymmetricBiphasicPulse:
        npt.assert_equal(pulse.amp1 != pulse.amp2, True)
        npt.assert_equal(pulse.phase_dur1 != pulse.phase_dur2, True)


@pytest.mark.parametrize('cls, build, params', PULSES)
def test_pulse_parameters_are_read_only(cls, build, params):
    # Assigning one would leave a cached waveform contradicting the pulse it
    # is supposed to describe. Build another pulse instead:
    pulse = build()
    for name in params:
        with pytest.raises(AttributeError):
            setattr(pulse, name, 1)


@pytest.mark.parametrize('cls, build, params', PULSES)
def test_pulse_renders_once_and_only_when_asked(cls, build, params):
    with counting_renders(cls) as counts:
        pulse = build()
        npt.assert_equal(_rendered(pulse), False)
        npt.assert_equal(counts, [])
        # Everything a pulse knows from its parameters alone:
        for name in params:
            getattr(pulse, name)
        npt.assert_equal(list(pulse.electrodes), [0])
        npt.assert_almost_equal(pulse.duration, params['stim_dur'])
        repr(pulse)
        npt.assert_equal(counts, [])
        npt.assert_equal(_rendered(pulse), False)
        # ...and the waveform, which is generated exactly once:
        npt.assert_equal(pulse.data.shape[0], 1)
        npt.assert_equal(len(counts), 1)
        npt.assert_equal(_rendered(pulse), True)
        for _ in range(3):
            pulse.data, pulse.time, pulse.shape, pulse[0, 0.001]
        npt.assert_equal(len(counts), 1)
        # `duration` is the same number the waveform ends on:
        npt.assert_almost_equal(pulse.duration, pulse.time[-1])


@pytest.mark.parametrize('cls, build, params', PULSES)
def test_pulse_rendered_state_is_immutable(cls, build, params):
    pulse = build()
    npt.assert_equal(pulse.data.flags.writeable, False)
    npt.assert_equal(pulse.time.flags.writeable, False)
    npt.assert_equal(pulse.data.dtype, np.float32)
    npt.assert_equal(pulse.time.dtype, np.float64)
    npt.assert_equal(pulse.data.flags['C_CONTIGUOUS'], True)


def test_pulse_electrode_names():
    # An unnamed pulse is electrode 0, as `Stimulus` has always numbered it:
    npt.assert_equal(list(MonophasicPulse(-20, 1).electrodes), [0])
    npt.assert_equal(list(BiphasicPulse(-20, 1, electrode='C3').electrodes),
                     ['C3'])
    # A pulse drives one electrode, and says so at construction rather than
    # leaving the mismatch to surface out of the waveform:
    for build in (lambda e: MonophasicPulse(-20, 1, electrode=e),
                  lambda e: BiphasicPulse(-20, 1, electrode=e),
                  lambda e: AsymmetricBiphasicPulse(-40, 10, 1, 4,
                                                    electrode=e)):
        with pytest.raises(ValueError):
            build(['A1', 'B2'])
