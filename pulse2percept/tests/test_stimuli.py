import numpy as np
import numpy.testing as npt
import pytest
try:
    # Python 3
    from unittest import mock
except ImportError:
    # Python 2
    import mock

try:
    # Python 3
    from imp import reload
except ImportError:
    pass

from .. import stimuli
from .. import implants
from .. import utils


def test_MonophasicPulse():
    tsample = 1.0

    for pdur in range(10):
        for ddur in range(10):
            for sdur in [None, 5, 10, 100]:
                for ptype in ['anodic', 'cathodic']:
                    pulse = stimuli.MonophasicPulse(ptype, pdur,
                                                    tsample, ddur,
                                                    sdur)

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
        stimuli.MonophasicPulse('anodicfirst', 10, 0.1)
    with pytest.raises(ValueError):
        stimuli.MonophasicPulse('cathodicfirst', 10, 0.1)


def test_BiphasicPulse():
    for ptype in ['cathodicfirst', 'anodicfirst']:
        for pdur in [0.25 / 1000, 0.45 / 1000, 0.65 / 1000]:
            for interphase_dur in [0, 0.25 / 1000, 0.45 / 1000, 0.65 / 1000]:
                for tsample in [5e-6, 1e-5, 5e-5]:
                    # generate pulse
                    pulse = stimuli.BiphasicPulse(ptype, pdur,
                                                  tsample,
                                                  interphase_dur)

                    # predicted length
                    pulse_gap_dur = 2 * pdur + interphase_dur

                    # make sure length is correct
                    npt.assert_equal(pulse.shape[-1],
                                     int(np.round(pulse_gap_dur
                                                  / tsample)))

                    # make sure amplitude is correct: negative peak,
                    # zero (in case of nonnegative interphase dur),
                    # positive peak
                    if interphase_dur > 0:
                        npt.assert_equal(np.unique(pulse.data),
                                         np.array([-1, 0, 1]))
                    else:
                        npt.assert_equal(np.unique(pulse.data),
                                         np.array([-1, 1]))

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
        stimuli.BiphasicPulse('anodic', 10, 0.1)
    with pytest.raises(ValueError):
        stimuli.BiphasicPulse('cathodic', 10, 0.1)


def test_PulseTrain():
    dur = 0.5
    pdur = 0.45 / 1000
    tsample = 5e-6
    ampl = 20.0
    freq = 5.0

    # First an easy one (sawtooth)...
    for scale in [1.0, 2.0, 5.0, 10.0]:
        pt = stimuli.PulseTrain(tsample=0.1 * scale,
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

    # Then some more general ones...
    # Size of array given stimulus duration
    stim_size = int(np.round(dur / tsample))

    # All empty pulse trains: Expect no division by zero errors
    for amp in [0, 20]:
        p2pt = stimuli.PulseTrain(freq=0, amp=ampl, dur=dur,
                                  pulse_dur=pdur,
                                  interphase_dur=pdur,
                                  tsample=tsample)
        npt.assert_equal(p2pt.data, np.zeros(stim_size))

    # Non-zero pulse trains: Expect right length, pulse order, etc.
    for freq in [8, 13.8, 20]:
        for pulsetype in ['cathodicfirst', 'anodicfirst']:
            for delay in [0, 10 / 1000]:
                for pulseorder in ['pulsefirst', 'gapfirst']:
                    p2pt = stimuli.PulseTrain(freq=freq,
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
                    npt.assert_equal(np.unique(p2pt.data),
                                     np.array([-ampl, 0, ampl]))

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
                    npt.assert_equal(idx_max[0].shape[-1],
                                     num_pulses * single_pulse_dur)
                    npt.assert_equal(idx_min[0].shape[-1],
                                     num_pulses * single_pulse_dur)

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
        stimuli.PulseTrain(0.1, delay=-10)
    with pytest.raises(ValueError):
        stimuli.PulseTrain(0.1, pulse_dur=-10)
    with pytest.raises(ValueError):
        stimuli.PulseTrain(0.1, freq=1000, pulse_dur=10)
    with pytest.raises(ValueError):
        stimuli.PulseTrain(0.1, pulseorder='cathodicfirst')
    with pytest.raises(ValueError):
        stimuli.PulseTrain(0)

    # Smoke test envelope_size > stim_size
    stimuli.PulseTrain(1, freq=0.01, dur=0.01)


def test_image2pulsetrain():
    # Range of values
    amp_min = 2
    amp_max = 15

    # Create a standard Argus I array
    implant = implants.ArgusI()

    # Create a small image with 1 pixel per electrode
    img = np.zeros((4, 4))

    # An all-zero image should give us a really boring stimulation protocol
    pulses = stimuli.image2pulsetrain(img, implant, valrange=[amp_min,
                                                              amp_max])
    for pt in pulses:
        npt.assert_equal(pt.data.max(), amp_min)

    # Now put some structure in the image
    img[1, 1] = img[1, 2] = img[2, 1] = img[2, 2] = 0.75

    expected_max = [amp_max, 0.75 * (amp_max - amp_min) + amp_min]
    for max_contrast, val_max in zip([True, False], expected_max):
        pt = stimuli.image2pulsetrain(img, implant, coding='amplitude',
                                      max_contrast=max_contrast,
                                      valrange=[amp_min, amp_max])

        # Make sure we have one pulse train per electrode
        npt.assert_equal(len(pt), implant.num_electrodes)

        # Make sure the brightest electrode has `amp_max`
        npt.assert_almost_equal(np.max([p.data.max() for p in pt]),
                                val_max)

        # Make sure the dimmest electrode has `amp_min` as max amplitude
        npt.assert_almost_equal(np.min([np.abs(p.data).max() for p in pt]),
                                amp_min)

    # Invalid implant
    with pytest.raises(TypeError):
        stimuli.image2pulsetrain("rainbow_cat.jpg", np.zeros(10))
    with pytest.raises(TypeError):
        e_array = implants.ElectrodeArray('epiretinal', 100, 0, 0)
        stimuli.image2pulsetrain("rainbow_cat.jpg", e_array)

    # Invalid image
    with pytest.raises(IOError):
        stimuli.image2pulsetrain("rainbow_cat.jpg", implants.ArgusI())

    # Smoke-test RGB
    stimuli.image2pulsetrain(np.zeros((10, 10, 3)), implants.ArgusI())

    # Smoke-test invert
    stimuli.image2pulsetrain(np.zeros((10, 10, 3)), implants.ArgusI(),
                             invert=True)

    # Smoke-test normalize
    stimuli.image2pulsetrain(np.ones((10, 10, 3)) * 2,
                             implants.ArgusI(), invert=True)

    # Smoke-test frequency coding
    stimuli.image2pulsetrain(np.zeros((10, 10, 3)), implants.ArgusI(),
                             coding='frequency')

    # Invalid coding
    with pytest.raises(ValueError):
        stimuli.image2pulsetrain(np.zeros((10, 10)), implants.ArgusI(),
                                 coding='n/a')

    # Trigger an import error
    with mock.patch.dict("sys.modules", {"skimage": {}, "skimage.io": {}}):
        with pytest.raises(ImportError):
            reload(stimuli)
            stimuli.image2pulsetrain(img, implant)


@pytest.mark.skip(reason='ffmpeg dependency')
def test_video2pulsetrain():
    reload(stimuli)
    implant = implants.ArgusI()

    with pytest.raises(OSError):
        stimuli.video2pulsetrain('no-such-file.avi', implant)

    # Smoke-test example video
    from skvideo import datasets
    stimuli.video2pulsetrain(datasets.bikes(), implant)


def test_parse_pulse_trains():
    # Specify pulse trains in a number of different ways and make sure they
    # are all identical after parsing

    # Create some p2p.implants
    argus = implants.ArgusI()
    simple = implants.ElectrodeArray('subretinal', 0, 0, 0, 0)

    pt_zero = utils.TimeSeries(1, np.zeros(1000))
    pt_nonzero = utils.TimeSeries(1, np.random.rand(1000))

    # Test 1
    # ------
    # Specify wrong number of pulse trains
    with pytest.raises(ValueError):
        stimuli.parse_pulse_trains(pt_nonzero, argus)
    with pytest.raises(ValueError):
        stimuli.parse_pulse_trains([pt_nonzero], argus)
    with pytest.raises(ValueError):
        stimuli.parse_pulse_trains([pt_nonzero]
                                  * (argus.num_electrodes - 1),
                                   argus)
    with pytest.raises(ValueError):
        stimuli.parse_pulse_trains([pt_nonzero] * 2, simple)

    # Test 2
    # ------
    # Send non-zero pulse train to specific electrode
    el_name = 'B3'
    el_idx = argus.get_index(el_name)

    # Specify a list of 16 pulse trains (one for each electrode)
    pt0_in = [pt_zero] * argus.num_electrodes
    pt0_in[el_idx] = pt_nonzero
    pt0_out = stimuli.parse_pulse_trains(pt0_in, argus)

    # Specify a dict with non-zero pulse trains
    pt1_in = {el_name: pt_nonzero}
    pt1_out = stimuli.parse_pulse_trains(pt1_in, argus)

    # Make sure the two give the same result
    for p0, p1 in zip(pt0_out, pt1_out):
        npt.assert_equal(p0.data, p1.data)

    # Test 3
    # ------
    # Smoke testing
    stimuli.parse_pulse_trains([pt_zero] * argus.num_electrodes, argus)
    stimuli.parse_pulse_trains(pt_zero, simple)
    stimuli.parse_pulse_trains([pt_zero], simple)
