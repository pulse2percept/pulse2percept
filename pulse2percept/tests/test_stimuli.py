import numpy as np
import numpy.testing as npt
import pytest
try:
    from unittest import mock
except ImportError:
    import mock

import pulse2percept as p2p


def test_MonophasicPulse():
    tsample = 1.0

    for pdur in range(10):
        for ddur in range(10):
            for sdur in [None, 5, 10, 100]:
                for ptype in ['anodic', 'cathodic']:
                    pulse = p2p.stimuli.MonophasicPulse(ptype, pdur,
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
        p2p.stimuli.MonophasicPulse(10, 0.1, 0, 'anodicfirst')
    with pytest.raises(ValueError):
        p2p.stimuli.MonophasicPulse(10, 0.1, 0, 'cathodicfirst')


def test_BiphasicPulse():
    for ptype in ['cathodicfirst', 'anodicfirst']:
        for pdur in [0.25 / 1000, 0.45 / 1000, 0.65 / 1000]:
            for interphase_dur in [0, 0.25 / 1000, 0.45 / 1000, 0.65 / 1000]:
                for tsample in [5e-6, 1e-5, 5e-5]:
                    # generate pulse
                    pulse = p2p.stimuli.BiphasicPulse(ptype, pdur,
                                                      tsample,
                                                      interphase_dur)

                    # predicted length
                    pulse_gap_dur = 2 * pdur + interphase_dur

                    # make sure length is correct
                    npt.assert_equal(pulse.shape[-1],
                                     int(np.round(pulse_gap_dur /
                                                  tsample)))

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
        p2p.stimuli.BiphasicPulse(10, 0.1, 0, 'anodic')
    with pytest.raises(ValueError):
        p2p.stimuli.BiphasicPulse(10, 0.1, 0, 'cathodic')


def test_PulseTrain():
    dur = 0.5
    pdur = 0.45 / 1000
    tsample = 5e-6
    ampl = 20.0
    freq = 5.0

    # First an easy one (sawtooth)...
    for scale in [1.0, 2.0, 5.0, 10.0]:
        pt = p2p.stimuli.PulseTrain(tsample=0.1 * scale,
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
        p2pt = p2p.stimuli.PulseTrain(freq=0, amp=ampl, dur=dur,
                                      pulse_dur=pdur,
                                      interphase_dur=pdur,
                                      tsample=tsample)
        npt.assert_equal(p2pt.data, np.zeros(stim_size))

    # Non-zero pulse trains: Expect right length, pulse order, etc.
    for freq in [8, 13.8, 20]:
        for pulsetype in ['cathodicfirst', 'anodicfirst']:
            for delay in [0, 10 / 1000]:
                for pulseorder in ['pulsefirst', 'gapfirst']:
                    p2pt = p2p.stimuli.PulseTrain(freq=freq,
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
        p2p.stimuli.PulseTrain(0.1, delay=-10)
    with pytest.raises(ValueError):
        p2p.stimuli.PulseTrain(0.1, pulse_dur=-10)
    with pytest.raises(ValueError):
        p2p.stimuli.PulseTrain(0.1, freq=1000, pulse_dur=10)
    with pytest.raises(ValueError):
        p2p.stimuli.PulseTrain(0.1, pulseorder='cathodicfirst')


def test_image2pulsetrain():
    # Range of values
    amp_min = 2
    amp_max = 15

    # Create a standard Argus I array
    implant = p2p.implants.ArgusI()

    # Create a small image with 1 pixel per electrode
    img = np.zeros((4, 4))

    # Trigger an import error
    with mock.patch.dict("sys.modules", {"skimage": {}}):
        with pytest.raises(ImportError):
            p2p.stimuli.image2pulsetrain(img, implant)

    # An all-zero image should give us a really boring stimulation protocol
    pulses = p2p.stimuli.image2pulsetrain(img, implant, valrange=[amp_min,
                                                                  amp_max])
    for pt in pulses:
        npt.assert_equal(pt.data.max(), amp_min)

    # Now put some structure in the image
    img[1, 1] = img[1, 2] = img[2, 1] = img[2, 2] = 0.75

    for max_contrast, val_max in zip([True, False], [amp_max, 0.75 * amp_max]):
        pt = p2p.stimuli.image2pulsetrain(img, implant, coding='amplitude',
                                          max_contrast=max_contrast,
                                          rftype='square', rfsize=50,
                                          valrange=[amp_min, amp_max])

        # Make sure we have one pulse train per electrode
        npt.assert_equal(len(pt), implant.num_electrodes)

        # Make sure the brightest electrode has `amp_max`
        npt.assert_equal(np.round(np.max([p.data.max() for p in pt])),
                         np.round(val_max))

        # Make sure the dimmest electrode has `amp_min` as max amplitude
        npt.assert_almost_equal(np.min([np.abs(p.data).max() for p in pt]),
                                amp_min, decimal=1)

    # Invalid implant
    with pytest.raises(TypeError):
        p2p.stimuli.image2pulsetrain("rainbow_cat.jpg", np.zeros(10))

    # Invalid image
    with pytest.raises(IOError):
        p2p.stimuli.image2pulsetrain("rainbow_cat.jpg", p2p.implants.ArgusI())

    # Smoke-test RGB
    p2p.stimuli.image2pulsetrain(np.zeros((10, 10, 3)), p2p.implants.ArgusI())

    # Smoke-test invert
    p2p.stimuli.image2pulsetrain(np.zeros((10, 10, 3)), p2p.implants.ArgusI(),
                                 invert=True)

    # Smoke-test normalize
    p2p.stimuli.image2pulsetrain(np.ones((10, 10, 3)) * 2,
                                 p2p.implants.ArgusI(), invert=True)

    # Smoke-test frequency coding
    p2p.stimuli.image2pulsetrain(np.zeros((10, 10, 3)), p2p.implants.ArgusI(),
                                 coding='frequency')

    # Invalid coding
    with pytest.raises(ValueError):
        p2p.stimuli.image2pulsetrain(np.zeros((10, 10)), p2p.implants.ArgusI(),
                                     coding='n/a')


def test_video2pulsetrain():
    implant = p2p.implants.ElectrodeArray('epiretinal', 100, 0, 0)

    # Trigger an import error
    with mock.patch.dict("sys.modules", {"skvideo": {}, "skvideo.utils": {}}):
        with pytest.raises(ImportError):
            p2p.stimuli.video2pulsetrain('invalid.avi', implant)

    with pytest.raises(FileNotFoundError):
        p2p.stimuli.video2pulsetrain('no-such-file.avi', implant)

    # Smoke-test example video
    from skvideo import datasets
    p2p.stimuli.video2pulsetrain(datasets.bikes(), implant)


def test_Movie2Pulsetrain():
    fps = 30.0
    amplitude_transform = 'linear'
    amp_max = 90
    freq = 20
    pulse_dur = .075 / 1000.
    interphase_dur = .075 / 1000.
    tsample = .005 / 1000.
    pulsetype = 'cathodicfirst'
    stimtype = 'pulsetrain'
    rflum = np.zeros(100)
    rflum[50] = 1
    m2pt = p2p.stimuli.Movie2Pulsetrain(rflum,
                                        fps=fps,
                                        amp_transform=amplitude_transform,
                                        amp_max=amp_max,
                                        freq=freq,
                                        pulse_dur=pulse_dur,
                                        interphase_dur=interphase_dur,
                                        tsample=tsample,
                                        pulsetype=pulsetype,
                                        stimtype=stimtype)
    npt.assert_equal(m2pt.shape[0], round((rflum.shape[-1] / fps) / tsample))
    npt.assert_(m2pt.data.max() < amp_max)


def test_parse_pulse_trains():
    # Specify pulse trains in a number of different ways and make sure they
    # are all identical after parsing

    # Create some p2p.implants
    argus = p2p.implants.ArgusI()
    simple = p2p.implants.ElectrodeArray('subretinal', 0, 0, 0, 0)

    pt_zero = p2p.utils.TimeSeries(1, np.zeros(1000))
    pt_nonzero = p2p.utils.TimeSeries(1, np.random.rand(1000))

    # Test 1
    # ------
    # Specify wrong number of pulse trains
    with pytest.raises(ValueError):
        p2p.stimuli.parse_pulse_trains(pt_nonzero, argus)
    with pytest.raises(ValueError):
        p2p.stimuli.parse_pulse_trains([pt_nonzero], argus)
    with pytest.raises(ValueError):
        p2p.stimuli.parse_pulse_trains([pt_nonzero] *
                                       (argus.num_electrodes - 1),
                                       argus)
    with pytest.raises(ValueError):
        p2p.stimuli.parse_pulse_trains([pt_nonzero] * 2, simple)

    # Test 2
    # ------
    # Send non-zero pulse train to specific electrode
    el_name = 'B3'
    el_idx = argus.get_index(el_name)

    # Specify a list of 16 pulse trains (one for each electrode)
    pt0_in = [pt_zero] * argus.num_electrodes
    pt0_in[el_idx] = pt_nonzero
    pt0_out = p2p.stimuli.parse_pulse_trains(pt0_in, argus)

    # Specify a dict with non-zero pulse trains
    pt1_in = {el_name: pt_nonzero}
    pt1_out = p2p.stimuli.parse_pulse_trains(pt1_in, argus)

    # Make sure the two give the same result
    for p0, p1 in zip(pt0_out, pt1_out):
        npt.assert_equal(p0.data, p1.data)

    # Test 3
    # ------
    # Smoke testing
    p2p.stimuli.parse_pulse_trains([pt_zero] * argus.num_electrodes, argus)
    p2p.stimuli.parse_pulse_trains(pt_zero, simple)
    p2p.stimuli.parse_pulse_trains([pt_zero], simple)
