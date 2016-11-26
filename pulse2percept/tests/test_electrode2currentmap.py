import numpy as np
import numpy.testing as npt

import pulse2percept.electrode2currentmap as e2cm


def test_Electrode():
    num_pts = 10
    r = np.linspace(1, 1000, num_pts)
    x = np.linspace(-1000, 1000, num_pts)
    y = np.linspace(-2000, 2000, num_pts)
    h = np.linspace(0, 1000, num_pts)
    t = ['subretinal', 'epiretinal'] * (num_pts // 2)

    for rr, xx, yy, hh, tt in zip(r, x, y, h, t):
        e = e2cm.Electrode(rr, xx, yy, hh, tt)
        npt.assert_equal(e.r, rr)
        npt.assert_equal(e.x, xx)
        npt.assert_equal(e.y, yy)
        # npt.assert_equal(e.h, hh)
        npt.assert_equal(e.ptype, tt)


def test_ElectrodeArray():
    # Make sure ElectrodeArray can accept ints, floats, lists, np.arrays
    implants = [None] * 4
    implants[0] = e2cm.ElectrodeArray([0], [1], [2], [3],
                                      ptype='epiretinal')
    implants[1] = e2cm.ElectrodeArray(0, 1, 2, 3,
                                      ptype='epiretinal')
    implants[2] = e2cm.ElectrodeArray(0.0, [1], 2.0, [3],
                                      ptype='epiretinal')
    implants[3] = e2cm.ElectrodeArray(np.array([0]), [1], [2], [[3]],
                                      ptype='epiretinal')
    for arr in implants:
        npt.assert_equal(arr.electrodes[0].r, 0)
        npt.assert_equal(arr.electrodes[0].x, 1)
        npt.assert_equal(arr.electrodes[0].y, 2)
        # npt.assert_equal(arr.electrodes[0].h, 3)
        npt.assert_equal(arr.electrodes[0].ptype, 'epiretinal')

    # However, all input arguments must have the same number of elements
    npt.assert_raises(AssertionError, e2cm.ElectrodeArray, [0], [1, 2],
                      [3, 4, 5], [6], 'epiretinal')


def test_ArgusI():
    # Create an ArgusI and make sure location is correct
    for htype in ['float', 'list']:
        for x in [0, -100, 200]:
            for y in [0, -200, 400]:
                for r in [0, -30, 45, 60, -90]:
                    # Height `h` can either be a float or a list
                    if htype == 'float':
                        h = 100
                    else:
                        h = np.ones(16) * 20

                    # Convert rotation angle to rad
                    rot = r * np.pi / 180
                    argus = e2cm.ArgusI(x, y, h=0, rot=rot)

                    # Coordinates of first electrode
                    xy = np.array([-1200, -1200]).T

                    # Rotate
                    R = np.array([np.cos(rot), np.sin(rot),
                                  -np.sin(rot), np.cos(rot)]).reshape((2, 2))
                    xy = np.matmul(R, xy)

                    # Then off-set: Make sure first electrode is placed
                    # correctly
                    npt.assert_almost_equal(argus.electrodes[0].x, xy[0] + x)
                    npt.assert_almost_equal(argus.electrodes[0].y, xy[1] + y)

    # `h` must have the right dimensions
    npt.assert_raises(ValueError, e2cm.ArgusI, -100, 10, h=np.zeros(5))


def test_TimeSeries():
    data_orig = np.zeros((10, 10, 1000))
    ts1 = e2cm.TimeSeries(1, data_orig)
    resample_factor = 10
    ts1.resample(resample_factor)
    npt.assert_equal(ts1.data.shape[-1],
                     data_orig.shape[-1] / resample_factor)


def test_get_pulse():
    for pulse_type in ['cathodicfirst', 'anodicfirst']:
        for pulse_dur in [0.25 / 1000, 0.45 / 1000, 0.65 / 1000]:
            for interphase_dur in [0, 0.25 / 1000, 0.45 / 1000, 0.65 / 1000]:
                for tsample in [5e-6, 1e-5, 5e-5]:
                    # generate pulse
                    pulse = e2cm.get_pulse(pulse_dur, tsample,
                                           interphase_dur,
                                           pulse_type)

                    # predicted length
                    pulse_gap_dur = 2 * pulse_dur + interphase_dur

                    # make sure length is correct
                    npt.assert_equal(pulse.shape[-1],
                                     int(np.round(pulse_gap_dur /
                                                  tsample)))

                    # make sure amplitude is correct: negative peak,
                    # zero (in case of nonnegative interphase dur),
                    # positive peak
                    if interphase_dur > 0:
                        npt.assert_equal(np.unique(pulse),
                                         np.array([-1, 0, 1]))
                    else:
                        npt.assert_equal(np.unique(pulse),
                                         np.array([-1, 1]))

                    # make sure pulse order is correct
                    idx_min = np.where(pulse == pulse.min())
                    idx_max = np.where(pulse == pulse.max())
                    if pulse_type == 'cathodicfirst':
                        # cathodicfirst should have min first
                        npt.assert_equal(idx_min[0] < idx_max[0], True)
                    else:
                        npt.assert_equal(idx_min[0] < idx_max[0], False)


def test_Retina_Electrodes():
    sampling = 1
    xlo = -2
    xhi = 2
    ylo = -3
    yhi = 3
    retina = e2cm.Retina(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
                         sampling=sampling, loadpath='')
    npt.assert_equal(retina.gridx.shape, ((yhi - ylo) / sampling + 1,
                                          (xhi - xlo) / sampling + 1))
    electrode1 = e2cm.Electrode(1, 0, 0, 0, ptype='epiretinal')

    # Calculate current spread for all retinal layers
    retinal_layers = ['INL', 'NFL']
    cs = dict()
    ecs = dict()
    for layer in retinal_layers:
        cs[layer] = electrode1.current_spread(retina.gridx, retina.gridy,
                                              layer=layer)
        ecs[layer] = retina.cm2ecm(cs[layer])

    electrode_array = e2cm.ElectrodeArray([1, 1], [0, 1], [0, 1],
                                          [0, 1], ptype='epiretinal')
    npt.assert_equal(electrode1.x, electrode_array.electrodes[0].x)
    npt.assert_equal(electrode1.y, electrode_array.electrodes[0].y)
    npt.assert_equal(electrode1.r, electrode_array.electrodes[0].r)
    ecs_list, cs_list = retina.electrode_ecs(electrode_array)
    print(ecs_list.shape)

    # Make sure cs_list has an entry for every layer
    npt.assert_equal(cs_list.shape[-2], len(retinal_layers))
    npt.assert_equal(ecs_list.shape[-2], len(retinal_layers))

    # Make sure manually generated current spreads match object
    for i, e in enumerate(retinal_layers):
        # last index: electrode, second-to-last: layer
        npt.assert_equal(cs[e], cs_list[..., i, 0])


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
    m2pt = e2cm.Movie2Pulsetrain(rflum,
                                 fps=fps,
                                 amplitude_transform=amplitude_transform,
                                 amp_max=amp_max,
                                 freq=freq,
                                 pulse_dur=pulse_dur,
                                 interphase_dur=interphase_dur,
                                 tsample=tsample,
                                 pulsetype=pulsetype,
                                 stimtype=stimtype)
    npt.assert_equal(m2pt.shape[0], round((rflum.shape[-1] / fps) / tsample))
    npt.assert_(m2pt.data.max() < amp_max)


def test_Psycho2Pulsetrain():
    dur = 0.5
    pdur = 0.45 / 1000
    tsample = 5e-6
    ampl = 20
    for freq in [8, 13.8, 20]:
        for pulsetype in ['cathodicfirst', 'anodicfirst']:
            for delay in [0, 10 / 1000]:
                for pulseorder in ['pulsefirst', 'gapfirst']:
                    p2pt = e2cm.Psycho2Pulsetrain(freq=freq,
                                                  dur=dur,
                                                  pulse_dur=pdur,
                                                  interphase_dur=pdur,
                                                  delay=delay,
                                                  tsample=tsample,
                                                  amp=ampl,
                                                  pulsetype=pulsetype,
                                                  pulseorder=pulseorder)

                    # make sure length is correct
                    npt.assert_equal(p2pt.data.size,
                                     int(np.round(dur / tsample)))

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


def test_Retina_ecm():
    sampling = 1
    xlo = -2
    xhi = 2
    ylo = -3
    yhi = 3
    retina = e2cm.Retina(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
                         sampling=sampling, loadpath='')

    s1 = e2cm.Psycho2Pulsetrain(freq=20, dur=0.5, pulse_dur=.075 / 1000.,
                                interphase_dur=.075 / 1000., delay=0.,
                                tsample=.075 / 1000., amp=20,
                                pulsetype='cathodicfirst')

    electrode_array = e2cm.ElectrodeArray([1, 1], [0, 1], [0, 1],
                                          [0, 1], ptype='epiretinal')
    ecs_list, cs_list = retina.electrode_ecs(electrode_array)
    xx = yy = 0
    ecs_vector = ecs_list[yy, xx]
    # Smoke testing, feed the same stimulus through both electrodes
    stim_data = np.array([s.data for s in [s1, s1]])
    e2cm.ecm(ecs_vector, stim_data, s1.tsample)

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
    m2pt = e2cm.Movie2Pulsetrain(rflum,
                                 fps=fps,
                                 amplitude_transform=amplitude_transform,
                                 amp_max=amp_max,
                                 freq=freq,
                                 pulse_dur=pulse_dur,
                                 interphase_dur=interphase_dur,
                                 tsample=tsample,
                                 pulsetype=pulsetype,
                                 stimtype=stimtype)
    # Smoke testing, feed the same stimulus through both electrodes to
    # make sure the code runs
    stim_data = np.array([s.data for s in [m2pt, m2pt]])
    e2cm.ecm(ecs_vector, stim_data, m2pt.tsample)
