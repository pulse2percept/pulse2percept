import tempfile
import numpy as np
import numpy.testing as npt

import pulse2percept.electrode2currentmap as e2cm

def test_TimeSeries():
    data_orig = np.zeros((10, 10, 1000))
    ts1 = e2cm.TimeSeries(1, data_orig)
    resample_factor = 10
    ts1.resample(resample_factor)
    npt.assert_equal(ts1.data.shape[-1],
                     data_orig.shape[-1]/resample_factor)


def test_Retina_Electrodes():
    retina_file = tempfile.NamedTemporaryFile().name
    sampling = 1
    xlo = -2
    xhi = 2
    ylo = -3
    yhi = 3
    retina = e2cm.Retina(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
                         sampling=sampling, axon_map=retina_file)
    npt.assert_equal(retina.gridx.shape, ((yhi - ylo) / sampling,
                                          (xhi - xlo) / sampling))
    electrode1 = e2cm.Electrode(1, 0, 0, 0)
    cs = electrode1.current_spread(retina.gridx, retina.gridy)
    electrode_array = e2cm.ElectrodeArray([1, 1], [0, 1], [0, 1],
                                          [0, 1])
    npt.assert_equal(electrode1.x, electrode_array.electrodes[0].x)
    npt.assert_equal(electrode1.y, electrode_array.electrodes[0].y)
    npt.assert_equal(electrode1.radius, electrode_array.electrodes[0].radius)
    ecs_list, cs_list = retina.electrode_ecs(electrode_array)
    npt.assert_equal(cs, cs_list[..., 0])


def test_Movie2Pulsetrain():
    fps = 30.0
    amplitude_transform = 'linear'
    amp_max = 90
    freq = 20
    pulse_dur = .075/1000.
    interphase_dur = .075/1000.
    tsample = .005/1000.
    pulsetype = 'cathodicfirst'
    stimtype = 'pulsetrain'
    dtype = np.int8
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
    freq = 20
    dur = 0.5
    pulse_dur = .075/1000.
    interphase_dur = .075/1000.
    delay = 0.
    tsample = .005/1000.
    current_amplitude = 20
    stimtype = 'pulsetrain'
    for dur in [1.0, 0.5]:
        for pulsetype in ['cathodicfirst', 'anodicfirst']:
            p2pt = e2cm.Psycho2Pulsetrain(freq=freq,
                                          dur=dur,
                                          pulse_dur=pulse_dur,
                                          interphase_dur=interphase_dur,
                                          delay=delay,
                                          tsample=tsample,
                                          current_amplitude=current_amplitude,
                                          pulsetype=pulsetype,
                                          stimtype=stimtype)
            npt.assert_equal(p2pt.shape[-1], int(dur / tsample))


def test_Retina_ecm():
    retina_file = tempfile.NamedTemporaryFile().name
    sampling = 1
    xlo = -2
    xhi = 2
    ylo = -3
    yhi = 3
    retina = e2cm.Retina(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
                         sampling=sampling, axon_map=retina_file)

    s1 = e2cm.Psycho2Pulsetrain(freq=20, dur=0.5, pulse_dur=.075/1000.,
                                interphase_dur=.075/1000., delay=0.,
                                tsample=.075/1000., current_amplitude=20,
                                pulsetype='cathodicfirst')

    electrode_array = e2cm.ElectrodeArray([1, 1], [0, 1], [0, 1],
                                          [0, 1])
    ecs_list, cs_list = retina.electrode_ecs(electrode_array)
    xx = yy = 0
    ecs_vector = ecs_list[yy, xx]
    # Smoke testing, feed the same stimulus through both electrodes
    stim_data = np.array([s.data for s in [s1, s1]])
    ecm = e2cm.ecm(ecs_vector, stim_data, s1.tsample)

    fps = 30.0
    amplitude_transform = 'linear'
    amp_max = 90
    freq = 20
    pulse_dur = .075/1000.
    interphase_dur = .075/1000.
    tsample = .005/1000.
    pulsetype = 'cathodicfirst'
    stimtype = 'pulsetrain'
    dtype = np.int8
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
    # Smoke testing, feed the same stimulus through both electrodes:
    stim_data = np.array([s.data for s in  [m2pt, m2pt]])
    ecm = e2cm.ecm(ecs_vector, stim_data, m2pt.tsample)
