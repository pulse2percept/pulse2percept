import tempfile
import numpy as np
import numpy.testing as npt

import electrode2currentmap as e2cm


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
    electrode = e2cm.Electrode(1, 0, 0)
    ecs = electrode.current_spread(retina.gridx, retina.gridy)


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
                                 stimtype=stimtype,
                                 dtype=dtype)
    npt.assert_equal(m2pt.shape[0], round((rflum.shape[-1] / fps) / tsample))


def test_Psycho2Pulsetrain():
    freq=20
    dur=0.5
    pulse_dur=.075/1000.
    interphase_dur=.075/1000.
    delay=0.
    tsample=.005/1000.
    current_amplitude=20
    pulsetype='cathodicfirst'
    stimtype='pulsetrain'
    p2pt = e2cm.Psycho2Pulsetrain(freq=freq, dur=dur, pulse_dur=pulse_dur,
                                  interphase_dur=interphase_dur,
                                  delay=delay,
                                  tsample=tsample,
                                  current_amplitude=current_amplitude,
                                  pulsetype=pulsetype,
                                  stimtype=stimtype)
    npt.assert_equal(p2pt.shape[-1], round(dur / tsample))
