import numpy as np
import numpy.testing as npt
import pulse2percept.electrode2currentmap as e2cm
import pulse2percept.effectivecurrent2brightness as ec2b


def test_brightness_movie():
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

    electrode_array = e2cm.ElectrodeArray([1, 1], [0, 1], [0, 1], [0, 1],
                                          ptype='epiretinal')
    ecs, cs = retina.electrode_ecs(electrode_array)
    temporal_model = ec2b.TemporalModel()
    fps = 30.
    rs = int(1 / (fps * s1.tsample))

    # Smoke testing, feed the same stimulus through both electrodes:
    brightness_movie = ec2b.pulse2percept(temporal_model, ecs, retina,
                                          [s1, s1], rs, dolayer='INL')

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

    rs = int(1 / (fps * m2pt.tsample))
    # Smoke testing, feed the same stimulus through both electrodes:
    brightness_movie = ec2b.pulse2percept(temporal_model, ecs, retina,
                                          [m2pt, m2pt], rs, dolayer='INL')

    npt.assert_almost_equal(brightness_movie.tsample,
                            m2pt.tsample * rs,
                            decimal=4)
