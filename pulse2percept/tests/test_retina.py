import numpy as np
import numpy.testing as npt
import logging

import pulse2percept as p2p


def test_Retina_Electrodes():
    logging.getLogger(__name__).info("test_Retina_Electrodes()")
    sampling = 1
    xlo = -2
    xhi = 2
    ylo = -3
    yhi = 3
    retina = p2p.retina.Grid(xlo=xlo, xhi=xhi,
                             ylo=ylo, yhi=yhi,
                             sampling=sampling,
                             save_data=False)
    npt.assert_equal(retina.gridx.shape, ((yhi - ylo) / sampling + 1,
                                          (xhi - xlo) / sampling + 1))
    npt.assert_equal(retina.range_x, retina.gridx.max() - retina.gridx.min())
    npt.assert_equal(retina.range_y, retina.gridy.max() - retina.gridy.min())

    electrode1 = p2p.implants.Electrode('epiretinal', 1, 0, 0, 0)

    # Calculate current spread for all retinal layers
    retinal_layers = ['INL', 'NFL']
    cs = dict()
    ecs = dict()
    for layer in retinal_layers:
        cs[layer] = electrode1.current_spread(retina.gridx, retina.gridy,
                                              layer=layer)
        ecs[layer] = retina.current2effectivecurrent(cs[layer])

    electrode_array = p2p.implants.ElectrodeArray('epiretinal', [1, 1], [0, 1],
                                                  [0, 1], [0, 1])
    npt.assert_equal(electrode1.x_center,
                     electrode_array.electrodes[0].x_center)
    npt.assert_equal(electrode1.y_center,
                     electrode_array.electrodes[0].y_center)
    npt.assert_equal(electrode1.radius, electrode_array.electrodes[0].radius)
    ecs_list, cs_list = retina.electrode_ecs(electrode_array)
    print(ecs_list.shape)

    # Make sure cs_list has an entry for every layer
    npt.assert_equal(cs_list.shape[-2], len(retinal_layers))
    npt.assert_equal(ecs_list.shape[-2], len(retinal_layers))

    # Make sure manually generated current spreads match object
    for i, e in enumerate(retinal_layers):
        # last index: electrode, second-to-last: layer
        npt.assert_equal(cs[e], cs_list[..., i, 0])


def test_Retina_ecm():
    logging.getLogger(__name__).info("test_Retina_ecm()")
    sampling = 1
    xlo = -2
    xhi = 2
    ylo = -3
    yhi = 3
    retina = p2p.retina.Grid(xlo=xlo, xhi=xhi,
                             ylo=ylo, yhi=yhi,
                             sampling=sampling,
                             save_data=False)

    s1 = p2p.stimuli.Psycho2Pulsetrain(freq=20, dur=0.5,
                                       pulse_dur=.075 / 1000.,
                                       interphase_dur=.075 / 1000., delay=0.,
                                       tsample=.075 / 1000., amp=20,
                                       pulsetype='cathodicfirst')

    electrode_array = p2p.implants.ElectrodeArray('epiretinal', [1, 1], [0, 1],
                                                  [0, 1], [0, 1])
    ecs_list, cs_list = retina.electrode_ecs(electrode_array)
    xx = yy = 0
    ecs_vector = ecs_list[yy, xx]
    # Smoke testing, feed the same stimulus through both electrodes
    stim_data = np.array([s.data for s in [s1, s1]])
    p2p.retina.effectivecurrentmap(ecs_vector, stim_data, s1.tsample)

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
    # Smoke testing, feed the same stimulus through both electrodes to
    # make sure the code runs
    stim_data = np.array([s.data for s in [m2pt, m2pt]])
    p2p.retina.effectivecurrentmap(ecs_vector, stim_data, m2pt.tsample)


def test_brightness_movie():
    logging.getLogger(__name__).info("test_brightness_movie()")

    tsample = 0.075 / 1000
    tsample_out = 1.0 / 30.0
    s1 = p2p.stimuli.Psycho2Pulsetrain(freq=20, dur=0.5,
                                       pulse_dur=.075 / 1000.,
                                       interphase_dur=.075 / 1000., delay=0.,
                                       tsample=tsample, amp=20,
                                       pulsetype='cathodicfirst')

    implant = p2p.implants.ElectrodeArray('epiretinal', [1, 1], [0, 1], [0, 1],
                                          [0, 1])

    # Smoke testing, feed the same stimulus through both electrodes:
    sim = p2p.Simulation("test movie", implant, engine='serial')

    sim.set_ofl(x_range=[-2, 2], y_range=[-3, 3], sampling=1, save_data=False)

    sim.set_gcl(tsample)

    logging.getLogger(__name__).info(" - Psycho2Pulsetrain")
    sim.pulse2percept([s1, s1], t_percept=tsample_out)


def test_debalthasar_threshold():
    # Make sure the threshold current fits for different implant heights
    logging.getLogger(__name__).info("test_debalthasar_threshold()")

    def get_baltha_pulse(curr, tsample):
        id = 0.975 / 1000
        pulse = curr * p2p.stimuli.biphasic_pulse('cathodicfirst',
                                                  0.975 / 1000,
                                                  tsample,
                                                  interphase_dur=id).data
        stim_dur = 0.1
        stim_size = int(round(stim_dur / tsample))
        pulse = np.concatenate((pulse, np.zeros(stim_size - pulse.size)))
        return p2p.utils.TimeSeries(tsample, pulse)

    def distance2threshold(el_dist):
        """Converts electrode distance (um) to threshold (uA)

        Based on linear regression of data presented in Fig. 7b of
        deBalthasar et al. (2008). Relationship is linear in log-log space.
        """
        slope = 1.5863261730600329
        intercept = -4.2496180725811659
        if el_dist > 0:
            return np.exp(np.log(el_dist) * slope + intercept)
        else:
            return np.exp(intercept)

    tsample = 0.005 / 1000

    bright = []
    for dist in np.linspace(150, 1000, 10):
        # Place the implant at various distances from the retina
        implant = p2p.implants.ElectrodeArray('epiretinal', 260, 0, 0, dist)

        sim = p2p.Simulation("debaltha", implant, engine='serial', dojit=True)
        sim.set_ofl(x_range=0, y_range=0, save_data=False,
                    streaks_enabled=False)
        sim.set_gcl(tsample)

        # For each distance to retina, find the threshold current according
        # to deBalthasar et al. (2008)
        pt = get_baltha_pulse(distance2threshold(dist), tsample)

        # Run the model
        resp = sim.pulse2percept(pt, t_percept=1.0 / 30.0)

        # Keep track of brightness
        bright.append(resp.data.max())

    # Make sure that all "threshold" currents give roughly the same brightness
    npt.assert_almost_equal(np.var(bright), 0, decimal=1)


def test_ret2dva():
    # Below 15mm eccentricity, relationship is linear with slope 3.731
    npt.assert_equal(p2p.retina.ret2dva(0.0), 0.0)
    for sign in [-1, 1]:
        for exp in [2, 3, 4]:
            ret = sign * 10 ** exp  # mm
            dva = 3.731 * sign * 10 ** (exp - 3)  # dva
            npt.assert_almost_equal(p2p.retina.ret2dva(ret), dva,
                                    decimal=3 - exp)  # adjust precision


# This function is deprecated
def test_micron2deg():
    npt.assert_almost_equal(p2p.retina.micron2deg(0.0), 0.0)
    npt.assert_almost_equal(p2p.retina.micron2deg(280.0), 1.0)


def test_dva2ret():
    # Below 50deg eccentricity, relationship is linear with slope 0.268
    npt.assert_equal(p2p.retina.dva2ret(0.0), 0.0)
    for sign in [-1, 1]:
        for exp in [-2, -1, 0]:
            dva = sign * 10 ** exp  # deg
            ret = 0.268 * sign * 10 ** (exp + 3)  # mm
            npt.assert_almost_equal(p2p.retina.dva2ret(dva), ret,
                                    decimal=-exp)  # adjust precision


# This function is deprecated
def test_deg2micron():
    npt.assert_almost_equal(p2p.retina.deg2micron(0.0), 0.0)
    npt.assert_almost_equal(p2p.retina.deg2micron(1.0), 280.0)
