import numpy as np
import numpy.testing as npt
import pytest
import pulse2percept.electrode2currentmap as e2cm
import pulse2percept.effectivecurrent2brightness as ec2b
from pulse2percept.utils import TimeSeries


def test_pulse2percept():
    # All ways to break the function call
    retina = e2cm.Retina(xlo=0, xhi=0, ylo=0, yhi=0, save_data=False)
    implant = e2cm.ElectrodeArray('epiretinal', 0, 0, 0, 0)
    electrode = e2cm.Electrode('epiretinal', 0, 0, 0, 0)
    pt = TimeSeries(0.001, np.ones(1000))
    tm = ec2b.TemporalModel()

    # Ways to break the function call (not including invalid pulse
    # trains, which are tested further down)
    fl_dummy = 10.2
    engine = 'serial'
    dojit = False
    with pytest.raises(TypeError):
        ec2b.pulse2percept(pt, implant, retina,
                           fl_dummy, fl_dummy, fl_dummy, True, engine, dojit)
    with pytest.raises(TypeError):
        ec2b.pulse2percept(pt, electrode,
                           tm, retina, fl_dummy, fl_dummy, fl_dummy, True,
                           engine, dojit)
    with pytest.raises(TypeError):
        ec2b.pulse2percept(pt, implant,
                           retina, retina, fl_dummy, fl_dummy, fl_dummy,
                           True, engine, dojit)

    # Smoke testing
    ec2b.pulse2percept(pt, implant, tm, retina, engine=engine, dojit=dojit)


def test_brightness_movie():
    sampling = 1
    xlo = -2
    xhi = 2
    ylo = -3
    yhi = 3
    retina = e2cm.Retina(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
                         sampling=sampling, save_data=False)

    s1 = e2cm.Psycho2Pulsetrain(freq=20, dur=0.5, pulse_dur=.075 / 1000.,
                                interphase_dur=.075 / 1000., delay=0.,
                                tsample=.075 / 1000., amp=20,
                                pulsetype='cathodicfirst')

    implant = e2cm.ElectrodeArray('epiretinal', [1, 1], [0, 1], [0, 1],
                                  [0, 1])
    tm = ec2b.TemporalModel()
    fps = 30.
    rs = int(1 / (fps * s1.tsample))

    # Smoke testing, feed the same stimulus through both electrodes:
    resp = ec2b.pulse2percept([s1, s1], implant, tm, retina, rs,
                              engine='serial', dojit=True, tol=1e-6)

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
    resp = ec2b.pulse2percept([m2pt, m2pt], implant, tm, retina, rs,
                              engine='serial', dojit=True, tol=1e-6)

    npt.assert_almost_equal(resp.tsample,
                            m2pt.tsample * rs,
                            decimal=4)


def test_debalthasar_threshold():
    # Make sure the threshold current fits for different implant heights

    def get_baltha_pulse(curr, tsample):
        pulse = curr * e2cm.get_pulse(0.975 / 1000, tsample, 0.975 / 1000,
                                      'cathodicfirst')
        stim_dur = 0.1
        stim_size = int(round(stim_dur / tsample))
        pulse = np.concatenate((pulse, np.zeros(stim_size - pulse.size)))
        return TimeSeries(tsample, pulse)

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
    tm = ec2b.TemporalModel(tsample)

    # Single-pixel retina
    retina = e2cm.Retina(xlo=0, xhi=0, ylo=0, yhi=0, save_data=False)

    bright = []
    for dist in np.linspace(150, 1000, 10):
        # Place the implant at various distances from the retina
        implant = e2cm.ElectrodeArray('epiretinal', 260, 0, 0, dist)

        # For each distance to retina, find the threshold current according
        # to deBalthasar et al. (2008)
        pt = get_baltha_pulse(distance2threshold(dist), tsample)

        # Run the model
        resp = ec2b.pulse2percept([pt], implant, tm, retina,
                                  use_ecs=False, engine='serial', dojit=True,
                                  rsample=30, tol=1e-6)

        # Keep track of brightness
        bright.append(resp.data.max())

    # Make sure that all "threshold" currents give roughly the same brightness
    npt.assert_almost_equal(np.var(bright), 0, decimal=1)


def test_parse_pulse_trains():
    # Specify pulse trains in a number of different ways and make sure they
    # are all identical after parsing

    # Create some implants
    argus = e2cm.ArgusI()
    simple = e2cm.ElectrodeArray('subretinal', 0, 0, 0, 0)

    pt_zero = TimeSeries(1, np.zeros(1000))
    pt_nonzero = TimeSeries(1, np.random.rand(1000))

    # Test 1
    # ------
    # Specify wrong number of pulse trains
    with pytest.raises(ValueError):
        ec2b.parse_pulse_trains(pt_nonzero, argus)
    with pytest.raises(ValueError):
        ec2b.parse_pulse_trains([pt_nonzero], argus)
    with pytest.raises(ValueError):
        ec2b.parse_pulse_trains([pt_nonzero] * (argus.num_electrodes - 1),
                                argus)
    with pytest.raises(ValueError):
        ec2b.parse_pulse_trains([pt_nonzero] * 2, simple)

    # Test 2
    # ------
    # Send non-zero pulse train to specific electrode
    el_name = 'B3'
    el_idx = argus.get_index(el_name)

    # Specify a list of 16 pulse trains (one for each electrode)
    pt0_in = [pt_zero] * argus.num_electrodes
    pt0_in[el_idx] = pt_nonzero
    pt0_out = ec2b.parse_pulse_trains(pt0_in, argus)

    # Specify a dict with non-zero pulse trains
    pt1_in = {el_name: pt_nonzero}
    pt1_out = ec2b.parse_pulse_trains(pt1_in, argus)

    # Make sure the two give the same result
    for p0, p1 in zip(pt0_out, pt1_out):
        npt.assert_equal(p0.data, p1.data)

    # Test 3
    # ------
    # Smoke testing
    ec2b.parse_pulse_trains([pt_zero] * argus.num_electrodes, argus)
    ec2b.parse_pulse_trains(pt_zero, simple)
    ec2b.parse_pulse_trains([pt_zero], simple)
