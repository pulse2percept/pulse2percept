import numpy as np
import numpy.testing as npt
import pytest

import pulse2percept as p2p


def test_Simulation_init():
    implant = p2p.implants.Electrode("epiretinal", 10, 0, 0, 0)
    with pytest.raises(TypeError):
        p2p.Simulation("test_Simulation_set_ofl", implant)


def test_Simulation_pulse2percept():
    implant = p2p.implants.ElectrodeArray("epiretinal", 10, 0, 0, 0)
    sim = p2p.Simulation("test_Simulation_pulse2percept", implant,
                         engine='serial')
    sim.set_ofl(x_range=[0, 0], y_range=[0, 0])

    # Smoke test
    pt = p2p.stimuli.biphasic_pulse("cathodicfirst", 0.1, 0.001)
    sim.pulse2percept(pt)


def test_Simulation_set_ofl():
    sim = p2p.Simulation("test_Simulation_set_ofl", p2p.implants.ArgusI(),
                         engine='serial')

    # Invalid grid ranges
    with pytest.raises(ValueError):
        sim.set_ofl(x_range=[10, 0])
    with pytest.raises(ValueError):
        sim.set_ofl(y_range=[10, 0])

    x_range = [-100, 100]
    y_range = [0, 200]
    sim.set_ofl(x_range=x_range, y_range=y_range, save_data=False)
    npt.assert_equal(sim.ofl.gridx.min(), x_range[0])
    npt.assert_equal(sim.ofl.gridx.max(), x_range[1])
    npt.assert_equal(sim.ofl.gridy.min(), y_range[0])
    npt.assert_equal(sim.ofl.gridy.max(), y_range[1])
    npt.assert_equal(sim.ofl.range_x, np.diff(x_range))
    npt.assert_equal(sim.ofl.range_x, np.diff(x_range))

    # Smoke test
    implant = p2p.implants.ElectrodeArray('epiretinal', 10, 0, 0, 0)
    sim = p2p.Simulation("test_Simulation_set_ofl", implant, engine='serial')
    sim.set_ofl(x_range=0, y_range=0)
    sim.set_ofl(x_range=[0, 0], y_range=[0, 0])
    sim.set_ofl()


def test_get_brightest_frame():
    # Pick a few raindom frames and place brightest pixel therein
    num_frames = 10
    rand_idx = np.random.randint(0, num_frames, 5)
    for idx in rand_idx:
        # Set the brightes pixel in frame `idx` of a random vector
        tsdata = np.random.rand(2, 2, num_frames)
        tsdata[1, 1, idx] = 2.0

        # Make sure function returns the right frame
        ts = p2p.utils.TimeSeries(1, tsdata)
        brightest = p2p.get_brightest_frame(ts)
        npt.assert_equal(brightest.data.max(), tsdata.max())
        npt.assert_equal(brightest.data, tsdata[:, :, idx])
