import numpy as np
import numpy.testing as npt
import pytest

import pulse2percept as p2p


def test_Simulation_init():
    implant = p2p.implants.Electrode("epiretinal", 10, 0, 0, 0)
    with pytest.raises(TypeError):
        p2p.Simulation(implant)


def test_Simulation_pulse2percept():
    implant = p2p.implants.ElectrodeArray("epiretinal", 10, 0, 0, 0)
    sim = p2p.Simulation(implant, engine='serial')
    sim.set_optic_fiber_layer(x_range=[0, 0], y_range=[0, 0])

    # PulseTrain must have the same tsample as (implicitly set up) GCL
    pt = p2p.stimuli.BiphasicPulse("cathodicfirst", 0.1, 0.001)
    with pytest.raises(ValueError):
        sim.pulse2percept(pt)

    pt = p2p.stimuli.BiphasicPulse("cathodicfirst", 0.1, 0.005 / 1000)
    with pytest.raises(ValueError):
        sim.pulse2percept(pt, layers=['GCL', 'invalid'])


def test_Simulation_set_optic_fiber_layer():
    sim = p2p.Simulation(p2p.implants.ArgusI(), engine='serial')

    # Invalid grid ranges
    with pytest.raises(ValueError):
        sim.set_optic_fiber_layer(x_range=[10, 0])
    with pytest.raises(ValueError):
        sim.set_optic_fiber_layer(x_range=[1, 2, 3])
    with pytest.raises(ValueError):
        sim.set_optic_fiber_layer(x_range='invalid')
    with pytest.raises(ValueError):
        sim.set_optic_fiber_layer(y_range=[10, 0])
    with pytest.raises(ValueError):
        sim.set_optic_fiber_layer(y_range=[1, 2, 3])
    with pytest.raises(ValueError):
        sim.set_optic_fiber_layer(y_range='invalid')

    x_range = [-100, 100]
    y_range = [0, 200]
    sim.set_optic_fiber_layer(x_range=x_range, y_range=y_range,
                              save_data=False)
    npt.assert_equal(sim.ofl.gridx.min(), x_range[0])
    npt.assert_equal(sim.ofl.gridx.max(), x_range[1])
    npt.assert_equal(sim.ofl.gridy.min(), y_range[0])
    npt.assert_equal(sim.ofl.gridy.max(), y_range[1])
    npt.assert_equal(sim.ofl.range_x, np.diff(x_range))
    npt.assert_equal(sim.ofl.range_x, np.diff(x_range))

    # Smoke test
    implant = p2p.implants.ElectrodeArray('epiretinal', 10, 0, 0, 0)
    sim = p2p.Simulation(implant, engine='serial')
    sim.set_optic_fiber_layer(x_range=0, y_range=0)
    sim.set_optic_fiber_layer(x_range=[0, 0], y_range=[0, 0])
    sim.set_optic_fiber_layer()


def test_Simulation_set_ganglion_cell_layer():
    # A valid ganglion cell model
    class Valid(p2p.retina.BaseModel):

        def model_cascade(self, inval):
            return inval

    # Smoke test custom model
    implant = p2p.implants.ElectrodeArray('epiretinal', 10, 0, 0, 0)
    sim = p2p.Simulation(implant, engine='serial')
    sim.set_optic_fiber_layer(x_range=0, y_range=0)

    valid_model = Valid(tsample=0.2)
    sim.set_ganglion_cell_layer(valid_model)
    npt.assert_equal(isinstance(sim.gcl, p2p.retina.BaseModel), True)
    npt.assert_equal(sim.gcl.tsample, 0.2)

    # Smoke test Nanduri model
    for modelstr in ['Nanduri2012', 'nanduri2012', 'NANDURI2012']:
        sim.set_ganglion_cell_layer(modelstr, tau3=42)
        npt.assert_equal(isinstance(sim.gcl, p2p.retina.Nanduri2012), True)
        npt.assert_equal(sim.gcl.tau3, 42)
        sim.set_ganglion_cell_layer(modelstr, unknown_param=2)  # smoke

    # Smoke test latest model
    for modelstr in ['latest', 'Latest', 'LATEST']:
        sim.set_ganglion_cell_layer(modelstr, lweight=42)
        npt.assert_equal(isinstance(sim.gcl, p2p.retina.TemporalModel), True)
        npt.assert_equal(sim.gcl.lweight, 42)
        sim.set_ganglion_cell_layer(modelstr, unknown_param=2)  # smoke

    # Model unknown
    with pytest.raises(ValueError):
        sim.set_ganglion_cell_layer('unknown-model')


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
