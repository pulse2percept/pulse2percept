import numpy as np
import numpy.testing as npt
import pytest

from .. import api as p2p
from .. import implants
from .. import stimuli
from .. import utils


def test_Simulation___init__():
    implant = implants.Electrode("epiretinal", 10, 0, 0, 0)
    with pytest.raises(TypeError):
        p2p.Simulation(implant)


def test_Simulation_pulse2percept():
    implant = implants.ElectrodeArray("epiretinal", 10, 0, 0, 0)
    sim = p2p.Simulation(implant, engine='serial')
    sim.set_optic_fiber_layer(x_range=[0, 0], y_range=[0, 0])
    pt = stimuli.BiphasicPulse('cathodicfirst', 0.45 / 1000, 0.005 / 1000)
    sim.pulse2percept(pt)
    sim.pulse2percept(pt, layers=['GCL'])
    sim.pulse2percept(pt, layers=['INL'])

    # PulseTrain must have the same tsample as (implicitly set up) GCL
    pt = stimuli.BiphasicPulse("cathodicfirst", 0.1, 0.001)
    with pytest.raises(ValueError):
        sim.pulse2percept(pt)

    pt = stimuli.BiphasicPulse("cathodicfirst", 0.1, 0.005 / 1000)
    with pytest.raises(ValueError):
        sim.pulse2percept(pt, layers=['GCL', 'invalid'])


def test_Simulation_set_optic_fiber_layer():
    sim = p2p.Simulation(implants.ArgusI(), engine='serial')

    # Invalid grid ranges
    with pytest.raises(ValueError):
        sim.set_optic_fiber_layer(x_range=(10, 0))
    with pytest.raises(ValueError):
        sim.set_optic_fiber_layer(x_range=(1, 2, 3))
    with pytest.raises(ValueError):
        sim.set_optic_fiber_layer(x_range='invalid')
    with pytest.raises(ValueError):
        sim.set_optic_fiber_layer(y_range=(10, 0))
    with pytest.raises(ValueError):
        sim.set_optic_fiber_layer(y_range=(1, 2, 3))
    with pytest.raises(ValueError):
        sim.set_optic_fiber_layer(y_range='invalid')

    x_range = (-100, 100)
    y_range = (0, 200)
    sim.set_optic_fiber_layer(x_range=x_range, y_range=y_range,
                              save_data=False, alpha=14000)
    npt.assert_equal(sim.ofl.gridx.min(), x_range[0])
    npt.assert_equal(sim.ofl.gridx.max(), x_range[1])
    npt.assert_equal(sim.ofl.gridy.min(), y_range[0])
    npt.assert_equal(sim.ofl.gridy.max(), y_range[1])
    npt.assert_equal(sim.ofl.x_range, x_range)
    npt.assert_equal(sim.ofl.y_range, y_range)
    npt.assert_equal(sim.ofl.alpha, 14000)

    # Smoke test
    implant = implants.ElectrodeArray('epiretinal', 10, 0, 0, 0)
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
    implant = implants.ElectrodeArray('epiretinal', 10, 0, 0, 0)
    sim = p2p.Simulation(implant, engine='serial')
    sim.set_optic_fiber_layer(x_range=0, y_range=0)

    valid_model = Valid(tsample=0.2)
    sim.set_ganglion_cell_layer(valid_model)
    npt.assert_equal(isinstance(sim.gcl, p2p.retina.BaseModel), True)
    npt.assert_equal(sim.gcl.tsample, 0.2)

    # Smoke test latest model
    for modelstr in ['latest', 'Latest', 'LATEST']:
        sim.set_ganglion_cell_layer(modelstr, lweight=42)
        npt.assert_equal(isinstance(sim.gcl, p2p.retina.TemporalModel), True)
        npt.assert_equal(sim.gcl.lweight, 42)
        sim.set_ganglion_cell_layer(modelstr, unknown_param=2)  # smoke

    # Smoke test Nanduri model
    for modelstr in ['Nanduri2012', 'nanduri2012', 'NANDURI2012']:
        sim.set_ganglion_cell_layer(modelstr, tau3=42)
        npt.assert_equal(isinstance(sim.gcl, p2p.retina.Nanduri2012), True)
        npt.assert_equal(sim.gcl.tau3, 42)
        sim.set_ganglion_cell_layer(modelstr, unknown_param=2)  # smoke

    # Smoke test Horsager model
    for modelstr in ['Horsager2009', 'horsager', 'HORSAGER2009']:
        sim.set_ganglion_cell_layer(modelstr, tau3=42)
        npt.assert_equal(isinstance(sim.gcl, p2p.retina.Horsager2009), True)
        npt.assert_equal(sim.gcl.tau3, 42)
        sim.set_ganglion_cell_layer(modelstr, unknown_param=2)  # smoke

    # Model unknown
    with pytest.raises(ValueError):
        sim.set_ganglion_cell_layer('unknown-model')
    with pytest.raises(ValueError):
        sim.set_ganglion_cell_layer(implants.ArgusII())


def test_get_brightest_frame():
    # Pick a few raindom frames and place brightest pixel therein
    num_frames = 10
    rand_idx = np.random.randint(0, num_frames, 5)
    for idx in rand_idx:
        # Set the brightes pixel in frame `idx` of a random vector
        tsdata = np.random.rand(2, 2, num_frames)
        tsdata[1, 1, idx] = 2.0

        # Make sure function returns the right frame
        ts = utils.TimeSeries(1, tsdata)
        brightest = p2p.get_brightest_frame(ts)
        npt.assert_equal(brightest.data.max(), tsdata.max())
        npt.assert_equal(brightest.data, tsdata[:, :, idx])
