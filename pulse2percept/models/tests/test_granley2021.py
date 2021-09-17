from multiprocessing import Value
from typing import Type
from pulse2percept.models.granley2021 import DefaultBrightModel, \
                                             DefaultSizeModel, DefaultStreakModel
from pulse2percept.utils.base import FreezeError
import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept.implants import ArgusI, ArgusII
from pulse2percept.percepts import Percept
from pulse2percept.stimuli import Stimulus, BiphasicPulseTrain
from pulse2percept.models import BiphasicAxonMapModel, BiphasicAxonMapSpatial, \
                                 AxonMapSpatial
from pulse2percept.utils.testing import assert_warns_msg


def test_effects_models():
    # Test thresholding on bright model
    model = DefaultBrightModel(do_thresholding=True)
    # Technically this could fail, but the probability is negliglible
    npt.assert_almost_equal(model(20, 0.01, 0.45), 0)

    # Test rho scaling on size model
    model = DefaultSizeModel(200)
    npt.assert_almost_equal(np.sqrt(model(0.01, 0.01, 0.45) * 200 * 200), model.min_rho)

    # Test lambda scaling on streak model
    model = DefaultStreakModel(200)
    npt.assert_almost_equal(np.sqrt(model(10, 1, 10000) * 200 * 200), model.min_lambda)

    coeffs = {'a' + str(i) : i for i in range(9)}
    # Models can take any coeffs, but only set the ones that they have
    model = DefaultBrightModel(**coeffs)
    npt.assert_equal(hasattr(model, 'a0'), True)
    npt.assert_equal(hasattr(model, 'a9'), False)
    model = DefaultSizeModel(200, **coeffs)
    npt.assert_equal(hasattr(model, 'a0'), True)
    npt.assert_equal(hasattr(model, 'a9'), False)
    model = DefaultStreakModel(200, **coeffs)
    npt.assert_equal(hasattr(model, 'a0'), False)
    npt.assert_equal(hasattr(model, 'a9'), True) 


@pytest.mark.parametrize('engine', ('serial', 'cython', 'jax'))
def test_biphasicAxonMapSpatial(engine):
    # Lambda cannot be too small:
    with pytest.raises(ValueError):
        BiphasicAxonMapSpatial(axlambda=9).build()

    model = BiphasicAxonMapModel(engine=engine, xystep=2).build()
    # Jax not implemented yet
    if engine == 'jax':
        with pytest.raises(NotImplementedError):
            implant = ArgusII()
            implant.stim = Stimulus({'A5' : BiphasicPulseTrain(20, 1, 0.45)})
            percept = model.predict_percept(implant)
        return

    # Only accepts biphasic pulse trains with no delay dur
    implant = ArgusI(stim=np.ones(16))
    with pytest.raises(TypeError):
        model.predict_percept(implant)

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(ArgusI()), None)

    # Zero in = zero out:
    implant = ArgusI(stim=np.zeros(16))
    percept = model.predict_percept(implant)
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [1])
    npt.assert_almost_equal(percept.data, 0)
    npt.assert_equal(percept.time, None)

    # Should be exactly equal to axon map model if effects models return 1
    model = BiphasicAxonMapSpatial(engine=engine, xystep=2)
    def bright_model(freq, amp, pdur): return 1
    def size_model(freq, amp, pdur): return 1
    def streak_model(freq, amp, pdur): return 1
    model.bright_model = bright_model
    model.size_model = size_model
    model.streak_model = streak_model
    model.build()
    axon_map = AxonMapSpatial(xystep=2).build()
    implant = ArgusII()
    implant.stim = Stimulus({'A5' : BiphasicPulseTrain(20, 1, 0.45)})
    percept = model.predict_percept(implant)
    percept_axon = axon_map.predict_percept(implant)
    npt.assert_almost_equal(percept.data[:, :, 0], percept_axon.get_brightest_frame())

    # Effect models must be callable
    model = BiphasicAxonMapSpatial(engine=engine, xystep=2)
    model.bright_model = 1.0
    with pytest.raises(AssertionError):
        model.build()

    # If t_percept is not specified, there should only be one frame 
    model = BiphasicAxonMapSpatial(engine=engine, xystep=2)
    model.build()
    implant = ArgusII()
    implant.stim = Stimulus({'A5' : BiphasicPulseTrain(20, 1, 0.45)})
    percept = model.predict_percept(implant)
    npt.assert_equal(percept.time is None, True)
    # If t_percept is specified, only first frame should have data
    # and the rest should be empty
    percept = model.predict_percept(implant, t_percept=[0, 1, 2, 5, 10])
    npt.assert_equal(len(percept.time), 5)
    npt.assert_equal(np.any(percept.data[:, :, 0]), True)
    npt.assert_equal(np.any(percept.data[:, :, 1:]), False)

    # Test that default models give expected values
    model = BiphasicAxonMapSpatial(engine=engine, rho=600, axlambda=600, xystep=1,
                           xrange=(-20, 20), yrange=(-15, 15))
    model.build()
    implant = ArgusII()
    implant.stim = Stimulus({'A4' : BiphasicPulseTrain(20, 1, 1)})
    percept = model.predict_percept(implant)
    npt.assert_equal(np.sum(percept.data > 1), 53)
    npt.assert_equal(np.sum(percept.data > 2), 36)
    npt.assert_equal(np.sum(percept.data > 3), 25)
    npt.assert_equal(np.sum(percept.data > 5), 12)
    npt.assert_equal(np.sum(percept.data > 7), 4)


@pytest.mark.parametrize('engine', ('serial', 'cython', 'jax'))
def test_biphasicAxonMapModel(engine):
    set_params = {'xystep': 2, 'engine': engine, 'rho': 432, 'axlambda': 20,
                  'n_axons': 9, 'n_ax_segments': 50,
                  'xrange': (-30, 30), 'yrange': (-20, 20),
                  'loc_od': (5, 6), 'do_thresholding' : False}
    model = BiphasicAxonMapModel(engine=engine)
    for param in set_params:
        npt.assert_equal(hasattr(model.spatial, param), True)

    # We can set and get effects model params, but only AFTER build
    assert(model.bright_model is None)
    with pytest.raises(FreezeError):
        model.a0 = 5
    model.build()
    model.a0 = 5
    # Should propogate to size and bright model
    # But should not be a member of streak or spatial
    npt.assert_equal(model.spatial.size_model.a0, 5)
    npt.assert_equal(model.spatial.bright_model.a0, 5)
    npt.assert_equal(hasattr(model.spatial.streak_model, 'a0'), False)
    with pytest.raises(AttributeError):
        model.spatial.__getattribute__('a0')

    # User can override default values
    model = BiphasicAxonMapModel(engine=engine)
    for key, value in set_params.items():
        setattr(model.spatial, key, value)
        npt.assert_equal(getattr(model.spatial, key), value)
    model = BiphasicAxonMapModel(**set_params)
    model.build(**set_params)
    for key, value in set_params.items():
        npt.assert_equal(getattr(model.spatial, key), value)

    # Zeros in, zeros out:
    implant = ArgusII(stim=np.zeros(60))
    npt.assert_almost_equal(model.predict_percept(implant).data, 0)
    implant.stim = np.zeros(60)
    npt.assert_almost_equal(model.predict_percept(implant).data, 0)

    # Implant and model must be built for same eye:
    with pytest.raises(ValueError):
        implant = ArgusII(eye='LE', stim=np.zeros(60))
        model.predict_percept(implant)
    with pytest.raises(ValueError):
        BiphasicAxonMapModel(eye='invalid').build()
    with pytest.raises(ValueError):
        BiphasicAxonMapModel(xystep=5).build(eye='invalid')

    # Lambda cannot be too small:
    with pytest.raises(ValueError):
        BiphasicAxonMapModel(axlambda=9).build()
