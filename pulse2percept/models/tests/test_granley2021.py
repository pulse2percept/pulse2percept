import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept.implants import ArgusI, ArgusII
from pulse2percept.percepts import Percept
from pulse2percept.stimuli import Stimulus, BiphasicPulseTrain
from pulse2percept.models import BiphasicAxonMapModel, BiphasicAxonMapSpatial, \
    AxonMapSpatial
from pulse2percept.models.granley2021 import DefaultBrightModel, \
    DefaultSizeModel, DefaultStreakModel
from pulse2percept.utils.base import FreezeError

try:
    import jax
    has_jax = True
except ImportError:
    has_jax = False

def test_effects_models():
    # Test rho scaling on size model
    model = DefaultSizeModel(200)
    npt.assert_almost_equal(
        np.sqrt(model(0.01, 0.01, 0.45) * 200 * 200), model.min_rho)

    # Test lambda scaling on streak model
    model = DefaultStreakModel(200)
    npt.assert_almost_equal(
        np.sqrt(model(10, 1, 10000) * 200 * 200), model.min_lambda)

    coeffs = {'a' + str(i): i for i in range(9)}
    # Models can take correct coeffs
    model_coeffs = {k: v for k, v in coeffs if hasattr(DefaultBrightModel(), k)}
    model = DefaultBrightModel(**model_coeffs)
    npt.assert_equal(hasattr(model, 'a0'), True)
    npt.assert_equal(hasattr(model, 'a9'), False)
    model_coeffs = {k: v for k, v in coeffs if hasattr(
        DefaultSizeModel(200), k)}
    model = DefaultSizeModel(200, **model_coeffs)
    npt.assert_equal(hasattr(model, 'a0'), True)
    npt.assert_equal(hasattr(model, 'a9'), False)
    model_coeffs = {k: v for k, v in coeffs if hasattr(
        DefaultStreakModel(200), k)}
    model = DefaultStreakModel(200, **model_coeffs)
    npt.assert_equal(hasattr(model, 'a0'), False)
    npt.assert_equal(hasattr(model, 'a9'), True)


@pytest.mark.parametrize('engine', ('serial', 'cython', 'jax'))
def test_biphasicAxonMapSpatial(engine):
    if engine == 'jax' and not has_jax:
        pytest.skip("Jax not installed")

    # Lambda cannot be too small:
    with pytest.raises(ValueError):
        BiphasicAxonMapSpatial(axlambda=9).build()

    model = BiphasicAxonMapModel(engine=engine, xystep=2).build()
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

    # Should be equal to axon map model if effects models return 1
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
    implant.stim = Stimulus({'A5': BiphasicPulseTrain(20, 1, 0.45)})
    percept = model.predict_percept(implant)
    percept_axon = axon_map.predict_percept(implant)
    npt.assert_almost_equal(
        percept.data[:, :, 0], percept_axon.max(axis='frames'))

    # Effect models must be callable
    model = BiphasicAxonMapSpatial(engine=engine, xystep=2)
    model.bright_model = 1.0
    with pytest.raises(TypeError):
        model.build()

    # If t_percept is not specified, there should only be one frame
    model = BiphasicAxonMapSpatial(engine=engine, xystep=2)
    model.build()
    implant = ArgusII()
    implant.stim = Stimulus({'A5': BiphasicPulseTrain(20, 1, 0.45)})
    percept = model.predict_percept(implant)
    npt.assert_equal(percept.time is None, True)
    # If t_percept is specified, only first frame should have data
    # and the rest should be empty
    percept = model.predict_percept(implant, t_percept=[0, 1, 2, 5, 10])
    npt.assert_equal(len(percept.time), 5)
    npt.assert_equal(np.any(percept.data[:, :, 0]), True)
    npt.assert_equal(np.any(percept.data[:, :, 1:]), False)

    # Test that default models give expected values
    model = BiphasicAxonMapSpatial(engine=engine, rho=400, axlambda=600,
                                   xystep=1, xrange=(-20, 20), yrange=(-15, 15))
    model.build()
    implant = ArgusII()
    implant.stim = Stimulus({'A4': BiphasicPulseTrain(20, 1, 1)})
    percept = model.predict_percept(implant)
    npt.assert_equal(np.sum(percept.data > 0.0813), 69)
    npt.assert_equal(np.sum(percept.data > 0.1626), 46)
    npt.assert_equal(np.sum(percept.data > 0.2439), 32)
    npt.assert_equal(np.sum(percept.data > 0.4065), 14)
    npt.assert_equal(np.sum(percept.data > 0.5691), 4)


def test_predict_spatial_jax():
    # ensure jax predict spatial is equal to normal
    if not has_jax:
        pytest.skip("Jax not installed")
    model1 = BiphasicAxonMapModel(engine='jax', xystep=2)
    model2 = BiphasicAxonMapModel(engine='cython', xystep=2)
    model1.build()
    model2.build()
    implant = ArgusII()
    implant.stim = {'A5' : BiphasicPulseTrain(25, 4, 0.45),
                    'C7' : BiphasicPulseTrain(50, 2.5, 0.75)}
    p1 = model1.predict_percept(implant)
    p2 = model2.predict_percept(implant)
    npt.assert_almost_equal(p1.data, p2.data, decimal=4)

    # test changing model parameters, make sure jax is clearing cache on build
    model1.axlambda = 800
    model2.axlambda = 800
    model1.rho = 50
    model2.rho = 50
    model1.build()
    model2.build()
    p1 = model1.predict_percept(implant)
    p2 = model2.predict_percept(implant)
    npt.assert_almost_equal(p1.data, p2.data, decimal=4)

@pytest.mark.parametrize('engine', ('serial', 'cython', 'jax'))
def test_predict_batched(engine):
    if not has_jax:
        pytest.skip("Jax not installed")

    # Allows mix of valid Stimulus types
    stims = [{'A5' : BiphasicPulseTrain(25, 4, 0.45),
              'C7' : BiphasicPulseTrain(50, 2.5, 0.75)},
             {'B4' : BiphasicPulseTrain(3, 1, 0.32)},
             Stimulus({'F3' : BiphasicPulseTrain(12, 3, 1.2)})]
    implant = ArgusII()
    model = BiphasicAxonMapModel(engine=engine, xystep=2)
    model.build()
    # Import error if we dont have jax
    if engine != 'jax':
        with pytest.raises(ImportError):
            model.predict_percept_batched(implant, stims)
        return
    
    percepts_batched = model.predict_percept_batched(implant, stims)
    percepts_serial = []
    for stim in stims:
        implant.stim = stim
        percepts_serial.append(model.predict_percept(implant))

    npt.assert_equal(len(percepts_serial), len(percepts_batched))
    for p1, p2 in zip(percepts_batched, percepts_serial):
        npt.assert_almost_equal(p1.data, p2.data)

@pytest.mark.parametrize('engine', ('serial', 'cython', 'jax'))
def test_biphasicAxonMapModel(engine):
    if engine == 'jax' and not has_jax:
        pytest.skip("Jax not installed")
    set_params = {'xystep': 2, 'engine': engine, 'rho': 432, 'axlambda': 20,
                  'n_axons': 9, 'n_ax_segments': 50,
                  'xrange': (-30, 30), 'yrange': (-20, 20),
                  'loc_od': (5, 6)}
    model = BiphasicAxonMapModel(engine=engine)
    for param in set_params:
        npt.assert_equal(hasattr(model.spatial, param), True)

    # We can set and get effects model params
    for atr in ['a' + str(i) for i in range(0, 10)]:
        npt.assert_equal(hasattr(model, atr), True)
    model.a0 = 5
    # Should propogate to size and bright model
    # But should not be a member of streak or spatial
    npt.assert_equal(model.spatial.size_model.a0, 5)
    npt.assert_equal(model.spatial.bright_model.a0, 5)
    npt.assert_equal(hasattr(model.spatial.streak_model, 'a0'), False)
    with pytest.raises(AttributeError):
        model.spatial.__getattribute__('a0')
    # If the spatial model and an effects model have a parameter with the
    # Same name, both need to be changed
    model.rho = 350
    model.axlambda = 450
    npt.assert_equal(model.spatial.size_model.rho, 350)
    npt.assert_equal(model.spatial.streak_model.axlambda, 450)
    npt.assert_equal(model.rho, 350)
    npt.assert_equal(model.axlambda, 450)

    # Effect model parameters can be passed even in constructor
    model = BiphasicAxonMapModel(engine=engine, a0=5, rho=432)
    npt.assert_equal(model.a0, 5)
    npt.assert_equal(model.spatial.bright_model.a0, 5)
    npt.assert_equal(model.rho, 432)
    npt.assert_equal(model.spatial.size_model.rho, 432)

    # If parameter is not an effects model param, it cant be set
    with pytest.raises(FreezeError):
        model.invalid_param = 5

    # Custom parameters also propogate to effects models
    model = BiphasicAxonMapModel(engine=engine)

    class TestSizeModel():
        def __init__(self):
            self.test_param = 5

        def __call__(self, freq, amp, pdur):
            return 1
    model.size_model = TestSizeModel()
    model.test_param = 10
    npt.assert_equal(model.spatial.size_model.test_param, 10)
    with pytest.raises(AttributeError):
        model.spatial.__getattribute__('test_param')

    # Values are passed correctly even in another classes __init__
    # This also tests for recursion error in another classes __init__
    class TestInitClassGood():
        def __init__(self):
            self.model = BiphasicAxonMapModel()
            # This shouldnt raise an error
            self.model.a0

    class TestInitClassBad():
        def __init__(self):
            self.model = BiphasicAxonMapModel()
            # This should
            self.model.a10 = 999
    # If this fails, something is wrong with getattr / setattr logic
    TestInitClassGood()
    with pytest.raises(FreezeError):
        TestInitClassBad()

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
