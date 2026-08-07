import copy
import warnings

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

# Building an axon map writes a cache to a relative path; keep it in a
# temporary directory instead of wherever pytest was started from:
pytestmark = pytest.mark.usefixtures('axon_cache_in_tmp')


def test_deepcopy_DefaultBrightModel():
    original = DefaultBrightModel()
    copied = copy.deepcopy(original)

    # Assert these are two different objects
    npt.assert_equal(id(original) != id(copied), True)

    # Assert the objects are equivalent
    npt.assert_equal(original.__dict__, copied.__dict__)

    # Assert changing copied doesn't change original
    copied.a4 = 5
    npt.assert_equal(original.a4 != copied.a4, True)


def test_deepcopy_DefaultSizeModel():
    original = DefaultSizeModel(rho=0)
    copied = copy.deepcopy(original)

    # Assert these are two different objects
    npt.assert_equal(id(original) != id(copied), True)

    # Assert the objects are equivalent
    npt.assert_equal(original.__dict__, copied.__dict__)

    # Assert changing copied doesn't change original
    copied.a0 = 5
    npt.assert_equal(original.a0 != copied.a0, True)

def test_deepcopy_DefaultStreakModel():
    original = DefaultStreakModel(200)
    copied = copy.deepcopy(original)

    # Assert these are two different objects
    npt.assert_equal(id(original) != id(copied), True)

    # Assert the objects are equivalent
    npt.assert_equal(original.__dict__, copied.__dict__)

    # Assert changing copied doesn't change original
    copied.a7 = 5
    npt.assert_equal(original.a7 != copied.a7, True)


def test_eq_DefaultStreakModel():
    model = DefaultStreakModel(axlambda=200)

    # Assert not equal for differing classes
    npt.assert_equal(model == DefaultSizeModel, False)

    # Assert equal to itself
    npt.assert_equal(model == model, True)

    # Assert equal for shallow references
    copied = model
    npt.assert_equal(model == copied, True)

    # Assert deep copies are equal
    copied = copy.deepcopy(model)
    npt.assert_equal(model == copied, True)

    # Assert different models do not equal each other
    differing_model = DefaultStreakModel(axlambda=300)
    npt.assert_equal(model != differing_model, True)


def test_eq_DefaultSizeModel():
    model = DefaultSizeModel(rho=1)

    # Assert not equal for differing classes
    npt.assert_equal(model == DefaultSizeModel, False)

    # Assert equal to itself
    npt.assert_equal(model == model, True)

    # Assert equal for shallow references
    copied = model
    npt.assert_equal(model == copied, True)

    # Assert deep copies are equal
    copied = copy.deepcopy(model)
    npt.assert_equal(model == copied, True)

    # Assert different models do not equal each other
    differing_model = DefaultSizeModel(rho=2)
    npt.assert_equal(model != differing_model, True)


def test_deepcopy_BiphasicAxonMapSpatial():
    original = BiphasicAxonMapSpatial()
    copied = copy.deepcopy(original)

    # Assert these are two different objects
    npt.assert_equal(id(original) != id(copied), True)

    # Assert the objects are equivalent
    npt.assert_equal(original == copied, True)
    npt.assert_equal(original == copied, True)

    # Assert changing copied doesn't change original
    copied.bright_model = None
    npt.assert_equal(original.bright_model != copied.bright_model, True)


def test_deepcopy_BiphasicAxonMapModel():
    original = BiphasicAxonMapModel()
    copied = copy.deepcopy(original)

    # Assert these are two different objects
    npt.assert_equal(id(original) != id(copied), True)

    # Assert the objects are equivalent
    npt.assert_equal(original.__dict__, copied.__dict__)

    # Assert changing copied doesn't change original
    copied.spatial.axlambda = 200
    npt.assert_equal(original.spatial != copied.spatial, True)

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


@pytest.mark.parametrize('cls, arg', [(DefaultSizeModel, 200),
                                      (DefaultStreakModel, 200)])
def test_effects_models_deprecated_engine(cls, arg):
    # 'engine' used to switch between the numpy and the (now removed) jax
    # backend. It is still accepted, but ignored:
    with pytest.deprecated_call():
        deprecated = cls(arg, engine='serial')
    # Passing it changes nothing about the model's output:
    npt.assert_almost_equal(deprecated(20, 1, 0.45), cls(arg)(20, 1, 0.45))
    # Even a value that used to select a different backend is a no-op:
    with pytest.deprecated_call():
        cls(arg, engine='jax')
    # Not passing it does not warn:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        cls(arg)


def test_biphasicAxonMapSpatial():
    # Lambda cannot be too small:
    with pytest.raises(ValueError):
        BiphasicAxonMapSpatial(axlambda=9).build()

    model = BiphasicAxonMapModel(xystep=2).build()
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
    model = BiphasicAxonMapSpatial(xystep=2)
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
    model = BiphasicAxonMapSpatial(xystep=2)
    model.bright_model = 1.0
    with pytest.raises(TypeError):
        model.build()

    # If t_percept is not specified, there should only be one frame
    model = BiphasicAxonMapSpatial(xystep=2)
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
    model = BiphasicAxonMapSpatial(rho=400, axlambda=600,
                                   xystep=1, xrange=(-20, 20), yrange=(-15, 15))
    model.build()
    implant = ArgusII()
    implant.stim = Stimulus({'A4': BiphasicPulseTrain(20, 1, 1)})
    percept = model.predict_percept(implant)
    npt.assert_equal(np.sum(percept.data > 0.0813), 70)
    npt.assert_equal(np.sum(percept.data > 0.1626), 50)
    npt.assert_equal(np.sum(percept.data > 0.2439), 33)
    npt.assert_equal(np.sum(percept.data > 0.4065), 16)
    npt.assert_equal(np.sum(percept.data > 0.5691), 4)


def test_biphasicAxonMapModel():
    set_params = {'xystep': 2, 'rho': 432, 'axlambda': 20,
                  'n_axons': 9, 'n_ax_segments': 50,
                  'xrange': (-30, 30), 'yrange': (-20, 20),
                  'loc_od': (5, 6)}
    model = BiphasicAxonMapModel()
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
    model = BiphasicAxonMapModel(a0=5, rho=432)
    npt.assert_equal(model.a0, 5)
    npt.assert_equal(model.spatial.bright_model.a0, 5)
    npt.assert_equal(model.rho, 432)
    npt.assert_equal(model.spatial.size_model.rho, 432)

    # If parameter is not an effects model param, it cant be set
    with pytest.raises(FreezeError):
        model.invalid_param = 5

    # Custom parameters also propogate to effects models
    model = BiphasicAxonMapModel()

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
    model = BiphasicAxonMapModel()
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


@pytest.mark.parametrize('model_cls', [BiphasicAxonMapModel,
                                       BiphasicAxonMapSpatial])
def test_find_threshold_not_supported(model_cls):
    # This model takes amplitude as a multiple of threshold and reads it from
    # the stimulus metadata, so the inherited `find_threshold` - which bisects
    # on a scaled copy of the stimulus data - cannot converge. It has to say
    # so rather than fail somewhere deeper with a confusing message.
    model = model_cls(xrange=(-3, 3), yrange=(-2, 2), xystep=1,
                      n_ax_segments=30).build()
    implant = ArgusII(stim={'A1': BiphasicPulseTrain(20, 1, 0.45,
                                                     stim_dur=100)})
    with pytest.raises(NotImplementedError) as excinfo:
        model.find_threshold(implant, 0.5)
    npt.assert_equal(model_cls.__name__ in str(excinfo.value), True)
    npt.assert_equal('metadata' in str(excinfo.value), True)
    # predict_percept is unaffected:
    npt.assert_equal(model.predict_percept(implant) is not None, True)
