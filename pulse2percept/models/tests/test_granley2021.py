from contextlib import contextmanager
import copy

import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept.implants import ArgusI, ArgusII
from pulse2percept.percepts import Percept
from pulse2percept.stimuli import (AmplitudeEncoder,
                                   AsymmetricBiphasicPulseTrain,
                                   BiphasicPulseTrain, ImageStimulus,
                                   MonophasicPulse, Stimulus)
from pulse2percept.models import (AlphaTemporal, AxonMapSpatial,
                                  BiphasicAxonMapModel,
                                  BiphasicAxonMapSpatial, FadingTemporal,
                                  Horsager2009Temporal, Model,
                                  Nanduri2012Temporal)
from pulse2percept.models.granley2021 import DefaultBrightModel, \
    DefaultSizeModel, DefaultStreakModel
from pulse2percept.units import (DimensionMismatchError, Quantity,
                                 dimensionless, mm, ms, s, uA, um,
                                 xTh)
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
    model = DefaultStreakModel(lam=200)

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
    differing_model = DefaultStreakModel(lam=300)
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
    original = BiphasicAxonMapSpatial(implant=ArgusII())
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
    original = BiphasicAxonMapModel(implant=ArgusII())
    copied = copy.deepcopy(original)

    # Assert these are two different objects
    npt.assert_equal(id(original) != id(copied), True)

    # Assert the objects are equivalent
    npt.assert_equal(original.__dict__, copied.__dict__)

    # Assert changing copied doesn't change original
    copied.spatial.lam = 200
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


def test_effects_models_units():
    # `rho` and `lam` are constructor arguments rather than entries in
    # `get_default_params`, but they are still lengths, and are still
    # normalized before being stored:
    size = DefaultSizeModel(0.2 * mm, min_rho=20 * um)
    npt.assert_almost_equal(size.rho, 200)
    npt.assert_almost_equal(size.min_rho, 20)
    streak = DefaultStreakModel(0.5 * mm, min_lambda=20 * um)
    npt.assert_almost_equal(streak.lam, 500)
    npt.assert_almost_equal(streak.min_lambda, 20)
    # Plain numbers, not quantities, so the equations can use them:
    for value in (size.rho, size.min_rho, streak.lam, streak.min_lambda):
        npt.assert_equal(isinstance(value, Quantity), False)
        npt.assert_equal(isinstance(value, (int, float)), True)
    # Both spellings give the same scaling factor:
    npt.assert_almost_equal(DefaultSizeModel(0.2 * mm)(20, 1, 0.45),
                            DefaultSizeModel(200)(20, 1, 0.45))
    npt.assert_almost_equal(DefaultStreakModel(0.5 * mm)(20, 1, 0.45),
                            DefaultStreakModel(500)(20, 1, 0.45))
    # And a current is not a length:
    with pytest.raises(DimensionMismatchError):
        DefaultSizeModel(200 * uA)
    with pytest.raises(DimensionMismatchError):
        DefaultStreakModel(500 * ms)


@pytest.mark.parametrize('cls, arg', [(DefaultSizeModel, 200),
                                      (DefaultStreakModel, 200)])
def test_effects_models_removed_engine(cls, arg):
    # 'engine' used to switch between the numpy and the (now removed) jax
    # backend. Deprecated in 0.9.1, removed in 0.10.0:
    with pytest.raises(AttributeError):
        cls(arg, engine='serial')


def test_biphasicAxonMapSpatial():
    # Lambda cannot be too small:
    with pytest.raises(ValueError):
        BiphasicAxonMapSpatial(implant=ArgusII(), lam=9).build()

    model = BiphasicAxonMapModel(implant=ArgusII(), step=2).build()
    # Only accepts biphasic pulse trains with no delay dur
    with pytest.raises(TypeError):
        model.predict_percept(np.ones(60))

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(None), None)

    # Zero in = zero out:
    source = np.zeros(60)
    percept = model.predict_percept(source)
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [1])
    npt.assert_almost_equal(percept.data, 0)
    npt.assert_equal(percept.time, None)

    # Should be equal to axon map model if effects models return 1
    model = BiphasicAxonMapSpatial(implant=ArgusII(), step=2)
    def bright_model(freq, amp, pdur): return 1
    def size_model(freq, amp, pdur): return 1
    def streak_model(freq, amp, pdur): return 1
    model.bright_model = bright_model
    model.size_model = size_model
    model.streak_model = streak_model
    model.build()
    axon_map = AxonMapSpatial(implant=ArgusII(), step=2).build()
    # The axon map reads current and this model reads threshold multiples, so
    # a 1 uA threshold is what makes the two numbers the same one:
    source = Stimulus({'A5': BiphasicPulseTrain(20, 1 * xTh, 0.45,
                                                      threshold_amp=1 * uA)})
    percept = model.predict_percept(source)
    percept_axon = axon_map.predict_percept(source)
    npt.assert_almost_equal(
        percept.data[:, :, 0], percept_axon.max(axis='frames'))

    # Effect models must be callable
    model = BiphasicAxonMapSpatial(implant=ArgusII(), step=2)
    model.bright_model = 1.0
    with pytest.raises(TypeError):
        model.build()

    # If t_percept is not specified, there should only be one frame
    model = BiphasicAxonMapSpatial(implant=ArgusII(), step=2)
    model.build()
    implant = ArgusII()
    source = Stimulus({'A5': BiphasicPulseTrain(20, 1 * xTh, 0.45)})
    percept = model.predict_percept(source)
    npt.assert_equal(percept.time is None, True)
    # If t_percept is specified, only first frame should have data
    # and the rest should be empty
    percept = model.predict_percept(source, t_percept=[0, 1, 2, 5, 10])
    npt.assert_equal(len(percept.time), 5)
    npt.assert_equal(np.any(percept.data[:, :, 0]), True)
    npt.assert_equal(np.any(percept.data[:, :, 1:]), False)

    # Test that default models give expected values
    model = BiphasicAxonMapSpatial(implant=ArgusII(), rho=400, lam=600,
                                   step=1, xrange=(-20, 20), yrange=(-15, 15))
    model.build()
    implant = ArgusII()
    source = Stimulus({'A4': BiphasicPulseTrain(20, 1 * xTh, 1)})
    percept = model.predict_percept(source)
    npt.assert_equal(np.sum(percept.data > 0.0813), 70)
    npt.assert_equal(np.sum(percept.data > 0.1626), 50)
    npt.assert_equal(np.sum(percept.data > 0.2439), 33)
    npt.assert_equal(np.sum(percept.data > 0.4065), 16)
    npt.assert_equal(np.sum(percept.data > 0.5691), 4)


def test_biphasicAxonMapModel():
    set_params = {'step': 2, 'rho': 432, 'lam': 20,
                  'n_axons': 9, 'n_ax_segments': 50,
                  'xrange': (-30, 30), 'yrange': (-20, 20),
                  'loc_od': (5, 6)}
    model = BiphasicAxonMapModel(implant=ArgusII())
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
    model.lam = 450
    npt.assert_equal(model.spatial.size_model.rho, 350)
    npt.assert_equal(model.spatial.streak_model.lam, 450)
    npt.assert_equal(model.rho, 350)
    npt.assert_equal(model.lam, 450)

    # Effect model parameters can be passed even in constructor
    model = BiphasicAxonMapModel(implant=ArgusII(), a0=5, rho=432)
    npt.assert_equal(model.a0, 5)
    npt.assert_equal(model.spatial.bright_model.a0, 5)
    npt.assert_equal(model.rho, 432)
    npt.assert_equal(model.spatial.size_model.rho, 432)

    # If parameter is not an effects model param, it cant be set
    with pytest.raises(FreezeError):
        model.invalid_param = 5

    # Custom parameters also propogate to effects models
    model = BiphasicAxonMapModel(implant=ArgusII())

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
            self.model = BiphasicAxonMapModel(implant=ArgusII())
            # This shouldnt raise an error
            self.model.a0

    class TestInitClassBad():
        def __init__(self):
            self.model = BiphasicAxonMapModel(implant=ArgusII())
            # This should
            self.model.a10 = 999
    # If this fails, something is wrong with getattr / setattr logic
    TestInitClassGood()
    with pytest.raises(FreezeError):
        TestInitClassBad()

    # User can override default values
    model = BiphasicAxonMapModel(implant=ArgusII())
    for key, value in set_params.items():
        setattr(model.spatial, key, value)
        npt.assert_equal(getattr(model.spatial, key), value)
    model = BiphasicAxonMapModel(implant=ArgusII(), **set_params)
    model.build(**set_params)
    for key, value in set_params.items():
        npt.assert_equal(getattr(model.spatial, key), value)

    # Zeros in, zeros out:
    source = np.zeros(60)
    npt.assert_almost_equal(model.predict_percept(source).data, 0)
    source = np.zeros(60)
    npt.assert_almost_equal(model.predict_percept(source).data, 0)

    # The eye is the implanted one, and is not settable on its own:
    npt.assert_equal(
        BiphasicAxonMapModel(implant=ArgusII(eye='LE'), step=5).eye, 'LE')
    with pytest.raises(AttributeError):
        BiphasicAxonMapModel(implant=ArgusII(), eye='LE')

    # Lambda cannot be too small:
    with pytest.raises(ValueError):
        BiphasicAxonMapModel(implant=ArgusII(), lam=9).build()


def test_DefaultStreakModel_removed_axlambda():
    # The streak model took `axlambda` as a keyword until 0.10.0; the old
    # name was removed in 0.11.0:
    with pytest.raises(TypeError):
        DefaultStreakModel(axlambda=200)
    npt.assert_equal(DefaultStreakModel(lam=200).lam, 200)
    npt.assert_equal(DefaultStreakModel(200).lam, 200)


@pytest.mark.parametrize('model_cls', [BiphasicAxonMapModel,
                                       BiphasicAxonMapSpatial])
# Scaling the train itself, and scaling the stimulus the implant made of it;
# the second is what a user reaches for:
@pytest.mark.parametrize('compose', [False, True])
def test_scaled_pulse_train_changes_percept(model_cls, compose):
    # Scaling has to give what building the train at that amplitude in the
    # first place gives:
    model = model_cls(implant=ArgusII(), xrange=(-12, 12), yrange=(-8, 8),
                      step=1, n_ax_segments=30).build()
    source = model.implant.prepare_stim(
        {'C5': BiphasicPulseTrain(20, 1 * xTh, 0.45, stim_dur=100)})
    single = model.predict_percept(source).data
    if compose:
        source = source * 2
    else:
        source = {'C5': BiphasicPulseTrain(20, 1 * xTh, 0.45,
                                           stim_dur=100) * 2}
    doubled = model.predict_percept(source).data
    direct = model.predict_percept(
        {'C5': BiphasicPulseTrain(20, 2 * xTh, 0.45, stim_dur=100)}).data
    npt.assert_equal(np.any(single), True)
    npt.assert_array_almost_equal(doubled, direct)
    npt.assert_equal(np.allclose(doubled, single), False)


@pytest.mark.parametrize('model_cls', [BiphasicAxonMapModel,
                                       BiphasicAxonMapSpatial])
@pytest.mark.parametrize('modify', [lambda s: s + 5, lambda s: s * np.inf,
                                    lambda s: s.append(s >> 1)])
def test_modified_pulse_train_rejected(model_cls, modify):
    # A DC offset, a non-finite factor and an appended second train all leave
    # something other than a biphasic pulse train. The model must say so rather
    # than predict from pulse parameters that no longer describe the stimulus:
    model = model_cls(implant=ArgusII(), xrange=(-3, 3), yrange=(-2, 2),
                      step=1, n_ax_segments=30).build()
    source = model.implant.prepare_stim(
        {'C5': BiphasicPulseTrain(20, 1 * xTh, 0.45, stim_dur=100)})
    with np.errstate(divide='ignore', invalid='ignore'):
        source = modify(source)
    with pytest.raises(TypeError):
        model.predict_percept(source)


def test_pulse_train_amp_sign_does_not_change_percept():
    # `BiphasicPulse` takes the magnitude of `amp`, so these two trains have
    # the very same waveform, and must therefore predict the same percept:
    model = BiphasicAxonMapModel(implant=ArgusII(), xrange=(-12, 12),
                                 yrange=(-8, 8), step=1,
                                 n_ax_segments=30).build()
    pos = model.implant.prepare_stim(
        {'C5': BiphasicPulseTrain(20, 1 * xTh, 0.45, stim_dur=100)})
    neg = model.implant.prepare_stim(
        {'C5': BiphasicPulseTrain(20, -1 * xTh, 0.45, stim_dur=100)})
    npt.assert_almost_equal(pos.data, neg.data)
    npt.assert_array_almost_equal(model.predict_percept(pos).data,
                                  model.predict_percept(neg).data)


def test_BiphasicAxonMapModel_min_current_spread():
    """The current-spread cutoff must reach this model's kernel too.

    ``min_current_spread`` lives on ``SpatialModel``, so
    ``BiphasicAxonMapSpatial`` inherits it; this pins that the biphasic
    kernel actually honours it rather than accepting it and ignoring it.
    """
    stim = {e: BiphasicPulseTrain(20, 30 * xTh, 0.45)
            for e in ('A2', 'C5', 'F8')}
    source = stim
    kwargs = {'xrange': (-8, 8), 'yrange': (-8, 8), 'step': 0.5,
              'rho': 200, 'verbose': False}

    exact = BiphasicAxonMapModel(implant=ArgusII(), 
        min_current_spread=0, **kwargs).build().predict_percept(source).data
    default = BiphasicAxonMapModel(implant=ArgusII(), 
        **kwargs).build().predict_percept(source).data
    # For three electrodes the default cutoff is not worth thinking about;
    # see `test_BiphasicAxonMapModel_min_current_spread_error_bound` for the
    # case where it is:
    npt.assert_allclose(default, exact, rtol=1e-5,
                        atol=1e-6 * np.abs(exact).max())

    # A coarse cutoff does change the result, which is how we know it is
    # wired through rather than silently dropped:
    coarse = BiphasicAxonMapModel(implant=ArgusII(), 
        min_current_spread=0.5, **kwargs).build().predict_percept(source).data
    assert np.abs(coarse - exact).max() > 1e-3


@pytest.mark.parametrize('amp', (2.0, 50.0))
def test_BiphasicAxonMapModel_min_current_spread_error_bound(amp):
    """The cutoff is an approximation here too, with a bound to match.

    The biphasic kernel tests ``r2`` against the cutoff before scaling the
    exponential by ``F_bright``, so what it drops at a segment is
    ``sum_i F_bright_i * exp(...)``. The error is therefore bounded by
    ``min_current_spread * sum(F_bright)``, which grows with the array size
    and with however hard the brightness model scales -- not by 1e-8 outright.
    """
    min_spread = 1e-8
    freq, pdur = 20, 0.45
    source = {e: BiphasicPulseTrain(freq, amp * xTh, pdur)
                            for e in ArgusII().electrode_names}
    kwargs = {'xrange': (-14, 14), 'yrange': (-10, 10), 'step': 0.75,
              'rho': 200, 'lam': 800, 'verbose': False}

    model = BiphasicAxonMapModel(implant=ArgusII(), min_current_spread=0, **kwargs).build()
    exact = model.predict_percept(source).data
    default = BiphasicAxonMapModel(implant=ArgusII(), 
        min_current_spread=min_spread,
        **kwargs).build().predict_percept(source).data

    n_el = model.implant.n_electrodes
    f_bright = np.asarray(model.spatial.bright_model(
        np.full(n_el, freq), np.full(n_el, amp), np.full(n_el, pdur)))
    dropped = min_spread * np.abs(f_bright).sum()
    assert np.abs(default - exact).max() <= dropped + 1e-6 * np.abs(exact).max()


@pytest.mark.parametrize('attr', ('size_model', 'streak_model'))
def test_BiphasicAxonMapModel_rejects_nonpositive_effects(attr):
    """A scaling factor of zero would surface as NaN, so it is rejected.

    Both factors enter the kernel through an exponent, so neither may be
    zero or negative. The default models cannot produce that, but a custom
    one could.
    """
    model = BiphasicAxonMapModel(implant=ArgusII(), xrange=(-4, 4), yrange=(-4, 4), step=1,
                                 verbose=False).build()
    setattr(model.spatial, attr, lambda freq, amp, pdur: np.zeros_like(amp))
    source = {'A2': BiphasicPulseTrain(20, 30 * xTh, 0.45)}
    with pytest.raises(ValueError, match=attr):
        model.predict_percept(source)

    # A positive factor is accepted:
    setattr(model.spatial, attr, lambda freq, amp, pdur: np.ones_like(amp))
    npt.assert_equal(model.predict_percept(source) is not None, True)


@pytest.mark.parametrize('attr', ('bright_model', 'size_model',
                                  'streak_model'))
@pytest.mark.parametrize('bad', (np.nan, np.inf, -np.inf))
def test_BiphasicAxonMapModel_rejects_nonfinite_effects(attr, bad):
    """A non-finite scaling factor is rejected before it reaches the kernel.

    ``nan <= 0`` is false, so the positivity check above does not catch NaN.
    Left alone it would flow into the kernel's exponent, and then
    ``abs(nan) > abs(px_bright)`` is false too -- so the affected segments
    would drop out of the max and the percept would come back quietly wrong
    instead of raising. Infinities are no better: they turn the sum into
    ``inf`` or ``nan`` depending on the signs involved. This covers
    ``bright_model`` as well, which the positivity check deliberately does
    not (a zero or negative brightness factor is legitimate).
    """
    model = BiphasicAxonMapModel(implant=ArgusII(), xrange=(-4, 4), yrange=(-4, 4), step=1,
                                 verbose=False).build()
    setattr(model.spatial, attr,
            lambda freq, amp, pdur: np.full_like(np.asarray(amp, dtype=float),
                                                 bad))
    source = {'A2': BiphasicPulseTrain(20, 30 * xTh, 0.45)}
    with pytest.raises(ValueError, match=attr):
        model.predict_percept(source)


def test_BiphasicAxonMapModel_reduces_to_AxonMapModel():
    """With every effect factor at 1, this model *is* the axon map model.

    The biphasic kernel computes
    ``F_bright * exp(-r^2 / (2 rho^2 F_size)) * sens ** (1 / F_streak)``
    as a single exponential of summed exponents. Setting all three factors to
    1 collapses that to ``exp(-r^2 / (2 rho^2)) * sens``, which is exactly
    what ``AxonMapModel`` computes for a unit-amplitude stimulus -- an
    independently written kernel, so this pins the fused arithmetic against
    something that does not share its code.
    """
    from pulse2percept.models import AxonMapModel

    kwargs = {'xrange': (-8, 8), 'yrange': (-8, 8), 'step': 0.5,
              'rho': 200, 'lam': 800, 'verbose': False}
    electrodes = ('A2', 'C5', 'F8')

    biphasic = BiphasicAxonMapModel(implant=ArgusII(), **kwargs)
    for attr in ('bright_model', 'size_model', 'streak_model'):
        setattr(biphasic.spatial, attr,
                lambda freq, amp, pdur: np.ones_like(np.asarray(amp,
                                                                dtype=float)))
    biphasic.build()
    got = biphasic.predict_percept({e: BiphasicPulseTrain(20, 30 * xTh, 0.45)
              for e in electrodes}).data

    plain = AxonMapModel(implant=ArgusII(), **kwargs).build()
    stim = np.zeros(60)
    names = list(ArgusII().electrode_names)
    for e in electrodes:
        stim[names.index(e)] = 1.0
    want = plain.predict_percept(stim).data

    npt.assert_allclose(got, want, rtol=1e-5, atol=1e-6 * np.abs(want).max())


def test_BiphasicAxonMap_t_percept_units():
    """This model overrides `predict_percept`, so it normalizes for itself"""
    source = {'A1': BiphasicPulseTrain(20, 1 * xTh, 0.45,
                                                     stim_dur=100)}
    for model in (BiphasicAxonMapSpatial(implant=ArgusII(), step=2).build(),
                  BiphasicAxonMapModel(implant=ArgusII(), step=2).build()):
        # A single time point is one time point, not something with a `len`:
        npt.assert_equal(model.predict_percept(source,
                                               t_percept=20).data.shape[-1], 1)
        bare = model.predict_percept(source, t_percept=[0, 20])
        for spelling in ([0, 20] * ms, np.array([0, 0.02]) * s, 20 * ms):
            unitful = model.predict_percept(source, t_percept=spelling)
            npt.assert_allclose(unitful.data.max(), bare.data.max(),
                                rtol=1e-12)
        with pytest.raises(DimensionMismatchError):
            model.predict_percept(source, t_percept=[0, 20] * uA)


def test_BiphasicAxonMap_dimension_before_waveform():
    """A picture is not an unsuitable pulse train, it is not a current at all

    The dimensional contract is the outermost one, so a dimensionless stimulus
    reports that rather than the model's own "must be BiphasicPulseTrains"
    complaint, which is about a stimulus it never had.
    """
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))

    # An ordinary implant refuses the picture outright (see
    # `ProsthesisSystem.stimulus_unit`); one that delivers something else is
    # what carries it as far as the model:
    class Projector(ArgusII):
        stimulus_unit = dimensionless

    projector = Projector(preprocess=False)
    for model in (BiphasicAxonMapSpatial(implant=projector, step=2).build(),
                  BiphasicAxonMapModel(implant=projector, step=2).build()):
        with pytest.raises(DimensionMismatchError) as excinfo:
            model.predict_percept(img)
        # Both dimensions this model reads are named, not just the one:
        for accepted in ('electric current', 'threshold ratio'):
            npt.assert_equal(accepted in str(excinfo.value), True)
        npt.assert_equal('dimensionless' in str(excinfo.value), True)
    # A current-valued stimulus of the wrong waveform still gets the
    # model-specific message:
    with pytest.raises(TypeError) as excinfo:
        BiphasicAxonMapSpatial(implant=ArgusII(), step=2).build().predict_percept(
            {'A1': MonophasicPulse(-1, 0.45, stim_dur=100)})
    npt.assert_equal('BiphasicPulseTrain' in str(excinfo.value), True)


@pytest.mark.parametrize('ModelClass', [BiphasicAxonMapSpatial,
                                        BiphasicAxonMapModel])
def test_BiphasicAxonMapSpatial_meridian_blend(ModelClass):
    # This model replaces `predict_percept` instead of customizing
    # `_predict_spatial`, so it has to call the postprocessing hook itself --
    # otherwise `meridian_blend`, inherited from `AxonMapSpatial`, would be
    # accepted and then quietly ignored.
    def make(**params):
        return ModelClass(implant=ArgusII(), xrange=(-6, 6), yrange=(-6, 6), step=0.25, rho=200,
                          lam=400, n_axons=250, n_ax_segments=200,
                          ignore_pickle=True, **params).build()

    source = {'C4': BiphasicPulseTrain(20, 20 * xTh, 0.45),
                            'C8': BiphasicPulseTrain(20, 20 * xTh, 0.45)}
    plain = make(meridian_blend=0)
    unblended = plain.predict_percept(source).data

    # Exercises the width inherited from `AxonMapSpatial` rather than one of
    # its own -- the point of the test is that the hook runs at all here:
    width = 1
    blended_model = make()
    npt.assert_equal(blended_model.meridian_blend, width)
    blended = blended_model.predict_percept(source).data
    npt.assert_equal(blended.shape, unblended.shape)
    npt.assert_equal(blended.dtype, unblended.dtype)
    npt.assert_equal(np.array_equal(blended, unblended), False)
    # It is the horizontal meridian here, so the change is a band around y=0:
    y = plain.grid.y[:, 0]
    delta = np.abs(blended - unblended)
    rows = delta.max(axis=(1, 2)) > delta.max() * 1e-3
    npt.assert_array_less(np.abs(y[rows]).max(), 4 * width)


@contextmanager
def _no_pulse_train_rendering():
    """Make generating a pulse train's waveform an error

    The only way to state "this model never asks for samples" as a test: if
    anything on the prediction path reaches for one, it fails loudly.
    """
    original = BiphasicPulseTrain._render

    def refuse(self):
        raise AssertionError('generated a pulse train waveform')
    BiphasicPulseTrain._render = refuse
    try:
        yield
    finally:
        BiphasicPulseTrain._render = original


@pytest.mark.parametrize('model_cls', [BiphasicAxonMapModel,
                                       BiphasicAxonMapSpatial])
@pytest.mark.parametrize('build_stim', [
    lambda: BiphasicPulseTrain(20, 1 * xTh, 0.45, stim_dur=100,
                               electrode='C5'),
    lambda: {'C5': BiphasicPulseTrain(20, 1 * xTh, 0.45, stim_dur=100),
             'A2': BiphasicPulseTrain(30, 2 * xTh, 0.45, stim_dur=100)},
    lambda: Stimulus({'C5': BiphasicPulseTrain(20, 1 * xTh, 0.45,
                                               stim_dur=100)}) * 2,
])
def test_BiphasicAxonMap_predicts_without_a_waveform(model_cls, build_stim):
    # This model is a function of frequency, amplitude and phase duration, and
    # it now takes them from the pulse trains themselves. None of them needs
    # the train sampled, so predicting must not sample one.
    model = model_cls(implant=ArgusII(), xrange=(-3, 3), yrange=(-2, 2), step=1,
                      n_ax_segments=30).build()
    source = build_stim()
    with _no_pulse_train_rendering():
        percept = model.predict_percept(source)
    npt.assert_equal(np.any(percept.data), True)


@pytest.mark.parametrize('model_cls', [BiphasicAxonMapModel,
                                       BiphasicAxonMapSpatial])
def test_BiphasicAxonMap_ignores_user_metadata(model_cls):
    # Metadata that happens to name pulse parameters is still just metadata:
    model = model_cls(implant=ArgusII(), xrange=(-3, 3), yrange=(-2, 2),
                      step=1, n_ax_segments=30).build()
    source = model.implant.prepare_stim(
        {'C5': BiphasicPulseTrain(20, 1 * xTh, 0.45, stim_dur=100)})
    expected = model.predict_percept(source).data
    source.metadata['user'] = {'amp': 999, 'freq': 1}
    npt.assert_array_equal(model.predict_percept(source).data, expected)


@pytest.mark.parametrize('model_cls', [BiphasicAxonMapModel,
                                       BiphasicAxonMapSpatial])
@pytest.mark.parametrize('build_stim', [
    lambda: {'C5': MonophasicPulse(-1, 0.45, stim_dur=100)},
    lambda: {'C5': AsymmetricBiphasicPulseTrain(20, 1, 2, 0.45, 0.9,
                                                stim_dur=100)},
    lambda: {'C5': BiphasicPulseTrain(20, 1 * xTh, 0.45, delay_dur=1,
                                      stim_dur=100)},
    lambda: (BiphasicPulseTrain(20, 1 * xTh, 0.45, stim_dur=100,
                                electrode='C5')
             .append(BiphasicPulseTrain(50, 1 * xTh, 0.45, stim_dur=100,
                                        electrode='C5'))),
    lambda: {'C5': Stimulus([[0, 1, 1, 0]], time=[0, 1, 99, 100])},
])
def test_BiphasicAxonMap_rejects_what_it_cannot_read(model_cls, build_stim):
    # A sequence of two trains has no single frequency, an asymmetric train is
    # not this model's protocol, a delayed one is outside what it was fit on,
    # and a raw waveform has no parameters at all. Each is refused rather than
    # predicted from whatever numbers happen to be lying around:
    model = model_cls(implant=ArgusII(), xrange=(-3, 3), yrange=(-2, 2), step=1,
                      n_ax_segments=30).build()
    source = build_stim()
    with pytest.raises(TypeError):
        model.predict_percept(source)


@pytest.mark.parametrize('model_cls', [BiphasicAxonMapModel,
                                       BiphasicAxonMapSpatial])
def test_BiphasicAxonMap_zero_amplitude_is_inactive(model_cls):
    # A train at zero amplitude drives nothing, which is a zero percept rather
    # than an error -- and is read off `amp`, not off the waveform:
    model = model_cls(implant=ArgusII(), xrange=(-3, 3), yrange=(-2, 2), step=1,
                      n_ax_segments=30).build()
    source = {'C5': BiphasicPulseTrain(20, 0 * xTh, 0.45,
                                                     stim_dur=100),
                            'A2': BiphasicPulseTrain(20, 0 * xTh, 0.45,
                                                     stim_dur=100)}
    with _no_pulse_train_rendering():
        percept = model.predict_percept(source)
    npt.assert_almost_equal(percept.data, 0)
    # One driven electrode among zeros still predicts from that one alone:
    source = {'C5': BiphasicPulseTrain(20, 0 * xTh, 0.45, stim_dur=100),
                    'A2': BiphasicPulseTrain(20, 1 * xTh, 0.45, stim_dur=100)}
    with _no_pulse_train_rendering():
        mixed = model.predict_percept(source).data
    only = model.predict_percept({'A2': BiphasicPulseTrain(20, 1 * xTh, 0.45, stim_dur=100)}).data
    npt.assert_array_almost_equal(mixed, only)


@pytest.mark.parametrize('model_cls', [BiphasicAxonMapModel,
                                       BiphasicAxonMapSpatial])
def test_BiphasicAxonMap_rejects_an_encoded_stimulus(model_cls):
    # An encoder's output is a schedule, not a pulse train: its amplitude and
    # frequency may differ from frame to frame, so there is no one `freq` for
    # this model to read. It stays refused, and refusing it does not expand
    # the schedule into a waveform either.
    model = model_cls(implant=ArgusII(), xrange=(-3, 3), yrange=(-2, 2), step=1,
                      n_ax_segments=30).build()
    implant = ArgusII()
    encoded = AmplitudeEncoder().encode(
        ImageStimulus(np.linspace(0, 1, 64).reshape(8, 8)), implant=implant)
    npt.assert_equal(encoded._structured_sources(), None)
    source = encoded
    with pytest.raises(TypeError):
        model.predict_percept(source)


# The temporal models a Granley composite is expected to work with. Their
# responses to one canonical drive peak at quite different moments, which is
# what most of these tests are about.
_TEMPORALS = [FadingTemporal, Nanduri2012Temporal, Horsager2009Temporal]

# Stimulation lasts this long, and the whole episode -- including the tail in
# which a lagging cascade can still be rising -- fits in this window (ms):
_STIM_DUR = 50
_EPISODE = 200


def _composite(temporal):
    """A Granley spatial model paired with ``temporal``, and Granley alone"""
    grid = dict(xrange=(-3, 3), yrange=(-2, 2), step=1, n_ax_segments=30)
    composite = Model(spatial=BiphasicAxonMapSpatial(implant=ArgusII(), **grid),
                      temporal=temporal)
    return composite.build(), BiphasicAxonMapModel(implant=ArgusII(), **grid).build()


def _composite_source(stim_dur=_STIM_DUR):
    return {'A5': BiphasicPulseTrain(20, 1 * xTh, 0.45, stim_dur=stim_dur)}


def _every_dt(temporal, until=_EPISODE):
    """Every instant ``temporal`` integrates, up to ``until`` (ms)

    Anything coarser reports the peak of the samples rather than the peak.
    """
    return np.arange(int(round(until / temporal.dt)) + 1) * temporal.dt


@pytest.mark.parametrize('temporal_cls', _TEMPORALS)
def test_BiphasicAxonMapSpatial_with_temporal_model_runs(temporal_cls):
    # Issue #565: the spatial model collapses time, so the percept it hands
    # over has no time axis for a temporal model to integrate.
    composite, _ = _composite(temporal_cls())
    # The drive is built from the retained pulse-train parameters, so pairing
    # the two still asks for no waveform:
    with _no_pulse_train_rendering():
        percept = composite.predict_percept(_composite_source())
    npt.assert_equal(percept.data.ndim, 3)
    npt.assert_equal(percept.data.shape[-1] > 1, True)
    npt.assert_equal(np.all(np.isfinite(percept.data)), True)


@pytest.mark.parametrize('temporal_cls', _TEMPORALS)
def test_BiphasicAxonMapSpatial_composite_peaks_at_the_granley_percept(
        temporal_cls):
    temporal = temporal_cls()
    composite, granley = _composite(temporal)
    source = _composite_source()
    percept = composite.predict_percept(source,
                                        t_percept=_every_dt(temporal))
    npt.assert_array_almost_equal(
        percept.max(axis='frames'),
        granley.predict_percept(source).data[..., 0])


@pytest.mark.parametrize('temporal_cls', _TEMPORALS)
def test_BiphasicAxonMapSpatial_composite_is_space_time_separable(
        temporal_cls):
    composite, _ = _composite(temporal_cls())
    percept = composite.predict_percept(_composite_source(),
                                        t_percept=[10, 20, 40, 80, 160])
    lit = percept.data[percept.data.max(axis=-1) > 0]
    npt.assert_equal(len(lit) > 1, True)
    # Every pixel rides the same envelope, so the spatial scale is all that
    # told the pixels apart:
    shapes = lit / lit.max(axis=-1, keepdims=True)
    npt.assert_array_almost_equal(shapes - shapes[0], 0)


@pytest.mark.parametrize('temporal_cls', _TEMPORALS)
def test_BiphasicAxonMapSpatial_composite_peak_ignores_requested_sampling(
        temporal_cls):
    composite, granley = _composite(temporal_cls())
    source = _composite_source()
    granley_max = granley.predict_percept(source).data.max()
    # Asked only for instants that fall well before the peak, the percept must
    # stay below the Granley maximum rather than renormalizing onto it:
    early = composite.predict_percept(source, t_percept=[10, 20])
    npt.assert_equal(early.data.max() < 0.9 * granley_max, True)
    # And those samples are the ones a longer run reports at the same times:
    longer = composite.predict_percept(source, t_percept=[10, 20, 40, 80])
    npt.assert_array_almost_equal(early.data, longer.data[..., :2])


@pytest.mark.parametrize('temporal_cls', _TEMPORALS)
def test_BiphasicAxonMapSpatial_composite_decays_after_stimulation(
        temporal_cls):
    composite, _ = _composite(temporal_cls())
    percept = composite.predict_percept(_composite_source(),
                                        t_percept=[100, 150, 200])
    frame_max = percept.data.max(axis=(0, 1))
    npt.assert_equal(np.all(np.diff(frame_max) < 0), True)
    npt.assert_equal(frame_max[-1] > 0, True)


@pytest.mark.parametrize('temporal_cls, param', [(FadingTemporal, 'tau'),
                                                 (Nanduri2012Temporal, 'tau3'),
                                                 (Horsager2009Temporal,
                                                  'tau3')])
def test_BiphasicAxonMapSpatial_composite_temporal_params_only_move_time(
        temporal_cls, param):
    source = _composite_source()
    traces = []
    for value in (10, 60):
        temporal = temporal_cls(**{param: value})
        composite, granley = _composite(temporal)
        percept = composite.predict_percept(source,
                                            t_percept=_every_dt(temporal))
        peak_frame = percept.max(axis='frames')
        npt.assert_array_almost_equal(
            peak_frame, granley.predict_percept(source).data[..., 0])
        brightest = np.unravel_index(np.argmax(peak_frame), peak_frame.shape)
        traces.append(percept.data[brightest])
    npt.assert_equal(np.allclose(traces[0], traces[1]), False)


@pytest.mark.parametrize('temporal_cls', _TEMPORALS)
def test_BiphasicAxonMapSpatial_composite_ignores_temporal_thresh_percept(
        temporal_cls):
    # The envelope is normalized, so a floor in the temporal model's own
    # brightness units would apply to the wrong quantity. Brightness is
    # Granley's to set, and the caller's own model is left alone:
    temporal = temporal_cls(thresh_percept=0.1)
    composite, _ = _composite(temporal)
    plain, _ = _composite(temporal_cls())
    source = _composite_source()
    npt.assert_array_almost_equal(
        composite.predict_percept(source, t_percept=[10, 20, 40]).data,
        plain.predict_percept(source, t_percept=[10, 20, 40]).data)
    npt.assert_almost_equal(temporal.thresh_percept, 0.1)


def test_BiphasicAxonMapSpatial_composite_ignores_inactive_stim_dur():
    # A train at zero amplitude is inactive everywhere, so it must not stretch
    # the envelope the active one rides on either:
    composite, _ = _composite(FadingTemporal())
    short = {'A5': BiphasicPulseTrain(20, 1 * xTh, 0.45, stim_dur=100)}
    padded = {'A5': BiphasicPulseTrain(20, 1 * xTh, 0.45, stim_dur=100),
              'A2': BiphasicPulseTrain(20, 0 * xTh, 0.45, stim_dur=1000)}
    t_percept = [40, 80, 100]
    npt.assert_array_almost_equal(
        composite.predict_percept(padded, t_percept=t_percept).data,
        composite.predict_percept(short, t_percept=t_percept).data)


@pytest.mark.parametrize('temporal_cls', _TEMPORALS)
def test_BiphasicAxonMapSpatial_composite_rejects_unequal_stim_dur(
        temporal_cls):
    # One separable envelope cannot have one electrode's contribution stop at
    # 100 ms and another's at 1000 ms:
    composite, _ = _composite(temporal_cls())
    source = {'A5': BiphasicPulseTrain(20, 1 * xTh, 0.45,
                                                     stim_dur=100),
                            'A2': BiphasicPulseTrain(20, 1 * xTh, 0.45,
                                                     stim_dur=1000)}
    with pytest.raises(NotImplementedError):
        composite.predict_percept(source)


def test_BiphasicAxonMapSpatial_composite_rides_an_alpha_envelope():
    """`AlphaTemporal` gives the Granley percept a rise, not just a fade.

    Sampled at every `dt`, so the peak the model normalizes by is one of the
    samples rather than something between two of them.
    """
    temporal = AlphaTemporal(tau=20)
    composite, granley = _composite(temporal)
    source = _composite_source()
    t = _every_dt(temporal)
    percept = composite.predict_percept(source, t_percept=t)

    # Time-varying, and still peaking at the Granley frame:
    granley_frame = granley.predict_percept(source).data[..., 0]
    npt.assert_equal(percept.data.shape[-1], len(t))
    npt.assert_array_almost_equal(percept.max(axis='frames'), granley_frame)

    trace = percept.data[np.unravel_index(np.argmax(granley_frame),
                                          granley_frame.shape)]
    # An alpha envelope, not an exponential fade: it starts at zero and peaks
    # well after onset, where `FadingTemporal` is already at its brightest one
    # step in.
    npt.assert_equal(trace[0], 0)
    peak = int(np.argmax(trace))
    npt.assert_equal(0 < peak < len(trace) - 1, True)
    npt.assert_array_less(_STIM_DUR, t[peak])
    npt.assert_array_less(trace[-1], trace[peak])

    fading, _ = _composite(FadingTemporal(tau=20))
    fade = fading.predict_percept(source, t_percept=t).data[
        np.unravel_index(np.argmax(granley_frame), granley_frame.shape)]
    npt.assert_array_less(np.argmax(fade), peak)


def test_BiphasicAxonMapSpatial_composite_normalizes_a_delayed_peak():
    # This one peaks ~226 ms after a 50 ms drive, several windows past the
    # first one searched. Normalizing by a value found before the peak would
    # let the percept outshine the Granley frame it is supposed to peak at:
    temporal = Horsager2009Temporal(tau3=100)
    composite, granley = _composite(temporal)
    source = _composite_source()
    granley_frame = granley.predict_percept(source).data[..., 0]
    percept = composite.predict_percept(source,
                                        t_percept=_every_dt(temporal, 400))
    npt.assert_equal(percept.data.max() <= granley_frame.max(), True)
    npt.assert_array_almost_equal(percept.max(axis='frames'), granley_frame)


def test_BiphasicAxonMapSpatial_composite_rejects_an_unlocatable_peak():
    # Still rising where the search gives up (~624 ms for a 50 ms drive):
    # saying so beats normalizing by a value known not to be the peak.
    composite, _ = _composite(Horsager2009Temporal(tau3=300))
    with pytest.raises(ValueError):
        composite.predict_percept(_composite_source())


def _threshold_model():
    return BiphasicAxonMapModel(implant=ArgusII(), xrange=(-3, 3), yrange=(-2, 2), step=1,
                                n_ax_segments=30).build()


def _percept_at(model, train, thresholds=None):
    if thresholds is not None:
        model.implant.thresholds = thresholds
    return model.predict_percept({'A2': train}).data


def test_BiphasicAxonMap_reads_threshold_multiples_not_current():
    model = _threshold_model()
    # Uncalibrated 2xTh and 160 uA on an 80 uA electrode are the same
    # stimulation to this model, though only the second is a current:
    relative = _percept_at(model, BiphasicPulseTrain(20, 2 * xTh, 0.45,
                                                     stim_dur=100))
    npt.assert_equal(np.any(relative), True)
    calibrated = _percept_at(model, BiphasicPulseTrain(20, 2 * xTh, 0.45,
                                                       stim_dur=100),
                             thresholds=80 * uA)
    npt.assert_array_equal(calibrated, relative)
    as_current = _percept_at(model, BiphasicPulseTrain(20, 160 * uA, 0.45,
                                                       stim_dur=100),
                             thresholds=80 * uA)
    npt.assert_array_equal(as_current, relative)
    on_the_train = _percept_at(model,
                               BiphasicPulseTrain(20, 160 * uA, 0.45,
                                                  stim_dur=100,
                                                  threshold_amp=80 * uA))
    npt.assert_array_equal(on_the_train, relative)


def test_BiphasicAxonMap_same_current_differs_by_threshold():
    model = _threshold_model()
    train = BiphasicPulseTrain(20, 160 * uA, 0.45, stim_dur=100)
    npt.assert_equal(np.allclose(_percept_at(model, train, thresholds=80 * uA),
                                 _percept_at(model, train,
                                             thresholds=40 * uA)),
                     False)


@pytest.mark.parametrize('model_cls', [BiphasicAxonMapModel,
                                       BiphasicAxonMapSpatial])
def test_BiphasicAxonMap_uncalibrated_current_raises(model_cls):
    model = model_cls(implant=ArgusII(), xrange=(-3, 3), yrange=(-2, 2), step=1,
                      n_ax_segments=30).build()
    source = {'A2': BiphasicPulseTrain(20, 160 * uA, 0.45,
                                                     stim_dur=100)}
    with pytest.raises(ValueError) as err:
        model.predict_percept(source)
    for remedy in ('2 * xTh', 'threshold_amp', 'implant.thresholds'):
        npt.assert_equal(remedy in str(err.value), True)
    # Any one of the three fixes it:
    model.implant.thresholds = 80 * uA
    npt.assert_equal(np.any(model.predict_percept(source).data), True)


@pytest.mark.parametrize('model_cls', [BiphasicAxonMapModel,
                                       BiphasicAxonMapSpatial])
def test_BiphasicAxonMap_zero_current_needs_no_threshold(model_cls):
    # Zero-current electrodes need no threshold.
    model = model_cls(implant=ArgusII(), xrange=(-3, 3), yrange=(-2, 2), step=1,
                      n_ax_segments=30).build()
    source = {'A2': BiphasicPulseTrain(20, 0 * uA, 0.45,
                                                     stim_dur=100)}
    with _no_pulse_train_rendering():
        percept = model.predict_percept(source)
    npt.assert_almost_equal(percept.data, 0)


@pytest.mark.parametrize('model_cls', [BiphasicAxonMapModel,
                                       BiphasicAxonMapSpatial])
def test_BiphasicAxonMap_n_gray(model_cls):
    # n_gray must reach the Percept, as in the generic spatial path:
    source = {'C5': BiphasicPulseTrain(20, 2 * xTh, 0.45, stim_dur=100)}
    model = model_cls(implant=ArgusII(), xrange=(-3, 3), yrange=(-2, 2),
                      step=1, n_ax_segments=30).build()
    full = model.predict_percept(source)
    npt.assert_equal(np.unique(full.data).size > 2, True)
    # The metadata should carry the stimulus itself, not just its metadata:
    npt.assert_equal(isinstance(full.metadata['stim'], Stimulus), True)

    model = model_cls(implant=ArgusII(), xrange=(-3, 3), yrange=(-2, 2),
                      step=1, n_ax_segments=30, n_gray=2).build()
    quantized = model.predict_percept(source)
    npt.assert_equal(np.unique(quantized.data).size, 2)


@pytest.mark.parametrize('model_cls', [BiphasicAxonMapModel,
                                       BiphasicAxonMapSpatial])
def test_BiphasicAxonMap_noise(model_cls):
    # noise must reach the Percept, as in the generic spatial path:
    source = {'C5': BiphasicPulseTrain(20, 2 * xTh, 0.45, stim_dur=100)}
    model = model_cls(implant=ArgusII(), xrange=(-4, 4), yrange=(-4, 4),
                      step=1, n_ax_segments=30, noise=0.5).build()
    frame = model.predict_percept(source).data[..., 0]
    # Salt and pepper are the brightest and darkest values in the frame, and
    # a fraction ``noise`` of the pixels are set to one or the other:
    n_extreme = (np.sum(np.isclose(frame, frame.max())) +
                 np.sum(np.isclose(frame, frame.min())))
    npt.assert_equal(n_extreme >= int(0.5 * frame.size), True)
