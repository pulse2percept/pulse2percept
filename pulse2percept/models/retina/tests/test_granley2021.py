from contextlib import contextmanager
import copy
import warnings

import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept.implants.retina import ArgusI, ArgusII
from pulse2percept.percepts import Percept
from pulse2percept.stimuli import (AmplitudeEncoder,
                                   AsymmetricBiphasicPulseTrain,
                                   BiphasicPulse, BiphasicPulseTrain,
                                   ImageStimulus, MonophasicPulse, samples,
                                   Stimulus, VideoStimulus)
from pulse2percept.models import FadingTemporal
from pulse2percept.models.base import _GAUSSIAN_CUTOFF, SpatialModel
from pulse2percept.models.retina import (AxonMapSpatial, BiphasicAxonMapModel,
                                         BiphasicScoreboardModel,
                                         ScoreboardModel)
from pulse2percept.models.retina.granley2021 import DefaultBrightModel, \
    DefaultSizeModel, DefaultStreakModel
from pulse2percept.units import (DimensionMismatchError, Hz, Quantity,
                                 dimensionless, mm, ms, s, uA, um,
                                 xTh)
from pulse2percept.utils.base import FreezeError

# Axon map caches use a relative path; write them to a temp directory:
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

    # Each model takes the coefficients it declares and no others:
    coeffs = {'a' + str(i): i for i in range(10)}
    model_coeffs = {k: v for k, v in coeffs.items()
                    if hasattr(DefaultBrightModel(), k)}
    npt.assert_equal(sorted(model_coeffs), ['a0', 'a1', 'a2', 'a3', 'a4'])
    model = DefaultBrightModel(**model_coeffs)
    npt.assert_almost_equal(model.a0, 0)
    npt.assert_almost_equal(model.a4, 4)
    npt.assert_equal(hasattr(model, 'a9'), False)
    model_coeffs = {k: v for k, v in coeffs.items()
                    if hasattr(DefaultSizeModel(200), k)}
    npt.assert_equal(sorted(model_coeffs), ['a0', 'a1', 'a5', 'a6'])
    model = DefaultSizeModel(200, **model_coeffs)
    npt.assert_almost_equal(model.a0, 0)
    npt.assert_almost_equal(model.a5, 5)
    npt.assert_equal(hasattr(model, 'a9'), False)
    model_coeffs = {k: v for k, v in coeffs.items()
                    if hasattr(DefaultStreakModel(200), k)}
    npt.assert_equal(sorted(model_coeffs), ['a7', 'a8', 'a9'])
    model = DefaultStreakModel(200, **model_coeffs)
    npt.assert_almost_equal(model.a7, 7)
    npt.assert_almost_equal(model.a9, 9)
    npt.assert_equal(hasattr(model, 'a0'), False)


def test_effects_models_units():
    size = DefaultSizeModel(0.2 * mm, min_rho=20 * um)
    npt.assert_almost_equal(size.rho, 200)
    npt.assert_almost_equal(size.min_rho, 20)
    streak = DefaultStreakModel(0.5 * mm, min_lambda=20 * um)
    npt.assert_almost_equal(streak.lam, 500)
    npt.assert_almost_equal(streak.min_lambda, 20)
    # Stored as plain floats:
    for value in (size.rho, size.min_rho, streak.lam, streak.min_lambda):
        npt.assert_equal(isinstance(value, Quantity), False)
        npt.assert_equal(isinstance(value, (int, float)), True)
    # Unitful and bare values give the same scaling factor:
    npt.assert_almost_equal(DefaultSizeModel(0.2 * mm)(20, 1, 0.45),
                            DefaultSizeModel(200)(20, 1, 0.45))
    npt.assert_almost_equal(DefaultStreakModel(0.5 * mm)(20, 1, 0.45),
                            DefaultStreakModel(500)(20, 1, 0.45))
    # Wrong dimensions:
    with pytest.raises(DimensionMismatchError):
        DefaultSizeModel(200 * uA)
    with pytest.raises(DimensionMismatchError):
        DefaultStreakModel(500 * ms)


@pytest.mark.parametrize('cls, arg', [(DefaultSizeModel, 200),
                                      (DefaultStreakModel, 200)])
def test_effects_models_removed_engine(cls, arg):
    # 'engine' (numpy vs. the removed jax backend) was deprecated in 0.9.1,
    # removed in 0.10.0:
    with pytest.raises(TypeError):
        cls(arg, engine='serial')


def test_BiphasicAxonMapModel_predict_percept():
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
    npt.assert_equal(percept.shape,
                     list(model.spatial.grid.x.shape) + [1])
    npt.assert_almost_equal(percept.data, 0)
    npt.assert_equal(percept.time, None)

    # Should be equal to axon map model if effects models return 1
    model = BiphasicAxonMapModel(implant=ArgusII(), step=2)
    def bright_model(freq, amp, pdur): return 1
    def size_model(freq, amp, pdur): return 1
    def streak_model(freq, amp, pdur): return 1
    model.spatial.bright_model = bright_model
    model.spatial.size_model = size_model
    model.spatial.streak_model = streak_model
    model.build()
    axon_map = AxonMapSpatial(implant=ArgusII(), step=2).build()
    source = Stimulus({'A5': BiphasicPulseTrain(20, 1 * xTh, 0.45,
                                                      threshold_amp=1 * uA)})
    percept = model.predict_percept(source)
    percept_axon = axon_map.predict_percept(source)
    npt.assert_almost_equal(
        percept.data[:, :, 0], percept_axon.max(axis='frames'))

    # Effect models must be callable
    model = BiphasicAxonMapModel(implant=ArgusII(), step=2)
    model.spatial.bright_model = 1.0
    with pytest.raises(TypeError):
        model.build()

    # If t_percept is not specified, there should only be one frame
    model = BiphasicAxonMapModel(implant=ArgusII(), step=2)
    model.build()
    implant = ArgusII()
    source = Stimulus({'A5': BiphasicPulseTrain(20, 1 * xTh, 0.45)})
    percept = model.predict_percept(source)
    npt.assert_equal(percept.time is None, True)
    # If t_percept is specified, only the first frame has data:
    percept = model.predict_percept(source, t_percept=[0, 1, 2, 5, 10])
    npt.assert_equal(len(percept.time), 5)
    npt.assert_equal(np.any(percept.data[:, :, 0]), True)
    npt.assert_equal(np.any(percept.data[:, :, 1:]), False)

    # Test that default models give expected values
    model = BiphasicAxonMapModel(implant=ArgusII(), rho=400, lam=600,
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

    # Effect-model coefficients are accessed only through their effect model:
    for atr in ['a' + str(i) for i in range(0, 10)]:
        npt.assert_equal(hasattr(model.spatial, atr), False)
    with pytest.raises(FreezeError):
        model.spatial.a0 = 5
    npt.assert_equal(model.spatial.bright_model.a0, 2.095)
    model.spatial.bright_model.a0 = 5
    npt.assert_equal(model.spatial.bright_model.a0, 5)
    # Effect-model coefficients are independent:
    npt.assert_equal(model.spatial.size_model.a0, 2.095)
    npt.assert_equal(hasattr(model.spatial.streak_model, 'a0'), False)

    # `rho` and `lam` are copied to the size and streak models:
    model.spatial.rho = 350
    model.spatial.lam = 450
    npt.assert_equal(model.spatial.size_model.rho, 350)
    npt.assert_equal(model.spatial.streak_model.lam, 450)
    npt.assert_equal(model.spatial.rho, 350)
    npt.assert_equal(model.spatial.lam, 450)

    # Effect model parameters are not constructor arguments:
    with pytest.raises(TypeError):
        BiphasicAxonMapModel(implant=ArgusII(), a0=5)
    model = BiphasicAxonMapModel(implant=ArgusII(), rho=432)
    npt.assert_equal(model.spatial.rho, 432)
    npt.assert_equal(model.spatial.size_model.rho, 432)

    # Unknown parameters are rejected:
    with pytest.raises(FreezeError):
        model.spatial.invalid_param = 5

    # A custom size model still receives `rho`:
    model = BiphasicAxonMapModel(implant=ArgusII())

    class TestSizeModel():
        def __init__(self):
            self.rho = None
            self.test_param = 5

        def __call__(self, freq, amp, pdur):
            return 1
    model.spatial.size_model = TestSizeModel()
    model.spatial.rho = 321
    npt.assert_equal(model.spatial.size_model.rho, 321)
    npt.assert_equal(model.spatial.size_model.test_param, 5)
    npt.assert_equal(hasattr(model.spatial, 'test_param'), False)

    # User can override default values
    model = BiphasicAxonMapModel(implant=ArgusII())
    for key, value in set_params.items():
        setattr(model.spatial, key, value)
        npt.assert_equal(getattr(model.spatial, key), value)
    model = BiphasicAxonMapModel(implant=ArgusII(), **set_params)
    model.spatial.build(**set_params)
    for key, value in set_params.items():
        npt.assert_equal(getattr(model.spatial, key), value)

    # Zeros in, zeros out:
    source = np.zeros(60)
    npt.assert_almost_equal(model.predict_percept(source).data, 0)
    source = np.zeros(60)
    npt.assert_almost_equal(model.predict_percept(source).data, 0)

    # `eye` comes from the implant and cannot be passed to the model:
    npt.assert_equal(
        BiphasicAxonMapModel(implant=ArgusII(eye='left'), step=5).spatial.eye,
        'left')
    with pytest.raises(TypeError):
        BiphasicAxonMapModel(implant=ArgusII(), eye='left')

    # Lambda cannot be too small:
    with pytest.raises(ValueError):
        BiphasicAxonMapModel(implant=ArgusII(), lam=9).build()


def test_DefaultStreakModel_removed_axlambda():
    # `axlambda` (renamed to `lam` in 0.10.0) was removed in 0.11.0:
    with pytest.raises(TypeError):
        DefaultStreakModel(axlambda=200)
    npt.assert_equal(DefaultStreakModel(lam=200).lam, 200)
    npt.assert_equal(DefaultStreakModel(200).lam, 200)


@pytest.mark.parametrize('compose', [False, True])
def test_scaled_pulse_train_changes_percept(compose):
    model = BiphasicAxonMapModel(implant=ArgusII(), xrange=(-12, 12),
                                 yrange=(-8, 8), step=1,
                                 n_ax_segments=30).build()
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


@pytest.mark.parametrize('modify', [lambda s: s + 5, lambda s: s * np.inf,
                                    lambda s: s.append(s >> 1)])
def test_modified_pulse_train_rejected(modify):
    model = _granley(ArgusII())
    source = model.implant.prepare_stim(
        {'C5': BiphasicPulseTrain(20, 1 * xTh, 0.45, stim_dur=100)})
    with np.errstate(divide='ignore', invalid='ignore'):
        source = modify(source)
    with pytest.raises(TypeError):
        model.predict_percept(source)


def test_pulse_train_amp_sign_does_not_change_percept():
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


@pytest.mark.parametrize('amp', (2.0, 50.0))
def test_BiphasicAxonMapModel_cutoff_error_bound(amp, monkeypatch):
    freq, pdur = 20, 0.45
    source = {e: BiphasicPulseTrain(freq, amp * xTh, pdur)
                            for e in ArgusII().electrode_names}
    kwargs = {'xrange': (-14, 14), 'yrange': (-10, 10), 'step': 0.75,
              'rho': 200, 'lam': 800, 'verbose': False}

    model = BiphasicAxonMapModel(implant=ArgusII(), **kwargs).build()
    default = model.predict_percept(source).data
    monkeypatch.setattr(SpatialModel, '_cutoff_r2',
                        lambda self, rho: np.float32(np.inf))
    exact = model.predict_percept(source).data

    n_el = model.implant.n_electrodes
    f_bright = np.asarray(model.spatial.bright_model(
        np.full(n_el, freq), np.full(n_el, amp), np.full(n_el, pdur)))
    dropped = _GAUSSIAN_CUTOFF * np.abs(f_bright).sum()
    assert np.abs(default - exact).max() <= dropped + 1e-6 * np.abs(exact).max()


@pytest.mark.parametrize('attr', ('size_model', 'streak_model'))
def test_BiphasicAxonMapModel_rejects_nonpositive_effects(attr):
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
    # Non-finite scaling factors raise ValueError:
    model = BiphasicAxonMapModel(implant=ArgusII(), xrange=(-4, 4), yrange=(-4, 4), step=1,
                                 verbose=False).build()
    setattr(model.spatial, attr,
            lambda freq, amp, pdur: np.full_like(np.asarray(amp, dtype=float),
                                                 bad))
    source = {'A2': BiphasicPulseTrain(20, 30 * xTh, 0.45)}
    with pytest.raises(ValueError, match=attr):
        model.predict_percept(source)


def test_BiphasicAxonMapModel_reduces_to_AxonMapModel():
    # With all effect factors at 1, this equals AxonMapModel:
    from pulse2percept.models.retina import AxonMapModel

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


def _one_axon_model(source, bright=1.0, size=1.0, streak=1.0, rho=200):
    """Return a built model and active-electrode ``(x, y)`` coordinates.

    Effect models return the given factors (scalar or callable)."""
    model = BiphasicAxonMapModel(ArgusII(), xrange=(-2, 2), yrange=(-2, 2),
                                 step=1, rho=rho, meridian_blend=0,
                                 n_axons=50, n_ax_segments=50,
                                 ignore_pickle=True, verbose=False)
    for attr, value in (('bright_model', bright), ('size_model', size),
                        ('streak_model', streak)):
        setattr(model.spatial, attr,
                value if callable(value) else lambda f, a, p, v=value: v)
    model.build()
    stim = model.implant.prepare_stim(source)
    x, y, _ = model.spatial._electrode_coords(model.implant.electrode_array,
                                              stim, electrodes=list(source))
    return model, np.c_[x, y]


def _give_every_pixel(model, segments):
    """Replace every pixel's axon with ``(x, y, sensitivity)`` rows"""
    spatial = model.spatial
    n_px, n_seg = spatial.grid.x.size, len(segments)
    spatial.axon_contrib = np.tile(np.array(segments, dtype=np.float32),
                                   (n_px, 1))
    spatial.axon_idx_start = np.arange(n_px) * n_seg
    spatial.axon_idx_end = spatial.axon_idx_start + n_seg


def test_BiphasicAxonMap_is_the_analytical_segment_response():
    # F_bright * exp(-r2 / (2 rho^2 F_size)) * sensitivity ** (1 / F_streak):
    rho, f_bright, f_size, f_streak, sens = 200, 1.3, 2.5, 0.4, 0.6
    source = {'C5': BiphasicPulseTrain(20, 2 * xTh, 0.45)}
    model, el = _one_axon_model(source, f_bright, f_size, f_streak, rho)
    dx, dy = 150, -80
    # A segment without a location contributes nothing:
    _give_every_pixel(model, [(np.nan, np.nan, 1),
                              (el[0, 0] + dx, el[0, 1] + dy, sens)])
    got = _frame(model.predict_percept(source))
    want = f_bright * np.exp(-(dx ** 2 + dy ** 2) / (2 * rho ** 2 * f_size) +
                             np.log(sens) / f_streak)
    npt.assert_allclose(got, want, rtol=1e-5)


def test_BiphasicAxonMap_sums_electrodes_before_picking_a_segment():
    # Each segment sums both (signed) electrodes; the pixel then takes the
    # segment with the largest |sum|, keeping its sign:
    rho = 400
    source = {'A1': BiphasicPulseTrain(20, 2 * xTh, 0.45),
              'A2': BiphasicPulseTrain(20, 3 * xTh, 0.45)}
    model, el = _one_axon_model(
        source, bright=lambda f, a, p: np.where(a > 2.5, -0.8, 1.0), rho=rho)
    mid = el.mean(axis=0)
    _give_every_pixel(model, [(*mid, 1), (*el[1], 1)])

    def gauss(p, q):
        return np.exp(-np.sum((p - q) ** 2) / (2 * rho ** 2))

    at_mid = gauss(mid, el[0]) - 0.8 * gauss(mid, el[1])
    at_a2 = gauss(el[1], el[0]) - 0.8
    npt.assert_array_less(abs(at_mid), abs(at_a2))
    npt.assert_allclose(_frame(model.predict_percept(source)), at_a2,
                        rtol=1e-5)
    # Picking each electrode's strongest segment first would differ:
    npt.assert_array_less(0.1, abs(gauss(mid, el[0]) - 0.8 - at_a2))


def test_BiphasicAxonMap_first_segment_wins_a_tie():
    # Mirror-image segments give sums of equal |value| and opposite sign:
    source = {'A1': BiphasicPulseTrain(20, 2 * xTh, 0.45),
              'A2': BiphasicPulseTrain(20, 3 * xTh, 0.45)}
    model, el = _one_axon_model(
        source, bright=lambda f, a, p: np.where(a > 2.5, -1.0, 1.0), rho=400)
    _give_every_pixel(model, [(*el[0], 1), (*el[1], 1)])
    got = _frame(model.predict_percept(source))
    npt.assert_array_less(0, got)
    _give_every_pixel(model, [(*el[1], 1), (*el[0], 1)])
    npt.assert_equal(_frame(model.predict_percept(source)), -got)


def test_BiphasicAxonMap_effects_follow_their_electrode():
    # The kernel sums electrodes in x order; here the active order is the
    # reverse, and F_size/F_streak differ per electrode:
    rho = 200
    source = {'A10': BiphasicPulseTrain(20, 2 * xTh, 0.45),
              'B1': BiphasicPulseTrain(20, 3 * xTh, 0.45)}
    f_size, f_streak = np.array([0.5, 3.0]), np.array([2.0, 0.3])
    model, el = _one_axon_model(
        source, size=lambda f, a, p: np.where(a > 2.5, *f_size[::-1]),
        streak=lambda f, a, p: np.where(a > 2.5, *f_streak[::-1]), rho=rho)
    npt.assert_array_less(el[1, 0], el[0, 0])
    segments = np.array([(*(el[0] + [150, 0]), 0.9),
                         (*(el[1] + [50, 0]), 0.5)])
    _give_every_pixel(model, segments)

    def oracle(size, streak):
        r2 = ((segments[:, None, :2] - el) ** 2).sum(axis=-1)
        resp = np.exp(-r2 / (2 * rho ** 2 * size) +
                      np.log(segments[:, 2:]) / streak).sum(axis=1)
        return resp[np.argmax(np.abs(resp))]

    want = oracle(f_size, f_streak)
    npt.assert_allclose(_frame(model.predict_percept(source)), want,
                        rtol=1e-5)
    # Swapped factors give a different percept:
    npt.assert_array_less(0.1, abs(oracle(f_size[::-1], f_streak[::-1]) -
                                   want))


@pytest.mark.parametrize('kernel', ('scoreboard', 'axon'))
def test_Biphasic_cutoff_scales_with_F_size(kernel):
    # Kept iff r2 <= cutoff_r2 * F_size, inclusive: 100 * 4 = 400 um^2.
    import torch
    from pulse2percept.models.base import _scoreboard_response
    from pulse2percept.models.retina.beyeler2019 import _axon_gauss
    x = np.array([19.99, 20, 20.01], dtype=np.float32)
    rho, cutoff_r2, f_size = 10, 100, np.array([4], dtype=np.float32)
    if kernel == 'scoreboard':
        got = _scoreboard_response(torch.ones((1, 1)), (x, np.zeros(3)),
                                   (np.zeros(1), np.zeros(1)), rho,
                                   cutoff_r2, 0, spread_scale=f_size)[:, 0]
    else:
        seg = torch.tensor(np.c_[x, np.zeros(3), np.ones(3)],
                           dtype=torch.float32)
        got = _axon_gauss(seg, torch.zeros(1), torch.zeros(1),
                          torch.tensor(2 * rho ** 2 * f_size),
                          torch.tensor(cutoff_r2 * f_size))[:-1, 0]
    npt.assert_equal(got.numpy() > 0, [True, True, False])
    npt.assert_allclose(got[1], np.exp(-400 / (2 * rho ** 2 * 4)),
                        rtol=1e-6)


def test_BiphasicAxonMap_t_percept_units():
    # `t_percept` accepts time units (model overrides `predict_percept`):
    source = {'A1': BiphasicPulseTrain(20, 1 * xTh, 0.45,
                                                     stim_dur=100)}
    model = BiphasicAxonMapModel(implant=ArgusII(), step=2).build()
    # A scalar `t_percept` gives one frame:
    npt.assert_equal(model.predict_percept(source,
                                           t_percept=20).data.shape[-1], 1)
    bare = model.predict_percept(source, t_percept=[0, 20])
    for spelling in ([0, 20] * ms, np.array([0, 0.02]) * s, 20 * ms):
        unitful = model.predict_percept(source, t_percept=spelling)
        npt.assert_allclose(unitful.data.max(), bare.data.max(), rtol=1e-12)
    with pytest.raises(DimensionMismatchError):
        model.predict_percept(source, t_percept=[0, 20] * uA)


def test_BiphasicAxonMap_dimension_before_waveform():
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))

    class Projector(ArgusII):
        stimulus_unit = dimensionless

    projector = Projector(preprocess=False)
    model = BiphasicAxonMapModel(implant=projector, step=2).build()
    with pytest.raises(DimensionMismatchError) as excinfo:
        model.predict_percept(img)
    for accepted in ('electric current', 'threshold ratio'):
        npt.assert_equal(accepted in str(excinfo.value), True)
    npt.assert_equal('dimensionless' in str(excinfo.value), True)
    # A current with the wrong waveform gives the model-specific error:
    with pytest.raises(TypeError) as excinfo:
        BiphasicAxonMapModel(implant=ArgusII(), step=2).build().predict_percept(
            {'A1': MonophasicPulse(-1, 0.45, stim_dur=100)})
    npt.assert_equal('BiphasicPulseTrain' in str(excinfo.value), True)


def test_BiphasicAxonMapModel_meridian_blend():
    def make(**params):
        return BiphasicAxonMapModel(implant=ArgusII(), xrange=(-6, 6),
                                    yrange=(-6, 6), step=0.25, rho=200,
                                    lam=400, n_axons=250, n_ax_segments=200,
                                    ignore_pickle=True, **params).build()

    source = {'C4': BiphasicPulseTrain(20, 20 * xTh, 0.45),
                            'C8': BiphasicPulseTrain(20, 20 * xTh, 0.45)}
    plain = make(meridian_blend=0)
    unblended = plain.predict_percept(source).data

    # Default width is inherited from `AxonMapSpatial`:
    width = 1
    blended_model = make()
    npt.assert_equal(blended_model.spatial.meridian_blend, width)
    blended = blended_model.predict_percept(source).data
    npt.assert_equal(blended.shape, unblended.shape)
    npt.assert_equal(blended.dtype, unblended.dtype)
    npt.assert_equal(np.array_equal(blended, unblended), False)
    y = plain.spatial.grid.y[:, 0]
    delta = np.abs(blended - unblended)
    rows = delta.max(axis=(1, 2)) > delta.max() * 1e-3
    npt.assert_array_less(np.abs(y[rows]).max(), 4 * width)


@contextmanager
def _no_pulse_train_rendering():
    # Rendering a pulse train's waveform raises AssertionError:
    original = BiphasicPulseTrain._render

    def refuse(self):
        raise AssertionError('generated a pulse train waveform')
    BiphasicPulseTrain._render = refuse
    try:
        yield
    finally:
        BiphasicPulseTrain._render = original


@pytest.mark.parametrize('build_stim', [
    lambda: BiphasicPulseTrain(20, 1 * xTh, 0.45, stim_dur=100,
                               electrode='C5'),
    lambda: {'C5': BiphasicPulseTrain(20, 1 * xTh, 0.45, stim_dur=100),
             'A2': BiphasicPulseTrain(30, 2 * xTh, 0.45, stim_dur=100)},
    lambda: Stimulus({'C5': BiphasicPulseTrain(20, 1 * xTh, 0.45,
                                               stim_dur=100)}) * 2,
])
def test_BiphasicAxonMap_predicts_without_a_waveform(build_stim):
    model = _granley(ArgusII())
    source = build_stim()
    with _no_pulse_train_rendering():
        percept = model.predict_percept(source)
    npt.assert_equal(np.any(percept.data), True)


def test_BiphasicAxonMap_ignores_user_metadata():
    # User metadata with pulse parameter names is ignored:
    model = _granley(ArgusII())
    source = model.implant.prepare_stim(
        {'C5': BiphasicPulseTrain(20, 1 * xTh, 0.45, stim_dur=100)})
    expected = model.predict_percept(source).data
    source.metadata['user'] = {'amp': 999, 'freq': 1}
    npt.assert_array_equal(model.predict_percept(source).data, expected)


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
def test_BiphasicAxonMap_rejects_what_it_cannot_read(build_stim):
    # Unsupported waveforms (a sequence of two trains has no single freq):
    model = _granley(ArgusII())
    source = build_stim()
    with pytest.raises(TypeError):
        model.predict_percept(source)


def test_BiphasicAxonMap_zero_amplitude_is_inactive():
    model = _granley(ArgusII())
    source = {'C5': BiphasicPulseTrain(20, 0 * xTh, 0.45,
                                                     stim_dur=100),
                            'A2': BiphasicPulseTrain(20, 0 * xTh, 0.45,
                                                     stim_dur=100)}
    with _no_pulse_train_rendering():
        percept = model.predict_percept(source)
    npt.assert_almost_equal(percept.data, 0)
    # Zero-amplitude electrodes do not affect the percept:
    source = {'C5': BiphasicPulseTrain(20, 0 * xTh, 0.45, stim_dur=100),
                    'A2': BiphasicPulseTrain(20, 1 * xTh, 0.45, stim_dur=100)}
    with _no_pulse_train_rendering():
        mixed = model.predict_percept(source).data
    only = model.predict_percept({'A2': BiphasicPulseTrain(20, 1 * xTh, 0.45, stim_dur=100)}).data
    npt.assert_array_almost_equal(mixed, only)


_GRID = dict(xrange=(-3, 3), yrange=(-2, 2), step=1, n_ax_segments=30)


def _granley(implant):
    return BiphasicAxonMapModel(implant=implant, **_GRID).build()


def test_BiphasicAxonMap_reads_an_encoded_image():
    model = _granley(ArgusII(thresholds=80 * uA))
    with _no_pulse_train_rendering():
        percept = model.predict_percept(samples.logo_bvl())
    npt.assert_equal(np.any(percept.data), True)
    relative = _granley(
        ArgusII(encoder=AmplitudeEncoder(amp_range=(0 * xTh, 0.625 * xTh),
                                         freq=6 * Hz)))
    with _no_pulse_train_rendering():
        npt.assert_array_almost_equal(
            relative.predict_percept(samples.logo_bvl()).data, percept.data)


def test_BiphasicAxonMap_ignores_the_raster():
    with _no_pulse_train_rendering():
        rastered = _granley(ArgusII(thresholds=80)).predict_percept(
            samples.logo_bvl())
        at_once = _granley(ArgusII(thresholds=80, raster=None)).predict_percept(
            samples.logo_bvl())
    npt.assert_array_equal(rastered.data, at_once.data)


@pytest.mark.parametrize('param, value', [('amp_range', (0, 100)),
                                          ('freq', 20 * Hz),
                                          ('phase_dur', 0.2 * ms)])
def test_BiphasicAxonMap_reads_the_encoder_parameters(param, value):
    default = _granley(ArgusII(thresholds=80))
    tweaked = _granley(ArgusII(thresholds=80,
                               encoder=AmplitudeEncoder(**{param: value})))
    with _no_pulse_train_rendering():
        base = default.predict_percept(samples.logo_bvl()).data
        other = tweaked.predict_percept(samples.logo_bvl()).data
    npt.assert_equal(np.allclose(base, other), False)


def test_BiphasicAxonMap_encoded_current_still_needs_a_threshold():
    model = _granley(ArgusII())
    with pytest.raises(ValueError) as err:
        model.predict_percept(samples.logo_bvl())
    npt.assert_equal('threshold' in str(err.value), True)


def test_BiphasicAxonMap_rejects_an_encoded_video():
    model = _granley(ArgusII(thresholds=80))
    video = VideoStimulus(np.random.rand(4, 4, 3), time=[0, 200, 400])
    with pytest.raises(NotImplementedError) as err:
        model.predict_percept(video)
    npt.assert_equal('frames' in str(err.value), True)


def test_BiphasicAxonMap_rejects_a_custom_encoder_pulse():
    encoder = AmplitudeEncoder(pulse=BiphasicPulse(1, 0.2), amp_range=(0, 50))
    model = _granley(ArgusII(thresholds=80, encoder=encoder))
    with pytest.raises(TypeError) as err:
        model.predict_percept(samples.logo_bvl())
    npt.assert_equal('pulse' in str(err.value), True)


def test_BiphasicAxonMap_encoded_extraction_is_lazy():
    # Reading the schedule's parameters does not render its samples:
    implant = ArgusII(thresholds=80)
    stim = implant.prepare_stim(samples.logo_bvl())
    _granley(implant).predict_percept(stim)
    npt.assert_equal(stim._Stimulus__stim['data'] is None, True)


def _threshold_model():
    return BiphasicAxonMapModel(implant=ArgusII(), xrange=(-3, 3), 
                                yrange=(-2, 2), step=1,
                                n_ax_segments=30).build()


def _percept_at(model, train, thresholds=None):
    if thresholds is not None:
        model.implant.thresholds = thresholds
    return model.predict_percept({'A2': train}).data


def test_BiphasicAxonMap_reads_threshold_multiples_not_current():
    model = _threshold_model()
    # 2xTh and 160 uA on an 80 uA threshold electrode give the same percept:
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


def test_BiphasicAxonMap_uncalibrated_current_raises():
    model = _granley(ArgusII())
    source = {'A2': BiphasicPulseTrain(20, 160 * uA, 0.45,
                                                     stim_dur=100)}
    with pytest.raises(ValueError) as err:
        model.predict_percept(source)
    for remedy in ('2 * xTh', 'threshold_amp', 'implant.thresholds'):
        npt.assert_equal(remedy in str(err.value), True)
    model.implant.thresholds = 80 * uA
    npt.assert_equal(np.any(model.predict_percept(source).data), True)


def test_BiphasicAxonMap_zero_current_needs_no_threshold():
    model = _granley(ArgusII())
    source = {'A2': BiphasicPulseTrain(20, 0 * uA, 0.45,
                                                     stim_dur=100)}
    with _no_pulse_train_rendering():
        percept = model.predict_percept(source)
    npt.assert_almost_equal(percept.data, 0)


def test_BiphasicAxonMap_n_gray():
    source = {'C5': BiphasicPulseTrain(20, 2 * xTh, 0.45, stim_dur=100)}
    model = _granley(ArgusII())
    full = model.predict_percept(source)
    npt.assert_equal(np.unique(full.data).size > 2, True)
    # Metadata stores the prepared Stimulus:
    npt.assert_equal(isinstance(full.metadata['stim'], Stimulus), True)

    model = BiphasicAxonMapModel(implant=ArgusII(), n_gray=2, **_GRID).build()
    quantized = model.predict_percept(source)
    npt.assert_equal(np.unique(quantized.data).size, 2)


# -----------------------------------------------------------------------------
# BiphasicScoreboardModel
# -----------------------------------------------------------------------------

_SB_GRID = dict(xrange=(-6, 6), yrange=(-6, 6), step=0.25, rho=200,
                verbose=False)


def _scoreboard(implant=None, **kwargs):
    return BiphasicScoreboardModel(
        implant=ArgusII() if implant is None else implant,
        **{**_SB_GRID, **kwargs}).build()


def _train(freq=20, amp=1, pdur=0.45):
    return {'C5': BiphasicPulseTrain(freq, amp * xTh, pdur)}


@pytest.mark.parametrize('bad', [np.nan, np.inf])
@pytest.mark.parametrize('model', [
    _scoreboard,
    lambda: BiphasicAxonMapModel(ArgusII(), n_axons=50, n_ax_segments=50,
                                 ignore_pickle=True, **_SB_GRID),
])
def test_Biphasic_rejects_nonfinite_pulse_amplitude(model, bad):
    with pytest.raises(ValueError, match='Pulse-train parameters must be'):
        model().predict_percept(_train(amp=bad))


def _frame(percept):
    """Return the single frame this model predicts"""
    return percept.data[..., 0]


def _effective_width(frame):
    """Return sum over peak, a height-independent Gaussian width"""
    return frame.sum() / frame.max()


def test_BiphasicScoreboard_amplitude_brightens_and_broadens():
    model = _scoreboard()
    weak = _frame(model.predict_percept(_train(amp=1)))
    strong = _frame(model.predict_percept(_train(amp=2)))
    npt.assert_array_less(weak.max(), strong.max())
    npt.assert_array_less(_effective_width(weak), _effective_width(strong))


def test_BiphasicScoreboard_frequency_brightens_only():
    # The default size model ignores frequency, so higher freq only brightens:
    model = _scoreboard()
    slow = _frame(model.predict_percept(_train(freq=20)))
    fast = _frame(model.predict_percept(_train(freq=40)))
    npt.assert_array_less(slow.max(), fast.max())
    npt.assert_almost_equal(_effective_width(slow), _effective_width(fast),
                            decimal=4)


def test_BiphasicScoreboard_is_the_analytical_gaussian(monkeypatch):
    freq, amp, pdur, rho = 20, 1.5, 0.45, 200
    # No cutoff, so the Gaussian is not truncated:
    monkeypatch.setattr(SpatialModel, '_cutoff_r2',
                        lambda self, rho: np.float32(np.inf))
    model = _scoreboard(rho=rho)
    got = _frame(model.predict_percept(_train(freq, amp, pdur)))

    spatial = model.spatial
    f_bright = DefaultBrightModel()(freq, amp, pdur)
    f_size = DefaultSizeModel(rho)(freq, amp, pdur)
    stim = model.implant.prepare_stim(_train(freq, amp, pdur))
    x, y, _ = spatial._electrode_coords(model.implant.electrode_array, stim,
                                        electrodes=['C5'])
    r2 = (spatial.grid.ret.x - x[0]) ** 2 + (spatial.grid.ret.y - y[0]) ** 2
    want = f_bright * np.exp(-r2 / (2 * rho ** 2 * f_size))
    npt.assert_allclose(got, want, rtol=1e-5, atol=1e-6 * want.max())


def test_BiphasicScoreboard_pairs_pulses_with_their_own_electrode(monkeypatch):
    # Two electrodes with different amp/pdur, in reverse implant order:
    # `_elec_params` and `_electrode_coords` must use the same order.
    rho, freq = 200, 20
    conditions = [('F10', 3.0, 0.9), ('A1', 1.0, 0.25)]
    names = list(ArgusII().electrode_names)
    npt.assert_array_less(names.index(conditions[1][0]),
                          names.index(conditions[0][0]))
    monkeypatch.setattr(SpatialModel, '_cutoff_r2',
                        lambda self, rho: np.float32(np.inf))
    model = _scoreboard(rho=rho, xrange=(-8, 8), yrange=(-6, 6), step=0.5)
    source = {name: BiphasicPulseTrain(freq, amp * xTh, pdur)
              for name, amp, pdur in conditions}
    got = _frame(model.predict_percept(source))

    spatial = model.spatial
    stim = model.implant.prepare_stim(source)

    def gaussian(name, amp, pdur):
        x, y, _ = spatial._electrode_coords(model.implant.electrode_array,
                                            stim, electrodes=[name])
        r2 = ((spatial.grid.ret.x - x[0]) ** 2 +
              (spatial.grid.ret.y - y[0]) ** 2)
        f_size = DefaultSizeModel(rho)(freq, amp, pdur)
        return DefaultBrightModel()(freq, amp, pdur) * np.exp(
            -r2 / (2 * rho ** 2 * f_size))

    want = sum(gaussian(*condition) for condition in conditions)
    npt.assert_allclose(got, want, rtol=1e-5, atol=1e-6 * want.max())
    # Swapping the conditions gives a different percept:
    swapped = sum(gaussian(name, amp, pdur)
                  for (name, _, _), (_, amp, pdur) in zip(conditions,
                                                          conditions[::-1]))
    npt.assert_array_less(0.1 * want.max(), np.abs(got - swapped).max())


def test_BiphasicScoreboard_uncalibrated_current_raises():
    model = _scoreboard()
    current = {'C5': BiphasicPulseTrain(20, 30 * uA, 0.45)}
    with pytest.raises(ValueError) as err:
        model.predict_percept(current)
    npt.assert_equal('threshold' in str(err.value), True)
    # Works once the implant has thresholds:
    calibrated = _scoreboard(implant=ArgusII(thresholds=30 * uA))
    npt.assert_array_almost_equal(calibrated.predict_percept(current).data,
                                  model.predict_percept(_train(amp=1)).data)


def test_BiphasicScoreboard_reads_an_encoded_image():
    model = _scoreboard(implant=ArgusII(thresholds=80 * uA), step=1)
    with _no_pulse_train_rendering():
        percept = model.predict_percept(samples.logo_bvl())
    npt.assert_equal(np.any(percept.data), True)
    # An encoder in threshold multiples needs no implant thresholds:
    relative = _scoreboard(
        step=1,
        implant=ArgusII(encoder=AmplitudeEncoder(
            amp_range=(0 * xTh, 0.625 * xTh), freq=6 * Hz)))
    with _no_pulse_train_rendering():
        npt.assert_array_almost_equal(
            relative.predict_percept(samples.logo_bvl()).data, percept.data)


@pytest.mark.parametrize('build_stim', [
    lambda: {'C5': 30},
    lambda: np.full(60, 30.0),
    lambda: {'C5': Stimulus([[0, 30, 30, 0]], time=[0, 1, 99, 100])},
    lambda: {'C5': MonophasicPulse(-30, 0.45, stim_dur=100)},
    lambda: {'C5': BiphasicPulseTrain(20, 30, 0.45, delay_dur=1,
                                      stim_dur=100)},
])
def test_BiphasicScoreboard_rejects_unstructured_stimuli(build_stim):
    # Amplitudes or raw waveforms do not specify the pulse:
    model = _scoreboard(implant=ArgusII(thresholds=30 * uA), step=1)
    with pytest.raises(TypeError):
        model.predict_percept(build_stim())


def test_BiphasicScoreboard_rejects_normalized_drive():
    # Normalized photovoltaic drive does not specify a pulse:
    from pulse2percept.implants.retina import PRIMAPivotal
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        model = BiphasicScoreboardModel(implant=PRIMAPivotal(), rho=200,
                                        step=0.5, xrange=(-2, 2),
                                        yrange=(-2, 2), verbose=False)
        with pytest.raises(DimensionMismatchError):
            model.predict_percept(samples.logo_bvl())


def test_BiphasicScoreboard_zero_stimulation_is_zero():
    model = _scoreboard(step=1)
    with _no_pulse_train_rendering():
        percept = model.predict_percept(
            {e: BiphasicPulseTrain(20, 0 * xTh, 0.45) for e in ('C5', 'A2')})
    npt.assert_almost_equal(percept.data, 0)
    # Zero-amplitude electrodes do not affect the percept:
    with _no_pulse_train_rendering():
        mixed = model.predict_percept(
            {'C5': BiphasicPulseTrain(20, 0 * xTh, 0.45),
             'A2': BiphasicPulseTrain(20, 1 * xTh, 0.45)}).data
    only = model.predict_percept(
        {'A2': BiphasicPulseTrain(20, 1 * xTh, 0.45)}).data
    npt.assert_array_almost_equal(mixed, only)


def test_BiphasicScoreboard_reduces_to_ScoreboardModel():
    # With all effect factors at 1, this equals ScoreboardModel:
    electrodes = ('A2', 'C5', 'F8')
    biphasic = BiphasicScoreboardModel(implant=ArgusII(), **_SB_GRID)
    for attr in ('bright_model', 'size_model'):
        setattr(biphasic.spatial, attr,
                lambda freq, amp, pdur: np.ones_like(np.asarray(amp,
                                                                dtype=float)))
    biphasic.build()
    got = biphasic.predict_percept(
        {e: BiphasicPulseTrain(20, 1 * xTh, 0.45) for e in electrodes}).data

    plain = ScoreboardModel(implant=ArgusII(), **_SB_GRID).build()
    unit = np.zeros(ArgusII().n_electrodes)
    names = list(ArgusII().electrode_names)
    for e in electrodes:
        unit[names.index(e)] = 1.0
    want = plain.predict_percept(unit).data
    npt.assert_allclose(got, want, rtol=1e-5, atol=1e-6 * np.abs(want).max())


@pytest.mark.parametrize('attr', ('bright_model', 'size_model'))
def test_BiphasicScoreboard_rejects_bad_effects(attr):
    model = _scoreboard(step=1)
    setattr(model.spatial, attr, lambda freq, amp, pdur: np.full_like(
        np.asarray(amp, dtype=float), np.nan))
    with pytest.raises(ValueError, match=attr):
        model.predict_percept(_train())
    if attr == 'size_model':
        # `F_size` is in an exponent denominator, so zero is invalid:
        setattr(model.spatial, attr,
                lambda freq, amp, pdur: np.zeros_like(amp))
        with pytest.raises(ValueError, match=attr):
            model.predict_percept(_train())
    setattr(model.spatial, attr, lambda freq, amp, pdur: np.ones_like(amp))
    npt.assert_equal(np.any(model.predict_percept(_train()).data), True)


def test_BiphasicScoreboard_rho_reaches_the_size_model():
    spatial = BiphasicScoreboardModel(implant=ArgusII(), **_SB_GRID).spatial
    npt.assert_almost_equal(spatial.size_model.rho, 200)
    spatial.rho = 0.3 * mm
    npt.assert_almost_equal(spatial.rho, 300)
    npt.assert_almost_equal(spatial.size_model.rho, 300)


def test_ScoreboardModel_is_unchanged():
    # Reference values from before BiphasicScoreboardModel was added:
    model = ScoreboardModel(implant=ArgusII(), xrange=(-8, 8), yrange=(-6, 6),
                            step=1, rho=200, verbose=False).build()
    stim = np.zeros(ArgusII().n_electrodes)
    names = list(ArgusII().electrode_names)
    for electrode, amp in (('A2', 20.0), ('C5', 30.0), ('F8', 40.0)):
        stim[names.index(electrode)] = amp
    data = model.predict_percept(stim).data
    npt.assert_equal(data.shape, (13, 17, 1))
    npt.assert_allclose(data.sum(), 297.30609130859375, rtol=1e-5)
    npt.assert_allclose(data.max(), 33.0367317199707, rtol=1e-5)
    npt.assert_allclose((data ** 2).sum(), 4960.07763671875, rtol=1e-5)


# -----------------------------------------------------------------------------
# Pulse-train input shared by both Granley models
# -----------------------------------------------------------------------------

#: Granley models and a small grid for each.
_GRANLEY = [(BiphasicAxonMapModel, dict(_GRID)),
            (BiphasicScoreboardModel, dict(xrange=(-3, 3), yrange=(-2, 2),
                                           step=1))]


@pytest.mark.parametrize('model_cls, grid', _GRANLEY)
def test_Granley_rejects_a_temporal_model(model_cls, grid):
    # One representative percept per train; there is no time course to shape:
    model = model_cls(implant=ArgusII(), verbose=False, **grid)
    with pytest.raises(TypeError, match='temporal'):
        model.temporal = FadingTemporal()
    npt.assert_equal(model.temporal, None)


@pytest.mark.parametrize('model_cls, grid', _GRANLEY)
@pytest.mark.parametrize('effect', ('bright_model', 'size_model'))
def test_Granley_effect_model_may_return_a_scalar(model_cls, grid, effect):
    # A scalar effect is broadcast to one factor per electrode (the kernels
    # have bounds checking off):
    model = model_cls(implant=ArgusII(), verbose=False, **grid).build()
    source = {e: BiphasicPulseTrain(20, 1 * xTh, 0.45)
              for e in ('A2', 'C5', 'F8')}
    spatial = model.spatial
    setattr(spatial, effect, lambda freq, amp, pdur: 1.0)
    scalar = model.predict_percept(source).data
    # Same as a per-electrode array:
    setattr(spatial, effect,
            lambda freq, amp, pdur: np.ones_like(np.asarray(amp, dtype=float)))
    npt.assert_array_equal(scalar, model.predict_percept(source).data)


@pytest.mark.parametrize('model_cls, grid', _GRANLEY)
def test_Granley_effect_model_length_must_match(model_cls, grid):
    # Effects must be scalar or one per electrode (kernels skip bounds
    # checks):
    model = model_cls(implant=ArgusII(), verbose=False, **grid).build()
    source = {e: BiphasicPulseTrain(20, 1 * xTh, 0.45)
              for e in ('A2', 'C5', 'F8')}
    model.spatial.bright_model = lambda freq, amp, pdur: np.ones(2)
    with pytest.raises(ValueError, match='bright_model'):
        model.predict_percept(source)


@pytest.mark.parametrize('model_cls, grid', _GRANLEY)
@pytest.mark.parametrize('build_train', [
    lambda: BiphasicPulseTrain(0, 1 * xTh, 0.45, stim_dur=100),
    lambda: BiphasicPulseTrain(20, 1 * xTh, 0.45, n_pulses=0, stim_dur=100),
])
def test_Granley_a_train_without_pulses_is_dark(model_cls, grid, build_train):
    # `freq=0` and `n_pulses=0` deliver no pulses, regardless of amplitude:
    model = model_cls(implant=ArgusII(), verbose=False, **grid).build()
    train = build_train()
    npt.assert_equal(train.n_pulses, 0)
    npt.assert_almost_equal(model.predict_percept({'C5': train}).data, 0)
    # A silent electrode does not affect the percept:
    driven = BiphasicPulseTrain(20, 1 * xTh, 0.45, stim_dur=100)
    npt.assert_array_almost_equal(
        model.predict_percept({'C5': build_train(), 'A2': driven}).data,
        model.predict_percept({'A2': driven}).data)


@pytest.mark.parametrize('model_cls, grid', _GRANLEY)
def test_Granley_rejects_anodic_first_trains(model_cls, grid):
    # [Granley2021]_ covers cathodic-first pulse trains only:
    model = model_cls(implant=ArgusII(), verbose=False, **grid).build()
    with pytest.raises(TypeError, match='cathodic-first'):
        model.predict_percept(
            {'C5': BiphasicPulseTrain(20, 1 * xTh, 0.45,
                                      cathodic_first=False)})


@pytest.mark.parametrize('model_cls, grid', _GRANLEY)
def test_Granley_rejects_an_anodic_first_encoder(model_cls, grid):
    # Same for encoded images:
    def encoded(cathodic_first):
        implant = ArgusII(encoder=AmplitudeEncoder(
            amp_range=(0 * xTh, 2 * xTh), cathodic_first=cathodic_first))
        return model_cls(implant=implant, verbose=False, **grid).build()

    with pytest.raises(TypeError, match='cathodic-first'):
        encoded(False).predict_percept(samples.logo_bvl())
    # Default (cathodic-first) polarity works:

    npt.assert_equal(
        np.any(encoded(True).predict_percept(samples.logo_bvl()).data), True)
