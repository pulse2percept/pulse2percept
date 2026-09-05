"""Tests for the public model constructor signatures.

Concrete model constructors expose explicit signatures without ``*args`` or
``**kwargs``.
"""
import inspect

import numpy.testing as npt
import pytest

from pulse2percept.implants import ArgusII
from pulse2percept.implants.cortex import Orion
from pulse2percept.models import (AlphaTemporal, AxonMapModel, AxonMapSpatial,
                                  BiphasicAxonMapModel,
                                  BiphasicAxonMapSpatial,
                                  BiphasicScoreboardModel,
                                  BiphasicScoreboardSpatial, FadingTemporal,
                                  Horsager2009Model, Horsager2009Temporal,
                                  Nanduri2012Model, Nanduri2012Spatial,
                                  Nanduri2012Temporal, ScoreboardModel,
                                  ScoreboardSpatial, Thompson2003Model,
                                  Thompson2003Spatial)
from pulse2percept.models import cortex
from pulse2percept.models.granley2021 import (DefaultBrightModel,
                                              DefaultSizeModel,
                                              DefaultStreakModel)

#: Public concrete models bound to an implant, with the parameter each one
#: adds on top of the inherited spatial parameters.
IMPLANT_MODELS = [
    (ScoreboardSpatial, 'rho'),
    (ScoreboardModel, 'rho'),
    (AxonMapSpatial, 'lam'),
    (AxonMapModel, 'lam'),
    (BiphasicAxonMapSpatial, 'bright_model'),
    (BiphasicAxonMapModel, 'bright_model'),
    (BiphasicScoreboardSpatial, 'bright_model'),
    (BiphasicScoreboardModel, 'bright_model'),
    (Thompson2003Spatial, 'radius'),
    (Thompson2003Model, 'radius'),
    (Nanduri2012Spatial, 'atten_a'),
    (Nanduri2012Model, 'atten_a'),
    (cortex.ScoreboardSpatial, 'regions'),
    (cortex.ScoreboardModel, 'regions'),
    (cortex.DynaphosModel, 'rheobase'),
]

#: Public concrete models without an implant, and their own parameter.
STANDALONE_MODELS = [
    (FadingTemporal, 'tau'),
    (AlphaTemporal, 'tau'),
    (Horsager2009Temporal, 'beta'),
    (Horsager2009Model, 'beta'),
    (Nanduri2012Temporal, 'asymptote'),
]

#: Granley effect models, and their own parameter.
EFFECT_MODELS = [
    (DefaultBrightModel, 'a2'),
    (DefaultSizeModel, 'a5'),
    (DefaultStreakModel, 'a7'),
]

#: The spatial parameter an effect model scales, its only positional argument.
EFFECT_ARG = {DefaultSizeModel: 'rho', DefaultStreakModel: 'lam'}

ALL_MODELS = IMPLANT_MODELS + STANDALONE_MODELS + EFFECT_MODELS


def _implant_for(cls):
    """Return an implant this model can be bound to."""
    return Orion() if cls.__module__.startswith('pulse2percept.models.cortex') \
        else ArgusII()


def _construct(cls, **params):
    """Construct ``cls``, supplying the argument it takes positionally."""
    if any(cls is model for model, _ in IMPLANT_MODELS):
        return cls(_implant_for(cls), **params)
    if cls in EFFECT_ARG:
        # `rho`/`lam`, in microns:
        return cls(200, **params)
    return cls(**params)


@pytest.mark.parametrize('cls,own_param', ALL_MODELS)
def test_constructor_signature_is_explicit(cls, own_param):
    params = inspect.signature(cls).parameters
    catch_all = [p.name for p in params.values()
                 if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL)]
    npt.assert_equal(catch_all, [])
    npt.assert_equal(own_param in params, True)


@pytest.mark.parametrize('cls,_', IMPLANT_MODELS + STANDALONE_MODELS)
def test_inherited_params_are_in_the_signature(cls, _):
    # Effect models are excluded: they declare no inherited parameters.
    params = inspect.signature(cls).parameters
    on_implant = any(cls is m for m, _ in IMPLANT_MODELS)
    inherited = 'step' if on_implant else 'dt'
    npt.assert_equal(inherited in params, True)
    npt.assert_equal('verbose' in params, True)
    if on_implant:
        # Every model that has electrodes to displace exposes the subject's
        # phosphene locations:
        npt.assert_equal('location_noise' in params, True)


@pytest.mark.parametrize('cls,_', IMPLANT_MODELS)
def test_implant_is_positional_or_keyword(cls, _):
    implant = inspect.signature(cls).parameters['implant']
    npt.assert_equal(implant.kind, implant.POSITIONAL_OR_KEYWORD)
    npt.assert_equal(implant.default, implant.empty)
    # Both spellings bind the implant:
    device = _implant_for(cls)
    for model in (cls(device), cls(implant=device)):
        npt.assert_equal(model.implant is device, True)


@pytest.mark.parametrize('cls,own_param', ALL_MODELS)
def test_model_params_are_keyword_only(cls, own_param):
    params = inspect.signature(cls).parameters
    positional = [name for name, p in params.items()
                  if p.kind is not p.KEYWORD_ONLY]
    # Only the implant and effect-model scale parameters may be positional:
    expected = [name for name in ('implant', EFFECT_ARG.get(cls))
                if name in params]
    npt.assert_equal(positional, expected)


@pytest.mark.parametrize('cls,_', ALL_MODELS)
def test_unknown_keyword_raises(cls, _):
    with pytest.raises(TypeError):
        _construct(cls, not_a_model_param=1)


def test_named_spatial_model_reaches_the_spatial_component():
    model = ScoreboardModel(ArgusII(), rho=222, step=2, thresh_percept=0.5)
    npt.assert_almost_equal(model.spatial.rho, 222)
    npt.assert_almost_equal(model.spatial.step, 2)
    npt.assert_almost_equal(model.spatial.thresh_percept, 0.5)


def test_named_spatiotemporal_model_reaches_both_components():
    model = Nanduri2012Model(ArgusII(), atten_a=1234, step=2, tau1=0.5,
                             thresh_percept=0.5, verbose=False)
    npt.assert_almost_equal(model.spatial.atten_a, 1234)
    npt.assert_almost_equal(model.spatial.step, 2)
    npt.assert_almost_equal(model.temporal.tau1, 0.5)
    # `thresh_percept` and `verbose` are declared by both components:
    for component in (model.spatial, model.temporal):
        npt.assert_almost_equal(component.thresh_percept, 0.5)
        npt.assert_equal(component.verbose, False)


@pytest.mark.parametrize('cls,_', ALL_MODELS)
def test_defaults_match_declared_defaults(cls, _):
    model = _construct(cls)
    for component in [getattr(model, 'spatial', model),
                      getattr(model, 'temporal', None)]:
        if component is None:
            continue
        for name, default in component.get_default_params().items():
            if name in ('visual_field_map', 'n_jobs', 'n_threads',
                        'bright_model',
                        'size_model', 'streak_model'):
                # Object defaults, and a machine-dependent thread count.
                continue
            npt.assert_equal(getattr(component, name), default,
                             err_msg=f'{cls.__name__}.{name}')


@pytest.mark.parametrize('kwargs', [{'n_threads': 2}, {'n_jobs': 2},
                                    {'n_threads': 3, 'n_jobs': 2}])
def test_n_jobs_is_an_alias_for_n_threads(kwargs):
    # `n_jobs` writes through to `n_threads`, and wins when both are given.
    npt.assert_equal(AxonMapModel(ArgusII(), **kwargs).spatial.n_threads, 2)
    npt.assert_equal(FadingTemporal(**kwargs).n_threads, 2)
