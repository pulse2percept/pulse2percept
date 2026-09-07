"""The public shape of :py:mod:`pulse2percept.models`

The root namespace is anatomy-neutral: it carries the abstract model classes,
the generic temporal models, and the two anatomical subpackages. Retinal
models were removed from the root in v0.11 without a deprecation alias, and
their implementation modules moved under ``models.retina``.
"""
import importlib

import numpy.testing as npt
import pytest

import pulse2percept.models as models

#: Names the root namespace still exports, and nothing else.
GENERIC = ['AlphaTemporal', 'BaseModel', 'FadingTemporal', 'Model',
           'SpatialModel', 'TemporalModel', 'cortex', 'retina']

#: Flat retinal names the root no longer exports.
MOVED = ['AxonMapModel', 'AxonMapSpatial', 'BiphasicAxonMapModel',
         'BiphasicAxonMapSpatial', 'BiphasicScoreboardModel',
         'BiphasicScoreboardSpatial', 'Horsager2009Model',
         'Horsager2009Temporal', 'Nanduri2012Model', 'Nanduri2012Spatial',
         'Nanduri2012Temporal', 'ScoreboardModel', 'ScoreboardSpatial',
         'Thompson2003Model', 'Thompson2003Spatial']

#: Implementation modules that moved out of the root.
MOVED_MODULES = ['beyeler2019', '_beyeler2019', 'granley2021', '_granley2021',
                 'horsager2009', '_horsager2009', 'nanduri2012',
                 '_nanduri2012', 'thompson2003', '_thompson2003']


def test_root_namespace_is_anatomy_neutral():
    npt.assert_equal(sorted(models.__all__), GENERIC)
    for name in GENERIC:
        npt.assert_equal(hasattr(models, name), True, err_msg=name)
    for name in MOVED:
        npt.assert_equal(hasattr(models, name), False, err_msg=name)


@pytest.mark.parametrize('module', MOVED_MODULES)
def test_old_implementation_modules_are_gone(module):
    with pytest.raises(ImportError):
        importlib.import_module(f'pulse2percept.models.{module}')


@pytest.mark.parametrize('module, names', [
    ('pulse2percept.models.retina', sorted(MOVED + ['RetinalSpatial'])),
    ('pulse2percept.models.cortex',
     ['CortexSpatial', 'DynaphosModel', 'ScoreboardModel',
      'ScoreboardSpatial']),
])
def test_canonical_imports(module, names):
    mod = importlib.import_module(module)
    npt.assert_equal(sorted(mod.__all__), names)
    for name in names:
        npt.assert_equal(hasattr(mod, name), True, err_msg=name)


@pytest.mark.parametrize('module, name', [
    ('pulse2percept.models.retina.base', 'RetinalSpatial'),
    ('pulse2percept.models.retina.beyeler2019', 'ScoreboardSpatial'),
    ('pulse2percept.models.retina.beyeler2019', 'ScoreboardModel'),
    ('pulse2percept.models.retina.beyeler2019', 'AxonMapSpatial'),
    ('pulse2percept.models.retina.beyeler2019', 'AxonMapModel'),
    ('pulse2percept.models.retina.granley2021', 'BiphasicAxonMapSpatial'),
    ('pulse2percept.models.retina.granley2021', 'BiphasicAxonMapModel'),
    ('pulse2percept.models.retina.granley2021', 'BiphasicScoreboardSpatial'),
    ('pulse2percept.models.retina.granley2021', 'BiphasicScoreboardModel'),
    ('pulse2percept.models.retina.horsager2009', 'Horsager2009Temporal'),
    ('pulse2percept.models.retina.horsager2009', 'Horsager2009Model'),
    ('pulse2percept.models.retina.nanduri2012', 'Nanduri2012Spatial'),
    ('pulse2percept.models.retina.nanduri2012', 'Nanduri2012Temporal'),
    ('pulse2percept.models.retina.nanduri2012', 'Nanduri2012Model'),
    ('pulse2percept.models.retina.thompson2003', 'Thompson2003Spatial'),
    ('pulse2percept.models.retina.thompson2003', 'Thompson2003Model'),
    ('pulse2percept.models.cortex.base', 'CortexSpatial'),
    ('pulse2percept.models.cortex.scoreboard', 'ScoreboardSpatial'),
    ('pulse2percept.models.cortex.scoreboard', 'ScoreboardModel'),
    ('pulse2percept.models.cortex.dynaphos', 'DynaphosModel'),
])
def test_defining_module(module, name):
    """Each model is defined in the module its package re-exports it from"""
    mod = importlib.import_module(module)
    package = importlib.import_module(module.rsplit('.', 1)[0])
    npt.assert_equal(getattr(mod, name) is getattr(package, name), True)


@pytest.mark.parametrize('name', ['ScoreboardSpatial', 'ScoreboardModel'])
def test_cortical_scoreboard_left_cortex_base(name):
    from pulse2percept.models.cortex import base
    npt.assert_equal(hasattr(base, name), False)


def test_scoreboard_kernels_are_shared_not_retinal():
    """The Gaussian spread kernels are generic, so cortex does not reach into
    a retinal module for them."""
    from pulse2percept.models import _scoreboard
    from pulse2percept.models.cortex import scoreboard
    from pulse2percept.models.retina import beyeler2019
    from pulse2percept.models.retina import _beyeler2019
    for name in ('fast_scoreboard', 'fast_scoreboard_3d'):
        npt.assert_equal(hasattr(_scoreboard, name), True, err_msg=name)
        npt.assert_equal(hasattr(_beyeler2019, name), False, err_msg=name)
    npt.assert_equal(scoreboard.fast_scoreboard is
                     _scoreboard.fast_scoreboard, True)
    npt.assert_equal(beyeler2019.fast_scoreboard is
                     _scoreboard.fast_scoreboard, True)
    # ... and the axon-map science stayed behind:
    for name in ('fast_axon_map', 'fast_jansonius', 'fast_find_closest_axon'):
        npt.assert_equal(hasattr(_beyeler2019, name), True, err_msg=name)
        npt.assert_equal(hasattr(_scoreboard, name), False, err_msg=name)
