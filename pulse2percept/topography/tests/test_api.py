"""The public shape of :py:mod:`pulse2percept.topography`

The root namespace is anatomy-neutral: it carries the generic machinery and
the two anatomical subpackages, and nothing else. Anatomy-specific maps were
removed from the root in v0.11 without a deprecation alias.
"""
import importlib

import numpy.testing as npt
import pytest

import pulse2percept.topography as topo


def test_root_namespace_is_anatomy_neutral():
    for name in ('Grid2D', 'VisualFieldMap', 'retina', 'cortex'):
        npt.assert_equal(hasattr(topo, name), True, err_msg=name)
    npt.assert_equal(sorted(topo.__all__),
                     ['Grid2D', 'VisualFieldMap', 'cortex', 'retina'])
    for name in ('RetinalMap', 'Curcio1990Map', 'Watson2014Map',
                 'Watson2014DisplaceMap', 'CorticalMap', 'Polimeni2006Map',
                 'NeuropythyMap'):
        npt.assert_equal(hasattr(topo, name), False, err_msg=name)
        with pytest.raises(ImportError):
            importlib.import_module(f'pulse2percept.topography.{name}')


@pytest.mark.parametrize('module, names', [
    ('pulse2percept.topography.retina',
     ['Curcio1990Map', 'RetinalMap', 'Watson2014DisplaceMap',
      'Watson2014Map']),
    ('pulse2percept.topography.cortex',
     ['CorticalMap', 'NeuropythyMap', 'Polimeni2006Map']),
])
def test_canonical_imports(module, names):
    mod = importlib.import_module(module)
    npt.assert_equal(sorted(mod.__all__), names)
    for name in names:
        npt.assert_equal(hasattr(mod, name), True, err_msg=name)


@pytest.mark.parametrize('module, name', [
    ('pulse2percept.topography.retina.base', 'RetinalMap'),
    ('pulse2percept.topography.retina.curcio1990', 'Curcio1990Map'),
    ('pulse2percept.topography.retina.watson2014', 'Watson2014Map'),
    ('pulse2percept.topography.retina.watson2014', 'Watson2014DisplaceMap'),
    ('pulse2percept.topography.cortex.base', 'CorticalMap'),
    ('pulse2percept.topography.cortex.polimeni2006', 'Polimeni2006Map'),
    ('pulse2percept.topography.cortex.neuropythy', 'NeuropythyMap'),
])
def test_defining_module(module, name):
    """Each map is defined in the module its package re-exports it from"""
    mod = importlib.import_module(module)
    package = importlib.import_module(module.rsplit('.', 1)[0])
    npt.assert_equal(getattr(mod, name) is getattr(package, name), True)
