"""The public shape of :py:mod:`pulse2percept.implants`

The root namespace is anatomy-neutral: it carries the generic device
machinery (electrodes, arrays, rasters, ensembles, the implant pipeline) and
the two anatomical subpackages, and nothing else. Retinal devices were
removed from the root in v0.11 without a deprecation alias, and anatomical
laterality lives with the target: ``eye`` on
:py:class:`~pulse2percept.implants.retina.RetinalImplant`, ``hemisphere`` on
:py:class:`~pulse2percept.implants.cortex.CorticalImplant`.
"""
import importlib
from inspect import signature

import numpy.testing as npt
import pytest

import pulse2percept.implants as implants
from pulse2percept.implants import (ElectrodeGrid, GridImplant, Implant,
                                    PointSource)
from pulse2percept.implants.cortex import (Cortivis, CorticalImplant, ICVP,
                                           LinearEdgeThread, Neuralink, Orion)
from pulse2percept.implants.retina import (AlphaAMS, AlphaIMS, ArgusI, ArgusII,
                                           BVT24, BVT44, Ho2019FlatArray,
                                           Huang2021Array, IMIE,
                                           Lorach2015Array, PRIMAPivotal,
                                           RetinalImplant)

RETINAL_DEVICES = [ArgusI, ArgusII, AlphaIMS, AlphaAMS, BVT24, BVT44, IMIE,
                   PRIMAPivotal, Lorach2015Array]
#: Retinal devices whose electrode naming or geometry depends on the eye.
EYE_SENSITIVE_DEVICES = [ArgusI, ArgusII, AlphaIMS, AlphaAMS, BVT24, BVT44,
                         IMIE]
CORTICAL_DEVICES = [Orion, Cortivis, ICVP]

GENERIC = ['CheckerboardRaster', 'cortex', 'CustomRaster', 'DiskElectrode',
           'Electrode', 'ElectrodeArray', 'ElectrodeGrid', 'EnsembleImplant',
           'GridImplant', 'HexElectrode', 'Implant', 'PointSource',
           'ProsthesisSystem', 'Raster', 'retina', 'SequentialRaster',
           'SquareElectrode']


def test_root_namespace_is_anatomy_neutral():
    npt.assert_equal(sorted(implants.__all__), sorted(GENERIC))
    for name in ('Implant', 'GridImplant', 'retina', 'cortex'):
        npt.assert_equal(hasattr(implants, name), True, err_msg=name)
    # No device, retinal or cortical, and no anatomical base class:
    for name in ('ArgusI', 'ArgusII', 'AlphaIMS', 'AlphaAMS', 'BVT24', 'BVT44',
                 'IMIE', 'PRIMAPivotal', 'PRIMA', 'PRIMA75', 'PRIMA55',
                 'PRIMA40', 'Lorach2015Array', 'Ho2019FlatArray',
                 'Huang2021Array', 'PhotovoltaicPixel', 'RectangleImplant',
                 'RetinalImplant', 'CorticalImplant', 'Orion', 'Cortivis',
                 'ICVP', 'Neuralink'):
        with pytest.raises(AttributeError):
            getattr(implants, name)


@pytest.mark.parametrize('name', ['argus', 'alpha', 'bvt', 'imie', 'prima'])
def test_flat_device_modules_are_gone(name):
    """Device implementations live under the target they stimulate"""
    with pytest.raises(ImportError):
        importlib.import_module(f'pulse2percept.implants.{name}')
    importlib.import_module(f'pulse2percept.implants.retina.{name}')


@pytest.mark.parametrize('module, names', [
    ('pulse2percept.implants.retina',
     ['AlphaAMS', 'AlphaIMS', 'ArgusI', 'ArgusII', 'BVT24', 'BVT44',
      'Ho2019FlatArray', 'Huang2021Array', 'IMIE', 'Lorach2015Array',
      'PRIMA', 'PRIMA40', 'PRIMA55', 'PRIMA75', 'PRIMAPivotal',
      'PhotovoltaicPixel', 'RetinalImplant']),
    ('pulse2percept.implants.cortex',
     ['CorticalImplant', 'Cortivis', 'EllipsoidElectrode', 'ICVP',
      'LinearEdgeThread', 'Neuralink', 'NeuralinkThread', 'Orion']),
])
def test_canonical_imports(module, names):
    mod = importlib.import_module(module)
    npt.assert_equal(sorted(mod.__all__), names)
    for name in names:
        npt.assert_equal(hasattr(mod, name), True, err_msg=name)


@pytest.mark.parametrize('module, name', [
    ('pulse2percept.implants.base', 'Implant'),
    ('pulse2percept.implants.base', 'GridImplant'),
    ('pulse2percept.implants.ensemble', 'EnsembleImplant'),
    ('pulse2percept.implants.retina.base', 'RetinalImplant'),
    ('pulse2percept.implants.retina.argus', 'ArgusI'),
    ('pulse2percept.implants.retina.argus', 'ArgusII'),
    ('pulse2percept.implants.retina.alpha', 'AlphaIMS'),
    ('pulse2percept.implants.retina.alpha', 'AlphaAMS'),
    ('pulse2percept.implants.retina.bvt', 'BVT24'),
    ('pulse2percept.implants.retina.bvt', 'BVT44'),
    ('pulse2percept.implants.retina.imie', 'IMIE'),
    ('pulse2percept.implants.retina.prima', 'PRIMAPivotal'),
    ('pulse2percept.implants.retina.prima', 'PhotovoltaicPixel'),
    ('pulse2percept.implants.cortex.base', 'CorticalImplant'),
    ('pulse2percept.implants.cortex.orion', 'Orion'),
    ('pulse2percept.implants.cortex.cortivis', 'Cortivis'),
    ('pulse2percept.implants.cortex.icvp', 'ICVP'),
    ('pulse2percept.implants.cortex.neuralink', 'Neuralink'),
])
def test_defining_module(module, name):
    """Each class is defined in the module its package re-exports it from"""
    mod = importlib.import_module(module)
    package = importlib.import_module(module.rsplit('.', 1)[0])
    npt.assert_equal(getattr(mod, name) is getattr(package, name), True)


@pytest.mark.parametrize('implant', [
    Implant(PointSource(0, 0, 0)),
    GridImplant((2, 3), 400),
])
def test_generic_implants_have_no_laterality(implant):
    for attr in ('eye', 'hemisphere'):
        npt.assert_equal(hasattr(implant, attr), False, err_msg=attr)
    for cls in (Implant, GridImplant):
        npt.assert_equal('eye' in signature(cls).parameters, False)
        npt.assert_equal('hemisphere' in signature(cls).parameters, False)


def test_generic_ensemble_knows_nothing_about_cortex():
    """``from_cortical_map`` became ``from_visual_field_map`` in 0.11"""
    npt.assert_equal(hasattr(implants.EnsembleImplant, 'from_cortical_map'),
                     False)
    npt.assert_equal(
        hasattr(implants.EnsembleImplant, 'from_visual_field_map'), True)
    # And the module itself does not reach into topography.cortex:
    source = importlib.import_module('pulse2percept.implants.ensemble')
    npt.assert_equal('CorticalMap' in open(source.__file__).read(), False)


def test_retinal_implant_owns_the_eye():
    array = ElectrodeGrid((2, 3), 400)
    npt.assert_equal(RetinalImplant(array).eye, 'right')
    npt.assert_equal(RetinalImplant(array, eye='left').eye, 'left')
    # Case-insensitive, stored lowercase:
    npt.assert_equal(RetinalImplant(array, eye='LEFT').eye, 'left')
    npt.assert_equal(RetinalImplant(array, eye='Right').eye, 'right')
    # A retinal implant is not a hemisphere:
    npt.assert_equal(hasattr(RetinalImplant(array), 'hemisphere'), False)
    # The pre-0.11 codes are gone, not deprecated aliases:
    for bad in ('LE', 'RE', 'both', 'left eye', ''):
        with pytest.raises(ValueError):
            RetinalImplant(array, eye=bad)
    for bad in (None, 1, ['left']):
        with pytest.raises(TypeError):
            RetinalImplant(array, eye=bad)
    # Device arguments still reach Implant:
    implant = RetinalImplant(array, eye='left', preprocess=True,
                             safe_mode=True, max_current=100)
    npt.assert_equal(implant.preprocess, True)
    npt.assert_equal(implant.safe_mode, True)
    npt.assert_almost_equal(implant.max_current, 100)


@pytest.mark.parametrize('eye', ['left', 'right'])
@pytest.mark.parametrize('implant_type', EYE_SENSITIVE_DEVICES)
def test_canonicalized_eye_drives_the_geometry(implant_type, eye):
    """Case is normalized before the device lays out its electrodes

    These devices reverse their column names in the left eye, so reading the
    raw constructor argument rather than the canonical
    :py:attr:`~pulse2percept.implants.retina.RetinalImplant.eye` would let the
    metadata say 'left' while the geometry stayed right-eye.
    """
    lower, upper = implant_type(eye=eye), implant_type(eye=eye.upper())
    npt.assert_equal(lower.eye, eye)
    npt.assert_equal(upper.eye, eye)
    npt.assert_equal(lower.electrode_names, upper.electrode_names)
    npt.assert_array_equal(lower.electrode_array.coordinates(),
                           upper.electrode_array.coordinates())


@pytest.mark.parametrize('implant_type', RETINAL_DEVICES)
def test_a_retinal_device_is_a_retinal_implant(implant_type):
    implant = implant_type()
    npt.assert_equal(isinstance(implant, RetinalImplant), True)
    # BVT44 is the one device published for the left eye:
    npt.assert_equal(implant.eye,
                     'left' if implant_type is BVT44 else 'right')
    npt.assert_equal(implant_type(eye='left').eye, 'left')
    npt.assert_equal(f"eye='{implant.eye}'" in repr(implant), True)


@pytest.mark.parametrize('implant_type', [Ho2019FlatArray, Huang2021Array])
def test_a_sized_retinal_device_is_a_retinal_implant(implant_type):
    """The two families whose constructor takes a pixel size first"""
    npt.assert_equal(isinstance(implant_type(55), RetinalImplant), True)
    npt.assert_equal(implant_type(55).eye, 'right')
    npt.assert_equal(implant_type(55, eye='left').eye, 'left')


def test_cortical_implant_owns_the_hemisphere():
    array = ElectrodeGrid((2, 3), 400)
    # Unspecified by default: cortical coordinates and the model's
    # `implant_position` already say which side the device is on.
    npt.assert_equal(CorticalImplant(array).hemisphere, None)
    npt.assert_equal(CorticalImplant(array, hemisphere='left').hemisphere,
                     'left')
    npt.assert_equal(CorticalImplant(array, hemisphere='RIGHT').hemisphere,
                     'right')
    npt.assert_equal(hasattr(CorticalImplant(array), 'eye'), False)
    # The pre-0.11 codes are gone, not deprecated aliases:
    for bad in ('LH', 'RH', 'both', 'L', ''):
        with pytest.raises(ValueError):
            CorticalImplant(array, hemisphere=bad)
    for bad in (1, ['left']):
        with pytest.raises(TypeError):
            CorticalImplant(array, hemisphere=bad)
    # Unspecified is not printed; a recorded side is:
    npt.assert_equal('hemisphere' in repr(CorticalImplant(array)), False)
    npt.assert_equal("hemisphere='left'" in repr(
        CorticalImplant(array, hemisphere='left')), True)


@pytest.mark.parametrize('implant_type', CORTICAL_DEVICES)
def test_a_cortical_device_is_a_cortical_implant(implant_type):
    npt.assert_equal(isinstance(implant_type(), CorticalImplant), True)
    npt.assert_equal(implant_type().hemisphere, None)
    npt.assert_equal(implant_type(hemisphere='LEFT').hemisphere, 'left')
    with pytest.raises(ValueError):
        implant_type(hemisphere='LH')


@pytest.mark.parametrize('implant_type', CORTICAL_DEVICES)
def test_hemisphere_does_not_move_a_cortical_device(implant_type):
    """Recording a side is metadata, not a placement"""
    plain = implant_type()
    for hemisphere in ('left', 'right'):
        sided = implant_type(hemisphere=hemisphere)
        npt.assert_equal(sided.electrode_names, plain.electrode_names)
        npt.assert_array_equal(sided.electrode_array.coordinates(),
                               plain.electrode_array.coordinates())


def test_neuralink_has_the_same_hemisphere_contract():
    """Neuralink is an ensemble, not a CorticalImplant, but says the same"""
    threads = [LinearEdgeThread(x, 0, 0) for x in (0, 1000)]
    npt.assert_equal(isinstance(Neuralink(threads), CorticalImplant), False)
    npt.assert_equal(Neuralink(threads).hemisphere, None)
    npt.assert_equal(Neuralink(threads, hemisphere='Right').hemisphere,
                     'right')
    with pytest.raises(ValueError):
        Neuralink(threads, hemisphere='RH')
    with pytest.raises(TypeError):
        Neuralink(threads, hemisphere=1)
    # ... and it does not move the threads:
    npt.assert_array_equal(
        Neuralink(threads, hemisphere='left').electrode_array.coordinates(),
        Neuralink(threads).electrode_array.coordinates())
