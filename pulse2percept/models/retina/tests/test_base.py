import numpy.testing as npt
import pytest

from pulse2percept.implants import ElectrodeGrid, Implant
from pulse2percept.implants.retina import RetinalImplant
from pulse2percept.models.retina import ScoreboardSpatial
from pulse2percept.topography.retina import (Curcio1990Map,
                                             Montesano2020Map,
                                             Watson2014Map)


def implant(eye='right', generic=False):
    array = ElectrodeGrid((2, 2), 400)
    return Implant(array) if generic else RetinalImplant(array, eye=eye)


def model(implant, **params):
    # Tiny grid: these tests are about laterality, not about the percept.
    return ScoreboardSpatial(implant, xrange=(-1, 1), yrange=(-1, 1), step=1,
                             verbose=False, **params)


@pytest.mark.parametrize('eye', ('left', 'right'))
def test_RetinalSpatial_matching_map(eye):
    spatial = model(implant(eye=eye),
                    visual_field_map=Montesano2020Map(eye=eye))
    spatial.build()
    npt.assert_equal(spatial.is_built, True)


def test_RetinalSpatial_map_eye_mismatch():
    spatial = model(implant(eye='left'),
                    visual_field_map=Montesano2020Map(eye='right'))
    with pytest.raises(ValueError):
        spatial.build()


def test_RetinalSpatial_map_eye_missing():
    spatial = model(implant(generic=True),
                    visual_field_map=Montesano2020Map(eye='right'))
    with pytest.raises(TypeError):
        spatial.build()


@pytest.mark.parametrize('eye', ('left', 'right'))
@pytest.mark.parametrize('vfmap', (Curcio1990Map, Watson2014Map))
def test_RetinalSpatial_eye_agnostic_map(eye, vfmap):
    spatial = model(implant(eye=eye), visual_field_map=vfmap())
    spatial.build()
    npt.assert_equal(spatial.is_built, True)
    # An eye-agnostic grid does not depend on laterality:
    spatial.implant.eye = 'left' if eye == 'right' else 'right'
    npt.assert_equal(spatial.is_built, True)


def test_RetinalSpatial_eye_mutation_after_build():
    # Mutating `implant.eye` or the bound map's `eye` bypasses parameter
    # assignment, so `is_built` has to catch it.
    for mutate in ['implant', 'map', 'both']:
        spatial = model(implant(eye='right'),
                        visual_field_map=Montesano2020Map(eye='right'))
        spatial.build()
        npt.assert_equal(spatial.is_built, True)
        if mutate in ('implant', 'both'):
            spatial.implant.eye = 'left'
        if mutate in ('map', 'both'):
            spatial.visual_field_map.eye = 'left'
        npt.assert_equal(spatial.is_built, False)
    # Rebuilding the now-matching left/left configuration works:
    spatial.build()
    npt.assert_equal(spatial.is_built, True)
