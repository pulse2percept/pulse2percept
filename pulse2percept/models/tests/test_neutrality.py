"""The generic spatial base carries no retinal assumptions

The generic spatial base supplies no ``visual_field_map`` and reads no
physical length as a visual-field extent. Both are properties of the retina,
and live on :py:class:`~pulse2percept.models.retina.RetinalSpatial` instead.
"""
import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.implants.retina import ArgusII
from pulse2percept.implants.cortex import Cortivis
from pulse2percept.models import SpatialModel
from pulse2percept.models.cortex import ScoreboardSpatial as CortexScoreboard
from pulse2percept.models.retina import (RetinalSpatial, ScoreboardSpatial,
                                         AxonMapSpatial, Nanduri2012Spatial,
                                         Thompson2003Spatial)
from pulse2percept.topography.retina import (Curcio1990Map, RetinalMap,
                                             Watson2014Map)
from pulse2percept.topography.cortex import Polimeni2006Map
from pulse2percept.units import DimensionMismatchError, mm, um


class BareSpatial(SpatialModel):
    """A concrete spatial model that declares nothing anatomical"""

    def _predict_spatial(self, electrode_array, stim):
        n_time = 1 if stim.time is None else stim.time.size
        return np.zeros((self.grid.x.size, n_time), dtype=np.float32)


class BareRetinal(RetinalSpatial):
    """The same model, on the retinal base"""

    def _predict_spatial(self, electrode_array, stim):
        n_time = 1 if stim.time is None else stim.time.size
        return np.zeros((self.grid.x.size, n_time), dtype=np.float32)


def test_generic_spatial_model_has_no_map():
    model = BareSpatial(ArgusII())
    npt.assert_equal(model.visual_field_map, None)
    with pytest.raises(ValueError) as excinfo:
        model.build()
    npt.assert_equal('visual_field_map' in str(excinfo.value), True)
    # ... and says so before it would trip over a missing attribute:
    npt.assert_equal('NoneType' in str(excinfo.value), False)
    # Supplying one is all it takes:
    npt.assert_equal(BareSpatial(ArgusII(),
                                 visual_field_map=Curcio1990Map(),
                                 step=5).build().is_built, True)


def test_retinal_spatial_supplies_the_retinal_default():
    npt.assert_equal(isinstance(BareRetinal(ArgusII()).visual_field_map,
                                Curcio1990Map), True)


@pytest.mark.parametrize('ModelClass', [ScoreboardSpatial, AxonMapSpatial,
                                        Nanduri2012Spatial,
                                        Thompson2003Spatial])
def test_retinal_models_are_retinal_spatial(ModelClass):
    npt.assert_equal(issubclass(ModelClass, RetinalSpatial), True)
    # Each keeps whichever map it installed before the split:
    npt.assert_equal(isinstance(ModelClass(ArgusII()).visual_field_map,
                                RetinalMap), True)


def test_beyeler_models_keep_their_own_map():
    """[Beyeler2019]_ overrides the `RetinalSpatial` default with Watson"""
    for ModelClass in (ScoreboardSpatial, AxonMapSpatial):
        npt.assert_equal(isinstance(ModelClass(ArgusII()).visual_field_map,
                                    Watson2014Map), True)


def test_retinal_length_shorthand_is_retinal_only():
    # `RetinalSpatial` resolves a retinal extent through its own map:
    model = BareRetinal(ArgusII(), xrange=(-2 * mm, 2 * mm))
    lo, hi = model.xrange
    npt.assert_equal(lo < 0 < hi, True)
    expected = Curcio1990Map().ret_to_dva(2000.0, 0)[0]
    npt.assert_almost_equal(hi, expected)
    npt.assert_almost_equal(lo, -expected)
    # ... the generic base does not:
    with pytest.raises(DimensionMismatchError) as excinfo:
        BareSpatial(ArgusII(), xrange=(-2 * mm, 2 * mm))
    npt.assert_equal('dva' in str(excinfo.value), True)
    with pytest.raises(DimensionMismatchError):
        BareSpatial(ArgusII(), yrange=(-2000 * um, 2000 * um))


def test_cortical_model_refuses_a_retinal_extent():
    with pytest.raises(DimensionMismatchError) as excinfo:
        CortexScoreboard(Cortivis(), xrange=(-2 * mm, 2 * mm))
    npt.assert_equal('dva' in str(excinfo.value), True)
    with pytest.raises(DimensionMismatchError):
        CortexScoreboard(Cortivis(), yrange=(-2 * mm, 2 * mm))
    # The cortical default map is unaffected:
    npt.assert_equal(isinstance(CortexScoreboard(Cortivis()).visual_field_map,
                                Polimeni2006Map), True)


def test_retinal_shorthand_needs_a_retinal_map():
    """A retinal model handed a cortical map has no extent to resolve"""
    with pytest.raises(DimensionMismatchError):
        BareRetinal(ArgusII(), visual_field_map=Polimeni2006Map(),
                    xrange=(-2 * mm, 2 * mm), ndim=[2])


def test_step_never_takes_a_length():
    """Only the *extent* is shorthand; the grid is always sampled in dva"""
    with pytest.raises(DimensionMismatchError):
        BareRetinal(ArgusII(), step=100 * um)
    with pytest.raises(DimensionMismatchError):
        BareSpatial(ArgusII(), step=100 * um)
