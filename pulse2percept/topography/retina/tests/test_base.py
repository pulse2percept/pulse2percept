import numpy.testing as npt
import pytest

from pulse2percept.topography.retina import (Curcio1990Map, Watson2014Map,
                                             Watson2014DisplaceMap)
from pulse2percept.units import (DimensionMismatchError, Quantity, dva, mm,
                                 ms, um)


def test_retinal_map_units():
    """dva in, microns out -- and either may be spelled with a unit"""
    for cls in (Curcio1990Map, Watson2014Map, Watson2014DisplaceMap):
        visual_field_map = cls()
        bare = visual_field_map.dva_to_ret(5, -2)
        npt.assert_allclose(visual_field_map.dva_to_ret(5 * dva, -2 * dva),
                            bare,
                            rtol=1e-12, err_msg=cls.__name__)
        # The output is a plain number of microns, never a Quantity:
        for value in bare:
            npt.assert_equal(isinstance(value, Quantity), False)
        with pytest.raises(DimensionMismatchError):
            visual_field_map.dva_to_ret(5 * um, -2)
        with pytest.raises(DimensionMismatchError):
            visual_field_map.dva_to_ret(5, -2 * ms)

    # The inverse takes a length, and mixed spellings round-trip:
    for cls in (Curcio1990Map, Watson2014Map):
        visual_field_map = cls()
        x_um, y_um = visual_field_map.dva_to_ret(5 * dva, -2 * dva)
        bare = visual_field_map.ret_to_dva(x_um, y_um)
        mixed = visual_field_map.ret_to_dva((x_um / 1000) * mm, y_um * um)
        npt.assert_allclose(mixed, bare, rtol=1e-9, err_msg=cls.__name__)
        with pytest.raises(DimensionMismatchError):
            visual_field_map.ret_to_dva(5 * dva, -2)
    # Curcio is exactly linear, so its round trip closes exactly. (Watson's
    # forward and inverse are separate fits and only agree to ~2%, which is a
    # property of that map and not of the unit conversion.)
    npt.assert_allclose(
        Curcio1990Map().ret_to_dva(*Curcio1990Map().dva_to_ret(5 * dva,
                                                               -2 * dva)),
        [5, -2], rtol=1e-12)

    # A non-coordinate keyword travels through untouched:
    watson = Watson2014Map()
    npt.assert_allclose(watson.dva_to_ret(3 * dva, 1 * dva, coords='cart'),
                        watson.dva_to_ret(3, 1, coords='cart'), rtol=1e-12)
    npt.assert_allclose(watson.ret_to_dva(1000 * um, 500 * um, coords='cart'),
                        watson.ret_to_dva(1000, 500, coords='cart'),
                        rtol=1e-12)
