import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.topography.cortex import Polimeni2006Map
from pulse2percept.units import DimensionMismatchError, Quantity, dva, mm, um


def test_cortical_map_units():
    """dva in, microns out, and a round trip that mixes the two spellings"""
    visual_field_map = Polimeni2006Map(regions=['v1', 'v2', 'v3'])
    xdva, ydva = np.array([5.0, 2.0]), np.array([-2.0, 3.0])
    for region in ('v1', 'v2', 'v3'):
        to_tissue = getattr(visual_field_map, f'dva_to_{region}')
        to_visual = getattr(visual_field_map, f'{region}_to_dva')
        bare = to_tissue(xdva, ydva)
        npt.assert_allclose(to_tissue(xdva * dva, ydva * dva), bare,
                            rtol=1e-12, err_msg=region)
        x_um, y_um = bare
        # The round trip the units exist for: microns back to degrees, with
        # the two coordinates spelled differently from each other.
        back_bare = to_visual(x_um, y_um)
        back_mixed = to_visual((x_um / 1000) * mm, y_um * um)
        npt.assert_allclose(back_mixed, back_bare, rtol=1e-6, err_msg=region)
        npt.assert_allclose(back_bare, [xdva, ydva], rtol=1e-4)
        # Plain arrays out, never quantities:
        for value in back_mixed:
            npt.assert_equal(isinstance(value, Quantity), False)
        with pytest.raises(DimensionMismatchError):
            to_tissue(xdva * um, ydva)
        with pytest.raises(DimensionMismatchError):
            to_visual(x_um * dva, y_um)
