from types import SimpleNamespace
import warnings

import numpy as np
import numpy.testing as npt
import pytest
from scipy.spatial import cKDTree

from pulse2percept.topography import NeuropythyMap


def _make_neuropythy_map():
    nmap = NeuropythyMap.__new__(NeuropythyMap)
    cortex_points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
    ], dtype='float32')
    mesh = SimpleNamespace(coordinates=np.array([
        [10, 20, 30, 40, 50],
        [1, 2, 3, 4, 5],
    ], dtype='float32'))
    object.__setattr__(nmap, 'cortex_tree', cKDTree(cortex_points))
    object.__setattr__(nmap, 'cort_nn_thresh', 2000)
    object.__setattr__(nmap, 'region_meshes', {'v1': (mesh,)})
    object.__setattr__(nmap, 'addr_idxs', {
        'addr': np.arange(5),
        'region': np.array(['v1'] * 5),
        'hemi': np.zeros(5, dtype=int),
    })
    return nmap


@pytest.mark.parametrize('shape', [(3,), (2, 2)])
def test_cortex_to_dva_preserves_input_shape_and_nans(shape):
    nmap = _make_neuropythy_map()
    xc = np.full(shape, 100, dtype='float32')
    yc = np.full(shape, 100, dtype='float32')
    zc = np.full(shape, 100, dtype='float32')
    xc.flat[1] = np.nan
    yc.flat[-1] = np.nan
    id_nan = np.isnan(xc) | np.isnan(yc) | np.isnan(zc)

    xdva, ydva = nmap.cortex_to_dva(xc, yc, zc)

    assert xdva.shape == shape
    assert ydva.shape == shape
    npt.assert_array_equal(np.isnan(xdva), id_nan)
    npt.assert_array_equal(np.isnan(ydva), id_nan)
    assert np.all(np.isfinite(xdva[~id_nan]))
    assert np.all(np.isfinite(ydva[~id_nan]))


def test_cortex_to_dva_exact_vertex_is_finite_without_warnings():
    nmap = _make_neuropythy_map()

    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        xdva, ydva = nmap.cortex_to_dva(0, 0, 0)

    assert np.shape(xdva) == ()
    assert np.shape(ydva) == ()
    npt.assert_equal([xdva, ydva], [10, 1])


def test_cortex_to_dva_all_nan_returns_two_same_shape_arrays():
    nmap = _make_neuropythy_map()
    shape = (2, 3)
    nan_coords = np.full(shape, np.nan, dtype='float32')

    result = nmap.cortex_to_dva(nan_coords, nan_coords, nan_coords)

    assert isinstance(result, tuple)
    assert len(result) == 2
    for values in result:
        assert values.shape == shape
        assert np.all(np.isnan(values))
