import numpy as np
import numpy.testing as npt
import pytest
from scipy.spatial import cKDTree
from types import SimpleNamespace
from pulse2percept.models.cortex import ScoreboardModel
from pulse2percept.models import ScoreboardModel as BeyelerScoreboard
from pulse2percept.implants.cortex import Neuralink
from pulse2percept.implants import EnsembleImplant
from pulse2percept.topography import CorticalMap, NeuropythyMap
from pulse2percept.units import (DimensionMismatchError, dva, mm, um)
import time
import os

def load_fsaverage_or_skip():
    """Load the Neuropythy 'fsaverage' subject, or skip the calling test.

    Loading a subject may download the Benson & Winawer (2018) dataset, which
    can fail for any number of reasons outside our control (neuropythy not
    installed, no network, moved download endpoints, no disk space, ...). All
    of them should skip the test rather than error it out.
    """
    try:
        return NeuropythyMap('fsaverage')
    except Exception as err:
        pytest.skip(f"Could not load the Neuropythy 'fsaverage' subject "
                    f"({type(err).__name__}: {err}). Download the Benson & "
                    f"Winawer 2018 dataset to run this test.")


@pytest.fixture(scope='session')
def neuropythy_available():
    """Skip the test unless the 'fsaverage' subject can be loaded.

    This is a fixture rather than a ``skipif`` condition on purpose:
    conditions are evaluated at collection time, so a ``skipif`` would hit the
    network on every test run, including runs where these tests are all
    skipped anyway.
    """
    return load_fsaverage_or_skip()


def test_load_fsaverage_or_skip_swallows_any_error():
    """Any failure to load the subject must skip, not error out.

    This runs without neuropythy installed and without touching the network,
    so it guards the fixture that gates every other test in this module.
    """
    import pulse2percept.topography.tests.test_neuropythy as mod

    for err in (ImportError('no neuropythy'), ValueError('no such subject'),
                OSError('network is down'), RuntimeError('something else')):
        def boom(*args, _err=err, **kwargs):
            raise _err

        orig = mod.NeuropythyMap
        mod.NeuropythyMap = boom
        try:
            with pytest.raises(pytest.skip.Exception) as excinfo:
                load_fsaverage_or_skip()
            # The skip reason names the underlying error, so a future
            # breakage is diagnosable straight from the CI log:
            npt.assert_equal(type(err).__name__ in str(excinfo.value), True)
        finally:
            mod.NeuropythyMap = orig


class ToyNeuropythyMap(NeuropythyMap):
    """A NeuropythyMap whose cortical mesh is a row of ``n`` vertices.

    ``cortex_to_dva`` only reads the k-d tree and the mesh lookup tables, so
    filling those in with a toy mesh exercises all of its bookkeeping (output
    shape, NaN handling, exact hits) with hand-checkable numbers and without a
    subject download. Vertex ``i`` sits at ``(i, 0, 0)`` mm on the cortex and
    maps to ``(i, -i)`` dva; vertices are 1 mm apart, which is exactly
    ``cort_nn_thresh``.
    """

    def __init__(self, n=6):
        # NeuropythyMap.__init__ needs neuropythy and a subject, so go
        # straight to the parameter defaults it would have set:
        CorticalMap.__init__(self)
        self.cortex_tree = cKDTree(np.stack([np.arange(n, dtype=float),
                                             np.zeros(n), np.zeros(n)], axis=-1))
        self.addr_idxs = {'addr': np.arange(n),
                          'region': np.array(['v1'] * n),
                          'hemi': np.zeros(n, dtype=int)}
        mesh = SimpleNamespace(coordinates=np.stack([np.arange(n, dtype=float),
                                                     -np.arange(n, dtype=float)]))
        self.region_meshes = {'v1': (mesh, mesh)}


def test_cortex_to_dva_shape_and_nans():
    """Every input point must get its own output slot (Issue #774)."""
    nmap = ToyNeuropythyMap()
    # Vertex i is at i mm == i * 1000 um, and maps to (i, -i) dva:
    xc = np.array([0., 1000., 2000.])
    xdva, ydva = nmap.cortex_to_dva(xc, np.zeros(3), np.zeros(3))
    npt.assert_almost_equal(xdva, [0, 1, 2])
    npt.assert_almost_equal(ydva, [0, -1, -2])

    # A NaN input must not shift the points after it into its slot:
    xc = np.array([0., np.nan, 2000.])
    xdva, ydva = nmap.cortex_to_dva(xc, np.zeros(3), np.zeros(3))
    npt.assert_equal(xdva.shape, (3,))
    npt.assert_almost_equal(xdva, [0, np.nan, 2])
    npt.assert_almost_equal(ydva, [0, np.nan, -2])
    # A NaN in any one of the three coordinates is enough:
    for coords in ([np.zeros(2), np.array([np.nan, 0.]), np.zeros(2)],
                   [np.zeros(2), np.zeros(2), np.array([np.nan, 0.])]):
        xdva, ydva = nmap.cortex_to_dva(*coords)
        npt.assert_almost_equal(xdva, [np.nan, 0])
        npt.assert_almost_equal(ydva, [np.nan, 0])

    # The output has the shape of the input, whatever that shape is:
    for shape in [(), (1,), (4,), (2, 3), (2, 3, 4)]:
        zeros = np.zeros(shape)
        xdva, ydva = nmap.cortex_to_dva(zeros, zeros, zeros)
        npt.assert_equal(xdva.shape, shape)
        npt.assert_equal(ydva.shape, shape)
        # ... including when every point is NaN, which must still return two
        # arrays rather than a single stacked one:
        nans = np.full(shape, np.nan)
        xdva, ydva = nmap.cortex_to_dva(nans, nans, nans)
        npt.assert_equal(xdva.shape, shape)
        npt.assert_equal(np.all(np.isnan(xdva)), True)
        npt.assert_equal(np.all(np.isnan(ydva)), True)

    with pytest.raises(ValueError):
        nmap.cortex_to_dva(np.zeros(3), np.zeros(2), np.zeros(3))


def test_cortex_to_dva_exact_vertex():
    """A point sitting exactly on a mesh vertex must not divide by zero."""
    nmap = ToyNeuropythyMap()
    verts = nmap.cortex_tree.data * 1000  # mm -> um
    with np.errstate(divide='raise', invalid='raise'):
        xdva, ydva = nmap.cortex_to_dva(verts[:, 0], verts[:, 1], verts[:, 2])
    # Each vertex maps to its own dva coordinate, not to a blend with its
    # neighbors and not to NaN:
    npt.assert_almost_equal(xdva, np.arange(len(verts)))
    npt.assert_almost_equal(ydva, -np.arange(len(verts)))

    # Halfway between two vertices, both weigh the same:
    xdva, ydva = nmap.cortex_to_dva(np.array([2500.]), np.zeros(1), np.zeros(1))
    npt.assert_almost_equal(xdva, [2.5])
    npt.assert_almost_equal(ydva, [-2.5])

    # Beyond cort_nn_thresh of every vertex, there is nothing to average:
    xdva, ydva = nmap.cortex_to_dva(np.array([-2000.]), np.zeros(1), np.zeros(1))
    npt.assert_almost_equal(xdva, [np.nan])
    npt.assert_almost_equal(ydva, [np.nan])


# use pytest.mark.slow because all neuropythy tests
# take a long time to run. This way, they will be skipped
# unless the user passes --runslow to pytest (which must be)
# done either from the root p2p directory or from this tests
# folder.
@pytest.mark.slow
def test_subject_parsing(neuropythy_available):
    import neuropythy as ny
    # random subject shouldn't download 
    start = time.time()
    with pytest.raises(ValueError):
        nmap = NeuropythyMap('invalid_subject')
    npt.assert_equal(time.time() - start < 10, True)

    # test non fsaverage subject first, to see if it downloads
    # (since this is non default behaviour for neuropythy)
    # this test will also pass if the subject has been previously downloaded
    nmap = NeuropythyMap('S1201')
    # smoke test
    nmap.dva_to_v1(1, 1)

    # should have been cached to cache_dir
    npt.assert_equal(os.path.exists(os.path.join(nmap.cache_dir, 'benson_winawer_2018', 'freesurfer_subjects')), True)
    npt.assert_equal(os.path.join(nmap.cache_dir, 'benson_winawer_2018', 'freesurfer_subjects') in ny.config['freesurfer_subject_paths'], True)
    npt.assert_equal(ny.config['benson_winawer_2018_path'], os.path.join(nmap.cache_dir, 'benson_winawer_2018'))

    # now any other subject should be loaded quickly (<40 sec)
    start = time.time()
    nmap = NeuropythyMap('fsaverage')
    npt.assert_equal(time.time() - start < 40, True)

    npt.assert_equal(nmap.predicted_retinotopy is not None, True)
    npt.assert_equal('v1' in nmap.region_meshes.keys(), True)


# these take long so dont do every combo
@pytest.mark.slow()
@pytest.mark.parametrize('regions', [['v1'], ['v1', 'v3'], ['v1', 'v2', 'v3']])
@pytest.mark.parametrize('jitter_boundary', [True, False])
def test_dva_to_cortex(regions, jitter_boundary, neuropythy_available):
    nmap = NeuropythyMap('fsaverage', regions=regions, jitter_boundary=jitter_boundary)
    npt.assert_equal(nmap.predicted_retinotopy is not None, True)
    npt.assert_equal(nmap.region_meshes is not None, True)
    if 'v1' in regions:
        npt.assert_equal(nmap.region_meshes['v1'] is not None, True)
    if 'v2' in regions:
        npt.assert_equal(nmap.region_meshes['v2'] is not None, True)
    if 'v3' in regions:
        npt.assert_equal(nmap.region_meshes['v3'] is not None, True)

    
    npt.assert_equal(list(nmap.region_meshes.keys()), regions)
    if 'v2' not in regions:
        with pytest.raises(ValueError):
            nmap.dva_to_v2(0, 0)

    if 'v3' not in regions:
        with pytest.raises(ValueError):
            nmap.dva_to_v3(0, 0)
    
    if 'v1' in regions:
        # smoke test
        nmap.dva_to_v1(0, 0)
        for surface in ['white', 'pial']:
            nmap.dva_to_v1(0, 0, surface=surface)
        
        x, y, z = nmap.dva_to_v1([1, 1, 0, 0, -1, -1], [1, -1, 1, -1, 1, -1])
        npt.assert_equal(x.shape, (6,))
        npt.assert_equal(y.shape, (6,))
        npt.assert_equal(z.shape, (6,))
        if jitter_boundary:
            npt.assert_almost_equal(x, np.array([-10035.355, -13315.073, -11266.07, -16252.549, 12075.739, 13630.971]), decimal=3)
            npt.assert_almost_equal(y, np.array([ -96637.12, -102852.29,  -96669.43, -102938.95,  -95358.4,  -101546.41]), decimal=2)
            npt.assert_almost_equal(z, np.array([-10769.129, -3861.491, -12831.113, -1908.735, -7168.826, 924.938]), decimal=3)
        else:
            npt.assert_almost_equal(x, np.array([-10035.355, -13315.073, np.nan, np.nan, 12075.739, 13630.971]), decimal=3)
            npt.assert_almost_equal(y, np.array([ -96637.12, -102852.29,  np.nan, np.nan,  -95358.4,  -101546.41]), decimal=2)
            npt.assert_almost_equal(z, np.array([-10769.129, -3861.491, np.nan, np.nan, -7168.826, 924.938]), decimal=3)

    if 'v2' in regions:
        # smoke test
        nmap.dva_to_v2(0, 0)
        for surface in ['white', 'pial']:
            nmap.dva_to_v2(0, 0, surface=surface)
        
        x, y, z = nmap.dva_to_v2([1, 1, 0, 0, -1, -1], [1, -1, 1, -1, 0, -1])
        npt.assert_equal(x.shape, (6,))
        npt.assert_equal(y.shape, (6,))
        npt.assert_equal(z.shape, (6,))
        if jitter_boundary:
            npt.assert_almost_equal(x, np.array([-11731.504, -20458.03, np.nan ,-18807.701, 26066.922,  22283.799] ), decimal=3)
            npt.assert_almost_equal(y, np.array([ -93461.92,  -100803.35, np.nan, -101528.13,   -96025.48,   -99334.945]), decimal=2)
            npt.assert_almost_equal(z, np.array([-11246.644,   1673.845, np.nan,   -313.502,   4501.598,   7011.859]), decimal=3)
        else:
            npt.assert_almost_equal(x, np.array([-11731.504, -20458.03, np.nan ,np.nan, np.nan,  22283.799] ), decimal=3)
            npt.assert_almost_equal(y, np.array([ -93461.92,  -100803.35, np.nan, np.nan,   np.nan,   -99334.945]), decimal=2)
            npt.assert_almost_equal(z, np.array([-11246.644,   1673.845, np.nan,   np.nan,   np.nan,   7011.859]), decimal=3)

    
    if 'v3' in regions:
        # smoke test
        nmap.dva_to_v3(0, 0)
        for surface in ['white', 'pial']:
            nmap.dva_to_v3(0, 0, surface=surface)
        
        x, y, z = nmap.dva_to_v3([1, 1, 0, 0, -1, -1], [1, -1, 1, -1, 0, -1])
        npt.assert_equal(x.shape, (6,))
        npt.assert_equal(y.shape, (6,))
        npt.assert_equal(z.shape, (6,))
        if jitter_boundary:
            npt.assert_almost_equal(x, np.array([-23812.113, -23514.828, -29542.21,  -25206.152,  27090.357,  28547.275]), decimal=3)
            npt.assert_almost_equal(y, np.array([-84409.51, -93015.07, -83442.17, -89647.35, -94726.14, -93238.63]), decimal=2)
            npt.assert_almost_equal(z, np.array([-15261.302,   4050.124, -16078.909,   3062.166,   4468.217,   8467.487]), decimal=3)
        else:
            npt.assert_almost_equal(x, np.array([-23812.113, -23514.828, np.nan,  np.nan,  np.nan,  28547.275]), decimal=3)
            npt.assert_almost_equal(y, np.array([-84409.51, -93015.07, np.nan, np.nan, np.nan, -93238.63]), decimal=2)
            npt.assert_almost_equal(z, np.array([-15261.302,   4050.124, np.nan,   np.nan,   np.nan,   8467.487]), decimal=3)


@pytest.mark.slow
def test_Neuralink_from_neuropythy(neuropythy_available):
    nmap = NeuropythyMap('fsaverage', regions=['v1'], jitter_boundary=False)
    nlink = Neuralink.from_neuropythy(nmap, locs=np.array([[0, 0], [3, 3], [-2, -2]]))
    # 0, 0 should be nan so it wont make one
    npt.assert_almost_equal(len(nlink.implants), 2)
    npt.assert_almost_equal(nlink.implants['A'].x, nmap.dva_to_v1(3, 3, surface='pial')[0])
    npt.assert_almost_equal(nlink.implants['A'].y, nmap.dva_to_v1(3, 3, surface='pial')[1])
    npt.assert_almost_equal(nlink.implants['A'].z, nmap.dva_to_v1(3, 3, surface='pial')[2])
    npt.assert_almost_equal(nlink.implants['B'].x, nmap.dva_to_v1(-2, -2, surface='pial')[0])
    npt.assert_almost_equal(nlink.implants['B'].y, nmap.dva_to_v1(-2, -2, surface='pial')[1])
    npt.assert_almost_equal(nlink.implants['B'].z, nmap.dva_to_v1(-2, -2, surface='pial')[2])

    orient1 = np.array(nmap.dva_to_v1(3, 3, surface='midgray')) - np.array(nmap.dva_to_v1(3, 3, surface='pial'))
    orient2 = np.array(nmap.dva_to_v1(-2, -2, surface='midgray')) - np.array(nmap.dva_to_v1(-2, -2, surface='pial'))
    orient1 = orient1 / np.linalg.norm(orient1)
    orient2 = orient2 / np.linalg.norm(orient2)
    npt.assert_almost_equal(nlink.implants['A'].direction, orient1, decimal=4)
    npt.assert_almost_equal(nlink.implants['B'].direction, orient2, decimal=4)

    nmap.jitter_boundary=True
    nlink = Neuralink.from_neuropythy(nmap, xrange=[-5, 5], yrange=(-3, 3), step=1)
    npt.assert_equal(len(nlink.implants), 77)
    # thank god for chatgpt
    npt.assert_equal(list(nlink.implants.keys()), ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                                                   'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                                                   'U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD',
                                                   'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM',
                                                   'AN', 'AO', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV',
                                                   'AW', 'AX', 'AY', 'AZ', 'BA', 'BB', 'BC', 'BD', 'BE',
                                                   'BF', 'BG', 'BH', 'BI', 'BJ', 'BK', 'BL', 'BM', 'BN',
                                                   'BO', 'BP', 'BQ', 'BR', 'BS', 'BT', 'BU', 'BV', 'BW',
                                                   'BX', 'BY'])
    idx = 0
    for vy in range(3, -4, -1):
        for vx in range(-5, 6, 1):
            # print(idx, vx, vy)
            implant = nlink.implants[list(nlink.implants.keys())[idx]]
            cx, cy, cz = nmap.dva_to_v1(vx, vy, surface='pial')
            npt.assert_almost_equal(implant.x, cx)
            npt.assert_almost_equal(implant.y, cy)
            npt.assert_almost_equal(implant.z, cz)

            orient = np.array(nmap.dva_to_v1(vx, vy, surface='midgray')) - np.array(nmap.dva_to_v1(vx, vy, surface='pial'))
            orient = orient / np.linalg.norm(orient)
            npt.assert_almost_equal(implant.direction, orient, decimal=3)

            idx += 1
            

@pytest.mark.slow
def test_ndim_mixup(neuropythy_available):
    nmap = NeuropythyMap('fsaverage')
    model = BeyelerScoreboard(vfmap=nmap)
    npt.assert_equal(2 in model.ndim, True)
    npt.assert_equal(3 in model.ndim, False)
    with pytest.raises(ValueError):
        model.build()


@pytest.mark.slow
def test_neuropythy_scoreboard(neuropythy_available):
    nmap = NeuropythyMap('fsaverage')
    model = ScoreboardModel(rho=800, step=.25, vfmap=nmap).build()
    implant = Neuralink.from_neuropythy(nmap, xrange=(-3, 3), yrange=(-3, 3))
    implant.stim = {e : 1 for e in implant.electrode_names}
    percept = model.predict_percept(implant)
    npt.assert_almost_equal(np.sum(percept.data), 5600.183, decimal=3)
    npt.assert_almost_equal(np.max(percept.data), 27.3698, decimal=3)

    nmap = NeuropythyMap('fsaverage', regions=['v2'])
    model = ScoreboardModel(rho=800, step=.25, vfmap=nmap).build()
    implant = Neuralink.from_neuropythy(nmap, xrange=(-3, 3), yrange=(-3, 3), region='v2')
    implant.stim = {e : 1 for e in implant.electrode_names}
    percept = model.predict_percept(implant)
    npt.assert_almost_equal(np.sum(percept.data), 5344.173, decimal=2)
    npt.assert_almost_equal(np.max(percept.data), 27.845, decimal=2)

    # mega implant
    nmap = NeuropythyMap('fsaverage', regions=['v1', 'v2', 'v3'])
    model = ScoreboardModel(rho=800, step=.25, vfmap=nmap).build()
    i1 = Neuralink.from_neuropythy(nmap, xrange=(-3, 3), yrange=(-3, 3), region='v1')
    i2 = Neuralink.from_neuropythy(nmap, xrange=(-3, 3), yrange=(-3, 3), region='v2')
    i3 = Neuralink.from_neuropythy(nmap, xrange=(-3, 3), yrange=(-3, 3), region='v3')
    implant = EnsembleImplant([i1, i2, i3])
    implant.stim = {e : 1 for e in implant.electrode_names}
    percept = model.predict_percept(implant)
    npt.assert_almost_equal(np.sum(percept.data), 20245.445, decimal=1)
    npt.assert_almost_equal(np.max(percept.data), 86.4913, decimal=1)


@pytest.mark.slow()
@pytest.mark.parametrize('regions', [['v1'], ['v1', 'v3'], ['v1', 'v2', 'v3']])
def test_cortex_to_dva(regions, neuropythy_available):
    nmap = NeuropythyMap('fsaverage', regions=regions, jitter_boundary=True)
    npt.assert_equal(nmap.predicted_retinotopy is not None, True)
    npt.assert_equal(nmap.region_meshes is not None, True)
    if 'v1' in regions:
        npt.assert_equal(nmap.region_meshes['v1'] is not None, True)
    if 'v2' in regions:
        npt.assert_equal(nmap.region_meshes['v2'] is not None, True)
    if 'v3' in regions:
        npt.assert_equal(nmap.region_meshes['v3'] is not None, True)

    
    npt.assert_equal(list(nmap.region_meshes.keys()), regions)
    
    if 'v1' in regions:
        # should work with all shapes, and keep them
        npt.assert_equal(np.isnan(nmap.v1_to_dva(0, 0, 0)[0]), True)
        npt.assert_equal(nmap.v1_to_dva([100, 200, 300], [100, 200, 300],
                                        [100, 200, 300])[0].shape, (3,))
        npt.assert_equal(nmap.v1_to_dva(np.eye(3), np.eye(3), np.eye(3))[0].shape,
                         (3, 3))

        x = np.array([-10035.355, -13315.073,  12075.739, 13630.971])
        y = np.array([ -96637.12, -102852.29,   -95358.4,  -101546.41])
        z = np.array([-10769.129, -3861.491, -7168.826, 924.938])


        xdva, ydva = nmap.v1_to_dva(x, y, z)
        npt.assert_equal(x.shape, (4,))
        npt.assert_equal(y.shape, (4,))
        npt.assert_almost_equal(xdva, np.array([1, 1, -1, -1]), decimal=1)
        npt.assert_almost_equal(ydva, np.array([1, -1,  1, -1]), decimal=1)

        # A NaN point keeps its own slot rather than dropping out and
        # shifting the points after it up (Issue #774):
        xnan, ynan, znan = x.copy(), y.copy(), z.copy()
        xnan[1], ynan[1], znan[1] = np.nan, np.nan, np.nan
        xdva, ydva = nmap.v1_to_dva(xnan, ynan, znan)
        npt.assert_equal(xdva.shape, (4,))
        npt.assert_almost_equal(xdva, np.array([1, np.nan, -1, -1]), decimal=1)
        npt.assert_almost_equal(ydva, np.array([1, np.nan, 1, -1]), decimal=1)
        # ... and the same points laid out as a 2D grid come back as one:
        xdva, ydva = nmap.v1_to_dva(*[np.stack([c, c]) for c in (x, y, z)])
        npt.assert_equal(xdva.shape, (2, 4))
        npt.assert_almost_equal(xdva, np.stack([[1, 1, -1, -1]] * 2), decimal=1)
        npt.assert_almost_equal(ydva, np.stack([[1, -1, 1, -1]] * 2), decimal=1)

        # Points that land exactly on a mesh vertex have zero distance to it,
        # which used to divide by zero (Issue #774). They map to that vertex:
        verts = nmap.cortex_tree.data * 1000  # mm -> um
        xdva, ydva = nmap.v1_to_dva(verts[:, 0], verts[:, 1], verts[:, 2])
        npt.assert_equal(np.any(np.isnan(xdva)), False)
        npt.assert_equal(np.any(np.isnan(ydva)), False)

        x = np.arange(-10, -1, .1)
        y = np.arange(-10, -1, .1)
        x1, y2 = nmap.v1_to_dva(*nmap.dva_to_v1(x, y))
        npt.assert_allclose(x, x1, rtol=.05, atol=0.1)
        npt.assert_allclose(y, y2, rtol=.05, atol=0.1)


        # test cort_nn_thresh
        idx = np.argmax(nmap.subject.hemis['rh'].surface('midgray').coordinates[0])
        x = np.array([nmap.subject.hemis['rh'].surface('midgray').coordinates[0][idx]])
        y = np.array([nmap.subject.hemis['rh'].surface('midgray').coordinates[1][idx]])
        z = np.array([nmap.subject.hemis['rh'].surface('midgray').coordinates[2][idx]])
        xdva, ydva = nmap.v1_to_dva(x, y, z)
        npt.assert_equal(xdva != np.array([np.nan]), True)
        npt.assert_equal(ydva != np.array([np.nan]), True)
        x1 = x + 999
        xdva, ydva = nmap.v1_to_dva(x1, y, z)
        npt.assert_equal(xdva != np.array([np.nan]), True)
        npt.assert_equal(ydva != np.array([np.nan]), True)
        x1 = x +1001
        xdva, ydva = nmap.v1_to_dva(x1, y, z)
        npt.assert_equal(xdva, np.array([np.nan]))
        npt.assert_equal(ydva, np.array([np.nan]))




    if 'v2' in regions:
        npt.assert_equal(np.isnan(nmap.v2_to_dva(0, 0, 0)[0]), True)
        npt.assert_equal(nmap.v2_to_dva([100, 200, 300], [100, 200, 300],
                                        [100, 200, 300])[0].shape, (3,))
        npt.assert_equal(nmap.v2_to_dva(np.eye(3), np.eye(3), np.eye(3))[0].shape,
                         (3, 3))


        x = np.array([-11731.504, -20458.03,  22283.799] )
        y = np.array([ -93461.92,  -100803.35, -99334.945])
        z = np.array([-11246.644,   1673.845,    7011.859])
        
        xdva, ydva = nmap.v2_to_dva(x, y, z)
        npt.assert_equal(xdva.shape, (3,))
        npt.assert_equal(ydva.shape, (3,))
        npt.assert_allclose(xdva, np.array([1, 1, -1]), rtol=.05, atol=0.1)
        npt.assert_allclose(ydva, np.array([1, -1, -1]), rtol=.05, atol=0.1)

        x = np.arange(-10, -1, .1)
        y = np.arange(-10, -1, .1)
        x1, y2 = nmap.v2_to_dva(*nmap.dva_to_v2(x, y))
        npt.assert_allclose(x, x1, rtol=.05, atol=0.1)
        npt.assert_allclose(y, y2, rtol=.05, atol=0.1)

    
    if 'v3' in regions:
        npt.assert_equal(np.isnan(nmap.v3_to_dva(0, 0, 0)[0]), True)
        npt.assert_equal(nmap.v3_to_dva([100, 200, 300], [100, 200, 300],
                                        [100, 200, 300])[0].shape, (3,))
        npt.assert_equal(nmap.v3_to_dva(np.eye(3), np.eye(3), np.eye(3))[0].shape,
                         (3, 3))


        x = np.array([-23812.113, -23514.828,  28547.275])
        y = np.array([-84409.51, -93015.07,  -93238.63])
        z = np.array([-15261.302,   4050.124,    8467.487])

        xdva, ydva = nmap.v3_to_dva(x, y, z)
        
        npt.assert_equal(xdva.shape, (3,))
        npt.assert_equal(ydva.shape, (3,))
        npt.assert_allclose(xdva, np.array([1, 1, -1]), rtol=.05, atol=0.1)
        npt.assert_allclose(ydva, np.array([1, -1, -1]), rtol=.05, atol=0.1)

        x = np.arange(-10, -1, .1)
        y = np.arange(-10, -1, .1)
        x1, y2 = nmap.v3_to_dva(*nmap.dva_to_v3(x, y))
        npt.assert_allclose(x, x1, rtol=.05, atol=0.1)
        npt.assert_allclose(y, y2, rtol=.05, atol=0.1)


def test_NeuropythyMap_units():
    """The FreeSurfer map converts between the same two sides as any other"""
    vfmap = load_fsaverage_or_skip()
    npt.assert_equal(vfmap.visual_unit, dva)
    npt.assert_equal(vfmap.tissue_unit, um)
    x, y = np.array([1.0, 3.0]), np.array([1.0, -2.0])
    bare = vfmap.dva_to_v1(x, y)
    npt.assert_allclose(vfmap.dva_to_v1(x * dva, y * dva), bare, rtol=1e-12)
    # `surface=` is not a coordinate and travels through untouched:
    npt.assert_allclose(vfmap.dva_to_v1(x * dva, y * dva, surface='pial'),
                        vfmap.dva_to_v1(x, y, surface='pial'), rtol=1e-12)
    # Back again, with the three coordinates spelled differently:
    xc, yc, zc = bare
    npt.assert_allclose(
        vfmap.v1_to_dva((xc / 1000) * mm, yc * um, (zc / 1000) * mm),
        vfmap.v1_to_dva(xc, yc, zc), rtol=1e-6)
    with pytest.raises(DimensionMismatchError):
        vfmap.dva_to_v1(x * um, y)
    with pytest.raises(DimensionMismatchError):
        vfmap.v1_to_dva(xc * dva, yc, zc)
