import numpy as np
import numpy.testing as npt
import pytest
import matplotlib.pyplot as plt
from pulse2percept.topography import NeuropythyMap
import time
import os

# use pytest.mark.slow because all neuropythy tests
# take a long time to run. This way, they will be skipped
# unless the user passes --runslow to pytest (which must be)
# done either from the root p2p directory or from this tests
# folder.
@pytest.mark.slow
def test_slow_test():
    print("This should not run")
    npt.assert_equal(True, False)

@pytest.mark.slow
def test_subject_parsing():
    import neuropythy as ny
    # random subject shouldn't download 
    start = time.time()
    with pytest.raises(ValueError):
        nmap = NeuropythyMap('invalid_subject')
    npt.assert_equal(time.time() - start < 20, True)

    # test non fsaverage subject first, to see if it downloads
    # (since this is non default behaviour for neuropythy)
    # this test will also pass if the subject has been previously downloaded
    nmap = NeuropythyMap('S1201')

    # should have been cached to cache_dir
    npt.assert_equal(os.path.exists(os.path.join(nmap.cache_dir, 'benson_winawer_2018', 'freesurfer_subjects')), True)
    npt.assert_equal(os.path.join(nmap.cache_dir, 'benson_winawer_2018', 'freesurfer_subjects') in ny.config['freesurfer_subject_paths'], True)
    npt.assert_equal(ny.config['benson_winawer_2018_path'], os.path.join(nmap.cache_dir, 'benson_winawer_2018'))

    # now any other subject should be loaded quickly (<40 sec)
    start = time.time()
    nmap = NeuropythyMap('fsaverage')
    npt.assert_equal(time.time() - start < 40, True)

    npt.assert_equal(nmap.predicted_retinotopy is not None, True)
    npt.assert_equal('v1' in nmap.region_meshes.keys())


# these take long so dont do every combo
@pytest.mark.slow()
@pytest.mark.parametrize('regions' : [['v1'], ['v1', 'v3'], ['v1', 'v2', 'v3']])
@pytest.mark.parametrize('jitter_boundary', [True, False])
def test_dva_to_cortex(regions, jitter_boundary):
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