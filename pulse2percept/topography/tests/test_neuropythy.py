import numpy as np
import numpy.testing as npt
import pytest
from pulse2percept.models.cortex import ScoreboardModel
from pulse2percept.models import ScoreboardModel as BeyelerScoreboard
from pulse2percept.topography import NeuropythyMap
from pulse2percept.implants.cortex import Neuralink, LinearEdgeThread
from pulse2percept.implants import EnsembleImplant
import time
import os

# use pytest.mark.slow because all neuropythy tests
# take a long time to run. This way, they will be skipped
# unless the user passes --runslow to pytest (which must be)
# done either from the root p2p directory or from this tests
# folder.
@pytest.mark.slow
def test_subject_parsing():
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


@pytest.mark.slow
def test_Neuralink_from_neuropythy():
    nmap = NeuropythyMap('fsaverage', regions=['v1'], jitter_boundary=False)
    nlink = Neuralink.from_neuropythy(nmap, locs=np.array([[0, 0], [3, 3], [-2, -2]]))
    # 0, 0 should be nan so it wont make one
    npt.assert_equal(len(nlink.implants), 2)
    npt.assert_equal(nlink.implants['A'].x, nmap.dva_to_v1(3, 3, surface='pial')[0])
    npt.assert_equal(nlink.implants['A'].y, nmap.dva_to_v1(3, 3, surface='pial')[1])
    npt.assert_equal(nlink.implants['A'].z, nmap.dva_to_v1(3, 3, surface='pial')[2])
    npt.assert_equal(nlink.implants['B'].x, nmap.dva_to_v1(-2, -2, surface='pial')[0])
    npt.assert_equal(nlink.implants['B'].y, nmap.dva_to_v1(-2, -2, surface='pial')[1])
    npt.assert_equal(nlink.implants['B'].z, nmap.dva_to_v1(-2, -2, surface='pial')[2])

    orient1 = np.array(nmap.dva_to_v1(3, 3, surface='midgray')) - np.array(nmap.dva_to_v1(3, 3, surface='pial'))
    orient2 = np.array(nmap.dva_to_v1(-2, -2, surface='midgray')) - np.array(nmap.dva_to_v1(-2, -2, surface='pial'))
    orient1 = orient1 / np.linalg.norm(orient1)
    orient2 = orient2 / np.linalg.norm(orient2)
    npt.assert_almost_equal(nlink.implants['A'].direction, orient1)
    npt.assert_almost_equal(nlink.implants['B'].direction, orient2)

    nmap.jitter_boundary=True
    nlink = Neuralink.from_neuropythy(nmap, xrange=[-5, 5], yrange=(-3, 3), xystep=1)
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
            print(idx, vx, vy)
            implant = nlink.implants[list(nlink.implants.keys())[idx]]
            cx, cy, cz = nmap.dva_to_v1(vx, vy, surface='pial')
            npt.assert_almost_equal(implant.x, cx)
            npt.assert_almost_equal(implant.y, cy)
            npt.assert_almost_equal(implant.z, cz)

            orient = np.array(nmap.dva_to_v1(vx, vy, surface='midgray')) - np.array(nmap.dva_to_v1(vx, vy, surface='pial'))
            orient = orient / np.linalg.norm(orient)
            npt.assert_almost_equal(implant.direction, orient)

            idx += 1
            


@pytest.mark.slow
def test_ndim_mixup():
    nmap = NeuropythyMap('fsaverage')
    model = BeyelerScoreboard(vfmap=nmap)
    npt.assert_equal(2 in model.ndim, True)
    npt.assert_equal(3 in model.ndim, False)
    with pytest.raises(ValueError):
        model.build()


@pytest.mark.slow
def test_neuropythy_scoreboard():
    nmap = NeuropythyMap('fsaverage')
    model = ScoreboardModel(rho=800, xystep=.25, vfmap=nmap).build()
    implant = Neuralink.from_neuropythy(nmap, xrange=(-3, 3), yrange=(-3, 3))
    implant.stim = {e : 1 for e in implant.electrode_names}
    percept = model.predict_percept(implant)
    npt.assert_almost_equal(np.sum(percept.data), 5600.183, decimal=3)
    npt.assert_almost_equal(np.max(percept.data), 27.3698, decimal=3)

    nmap = NeuropythyMap('fsaverage', regions=['v2'])
    model = ScoreboardModel(rho=800, xystep=.25, vfmap=nmap).build()
    implant = Neuralink.from_neuropythy(nmap, xrange=(-3, 3), yrange=(-3, 3), region='v2')
    implant.stim = {e : 1 for e in implant.electrode_names}
    percept = model.predict_percept(implant)
    npt.assert_almost_equal(np.sum(percept.data), 5344.173, decimal=3)
    npt.assert_almost_equal(np.max(percept.data), 27.845, decimal=3)

    # mega implant
    nmap = NeuropythyMap('fsaverage', regions=['v1', 'v2', 'v3'])
    model = ScoreboardModel(rho=800, xystep=.25, vfmap=nmap).build()
    i1 = Neuralink.from_neuropythy(nmap, xrange=(-3, 3), yrange=(-3, 3), region='v1')
    i2 = Neuralink.from_neuropythy(nmap, xrange=(-3, 3), yrange=(-3, 3), region='v2')
    i3 = Neuralink.from_neuropythy(nmap, xrange=(-3, 3), yrange=(-3, 3), region='v3')
    implant = EnsembleImplant([i1, i2, i3])
    implant.stim = {e : 1 for e in implant.electrode_names}
    percept = model.predict_percept(implant)
    npt.assert_almost_equal(np.sum(percept.data), 20245.45, decimal=3)
    npt.assert_almost_equal(np.max(percept.data), 86.4913, decimal=3)



@pytest.mark.slow()
@pytest.mark.parametrize('regions', [['v1'], ['v1', 'v3'], ['v1', 'v2', 'v3']])
def test_cortex_to_dva(regions):
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
        # should work with all shapes
        npt.assert_equal(nmap.v1_to_dva(0, 0, 0)[0], np.array([np.nan]))
        nmap.v1_to_dva([100, 200, 300], [100, 200, 300], [100, 200, 300])
        nmap.v1_to_dva(np.eye(3), np.eye(3), np.eye(3))       
        
        x = np.array([-10035.355, -13315.073,  12075.739, 13630.971])
        y = np.array([ -96637.12, -102852.29,   -95358.4,  -101546.41])
        z = np.array([-10769.129, -3861.491, -7168.826, 924.938])


        xdva, ydva = nmap.v1_to_dva(x, y, z)
        npt.assert_equal(x.shape, (4,))
        npt.assert_equal(y.shape, (4,))
        npt.assert_almost_equal(xdva, np.array([1, 1, -1, -1]), decimal=1)
        npt.assert_almost_equal(ydva, np.array([1, -1,  1, -1]), decimal=1)

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
        npt.assert_equal(nmap.v2_to_dva(0, 0, 0)[0], np.array([np.nan]))
        nmap.v2_to_dva([100, 200, 300], [100, 200, 300], [100, 200, 300])
        nmap.v2_to_dva(np.eye(3), np.eye(3), np.eye(3))     


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
        npt.assert_equal(nmap.v3_to_dva(0, 0, 0)[0], np.array([np.nan]))
        nmap.v3_to_dva([100, 200, 300], [100, 200, 300], [100, 200, 300])
        nmap.v3_to_dva(np.eye(3), np.eye(3), np.eye(3))


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



    