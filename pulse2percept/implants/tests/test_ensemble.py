import numpy as np
import numpy.testing as npt
import pytest
from pulse2percept.implants import (EnsembleImplant, PointSource, ProsthesisSystem)
from pulse2percept.implants.cortex import Cortivis
from pulse2percept.topography import Polimeni2006Map
from pulse2percept.models.cortex.base import ScoreboardModel

def test_EnsembleImplant():
    # Invalid instantiations:
    with pytest.raises(TypeError):
        EnsembleImplant(implants="this can't happen")
    with pytest.raises(TypeError):
        EnsembleImplant(implants=[3,Cortivis()])
    with pytest.raises(TypeError):
        EnsembleImplant(implants={'1': Cortivis(), '2': 'abcd'})

    # Instantiate with list
    p1 = ProsthesisSystem(PointSource(0,0,0))
    p2 = ProsthesisSystem(PointSource(1,1,1))
    ensemble = EnsembleImplant(implants=[p1,p2])
    npt.assert_equal(ensemble.n_electrodes, 2)
    npt.assert_equal(ensemble[0], p1[0])
    npt.assert_equal(ensemble[1], p2[0])
    npt.assert_equal(ensemble.electrode_names, ['0-0','1-0'])

    # Instantiate with dict
    ensemble = EnsembleImplant(implants={'A': p2, 'B': p1})
    npt.assert_equal(ensemble.n_electrodes, 2)
    npt.assert_equal(ensemble[0], p2[0])
    npt.assert_equal(ensemble[1], p1[0])
    npt.assert_equal(ensemble.electrode_names, ['A-0','B-0'])

    # predict_percept smoke test
    ensemble.stim = [1,1]
    model = ScoreboardModel().build()
    model.predict_percept(ensemble)

# we essentially just need to make sure that electrode names are
# set properly, the rest of the EnsembleImplant functionality 
# (electrode placement, etc) is determined by the implants passed in
# and thus already tested
# but we'll test it again just to make sure
def test_ensemble_cortivis():
    cortivis0 = Cortivis(0)
    cortivis1 = Cortivis(x=10000)

    ensemble = EnsembleImplant([cortivis0, cortivis1])

    # check that positions are the same
    npt.assert_equal(ensemble['0-1'].x, cortivis0['1'].x)
    npt.assert_equal(ensemble['0-1'].y, cortivis0['1'].y)
    npt.assert_equal(ensemble['1-1'].x, cortivis1['1'].x)
    npt.assert_equal(ensemble['1-1'].y, cortivis1['1'].y)

# test from_coords initialization (physical coords in um)
def test_from_coords():
    locs = np.array([(0,0), (10000,0)])

    # check invalid instantiations
    with pytest.raises(TypeError):
        EnsembleImplant.from_coords(Cortivis(0), locs=locs)

    locs = np.array([(0,0), (10000,0), (0, 10000)])

    c0 = Cortivis(x=0,y=0)
    c1 = Cortivis(x=10000,y=0)
    c2 = Cortivis(x=0, y=10000)
    ensemble = EnsembleImplant.from_coords(Cortivis, locs=locs)

    # check that positions are the same
    npt.assert_equal(ensemble['0-1'].x, c0['1'].x)
    npt.assert_equal(ensemble['0-1'].y, c0['1'].y)
    npt.assert_equal(ensemble['0-1'].z, c0['1'].z)
    npt.assert_equal(ensemble['1-1'].x, c1['1'].x)
    npt.assert_equal(ensemble['1-1'].y, c1['1'].y)
    npt.assert_equal(ensemble['1-1'].z, c1['1'].z)
    npt.assert_equal(ensemble['2-1'].x, c2['1'].x)
    npt.assert_equal(ensemble['2-1'].y, c2['1'].y)
    npt.assert_equal(ensemble['2-1'].z, c2['1'].z)

# test from_cortical_map initialization (vf coords in dva)
def test_from_cortical_map():
    vfmap = Polimeni2006Map()

    locs = np.array([(2000,2000), (10000,0), (5000, 5000)]).astype(np.float64)

    # find locations in dva
    dva_x, dva_y = vfmap.to_dva()['v1'](locs[:,0], locs[:,1])
    dva_list = [(x,y) for x,y in zip(dva_x, dva_y)]
    dva_locs = np.array(dva_list)

    c0 = Cortivis(x=2000, y=2000)
    c1 = Cortivis(x=10000, y=0)
    c2 = Cortivis(x=5000, y=5000)

    # use dva coords to create ensemble
    ensemble = EnsembleImplant.from_cortical_map(Cortivis, vfmap, dva_locs)

    # check that positions are approx. the same
    npt.assert_approx_equal(ensemble['0-1'].x, c0['1'].x, 5)
    npt.assert_approx_equal(ensemble['0-1'].y, c0['1'].y, 5)
    npt.assert_approx_equal(ensemble['0-1'].z, c0['1'].z, 5)
    npt.assert_approx_equal(ensemble['1-1'].x, c1['1'].x, 5)
    npt.assert_approx_equal(ensemble['1-1'].y, c1['1'].y, 5)
    npt.assert_approx_equal(ensemble['1-1'].z, c1['1'].z, 5)
    npt.assert_approx_equal(ensemble['2-1'].x, c2['1'].x, 5)
    npt.assert_approx_equal(ensemble['2-1'].y, c2['1'].y, 5)
    npt.assert_approx_equal(ensemble['2-1'].z, c2['1'].z, 5)
