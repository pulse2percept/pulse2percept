import numpy as np
import numpy.testing as npt
import pytest
from pulse2percept.implants import (EnsembleImplant, PointSource, ProsthesisSystem)
from pulse2percept.implants.cortex import Cortivis
from pulse2percept.models.cortex.base import ScoreboardModel

def test_EnsembleImplant():
    # Invalid instantiations:
    with pytest.raises(TypeError):
        EnsembleImplant(implants="this can't happen")

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