import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept.implants.cortex import Cortivis

@pytest.mark.parametrize('x', (-100, 200))
@pytest.mark.parametrize('y', (-200, 400))
@pytest.mark.parametrize('rot', (-45, 60))
def test_Cortivis(x, y, rot):
    # Create a Cortivis implant and make sure location is correct
    # Depth 'z' must be 0
    cortivis = Cortivis(x=x, y=y, rot=rot)
    cortivis0 = Cortivis(x=0, y=0, rot=0)
    
    # Slots:
    npt.assert_equal(hasattr(cortivis, '__slots__'), True)
    npt.assert_equal(hasattr(cortivis, '__dict__'), False)

    # Check if there are 96 electrodes in the array
    npt.assert_equal(len(cortivis.earray.electrodes), 96)

    # Coordinates of electrode '1'
    xy = np.array([cortivis0['1'].x, cortivis0['1'].y]).T

    # Rotate
    rot_rad = np.deg2rad(rot)
    R = np.array([np.cos(rot_rad), -np.sin(rot_rad),
                  np.sin(rot_rad), np.cos(rot_rad)]).reshape((2, 2))
    xy = np.matmul(R, xy)

    # Then off-set: Make sure first electrode is placed
    # correctly
    npt.assert_almost_equal(cortivis['1'].x, xy[0] + x)
    npt.assert_almost_equal(cortivis['1'].y, xy[1] + y)

    # Check radii of electrodes
    for e in cortivis.earray.electrode_objects:
        npt.assert_almost_equal(e.r, 40)
    