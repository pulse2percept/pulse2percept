import numpy as np
import pytest
import numpy.testing as npt
from pulse2percept.implants.base import ProsthesisSystem
from pulse2percept.implants.bva import BVA24

@pytest.mark.parametrize('x', (-100, 200))
@pytest.mark.parametrize('y', (-200, 400))
@pytest.mark.parametrize('r', (-45, 60))
def test_BVA24(x, y, r):
    # Create a BVA24 and make sure location is correct
    # Convert rotation angle to rad
    rot = np.deg2rad(r)
    bva = BVA24(x_center=x, y_center=y, rot=r)
    
    # Coordinate of first electrode (electrode '1')
    xy = np.array([-1275.0, 1520.0]).T
    # Coordinate of last electrode (electrode '21m')
    xy2 = np.array([-850.0, -2280.0]).T
    
    # Rotate
    R = np.array([np.cos(rot), -np.sin(rot),
                  np.sin(rot), np.cos(rot)]).reshape((2,2))
    xy = np.matmul(R, xy)
    xy2 = np.matmul(R, xy2)
    
    # Then off-set: Make sure first electrode is placed
    # correctly
    npt.assert_almost_equal(bva['1'].x, xy[0] + x)
    npt.assert_almost_equal(bva['1'].y, xy[1] + y)
    npt.assert_almost_equal(bva['21m'].x, xy2[0] + x)
    npt.assert_almost_equal(bva['21m'].y, xy2[1] + y)
    
    # Check radii of electrodes
    for e in ['1', '5', '8', '15', '20']:
        npt.assert_almost_equal(bva[e].r, 300.0)
    for e in ['9', '17', '19']:
        npt.assert_almost_equal(bva[e].r, 200.0)
    for e in ['R1', 'R2']:
        npt.assert_almost_equal(bva[e].r, 1000.0)
    
    # Check the center is still at (x,y)
    y_center = (bva['8'].y + bva['13'].y)/2
    npt.assert_almost_equal(y_center, y)
    x_center = (bva['8'].x + bva['13'].x)/2
    npt.assert_almost_equal(x_center, x)
    
    # Right-eye implant:
    xc,yc = 500, -500
    bva_re = BVA24(eye='RE', x_center=xc, y_center=yc)
    npt.assert_equal(bva_re['1'].x < bva_re['6'].x, True)
    npt.assert_equal(bva_re['1'].y, bva_re['1'].y)
    
    # Left-eye implant:
    xc,yc = 500, -500
    bva_le = BVA24(eye='LE', x_center=xc, y_center=yc)
    npt.assert_equal(bva_le['1'].x > bva_le['6'].x, True)
    npt.assert_equal(bva_le['1'].y, bva_le['1'].y)