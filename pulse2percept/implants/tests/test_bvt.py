import numpy as np
import pytest
import numpy.testing as npt
from pulse2percept.implants.base import ProsthesisSystem
from pulse2percept.implants.bvt import BVT24


@pytest.mark.parametrize('x', (-100, 200))
@pytest.mark.parametrize('y', (-200, 400))
@pytest.mark.parametrize('rot', (-45, 60))
def test_BVT24(x, y, rot):
    # Create a BVT24 and make sure location is correct
    bva = BVT24(x=x, y=y, rot=rot)

    # Slots:
    npt.assert_equal(hasattr(bva, '__slots__'), True)
    npt.assert_equal(hasattr(bva, '__dict__'), False)

    # Coordinate of first electrode (electrode '1')
    xy = np.array([-1275.0, 1520.0]).T
    # Coordinate of last electrode (electrode '21m')
    xy2 = np.array([-850.0, -2280.0]).T

    # Rotate
    rot_rad = np.deg2rad(rot)
    R = np.array([np.cos(rot_rad), -np.sin(rot_rad),
                  np.sin(rot_rad), np.cos(rot_rad)]).reshape((2, 2))
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
    y_center = (bva['8'].y + bva['13'].y) / 2
    npt.assert_almost_equal(y_center, y)
    x_center = (bva['8'].x + bva['13'].x) / 2
    npt.assert_almost_equal(x_center, x)

    # Right-eye implant:
    xc, yc = 500, -500
    bva_re = BVT24(eye='RE', x=xc, y=yc)
    npt.assert_equal(bva_re['1'].x < bva_re['6'].x, True)
    npt.assert_equal(bva_re['1'].y, bva_re['1'].y)

    # Left-eye implant:
    xc, yc = 500, -500
    bva_le = BVT24(eye='LE', x=xc, y=yc)
    npt.assert_equal(bva_le['1'].x > bva_le['6'].x, True)
    npt.assert_equal(bva_le['1'].y, bva_le['1'].y)


def test_BVT24_stim():
    # Assign a stimulus:
    implant = BVT24()
    implant.stim = {'1': 1}
    npt.assert_equal(implant.stim.electrodes, ['1'])
    npt.assert_equal(implant.stim.time, None)
    npt.assert_equal(implant.stim.data, [[1]])

    # You can also assign the stimulus in the constructor:
    BVT24(stim={'1': 1})
    npt.assert_equal(implant.stim.electrodes, ['1'])
    npt.assert_equal(implant.stim.time, None)
    npt.assert_equal(implant.stim.data, [[1]])

    # Set a stimulus via array:
    implant = BVT24(stim=np.ones(35))
    npt.assert_equal(implant.stim.shape, (35, 1))
    npt.assert_almost_equal(implant.stim.data, 1)
