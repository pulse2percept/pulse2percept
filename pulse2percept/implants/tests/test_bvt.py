import numpy as np
import pytest
import numpy.testing as npt
from pulse2percept.implants.base import ProsthesisSystem
from pulse2percept.implants.bvt import BVT24, BVT44


@pytest.mark.parametrize('x', (-100, 200))
@pytest.mark.parametrize('y', (-200, 400))
@pytest.mark.parametrize('rot', (-45, 60))
@pytest.mark.parametrize('eye', ('LE', 'RE'))
def test_BVT24(x, y, rot, eye):
    # Create a BVT24 and make sure location is correct
    bva = BVT24(x=x, y=y, rot=rot, eye=eye)

    # Slots:
    npt.assert_equal(hasattr(bva, '__slots__'), True)
    npt.assert_equal(hasattr(bva, '__dict__'), False)

    # Make sure rotation + translation is applied correctly:
    bva0 = BVT24(eye=eye)  # centered
    xy = np.array([bva0['C1'].x, bva0['C1'].y]).T
    xy2 = np.array([bva0['C21m'].x, bva0['C21m'].y]).T
    # Rotate:
    rot_rad = np.deg2rad(rot)
    R = np.array([np.cos(rot_rad), -np.sin(rot_rad),
                  np.sin(rot_rad), np.cos(rot_rad)]).reshape((2, 2))
    xy = np.matmul(R, xy)
    xy2 = np.matmul(R, xy2)
    # Translate:
    npt.assert_almost_equal(bva['C1'].x, xy[0] + x)
    npt.assert_almost_equal(bva['C1'].y, xy[1] + y)
    npt.assert_almost_equal(bva['C21m'].x, xy2[0] + x)
    npt.assert_almost_equal(bva['C21m'].y, xy2[1] + y)

    # Check radii of electrodes
    for e in ['C1', 'C5', 'C8', 'C15', 'C20']:
        npt.assert_almost_equal(bva[e].r, 300.0)
    for e in ['C9', 'C17', 'C19']:
        npt.assert_almost_equal(bva[e].r, 200.0)
    for e in ['R1', 'R2']:
        npt.assert_almost_equal(bva[e].r, 1000.0)

    # Check the center is still at (x,y)
    y_center = (bva['C8'].y + bva['C13'].y) / 2
    npt.assert_almost_equal(y_center, y)
    x_center = (bva['C8'].x + bva['C13'].x) / 2
    npt.assert_almost_equal(x_center, x)

    # Right-eye implant:
    xc, yc = -500, -500
    bva_re = BVT24(eye='RE', x=xc, y=yc)
    npt.assert_equal(bva_re['C1'].x > bva_re['C6'].x, True)
    npt.assert_equal(bva_re['C1'].y, bva_re['C1'].y)

    # Left-eye implant:
    xc, yc = -500, -500
    bva_le = BVT24(eye='LE', x=xc, y=yc)
    npt.assert_equal(bva_le['C1'].x < bva_le['C6'].x, True)
    npt.assert_equal(bva_le['C1'].y, bva_le['C1'].y)


def test_BVT24_stim():
    # Assign a stimulus:
    implant = BVT24()
    implant.stim = {'C1': 1}
    npt.assert_equal(implant.stim.electrodes, ['C1'])
    npt.assert_equal(implant.stim.time, None)
    npt.assert_equal(implant.stim.data, [[1]])

    # You can also assign the stimulus in the constructor:
    BVT24(stim={'C1': 1})
    npt.assert_equal(implant.stim.electrodes, ['C1'])
    npt.assert_equal(implant.stim.time, None)
    npt.assert_equal(implant.stim.data, [[1]])

    # Set a stimulus via array:
    implant = BVT24(stim=np.ones(35))
    npt.assert_equal(implant.stim.shape, (35, 1))
    npt.assert_almost_equal(implant.stim.data, 1)


@pytest.mark.parametrize('x', (-100, 200))
@pytest.mark.parametrize('y', (-200, 400))
@pytest.mark.parametrize('rot', (-45, 60))
@pytest.mark.parametrize('eye', ('LE', 'RE'))
def test_BVT44(x, y, rot, eye):
    # Create a BVT44 and make sure location is correct
    bva = BVT44(x=x, y=y, rot=rot, eye=eye)

    # Slots:
    npt.assert_equal(hasattr(bva, '__slots__'), True)
    npt.assert_equal(hasattr(bva, '__dict__'), False)

    # Make sure array is rotated + translated correctly:
    bva0 = BVT44(eye=eye)
    xy = np.array([bva0['A1'].x, bva0['A1'].y]).T
    xy2 = np.array([bva0['G6'].x, bva0['G6'].y]).T
    # Rotate:
    rot_rad = np.deg2rad(rot)
    R = np.array([np.cos(rot_rad), -np.sin(rot_rad),
                  np.sin(rot_rad), np.cos(rot_rad)]).reshape((2, 2))
    xy = np.matmul(R, xy)
    xy2 = np.matmul(R, xy2)
    # Translate:
    npt.assert_almost_equal(bva['A1'].x, xy[0] + x)
    npt.assert_almost_equal(bva['A1'].y, xy[1] + y)
    npt.assert_almost_equal(bva['G6'].x, xy2[0] + x)
    npt.assert_almost_equal(bva['G6'].y, xy2[1] + y)

    # Check radii of electrodes
    for e in ['A1', 'A5', 'B3', 'C5', 'D2']:
        npt.assert_almost_equal(bva[e].r, 500.0)
    for e in ['R1', 'R2']:
        npt.assert_almost_equal(bva[e].r, 1000.0)

    # Check the center is still at (x,y)
    npt.assert_almost_equal((bva['D4'].x + bva['D5'].x) / 2.0, x)
    npt.assert_almost_equal((bva['E4'].y + bva['C4'].y) / 2.0, y)

    # Right-eye implant:
    xc, yc = -500, -500
    bva_re = BVT44(eye='RE', x=xc, y=yc)
    npt.assert_equal(bva_re['A6'].x > bva_re['A1'].x, True)
    npt.assert_equal(bva_re['A6'].y, bva_re['A1'].y)

    # Left-eye implant:
    xc, yc = -500, -500
    bva_le = BVT44(eye='LE', x=xc, y=yc)
    npt.assert_equal(bva_le['A6'].x < bva_le['A1'].x, True)
    npt.assert_equal(bva_le['A6'].y, bva_le['A1'].y)


def test_BVT44_stim():
    # Assign a stimulus:
    implant = BVT44()
    implant.stim = {'A1': 1}
    npt.assert_equal(implant.stim.electrodes, ['A1'])
    npt.assert_equal(implant.stim.time, None)
    npt.assert_equal(implant.stim.data, [[1]])

    # You can also assign the stimulus in the constructor:
    BVT44(stim={'A1': 1})
    npt.assert_equal(implant.stim.electrodes, ['A1'])
    npt.assert_equal(implant.stim.time, None)
    npt.assert_equal(implant.stim.data, [[1]])

    # Set a stimulus via array:
    implant = BVT44(stim=np.ones(46))
    npt.assert_equal(implant.stim.shape, (46, 1))
    npt.assert_almost_equal(implant.stim.data, 1)
