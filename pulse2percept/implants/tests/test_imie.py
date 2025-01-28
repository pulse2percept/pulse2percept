import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept import implants

@pytest.mark.parametrize('x', (-100, 200))
@pytest.mark.parametrize('y', (-200, 400))
@pytest.mark.parametrize('rot', (-45, 60))
@pytest.mark.parametrize('eye', ('LE', 'RE'))
def test_IMIE(x, y, rot, eye):
    # Create an IMIE and make sure location is correct

    imie = implants.IMIE(x, y, rot=rot, eye = eye)
    imie0 = implants.IMIE(eye = eye)
    # Slots:
    npt.assert_equal(hasattr(imie, '__slots__'), True)
    npt.assert_equal(hasattr(imie, '__dict__'), False)

    # Check if there is 256 electrodes in the array
    npt.assert_equal(len(imie.earray.electrodes), 256)

    # Coordinates of electrode 'N3'
    xy = np.array([imie0['N3'].x, imie0['N3'].y]).T

    # Rotate
    rot_rad = np.deg2rad(rot)
    R = np.array([np.cos(rot_rad), -np.sin(rot_rad),
                  np.sin(rot_rad), np.cos(rot_rad)]).reshape((2, 2))
    xy = np.matmul(R, xy)

    # Then off-set: Make sure electrode N3 is placed
    # correctly
    npt.assert_almost_equal(imie['N3'].x, xy[0] + x)
    npt.assert_almost_equal(imie['N3'].y, xy[1] + y)

    # Make sure array center is still (x,y)
    y_center = imie['H10'].y + (imie['G10'].y - imie['H10'].y) / 2
    npt.assert_almost_equal(y_center, y)
    x_center = imie['H10'].x + (imie['G10'].x - imie['H10'].x) / 2
    npt.assert_almost_equal(x_center, x)

    # Make sure the center to center pitch is correct
    npt.assert_almost_equal((imie['L1'].x - imie['K1'].x) ** 2 + 
                            (imie['L1'].y - imie['K1'].y) ** 2,
                            300**2)
    npt.assert_almost_equal((imie['A3'].x - imie['A4'].x) ** 2 + 
                            (imie['A3'].y - imie['A4'].y) ** 2,
                            350**2)

    # Check radii of electrodes
    for e in ['N16', 'N17', 'A16', 'A17', 'L1', 'K1', 'C1', 'D1']:
        npt.assert_almost_equal(imie[e].r, 80.0)
    for e in ['A3', 'M15', 'B19', 'C15', 'D13']:
        npt.assert_almost_equal(imie[e].r, 105.0)

    # `h` must have the right dimensions
    with pytest.raises(ValueError):
        implants.IMIE(x=-100, y=10, z=np.zeros(5))
    with pytest.raises(ValueError):
        implants.IMIE(x=-100, y=10, z=[1, 2, 3])

    # Right-eye implant:
    xc, yc = 500, -500
    imie_re = implants.IMIE(eye='RE', x=xc, y=yc)
    npt.assert_equal(imie_re['A4'].x > imie_re['A3'].x, True)
    npt.assert_almost_equal(imie_re['A4'].y, imie_re['A3'].y)

    # need to adjust for reflection about y-axis
    # Left-eye implant:
    imie_le = implants.IMIE(eye='LE', x=xc, y=yc)
    npt.assert_equal(imie_le['A3'].x > imie_le['A4'].x, True)
    npt.assert_almost_equal(imie_le['A3'].y, imie_le['A4'].y)

    # In both left and right eyes, rotation with positive angle should be
    # counter-clock-wise (CCW): for (x>0,y>0), decreasing x and increasing y
    for eye, el in zip(['LE', 'RE'], ['L5', 'L17']):
        before = implants.IMIE(eye=eye)
        after = implants.IMIE(eye=eye, rot=10)
        npt.assert_equal(after[el].x < before[el].x, True)
        npt.assert_equal(after[el].y > before[el].y, True)

def test_IMIE_stim():
    # Assign a stimulus:
    implant = implants.IMIE()
    implant.stim = {'A3': 1}
    npt.assert_equal(implant.stim.electrodes, ['A3'])
    npt.assert_equal(implant.stim.time, None)
    npt.assert_equal(implant.stim.data, [[1]])

    # You can also assign the stimulus in the constructor:
    implants.IMIE(stim={'A3': 1})
    npt.assert_equal(implant.stim.electrodes, ['A3'])
    npt.assert_equal(implant.stim.time, None)
    npt.assert_equal(implant.stim.data, [[1]])

    # Set a stimulus via array:
    implant = implants.IMIE(stim=np.ones(256))
    npt.assert_equal(implant.stim.shape, (256, 1))
    npt.assert_almost_equal(implant.stim.data, 1)