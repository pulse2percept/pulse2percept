import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept import implants

@pytest.mark.parametrize('rot', (-45, 60))
@pytest.mark.parametrize('eye', ('LE', 'RE'))
def test_IMIE(rot, eye):
    # Create an IMIE and make sure location is correct

    imie = implants.IMIE(rot=rot, eye = eye)
    imie0 = implants.IMIE(eye = eye)
    # Slots:
    npt.assert_equal(hasattr(imie, '__slots__'), True)
    npt.assert_equal(hasattr(imie, '__dict__'), False)

    # Check if there is 256 electrodes in the array
    npt.assert_equal(len(imie.electrode_array.electrodes), 256)

    # Coordinates of electrode 'N3'
    xy = np.array([imie0['N3'].x, imie0['N3'].y]).T

    # Rotate
    rot_rad = np.deg2rad(rot)
    R = np.array([np.cos(rot_rad), -np.sin(rot_rad),
                  np.sin(rot_rad), np.cos(rot_rad)]).reshape((2, 2))
    xy = np.matmul(R, xy)

    # Then off-set: Make sure electrode N3 is placed
    # correctly
    npt.assert_almost_equal(imie['N3'].x, xy[0])
    npt.assert_almost_equal(imie['N3'].y, xy[1])

    # The array is centered on the device's own origin
    y_center = imie['H10'].y + (imie['G10'].y - imie['H10'].y) / 2
    npt.assert_almost_equal(y_center, 0)
    x_center = imie['H10'].x + (imie['G10'].x - imie['H10'].x) / 2
    npt.assert_almost_equal(x_center, 0)

    # Make sure the center to center pitch is correct
    npt.assert_almost_equal((imie['L1'].x - imie['K1'].x) ** 2 + 
                            (imie['L1'].y - imie['K1'].y) ** 2,
                            300**2)
    npt.assert_almost_equal((imie['A3'].x - imie['A4'].x) ** 2 + 
                            (imie['A3'].y - imie['A4'].y) ** 2,
                            350**2)

    # Check radii of electrodes
    for e in ['N16', 'N17', 'A16', 'A17', 'L1', 'K1', 'C1', 'D1']:
        npt.assert_almost_equal(imie[e].radius, 80.0)
    for e in ['A3', 'M15', 'B19', 'C15', 'D13']:
        npt.assert_almost_equal(imie[e].radius, 105.0)

    # `h` must have the right dimensions
    with pytest.raises(ValueError):
        implants.IMIE(z=np.zeros(5))
    with pytest.raises(ValueError):
        implants.IMIE(z=[1, 2, 3])

    # Right-eye implant:
    imie_re = implants.IMIE(eye='RE')
    npt.assert_equal(imie_re['A4'].x > imie_re['A3'].x, True)
    npt.assert_almost_equal(imie_re['A4'].y, imie_re['A3'].y)

    # need to adjust for reflection about y-axis
    # Left-eye implant:
    imie_le = implants.IMIE(eye='LE')
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
    # Prepare a stimulus via dict:
    implant = implants.IMIE()
    stim = implant.prepare_stim({'A3': 1})
    npt.assert_equal(stim.electrodes, ['A3'])
    npt.assert_equal(stim.time, None)
    npt.assert_equal(stim.data, [[1]])

    # Prepare a stimulus via array:
    stim = implant.prepare_stim(np.ones(256))
    npt.assert_equal(stim.shape, (256, 1))
    npt.assert_almost_equal(stim.data, 1)