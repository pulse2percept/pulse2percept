import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept.implants.cortex.icvp import ICVP

@pytest.mark.parametrize('x', (-100, 200))
@pytest.mark.parametrize('y', (-200, 400))
@pytest.mark.parametrize('rot', (-45, 60))
def test_icvp(x, y, rot):
    icvp = ICVP(x, y, rot=rot)
    non_rot_icvp = ICVP(0)

    n_elec = 18
    spacing = 400
    radius = 50
    length_650 = {'9', '2', '6', '11', '15', '4', '8', '13'}
    deactivated_electrodes = {'R', 'C'}

    # Slots:
    npt.assert_equal(hasattr(icvp, '__slots__'), True)
    npt.assert_equal(hasattr(icvp, '__dict__'), False)

    # Make sure number of electrodes is correct
    npt.assert_equal(icvp.n_electrodes, n_elec)
    npt.assert_equal(len(icvp.earray.electrodes), n_elec)

    # Coordinates of 11 when device is not rotated:
    xy = np.array([non_rot_icvp['11'].x, non_rot_icvp['11'].y])
    # Rotate
    rot_rad = np.deg2rad(rot)
    R = np.array([np.cos(rot_rad), -np.sin(rot_rad),
                  np.sin(rot_rad), np.cos(rot_rad)]).reshape((2, 2))
    xy = R @ xy
    # Then off-set: Make sure first electrode is placed
    # correctly
    npt.assert_almost_equal(icvp['11'].x, xy[0] + x, decimal=2)
    npt.assert_almost_equal(icvp['11'].y, xy[1] + y, decimal=2)

    for electrode in icvp.earray.electrode_objects:
        npt.assert_almost_equal(electrode.r, radius)

        if electrode.name in deactivated_electrodes:
            npt.assert_equal(electrode.activated, False)
        else:
            npt.assert_equal(electrode.activated, True)

        if electrode.name in length_650:
            npt.assert_equal(electrode.z, -650)
        else:
            npt.assert_equal(electrode.z, -850)

    # Make sure center to center spacing is correct
    npt.assert_almost_equal(np.sqrt(
        (icvp['11'].x - icvp['7'].x) ** 2 +
        (icvp['11'].y - icvp['7'].y) ** 2),
        spacing
    )
    npt.assert_almost_equal(np.sqrt(
        (icvp['11'].x - icvp['10'].x) ** 2 +
        (icvp['11'].y - icvp['10'].y) ** 2),
        spacing
    )
    npt.assert_almost_equal(np.sqrt(
        (icvp['11'].x - icvp['15'].x) ** 2 +
        (icvp['11'].y - icvp['15'].y) ** 2),
        spacing
    )
