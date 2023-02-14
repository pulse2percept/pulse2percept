import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept.implants.cortex.orion import Orion


@pytest.mark.parametrize('x', (-100, 200))
@pytest.mark.parametrize('y', (-200, 400))
@pytest.mark.parametrize('rot', (-45, 60))
def test_orion(x, y, rot):
    orion = Orion(x, y, rot=rot)
    non_rot_orion = Orion(0)

    n_elec = 60
    spacing = (4200, np.sqrt(3**2-2.1**2)*1000)

    # Slots:
    npt.assert_equal(hasattr(orion, '__slots__'), True)
    npt.assert_equal(hasattr(orion, '__dict__'), False)

    # Make sure number of electrodes is correct
    npt.assert_equal(orion.n_electrodes, n_elec)
    npt.assert_equal(len(orion.earray.electrodes), n_elec)

    # Coordinates of 55 when device is not rotated:
    xy = np.array([non_rot_orion['55'].x, non_rot_orion['55'].y])
    # Rotate
    rot_rad = np.deg2rad(rot)
    R = np.array([np.cos(rot_rad), -np.sin(rot_rad),
                  np.sin(rot_rad), np.cos(rot_rad)]).reshape((2, 2))
    xy = R @ xy
    # Then off-set: Make sure first electrode is placed
    # correctly
    npt.assert_almost_equal(orion['55'].x, xy[0] + x, decimal=2)
    npt.assert_almost_equal(orion['55'].y, xy[1] + y, decimal=2)

    # Make sure the radius is correct
    for electrode in orion.earray.electrode_objects:
        npt.assert_almost_equal(electrode.r, 1000)

    # Make sure the pitch is correct:
    # distance between two electrodes that are one row apart and adjacent horizontally
    diag_dist = np.sqrt((spacing[0] / 2) ** 2 + spacing[1] ** 2)
    npt.assert_almost_equal(np.sqrt(
        (orion['55'].x - orion['49'].x) ** 2 +
        (orion['55'].y - orion['49'].y) ** 2),
        diag_dist
    )
    npt.assert_almost_equal(np.sqrt(
        (orion['55'].x - orion['60'].x) ** 2 +
        (orion['55'].y - orion['60'].y) ** 2),
        diag_dist
    )
