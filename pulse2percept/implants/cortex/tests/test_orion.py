import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept.implants.cortex.orion import Orion


def test_orion():
    orion = Orion()

    n_elec = 60
    spacing = (4200, np.sqrt(3**2-2.1**2)*1000)

    # Slots:
    npt.assert_equal(hasattr(orion, '__slots__'), True)
    npt.assert_equal(hasattr(orion, '__dict__'), False)

    # Make sure number of electrodes is correct
    npt.assert_equal(orion.n_electrodes, n_elec)
    npt.assert_equal(len(orion.electrode_array.electrodes), n_elec)

    # Make sure the radius is correct
    for electrode in orion.electrode_array.electrode_objects:
        npt.assert_almost_equal(electrode.radius, 1000)

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
