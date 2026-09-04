import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept.implants.cortex.icvp import ICVP


def test_icvp():
    icvp = ICVP()

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
    npt.assert_equal(len(icvp.electrode_array.electrodes), n_elec)

    for electrode in icvp.electrode_array.electrode_objects:
        npt.assert_almost_equal(electrode.radius, radius)

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
