import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept.implants.cortex import Cortivis


def test_Cortivis():
    # Create a Cortivis implant and make sure location is correct
    # Depth 'z' must be 0
    cortivis = Cortivis()

    # Slots:
    npt.assert_equal(hasattr(cortivis, '__slots__'), True)
    npt.assert_equal(hasattr(cortivis, '__dict__'), False)

    # Check if there are 96 electrodes in the array
    npt.assert_equal(len(cortivis.electrode_array.electrodes), 96)

    # Check radii of electrodes
    for e in cortivis.electrode_array.electrode_objects:
        npt.assert_almost_equal(e.radius, 40)
    