import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept.implants.cortex import NeuroPortArray


def test_NeuroPortArray():
    # Create a NeuroPortArray implant and make sure location is correct
    # Depth 'z' must be 0
    neuroport = NeuroPortArray()

    # Slots:
    npt.assert_equal(hasattr(neuroport, '__slots__'), True)
    npt.assert_equal(hasattr(neuroport, '__dict__'), False)

    # Check if there are 96 electrodes in the array
    npt.assert_equal(len(neuroport.electrode_array.electrodes), 96)

    # Check radii of electrodes
    for e in neuroport.electrode_array.electrode_objects:
        npt.assert_almost_equal(e.radius, 40)
    