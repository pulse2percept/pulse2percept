import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept.implants.cortex import NeuroPortArray


def test_NeuroPortArray():
    neuroport = NeuroPortArray()

    # Slots:
    npt.assert_equal(hasattr(neuroport, '__slots__'), True)
    npt.assert_equal(hasattr(neuroport, '__dict__'), False)

    # Check if there are 96 electrodes in the array
    npt.assert_equal(len(neuroport.electrode_array.electrodes), 96)

    # Check radii of electrodes; shank tips sit 1.5 mm deep:
    for e in neuroport.electrode_array.electrode_objects:
        npt.assert_almost_equal(e.radius, 40)
        npt.assert_almost_equal(e.z, -1500)
    