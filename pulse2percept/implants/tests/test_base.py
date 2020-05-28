import numpy as np
import collections as coll
import pytest
import numpy.testing as npt

from pulse2percept.implants import (PointSource, ElectrodeArray,
                                    ProsthesisSystem)
from pulse2percept.stimuli import Stimulus


def test_ProsthesisSystem():
    # Invalid instantiations:
    with pytest.raises(ValueError):
        ProsthesisSystem(ElectrodeArray(PointSource(0, 0, 0)),
                         eye='both')
    with pytest.raises(TypeError):
        ProsthesisSystem(Stimulus)

    # Iterating over the electrode array:
    implant = ProsthesisSystem(PointSource(0, 0, 0))
    npt.assert_equal(implant.n_electrodes, 1)
    npt.assert_equal(implant[0], implant.earray[0])
    npt.assert_equal(implant.keys(), implant.earray.keys())
    for i, e in zip(implant, implant.earray):
        npt.assert_equal(i, e)

    # Set a stimulus after the constructor:
    npt.assert_equal(implant.stim, None)
    implant.stim = 3
    npt.assert_equal(isinstance(implant.stim, Stimulus), True)
    npt.assert_equal(implant.stim.shape, (1, 1))
    npt.assert_equal(implant.stim.time, None)
    npt.assert_equal(implant.stim.electrodes, [0])

    with pytest.raises(ValueError):
        # Wrong number of stimuli
        implant.stim = [1, 2]
    with pytest.raises(TypeError):
        # Invalid stim type:
        implant.stim = "stim"
    # Invalid electrode names:
    with pytest.raises(ValueError):
        implant.stim = {'A1': 1}
    with pytest.raises(ValueError):
        implant.stim = Stimulus({'A1': 1})

    # Slots:
    npt.assert_equal(hasattr(implant, '__slots__'), True)
    npt.assert_equal(hasattr(implant, '__dict__'), False)
