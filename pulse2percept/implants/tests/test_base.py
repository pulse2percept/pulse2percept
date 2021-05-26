import numpy as np
import collections as coll
import pytest
import numpy.testing as npt
from matplotlib.patches import Circle

from pulse2percept.implants import (PointSource, ElectrodeArray, ElectrodeGrid,
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
    npt.assert_equal(implant.electrode_names, implant.earray.electrode_names)
    for i, e in zip(implant, implant.earray):
        npt.assert_equal(i, e)

    # Set a stimulus after the constructor:
    npt.assert_equal(implant.stim, None)
    implant.stim = 3
    npt.assert_equal(isinstance(implant.stim, Stimulus), True)
    npt.assert_equal(implant.stim.shape, (1, 1))
    npt.assert_equal(implant.stim.time, None)
    npt.assert_equal(implant.stim.electrodes, [0])

    ax = implant.plot()
    npt.assert_equal(len(ax.texts), 0)
    npt.assert_equal(len(ax.collections), 1)

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
    # Safe mode requires charge-balanced pulses:
    with pytest.raises(ValueError):
        implant = ProsthesisSystem(PointSource(0, 0, 0), safe_mode=True)
        implant.stim = 1

    # Slots:
    npt.assert_equal(hasattr(implant, '__slots__'), True)
    npt.assert_equal(hasattr(implant, '__dict__'), False)


def test_ProsthesisSystem_stim():
    implant = ProsthesisSystem(ElectrodeGrid((13, 13), 20))
    stim = Stimulus(np.ones((13 * 13 + 1, 5)))
    with pytest.raises(ValueError):
        implant.stim = stim

    # Deactivated electrodes cannot receive stimuli:
    implant.deactivate('H4')
    npt.assert_equal(implant['H4'].activated, False)
    with pytest.raises(ValueError):
        implant.stim = {'H4': 1}
    implant.deactivate('all')
    with pytest.raises(ValueError):
        implant.stim = [1] * implant.n_electrodes
    implant.activate('all')
    implant.stim = {'H4': 1}
