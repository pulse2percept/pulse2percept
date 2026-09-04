import numpy as np
import pytest
import numpy.testing as npt
from pulse2percept.implants.base import Implant
from pulse2percept.implants.bvt import BVT24, BVT44


@pytest.mark.parametrize('eye', ('LE', 'RE'))
def test_BVT24(eye):
    # Create a BVT24 and make sure location is correct
    bva = BVT24(eye=eye)

    # Slots:
    npt.assert_equal(hasattr(bva, '__slots__'), True)
    npt.assert_equal(hasattr(bva, '__dict__'), False)

    # Check radii of electrodes
    for e in ['C1', 'C5', 'C8', 'C15', 'C20']:
        npt.assert_almost_equal(bva[e].radius, 300.0)
    for e in ['C9', 'C17', 'C19']:
        npt.assert_almost_equal(bva[e].radius, 200.0)
    for e in ['R1', 'R2']:
        npt.assert_almost_equal(bva[e].radius, 1000.0)

    # The array is centered on the device's own origin
    y_center = (bva['C8'].y + bva['C13'].y) / 2
    npt.assert_almost_equal(y_center, 0)
    x_center = (bva['C8'].x + bva['C13'].x) / 2
    npt.assert_almost_equal(x_center, 0)

    # Right-eye implant:
    bva_re = BVT24(eye='RE')
    npt.assert_equal(bva_re['C1'].x > bva_re['C6'].x, True)
    npt.assert_equal(bva_re['C1'].y, bva_re['C1'].y)

    # Left-eye implant:
    bva_le = BVT24(eye='LE')
    npt.assert_equal(bva_le['C1'].x < bva_le['C6'].x, True)
    npt.assert_equal(bva_le['C1'].y, bva_le['C1'].y)


def test_BVT24_stim():
    # Prepare a stimulus via dict:
    implant = BVT24()
    stim = implant.prepare_stim({'C1': 1})
    npt.assert_equal(stim.electrodes, ['C1'])
    npt.assert_equal(stim.time, None)
    npt.assert_equal(stim.data, [[1]])

    # Prepare a stimulus via array:
    stim = implant.prepare_stim(np.ones(35))
    npt.assert_equal(stim.shape, (35, 1))
    npt.assert_almost_equal(stim.data, 1)


@pytest.mark.parametrize('eye', ('LE', 'RE'))
def test_BVT44(eye):
    # Create a BVT44 and make sure location is correct
    bva = BVT44(eye=eye)

    # Slots:
    npt.assert_equal(hasattr(bva, '__slots__'), True)
    npt.assert_equal(hasattr(bva, '__dict__'), False)

    # Check radii of electrodes
    for e in ['A1', 'A5', 'B3', 'C5', 'D2']:
        npt.assert_almost_equal(bva[e].radius, 500.0)
    for e in ['R1', 'R2']:
        npt.assert_almost_equal(bva[e].radius, 1000.0)

    # The array is centered on the device's own origin
    npt.assert_almost_equal((bva['D4'].x + bva['D5'].x) / 2.0, 0)
    npt.assert_almost_equal((bva['E4'].y + bva['C4'].y) / 2.0, 0)

    # Right-eye implant:
    bva_re = BVT44(eye='RE')
    npt.assert_equal(bva_re['A6'].x > bva_re['A1'].x, True)
    npt.assert_equal(bva_re['A6'].y, bva_re['A1'].y)

    # Left-eye implant:
    bva_le = BVT44(eye='LE')
    npt.assert_equal(bva_le['A6'].x < bva_le['A1'].x, True)
    npt.assert_equal(bva_le['A6'].y, bva_le['A1'].y)


def test_BVT44_stim():
    # Prepare a stimulus via dict:
    implant = BVT44()
    stim = implant.prepare_stim({'A1': 1})
    npt.assert_equal(stim.electrodes, ['A1'])
    npt.assert_equal(stim.time, None)
    npt.assert_equal(stim.data, [[1]])

    # Prepare a stimulus via array:
    stim = implant.prepare_stim(np.ones(46))
    npt.assert_equal(stim.shape, (46, 1))
    npt.assert_almost_equal(stim.data, 1)


@pytest.mark.parametrize('cls', (BVT24, BVT44))
def test_BVT_rejects_rot(cls):
    """Orientation in tissue is the model's `implant_rotation`"""
    with pytest.raises(TypeError):
        cls(rot=30)
