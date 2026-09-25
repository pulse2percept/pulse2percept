import numpy as np
import pytest
import numpy.testing as npt
from pulse2percept.implants.base import Implant
from pulse2percept.units import mm, um
from pulse2percept.implants.retina.suprachoroidal import (Suprachoroidal24,
                                                          Suprachoroidal44)


@pytest.mark.parametrize('eye', ('left', 'right'))
def test_Suprachoroidal24(eye):
    # Create a Suprachoroidal24 and make sure location is correct
    bva = Suprachoroidal24(eye=eye)

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
    bva_re = Suprachoroidal24(eye='right')
    npt.assert_equal(bva_re['C1'].x > bva_re['C6'].x, True)
    npt.assert_equal(bva_re['C1'].y, bva_re['C1'].y)

    # Left-eye implant:
    bva_le = Suprachoroidal24(eye='left')
    npt.assert_equal(bva_le['C1'].x < bva_le['C6'].x, True)
    npt.assert_equal(bva_le['C1'].y, bva_le['C1'].y)


def test_Suprachoroidal24_stim():
    # Prepare a stimulus via dict:
    implant = Suprachoroidal24()
    stim = implant.prepare_stim({'C1': 1})
    npt.assert_equal(stim.electrodes, ['C1'])
    npt.assert_equal(stim.time, None)
    npt.assert_equal(stim.data, [[1]])

    # Prepare a stimulus via array:
    stim = implant.prepare_stim(np.ones(35))
    npt.assert_equal(stim.shape, (35, 1))
    npt.assert_almost_equal(stim.data, 1)


@pytest.mark.parametrize('eye', ('left', 'right'))
def test_Suprachoroidal44(eye):
    # Create a Suprachoroidal44 and make sure location is correct
    bva = Suprachoroidal44(eye=eye)

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
    bva_re = Suprachoroidal44(eye='right')
    npt.assert_equal(bva_re['A6'].x > bva_re['A1'].x, True)
    npt.assert_equal(bva_re['A6'].y, bva_re['A1'].y)

    # Left-eye implant:
    bva_le = Suprachoroidal44(eye='left')
    npt.assert_equal(bva_le['A6'].x < bva_le['A1'].x, True)
    npt.assert_equal(bva_le['A6'].y, bva_le['A1'].y)


def test_Suprachoroidal44_stim():
    # Prepare a stimulus via dict:
    implant = Suprachoroidal44()
    stim = implant.prepare_stim({'A1': 1})
    npt.assert_equal(stim.electrodes, ['A1'])
    npt.assert_equal(stim.time, None)
    npt.assert_equal(stim.data, [[1]])

    # Prepare a stimulus via array:
    stim = implant.prepare_stim(np.ones(46))
    npt.assert_equal(stim.shape, (46, 1))
    npt.assert_almost_equal(stim.data, 1)


@pytest.mark.parametrize('cls', (Suprachoroidal24, Suprachoroidal44))
def test_Suprachoroidal_rejects_rot(cls):
    """Orientation in tissue is the model's `implant_rotation`"""
    with pytest.raises(TypeError):
        cls(rot=30)


@pytest.mark.parametrize('cls, n_elecs', ((Suprachoroidal24, 35),
                                          (Suprachoroidal44, 46)))
@pytest.mark.parametrize('eye', ('left', 'right'))
def test_Suprachoroidal_per_electrode_z(cls, n_elecs, eye):
    z = np.arange(n_elecs, dtype=float)
    for z_in, z_um in ((list(z), z), (z, z), (z * um, z),
                       (z / 1000 * mm, z)):
        implant = cls(z=z_in, eye=eye)
        npt.assert_almost_equal([e.z for e in implant.electrode_objects],
                                z_um)
    for bad in (np.ones(n_elecs - 1), np.ones(n_elecs + 1),
                np.ones(n_elecs + 1) * um):
        with pytest.raises(ValueError):
            cls(z=bad)
